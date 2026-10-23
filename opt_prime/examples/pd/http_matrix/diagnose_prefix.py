"""Teacher-force equal histories while varying prefix-cache history and precision (#212)."""

import argparse
import json
from pathlib import Path

import torch
from opt_prime.accel.pd import PDRequest, SamplingParams, batching
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
head = [tok.bos_token_id] if tok.bos_token_id is not None else []
fill = tok.encode(
    "This is background context. The red box contains apples. The blue box contains books.\n",
    add_special_tokens=False,
)
tail = tok.encode(
    "\nQuestion: What is the capital of France?\nAnswer:", add_special_tokens=False
)
inputs = {
    n: head
    + (fill * ((n - len(head) - len(tail) + len(fill) - 1) // len(fill)))[
        : n - len(head) - len(tail)
    ]
    + tail
    for n in (128, 2048, 8192)
}
model = (
    AutoModelForCausalLM.from_pretrained(
        a.model, local_files_only=True, dtype=torch.bfloat16
    )
    .cuda()
    .eval()
)
results = []
original = batching.select_token
try:
    with torch.inference_mode():
        for dtype in ("bfloat16", "float32"):
            model.to(getattr(torch, dtype))
            oracle = None
            reference = None
            for mode, history in [
                ("cold", []),
                ("partial", [128]),
                ("warm", [128, 2048, 8192]),
                ("warm_concurrent", [128, 2048, 8192]),
            ]:
                b = batching.ContinuousBatcher(
                    model,
                    slots=4,
                    max_context=9216,
                    kv="paged",
                    page_size=64,
                    prefill_chunk=256,
                    prefix_cache=True,
                )
                for i, n in enumerate(history):
                    f = b.submit(PDRequest(f"prime-{i}", tuple(inputs[n]), 32))
                    b.run_until_idle()
                    f.result()
                vectors = []
                predicted = []

                def sample(
                    logits,
                    params,
                    position,
                    vectors=vectors,
                    predicted=predicted,
                    oracle=oracle,
                ):
                    if params.seed != 212:
                        return original(logits, params, position)
                    vectors.append(logits.float().cpu().clone())
                    predicted.append(int(logits.argmax()))
                    return (
                        original(logits, params, position)
                        if oracle is None
                        else oracle[len(vectors) - 1]
                    )

                batching.select_token = sample
                before = b.kv.hit_tokens
                f = b.submit(
                    PDRequest(
                        "target", tuple(inputs[2048]), 32, SamplingParams(seed=212)
                    )
                )
                if mode == "warm_concurrent":
                    for j, n in enumerate([128, 8192, 128]):
                        b.submit(PDRequest(f"companion-{j}", tuple(inputs[n]), 32))
                b.run_until_idle()
                tokens = f.result()
                batching.select_token = original
                values = torch.stack(vectors)
                assert values.shape[0] == 32 and torch.isfinite(values).all()
                if oracle is None:
                    oracle = tokens
                    reference = values
                delta = values - reference
                different = [
                    i for i, (u, v) in enumerate(zip(predicted, oracle)) if u != v
                ]
                rows = []
                for i in different:
                    tops = [z[i].topk(3) for z in (reference, values)]
                    rows.append(
                        {
                            "position": i,
                            "top_ids": [x.indices.tolist() for x in tops],
                            "top_logits": [x.values.tolist() for x in tops],
                        }
                    )
                result = {
                    "dtype": dtype,
                    "oracle": oracle,
                    "mode": mode,
                    "cache_hit_tokens": b.kv.hit_tokens - before,
                    "argmax_disagreements": different,
                    "max_abs_logit_difference": float(delta.abs().max()),
                    "relative_rms": float(
                        delta.square().mean().sqrt() / reference.square().mean().sqrt()
                    ),
                    "divergence_details": rows,
                }
                results.append(result)
                print(
                    json.dumps(
                        {k: v for k, v in result.items() if k != "divergence_details"}
                    ),
                    flush=True,
                )
                del b
finally:
    batching.select_token = original
Path(a.output).write_text(json.dumps(results, indent=2) + "\n")
