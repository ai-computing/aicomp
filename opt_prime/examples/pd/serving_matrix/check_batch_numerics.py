"""Teacher-force the first B1/B4 divergence under BF16, then FP32."""

import argparse
import json
from pathlib import Path

import torch
from inputs import exact_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--baseline", required=True)
p.add_argument("--output", required=True)
p.add_argument("--batch", type=int, default=4, choices=[4, 16])
a = p.parse_args()
rows = [json.loads(s) for s in Path(a.baseline).read_text().splitlines()]
ref = {
    (r["input_tokens"], r["batch"]): r["tokens"] for r in rows if r.get("repeat") == 1
}
tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
prefix = [tok.bos_token_id] if tok.bos_token_id is not None else []
filler = tok.encode(
    "This is background context. The red box contains apples. The blue box contains books.\n",
    add_special_tokens=False,
)
tails = [
    tok.encode(s, add_special_tokens=False)
    for s in [
        "\nQuestion: What is the capital of France?\nAnswer:",
        "\nQuestion: Compute 2 + 3.\nAnswer:",
        "\nQuestion: What does the blue box contain?\nAnswer:",
        "\nQuestion: 한국의 수도는 어디인가요?\nAnswer:",
    ]
]
cases = []
for n in [128, 512, 2048, 8192]:
    x, y = ref[n, 1][0], ref[n, a.batch][0]
    if x != y:
        cases.append((n, next(i for i, (u, v) in enumerate(zip(x, y)) if u != v)))
model = (
    AutoModelForCausalLM.from_pretrained(
        a.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    .cuda()
    .eval()
)
torch.backends.cuda.matmul.allow_tf32 = False
results = []
with torch.inference_mode():
    for dtype in ["bfloat16", "float32"]:
        if dtype == "float32":
            model.float()
        for n, step in cases:
            vectors = []
            for batch in [1, a.batch]:
                ids = [
                    exact_prompt(prefix, filler, tails[i % len(tails)], n)
                    for i in range(batch)
                ]
                out = model(
                    input_ids=torch.tensor(ids, device="cuda"),
                    use_cache=True,
                    logits_to_keep=1,
                )
                for j in range(step):
                    forced = [ref[n, batch][i][j] for i in range(batch)]
                    out = model(
                        input_ids=torch.tensor(forced, device="cuda")[:, None],
                        past_key_values=out.past_key_values,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                vectors.append(out.logits[0, -1].float().cpu())
            top = [v.topk(5) for v in vectors]
            row = {
                "dtype": dtype,
                "batch": a.batch,
                "input_tokens": n,
                "generated_position": step,
                "argmax_equal": int(vectors[0].argmax()) == int(vectors[1].argmax()),
                "max_abs_logit_difference": float(
                    (vectors[0] - vectors[1]).abs().max()
                ),
                "top_ids": [t.indices.tolist() for t in top],
                "top_values": [t.values.tolist() for t in top],
                "top1_margin": [float(t.values[0] - t.values[1]) for t in top],
            }
            results.append(row)
            print(json.dumps(row), flush=True)
Path(a.output).write_text(json.dumps(results, indent=2) + "\n")
