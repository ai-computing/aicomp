"""Direct HF GPU baseline; not an HTTP/PD or factual-quality benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from inputs import exact_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=2)
    a = p.parse_args()
    if min(*a.lengths, *a.batches, a.new_tokens, a.repeats) <= 0:
        p.error("sizes and repeats must be positive")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    model_path = (
        a.model
        if Path(a.model).is_dir()
        else snapshot_download(a.model, revision=a.revision, local_files_only=True)
    )
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    metadata = {
        "revision": a.revision,
        "model_type": model.config.model_type,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "dtype": "bfloat16",
        "attention": "sdpa",
        "fixed_outputs_ignore_eos": True,
        "prefix_cache": False,
        "validation_scope": "finite logits and repeat stability; not numerical oracle or quality",
    }
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

    @torch.inference_mode()
    def run(ids):
        x = torch.tensor(ids, device=device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        t = time.perf_counter()
        out = model(input_ids=x, use_cache=True, logits_to_keep=1)
        z = out.logits[:, -1]
        finite = torch.isfinite(z).all()
        token = z.argmax(-1)
        tokens = [token]
        cache = out.past_key_values
        torch.cuda.synchronize(device)
        first = time.perf_counter()
        for _ in range(a.new_tokens - 1):
            out = model(
                input_ids=token[:, None],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            z = out.logits[:, -1]
            finite = finite & torch.isfinite(z).all()
            token = z.argmax(-1)
            tokens.append(token)
            cache = out.past_key_values
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        assert bool(finite), "nonfinite logits"
        return {
            "ttft_s": first - t,
            "e2e_s": end - t,
            "tpot_s": (end - first) / (a.new_tokens - 1) if a.new_tokens > 1 else 0,
            "output_tok_s": len(ids) * a.new_tokens / (end - t),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "tokens": torch.stack(tokens, dim=1).cpu().tolist(),
        }

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w", buffering=1) as f:
        f.write(json.dumps({"metadata": metadata}) + "\n")
        run([exact_prompt(prefix, filler, tails[0], 128)])
        for length in a.lengths:
            for batch in a.batches:
                ids = [
                    exact_prompt(prefix, filler, tails[i % len(tails)], length)
                    for i in range(batch)
                ]
                digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                previous = None
                for repeat in range(a.repeats):
                    try:
                        result = run(ids)
                        result["repeat_equal"] = (
                            previous is None or previous == result["tokens"]
                        )
                        assert result["repeat_equal"], "greedy repeat mismatch"
                        previous = result["tokens"]
                        row = {
                            "status": "passed",
                            "input_tokens": length,
                            "batch": batch,
                            "output_tokens": a.new_tokens,
                            "repeat": repeat,
                            "input_sha256": digest,
                            **result,
                        }
                    except torch.cuda.OutOfMemoryError as error:
                        torch.cuda.empty_cache()
                        row = {
                            "status": "oom",
                            "input_tokens": length,
                            "batch": batch,
                            "repeat": repeat,
                            "error": str(error),
                        }
                    f.write(json.dumps(row) + "\n")
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "tokens"}),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
