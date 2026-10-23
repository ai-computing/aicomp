import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from prefill import Prefill
from safetensors import safe_open
from transformers import AutoTokenizer

mid = "Qwen/Qwen3-0.6B"
path = Path(snapshot_download(mid, local_files_only=True))
c = json.loads((path / "config.json").read_text())
w = {}
for shard in sorted(path.glob("*.safetensors")):
    with safe_open(shard, framework="pt") as f:
        for key in f:
            w[key] = f.get_tensor(key).float()
e = Prefill(c, w, width=128, chip=0)
print("READY", flush=True)
tok = AutoTokenizer.from_pretrained(mid)
records = []
for prompt in [
    "What is the capital of France?",
    "Compute 2 + 3.",
    "한국의 수도는 어디인가요?",
]:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        enable_thinking=False,
    )
    t = time.perf_counter()
    k, v, l = e.prefill(ids)
    row = {
        "prompt": prompt,
        "ids": ids,
        "keys": k,
        "values": v,
        "logits": l,
        "prefill_s": time.perf_counter() - t,
    }
    records.append(row)
    print(prompt, "first", int(l.argmax()), "seconds", row["prefill_s"], flush=True)
torch.save(records, "qwen3_197_kv.pt")
