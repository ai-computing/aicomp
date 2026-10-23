import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from huggingface_hub import snapshot_download
from rngd_pd.engines.fsdk import sync
from rngd_pd.engines.model import ModelConfig
from rngd_pd.engines.prefill_full import FullPrefillEngine
from safetensors import safe_open
from transformers import AutoTokenizer

mid = "KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024"
path = Path(snapshot_download(mid, local_files_only=True))
tok = AutoTokenizer.from_pretrained(mid)
texts = [
    "What is the capital of France?",
    "Compute 2 + 3.",
    "한국의 수도는 어디인가요?",
    "What is the capital of Japan?",
]
prompts = [
    tok.apply_chat_template(
        [{"role": "user", "content": s}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    for s in texts
]
capacity = ((max(map(len, prompts)) - 1 + 31) // 32) * 32
assert capacity <= 384
print("PROMPT_LENGTHS", list(map(len, prompts)), "CAPACITY", capacity, flush=True)


class KVOnly(FullPrefillEngine):
    def _build_lmhead(self, cache_dir):
        pass

    def _next_token(self, x, P):
        return None


sd = {}
for shard in sorted(path.glob("*.safetensors")):
    with safe_open(shard, framework="pt") as f:
        for key in f:
            sd[key] = f.get_tensor(key).float().numpy()
print("WEIGHTS", flush=True)
cfg = ModelConfig(SimpleNamespace(**json.loads((path / "config.json").read_text())))
e = KVOnly(cfg, sd, C=capacity, group=2, chip=0, cache_dir=None, S=512)
print("READY", flush=True)
rows = []
for i, p in enumerate(prompts):
    t = time.perf_counter()
    k, v, _ = e.prefill_device(p[:-1])
    sync(0)
    np.savez(
        f"koni196_kv_{i}.npz",
        keys=k.cpu().float().numpy(),
        values=v.cpu().float().numpy(),
    )
    rows.append(
        {"prompt": texts[i], "ids": p, "prefill_seconds": time.perf_counter() - t}
    )
    print(rows[-1], flush=True)
Path("koni196_prompts.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
)
