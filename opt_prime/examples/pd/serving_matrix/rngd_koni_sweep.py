"""KONI long-prefix correctness probe using the existing Llama3.1 RNGD graph."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from inputs import exact_prompt
from koni_export import export_groups, validate
from rngd_pd.engines.fsdk import sync
from rngd_pd.engines.model import ModelConfig
from rngd_pd.engines.prefill_full import FullPrefillEngine
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = "KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024"
REVISION = "0172bf7e73977ae213613529f3d2d5400a33f678"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--width", type=int, choices=[128, 512, 2048, 8192], required=True)
p.add_argument("--output", required=True)
p.add_argument("--rms-tile", type=int, choices=[0, 256, 512], default=0)
p.add_argument("--group", type=int, choices=[1, 2], default=2)
p.add_argument("--partitioned", action="store_true")
a = p.parse_args()
if a.partitioned and (a.group != 1 or not a.rms_tile or a.width % a.rms_tile):
    p.error("partitioned requires group1 and divisible RMS tiles")
if a.rms_tile and not a.partitioned:
    if a.width % a.rms_tile:
        p.error("width must be divisible by rms tile")
    from rngd_pd.engines.fsdk import TacticKernelModule
    from rngd_pd.kernels import prefill_full as full
    from rngd_pd.kernels.prefill_chunk import rms_dsl
    from rowwise_rms import RowwiseRMS

    class TiledLayer(full.PrefillLayer):
        def __init__(self, width):
            super().__init__(width)
            self.rms1 = RowwiseRMS(
                TacticKernelModule(rms_dsl(a.rms_tile, 4096)), a.rms_tile
            )
            self.rms2 = RowwiseRMS(
                TacticKernelModule(rms_dsl(a.rms_tile, 4096)), a.rms_tile
            )

    # Process-local graph selection; installed SDK files remain unchanged.
    full.PrefillLayer = TiledLayer
path = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
config = json.loads((path / "config.json").read_text())
validate(config)
weights = {}
for shard in sorted(path.glob("*.safetensors")):
    with safe_open(shard, framework="pt") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open is not iterable
            weights[key] = f.get_tensor(key).float().numpy()
tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
ids = exact_prompt(
    [tok.bos_token_id] if tok.bos_token_id is not None else [],
    tok.encode(
        "This is background context. The red box contains apples. The blue box contains books.\n",
        add_special_tokens=False,
    ),
    tok.encode(
        "\nQuestion: What is the capital of France?\nAnswer:", add_special_tokens=False
    ),
    a.width,
)


class KVOnly(FullPrefillEngine):
    def _build_lmhead(self, cache_dir):
        # Use the same CPU FP32 final normalization/head as the Qwen export probe.
        pass


engine_type = KVOnly
extra = {}
if a.partitioned:
    from koni_partitioned import PartitionedPrefill

    engine_type = PartitionedPrefill
    extra["rms_tile"] = a.rms_tile
t = time.perf_counter()
engine = engine_type(
    ModelConfig(SimpleNamespace(**config)),
    weights,
    C=a.width,
    group=a.group,
    chip=0,
    cache_dir=None,
    S=a.width,
    **extra,
)
sync(0)
print(
    json.dumps(
        {"ready": True, "init_s": time.perf_counter() - t, "revision": REVISION}
    ),
    flush=True,
)
norm = torch.from_numpy(weights["model.norm.weight"])
head = torch.from_numpy(weights["lm_head.weight"])
previous = None
with open(a.output + ".jsonl", "w", buffering=1) as f, torch.inference_mode():
    for repeat in range(3):
        t = time.perf_counter()
        stage = engine.begin_device_prefill(ids)
        while not stage.advance():
            pass
        sync(0)
        k, v, logits = export_groups(stage, norm, head, config["rms_norm_eps"])
        elapsed = time.perf_counter() - t
        assert k.shape == v.shape == (32, 8, a.width, 128), (
            f"keys={tuple(k.shape)}, values={tuple(v.shape)}, groups={stage.g}, "
            f"first_device_k={tuple(stage.Ks[0].shape)}, first_host_k={tuple(stage.Ks[0].cpu().shape)}"
        )
        assert all(bool(torch.isfinite(x).all()) for x in (k, v, logits))
        exact = previous is None or all(
            torch.equal(x, y) for x, y in zip(previous, (k, v, logits))
        )
        assert exact, "repeat mismatch"
        previous = (k, v, logits)
        row = {
            "model": MODEL,
            "revision": REVISION,
            "width": a.width,
            "repeat": repeat,
            "rms_tile": a.rms_tile,
            "group": a.group,
            "partitioned": a.partitioned,
            "prefill_s": elapsed,
            "repeat_exact": exact,
            "first_token": int(logits.argmax()),
            "kv_shape": list(k.shape),
            "dtype": str(k.dtype),
        }
        f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
torch.save(
    {"ids": ids, "keys": k, "values": v, "logits": logits, "metadata": row},
    a.output + ".pt",
)
