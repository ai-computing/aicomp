#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# Usage: python3 moe_offload_precompile.py <hf model dir or id> [--backend rngd|rngd-opt]
#
# Compiles and caches what an NPU backend needs for a model's expert shape without using the device (e.g. while it
# is busy): every SlotFFN graph for `rngd`, the generated per-shape kernel + server binary for `rngd-opt`. Later runs load the graphs from ~/.cache/opt_prime/moe_offload/<backend>.
#
import argparse, os, sys, time

import torch
from transformers import AutoConfig

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.moe_offload.backends import get_backend  # noqa: E402
from opt_prime.moe_offload.spec import ExpertSpec  # noqa: E402


def spec_from_config(path):
    c = AutoConfig.from_pretrained(path)
    c = getattr(c, "text_config", c)
    E = next(getattr(c, k) for k in ("num_experts", "n_routed_experts", "num_local_experts") if getattr(c, k, None))
    return ExpertSpec(E, c.hidden_size, c.moe_intermediate_size, dtype=torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--backend", default="rngd")
    a = ap.parse_args()
    spec = spec_from_config(a.model)
    b = get_backend(a.backend, emulate=False)
    t0 = time.time()
    done = b.precompile(spec)
    print(f"{spec.signature()}: {len(done)} artifacts cached in {b.cache_dir()} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
