"""Local experimental host-KV producer; one synchronous request at a time."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shared-fd", type=int)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    from opt_prime.pd.npu.provenance import record_run

    record_run(a.out, __file__)
    cfg = json.loads(a.config.read_text())
    import numpy as np
    import torch
    from hetero_p2p.serving import RngdBinding
    from opt_prime.pd.npu.model import DenseModel, load_weights
    from opt_prime.pd.npu.rngd import QwenRngdPrefill

    binding = RngdBinding.from_path(cfg["rngd_device_path"], cfg["rngd_logical_index"])
    state = subprocess.check_output(["furiosa-smi", "ps"], text=True)
    (a.out / "before.txt").write_text(state)
    if f"npu{binding.index}:" in state:
        raise RuntimeError("selected RNGD is in use")
    torch.set_num_threads(8)
    spec = DenseModel.load(cfg["model"], cfg["model_id"])
    engine = QwenRngdPrefill(
        spec,
        load_weights(cfg["model"], spec),
        width=64,
        chip=binding.compute_index,
        accurate_attention=True,
    )
    descriptors = []
    for f in Path("/proc/self/fd").iterdir():
        try:
            descriptors.append(os.readlink(f))
        except OSError:
            pass
    binding.verify_open_channels(descriptors)
    (a.out / "binding.json").write_text(json.dumps(descriptors))
    shared = None
    if a.shared_fd is not None:
        from opt_prime.pd.npu.shared_kv import SharedKVSlab

        slab = SharedKVSlab(
            a.shared_fd, (spec.caches, spec.kv_heads, 512, spec.head_dim), writable=True
        )
        shared = slab.view()
    print("PREFIX_READY", flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        index = request["index"]
        if type(index) is not int or not 0 <= index < len(cfg["prompt_token_ids"]):
            raise ValueError("invalid prompt index")
        k, v, _ = engine.prefill_device(cfg["prompt_token_ids"][index][:-1])
        if request.get("transport") == "shared":
            if shared is None:
                raise ValueError("shared slab not configured")
            shared[0::2] = k.cpu().float().numpy().transpose(0, 1, 3, 2)
            shared[1::2] = v.cpu().float().numpy()
            print(
                "PREFIX_RESULT "
                + json.dumps(dict(sequence=request["sequence"], index=index)),
                flush=True,
            )
            continue
        path = a.out / f'prefix-{request["sequence"]}.npz'
        np.savez(path, keys=k.cpu().float().numpy(), values=v.cpu().float().numpy())
        print("PREFIX_RESULT " + str(path.resolve()), flush=True)


if __name__ == "__main__":
    main()
