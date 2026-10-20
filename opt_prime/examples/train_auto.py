#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# One training job, backend chosen from the model's block spec and the device:
#   python3 train_auto.py --model unsloth/Llama-3.2-1B-Instruct --data data.jsonl --device rngd:0 --steps 20
#   torchrun --nproc_per_node=1 train_auto.py --model ... --device cuda:0           (Optimus pipeline on GPUs)
# Data: JSONL with {"text"} | {"prompt","response"} | {"messages"}. --make-data writes a small synthetic set.
import argparse, json, os, sys

# an installed NPU torch backend (furiosa-torch) autoloads with torch and breaks CUDA autograd; RNGD jobs need it
if not any(a.startswith("rngd") for a in sys.argv):
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.train import TrainJob, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--method", default="lora", choices=["lora", "full"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--offload-gpu-gb", type=float, default=None, help="MoE: GPU budget for experts, rest on host")
    ap.add_argument("--moe-hot-fraction", type=float, default=0.0,
                    help="MoE on an NPU (npu-lora): fraction of each layer's experts computed on the device")
    ap.add_argument("--make-data", type=int, default=0, help="write N synthetic prompt/response examples to --data")
    a = ap.parse_args()
    if a.make_data:
        with open(a.data, "w") as f:
            for i in range(a.make_data):
                f.write(json.dumps({"prompt": f"What is the code name of project {i}?",
                                    "response": f"Project {i} is code-named Optimus-{i * 37 % 1000}."}) + "\n")
    res = train(TrainJob(model=a.model, data=a.data, method=a.method, device=a.device, steps=a.steps,
                         seq_len=a.seq_len, lr=a.lr, out=a.out, offload_gpu_gb=a.offload_gpu_gb,
                         moe_hot_fraction=a.moe_hot_fraction))
    if res.losses:
        print(f"[train] backend {res.backend}: {len(res.losses)} steps, loss {res.losses[0]:.4f} -> "
              f"{res.losses[-1]:.4f} in {res.seconds:.1f}s; {json.dumps({k: v for k, v in res.info.items() if k != 'summary'})}")


if __name__ == "__main__":
    main()
