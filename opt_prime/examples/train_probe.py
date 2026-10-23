#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# Training probe: a tiny random model of a transformers family through Optimus_p (pipeline training), full fine-tuning
# or LoRA, a few steps on a fixed batch.
#
# Checks: the first step's loss equals the plain HF model's loss on the same batch, the loss decreases, and with
# LoRA the base weights stay unchanged.
#
# Usage:  torchrun --nproc_per_node=1 train_probe.py --family llama [--lora] [--pp-size 1] [--dynamo-capture]
#         torchrun --nproc_per_node=2 train_probe.py --family qwen3_moe --pp-size 2
import argparse, copy, os, sys

# an installed NPU torch backend (furiosa-torch) autoloads with torch and breaks CUDA autograd streams
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.IR import IR_Anal  # noqa: E402
from opt_prime.lora import LoRAConfig  # noqa: E402
from opt_prime.accel.testing import build_tiny  # noqa: E402
from opt_prime.opti_pri import Optimus_p  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--pp-size", type=int, default=1)
    ap.add_argument("--dynamo-capture", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dtype", default="float32", help="bfloat16 for MoE families (transformers' grouped_mm experts)")
    ap.add_argument("--offload-hot", type=float, default=None,
                    help="MoE models: keep this fraction of each layer's experts on the GPU, the rest on the host "
                         "(experts frozen; non-expert parameters train)")
    a = ap.parse_args()
    rank = int(os.environ.get("RANK", "0"))

    dt = getattr(torch, a.dtype)
    model = build_tiny(a.family, dt)
    model.config.use_cache = False
    if os.environ.get("EXPERTS_IMPL"): model.config._experts_implementation = os.environ["EXPERTS_IMPL"]
    V = model.config.get_text_config().vocab_size if hasattr(model.config, "get_text_config") else model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    data = torch.randint(3, V, (4, 16), generator=g)
    labels = data.clone()
    labels[:, :-1] = data[:, 1:]
    labels[:, -1] = -100
    with torch.no_grad():
        ref_logits = copy.deepcopy(model)(data).logits.float()
    ref_loss = F.cross_entropy(ref_logits.reshape(-1, ref_logits.shape[-1]), labels.reshape(-1), ignore_index=-100).item()

    offload = None
    if a.offload_hot is not None:
        from opt_prime.accel.moe import ExpertProfiler, MoEOffloadConfig
        with ExpertProfiler(model) as prof, torch.no_grad():
            model(data)
        counts = prof.as_lists()
        E = len(counts[0])
        offload = MoEOffloadConfig(hot=[sorted(range(E), key=lambda e: -c[e])[:int(E * a.offload_hot)] for c in counts])
    opt = Optimus_p(model, 1, use_gpu=True, pp_size=a.pp_size, ir_analyze=IR_Anal.SEQUENTIAL,
                    dynamo_capture=a.dynamo_capture, moe_offload=offload)
    if a.lora:
        opt.apply_lora(LoRAConfig(r=4, alpha=8.0, dropout=0.0,
                                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    opt.train()
    params = [p for p in opt.parameters() if p.requires_grad]
    base = {n: p.detach().clone() for n, p in opt.run_info.submod.named_parameters() if not p.requires_grad}
    opt.optimizer = torch.optim.SGD(params, lr=a.lr or (1e-2 if a.lora else 5e-2))
    losses = []
    for step in range(a.steps):
        d, l = (data, labels) if opt.is_first_stage() else (None, None)
        l = opt.move_labels2last_stage(l)
        opt.optimizer.zero_grad()
        opt.run(d, l, mode="1f1b")
        opt.optimizer.step()
        if opt.is_last_stage():
            losses.append(float(sum(opt.get_loss()) / opt.num_mb))
    unchanged = all(torch.equal(p.detach(), base[n]) for n, p in opt.run_info.submod.named_parameters() if n in base)
    if opt.is_last_stage():
        tol = 1e-4 if dt == torch.float32 else 2e-2
        first_ok = abs(losses[0] - ref_loss) <= tol * max(1.0, abs(ref_loss))
        dec_ok = losses[-1] < losses[0]
        lora_ok = (not a.lora) or unchanged
        ok = first_ok and dec_ok and lora_ok
        print(f"[probe] {a.family} {'lora' if a.lora else 'ft'} pp={a.pp_size}"
              f"{'' if a.offload_hot is None else f' offload-hot={a.offload_hot}'}: ref {ref_loss:.5f} first {losses[0]:.5f} "
              f"last {losses[-1]:.5f} trainable {sum(p.numel() for p in params)} base unchanged {unchanged} -> "
              f"{'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
