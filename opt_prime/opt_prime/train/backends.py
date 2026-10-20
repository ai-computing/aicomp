#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""TrainJob / TrainBackend contract and the built-in backends."""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from ..model.blocks import ModelSpec

#: LoRA target names: HF module names -> npu-lora's short names
_NPU_LORA_TARGETS = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"}


@dataclass
class TrainJob:
    model: str                              # HF id or local path
    data: str                               # JSONL: {"text"} | {"prompt", "response"} | {"messages"}
    method: str = "lora"                    # "lora" | "full"
    device: str = "cuda:0"                  # "cuda:N", "cpu", "rngd:N"
    steps: int = 100
    seq_len: int = 512
    lr: float = 2e-4
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_targets: Tuple[str, ...] = ("q_proj", "v_proj")
    out: Optional[str] = None               # adapter (LoRA, PEFT format) or weights (full) directory
    seed: int = 0
    offload_gpu_gb: Optional[float] = None  # MoE models: experts beyond this GPU budget stay frozen on the host

    @property
    def device_kind(self) -> str:
        return self.device.split(":")[0]


@dataclass
class TrainResult:
    backend: str
    losses: List[float]
    out: Optional[str]
    seconds: float
    info: Dict = field(default_factory=dict)


class TrainBackend:
    """supports(spec, job) -> reasons it cannot run the job ([] = can); run(job, spec) -> TrainResult."""
    name = "base"
    device_kinds: Tuple[str, ...] = ()

    def supports(self, spec: ModelSpec, job: TrainJob) -> List[str]:
        return [] if job.device_kind in self.device_kinds else [f"device {job.device_kind} not in {self.device_kinds}"]

    def run(self, job: TrainJob, spec: ModelSpec) -> TrainResult:
        raise NotImplementedError


_BACKENDS: Dict[str, TrainBackend] = {}


def register_train_backend(backend: TrainBackend):
    _BACKENDS[backend.name] = backend
    return backend


def registered_train_backends() -> List[TrainBackend]:
    return list(_BACKENDS.values())


def select_train_backend(spec: ModelSpec, job: TrainJob, candidates=None):
    """-> (backend, table of (name, reasons)); the first registered backend that supports the job."""
    rows = [(b, b.supports(spec, job)) for b in (candidates if candidates is not None else registered_train_backends())]
    for b, reasons in rows:
        if not reasons:
            return b, [(x.name, r) for x, r in rows]
    raise ValueError("no training backend supports this job: " + "; ".join(f"{b.name}: {', '.join(r)}" for b, r in rows))


def train(job: TrainJob, candidates=None) -> TrainResult:
    from ..devices import acquire
    from ..model import model_spec_from_pretrained
    spec = model_spec_from_pretrained(job.model)
    backend, table = select_train_backend(spec, job, candidates)
    lease = None
    if job.device_kind in ("rngd", "atom"):
        lease = acquire(job.device, "train", f"train {backend.name} {job.model}")
    try:
        result = backend.run(job, spec)
    finally:
        if lease is not None:
            lease.release()
    result.info["selection"] = table
    return result


# ---------------------------------------------------------------------------------------------------------- data

def _examples(path, tok, seq_len):
    """(ids, loss mask) per JSONL example, truncated to seq_len + 1 tokens."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        if "text" in ex:
            ids = tok(ex["text"])["input_ids"]
            mask = [True] * len(ids)
        else:
            msgs = ex.get("messages") or [{"role": "user", "content": ex["prompt"]},
                                          {"role": "assistant", "content": ex["response"]}]
            p = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
            f = tok.apply_chat_template(msgs, tokenize=True)
            p = list(p["input_ids"] if hasattr(p, "keys") else p)
            ids = list(f["input_ids"] if hasattr(f, "keys") else f)
            mask = [False] * len(p) + [True] * (len(ids) - len(p))
        out.append((ids[:seq_len + 1], mask[:seq_len + 1]))
    return out


# ---------------------------------------------------------------------------------------------------- backends

class OptimusTrainBackend(TrainBackend):
    """Optimus_p (pipeline, one GPU per stage) with opt_prime.lora or full fine-tuning; one sequence per step."""
    name = "optimus"
    device_kinds = ("cuda",)

    def supports(self, spec, job):
        reasons = super().supports(spec, job)
        if job.method not in ("lora", "full"):
            reasons.append(f"method {job.method}")
        return reasons

    def run(self, job, spec):
        import sys
        import torch
        if "furiosa.torch" in sys.modules:
            raise RuntimeError("furiosa-torch is loaded in this process (torch device-backend autoload); it breaks CUDA "
                               "autograd streams. Set TORCH_DEVICE_BACKEND_AUTOLOAD=0 before importing torch.")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ..IR import IR_Anal
        from ..lora import LoRAConfig
        from ..opti_pri import Optimus_p
        tok = AutoTokenizer.from_pretrained(job.model)
        model = AutoModelForCausalLM.from_pretrained(job.model, dtype=torch.bfloat16)
        model.config.use_cache = False
        offload = None
        if job.offload_gpu_gb is not None and any(l.is_moe for l in spec.layers):
            from ..moe_offload import ExpertProfiler, MoEOffloadConfig
            sample = [torch.tensor([ids[:-1]]) for ids, _ in _examples(job.data, tok, job.seq_len)[:4]]
            with ExpertProfiler(model) as prof, torch.no_grad():
                for x in sample:
                    model(x)
            offload = MoEOffloadConfig(gpu_budget_gb=job.offload_gpu_gb, profile=prof.as_lists())
        opt = Optimus_p(model, 1, use_gpu=True, pp_size=int(os.environ.get("WORLD_SIZE", "1")),
                        ir_analyze=IR_Anal.SEQUENTIAL, dynamo_capture=True, moe_offload=offload)
        if job.method == "lora":
            opt.apply_lora(LoRAConfig(r=job.lora_r, alpha=job.lora_alpha, dropout=0.0,
                                      target_modules=list(job.lora_targets)))
        opt.train()
        opt.optimizer = torch.optim.AdamW([p for p in opt.parameters() if p.requires_grad], lr=job.lr)
        data = _examples(job.data, tok, job.seq_len)
        losses, t0 = [], time.time()
        for step in range(job.steps):
            ids, mask = data[step % len(data)]
            x = torch.tensor([ids[:-1]])
            y = torch.tensor([[t if m else -100 for t, m in zip(ids[1:], mask[1:])]])
            d, l = (x, y) if opt.is_first_stage() else (None, None)
            l = opt.move_labels2last_stage(l)
            opt.optimizer.zero_grad()
            opt.run(d, l, mode="1f1b")
            torch.nn.utils.clip_grad_norm_([p for p in opt.parameters() if p.requires_grad], 1.0)
            opt.optimizer.step()
            if opt.is_last_stage():
                losses.append(float(sum(opt.get_loss()) / opt.num_mb))
        out = None
        if job.out:                                   # every stage saves its own trainable parameters
            os.makedirs(job.out, exist_ok=True)
            out = job.out
            state = {n: p.detach().cpu() for n, p in opt.run_info.submod.named_parameters() if p.requires_grad}
            torch.save(state, os.path.join(out, f"stage{opt.tpl.stage}_trainable.pt"))
        return TrainResult(self.name, losses, out, time.time() - t0,
                           dict(stage=opt.tpl.stage, offload=None if opt.moe_offload_report is None else
                                [(t["tier"], t["experts"]) for t in opt.moe_offload_report.tiers]))


class NpuLoraTrainBackend(TrainBackend):
    """The npu-lora library: its functional model, backward graphs compiled for the NPU, host optimizer. Supports the
    families npu-lora implements: dense full-attention GQA blocks with a gated SiLU MLP - Llama, Mistral (no sliding
    window), Qwen2 (q/k/v biases), Qwen3 (per-head q/k norm), Gemma 3 text (sequence within the sliding window) - and,
    with an npu-lora that has the MoE split, LoRA on Qwen3-MoE / Qwen2-MoE / Mixtral (attention on the NPU, the
    post-attention norm and MoE block frozen on the host)."""
    name = "npu-lora"
    device_kinds = ("rngd", "cuda", "cpu")

    def supports(self, spec, job):
        reasons = super().supports(spec, job)
        try:
            import npu_lora  # noqa: F401
        except ImportError:
            return reasons + ["npu-lora not installed (pip install npu-lora)"]
        families = {"llama": (False, False), "mistral": (False, False), "qwen2": (True, False), "qwen3": (False, True),
                    "gemma3_text": (False, True)}
        # MoE checkpoints (npu-lora >= the MoE split): attention on the NPU, norm + MoE block frozen on the host
        moe_families = {"qwen3_moe": (False, True), "qwen2_moe": (True, False), "mixtral": (False, False)}
        try:
            from npu_lora.llama import MOE_FAMILIES
            families.update({k: v for k, v in moe_families.items() if k in MOE_FAMILIES})
        except ImportError:
            pass
        moe = spec.model_type in moe_families
        if spec.model_type not in families:
            reasons.append(f"npu-lora implements {sorted(families)}, not {spec.model_type}")
        else:
            bias, qkn = families[spec.model_type]
            for l in spec.layers:
                a = l.attention
                window_ok = not a.sliding_window or (spec.model_type == "gemma3_text" and job.seq_len <= a.sliding_window)
                mlp_ok = l.is_moe or (l.mlp and l.mlp.kind == "gated") if moe else (not l.is_moe and (
                    not l.mlp or l.mlp.kind == "gated"))
                if (a.kind != "gqa" or not window_ok or not mlp_ok or a.qk_norm != qkn or (a.qkv_bias and not bias)):
                    reasons.append(f"layer {l.index} is not a {spec.model_type} block npu-lora implements")
                    break
            if moe and job.method == "full":
                reasons.append("npu-lora trains MoE checkpoints with LoRA only (experts stay frozen on the host)")
        bad = [t for t in job.lora_targets if t not in _NPU_LORA_TARGETS]
        if job.method == "lora" and bad:
            reasons.append(f"LoRA targets {bad} (npu-lora: {sorted(_NPU_LORA_TARGETS)})")
        return reasons

    def run(self, job, spec):
        from transformers import AutoTokenizer
        from npu_lora import backends, export
        from npu_lora.data import PackedDataset
        from npu_lora.trainer import LoRAConfig, LoRATrainer, TrainConfig
        dev = job.device.replace("rngd", "furiosa")
        tc = TrainConfig(seq_len=job.seq_len, lr=job.lr, steps=job.steps, seed=job.seed, full=job.method == "full")
        lora = LoRAConfig(r=job.lora_r, alpha=job.lora_alpha,
                          targets=tuple(_NPU_LORA_TARGETS[t] for t in job.lora_targets))
        import sys

        def log(line):                     # npu-lora's JSON events (compile progress, steps) -> stderr
            print(line, file=sys.stderr, flush=True)
        tr = LoRATrainer(job.model, backends.make(dev), lora, tc, log=log)
        tok = AutoTokenizer.from_pretrained(tr.path)
        ds = PackedDataset(job.data, tok, job.seq_len, seed=job.seed)
        t0 = time.time()
        tr.setup()
        setup_s = time.time() - t0
        losses = [tr.train_step(b)["loss"] for b in ds.batches(1, job.steps)]
        if job.out:
            (export.save_full(tr, job.out) if tc.full else export.save_peft(tr, job.out, job.model))
        return TrainResult(self.name, losses, job.out, time.time() - t0,
                           dict(setup_s=round(setup_s, 1), backend=dev, summary=tr.summary()))


# order = preference among backends that support a job: Optimus on GPUs (any family), npu-lora on RNGD
register_train_backend(OptimusTrainBackend())
register_train_backend(NpuLoraTrainBackend())
