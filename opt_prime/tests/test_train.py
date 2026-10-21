#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# opt_prime.accel.train: backend selection per model / device, and npu-lora (CPU reference backend) on a tiny Llama.
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opt_prime.accel.spec import model_spec  # noqa: E402
from opt_prime.accel.testing import build_tiny  # noqa: E402
from opt_prime.accel.train import NpuLoraTrainBackend, TrainJob, select_train_backend, train  # noqa: E402

TOK = "unsloth/Llama-3.2-1B-Instruct"


def test_selection_by_model_and_device():
    from transformers import AutoConfig, AutoModelForCausalLM
    llama = model_spec(AutoModelForCausalLM.from_config(AutoConfig.for_model(
        "llama", hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128)))
    qwen3_moe = model_spec(build_tiny("qwen3_moe"))
    job = lambda dev: TrainJob(model="x", data="x", device=dev)
    assert select_train_backend(llama, job("cuda:0"))[0].name == "optimus"
    assert select_train_backend(qwen3_moe, job("cuda:0"))[0].name == "optimus"
    qwen3 = model_spec(AutoModelForCausalLM.from_config(AutoConfig.for_model(
        "qwen3", hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, vocab_size=128)))
    if NpuLoraTrainBackend().supports(llama, job("rngd:0")) == []:      # npu-lora installed
        assert select_train_backend(llama, job("rngd:0"))[0].name == "npu-lora"
        assert select_train_backend(qwen3, job("rngd:0"))[0].name == "npu-lora"
    try:
        from npu_lora.llama import MOE_FAMILIES                          # npu-lora with the MoE split
    except ImportError:
        MOE_FAMILIES = {}
    if "qwen3_moe" in MOE_FAMILIES:
        assert select_train_backend(qwen3_moe, job("rngd:0"))[0].name == "npu-lora"
        full = TrainJob(model="x", data="x", device="rngd:0", method="full")
        from opt_prime.accel.train.backends import _npu_lora_has
        assert (NpuLoraTrainBackend().supports(qwen3_moe, full) == []) == _npu_lora_has("moe", "HostMLP", "step")
    else:
        with pytest.raises(ValueError, match="npu-lora|optimus"):
            select_train_backend(qwen3_moe, job("rngd:0"))               # this npu-lora has no MoE split


def test_npu_lora_moe_families_follow_the_installed_library(monkeypatch):
    """MoE checkpoints go to npu-lora only when the installed npu-lora declares the MoE split."""
    import types
    qwen3_moe = model_spec(build_tiny("qwen3_moe"))
    job = TrainJob(model="x", data="x", device="rngd:0")
    pkg, mod = types.ModuleType("npu_lora"), types.ModuleType("npu_lora.llama")
    pkg.llama = mod
    monkeypatch.setitem(sys.modules, "npu_lora", pkg)
    monkeypatch.setitem(sys.modules, "npu_lora.llama", mod)
    for sub in ("moe", "moe_device"):                  # an old npu-lora: neither module
        monkeypatch.setitem(sys.modules, f"npu_lora.{sub}", None)
    assert NpuLoraTrainBackend().supports(qwen3_moe, job)                 # old npu-lora: rejected
    mod.MOE_FAMILIES = {"qwen3_moe": "qwen3"}
    assert NpuLoraTrainBackend().supports(qwen3_moe, job) == []
    full = TrainJob(model="x", data="x", device="rngd:0", method="full")
    assert any("LoRA only" in r for r in NpuLoraTrainBackend().supports(qwen3_moe, full))   # no moe.HostMLP.step
    moe = types.ModuleType("npu_lora.moe")
    moe.HostMLP = type("HostMLP", (), {"step": lambda self: None})
    monkeypatch.setitem(sys.modules, "npu_lora.moe", moe)
    assert NpuLoraTrainBackend().supports(qwen3_moe, full) == []
    hot = TrainJob(model="x", data="x", device="rngd:0", moe_hot_fraction=0.25)
    assert any("device experts" in r for r in NpuLoraTrainBackend().supports(qwen3_moe, hot))


@pytest.fixture
def tiny_llama_dir(tmp_path):
    pytest.importorskip("npu_lora")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        tok = AutoTokenizer.from_pretrained(TOK)
    except Exception as e:
        pytest.skip(f"tokenizer {TOK} not cached: {e}")
    cfg = AutoConfig.for_model("llama", hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                               vocab_size=len(tok), tie_word_embeddings=False, max_position_embeddings=256)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).save_pretrained(tmp_path / "model")
    tok.save_pretrained(tmp_path / "model")
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"text": f"Optimus fact {i}: the answer is {i * 7}."}) for i in range(8)))
    return tmp_path


def test_npu_lora_cpu_reference_backend(tiny_llama_dir):
    job = TrainJob(model=str(tiny_llama_dir / "model"), data=str(tiny_llama_dir / "data.jsonl"), device="cpu",
                   steps=6, seq_len=32, lr=5e-3, lora_r=4, lora_alpha=8, lora_targets=("q_proj", "v_proj"),
                   out=str(tiny_llama_dir / "adapter"))
    res = train(job)
    assert res.backend == "npu-lora" and len(res.losses) == 6
    assert res.losses[-1] < res.losses[0]
    assert os.path.exists(tiny_llama_dir / "adapter" / "adapter_config.json")
    assert dict(res.info["selection"])["optimus"]                    # optimus: no cpu device -> reasons recorded


# ------------------------------------------------- optimus backend output: Hugging Face model / PEFT adapter

def _tiny_hf(tmp_path, tie=False):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model("llama", hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=128,
                               tie_word_embeddings=tie, max_position_embeddings=256)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    model.save_pretrained(tmp_path / "model")
    return model


def _stages(model, n=2):
    """the model split like Optimus_p's IR does (split_module names: "model_layers_0_self_attn_q_proj.weight"),
    layers spread over n stages"""
    import copy
    model = copy.deepcopy(model)
    subs = [torch.nn.Module() for _ in range(n)]
    per = (model.config.num_hidden_layers + n - 1) // n
    for name, m in model.named_modules():
        if list(m.children()) or not list(m.parameters(recurse=False)):
            continue
        s = 0 if name.startswith("model.embed") else n - 1 if not name.startswith("model.layers.") else \
            int(name.split(".")[2]) // per
        subs[s].add_module(name.replace(".", "_"), m)
    return subs


def _fake_opt(sub, stage, n, lora=None):
    """the Optimus_p surface _save_hf uses; save_hf_ckpt is Optimus_p's own"""
    from types import SimpleNamespace
    from opt_prime.opti_pri import Optimus_p
    o = SimpleNamespace(use_gpu=False, run_info=SimpleNamespace(submod=sub), lora_config=lora,
                        tpl=SimpleNamespace(stage=stage, world_size=n, num_stage=n, dp_size=1, tp_size=1),
                        get_rank=lambda: stage)
    o.save_hf_ckpt = lambda d, step=None, epoch=None: Optimus_p.save_hf_ckpt(o, d, step, epoch)
    return o


def _train_step(params):
    """one SGD step on a random objective (the stage modules here have no pipeline forward)"""
    params = [p for p in params if p.requires_grad]
    loss = sum((p.float() * torch.randn_like(p, dtype=torch.float32)).sum() for p in params)
    loss.backward()
    with torch.no_grad():
        for p in params:
            p -= 0.05 * p.grad.sign()
            p.grad = None


@pytest.mark.parametrize("tie", [False, True])
def test_optimus_full_ft_output_loads_with_from_pretrained(tmp_path, tie):
    from transformers import AutoModelForCausalLM
    from opt_prime.accel.train.backends import _save_hf
    from opt_prime.hf_ckpt import build_mangled_map, restore_hf_key
    base = _tiny_hf(tmp_path, tie)
    subs = _stages(base)
    for s in subs:
        _train_step(s.parameters())
    job = TrainJob(model=str(tmp_path / "model"), data="x", method="full", out=str(tmp_path / "out"))
    for stage in (1, 0):                                    # rank 0 assembles after its own part
        _save_hf(_fake_opt(subs[stage], stage, 2), job)
    assert not os.path.exists(tmp_path / "out" / ".stages") and not list((tmp_path / "out").glob("stage*"))
    got = AutoModelForCausalLM.from_pretrained(tmp_path / "out", dtype=torch.bfloat16).state_dict()
    keys = set(base.state_dict())
    mangled = build_mangled_map(keys)
    trained = {restore_hf_key(k, keys, mangled): v for s in subs for k, v in s.state_dict().items()}
    assert None not in trained and set(trained) <= set(got)
    for k, v in trained.items():
        assert torch.equal(got[k], v), k
    w = "model.layers.1.mlp.down_proj.weight"
    assert not torch.equal(got[w], base.state_dict()[w])   # trained, not the base weights


def test_optimus_lora_output_is_a_peft_adapter(tmp_path):
    from safetensors.torch import load_file
    from opt_prime.accel.train.backends import _save_hf
    from opt_prime.lora import LoRAConfig, LoRALinear, apply_lora_to_submod
    base = _tiny_hf(tmp_path)
    subs = _stages(base)
    cfg = LoRAConfig(r=4, alpha=8.0, dropout=0.0, target_modules=["q_proj", "v_proj"])
    for s in subs:
        apply_lora_to_submod(s, cfg)
        _train_step(s.parameters())
    job = TrainJob(model=str(tmp_path / "model"), data="x", method="lora", out=str(tmp_path / "adapter"))
    for stage in (1, 0):
        _save_hf(_fake_opt(subs[stage], stage, 2, cfg), job)
    out = tmp_path / "adapter"
    assert sorted(os.listdir(out)) == ["adapter_config.json", "adapter_model.safetensors"]
    acfg = json.load(open(out / "adapter_config.json"))
    assert (acfg["peft_type"], acfg["r"], acfg["lora_alpha"], acfg["target_modules"]) == ("LORA", 4, 8.0,
                                                                                          ["q_proj", "v_proj"])
    ad = load_file(out / "adapter_model.safetensors")
    assert len(ad) == 2 * 2 * 2                             # 2 layers x {q, v} x {A, B}
    hf_modules = dict(base.named_modules())
    for s in subs:
        for name, m in s.named_modules():
            if not isinstance(m, LoRALinear):
                continue
            hf = name.replace("model_layers_", "model.layers.").replace("_self_attn_", ".self_attn.")
            assert isinstance(hf_modules[hf], torch.nn.Linear)
            assert torch.equal(ad[f"base_model.model.{hf}.lora_A.weight"], m.lora_A.weight)
            assert torch.equal(ad[f"base_model.model.{hf}.lora_B.weight"], m.lora_B.weight)
    try:                                                    # PEFT itself, where installed
        from peft import PeftModel
    except ImportError:
        return
    from transformers import AutoModelForCausalLM
    pm = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(tmp_path / "model", dtype=torch.bfloat16),
                                   str(out))
    sd = {k.replace(".default", ""): v for k, v in pm.state_dict().items() if "lora_" in k}
    for k, v in ad.items():
        assert torch.equal(sd[k].to(v.dtype), v), k


def test_full_ft_merge_fills_frozen_offloaded_experts_from_base(tmp_path):
    """with MoE expert offloading the experts are not stage parameters (frozen on the tiers): the merged model
    takes them from the base model, the trained stage parameters from the stages"""
    from transformers import AutoModelForCausalLM
    from opt_prime.accel.train.backends import _save_hf
    base = _tiny_hf(tmp_path)
    subs = _stages(base)
    held_out = "model_layers_1_mlp_down_proj"                   # stands in for the offloaded experts
    del subs[1]._modules[held_out]
    for s in subs:
        _train_step(s.parameters())
    job = TrainJob(model=str(tmp_path / "model"), data="x", method="full", out=str(tmp_path / "out"))
    for stage in (1, 0):
        o = _fake_opt(subs[stage], stage, 2)
        o.moe_offload_report = object()
        _save_hf(o, job)
    got = AutoModelForCausalLM.from_pretrained(tmp_path / "out", dtype=torch.bfloat16).state_dict()
    ref = base.state_dict()
    assert torch.equal(got["model.layers.1.mlp.down_proj.weight"], ref["model.layers.1.mlp.down_proj.weight"])
    assert not torch.equal(got["model.layers.0.mlp.down_proj.weight"], ref["model.layers.0.mlp.down_proj.weight"])
