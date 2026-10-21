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
