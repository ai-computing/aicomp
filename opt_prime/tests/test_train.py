#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# opt_prime.train: backend selection per model / device, and npu-lora (CPU reference backend) on a tiny Llama.
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opt_prime.model import model_spec  # noqa: E402
from opt_prime.moe_offload.testing import build_tiny  # noqa: E402
from opt_prime.train import NpuLoraTrainBackend, TrainJob, select_train_backend, train  # noqa: E402

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
    if NpuLoraTrainBackend().supports(llama, job("rngd:0")) == []:      # npu-lora installed
        assert select_train_backend(llama, job("rngd:0"))[0].name == "npu-lora"
    with pytest.raises(ValueError, match="npu-lora|optimus"):
        select_train_backend(qwen3_moe, job("rngd:0"))                   # no backend trains MoE on RNGD yet


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
