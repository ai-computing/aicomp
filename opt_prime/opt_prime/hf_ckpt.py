#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#
# Stage checkpoints -> Hugging Face formats, on CPU:
#   merge_hf_ckpt()     stage{N}_tp{M}.pt files from Optimus_p.save_hf_ckpt() -> one model directory that
#                       AutoModelForCausalLM.from_pretrained() loads (examples/merge_hf_ckpt.py is its CLI)
#   save_peft_adapter() opt_prime.lora adapter weights of every stage -> a PEFT adapter
#                       (adapter_config.json + adapter_model.safetensors)
#
import glob
import json
import os

import torch


# TP sharding plan for LLaMA (extend for other models as needed)
COLWISE_NAMES = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}  # concat dim=0
ROWWISE_NAMES = {"o_proj", "down_proj"}                                   # concat dim=1


def restore_hf_key(key, original_keys_set, mangled_map):
    """Restore a mangled state_dict key back to the original HuggingFace FQN.

    After split_module(), keys come in several forms:
      1. Original FQN preserved (call_module internal params):
         "model.layers.0.self_attn.q_proj.weight"  → already in original_keys
      2. Underscore-mangled (call_module with renamed submodules):
         "model_layers_0_self_attn_q_proj.weight"  → dot at attr boundary
      3. Fully mangled (moved_ parameters, no dots):
         "moved_model_layers_0_input_layernorm_weight"
      4. Underscore-mangled without moved_ prefix:
         "model_embed_tokens.weight"

    Strategy: strip "moved_" if present, then convert the entire key to
    underscore form and look up in the mangled_map.
    """
    # Already in original form
    if key in original_keys_set:
        return key

    # Strip "moved_" prefix if present
    bare = key[len("moved_"):] if key.startswith("moved_") else key

    # Normalize: replace remaining dots with underscores for uniform lookup
    normalized = bare.replace(".", "_")

    if normalized in mangled_map:
        return mangled_map[normalized]

    # Non-persistent buffers (e.g., rotary_emb.inv_freq) are not in
    # the original state_dict — they are regenerated at model init.
    # Return None to signal the caller to skip this key.
    return None


def is_tp_sharded(hf_key):
    """Check if a key corresponds to a TP-sharded parameter."""
    for name in COLWISE_NAMES | ROWWISE_NAMES:
        if f".{name}." in hf_key and hf_key.endswith(".weight"):
            return True
    return False


def get_tp_concat_dim(hf_key):
    """Get the concatenation dimension for TP shard reconstruction."""
    for name in COLWISE_NAMES:
        if f".{name}." in hf_key:
            return 0
    for name in ROWWISE_NAMES:
        if f".{name}." in hf_key:
            return 1
    return None


def build_mangled_map(original_keys):
    """Build mangled_full_key -> original_key mapping for FQN restoration.

    move_parameters() applies node.target.replace('.', '_') to the ENTIRE FQN:
      "model.layers.0.input_layernorm.weight" -> "model_layers_0_input_layernorm_weight"

    We build the reverse: "model_layers_0_input_layernorm_weight" -> "model.layers.0.input_layernorm.weight"
    """
    mangled_map = {}
    for key in original_keys:
        mangled = key.replace(".", "_")
        mangled_map[mangled] = key
    return mangled_map


def hf_state_keys(model, token=None):
    """-> (config, the model's state_dict keys); the model is built on the meta device (no weights read)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model, token=token)
    with torch.device("meta"):
        ref_model = AutoModelForCausalLM.from_config(config)
    return config, set(ref_model.state_dict().keys())


def merge_stage_state_dicts(ckpt_dir, original_keys, log=print):
    """stage*_tp*.pt in ckpt_dir -> {HF key: tensor} (TP shards concatenated)"""
    mangled_map = build_mangled_map(original_keys)
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "stage*_tp*.pt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No stage*_tp*.pt files found in {ckpt_dir}")

    log(f"Found {len(ckpt_files)} checkpoint files:")
    for f in ckpt_files:
        log(f"  {os.path.basename(f)}")

    # Load first checkpoint to get metadata
    first_ckpt = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)
    pp_size = first_ckpt['pp_size']
    tp_size = first_ckpt['tp_size']
    log(f"PP size: {pp_size}, TP size: {tp_size}")
    del first_ckpt

    full_sd = {}
    for stage in range(pp_size):
        # all TP shards of this stage (one file when tp_size == 1)
        shards = [torch.load(os.path.join(ckpt_dir, f"stage{stage}_tp{tp_rank}.pt"), map_location="cpu",
                             weights_only=False)['state_dict'] for tp_rank in range(tp_size)]
        restored_count = 0
        skipped_count = 0
        for key in shards[0].keys():
            hf_key = restore_hf_key(key, original_keys, mangled_map)
            if hf_key is None:
                skipped_count += 1
                continue
            if tp_size > 1 and is_tp_sharded(hf_key):
                full_sd[hf_key] = torch.cat([s[key] for s in shards], dim=get_tp_concat_dim(hf_key))
            else:
                # Non-sharded: all TP ranks have identical copy
                full_sd[hf_key] = shards[0][key]
            restored_count += 1

        log(f"  Stage {stage}: {restored_count} params restored{' (TP shards merged)' if tp_size > 1 else ''}"
            f"{f', {skipped_count} non-persistent buffers skipped' if skipped_count else ''}")
        del shards
    return full_sd


def merge_hf_ckpt(model, ckpt_dir, output, token=None, dtype=None, log=print):
    """Stage checkpoints of `model` (HF id or path) in ckpt_dir -> a HuggingFace model directory `output`
    (weights + config + tokenizer); dtype: the saved weights' dtype (None: transformers' default)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"Loading model config: {model}")
    config, original_keys = hf_state_keys(model, token)
    log(f"Original model has {len(original_keys)} parameters/buffers")
    full_sd = merge_stage_state_dicts(ckpt_dir, original_keys, log)

    # Verify
    missing = original_keys - set(full_sd.keys())
    extra = set(full_sd.keys()) - original_keys
    if missing:
        log(f"\nWARNING: {len(missing)} missing keys:")
        for k in sorted(missing)[:10]:
            log(f"  {k}")
    if extra:
        log(f"\nWARNING: {len(extra)} extra keys:")
        for k in sorted(extra)[:10]:
            log(f"  {k}")
    if not missing and not extra:
        log(f"\nAll {len(original_keys)} keys matched successfully!")

    # Load into fresh model and save
    log(f"\nLoading merged weights into model...")
    hf_model = AutoModelForCausalLM.from_config(config, **({} if dtype is None else {"dtype": dtype}))
    hf_model.load_state_dict(full_sd, strict=True)
    try:                                              # the model's own generation defaults (eos ids, sampling)
        from transformers import GenerationConfig
        hf_model.generation_config = GenerationConfig.from_pretrained(model, token=token)
    except Exception:
        pass

    os.makedirs(output, exist_ok=True)
    hf_model.save_pretrained(output)
    log(f"Model saved to {output}/")

    # Save tokenizer too
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, token=token)
        tokenizer.save_pretrained(output)
        log(f"Tokenizer saved to {output}/")
    except Exception as e:
        log(f"Warning: Could not save tokenizer: {e}")
    return output


def lora_state_to_peft(lora_sd, original_keys):
    """opt_prime.lora state ("model_layers_0_self_attn_q_proj.lora_A.weight") -> PEFT names
    ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight")"""
    mangled_map = build_mangled_map(original_keys)
    out = {}
    for key, val in lora_sd.items():
        mod, ab = key.rsplit(".lora_", 1)                       # ab: "A.weight" | "B.weight"
        base = restore_hf_key(mod + ".weight", original_keys, mangled_map)
        if base is None:
            raise KeyError(f"LoRA module {mod} has no Linear weight in the HF model")
        val = val.to_local() if hasattr(val, "to_local") else val
        out[f"base_model.model.{base[:-len('.weight')]}.lora_{ab}"] = val.detach().cpu().contiguous()
    return out


def save_peft_adapter(output, lora_sd, base_model, lora_config, original_keys, dtype=torch.bfloat16):
    """opt_prime.lora weights of all stages (merged dict) -> PEFT adapter directory `output`; scaling alpha/r
    matches PEFT's (no rslora)."""
    from safetensors.torch import save_file
    os.makedirs(output, exist_ok=True)
    tensors = {k: v.to(dtype) for k, v in lora_state_to_peft(lora_sd, original_keys).items()}
    save_file(tensors, os.path.join(output, "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_model,
           "r": lora_config.r, "lora_alpha": lora_config.alpha, "lora_dropout": lora_config.dropout, "bias": "none",
           "target_modules": sorted({k.split(".")[-3] for k in tensors}),       # the modules that got adapters
           "fan_in_fan_out": False, "inference_mode": True,
           "use_rslora": False, "modules_to_save": None}
    with open(os.path.join(output, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return output
