#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#
# CPU utility to merge stage-level checkpoints (from save_hf_ckpt) into
# a single HuggingFace-compatible model directory.
#
# Usage:
#   python3 merge_hf_ckpt.py \
#     --model meta-llama/Llama-3.2-1B \
#     --ckpt-dir ./hf_ckpt \
#     --output ./merged_model
#
#   python3 merge_hf_ckpt.py \
#     --model meta-llama/Llama-3.2-1B \
#     --ckpt-dir ./hf_ckpt \
#     --output ./merged_model \
#     --token <hf_access_token>
#
# No GPU required. Runs on CPU.
#

import os
import sys
import glob
import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


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


def main():
    parser = argparse.ArgumentParser(description="Merge OptimusPrime stage checkpoints into HuggingFace model")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name (e.g., meta-llama/Llama-3.2-1B)")
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Directory containing stage*_tp*.pt files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for merged HuggingFace model")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace access token")
    args = parser.parse_args()

    # Token resolution: --token > LLAMA_ACCESS_TOKEN env > huggingface-cli login cache
    access_token = args.token or os.getenv('LLAMA_ACCESS_TOKEN')
    # access_token=None is OK — HuggingFace will use cached token from `huggingface-cli login`

    print(f"Loading model config: {args.model}")
    config = AutoConfig.from_pretrained(args.model, token=access_token)

    # Create model structure without pretrained weights (saves memory)
    with torch.device("meta"):
        ref_model = AutoModelForCausalLM.from_config(config)
    original_keys = set(ref_model.state_dict().keys())
    mangled_map = build_mangled_map(original_keys)
    del ref_model

    print(f"Original model has {len(original_keys)} parameters/buffers")

    # Discover checkpoint files
    ckpt_files = sorted(glob.glob(os.path.join(args.ckpt_dir, "stage*_tp*.pt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No stage*_tp*.pt files found in {args.ckpt_dir}")

    print(f"Found {len(ckpt_files)} checkpoint files:")
    for f in ckpt_files:
        print(f"  {os.path.basename(f)}")

    # Load first checkpoint to get metadata
    first_ckpt = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)
    pp_size = first_ckpt['pp_size']
    tp_size = first_ckpt['tp_size']
    print(f"PP size: {pp_size}, TP size: {tp_size}")
    del first_ckpt

    # Merge
    full_sd = {}
    for stage in range(pp_size):
        if tp_size == 1:
            # PP only: single file per stage
            fpath = os.path.join(args.ckpt_dir, f"stage{stage}_tp0.pt")
            ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
            sd = ckpt['state_dict']

            restored_count = 0
            skipped_count = 0
            for key, val in sd.items():
                hf_key = restore_hf_key(key, original_keys, mangled_map)
                if hf_key is None:
                    skipped_count += 1
                    continue
                full_sd[hf_key] = val
                restored_count += 1

            print(f"  Stage {stage}: {restored_count} params restored"
                  f"{f', {skipped_count} non-persistent buffers skipped' if skipped_count else ''}")
            del ckpt, sd
        else:
            # PP + TP: load all TP shards for this stage
            shards = []
            for tp_rank in range(tp_size):
                fpath = os.path.join(args.ckpt_dir, f"stage{stage}_tp{tp_rank}.pt")
                ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
                shards.append(ckpt['state_dict'])
                del ckpt

            restored_count = 0
            skipped_count = 0
            for key in shards[0].keys():
                hf_key = restore_hf_key(key, original_keys, mangled_map)
                if hf_key is None:
                    skipped_count += 1
                    continue

                if is_tp_sharded(hf_key):
                    dim = get_tp_concat_dim(hf_key)
                    vals = [shards[tp_rank][key] for tp_rank in range(tp_size)]
                    full_sd[hf_key] = torch.cat(vals, dim=dim)
                else:
                    # Non-sharded: all TP ranks have identical copy
                    full_sd[hf_key] = shards[0][key]
                restored_count += 1

            print(f"  Stage {stage}: {restored_count} params restored (TP shards merged)"
                  f"{f', {skipped_count} non-persistent buffers skipped' if skipped_count else ''}")
            del shards

    # Verify
    missing = original_keys - set(full_sd.keys())
    extra = set(full_sd.keys()) - original_keys
    if missing:
        print(f"\nWARNING: {len(missing)} missing keys:")
        for k in sorted(missing)[:10]:
            print(f"  {k}")
    if extra:
        print(f"\nWARNING: {len(extra)} extra keys:")
        for k in sorted(extra)[:10]:
            print(f"  {k}")

    if not missing and not extra:
        print(f"\nAll {len(original_keys)} keys matched successfully!")

    # Load into fresh model and save
    print(f"\nLoading merged weights into model...")
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(full_sd, strict=True)

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    print(f"Model saved to {args.output}/")

    # Save tokenizer too
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=access_token)
        tokenizer.save_pretrained(args.output)
        print(f"Tokenizer saved to {args.output}/")
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")

    print(f"\nDone! You can now load the model with:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{args.output}")')


if __name__ == "__main__":
    main()
