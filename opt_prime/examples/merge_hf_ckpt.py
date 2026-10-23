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
# No GPU required. Runs on CPU. The merge itself lives in opt_prime.hf_ckpt
# (also used by opt_prime.accel.train for its Hugging Face output).
#

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.hf_ckpt import (merge_hf_ckpt, restore_hf_key, is_tp_sharded,  # noqa: F401,E402
                               get_tp_concat_dim, build_mangled_map)


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

    merge_hf_ckpt(args.model, args.ckpt_dir, args.output, token=access_token)

    print(f"\nDone! You can now load the model with:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{args.output}")')


if __name__ == "__main__":
    main()
