#!/usr/bin/env python3
"""
Convert JSON profile files to NPZ format for FASOP.

This script reads JSON profiling results and converts them to NPZ format
with shape (M, N) where:
- M = number of micro batch sizes (rows)
- N = 10 columns: [embed, attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_act_fn, mlp_up, mlp_down, lm_head]

JSON structure (from profile_fx.py):
- profile_result['combined_timing']['embedding']['mean_ms']
- profile_result['combined_timing']['lm_head']['mean_ms']
- profile_result['combined_timing']['modules'] - list of {name, mean_ms}
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Column mapping for NPZ output
COLUMN_NAMES = [
    'embed',
    'attn_q', 'attn_k', 'attn_v', 'attn_o',
    'mlp_gate', 'mlp_act_fn', 'mlp_up', 'mlp_down',
    'lm_head'
]

# Suffix patterns to column index mapping
SUFFIX_MAP = {
    'attn_q': 'self_attn_q_proj',
    'attn_k': 'self_attn_k_proj',
    'attn_v': 'self_attn_v_proj',
    'attn_o': 'self_attn_o_proj',
    'mlp_gate': 'mlp_gate_proj',
    'mlp_act_fn': 'mlp_act_fn',
    'mlp_up': 'mlp_up_proj',
    'mlp_down': 'mlp_down_proj',
}


def extract_config_from_filename(filename: str) -> Optional[Dict]:
    """
    Extract configuration from filename.

    Patterns:
    - profile_<model>-<batch_size>-<micro_batch_size>-<pp>-<tp>-<dp>.json
    - profile_<run_id>.json

    Returns dict with batch_size, micro_batch_size, etc. or None if not parsable.
    """
    basename = os.path.basename(filename)
    if not basename.startswith('profile_') or not basename.endswith('.json'):
        return None

    # Try structured pattern first
    pattern = r'profile_(.+?)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)\.json'
    match = re.match(pattern, basename)
    if match:
        model, batch_size, micro_batch_size, pp, tp, dp = match.groups()
        return {
            'model': model,
            'batch_size': int(batch_size),
            'micro_batch_size': int(micro_batch_size),
            'pp_size': int(pp),
            'tp_size': int(tp),
            'dp_size': int(dp),
        }

    # Fall back to extracting from JSON content
    return None


def load_profile_json(filepath: str) -> Optional[Dict]:
    """Load and validate profile JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'combined_timing' not in data:
            print(f"Warning: {filepath} missing 'combined_timing' key, skipping", file=sys.stderr)
            return None

        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}", file=sys.stderr)
        return None


def extract_timing_row(profile_result: Dict) -> np.ndarray:
    """
    Extract timing data from profile result and build a single row.

    Returns:
        np.ndarray of shape (10,) with values for each column.
        NaN values indicate missing data.
    """
    combined = profile_result.get('combined_timing', {})

    # Extract embedding and lm_head
    embed_mean = combined.get('embedding', {}).get('mean_ms', np.nan)
    lm_head_mean = combined.get('lm_head', {}).get('mean_ms', np.nan)

    # Extract module timings
    modules = combined.get('modules', [])

    # Group module timings by suffix type
    suffix_vals = {k: [] for k in SUFFIX_MAP}

    # Pattern to extract layer number and suffix from module name
    # Example: model_layers_0_self_attn_q_proj -> layer 0, suffix self_attn_q_proj
    name_pattern = re.compile(r'^model_layers_(\d+)_([a-z0-9_]+)$')

    for module in modules:
        name = module.get('name', '')
        mean = module.get('mean_ms')

        if mean is None:
            continue

        match = name_pattern.match(name)
        if not match:
            continue

        # layer_num = match.group(1)  # Not used, but available if needed
        suffix = match.group(2)

        # Map suffix to column key
        for col_key, suffix_pattern in SUFFIX_MAP.items():
            if suffix == suffix_pattern:
                suffix_vals[col_key].append(mean)
                break

    # Build row: average values for each column
    row = [embed_mean]

    for col_name in COLUMN_NAMES[1:]:  # Skip 'embed' (already added)
        if col_name == 'lm_head':
            row.append(float(lm_head_mean) if not np.isnan(lm_head_mean) else np.nan)
        else:
            vals = suffix_vals.get(col_name, [])
            row.append(float(np.mean(vals)) if vals else np.nan)

    return np.array(row, dtype=float)


def get_gpu_type() -> str:
    """Auto-detect GPU type from torch."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            # Normalize common NVIDIA names
            for tag in ["A40", "A100", "H100", "V100", "L40", "L4", "T4"]:
                if tag in name:
                    return tag
            # Fallback: first token without spaces
            return name.split()[0]
    except Exception:
        pass
    return "unknown"


def extract_model_size(profile_result: Dict) -> str:
    """Extract model size string from config (e.g., '70b', '1b')."""
    config = profile_result.get('config', {})
    model_size = str(config.get('model_size', '')).lower()

    # Try to extract number + 'b'
    match = re.search(r'(\d+)\s*b', model_size)
    if match:
        return f"{match.group(1)}b"

    # Fallback: check model_name
    model_name = str(config.get('model_name', '')).lower()
    match = re.search(r'(\d+)\s*b', model_name)
    if match:
        return f"{match.group(1)}b"

    return "unknown"


def process_json_files(input_dir: str, output_dir: str, model_size: Optional[str],
                       gpu_type: Optional[str], tp: Optional[int]) -> None:
    """
    Process all JSON files in input directory and generate NPZ files.

    Groups profiles by (model_size, gpu_type, tp) and stacks rows by micro_batch_size.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Find all JSON files
    json_files = list(input_path.glob("profile_*.json"))

    if not json_files:
        print(f"Warning: No profile_*.json files found in {input_dir}", file=sys.stderr)
        return

    print(f"Found {len(json_files)} JSON files")

    # Group profiles by (model_size, gpu_type, tp, mbs)
    # Key: (model_size, gpu_type, tp, mbs) -> List[row]
    grouped_data: Dict[Tuple[str, str, int, int], List[np.ndarray]] = defaultdict(list)

    for json_file in json_files:
        print(f"Processing {json_file.name}...")

        profile_data = load_profile_json(str(json_file))
        if profile_data is None:
            continue

        # Extract configuration
        config = profile_data.get('config', {})

        # Determine model size
        file_model_size = model_size or extract_model_size(profile_data)

        # Determine GPU type
        file_gpu_type = gpu_type or get_gpu_type()

        # Determine TP
        file_tp = tp if tp is not None else config.get('tp_size', 1)

        # Extract micro batch size
        mbs = config.get('micro_batch_size', 1)

        # Extract timing row
        row = extract_timing_row(profile_data)

        # Group by key
        key = (file_model_size, file_gpu_type, file_tp, mbs)
        grouped_data[key].append(row)

        print(f"  -> model={file_model_size}, gpu={file_gpu_type}, tp={file_tp}, mbs={mbs}")

    if not grouped_data:
        print("Error: No valid profile data extracted", file=sys.stderr)
        return

    # Generate NPZ files
    # Group by (model_size, gpu_type, tp) and stack rows by mbs
    npz_groups: Dict[Tuple[str, str, int], Dict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    for (m_size, g_type, t_size, mbs), rows in grouped_data.items():
        npz_groups[(m_size, g_type, t_size)][mbs].extend(rows)

    # Create NPZ files
    for (m_size, g_type, t_size), mbs_data in npz_groups.items():
        # Sort by mbs for consistent row ordering
        sorted_mbs = sorted(mbs_data.keys())

        # Stack rows: each mbs gets one row (average if multiple profiles)
        rows = []
        for mbs in sorted_mbs:
            mbs_rows = np.array(mbs_data[mbs])
            # Average across multiple profiles with same mbs
            avg_row = np.nanmean(mbs_rows, axis=0)
            rows.append(avg_row)

        data_matrix = np.array(rows)

        # Generate filename
        npz_filename = f"llama{m_size}_{g_type}_{t_size}.npz"
        npz_path = output_path / npz_filename

        # Save NPZ
        np.savez(npz_path, data=data_matrix)

        print(f"\nCreated {npz_filename}")
        print(f"  Shape: {data_matrix.shape} (rows=mbs, cols=operations)")
        print(f"  MBS values: {sorted_mbs}")
        print(f"  Columns: {COLUMN_NAMES}")
        print(f"  Sample row [0]: {data_matrix[0]}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON profile files to NPZ format for FASOP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect all parameters from JSON files
  python make_npz.py

  # Specify model size and GPU type
  python make_npz.py --model-size 70b --gpu-type A40 --tp 2

  # Custom input/output directories
  python make_npz.py --input-dir ./profiles --output-dir ./costs
"""
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='results/',
        help='Directory containing JSON profile files (default: results/)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='../known_cost/',
        help='Directory for NPZ output files (default: ../known_cost/)'
    )

    parser.add_argument(
        '--model-size',
        type=str,
        default=None,
        help='Model size string (e.g., "70b", "1b"). Auto-detected if not specified.'
    )

    parser.add_argument(
        '--gpu-type',
        type=str,
        default=None,
        help='GPU type (e.g., "A40", "A100"). Auto-detected if not specified.'
    )

    parser.add_argument(
        '--tp',
        type=int,
        default=None,
        help='Tensor parallel degree. Extracted from JSON if not specified.'
    )

    args = parser.parse_args()

    print("="*80)
    print("JSON to NPZ Converter for FASOP Profile Database")
    print("="*80)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model size:       {args.model_size or 'auto-detect'}")
    print(f"GPU type:         {args.gpu_type or 'auto-detect'}")
    print(f"TP degree:        {args.tp if args.tp is not None else 'auto-detect'}")
    print("="*80)
    print()

    process_json_files(
        args.input_dir,
        args.output_dir,
        args.model_size,
        args.gpu_type,
        args.tp
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
