import numpy as np
import glob
import os
import argparse

# FX graph node types (10 columns)
# [embed, attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_act_fn, mlp_up, mlp_down, lm_head]
FX_NODE_TYPES = [
    "embed", "attn_q", "attn_k", "attn_v", "attn_o",
    "mlp_gate", "mlp_act_fn", "mlp_up", "mlp_down", "lm_head"
]
LAYER_NODE_TYPES = FX_NODE_TYPES[1:-1]  # 8 layer nodes
SHORT_NAMES = ["Q", "K", "V", "O", "gate", "act", "up", "down"]


def get_mbs_value(row_idx):
    """Get mbs value from row index (mbs = 2^row_idx)"""
    return 2 ** row_idx


def show_profiles(profile_dir=None):
    """Show npz profile files with detailed breakdown."""
    if profile_dir is None:
        profile_dir = os.path.dirname(__file__)

    files = sorted(glob.glob(os.path.join(profile_dir, "*.npz")))
    if not files:
        print("No .npz files found")
        return

    # Collect all data
    all_data = {}
    tp_files = []
    for f in files:
        basename = os.path.basename(f)
        parts = basename.replace(".npz", "").split("_")
        tp = int(parts[-1])
        tp_files.append((tp, f, basename))
    tp_files.sort(key=lambda x: x[0])

    # Load data
    for tp, filepath, basename in tp_files:
        data = np.load(filepath)['data']
        all_data[tp] = {}
        for mbs_idx in range(data.shape[0]):
            mbs = get_mbs_value(mbs_idx)
            all_data[tp][mbs] = data[mbs_idx]

    # Get all mbs values
    all_mbs = sorted(set(mbs for tp_data in all_data.values() for mbs in tp_data.keys()))
    all_tp = sorted(all_data.keys())

    # =========================================================================
    # SECTION 1: Summary Table (embed | layer | lm_head)
    # =========================================================================
    print("\n" + "=" * 90)
    print(" PROFILE SUMMARY (ms)")
    print("=" * 90)

    header = f"{'TP':<4} {'mbs':<5} │ {'embed':>10} {'layer':>10} {'lm_head':>10} │ {'total':>10}"
    print(header)
    print("─" * 4 + "─" * 5 + "─┼" + "─" * 33 + "─┼" + "─" * 11)

    for tp in all_tp:
        for mbs in all_mbs:
            if mbs not in all_data[tp]:
                continue
            row = all_data[tp][mbs]
            embed = row[0]
            layer = np.sum(row[1:-1])
            lm_head = row[-1]
            total = embed + layer + lm_head
            print(f"{tp:<4} {mbs:<5} │ {embed:>10.2f} {layer:>10.2f} {lm_head:>10.2f} │ {total:>10.2f}")
        if tp != all_tp[-1]:
            print("─" * 4 + "─" * 5 + "─┼" + "─" * 33 + "─┼" + "─" * 11)

    # =========================================================================
    # SECTION 2: Layer Breakdown Table
    # =========================================================================
    print("\n" + "=" * 85)
    print(" LAYER BREAKDOWN (ms)")
    print(" [Q=attn_q, K=attn_k, V=attn_v, O=attn_o, gate=mlp_gate, act=mlp_act_fn, up=mlp_up, down=mlp_down]")
    print("=" * 85)

    col_width = 8
    header_parts = [f"{'TP':<3}", f"{'mbs':<4}"]
    for name in SHORT_NAMES:
        header_parts.append(f"{name:>{col_width}}")
    print(" ".join(header_parts))
    print("─" * 85)

    for tp in all_tp:
        for mbs in all_mbs:
            if mbs not in all_data[tp]:
                continue
            row = all_data[tp][mbs]
            layer_nodes = row[1:-1]
            line_parts = [f"{tp:<3}", f"{mbs:<4}"]
            for val in layer_nodes:
                line_parts.append(f"{val:>{col_width}.2f}")
            print(" ".join(line_parts))
        if tp != all_tp[-1]:
            print()

    # =========================================================================
    # SECTION 3: Scaling Statistics
    # =========================================================================
    if 1 not in all_data or 1 not in all_data[1]:
        print("\nBaseline (TP=1, mbs=1) not found, skipping statistics")
        return

    baseline = all_data[1][1]
    base_layer = np.sum(baseline[1:-1])

    print("\n" + "=" * 70)
    print(" SCALING RATIOS (baseline: TP=1, mbs=1)")
    print("=" * 70)

    header = f"{'TP':<4} {'mbs':<5} │ {'layer_ratio':>12} {'vs TP=1':>12} {'vs mbs=1':>12}"
    print(header)
    print("─" * 4 + "─" * 5 + "─┼" + "─" * 39)

    for tp in all_tp:
        tp_base_layer = np.sum(all_data[tp][1][1:-1]) if 1 in all_data[tp] else None
        for mbs in all_mbs:
            if mbs not in all_data[tp]:
                continue
            row = all_data[tp][mbs]
            layer = np.sum(row[1:-1])

            ratio_base = layer / base_layer if base_layer > 0 else 0

            if 1 in all_data and mbs in all_data[1]:
                tp1_layer = np.sum(all_data[1][mbs][1:-1])
                ratio_tp1 = layer / tp1_layer if tp1_layer > 0 else 0
            else:
                ratio_tp1 = float('nan')

            if tp_base_layer:
                ratio_mbs1 = layer / tp_base_layer
            else:
                ratio_mbs1 = float('nan')

            print(f"{tp:<4} {mbs:<5} │ {ratio_base:>12.3f} {ratio_tp1:>12.3f} {ratio_mbs1:>12.3f}")
        if tp != all_tp[-1]:
            print()

    # =========================================================================
    # SECTION 4: Quick Reference
    # =========================================================================
    print("\n" + "=" * 50)
    print(" QUICK REFERENCE")
    print("=" * 50)
    print(f" Baseline layer time: {base_layer:.2f} ms (TP=1, mbs=1)")
    print(f" Files loaded: {len(tp_files)}")
    print(f" TP values: {all_tp}")
    print(f" MBS values: {all_mbs}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProfileDB inspection utility")
    parser.add_argument("--dir", type=str, default=None, help="Profile directory path")
    args = parser.parse_args()

    show_profiles(profile_dir=args.dir)
