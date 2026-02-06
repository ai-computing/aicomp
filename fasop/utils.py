"""
Portions of this code adapted from the 'AMP' project (https://github.com/DachengLi1/AMP). 
@article{li2022amp,
  title={AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness},
  author={Li, Dacheng and Wang, Hongyi and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2210.07297},
  year={2022}
}
"""

from collections import defaultdict
from typing import Optional, Any, Dict
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Logging Utilities
# =============================================================================

class TeeLogger:
    """
    Tee-style logger that writes to both stdout and a file simultaneously.
    """
    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log_file = log_file
        # Create directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.file = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


class JSONLLogger:
    """
    Real-time JSONL logger for search results.
    Each result is written as a single JSON line immediately.
    """
    def __init__(self, jsonl_file: str):
        self.jsonl_file = jsonl_file
        # Create directory if needed
        log_dir = os.path.dirname(jsonl_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.file = open(jsonl_file, 'w', encoding='utf-8')
        self.count = 0

    def log(self, result: Any) -> None:
        """Write a result as a JSON line and print to console."""
        from dataclasses import asdict
        self.count += 1
        result_dict = {"index": self.count, **asdict(result)}
        json_line = json.dumps(result_dict, ensure_ascii=False)
        self.file.write(json_line + '\n')
        self.file.flush()  # Ensure immediate write
        print(json_line)  # Real-time console output

    def log_meta(self, meta: Dict) -> None:
        """Write metadata as a JSON line (e.g., total_count at start)."""
        json_line = json.dumps(meta, ensure_ascii=False)
        self.file.write(json_line + '\n')
        self.file.flush()
        print(json_line)

    def close(self):
        self.file.close()


# =============================================================================
# Parallelism Utilities
# =============================================================================

# returns the rank to axis. If pp_deg=dp_deg=mp_deg=2, rank 3 gives (0,1,1).
# This is deepspeed method
def rank2axis(rank, mp_deg, dp_deg, pp_deg):
    pp = rank // (mp_deg * dp_deg)
    remainder = rank % (mp_deg * dp_deg)

    dp = remainder // (mp_deg)
    remainder = remainder % mp_deg

    mp = remainder

    return (pp, dp, mp)

# returns the axis to rank. If pp_deg=dp_deg=mp_deg=2, (0,1,1) gives 3
def axis2rank(axis, mp_deg, dp_deg, pp_deg):
    pp, dp, mp = axis
    return mp + mp_deg * dp + (mp_deg * dp_deg) * pp

def factor(N, upper=None):
    if upper is None:
        upper = N
    ret = []
    for i in range(1, upper+1):
        if N % i == 0:
            ret.append(i)
    return ret


def count_parallel_strategies(M, N, gbs, num_layers, available_tps=None):
    """
    Count total number of valid parallelism configurations.

    Args:
        M: GPUs per node
        N: Number of nodes
        gbs: Global batch size
        num_layers: Number of model layers (for pp constraint)
        available_tps: List of available TP values from NPZ files (optional).
                      If None, uses factor(min(M, 4)) for backward compatibility.

    Returns:
        Total count of valid (tp, dp, pp, mbs) combinations
    """
    count = 0
    W = M * N

    # Determine valid TP values
    valid_tp_divisors = factor(M)  # TP must divide M (GPUs per node)
    if available_tps is not None:
        # Use intersection of available_tps and valid divisors
        tp_candidates = sorted([tp for tp in available_tps if tp in valid_tp_divisors])
    else:
        # Backward compatibility: use factor(min(M, 4))
        tp_candidates = factor(min(M, 4))

    for tp_deg in tp_candidates:
        if M * N % tp_deg != 0:
            continue
        remain = M * N // tp_deg
        for dp_deg in factor(remain):
            pp_degree = M * N / (tp_deg * dp_deg)
            if pp_degree != int(pp_degree):
                continue
            if (W / pp_degree) % dp_deg != 0:
                continue
            if gbs % dp_deg != 0:
                continue
            if pp_degree > num_layers:
                continue
            for mbs in factor(gbs // dp_deg):
                count += 1
    return count


def enumerate_parallel_strategies(M, N, gbs, known, num_layers, available_tps=None):
    """
    Enumerate all valid parallelism configurations (tp, dp, pp, mbs) for a given cluster.

    Args:
        M: GPUs per node
        N: Number of nodes
        gbs: Global batch size
        known: Previously enumerated configs (for iteration), or None to start fresh
        num_layers: Number of model layers (for pp constraint)
        available_tps: List of available TP values from NPZ files (optional).
                      If None, uses factor(min(M, 4)) for backward compatibility.

    Returns:
        (tp_deg, dp_deg, mbs, known) tuple, or None if no more configs
    """
    if known is None:
        known = defaultdict(list)
        ele_count = 0
        W = M * N

        # Determine valid TP values
        valid_tp_divisors = factor(M)  # TP must divide M (GPUs per node)
        if available_tps is not None:
            # Use intersection of available_tps and valid divisors
            tp_candidates = sorted([tp for tp in available_tps if tp in valid_tp_divisors])
        else:
            # Backward compatibility: use factor(min(M, 4))
            tp_candidates = factor(min(M, 4))

        for tp_deg in tp_candidates:
            if M*N % tp_deg != 0:
                continue
            remain = M*N // tp_deg
            for dp_deg in factor(remain): # data parallelism
                pp_degree = M*N / (tp_deg*dp_deg)
                # if pp_degree is not int
                if pp_degree != int(pp_degree):
                    continue
                if (W / pp_degree) % dp_deg != 0:
                    continue
                if gbs % (dp_deg) != 0:
                    continue
                if pp_degree > num_layers:
                    continue
                for mbs in factor(gbs // dp_deg):
                    ele_count += 1
                    known[mbs].append((tp_deg, dp_deg))
    if len(known.keys()) == 0:
        return None

    mbs = list(known.keys())[0]
    (tp_deg, dp_deg) = known[mbs].pop(0)
    if len(known[mbs]) == 0:
       known.pop(mbs, None)

    return tp_deg, dp_deg, mbs, known


def compute_pareto_frontier(costs: np.ndarray, throughputs: np.ndarray) -> np.ndarray:
    """
    Compute Pareto frontier mask for cost-throughput tradeoff using O(n log n) skyline algorithm.

    A point is on the Pareto frontier if no other point has both:
    - Lower or equal cost AND higher or equal throughput (with at least one strictly better)

    Algorithm:
        1. Sort points by cost (ascending), with throughput (descending) for tie-breaking
        2. Scan once: a point is on frontier if its throughput exceeds all previous points
        
    Args:
        costs: Array of cost values (lower is better)
        throughputs: Array of throughput values (higher is better)

    Returns:
        Boolean mask where True indicates Pareto frontier points
    """
    n = len(costs)
    
    # Sort by cost (ascending), then by throughput (descending) for tie-breaking
    # lexsort sorts by the last key first, so we put costs last
    indices = np.lexsort((-throughputs, costs))
    sorted_throughputs = throughputs[indices]
    
    # Track which points are on the frontier
    is_pareto = np.zeros(n, dtype=bool)
    max_throughput = -np.inf
    
    # Single pass: if throughput exceeds max seen so far, it's on frontier
    for i, idx in enumerate(indices):
        if sorted_throughputs[i] > max_throughput:
            is_pareto[idx] = True
            max_throughput = sorted_throughputs[i]
    
    return is_pareto


def draw_pareto_graph(
    csv_path: str,
    output_path: Optional[str] = None,
    title: str = "LLaMA 70B training pareto frontier analysis"
) -> str:
    """
    Draw Pareto frontier graph from CSV results.

    Args:
        csv_path: Path to CSV file with search results
        output_path: Output PNG path. If None, auto-generates from csv_path.
        title: Graph title

    Returns:
        Path to saved PNG file
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Filter out OOM configurations
    df = df[df['is_oom'] == False].reset_index(drop=True)

    if len(df) == 0:
        print("[WARNING] No valid (non-OOM) configurations found for Pareto graph.")
        return ""

    # Extract cost and throughput
    costs = df['cost($)'].values
    throughputs = df['throughput(samples/s)'].values

    # Compute Pareto frontier
    pareto_mask = compute_pareto_frontier(costs, throughputs)

    # Find highest throughput point
    max_throughput_idx = np.argmax(throughputs)
    max_point = df.iloc[max_throughput_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot non-Pareto points (gray)
    non_pareto_mask = ~pareto_mask
    ax.scatter(
        costs[non_pareto_mask],
        throughputs[non_pareto_mask],
        c='#888888',
        alpha=0.6,
        s=30,
        label='Other configurations',
        zorder=1
    )

    # Plot Pareto frontier points (blue)
    ax.scatter(
        costs[pareto_mask],
        throughputs[pareto_mask],
        c='#3b82f6',
        s=100,
        label='Pareto frontier',
        zorder=3,
        edgecolors='#1d4ed8',
        linewidths=1.5
    )

    # Add annotation for highest throughput point
    annotation_text = (
        f"gbs={int(max_point['gbs'])} mbs={int(max_point['mbs'])} tp={int(max_point['tp'])} "
        f"pp={int(max_point['pp'])} dp={int(max_point['dp'])}\n"
        f"total_time(s)={max_point['total_time(s)']:.2f}"
    )
    ax.annotate(
        annotation_text,
        xy=(costs[max_throughput_idx], throughputs[max_throughput_idx]),
        xytext=(costs[max_throughput_idx] + (costs.max() * 0.1),
                throughputs[max_throughput_idx] - (throughputs.max() * 0.05)),
        fontsize=10,
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='#3b82f6',
            linewidth=1.5
        ),
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='arc3,rad=0',
            color='#3b82f6'
        ),
        zorder=5
    )

    # Styling
    ax.set_xlabel('Training Cost ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Throughput (samples/s)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set axis limits (X starts from 0)
    ax.set_xlim(0, costs.max() * 1.05)
    ax.set_ylim(0, throughputs.max() * 1.1)

    # Background color
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('white')

    # Determine output path
    if output_path is None:
        # Generate from CSV path: same directory, .png extension
        output_path = csv_path.rsplit('.', 1)[0] + '.png'

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return output_path