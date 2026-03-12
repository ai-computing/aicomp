"""
ILP partition test with llama70b @ num_layers=10.

Builds cost_e / cost_c from the NPZ profile directly,
then compares ILP vs DP vs minmax vs bruteforce results.
"""

import sys
import numpy as np
import torch

from config import (
    MODEL_CONFIGS, FX_NODE_TYPES, FX_NODE_INDEX,
    LAYER_NODE_TYPES, NODES_PER_LAYER, get_mbs_index,
)
from partition import get_node_type_sequence, get_total_nodes
from pipe import (
    ILP, dynamic_programming2, minmax, exhaustive_partition,
    _solve_ilp_cpsat, _get_cost_e_for_gpu, _get_comm_cost,
    get_stage_latency, _extract_stage_times,
)


# ── Settings ──────────────────────────────────────────────────────
NUM_LAYERS = 10          # reduced from 80
PP_DEGREE  = 4           # pipeline parallel stages
TP_DEGREE  = 2           # tensor parallel (selects which NPZ to load)
MBS        = 4           # micro-batch size
GBS        = 32          # global batch size
DP_DEGREE  = GBS // (MBS * PP_DEGREE)  # will be used for num_mb
NUM_MB     = max(1, GBS // (DP_DEGREE * MBS) if DP_DEGREE > 0 else GBS // MBS)

# ── 1. Build node sequence ────────────────────────────────────────
node_sequence = get_node_type_sequence(NUM_LAYERS)
num_nodes = len(node_sequence)

print(f"=== ILP Test: llama70b with {NUM_LAYERS} layers ===")
print(f"Total FX nodes : {num_nodes}  (1 embed + {NUM_LAYERS}×8 layers + 1 lm_head)")
print(f"PP={PP_DEGREE}, TP={TP_DEGREE}, MBS={MBS}, NUM_MB={NUM_MB}")
print()

# ── 2. Build cost_e from NPZ profile ─────────────────────────────
npz_path = f"known_cost/llama70b_A40_{TP_DEGREE}.npz"
try:
    profile_raw = np.load(npz_path)['data']
except FileNotFoundError:
    print(f"ERROR: Profile not found: {npz_path}")
    sys.exit(1)

profile_data = 3.0 * profile_raw  # forward + backward multiplier

mbs_idx = get_mbs_index(MBS)
if mbs_idx >= profile_data.shape[0]:
    mbs_idx = profile_data.shape[0] - 1

cost_e = np.zeros(num_nodes)
for i, node_type in enumerate(node_sequence):
    if node_type in FX_NODE_INDEX:
        node_idx = FX_NODE_INDEX[node_type]
        if node_idx < profile_data.shape[1]:
            cost_e[i] = profile_data[mbs_idx, node_idx]

cost_e = cost_e / 1000.0  # ms → sec

# Use same profile for all stages (homogeneous A40 cluster)
cost_e_per_gpu = {"A40": cost_e}
gpu_type_lst = ["A40"] * PP_DEGREE

print(f"cost_e shape: {cost_e.shape}")
print(f"cost_e sample (first 10): {cost_e[:10].round(6)}")
print(f"cost_e total : {cost_e.sum():.6f} sec")
print()

# ── 3. Build cost_c (simplified: uniform comm cost) ──────────────
# Approximate: activation = mbs * seq_len * hidden_size * precision / bandwidth
cfg = MODEL_CONFIGS["llama70b"]
h = cfg["hidden_size"]       # 8192
s = cfg["sequence_length"]   # 1024
precision_bits = 16
bandwidth = 100 * 1e9        # 100 GB/s inter-node

# Per-node communication volume depends on node type
def node_comm_volume(node_type):
    """Activation bytes if stage boundary is after this node."""
    if node_type in ("embed", "attn_o", "mlp_down"):
        return MBS * s * h * precision_bits / bandwidth
    elif node_type in ("attn_q", "attn_k", "attn_v"):
        return MBS * s * h // TP_DEGREE * precision_bits / bandwidth
    elif node_type in ("mlp_gate", "mlp_act_fn", "mlp_up"):
        return MBS * s * 4 * h // TP_DEGREE * precision_bits / bandwidth
    elif node_type == "lm_head":
        return MBS * s * cfg["vocab_size"] * precision_bits / bandwidth
    return MBS * s * h * precision_bits / bandwidth

# cost_c shape: (num_nodes, pp_degree - 1)
cost_c = np.zeros((num_nodes, PP_DEGREE - 1))
for i, nt in enumerate(node_sequence):
    vol = node_comm_volume(nt)
    for j in range(PP_DEGREE - 1):
        cost_c[i][j] = vol

print(f"cost_c shape: {cost_c.shape}")
print()

# ── 4. Run partition methods ──────────────────────────────────────
import time

methods = {}
timings = {}

# Shared prefix sums for ILP solvers
cost_e_list = [_get_cost_e_for_gpu(cost_e_per_gpu, gpu_type_lst[j]) for j in range(PP_DEGREE)]
prefix = []
for j in range(PP_DEGREE):
    psum = [0.0]
    for i in range(num_nodes):
        psum.append(psum[-1] + float(cost_e_list[j][i]))
    prefix.append(psum)
max_comp = max(prefix[j][num_nodes] for j in range(PP_DEGREE))
max_comm = float(np.max(cost_c)) if cost_c.size else 0.0
M = (max_comp + max_comm) * 2 + 1e6

def _partition_to_result(partition):
    """Convert a partition list to the standard result tuple via get_stage_latency."""
    stage_latency = get_stage_latency(partition, cost_e_per_gpu, cost_c, gpu_type_lst)
    return (partition,) + _extract_stage_times(stage_latency)

# ILP (CP-SAT)
try:
    t0 = time.perf_counter()
    partition = _solve_ilp_cpsat(num_nodes, prefix, cost_c, PP_DEGREE, NUM_MB, M, verbose=True)
    timings["ILP(CP-SAT)"] = time.perf_counter() - t0
    if partition:
        methods["ILP(CP-SAT)"] = _partition_to_result(partition)
    else:
        print("ILP(CP-SAT): No solution found")
except Exception as e:
    print(f"ILP(CP-SAT) FAILED: {e}")

# ILP (SCIP/MIP) - force MIP path by calling the MIP solver directly
try:
    from ortools.linear_solver import pywraplp
    from pipe import _create_mip_solver

    t0 = time.perf_counter()
    solver, solver_name = _create_mip_solver(pywraplp)
    infinity = solver.infinity()
    cut_vars = {}
    cut_ints = []

    for j in range(PP_DEGREE - 1):
        lb, ub = j + 1, num_nodes - (PP_DEGREE - 1 - j)
        for k in range(lb, ub + 1):
            cut_vars[(j, k)] = solver.BoolVar(f'y_{j}_{k}')
        cut_int = solver.IntVar(lb, ub, f'cut_{j}')
        solver.Add(cut_int == sum(k * cut_vars[(j, k)] for k in range(lb, ub + 1)))
        cut_ints.append(cut_int)
        solver.Add(sum(cut_vars[(j, k)] for k in range(lb, ub + 1)) == 1)

    for j in range(len(cut_ints) - 1):
        solver.Add(cut_ints[j] + 1 <= cut_ints[j + 1])

    stage_costs = [solver.NumVar(0.0, infinity, f'stage_cost_{j}') for j in range(PP_DEGREE)]

    lb0, ub0 = 1, num_nodes - (PP_DEGREE - 1)
    comm0 = _get_comm_cost(cost_c, 0, 0)
    for k in range(lb0, ub0 + 1):
        val = prefix[0][k] + comm0
        y = cut_vars[(0, k)]
        solver.Add(stage_costs[0] <= val + M * (1 - y))
        solver.Add(stage_costs[0] >= val - M * (1 - y))

    for j in range(1, PP_DEGREE - 1):
        lb_prev = j
        ub_prev = num_nodes - (PP_DEGREE - j)
        lb_curr = j + 1
        ub_curr = num_nodes - (PP_DEGREE - 1 - j)
        for k_start in range(lb_prev, ub_prev + 1):
            for k_end in range(max(k_start + 1, lb_curr), ub_curr + 1):
                comp_val = prefix[j][k_end] - prefix[j][k_start]
                comm_val = _get_comm_cost(cost_c, k_start, j - 1)
                val = comp_val + comm_val
                both = solver.BoolVar(f'both_{j}_{k_start}_{k_end}')
                solver.Add(both <= cut_vars[(j - 1, k_start)])
                solver.Add(both <= cut_vars[(j, k_end)])
                solver.Add(both >= cut_vars[(j - 1, k_start)] + cut_vars[(j, k_end)] - 1)
                solver.Add(stage_costs[j] <= val + M * (1 - both))
                solver.Add(stage_costs[j] >= val - M * (1 - both))

    lb_last = PP_DEGREE - 1
    ub_last = num_nodes - 1
    for k in range(lb_last, ub_last + 1):
        val = prefix[-1][num_nodes] - prefix[-1][k] + _get_comm_cost(cost_c, k, PP_DEGREE - 2)
        y = cut_vars[(PP_DEGREE - 2, k)]
        solver.Add(stage_costs[-1] <= val + M * (1 - y))
        solver.Add(stage_costs[-1] >= val - M * (1 - y))

    max_stage_cost = solver.NumVar(0.0, infinity, 'max_stage_cost')
    for j in range(PP_DEGREE):
        solver.Add(max_stage_cost >= stage_costs[j])

    solver.Minimize((NUM_MB - 1) * max_stage_cost + sum(stage_costs))

    print(f"ILP ({solver_name}): variables={solver.NumVariables()}, constraints={solver.NumConstraints()}")
    status = solver.Solve()

    timings[f"ILP({solver_name})"] = time.perf_counter() - t0

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        cut_values = [0] + [int(cut_ints[j].solution_value()) for j in range(len(cut_ints))] + [num_nodes]
        partition = [cut_values[j + 1] - cut_values[j] for j in range(PP_DEGREE)]
        print(f"ILP ({solver_name}) solution: partition={partition}, cuts={cut_values[1:-1]}")
        methods[f"ILP({solver_name})"] = _partition_to_result(partition)
    else:
        print(f"ILP ({solver_name}): No solution found")
except Exception as e:
    print(f"ILP(MIP) FAILED: {e}")

# DP
try:
    t0 = time.perf_counter()
    result = dynamic_programming2(num_nodes, cost_e_per_gpu, cost_c, PP_DEGREE, NUM_MB, gpu_type_lst)
    timings["DP"] = time.perf_counter() - t0
    methods["DP"] = result
except Exception as e:
    print(f"DP FAILED: {e}")

# Minmax
try:
    t0 = time.perf_counter()
    result = minmax(num_nodes, cost_e_per_gpu, cost_c, PP_DEGREE, gpu_type_lst)
    timings["Minmax"] = time.perf_counter() - t0
    methods["Minmax"] = result
except Exception as e:
    print(f"Minmax FAILED: {e}")

# Bruteforce (feasible because num_nodes=82, pp=4 is still manageable... actually might be slow)
if num_nodes <= 30:  # only for very small
    try:
        t0 = time.perf_counter()
        result = exhaustive_partition(num_nodes, cost_e_per_gpu, cost_c, PP_DEGREE, gpu_type_lst)
        timings["Bruteforce"] = time.perf_counter() - t0
        methods["Bruteforce"] = result
    except Exception as e:
        print(f"Bruteforce FAILED: {e}")
else:
    print(f"Bruteforce skipped (num_nodes={num_nodes} too large for exhaustive search)")

# ── 5. Compare results ───────────────────────────────────────────
print()
print("=" * 80)
print(f"{'Method':<12} {'Partition':<30} {'Max Stage':>12} {'Total':>12} {'Time(ms)':>10}")
print("=" * 80)

for name, (partition, comp_lst, comm_lst, time_lst, *_) in methods.items():
    max_stage = max(time_lst)
    total = sum(time_lst)
    elapsed = timings.get(name, 0) * 1000
    part_str = str(partition)
    if len(part_str) > 28:
        part_str = part_str[:25] + "..."
    print(f"{name:<12} {part_str:<30} {max_stage:>12.6f} {total:>12.6f} {elapsed:>10.2f}")

# ── 6. Validate ILP partition ────────────────────────────────────
ilp_key = next((k for k in methods if k.startswith("ILP")), None)
if ilp_key:
    partition = methods[ilp_key][0]
    print()
    print("--- ILP Partition Validation ---")
    print(f"Partition: {partition}")
    print(f"Sum = {sum(partition)} (expected {num_nodes})")
    assert sum(partition) == num_nodes, f"FAIL: sum={sum(partition)} != {num_nodes}"
    assert all(p >= 1 for p in partition), f"FAIL: empty stage in {partition}"
    print("[OK] All stages have >= 1 node")
    print("[OK] Partition sums to total nodes")

    # Show what nodes each stage gets
    idx = 0
    for stage_id, count in enumerate(partition):
        nodes = node_sequence[idx:idx+count]
        layers_in_stage = set()
        for n_i, nt in enumerate(nodes):
            if nt == "embed":
                layers_in_stage.add("embed")
            elif nt == "lm_head":
                layers_in_stage.add("lm_head")
            else:
                layer_idx = (idx + n_i - 1) // NODES_PER_LAYER
                layers_in_stage.add(f"L{layer_idx}")
        print(f"  Stage {stage_id}: {count} nodes → {sorted(layers_in_stage)}")
        idx += count

print()
print("Done.")
