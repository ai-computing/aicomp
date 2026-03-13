"""
Pipeline scheduling and partitioning module for FASOP.

This module provides functions for:
- Stage latency calculation
- Partition optimization (minmax, DP, ILP, exhaustive)
- Pipeline scheduling simulation
"""

import torch
import time
import numpy as np
import copy
from typing import Dict, List, Optional, Union

from stage import PPGroup


# Default GPU type for fallback when GPU type not found in cost_e_per_gpu
DEFAULT_GPU_TYPE = "A10"


class Stage:
    def __init__(self):
        self.comm_time = 0.
        self.comp_time = 0.
        self.for_send_time = 0.
        self.back_send_time = 0.

    def set_comp_time(self, comp_time):
        self.comp_time = comp_time

    def set_comm_time(self, comm_time):
        self.comm_time = comm_time
    
    def set_for_send_time(self, for_send_time):
        self.for_send_time = for_send_time
    
    def set_back_send_time(self, back_send_time):
        self.back_send_time = back_send_time

    def get_comp_time(self):
        return self.comp_time
    
    def get_comm_time(self):
        return self.comm_time
    
    def get_for_send_time(self):
        return self.for_send_time

    def get_back_send_time(self):
        return self.back_send_time

    def get_stage_time(self):
        return self.comm_time+self.comp_time


def _extract_stage_times(stage_latency: List['Stage']):
    """
    Extract all timing lists from stage latency objects.

    Args:
        stage_latency: List of Stage objects

    Returns:
        Tuple of (stage_time_lst, stage_comp_time_lst, stage_comm_time_lst,
                  stage_for_send_time_lst, stage_back_send_time_lst)
    """
    return (
        [s.get_stage_time() for s in stage_latency],
        [s.get_comp_time() for s in stage_latency],
        [s.get_comm_time() for s in stage_latency],
        [s.get_for_send_time() for s in stage_latency],
        [s.get_back_send_time() for s in stage_latency],
    )


def minmax(
    num_layer: int,
    cost_e_per_gpu: Dict[str, np.ndarray],
    cost_c: np.ndarray,
    pp_degree: int,
    gpu_type_lst: List[str],
    even_split: bool = False,
    verbose: bool = False
):
    """
    Min-max partitioning algorithm for load balancing.

    Args:
        num_layer: Total number of layers/nodes
        cost_e_per_gpu: Dict mapping GPU type to execution cost array
        cost_c: Communication cost array
        pp_degree: Pipeline parallel degree
        gpu_type_lst: List of GPU types for each stage
        even_split: If True, use even split without optimization
        verbose: If True, print partition progress for debugging

    Returns:
        Tuple of (partition, stage_comp_time_lst, stage_comm_time_lst,
                  stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst)
    """
    num_balanced_layer = num_layer // pp_degree
    partition = []
    for i in range(pp_degree):
        partition.append(num_balanced_layer)
    rest = int(num_layer - (num_balanced_layer * pp_degree))
    for i in range(rest):
        partition[i-1] += 1

    partition_history = []
    partition_history.append(partition[:])

    last_max_latency = 1000000
    counted = False
    max_latency_index = 0
    min_latency_index = 0

    if even_split:
        stage_latency = get_stage_latency(partition, cost_e_per_gpu, cost_c, gpu_type_lst)
    else:
        while(1):
            stage_latency = get_stage_latency(partition, cost_e_per_gpu, cost_c, gpu_type_lst)
            stage_time_lst = [stage.get_stage_time() for stage in stage_latency]


            max_latency = max(stage_time_lst)
            if max_latency > last_max_latency:
                if counted:
                    partition[max_latency_index] += 1
                    partition[min_latency_index] -= 1
                    stage_latency = get_stage_latency(partition, cost_e_per_gpu, cost_c, gpu_type_lst)
                    break
            if max_latency == last_max_latency:
                if counted and partition in partition_history[:-1]:
                    partition[max_latency_index] += 1
                    partition[min_latency_index] -= 1
                    stage_latency = get_stage_latency(partition, cost_e_per_gpu, cost_c, gpu_type_lst)
                    break
            last_max_latency = max_latency
            max_latency_index = stage_time_lst.index(max_latency)

            min_latency = min(stage_time_lst)
            min_latency_index = stage_time_lst.index(min_latency)

            if (max_latency_index == 0 or max_latency_index == pp_degree-1) and partition[max_latency_index] == 2:
                if counted:
                    partition[max_latency_index] += 1
                    partition[min_latency_index] -= 1
                break
            if partition[max_latency_index]>1:
                partition[max_latency_index] -= 1
                partition[min_latency_index] += 1
                counted=True
                partition_history.append(partition[:])
            else: # no layers to substract
                break
    
    stage_time_lst, stage_comp_time_lst, stage_comm_time_lst, stage_for_send_time_lst, stage_back_send_time_lst = _extract_stage_times(stage_latency)

    return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst


def explain_minmax(
    num_layer: int,
    cost_e_per_gpu: Dict[str, np.ndarray],
    cost_c: np.ndarray,
    pp_degree: int,
    gpu_type_lst: List[str]
):
    """
    Min-max partitioning with verbose output for debugging.

    This is a convenience wrapper for minmax() with verbose=True.
    See minmax() for full documentation.
    """
    return minmax(
        num_layer=num_layer,
        cost_e_per_gpu=cost_e_per_gpu,
        cost_c=cost_c,
        pp_degree=pp_degree,
        gpu_type_lst=gpu_type_lst,
        even_split=False,
        verbose=True
    )


def get_stage_latency(
    partition: List[int],
    cost_e_per_gpu: Dict[str, np.ndarray],
    cost_c: np.ndarray,
    gpu_type_lst: List[str]
) -> List[Stage]:
    """
    Calculate latency for each pipeline stage.
    
    Args:
        partition: List of node/layer counts per stage [n0, n1, n2, ...]
        cost_e_per_gpu: Dict mapping GPU type to execution cost array
                        e.g., {"A100": array, "A40": array, "A10": array}
        cost_c: Communication cost array
        gpu_type_lst: List of GPU types for each stage
    
    Returns:
        List of Stage objects with computed latencies
    """
    num_bw_share = 1  # which should be calculated in get_cost_c considering PCIe
    num_stage = len(partition)

    stage_latency = [Stage() for _ in range(num_stage)]

    if num_stage == 1:
        gpu_type = gpu_type_lst[0]
        cost_e = _get_cost_e_for_gpu(cost_e_per_gpu, gpu_type)
        stage_latency[0].set_comp_time(sum(cost_e))
        return stage_latency
    
    for stage in range(num_stage):
        num_layer_til_last_stage = sum(partition[:stage])
        num_layer_til_cur_stage = sum(partition[:stage+1])
        
        # Get cost_e for this stage's GPU type
        gpu_type = gpu_type_lst[stage]
        cost_e = _get_cost_e_for_gpu(cost_e_per_gpu, gpu_type)

        if stage == 0:
            stage_latency[stage].set_comp_time(sum(cost_e[:num_layer_til_cur_stage]))
            stage_latency[stage].set_for_send_time((cost_c[sum(partition[:stage])][stage]*num_bw_share).item())
        elif stage == num_stage-1:
            stage_latency[stage].set_comp_time(sum(cost_e[num_layer_til_last_stage:num_layer_til_cur_stage]))
            stage_latency[stage].set_back_send_time((cost_c[sum(partition[:stage])][stage-1]*num_bw_share).item())
        else:
            stage_latency[stage].set_comp_time(sum(cost_e[num_layer_til_last_stage:num_layer_til_cur_stage]))
            stage_latency[stage].set_comm_time((cost_c[sum(partition[:stage])][stage-1]*num_bw_share).item())
            
    return stage_latency


def _get_cost_e_for_gpu(
    cost_e_per_gpu: Dict[str, np.ndarray],
    gpu_type: str
) -> np.ndarray:
    """
    Get execution cost array for a specific GPU type.
    
    Args:
        cost_e_per_gpu: Dict mapping GPU type to cost array
        gpu_type: GPU type string (e.g., "A100", "A40", "A10")
    
    Returns:
        Cost array for the specified GPU type, with fallback to DEFAULT_GPU_TYPE
    """
    if gpu_type in cost_e_per_gpu:
        return cost_e_per_gpu[gpu_type]
    
    # Fallback: try common mappings
    # A40 might use A10 profile if A40 not available
    fallback_map = {
        "A40": ["A10", "A100"],
        "A10": ["A40", "A100"],
        "A100": ["A10", "A40"],
    }
    
    for fallback in fallback_map.get(gpu_type, []):
        if fallback in cost_e_per_gpu:
            return cost_e_per_gpu[fallback]
    
    # Last resort: return first available
    if cost_e_per_gpu:
        return list(cost_e_per_gpu.values())[0]
    
    raise ValueError(f"No cost_e available for GPU type: {gpu_type}")



def schedule(pp_degree, num_mb, stage_comp_time_lst, stage_for_send_time_lst, stage_back_send_time_lst):

    ppgroup_cfg = {"num_mb": None,
                   "pp_degree": None,
                   "stage_comp_time_lst": stage_comp_time_lst,
                   "stage_for_send_time_lst": stage_for_send_time_lst,
                   "stage_back_send_time_lst": stage_back_send_time_lst
                   }

    if isinstance(num_mb, torch.Tensor):
        ppgroup_cfg["num_mb"] = int(num_mb.item())
    else:
        ppgroup_cfg["num_mb"] = num_mb
    
    if isinstance(pp_degree, torch.Tensor):
        ppgroup_cfg["pp_degree"] = int(pp_degree.item())
    else:
        ppgroup_cfg["pp_degree"] = pp_degree

    if ppgroup_cfg["pp_degree"] == 1:
        cost = num_mb * sum(stage_comp_time_lst)

    else:    
        my_pp_group = PPGroup(**ppgroup_cfg)
        
        my_pp_group.simulate_full_pipeline()
        cost = my_pp_group.get_pipe_cost()

    if not isinstance(cost, torch.Tensor):
        cost = torch.tensor(cost)

    if ppgroup_cfg["pp_degree"] == 1:
        stage_wise_cost_lst = [cost]
    else:
        stage_wise_cost_lst = my_pp_group.get_stagewise_end_time_lst()

    return cost, stage_wise_cost_lst


def exhaustive_partition(
    num_layer: int,
    cost_e_per_gpu: Dict[str, np.ndarray],
    cost_c: np.ndarray,
    pp_degree: int,
    gpu_type_lst: List[str]
):
    """
    Exhaustive search for optimal partition.
    
    Args:
        num_layer: Total number of layers/nodes
        cost_e_per_gpu: Dict mapping GPU type to execution cost array
        cost_c: Communication cost array
        pp_degree: Pipeline parallel degree
        gpu_type_lst: List of GPU types for each stage
    """
    s_time = time.time()
    P = compositions(num_layer, pp_degree)
    max_latency = np.inf
    partition = []
    stage_latency = []
    
    for p in P:
        cur_latency = get_stage_latency(list(p), cost_e_per_gpu, cost_c, gpu_type_lst)
        stage_time_lst = [stage.get_comp_time() for stage in cur_latency]
        
        if max(stage_time_lst) < max_latency:
            partition = list(p)
            stage_latency = cur_latency
            max_latency = max(stage_time_lst)

    stage_time_lst, stage_comp_time_lst, stage_comm_time_lst, stage_for_send_time_lst, stage_back_send_time_lst = _extract_stage_times(stage_latency)
    # print(f"exhaustive_partition: {time.time()-s_time:.4f} sec")
    # print(f"partition: {partition}")

    return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst
    
        
from itertools import permutations
def compositions(n, k):
    def inner(n, k):
        if k == 1:
            yield (n,)
        else:
            for i in range(1, n):
                for rest in inner(n-i, k-1):
                    yield (i,) + rest
    return list(inner(n, k))

def dynamic_programming(
    total_layers: int,
    cost_e_per_gpu: Dict[str, np.ndarray],
    cost_c: np.ndarray,
    pp: int,
    num_mb: int,
    gpu_type_list: List[str],
    verbose: bool = False
):
    """
    Dynamic programming model partitioning method (from AMP).

    Args:
        total_layers: Total number of layers/nodes
        cost_e_per_gpu: Dict mapping GPU type to execution cost array
        cost_c: Communication cost array
        pp: Pipeline parallel degree
        num_mb: Number of micro-batches
        gpu_type_list: List of GPU types for each stage
        verbose: Enable verbose logging
    """
    time_dp_s = time.time()

    if pp == 1:
        cost_e = _get_cost_e_for_gpu(cost_e_per_gpu, gpu_type_list[0])
        S = [total_layers]
        stage_latency = get_stage_latency(S, cost_e_per_gpu, cost_c, gpu_type_list)
        stage_time_lst, stage_comp_time_lst, stage_comm_time_lst, stage_for_send_time_lst, stage_back_send_time_lst = _extract_stage_times(stage_latency)

        return S, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst

    # Build possible cost values from all GPU types
    possible = [0]
    for gpu_type in cost_e_per_gpu.keys():
        cost_e_arr = cost_e_per_gpu[gpu_type]
        for i in range(1, total_layers+1):
            ptr = 0
            while ptr + i <= total_layers:
                possible.append(sum(cost_e_arr[ptr:ptr+i]))
                ptr += 1

    # Build cost_e list for each stage based on GPU type
    cost_e_list = [_get_cost_e_for_gpu(cost_e_per_gpu, gpu_type_list[i]) for i in range(pp)]

    possible = sorted(list(set(possible)))
    trace = []
    for i in range(total_layers):
        outer = []
        for j in range(pp):
            inner = []
            for m in range(len(possible)):
                inner.append(([],np.infty))
            outer.append(inner)
        trace.append(outer)

    for i in range(total_layers):
        for j in range(pp):
            for m in range(len(possible)):
                if i+1 <= j: # invalid
                    pass
                else:
                    if j == 0: # base case: 0 cut
                        comp_t = sum(cost_e_list[j][:i+1])
                        comp_t = max(comp_t, possible[m])
                        stage_time = comp_t * (num_mb-1)
                        trace[i][j][m] = ([i+1], stage_time)
                        # if verbose:
                        #     print(f"trace[{i}][{j}][{m}] : {trace[i][j][m]}")
                    else:
                        cost_best = np.infty
                        S_best = []
                        for cut in range(j-1, i):
                            comp_t = sum(cost_e_list[j][cut+1:i+1])
                            S, past_stage_time = trace[cut][j-1][possible.index(max(comp_t, possible[m]))]
                            cur_stage_time = comp_t
                            cur_stage_time += cost_c[cut][j-1]
                            if j != pp-1:
                                cur_stage_time += cost_c[cut][j]
                            cur_stage_time = cur_stage_time * (num_mb-1)
                            stage_time = max(cur_stage_time, past_stage_time)
                            if stage_time < cost_best:
                                cost_best = stage_time
                                S_ = copy.deepcopy(S)
                                S_.append(i-cut)
                                S_best = S_
                        trace[i][j][m] = (S_best, cost_best)
                        # if verbose:
                        #     print(f"trace[{i}][{j}][{m}] : {trace[i][j][m]}")

    time_dp_used = time.time() - time_dp_s

    # add each stage cost at the end
    S, cost = trace[total_layers-1][pp-1][0]
    # if verbose:
    #     print(f"trace: {trace[total_layers-1][pp-1][0]}")
    #     print(f"dynamic programming used {round(time_dp_used,2)} seconds with {total_layers} layers and {pp} stages.")
    #     print(f"S: {S}, cost: {cost}")
    
    stage_latency = get_stage_latency(S, cost_e_per_gpu, cost_c, gpu_type_list)
    stage_time_lst, stage_comp_time_lst, stage_comm_time_lst, stage_for_send_time_lst, stage_back_send_time_lst = _extract_stage_times(stage_latency)

    return S, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst


def dynamic_programming2(
    total_layers: int,
    cost_e_per_gpu: Dict[str, np.ndarray],
    cost_c: np.ndarray,
    pp: int,
    num_mb: int,
    gpu_type_list: List[str],
    verbose: bool = False
):
    """
    Dynamic programming model partitioning method (variant 2).

    Args:
        total_layers: Total number of layers/nodes
        cost_e_per_gpu: Dict mapping GPU type to execution cost array
        cost_c: Communication cost array
        pp: Pipeline parallel degree
        num_mb: Number of micro-batches
        gpu_type_list: List of GPU types for each stage
        verbose: Enable verbose logging
    """
    time_dp_s = time.time()

    if pp == 1:
        cost_e = _get_cost_e_for_gpu(cost_e_per_gpu, gpu_type_list[0])
        partition = [total_layers]
        stage_latency = get_stage_latency(partition, cost_e_per_gpu, cost_c, gpu_type_list)
        stage_time_lst, stage_comp_time_lst, stage_comm_time_lst, stage_for_send_time_lst, stage_back_send_time_lst = _extract_stage_times(stage_latency)

        return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst

    # Build cost_e list for each stage
    cost_e_list = [_get_cost_e_for_gpu(cost_e_per_gpu, gpu_type_list[i]) for i in range(pp)]

    DP = [[([], np.inf) for _ in range(pp)] for _ in range(total_layers)]
    for i in range(total_layers):
        DP[i][0] = ([i+1], sum(cost_e_list[0][:i+1])*(num_mb-1))
    
    if num_mb == 1:
        for j in range(1, pp):
            for i in range(j, total_layers):
                for cut in range(j-1,i):
                    comp_t = sum(cost_e_list[j][cut+1:i+1])
                    partition, past_stage_time = DP[cut][j-1]
                    cur_stage_time = comp_t
                    cur_stage_time += cost_c[cut][j-1]
                    if j != pp-1:
                        cur_stage_time += cost_c[cut][j]
                    stage_time = past_stage_time + cur_stage_time
                    if stage_time < DP[i][j][1]:
                        DP[i][j] = (partition+[i-cut], stage_time)
    else:
        for j in range(1, pp):
            for i in range(j, total_layers):
                for cut in range(j-1, i):
                    comp_t = sum(cost_e_list[j][cut+1:i+1])
                    partition, past_stage_time = DP[cut][j-1]
                    cur_stage_time = comp_t
                    cur_stage_time += cost_c[cut][j-1]
                    if j != pp-1:
                        cur_stage_time += cost_c[cut][j]
                    cur_stage_time = cur_stage_time * (num_mb-1)
                    stage_time = max(cur_stage_time, past_stage_time)
                    if stage_time < DP[i][j][1]:
                        DP[i][j] = (partition+[i-cut], stage_time)
                            
    time_dp_used = time.time() - time_dp_s

    partition, cost = DP[total_layers-1][pp-1]

    # if verbose:
    #     print(f"dynamic_programming2 used {round(time_dp_used,2)} seconds with {total_layers} nodes and {pp} stages.")
    #     print(f"partition: {partition}, cost: {cost}")

    stage_latency = get_stage_latency(partition, cost_e_per_gpu, cost_c, gpu_type_list)
    stage_time_lst, stage_comp_time_lst, stage_comm_time_lst, stage_for_send_time_lst, stage_back_send_time_lst = _extract_stage_times(stage_latency)

    return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst


def _get_comm_cost(cost_c: np.ndarray, cut_pos: int, stage_idx: int) -> float:
    """Safe lookup of communication cost at (cut_pos, stage_idx)."""
    r = min(cut_pos, cost_c.shape[0] - 1) if cost_c.size else 0
    c = min(stage_idx, cost_c.shape[1] - 1) if cost_c.size else 0
    return float(cost_c[r, c]) if cost_c.size else 0.0


def _create_mip_solver(pywraplp):
    """
    Create a MIP-capable solver using the best available backend.

    Tries in order: SCIP (best open-source MIP), HiGHS (fast), CBC (fallback), GLPK.
    Returns (solver, solver_name). Raises RuntimeError if none available.
    """
    # Prefer SCIP (strong for MIP with continuous vars), then HiGHS, CBC, GLPK
    for solver_id, name in [
        ('SCIP', 'SCIP'),
        ('HIGHS_MIP', 'HiGHS'),
        ('HIGHS', 'HiGHS'),
        ('CBC', 'CBC'),
        ('GLPK_MIP', 'GLPK'),
    ]:
        solver = pywraplp.Solver.CreateSolver(solver_id)
        if solver is not None:
            return solver, name
    raise RuntimeError(
        "ILP requires a MIP solver (SCIP, HiGHS, CBC, or GLPK). "
        "Install ortools: pip install ortools"
    )


# Scale factor for CP-SAT integer formulation (costs -> integers).
# Kept so scaled values fit in 32-bit for CP-SAT (max_comp ~ 1e3–1e5 ms).
_ILP_CP_SAT_SCALE = int(1e6)


def _solve_ilp_cpsat(
    num_nodes: int,
    prefix: List[List[float]],
    cost_c: np.ndarray,
    pp_degree: int,
    num_mb_val: int,
    M_float: float,
    verbose: bool,
) -> Optional[List[int]]:
    """
    Solve the partition ILP using CP-SAT (integer-only, often faster than SCIP/CBC).

    All costs are scaled to integers. Returns partition list if solved, else None.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(f"[ILP-CPSAT] {msg}", flush=True)

    try:
        from ortools.sat.python import cp_model
    except ImportError:
        _log("ortools.sat not available, skipping CP-SAT")
        return None

    _log(f"num_nodes={num_nodes}, pp_degree={pp_degree}, num_mb={num_mb_val}, building model...")
    t_build = time.time()

    scale = _ILP_CP_SAT_SCALE
    M_int = min(int(M_float * scale) + 1, 2**31 - 1)  # CP-SAT friendly upper bound

    model = cp_model.CpModel()

    # Cut position binary variables
    cut_vars = {}
    cut_ints = []
    for j in range(pp_degree - 1):
        lb, ub = j + 1, num_nodes - (pp_degree - 1 - j)
        for k in range(lb, ub + 1):
            cut_vars[(j, k)] = model.new_bool_var(f"y_{j}_{k}")
        cut_int = model.new_int_var(lb, ub, f"cut_{j}")
        model.add(cut_int == sum(k * cut_vars[(j, k)] for k in range(lb, ub + 1)))
        cut_ints.append(cut_int)
        model.add(sum(cut_vars[(j, k)] for k in range(lb, ub + 1)) == 1)

    for j in range(len(cut_ints) - 1):
        model.add(cut_ints[j] + 1 <= cut_ints[j + 1])

    # Stage costs as integer variables (scaled)
    stage_costs = [
        model.new_int_var(0, M_int, f"stage_cost_{j}") for j in range(pp_degree)
    ]

    # Stage 0
    lb0, ub0 = 1, num_nodes - (pp_degree - 1)
    comm0 = _get_comm_cost(cost_c, 0, 0)
    for k in range(lb0, ub0 + 1):
        val = int(round((prefix[0][k] + comm0) * scale))
        val = max(0, min(val, M_int))
        y = cut_vars[(0, k)]
        model.add(stage_costs[0] <= val + M_int * (1 - y))
        model.add(stage_costs[0] >= val - M_int * (1 - y))

    # Middle stages
    for j in range(1, pp_degree - 1):
        lb_prev = j
        ub_prev = num_nodes - (pp_degree - j)
        lb_curr = j + 1
        ub_curr = num_nodes - (pp_degree - 1 - j)
        for k_start in range(lb_prev, ub_prev + 1):
            for k_end in range(max(k_start + 1, lb_curr), ub_curr + 1):
                comp_val = prefix[j][k_end] - prefix[j][k_start]
                comm_val = _get_comm_cost(cost_c, k_start, j - 1)
                val = int(round((comp_val + comm_val) * scale))
                val = max(0, min(val, M_int))
                both = model.new_bool_var(f"both_{j}_{k_start}_{k_end}")
                model.add(both <= cut_vars[(j - 1, k_start)])
                model.add(both <= cut_vars[(j, k_end)])
                model.add(
                    both >= cut_vars[(j - 1, k_start)] + cut_vars[(j, k_end)] - 1
                )
                model.add(stage_costs[j] <= val + M_int * (1 - both))
                model.add(stage_costs[j] >= val - M_int * (1 - both))

    # Last stage
    lb_last = pp_degree - 1
    ub_last = num_nodes - 1
    for k in range(lb_last, ub_last + 1):
        val = int(
            round(
                (prefix[-1][num_nodes] - prefix[-1][k] + _get_comm_cost(cost_c, k, pp_degree - 2))
                * scale
            )
        )
        val = max(0, min(val, M_int))
        y = cut_vars[(pp_degree - 2, k)]
        model.add(stage_costs[-1] <= val + M_int * (1 - y))
        model.add(stage_costs[-1] >= val - M_int * (1 - y))

    max_stage_cost = model.new_int_var(0, M_int, "max_stage_cost")
    for j in range(pp_degree):
        model.add(max_stage_cost >= stage_costs[j])

    model.minimize(
        (num_mb_val - 1) * max_stage_cost + sum(stage_costs)
    )

    build_sec = time.time() - t_build
    _log(f"model built in {build_sec:.2f}s, starting solver...")

    solver = cp_model.CpSolver()
    if not verbose:
        solver.parameters.log_search_progress = False
    t_solve = time.time()
    status = solver.solve(model)
    solve_sec = time.time() - t_solve

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, f"status={status}")
    _log(f"solver finished: {status_name}, wall_time={solve_sec:.2f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        _log("no feasible solution, returning None")
        return None

    cut_values = [0] + [int(solver.value(cut_ints[j])) for j in range(len(cut_ints))] + [num_nodes]
    partition = [cut_values[j + 1] - cut_values[j] for j in range(pp_degree)]

    _log(f"solved in {solve_sec:.2f}s: partition={partition}, cuts={cut_values[1:-1]}")
    if verbose:
        sc = [solver.value(stage_costs[j]) / scale for j in range(pp_degree)]
        print(f"ILP (CP-SAT): partition={partition}, cuts={cut_values[1:-1]}", flush=True)
        print(f"ILP stage_costs (scaled back): {[round(x, 6) for x in sc]}", flush=True)

    return partition


def ILP(
    num_nodes: int,
    cost_e_per_gpu: Dict[str, np.ndarray],
    cost_c: np.ndarray,
    pp_degree: int,
    gpu_type_lst: List[str],
    num_mb: int,
    verbose: bool = False
):
    """
    Integer Linear Programming for optimal partition using OR-Tools.

    Stage time = computation + communication
    Uses binary cut-position variables and big-M for indicator constraints.

    Args:
        num_nodes: Total number of nodes to partition
        cost_e_per_gpu: Dict mapping GPU type to execution cost array
        cost_c: Communication cost array (shape: num_nodes x pp_degree)
        pp_degree: Pipeline parallel degree
        gpu_type_lst: List of GPU types for each stage
        num_mb: Number of micro-batches
        verbose: Enable verbose logging

    Returns:
        Tuple of (partition, stage_comp_time_lst, stage_comm_time_lst,
                  stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst)
    """
    s_time = time.time()

    try:
        from ortools.linear_solver import pywraplp
    except ImportError:
        raise ImportError("ILP requires ortools. Install with: pip install ortools")

    # Build cost_e list for each stage based on GPU type
    cost_e_list = [_get_cost_e_for_gpu(cost_e_per_gpu, gpu_type_lst[j]) for j in range(pp_degree)]

    # Precompute prefix sums: prefix[j][i] = sum of cost_e_list[j][0:i]
    prefix = []
    for j in range(pp_degree):
        psum = [0.0]
        for i in range(num_nodes):
            psum.append(psum[-1] + float(cost_e_list[j][i]))
        prefix.append(psum)

    # Big-M upper bound for stage cost (comp + comm)
    max_comp = max(prefix[j][num_nodes] for j in range(pp_degree))
    max_comm = float(np.max(cost_c)) if cost_c.size else 0.0
    M = (max_comp + max_comm) * 2 + 1e6

    num_mb_val = int(num_mb.item()) if hasattr(num_mb, "item") else int(num_mb)

    # Try CP-SAT first (integer formulation, often much faster than SCIP/CBC)
    if verbose:
        print(f"[ILP] trying CP-SAT first (num_nodes={num_nodes}, pp_degree={pp_degree})", flush=True)
    try:
        partition = _solve_ilp_cpsat(
            num_nodes, prefix, cost_c, pp_degree, num_mb_val, M, verbose
        )
    except Exception as e:
        print(f"[ILP] CP-SAT failed (exception): {e}", flush=True)
        import traceback
        traceback.print_exc()
        partition = None

    if partition is None:
        if verbose:
            print("[ILP] CP-SAT returned no solution, falling back to MIP", flush=True)
        # Fallback: MIP with linear_solver (SCIP/CBC/etc.)
        solver, solver_name = _create_mip_solver(pywraplp)
        infinity = solver.infinity()
        cut_vars = {}
        cut_ints = []

        for j in range(pp_degree - 1):
            lb, ub = j + 1, num_nodes - (pp_degree - 1 - j)
            for k in range(lb, ub + 1):
                cut_vars[(j, k)] = solver.BoolVar(f'y_{j}_{k}')
            cut_int = solver.IntVar(lb, ub, f'cut_{j}')
            solver.Add(cut_int == sum(k * cut_vars[(j, k)] for k in range(lb, ub + 1)))
            cut_ints.append(cut_int)
            solver.Add(sum(cut_vars[(j, k)] for k in range(lb, ub + 1)) == 1)

        for j in range(len(cut_ints) - 1):
            solver.Add(cut_ints[j] + 1 <= cut_ints[j + 1])

        stage_costs = [solver.NumVar(0.0, infinity, f'stage_cost_{j}') for j in range(pp_degree)]

        lb0, ub0 = 1, num_nodes - (pp_degree - 1)
        comm0 = _get_comm_cost(cost_c, 0, 0)
        for k in range(lb0, ub0 + 1):
            val = prefix[0][k] + comm0
            y = cut_vars[(0, k)]
            solver.Add(stage_costs[0] <= val + M * (1 - y))
            solver.Add(stage_costs[0] >= val - M * (1 - y))

        for j in range(1, pp_degree - 1):
            lb_prev = j
            ub_prev = num_nodes - (pp_degree - j)
            lb_curr = j + 1
            ub_curr = num_nodes - (pp_degree - 1 - j)
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

        lb_last = pp_degree - 1
        ub_last = num_nodes - 1
        for k in range(lb_last, ub_last + 1):
            val = prefix[-1][num_nodes] - prefix[-1][k] + _get_comm_cost(cost_c, k, pp_degree - 2)
            y = cut_vars[(pp_degree - 2, k)]
            solver.Add(stage_costs[-1] <= val + M * (1 - y))
            solver.Add(stage_costs[-1] >= val - M * (1 - y))

        max_stage_cost = solver.NumVar(0.0, infinity, 'max_stage_cost')
        for j in range(pp_degree):
            solver.Add(max_stage_cost >= stage_costs[j])

        solver.Minimize((num_mb_val - 1) * max_stage_cost + sum(stage_costs))

        if verbose:
            print(f"ILP (ortools, solver={solver_name}): variables={solver.NumVariables()}, constraints={solver.NumConstraints()}")

        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
            print("ILP: No solution found, falling back to even partition")
            base = num_nodes // pp_degree
            remainder = num_nodes % pp_degree
            partition = [base + (1 if i < remainder else 0) for i in range(pp_degree)]
        else:
            cut_values = [0] + [int(cut_ints[j].solution_value()) for j in range(len(cut_ints))] + [num_nodes]
            partition = [cut_values[j + 1] - cut_values[j] for j in range(pp_degree)]
            if verbose:
                sc = [stage_costs[j].solution_value() for j in range(pp_degree)]
                print(f"ILP solution: partition={partition}, cuts={cut_values[1:-1]}")
                print(f"ILP stage_costs: {sc}")

    stage_latency = get_stage_latency(partition, cost_e_per_gpu, cost_c, gpu_type_lst)
    stage_time_lst, stage_comp_time_lst, stage_comm_time_lst, stage_for_send_time_lst, stage_back_send_time_lst = _extract_stage_times(stage_latency)

    return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst


# =============================================================================
# Partition Method Wrappers
# =============================================================================
# Unified interface for --pp-partition-method argument
# Each function returns: (partition, stage_comp_time_lst, stage_comm_time_lst,
#                         stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst)

def partition_even(
    num_layer: int,
    cost_e_per_gpu: Dict[str, np.ndarray],
    cost_c: np.ndarray,
    pp_degree: int,
    gpu_type_lst: List[str],
    num_mb: int = 1,
    verbose: bool = False
):
    """
    Even partition method (Optimus Prime style).

    Distributes layers evenly across stages with minimal rebalancing.
    Uses minmax() with even_split=True.
    """
    return minmax(
        num_layer=num_layer,
        cost_e_per_gpu=cost_e_per_gpu,
        cost_c=cost_c,
        pp_degree=pp_degree,
        gpu_type_lst=gpu_type_lst,
        even_split=True,
        verbose=verbose
    )


def partition_minmax(
    num_layer: int,
    cost_e_per_gpu: Dict[str, np.ndarray],
    cost_c: np.ndarray,
    pp_degree: int,
    gpu_type_lst: List[str],
    num_mb: int = 1,
    verbose: bool = False
):
    """
    Min-max partition method.

    Iteratively rebalances layers to minimize the maximum stage latency.
    Uses minmax() with even_split=False.
    """
    return minmax(
        num_layer=num_layer,
        cost_e_per_gpu=cost_e_per_gpu,
        cost_c=cost_c,
        pp_degree=pp_degree,
        gpu_type_lst=gpu_type_lst,
        even_split=False,
        verbose=verbose
    )


def partition_dp(
    num_layer: int,
    cost_e_per_gpu: Dict[str, np.ndarray],
    cost_c: np.ndarray,
    pp_degree: int,
    gpu_type_lst: List[str],
    num_mb: int = 1,
    verbose: bool = False
):
    """
    Dynamic programming partition method.

    Uses dynamic programming to find optimal partition.
    Internally calls dynamic_programming2().
    """
    return dynamic_programming2(
        total_layers=num_layer,
        cost_e_per_gpu=cost_e_per_gpu,
        cost_c=cost_c,
        pp=pp_degree,
        num_mb=num_mb,
        gpu_type_list=gpu_type_lst,
        verbose=verbose
    )


def partition_ilp(
    num_layer: int,
    cost_e_per_gpu: Dict[str, np.ndarray],
    cost_c: np.ndarray,
    pp_degree: int,
    gpu_type_lst: List[str],
    num_mb: int = 1,
    verbose: bool = False
):
    """
    ILP (Integer Linear Programming) partition method.

    Uses OR-Tools (SCIP/CBC) to solve optimal partition with comp+comm stage costs.
    Supports both FX node-based and legacy layer-based profiling.
    Internally calls ILP().
    """
    return ILP(
        num_nodes=num_layer,
        cost_e_per_gpu=cost_e_per_gpu,
        cost_c=cost_c,
        pp_degree=pp_degree,
        gpu_type_lst=gpu_type_lst,
        num_mb=num_mb,
        verbose=verbose
    )


def partition_bruteforce(
    num_layer: int,
    cost_e_per_gpu: Dict[str, np.ndarray],
    cost_c: np.ndarray,
    pp_degree: int,
    gpu_type_lst: List[str],
    num_mb: int = 1,
    verbose: bool = False
):
    """
    Brute-force exhaustive search partition method.

    Tries all possible partition combinations and selects the best one.
    WARNING: Exponential complexity - only use for small num_layer and pp_degree.
    Internally calls exhaustive_partition().
    """
    return exhaustive_partition(
        num_layer=num_layer,
        cost_e_per_gpu=cost_e_per_gpu,
        cost_c=cost_c,
        pp_degree=pp_degree,
        gpu_type_lst=gpu_type_lst
    )


# Mapping from method name to function
PARTITION_METHODS = {
    "even": partition_even,
    "minmax": partition_minmax,
    "dp": partition_dp,
    "ilp": partition_ilp,
    "bruteforce": partition_bruteforce,
}


def get_partition_function(method: str):
    """
    Get partition function by method name.

    Args:
        method: Partition method name ("even", "minmax", "dp", "ilp", "bruteforce")

    Returns:
        Partition function

    Raises:
        ValueError: If method is unknown
    """
    if method not in PARTITION_METHODS:
        raise ValueError(f"Unknown partition method: {method}. Available: {list(PARTITION_METHODS.keys())}")
    return PARTITION_METHODS[method]