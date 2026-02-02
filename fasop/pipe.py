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
    Integer Linear Programming for optimal partition using CPLEX.

    Supports both FX node-based and legacy layer-based profiling.
    Uses binary assignment variables for each possible (node, stage) pair.

    Args:
        num_nodes: Total number of nodes/layers to partition
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
        from docplex.mp.model import Model
    except ImportError:
        raise ImportError("ILP requires docplex. Install with: pip install docplex")

    # Build cost_e list for each stage based on GPU type
    cost_e_list = [_get_cost_e_for_gpu(cost_e_per_gpu, gpu_type_lst[j]) for j in range(pp_degree)]

    # Precompute prefix sums for each GPU type's cost_e
    # prefix[j][i] = sum of cost_e_list[j][0:i]
    prefix = []
    for j in range(pp_degree):
        psum = [0.0]
        for i in range(num_nodes):
            psum.append(psum[-1] + cost_e_list[j][i])
        prefix.append(psum)

    # Create ILP model
    m = Model(name='fx_node_partitioning')

    # Binary assignment variables: x[i][j] = 1 if node i is the start of stage j+1
    # (i.e., node i is the first node after cut point j)
    # For pp_degree stages, we need pp_degree-1 cut points
    # cut[j] indicates where stage j+1 starts (1-indexed cut points)

    # Create binary variables for cut positions
    # y[j][k] = 1 if cut point j is at position k
    # j ranges from 0 to pp_degree-2 (pp_degree-1 cut points)
    # k ranges based on valid positions for each cut

    cut_vars = {}  # (j, k) -> binary variable
    cuts = []  # Integer variables representing actual cut positions

    for j in range(pp_degree - 1):
        # Cut j can be at positions from j+1 to num_nodes-(pp_degree-1-j)
        lb = j + 1
        ub = num_nodes - (pp_degree - 1 - j)

        # Create binary variables for each possible position
        for k in range(lb, ub + 1):
            cut_vars[(j, k)] = m.binary_var(name=f'y_{j}_{k}')

        # Exactly one position is selected for this cut
        m.add_constraint(
            m.sum(cut_vars[(j, k)] for k in range(lb, ub + 1)) == 1,
            f'one_cut_{j}'
        )

        # Create integer variable and link to binary
        cut_int = m.integer_var(lb=lb, ub=ub, name=f'cut_{j}')
        m.add_constraint(
            cut_int == m.sum(k * cut_vars[(j, k)] for k in range(lb, ub + 1)),
            f'cut_link_{j}'
        )
        cuts.append(cut_int)

    # Ordering constraints: cut[j-1] < cut[j]
    for j in range(len(cuts) - 1):
        m.add_constraint(cuts[j] + 1 <= cuts[j + 1], f'order_{j}')

    # Stage computation cost variables
    stage_costs = [m.continuous_var(name=f'stage_cost_{j}') for j in range(pp_degree)]

    # Stage 0: nodes [0, cuts[0])
    # Use indicator constraints with binary variables
    lb0, ub0 = 1, num_nodes - (pp_degree - 1)
    for k in range(lb0, ub0 + 1):
        m.add_indicator(
            cut_vars[(0, k)],
            stage_costs[0] == prefix[0][k],
            name=f'ind_0_{k}'
        )

    # Middle stages: stage j covers [cuts[j-1], cuts[j]) for j = 1 to pp_degree-2
    for j in range(1, pp_degree - 1):
        lb_prev = j
        ub_prev = num_nodes - (pp_degree - j)
        lb_curr = j + 1
        ub_curr = num_nodes - (pp_degree - 1 - j)

        for k_start in range(lb_prev, ub_prev + 1):
            for k_end in range(max(k_start + 1, lb_curr), ub_curr + 1):
                # Create auxiliary binary for conjunction
                both = m.binary_var(name=f'both_{j}_{k_start}_{k_end}')
                m.add_constraint(both <= cut_vars[(j - 1, k_start)], f'both_a_{j}_{k_start}_{k_end}')
                m.add_constraint(both <= cut_vars[(j, k_end)], f'both_b_{j}_{k_start}_{k_end}')
                m.add_constraint(both >= cut_vars[(j - 1, k_start)] + cut_vars[(j, k_end)] - 1, f'both_c_{j}_{k_start}_{k_end}')

                m.add_indicator(
                    both,
                    stage_costs[j] == prefix[j][k_end] - prefix[j][k_start],
                    name=f'ind_{j}_{k_start}_{k_end}'
                )

    # Last stage: nodes [cuts[-1], num_nodes)
    lb_last = pp_degree - 1
    ub_last = num_nodes - 1
    for k in range(lb_last, ub_last + 1):
        m.add_indicator(
            cut_vars[(pp_degree - 2, k)],
            stage_costs[-1] == prefix[-1][num_nodes] - prefix[-1][k],
            name=f'ind_last_{k}'
        )

    # Max stage cost variable
    max_stage_cost = m.continuous_var(name='max_stage_cost')
    for j in range(pp_degree):
        m.add_constraint(max_stage_cost >= stage_costs[j], f'max_stage_{j}')

    # Objective: minimize pipeline latency
    # For 1F1B: (num_mb + pp - 1) * max_stage
    # Simplified: (num_mb - 1) * max_stage + sum(stage_costs)
    total_stage_cost = m.sum(stage_costs)
    m.minimize((num_mb - 1) * max_stage_cost + total_stage_cost)

    # if verbose:
    #     m.print_information()

    # Solve
    solution = m.solve()

    if solution is None:
        print("ILP: No solution found, falling back to even partition")
        # Fallback to even partition
        base = num_nodes // pp_degree
        remainder = num_nodes % pp_degree
        partition = [base + (1 if i < remainder else 0) for i in range(pp_degree)]
    else:
        # Extract partition from cut points
        cut_values = [0] + [int(solution.get_value(cuts[j])) for j in range(len(cuts))] + [num_nodes]
        partition = [cut_values[j + 1] - cut_values[j] for j in range(pp_degree)]

        # if verbose:
        #     print(f"ILP solution: partition={partition}, cuts={cut_values[1:-1]}")
        #     print(f"ILP stage_costs: {[solution.get_value(stage_costs[j]) for j in range(pp_degree)]}")

    elapsed = time.time() - s_time
    # if verbose:
    #     print(f"ILP: {elapsed:.4f} sec")

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

    Uses CPLEX to solve optimal partition as ILP problem.
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