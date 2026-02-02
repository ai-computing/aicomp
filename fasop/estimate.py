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

import os

import torch
import torch.nn as nn
import numpy as np

from utils import axis2rank
from pipe import (
    minmax, schedule, get_stage_latency, explain_minmax,
    dynamic_programming, dynamic_programming2, exhaustive_partition, ILP,
    # Partition method wrappers
    partition_even, partition_minmax, partition_dp, partition_ilp, partition_bruteforce,
    get_partition_function, PARTITION_METHODS
)
from device_placement import get_gpu_for_stage
from config import (
    TRAINING_CONFIG, 
    get_gpu_memory_gb, 
    get_memory_max_from_cluster_info,
    FX_NODE_TYPES,
    FX_NODE_INDEX,
    get_mbs_index,
    get_node_param_count,
    get_node_activation_memory,
    A40_INTRA_BW_MARKER,
    TRANSFORMER_LAYER_PARAM_MULTIPLIER,
    TRANSFORMER_LAYER_EXTRA_PARAMS,
    OPTIMIZER_STATE_MULTIPLIER,
    ZERO_BASE_MEMORY_MULTIPLIER,
    ZERO_OPTIM_STATE_MEMORY,
    GPUConfig,
    ProfileDB,
)
from partition import (
    SplitMethod,
    PartitionAlgorithm,
    StageConfig,
    PartitionResult,
    get_layers_per_stage,
    get_nodes_per_stage,
    build_stage_configs,
    build_partition_result,
    expand_node_costs,
    get_node_type_sequence,
    get_total_nodes,
    compute_stage_latencies,
    compute_max_stage_latency,
    NODES_PER_LAYER,
    LAYER_NODE_TYPES,
)

import copy


class FASOP(nn.Module):
    def __init__(self, model_config, exp_name, gpu_config: GPUConfig, num_node, partition_method: str = "minmax"):
        """
        Initialize FASOP model.

        Args:
            model_config: Model configuration dict
            exp_name: Experiment name
            gpu_config: GPUConfig object containing GPU bandwidth settings
            num_node: Number of nodes
            partition_method: PP partition method - "even", "minmax", "dp", "ilp", "exhaustive"
                - even: Optimus Prime style (simple_split if tp=1, llama_tp_split if tp>1)
                - minmax: Min-max load balancing
                - dp: Dynamic programming
                - ilp: Integer Linear Programming (CPLEX)
                - exhaustive: Exhaustive search
        """
        super().__init__()
        self.model_config = model_config
        self.exp_name = "init_" + exp_name
        self.model_type = model_config["type"]
        self.gpu_config = gpu_config
        self.num_node = num_node
        self.partition_method = partition_method
        
        # Initialize ProfileDB with dynamic GPU type loading
        self._init_profile_db()
        
    def _init_profile_db(self):
        """
        Initialize ProfileDB for dynamic profile loading.
        
        Uses ProfileDB to automatically discover and load profile costs
        for all GPU types in the cluster. Supports both NPZ (FX node-level)
        and NPY (layer-level) formats.
        """
        # Get GPU types from the gpu_config
        gpu_types = self.gpu_config.gpu_types
        
        # Create ProfileDB instance
        self.profile_db = ProfileDB(
            gpu_types=gpu_types,
            model_type=self.model_type,
            profile_dir="known_cost",
            forward_backward_multiplier=3.0,
        )
        
        # Log loaded profiles
        print(f"[FASOP] {self.profile_db.summary()}")
        
        # For backward compatibility, also create legacy format dict
        # This maps to the old "profile_cost_a100" / "profile_cost_a10" structure
        legacy_dict = self.profile_db.get_legacy_format_dict()
        self.profile_cost_A100 = legacy_dict["profile_cost_a100"]
        self.profile_cost_A10 = legacy_dict["profile_cost_a10"]
        
    def get_profile_for_gpu(self, gpu_type: str, tp: int):
        """
        Get profile data for a specific GPU type and TP value.
        
        Args:
            gpu_type: GPU type string (e.g., "A40", "A100")
            tp: Tensor parallel degree
            
        Returns:
            Profile data array, or None if not available
        """
        return self.profile_db.get_profile(gpu_type, tp)
    
    def get_available_tp_for_gpu(self, gpu_type: str):
        """
        Get available TP values for a GPU type.
        
        Args:
            gpu_type: GPU type string
            
        Returns:
            List of available TP values
        """
        return self.profile_db.get_available_tp(gpu_type)
        
    def forward(self, args, node_type, exhaustive):
        config, bs, micro_bs, cluster_info, model_config, parallel_dims = args
        
        # Build gpu_profiledb from ProfileDB
        # For now, use legacy format for backward compatibility with existing code
        gpu_profiledb = {
            "profile_cost_a100": self.profile_cost_A100,
            "profile_cost_a10": self.profile_cost_A10,
            # Also include full ProfileDB for future use
            "_profile_db": self.profile_db,
        }
        
        rank_map, partition, estimated_latency, pipecost, dp_side_cost, all_reduce_embedding_cost, is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem = predict(
            config, bs, micro_bs, cluster_info, model_config, gpu_profiledb, parallel_dims,
            node_type, exhaustive, self.partition_method, self.gpu_config
        )
        return rank_map, partition, estimated_latency, pipecost, dp_side_cost, all_reduce_embedding_cost, is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem


# =============================================================================
# FX Node Helper Functions
# =============================================================================

def _get_fx_node_activation_volume(node_type: str, bs, s, h, v, tp):
    """
    Get the activation output volume for an FX node type.
    
    This is used for calculating pipeline communication cost at stage boundaries.
    
    Args:
        node_type: FX node type string
        bs: micro-batch size
        s: sequence length
        h: hidden size
        v: vocab size
        tp: tensor parallel degree
        
    Returns:
        Activation volume (number of elements)
    """
    # Convert tensors to values if needed
    bs_val = bs.item() if hasattr(bs, 'item') else bs
    s_val = s.item() if hasattr(s, 'item') else s
    h_val = h.item() if hasattr(h, 'item') else h
    v_val = v.item() if hasattr(v, 'item') else v
    tp_val = tp.item() if hasattr(tp, 'item') else tp
    
    # Nodes that output full hidden dimension (after all-gather/all-reduce)
    if node_type in ["embed", "attn_o", "mlp_down"]:
        return bs_val * s_val * h_val
    
    # Attention Q/K/V projections (TP-sharded)
    elif node_type in ["attn_q", "attn_k", "attn_v"]:
        return bs_val * s_val * h_val / tp_val
    
    # MLP intermediate activations (TP-sharded)
    elif node_type in ["mlp_gate", "mlp_act_fn", "mlp_up"]:
        return bs_val * s_val * 4 * h_val / tp_val
    
    # LM head output (logits over vocabulary)
    elif node_type == "lm_head":
        return bs_val * s_val * v_val
    
    # Default fallback
    else:
        return bs_val * s_val * h_val


def _get_fx_node_param_count(node_type: str, h, v, num_heads, num_kv_heads, tp):
    """
    Get the parameter count for an FX node type.
    
    This is used for calculating data parallel gradient synchronization cost.
    
    Args:
        node_type: FX node type string
        h: hidden size
        v: vocab size
        num_heads: number of attention heads
        num_kv_heads: number of key-value heads (for GQA)
        tp: tensor parallel degree
        
    Returns:
        Parameter count (number of elements, not bytes)
    """
    # Convert tensors to values if needed
    h_val = h.item() if hasattr(h, 'item') else h
    v_val = v.item() if hasattr(v, 'item') else v
    num_heads_val = num_heads.item() if hasattr(num_heads, 'item') else num_heads
    num_kv_heads_val = num_kv_heads.item() if hasattr(num_kv_heads, 'item') else num_kv_heads
    tp_val = tp.item() if hasattr(tp, 'item') else tp
    
    head_dim = h_val // num_heads_val
    
    # Embedding and LM head: h × v / tp
    if node_type in ["embed", "lm_head"]:
        return (h_val * v_val) / tp_val
    
    # Attention Q and O projections: h × h / tp
    elif node_type in ["attn_q", "attn_o"]:
        return (h_val * h_val) / tp_val
    
    # Attention K and V projections: h × num_kv_heads × head_dim / tp (for GQA)
    elif node_type in ["attn_k", "attn_v"]:
        return (h_val * num_kv_heads_val * head_dim) / tp_val
    
    # MLP gate and up projections: h × 4h / tp
    elif node_type in ["mlp_gate", "mlp_up"]:
        return (h_val * 4 * h_val) / tp_val
    
    # MLP down projection: 4h × h / tp
    elif node_type == "mlp_down":
        return (4 * h_val * h_val) / tp_val
    
    # Activation function: no parameters
    elif node_type == "mlp_act_fn":
        return 0
    
    # Default fallback
    else:
        return 0


# =============================================================================
# Pipeline Communication Cost
# =============================================================================

# pipeline communication cost, return shape: (total_layers-1, pp-1)
def get_cost_c(cluster_info, model_config, parallel_config, gpu_profiledb, gpu_config: GPUConfig, dp_index=0, layer_types=None):
    """
    Calculate pipeline communication cost.
    
    Supports both:
        - Legacy layer types: "embedding_layer", "transformer_layer", etc.
        - FX node types: "embed", "attn_q", "attn_o", "mlp_down", "lm_head", etc.
    
    For FX nodes, activation volume varies by boundary type (output shape of the cut node):
        - embed, attn_o, mlp_down: bs * s * h (natural boundaries, full hidden dim)
        - attn_q, attn_k, attn_v: bs * s * h / tp (TP-sharded attention projections)
        - mlp_gate, mlp_act_fn, mlp_up: bs * s * 4h / tp (TP-sharded MLP intermediate)
        - lm_head: bs * s * v (logits over vocabulary)
    
    For legacy layer types, volume is always bs * s * h (layer boundaries).
    """
    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    bs = parallel_config["micro_bs"]
    rank_node_map = parallel_config["rank_node_map"]
    tp = parallel_config["tp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
    
    precision = torch.ones(1) * TRAINING_CONFIG["default_precision"]  # from config.py

    _num_layer = len(layer_types)

    if pp == 1:
        return torch.zeros(int(n.item())), layer_types
      
    # Build node/layer activation volume lookup table
    # Each entry represents the communication volume if stage boundary is AFTER that node
    layer_volume = []
    for i in range(_num_layer):
        node_or_layer_type = layer_types[i]
        
        # FX node types: volume depends on node output shape
        if node_or_layer_type in FX_NODE_TYPES:
            volume = _get_fx_node_activation_volume(node_or_layer_type, bs, s, h, v, tp)
            layer_volume.append(volume)
        # Legacy layer types: volume is always bs * s * h
        elif node_or_layer_type in ["embedding_layer", "transformer_layer", "encoder", "decoder", "post_process"]:
            layer_volume.append(bs * s * h)
        else:
            raise ValueError(f"Unrecognized layer/node type: {node_or_layer_type}")
            
    # Build communication cost between pipeline stages by looking up the cluster information
    cost_c = torch.zeros((int(dp.item()), _num_layer, int(pp.item()-1)))
    for i in range(int(dp.item())):    
        for j in range(int(pp.item()-1)):
            # get the slowest tp gpu connection
            slowest_bandwidth = np.inf
            for k in range(int(tp.item())):    
                rank_cur = axis2rank(axis=(j,i,k), mp_deg=tp, dp_deg=dp, pp_deg=pp)
                rank_peer = axis2rank(axis=(j+1,i,k), mp_deg=tp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]
                node_peer = rank_node_map[int(rank_peer.item())]
                
                if node_cur != node_peer: 
                    cur_bandwidth = min(cluster_info[node_cur][0], cluster_info[node_peer][0])
                    cur_bandwidth = cur_bandwidth / int(dp.item()) / int(tp.item()) / 1.05
                else:
                    cur_bandwidth = cluster_info[node_cur][1] / 3.5
                    # A40-specific bandwidth handling
                    if gpu_config.has_a40() and cluster_info[node_cur][1] == A40_INTRA_BW_MARKER:
                        if (rank_cur // 2) == (rank_peer // 2):
                            cur_bandwidth = gpu_config.get_a40_detail("sendrecv_nvlink")
                        elif (rank_cur // 4) == (rank_peer // 4):
                            cur_bandwidth = gpu_config.get_a40_detail("sendrecv_pcie1")
                        else:
                            cur_bandwidth = gpu_config.get_a40_detail("sendrecv_pcie2")
                # print(f"(pp-{j}, dp-{i}, tp-{k}): cur: {rank_cur.item()}, peer: {rank_peer.item()}, BW: {cur_bandwidth.item()/1e9:.2f}e9")
                if cur_bandwidth < slowest_bandwidth:
                    slowest_bandwidth = cur_bandwidth
            # print(f"SLOWEST = {slowest_bandwidth.item()}")
            # Communication volume depends on boundary node type
            for k in range(_num_layer):
                cost_c[i][k][j] = layer_volume[k] * precision / slowest_bandwidth
    cost_c = torch.max(cost_c, dim=0)
    return cost_c.values, layer_types


# =============================================================================
# FX Node-based Cost Estimation
# =============================================================================

def get_cost_e_fx(model_config, parallel_config, profile_cost, node_sequence, model_type=None):
    """
    Get execution cost for each FX node in the sequence.
    
    Args:
        model_config: Model configuration dict
        parallel_config: Parallel configuration (tp, dp, pp, micro_bs, etc.)
        profile_cost: Profile cost data from NPZ file, shape (num_mbs, num_fx_nodes)
        node_sequence: List of FX node types for the model
        model_type: Model type string
    
    Returns:
        cost_e: numpy array of shape (num_nodes,) with execution cost for each node
    """
    bs = parallel_config["micro_bs"]
    tp = parallel_config["tp"]
    dp = parallel_config["dp"]
    
    tp_key = str(int(tp.item()))
    num_nodes = len(node_sequence)
    
    # Get the profile data for the given TP degree
    if tp_key not in profile_cost:
        # Fallback to TP=1 if not available
        tp_key = "1"
    
    profile_data = profile_cost[tp_key]
    
    # Handle different profile data formats
    if profile_data.ndim == 1:
        # Old format: 1D array (layer-level costs)
        # Fall back to legacy behavior
        return _get_cost_e_legacy(model_config, parallel_config, profile_cost, node_sequence, model_type)
    
    # New NPZ format: 2D array (mbs x fx_nodes)
    # Find the row index for the given micro-batch size
    mbs_idx = get_mbs_index(int(bs))
    if mbs_idx >= profile_data.shape[0]:
        mbs_idx = profile_data.shape[0] - 1
    
    # Get costs for each FX node type
    cost_e = np.zeros(num_nodes)
    
    for i, node_type in enumerate(node_sequence):
        if node_type in FX_NODE_INDEX:
            node_idx = FX_NODE_INDEX[node_type]
            if node_idx < profile_data.shape[1]:
                cost_e[i] = profile_data[mbs_idx, node_idx]

    # Convert ms to seconds (NPZ files store latency in milliseconds)
    cost_e = cost_e / 1000.0

    return torch.from_numpy(cost_e).float()


def _get_cost_e_legacy(model_config, parallel_config, profile_cost, layer_types, model_type):
    """
    Legacy get_cost_e for backward compatibility with old NPY format.
    """
    n = model_config["num_layers"]
    bs = parallel_config["micro_bs"]
    tp = parallel_config["tp"]
    dp = parallel_config["dp"]

    _num_layer = len(layer_types)
    cost_e = np.zeros((int(dp.item()), _num_layer))

    for i in range(int(dp.item())):
        for layer_id in range(_num_layer):
            layer_type = layer_types[layer_id]
            tp_key = str(int(tp.item()))
            
            if tp_key in profile_cost and layer_id < len(profile_cost[tp_key]):
                cur_layer = bs * profile_cost[tp_key][layer_id]
            else:
                cur_layer = 0
            cost_e[i][layer_id] = cur_layer

    cost_e = torch.from_numpy(np.stack(cost_e, axis=0))
    cost_e = torch.mean(cost_e, dim=0)
    # Convert ms to seconds (NPY files store latency in milliseconds)
    cost_e = cost_e / 1000.0
    return cost_e


def _check_fx_node_format(amp_config):
    """
    Check if the profile cost data is in FX node format (2D NPZ) or legacy format (1D NPY).
    
    Returns:
        True if FX node format, False if legacy format
    """
    profile_cost = amp_config.get("profile_cost_a10", amp_config.get("profile_cost_a100", {}))
    if not profile_cost:
        return False
    
    sample_key = list(profile_cost.keys())[0]
    if sample_key in profile_cost and isinstance(profile_cost[sample_key], np.ndarray):
        return profile_cost[sample_key].ndim == 2
    return False


# Keep old function name for backward compatibility
def get_cost_e(is_a100, model_config, parallel_config, profile_cost, layer_types=None, model_type=None):
    """
    Wrapper for backward compatibility. Routes to FX-based or legacy implementation.
    """
    # Check if using new FX node format (2D array in profile_cost)
    sample_key = list(profile_cost.keys())[0] if profile_cost else "1"
    if sample_key in profile_cost and isinstance(profile_cost[sample_key], np.ndarray):
        if profile_cost[sample_key].ndim == 2:
            # New FX format
            return get_cost_e_fx(model_config, parallel_config, profile_cost, layer_types, model_type)
    
    # Legacy format - use original implementation
    return _get_cost_e_legacy(model_config, parallel_config, profile_cost, layer_types, model_type)


def cost_all_reduce_embedding(model_config, cluster_info, parallel_config, gpu_per_node, gpu_config: GPUConfig):
    tp_degree = int(parallel_config["tp"].item())
    dp_degree = int(parallel_config["dp"].item())
    pp_degree = int(parallel_config["pp"].item())
    rank_node_map = parallel_config["rank_node_map"]
    hidden_size = int(model_config["hidden_size"].item())
    vocab_size = int(model_config["vocab_size"].item())
    
    # Get precision from model_config, fallback to default if not present
    precision = int(model_config.get("precision", torch.tensor(TRAINING_CONFIG["default_precision"])).item())
    
    if pp_degree>1:
        # Get communication bandwidth between pipeline stage 0 and -1
        for i in range(dp_degree):    
            # get the slowest tp gpu connection
            slowest_bandwidth = np.inf
            for k in range(tp_degree):    
                rank_cur = axis2rank(axis=(0,i,k), mp_deg=tp_degree, dp_deg=dp_degree, pp_deg=pp_degree)
                rank_peer = axis2rank(axis=(pp_degree-1,i,k), mp_deg=tp_degree, dp_deg=dp_degree, pp_deg=pp_degree)
                node_cur = rank_node_map[rank_cur]
                node_peer = rank_node_map[rank_peer]
               
                # print(f"(pp-{pp_degree}, tp-{tp_degree}, dp-{dp_degree}), cur: {rank_cur}, peer: {rank_peer}")
                if node_cur != node_peer: # use inter-node bandwidth
                    cur_bandwidth = min(cluster_info[node_cur][0], cluster_info[node_peer][0])
                else: # use intra-node bandwidth
                    cur_bandwidth = cluster_info[node_cur][1]
                    # A40-specific bandwidth handling
                    if gpu_config.has_a40() and cluster_info[node_cur][1] == A40_INTRA_BW_MARKER:
                        if (rank_cur // 2) == (rank_peer // 2):
                            cur_bandwidth = gpu_config.get_a40_detail("allreduce_nvlink")
                        elif (rank_cur // 4) == (rank_peer // 4):
                            cur_bandwidth = gpu_config.get_a40_detail("allreduce_pcie1")
                        else:
                            cur_bandwidth = gpu_config.get_a40_detail("allreduce_pcie2")
                if cur_bandwidth < slowest_bandwidth:
                    slowest_bandwidth = cur_bandwidth
        
        # if dp_degree<gpu_per_node, we assume the bandwidth is shared by all dp_degree
        # else, we assume the bandwidth is shared by all gpu_per_node
        band_width = slowest_bandwidth/min(dp_degree, gpu_per_node) 
        embedding_syn_cost = 2*(2-1)*(hidden_size*vocab_size*precision)/(2*band_width)/tp_degree
        return embedding_syn_cost.item()
    else:
        return 0
        

def dp_cost(config, cluster_info, model_config, parallel_config, gpu_profiledb, partition, gpu_config: GPUConfig, layer_types=None, gpu_per_node=4, gpu_type_lst=None):
    """
    Calculate data parallel gradient synchronization cost.
    
    Args:
        partition: Optimus Prime partition format - list of node/layer counts per stage
                   e.g., [10, 8, 8, 6] means stage 0 has 10 nodes, stage 1 has 8, etc.
        layer_types: List of node/layer types for the entire model
    
    Supports both:
        - Legacy layer types: "embedding_layer", "transformer_layer", etc.
        - FX node types: "embed", "attn_q", "attn_o", "mlp_down", "lm_head", etc.
    
    For FX nodes, parameter count is calculated exactly per node type:
        - embed, lm_head: h * v / tp
        - attn_q, attn_o: h * h / tp
        - attn_k, attn_v: h * num_kv_heads * head_dim / tp (for GQA)
        - mlp_gate, mlp_up: h * 4h / tp
        - mlp_down: 4h * h / tp
        - mlp_act_fn: 0 (no parameters)
    
    Returns:
        Tuple of (stage_boundaries, dp_cost_list)
        - stage_boundaries: cumulative indices [0, n0, n0+n1, ...] for stage boundaries
        - dp_cost_list: list of DP gradient sync cost for each stage
    """
    h = model_config["hidden_size"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    a = model_config["num_attention_heads"]
    kv_heads = model_config.get("num_key_value_heads", a)  # For GQA
    rank_node_map = parallel_config["rank_node_map"]
    tp = parallel_config["tp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
    
    _num_layer = len(layer_types)
        
    # Convert partition to cumulative stage boundaries
    # partition: [n0, n1, n2, ...] -> stage_boundaries: [0, n0, n0+n1, n0+n1+n2, ...]
    stage_boundaries = [0]
    for i in range(len(partition)):
        stage_boundaries.append(stage_boundaries[-1] + partition[i])
    assert stage_boundaries[-1] == _num_layer, f"Partition sum {stage_boundaries[-1]} != total nodes {_num_layer}"
    assert len(stage_boundaries) == pp + 1, f"Stage boundaries {len(stage_boundaries)} != pp+1 {pp+1}"

    counted = False
    # debug
    # print(f"tp: {tp.item()}, dp: {dp.item()}, pp: {pp.item()}")
    
    # Calculate parameter count for stage 0 (used for DP cost estimation)
    param_count = 0    
    for layer_id in range(stage_boundaries[0], stage_boundaries[1]):
        node_or_layer_type = layer_types[layer_id]
        
        # Check if this is an FX node type
        if node_or_layer_type in FX_NODE_TYPES:
            param_count += _get_fx_node_param_count(node_or_layer_type, h, v, a, kv_heads, tp)
        # Legacy layer types
        elif node_or_layer_type == "embedding_layer" or node_or_layer_type == "post_process":
            if not counted:
                counted = True
                param_count += (h*v)
        elif node_or_layer_type == "transformer_layer" or node_or_layer_type == "encoder" or node_or_layer_type == "decoder":
            param_count += ((TRANSFORMER_LAYER_PARAM_MULTIPLIER * h ** 2) + TRANSFORMER_LAYER_EXTRA_PARAMS) / tp
    # debug
    # print(f" param_count: {param_count.item()}")
    
    # Get communication bandwidth of pipeline stage 0
    dp_cost_list = []
    for i in range(int(pp.item())):
        for j in range(int(tp.item())):
            bandwidth_lst = []
            for k in range(int(dp.item())):
                rank_cur = axis2rank(axis=(0,k,j), mp_deg=tp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]
                rank_next = axis2rank(axis=(0,(k+1)%(dp.item()),j), mp_deg=tp, dp_deg=dp, pp_deg=pp)
                node_next = rank_node_map[int(rank_next.item())]
                #print(f"dp_Cost (pp-{i}, dp-{k}, tp-{j}): cur: {rank_cur.item()}, peer: {rank_next.item()}")

                if node_cur == node_next:
                    connectivity = cluster_info[node_cur][1]
                    # A40-specific bandwidth handling
                    if gpu_config.has_a40() and cluster_info[node_cur][1] == A40_INTRA_BW_MARKER:
                        connect_degree = int(tp.item() * dp.item())
                        if connect_degree <= 2:
                            connectivity = gpu_config.get_a40_detail("allreduce_nvlink")
                        elif connect_degree == 4:
                            connectivity = gpu_config.get_a40_detail("allreduce_pcie1")
                        elif connect_degree >= 8:
                            connectivity = gpu_config.get_a40_detail("allreduce_pcie2")
                else:
                    connectivity = min(cluster_info[node_cur][0], cluster_info[node_next][0])
                bandwidth_lst.append(connectivity)
        # get slowest of bandwidth
        bandwidth = min(bandwidth_lst)
        # print(f"dp_Cost (pp-{i}, dp-{k}, tp-{j}): cur: {rank_cur.item()}, peer: {rank_next.item()}, BW: {bandwidth.item()/1e9:.2f}e9")

        # Inter-node bandwidth share
        if int(tp.item())*int(dp.item()) >= gpu_per_node and int(dp.item())>1:
            bandwidth = bandwidth / int(tp.item())
        
        # Intra-node bandwidth share
        elif int(tp.item())*int(dp.item()) < gpu_per_node and int(dp.item())>1 and int(tp.item()) > 1:
            if gpu_type_lst[i] == "A10":
                bandwidth = bandwidth / (gpu_per_node/int(dp.item()))
    
        # All-reduce cost: 2(n-1)M / nB
        precision = TRAINING_CONFIG["default_precision"]  # from config.py
        dp_cost_list.append(2 * (int(dp.item()) - 1) * (param_count * precision) / (int(dp.item()) * bandwidth))
    
    # print(f"bandwidth: {bandwidth}")
    # print(f"dp_cost_list: {dp_cost_list}")
        
    return stage_boundaries, dp_cost_list


def predict(config, gbs, mbs, cluster_info, model_config, gpu_profiledb, parallel_dims, node_type, exhaustive, partition_method: str, gpu_config: GPUConfig):
    """
    Predict training cost for a given configuration.

    Args:
        partition_method: PP partition method
            - "even": Optimus Prime style (even split with minmax)
            - "minmax": Min-max load balancing (default)
            - "dp": Dynamic programming
            - "ilp": Integer Linear Programming
            - "exhaustive": Exhaustive search
    """
    total_layers = int(model_config["num_layers"])
    model_type = model_config["type"]
    cost = torch.zeros(1,)
    M, N = config.shape
    config = np.asarray(config)

    # Convert partition_method to even_split flag for backward compatibility
    even_split = (partition_method == "even")
    
    # Check if using FX node-level profiling (NPZ format)
    use_fx_nodes = _check_fx_node_format(gpu_profiledb)
       
    if np.all(config == -1):
        rank_map = defaultdict(list)
        rank_node_map = dict()

        tp_degree = parallel_dims["tp_deg"]
        dp_degree = parallel_dims["dp_deg"]
        pp_degree = parallel_dims["pp_deg"]                   
        
        # infer a GPU rank map                
        counter = 0 
        for j in range(N):
            for k in range(M):
                # TODO: bad code here, config counts from 1
                rank_map[j].append(counter)
                rank_node_map[counter] = j
                counter += 1
    
    # valid config, inferred from sa 
    else:
        config = torch.from_numpy(config)
        pp = torch.max(config).float()
        
        # infer rank_map: given node name, returns the global mapped rank(int) in (pp, dp, tp) order
        # rank_node_map: given rank, returns the node
        rank_map = defaultdict(list)
        rank_node_map = dict()
           
        tp_degree = parallel_dims["tp_deg"]
        dp_degree = parallel_dims["dp_deg"]
        pp_degree = parallel_dims["pp_deg"]                  
        
        rank_counter = np.zeros(int(pp.item()))
            
        # infer a GPU rank map                    
        for j in range(N):
            for k in range(M):
                # TODO: bad code here, config counts from 1
                cur_pp = int(config[k][j] - 1)
                rank_map[j].append(int((rank_counter[cur_pp] + cur_pp * tp_degree * dp_degree).item()))
                rank_node_map[int((rank_counter[cur_pp] + cur_pp * tp_degree * dp_degree).item())] = j
                rank_counter[cur_pp] += 1
            
    num_mb = gbs / (dp_degree * mbs)
            
    parallel_config = {"tp" : tp_degree, "dp" : dp_degree, "pp" : pp_degree, "micro_bs" : mbs, "rank_map" : rank_map, "rank_node_map": rank_node_map}
    pp_degree = int(pp_degree.item())
    
    # Get node/layer sequence based on profiling format
    if use_fx_nodes:
        # Use FX node-level sequence
        layer_types = get_node_sequence(model_type=model_type, num_layers=total_layers, use_fx_nodes=True)
    else:
        # Use legacy layer-level sequence
        layer_types = get_layer_type(model_type=model_type, n=total_layers, pp=pp_degree)

    # Build cost_e_per_gpu dict for all GPU types
    # This replaces the old cost_e_a100 / cost_e_a10 pattern
    cost_e_per_gpu = {}
    
    # Get cost_e for A100 if profile available
    if "profile_cost_a100" in gpu_profiledb and gpu_profiledb["profile_cost_a100"]:
        cost_e_per_gpu["A100"] = np.asarray(get_cost_e(
            is_a100=True, 
            model_config=model_config, 
            parallel_config=parallel_config, 
            profile_cost=gpu_profiledb["profile_cost_a100"], 
            layer_types=layer_types, 
            model_type=model_type
        ))
    
    # Get cost_e for other GPU types (A10, A40, etc.)
    if "profile_cost_a10" in gpu_profiledb and gpu_profiledb["profile_cost_a10"]:
        cost_e_other = np.asarray(get_cost_e(
            is_a100=False, 
            model_config=model_config, 
            parallel_config=parallel_config, 
            profile_cost=gpu_profiledb["profile_cost_a10"], 
            layer_types=layer_types, 
            model_type=model_type
        ))
        # Map to appropriate GPU type (A10, A40, etc.)
        # For backward compatibility, use A10 as default non-A100 type
        cost_e_per_gpu["A10"] = cost_e_other
        cost_e_per_gpu["A40"] = cost_e_other  # A40 uses same profile as A10 if not specified
    
    # Ensure we have at least one cost_e
    if not cost_e_per_gpu:
        raise ValueError("No GPU profile cost data available")
    
    # Get a reference cost_e for length calculation (use first available)
    cost_e_ref = list(cost_e_per_gpu.values())[0]
    num_nodes_or_layers = len(cost_e_ref)
    
    cost_c, layer_type = get_cost_c(cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, gpu_profiledb=gpu_profiledb, gpu_config=gpu_config, layer_types=layer_types)
    cost_c_arr = np.asarray(cost_c)

    if "T5" not in model_type:
        gpu_type_lst = get_gpu_for_stage(pp_degree, N, node_type)
        if exhaustive["exhaustive"]:
            gpu_type_lst = exhaustive["gpu_type_lst"]

        # Select partition method using unified interface
        partition_fn = get_partition_function(partition_method)
        partition, stage_comp_time_lst, _, _, stage_for_send_time_lst, stage_back_send_time_lst = partition_fn(
            num_layer=num_nodes_or_layers,
            cost_e_per_gpu=cost_e_per_gpu,
            cost_c=cost_c_arr,
            pp_degree=pp_degree,
            gpu_type_lst=gpu_type_lst,
            num_mb=num_mb,
            verbose=exhaustive.get("verbose", False)
        )

        if exhaustive["exhaustive"]:
            assert False, "Done!"
        pipecost_last, stage_wise_cost_lst = schedule(pp_degree, 
                                                    num_mb, stage_comp_time_lst, 
                                                    stage_for_send_time_lst, 
                                                    stage_back_send_time_lst)
        is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem  = EstimatePeakMemory(partition, model_config, parallel_config, layer_type, cluster_info, gbs, num_mb)
    else: # If the model is T5
        # get gpu type for each stage
        gpu_type_lst = get_gpu_for_stage(pp_degree, N, node_type)

        if pp_degree>1:
            PP_C = []
            for pp_encoder in range(1, min(pp_degree, int(num_nodes_or_layers/2))):
                pp_decoder = pp_degree - pp_encoder
                if pp_decoder <= total_layers/2:
                    PP_C.append([pp_encoder, pp_decoder])
            pipecost_last = 1000000
            if exhaustive["exhaustive"] is True:
                gpu_type_lst = exhaustive["gpu_type_lst"]
                max_t = 100000.0
                for pp_c in PP_C:
                    pp_en, pp_de = pp_c
                    partition_en, stage_comp_time_lst_en, _, _, stage_for_send_time_lst_en, stage_back_send_time_lst_en = exhaustive_partition(
                        int(num_nodes_or_layers/2), cost_e_per_gpu, cost_c_arr, pp_en, gpu_type_lst[:pp_en]
                    )
                    partition_de, stage_comp_time_lst_de, _, _, stage_for_send_time_lst_de, stage_back_send_time_lst_de = exhaustive_partition(
                        int(num_nodes_or_layers/2), cost_e_per_gpu, cost_c_arr, pp_de, gpu_type_lst[pp_en:]
                    )
                    partition = partition_en + partition_de
                    stage_latency = get_stage_latency(partition, cost_e_per_gpu, cost_c_arr, gpu_type_lst)
                    stage_time_lst_temp = [stage.get_stage_time() for stage in stage_latency]
                    stage_comp_time_lst_temp = [stage.get_comp_time() for stage in stage_latency]
                    stage_for_send_time_lst_temp = [stage.get_for_send_time() for stage in stage_latency]
                    stage_back_send_time_lst_temp = [stage.get_back_send_time() for stage in stage_latency]
                    pipecost_last_temp, stage_wise_cost_lst_temp = schedule(pp_degree, num_mb, stage_comp_time_lst_temp, stage_for_send_time_lst_temp, stage_back_send_time_lst_temp)
                    if max_t > pipecost_last_temp:
                        max_t = pipecost_last_temp
                        partition_last=partition[:]
                print("partition_last", partition_last, "max_t", max_t)
                assert False, "Done!"
                
            for pp_c in PP_C:
                pp_en, pp_de = pp_c
                partition_en, stage_comp_time_lst_en, _, _, stage_for_send_time_lst_en, stage_back_send_time_lst_en = minmax(
                    int(num_nodes_or_layers/2), cost_e_per_gpu, cost_c_arr, pp_en, gpu_type_lst[:pp_en], even_split
                )
                partition_de, stage_comp_time_lst_de, _, _, stage_for_send_time_lst_de, stage_back_send_time_lst_de = minmax(
                    int(num_nodes_or_layers/2), cost_e_per_gpu, cost_c_arr, pp_de, gpu_type_lst[pp_en:], even_split
                )
                
                partition_temp = partition_en + partition_de

                # re-minmax
                stage_latency = get_stage_latency(partition_temp, cost_e_per_gpu, cost_c_arr, gpu_type_lst)
                stage_time_lst_temp = [stage.get_stage_time() for stage in stage_latency]
                stage_comp_time_lst_temp = [stage.get_comp_time() for stage in stage_latency]
                stage_for_send_time_lst_temp = [stage.get_for_send_time() for stage in stage_latency]
                stage_back_send_time_lst_temp = [stage.get_back_send_time() for stage in stage_latency]

                pipecost_last_temp, stage_wise_cost_lst_temp = schedule(pp_degree, num_mb, stage_comp_time_lst_temp, stage_for_send_time_lst_temp, stage_back_send_time_lst_temp)
                
                if pipecost_last_temp < pipecost_last:
                    pipecost_last = pipecost_last_temp
                    partition = partition_temp
                    stage_comp_time_lst = stage_comp_time_lst_temp
                    stage_for_send_time_lst = stage_for_send_time_lst_temp
                    stage_back_send_time_lst = stage_back_send_time_lst_temp
                    stage_wise_cost_lst = stage_wise_cost_lst_temp
                    
                is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem  = EstimatePeakMemory(partition, model_config, parallel_config, layer_type, cluster_info, gbs, num_mb) # use gpu_type_lst
        else:
            partition, stage_comp_time_lst, _, _, stage_for_send_time_lst, stage_back_send_time_lst = minmax(
                int(num_nodes_or_layers), cost_e_per_gpu, cost_c_arr, pp_degree, gpu_type_lst, even_split
            )
            is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem  = EstimatePeakMemory(partition, model_config, parallel_config, layer_type, cluster_info, gbs, num_mb) # use gpu_type_lst

            pipecost_last, stage_wise_cost_lst = schedule(pp_degree, num_mb, stage_comp_time_lst, stage_for_send_time_lst, stage_back_send_time_lst)
        
    # Calculate data parallelism gradient synchronization cost
    _, dp_cost_list = dp_cost(config, cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, 
                        gpu_profiledb=gpu_profiledb, partition=partition, gpu_config=gpu_config, layer_types=layer_types, gpu_per_node=M, gpu_type_lst=gpu_type_lst)
    
    if model_type != "T5":
        all_reduce_embedding_cost = cost_all_reduce_embedding(model_config, cluster_info, parallel_config, M, gpu_config)
    else:
        all_reduce_embedding_cost = 0

    end2end_stage_latency=[]
    for i in range(len(stage_wise_cost_lst)):
        end2end_stage_latency.append(stage_wise_cost_lst[i] + dp_cost_list[i])
    cost_last = max(end2end_stage_latency) + all_reduce_embedding_cost

    max_latency = max(end2end_stage_latency)
    max_latency_index = end2end_stage_latency.index(max_latency)
    
    dp_side_cost_last = dp_cost_list[max_latency_index]
    if exhaustive["exhaustive"] is True:
        print(f"partition: {partition}")
        print(f"estimated time / pp latency / dp cost / allreduce embedding")
        print(f"{cost_last.item():.4f}, {pipecost_last.item():.4f}, {dp_side_cost_last.item():.4f}, {all_reduce_embedding_cost:.4f}")
        assert False, "Done!"
    
    return rank_map, partition, cost_last, pipecost_last, dp_side_cost_last, all_reduce_embedding_cost, is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem
    

def EstimatePeakMemory(partition, model_config, parallel_config, layer_type, cluster_info, gbs, num_mb):
    """
    Estimate peak memory usage for each pipeline stage.
    Supports both legacy layer-level and FX node-level partitioning.
    """
    h = model_config["hidden_size"] 
    v = model_config["vocab_size"]
    s = model_config["sequence_length"]
    a = model_config["num_attention_heads"]
    # For GQA
    kv_heads = model_config.get("num_key_value_heads", a)
    g = a / kv_heads
    tp = parallel_config["tp"]
    pp = parallel_config["pp"] 
    dp = parallel_config["dp"]
    b = parallel_config["micro_bs"]
    gbs = gbs
    num_mb = num_mb
    N = len(cluster_info)
    
    memory = []
    memory_zero = []
    p = pp
    if num_mb > pp:
        p = pp
    else:
        p = num_mb

    st = 0
    en = 0
    for j, stage in enumerate(partition):
        st = 0 + en 
        en += stage
        param_count = 0 # unit: bytes
        activation = 0 # unit: bytes
        
        for i in range(st, en):
            node_or_layer = layer_type[i]
            
            # Check if this is FX node type or legacy layer type
            if node_or_layer in FX_NODE_TYPES:
                # FX node-level memory estimation
                h_val = int(h.item()) if hasattr(h, 'item') else int(h)
                v_val = int(v.item()) if hasattr(v, 'item') else int(v)
                s_val = int(s.item()) if hasattr(s, 'item') else int(s)
                a_val = int(a.item()) if hasattr(a, 'item') else int(a)
                kv_val = int(kv_heads.item()) if hasattr(kv_heads, 'item') else int(kv_heads)
                tp_val = int(tp.item()) if hasattr(tp, 'item') else int(tp)
                b_val = int(b) if isinstance(b, (int, float)) else int(b.item()) if hasattr(b, 'item') else int(b)
                
                param_count += get_node_param_count(node_or_layer, h_val, v_val, a_val, kv_val, tp_val)
                activation += get_node_activation_memory(node_or_layer, b_val, s_val, h_val, a_val, tp_val) * p
            
            # Legacy layer-level memory estimation
            elif node_or_layer == "embedding_layer":
                param_count += ( h * v ) / tp
                activation += ( s * b * h * pp * p) / tp
            elif node_or_layer in ["transformer_layer", "encoder", "decoder"]:
                param_count += ( TRANSFORMER_LAYER_PARAM_MULTIPLIER * h ** 2 ) / tp
                # For GQA
                activation += (s * b * p * h) * (10 + 20 / tp + 4 / (g * tp) + (5 * a * s) / (h * tp))
            elif node_or_layer == "post_process":
                param_count += ( h * v ) / tp
                activation += (4 * s * b * h ) / tp + ( 4 * s * b * v ) / tp
                
        major = param_count * OPTIMIZER_STATE_MULTIPLIER
        major_zero = param_count * (ZERO_BASE_MEMORY_MULTIPLIER + int(ZERO_OPTIM_STATE_MEMORY / dp))
        memory.append((major + activation) / 1024 /1024 /1024)
        memory_zero.append((major_zero + activation) / 1024 / 1024 /1024)
    

    oom = False
    oom_zero = False
    error_percent=1.10
    # oom_gpumem = 0.0
    # zerooom_gpumem = 0.0
    oom_gpumem = max(memory)
    zerooom_gpumem = max(memory_zero)
    # debug    
    # print(f"partition size: {len(partition)}, \n partition: {partition}")
    # print(f"cluster size: {len(cluster_info)}, \n cluster_info: {cluster_info}")
    # print(f"memory size: {len(memory)}, oom_gpumem: {oom_gpumem}, \n {memory}")
    # print(f"memory zero size: {len(memory_zero)}, zerooom_gpumem: {zerooom_gpumem}, \n {memory_zero}")
    
    # print(f"oom_gpumem: {oom_gpumem}, \n {memory}")
    # print(f"zerooom_gpumem: {zerooom_gpumem}, \n {memory_zero}")
    
    for i in range(len(partition)):
        if len(partition) > N:
            a = int(len(partition) / N)
            j = int(i / a)
        elif len(partition) == N:
            j = i
        else:
            j = None

        if j is not None:
            # Get GPU memory limit from centralized config
            memory_max = get_memory_max_from_cluster_info(cluster_info[j])
        else:
            s = i * int(N/len(partition))
            e = (i + 1) * int(N/len(partition)) -1
            for j in range(s, e):
                # Get GPU memory limit from centralized config
                memory_max = get_memory_max_from_cluster_info(cluster_info[j])    
        
        # print(f"memory_max: {memory_max}")
        if (memory[i] * error_percent) > memory_max:
            oom = True
            oom_gpumem = memory[i] * error_percent
        
        if (memory_zero[i]) > memory_max:
            oom_zero = True
            zerooom_gpumem = memory_zero[i] * error_percent
    # debug              
    # print(f"is oom: {oom}")
    # print(f"is zero oom: {oom_zero}")
    
    # print(f"is oom_gpumem: {oom_gpumem}")
    # print(f"is zerooom_gpumem: {zerooom_gpumem}")

    # Convert tensor to float if needed
    if hasattr(oom_gpumem, 'item'):
        oom_gpumem = oom_gpumem.item()
    elif hasattr(oom_gpumem, '__float__'):
        oom_gpumem = float(oom_gpumem)
    if hasattr(zerooom_gpumem, 'item'):
        zerooom_gpumem = zerooom_gpumem.item()
    elif hasattr(zerooom_gpumem, '__float__'):
        zerooom_gpumem = float(zerooom_gpumem)

    return oom, oom_gpumem, oom_zero, zerooom_gpumem


def get_layer_type(model_type, n, pp):
    """
    Legacy function for backward compatibility.
    Returns layer-level types.
    """
    layer_types = ["embedding_layer"]
    if model_type != "T5":
        for i in range(n):
            layer_types.append("transformer_layer")
        layer_types.append("post_process")
    else:
        for i in range(int(n/2)):
            layer_types.append("encoder")
        layer_types.append("embedding_layer")
        for i in range(int(n/2)):
            layer_types.append("decoder")
    return layer_types


def get_node_sequence(model_type, num_layers, use_fx_nodes=True):
    """
    Generate the sequence of FX graph nodes for the entire model.
    
    Args:
        model_type: Model type string (e.g., "llama70b", "llama1b")
        num_layers: Number of transformer layers
        use_fx_nodes: If True, return FX node-level sequence; if False, return layer-level
    
    Returns:
        List of node types representing the model's computation graph
    
    FX Node sequence for one transformer layer (8 nodes):
        - attn_q, attn_k, attn_v, attn_o (attention)
        - mlp_gate, mlp_act_fn, mlp_up, mlp_down (MLP)
    
    Full model:
        - embed (embedding layer)
        - [attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_act_fn, mlp_up, mlp_down] x num_layers
        - lm_head (language model head)
    
    Total nodes = 1 (embed) + num_layers * 8 + 1 (lm_head)
    """
    if not use_fx_nodes:
        # Fall back to legacy layer-level
        return get_layer_type(model_type, num_layers, 1)
    
    # Use the centralized node sequence builder from partition.py
    # This ensures consistency with partition logic
    return get_node_type_sequence(num_layers)


def get_nodes_per_layer():
    """
    Returns the number of FX nodes per transformer layer.
    """
    return 8  # attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_up, mlp_down, mlp_act_fn


def node_sequence_to_layer_costs(node_costs, num_layers):
    """
    Aggregate FX node costs back to layer-level costs.
    Useful for compatibility with layer-based partitioning.
    
    Args:
        node_costs: Array of costs for each FX node
        num_layers: Number of transformer layers
    
    Returns:
        layer_costs: Array of costs for [embed, layer_0, layer_1, ..., layer_n-1]
    """
    nodes_per_layer = get_nodes_per_layer()
    
    layer_costs = []
    
    # Embedding cost (first node)
    layer_costs.append(node_costs[0])
    
    # Transformer layer costs
    for i in range(num_layers):
        start_idx = 1 + i * nodes_per_layer
        end_idx = start_idx + nodes_per_layer
        layer_cost = np.sum(node_costs[start_idx:end_idx])
        layer_costs.append(layer_cost)
    
    return np.array(layer_costs)
