"""
Partition module for FASOP.

This module handles pipeline parallel stage partitioning logic,
simulating the split methods from Optimus Prime's IR.py.

Key Concepts:
    - Node: A single torch.nn.Module instance (e.g., q_proj, k_proj, mlp_down)
    - Layer: A transformer layer containing multiple nodes
    - Stage: A pipeline stage containing multiple nodes (not layers!)

NPZ Profile Structure:
    - Shape: (num_mbs_options, 10)
    - Columns: [embed, attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_act_fn, mlp_up, mlp_down, lm_head]
    - embed (col 0): Single node at model start
    - attn_q to mlp_down (cols 1-8): Repeated for each transformer layer
    - lm_head (col 9): Single node at model end

Node Array Construction:
    - Total nodes = 1 (embed) + num_layers * 8 (layer nodes) + 1 (lm_head)
    - node_costs = [embed] + [attn_q, ..., mlp_down] * num_layers + [lm_head]

Split Methods:
    - SIMPLE: Split by node count evenly (TP=1)
    - TP_SPLIT: Split by layer boundaries (TP>1, LLaMA-style)

Partition Algorithms:
    - MINMAX: Iterative min-max balancing
    - DP: Dynamic programming
    - ILP: Integer linear programming
    - EXHAUSTIVE: Exhaustive search
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

# Import centralized constants from config.py
from config import (
    FX_NODE_TYPES,
    LAYER_NODE_TYPES,
    NODES_PER_LAYER,
    FX_NODE_INDEX as NODE_TYPE_TO_INDEX,
)


# =============================================================================
# Enums
# =============================================================================

class SplitMethod(Enum):
    """
    Pipeline stage split methods.
    
    Corresponds to IR.py's split_IR method parameter:
        - "simple" -> SIMPLE: Split by node count
        - "llama-tp-split" -> TP_SPLIT: Split by layer boundaries
    """
    SIMPLE = "simple"       # Split by node count, for TP=1
    TP_SPLIT = "tp_split"   # Split by layer boundaries, for TP>1 (LLaMA-style)
    
    @classmethod
    def from_tp_degree(cls, tp_degree: int) -> 'SplitMethod':
        """
        Auto-select split method based on TP degree.
        
        Args:
            tp_degree: Tensor parallel degree
            
        Returns:
            SIMPLE if TP=1, TP_SPLIT if TP>1
        """
        if tp_degree == 1:
            return cls.SIMPLE
        return cls.TP_SPLIT


class PartitionAlgorithm(Enum):
    """
    Partition search algorithms for load balancing.
    """
    MINMAX = "minmax"           # Iterative min-max balancing (default)
    DP = "dp"                   # Dynamic programming
    ILP = "ilp"                 # Integer linear programming (CPLEX)
    EXHAUSTIVE = "exhaustive"   # Exhaustive search


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StageConfig:
    """
    Configuration for a single pipeline stage (node-based).
    
    Attributes:
        stage_id: Stage index (0-based)
        start_node_idx: First node index in this stage (inclusive)
        end_node_idx: Last node index in this stage (inclusive)
        num_nodes: Number of nodes in this stage
        node_indices: List of node indices in this stage
        node_types: List of node type strings in this stage
        last_node_type: Type of the last node (for comm cost calculation)
        gpu_type: GPU type assigned to this stage (optional)
        
        # Layer-level info (derived)
        start_layer: First layer index (-1 if only embed)
        end_layer: Last layer index (-1 if only embed)
        num_layers: Number of complete layers
    """
    stage_id: int
    start_node_idx: int
    end_node_idx: int
    num_nodes: int = 0
    node_indices: List[int] = field(default_factory=list)
    node_types: List[str] = field(default_factory=list)
    last_node_type: str = "mlp_act_fn"
    gpu_type: Optional[str] = None
    
    # Layer-level info
    start_layer: int = -1
    end_layer: int = -1
    num_layers: int = 0
    
    def __post_init__(self):
        """Compute derived fields."""
        if not self.node_indices:
            self.node_indices = list(range(self.start_node_idx, self.end_node_idx + 1))
        if self.num_nodes == 0:
            self.num_nodes = len(self.node_indices)


@dataclass 
class PartitionResult:
    """
    Result of a partition operation (node-based).
    
    Attributes:
        stages: List of StageConfig objects
        partition: List of node counts per stage
        node_costs: Full node cost array used for partitioning
        split_method: Split method used
        algorithm: Partition algorithm used
        total_nodes: Total number of nodes
    """
    stages: List[StageConfig]
    partition: List[int]  # [num_nodes_stage0, num_nodes_stage1, ...]
    node_costs: Optional[np.ndarray] = None
    split_method: SplitMethod = SplitMethod.SIMPLE
    algorithm: PartitionAlgorithm = PartitionAlgorithm.MINMAX
    total_nodes: int = 0
    
    @property
    def num_stages(self) -> int:
        return len(self.stages)
    
    def get_stage(self, stage_id: int) -> StageConfig:
        """Get StageConfig by stage_id."""
        return self.stages[stage_id]
    
    def get_stage_costs(self, stage_id: int) -> np.ndarray:
        """Get the cost array for a specific stage."""
        if self.node_costs is None:
            return np.array([])
        stage = self.stages[stage_id]
        return self.node_costs[stage.start_node_idx:stage.end_node_idx + 1]


# =============================================================================
# Node Cost Array Construction
# =============================================================================

def expand_node_costs(
    profile_data: np.ndarray,
    num_layers: int,
    mbs_index: int = 0,
) -> np.ndarray:
    """
    Expand NPZ profile data into a full node cost array.
    
    NPZ structure: (num_mbs_options, 10) where 10 = [embed, attn_q, ..., mlp_down, lm_head]
    
    Expansion:
        - embed (1 node)
        - [attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_act_fn, mlp_up, mlp_down] × num_layers
        - lm_head (1 node)
    
    Args:
        profile_data: NPZ data array, shape (num_mbs, 10) or (num_mbs, 9) for legacy
        num_layers: Number of transformer layers
        mbs_index: Row index for micro-batch size (default: 0)
        
    Returns:
        1D array of node costs, length = 1 + num_layers * 8 + 1
        
    Example:
        >>> data = np.load('llama70b_A40_1.npz')['data']  # shape (4, 10)
        >>> costs = expand_node_costs(data, num_layers=80, mbs_index=0)
        >>> costs.shape
        (642,)  # 1 + 80 * 8 + 1
    """
    if profile_data.ndim == 1:
        # Already 1D (legacy format)
        row = profile_data
    else:
        # 2D format: select row for mbs
        if mbs_index >= profile_data.shape[0]:
            mbs_index = profile_data.shape[0] - 1
        row = profile_data[mbs_index]
    
    # Detect format: 10 columns (new) vs 9 columns (legacy)
    has_lm_head = (len(row) >= 10)
    
    # row = [embed, attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_act_fn, mlp_up, mlp_down, lm_head]
    embed_cost = row[0]
    
    if has_lm_head:
        layer_node_costs = row[1:-1]  # [attn_q, ..., mlp_down] = 8 values
        lm_head_cost = row[-1]
    else:
        # Legacy 9-column format: no lm_head
        layer_node_costs = row[1:]  # [attn_q, ..., mlp_down] = 8 values
        lm_head_cost = embed_cost  # Fallback: use embed cost (often shared weights)
    
    # Build full node array
    # Total: 1 (embed) + num_layers * 8 (layer nodes) + 1 (lm_head)
    total_nodes = get_total_nodes(num_layers)
    node_costs = np.zeros(total_nodes)
    
    # First node is embed
    node_costs[0] = embed_cost
    
    # Repeat layer nodes for each transformer layer
    for layer_idx in range(num_layers):
        start_idx = 1 + layer_idx * NODES_PER_LAYER
        end_idx = start_idx + NODES_PER_LAYER
        node_costs[start_idx:end_idx] = layer_node_costs
    
    # Last node is lm_head
    node_costs[-1] = lm_head_cost
    
    return node_costs


def get_node_type_sequence(num_layers: int) -> List[str]:
    """
    Get the sequence of node types for the entire model.
    
    Args:
        num_layers: Number of transformer layers
        
    Returns:
        List of node type strings
        
    Example:
        >>> get_node_type_sequence(2)
        ['embed', 
         'attn_q', 'attn_k', 'attn_v', 'attn_o', 'mlp_gate', 'mlp_act_fn', 'mlp_up', 'mlp_down',
         'attn_q', 'attn_k', 'attn_v', 'attn_o', 'mlp_gate', 'mlp_act_fn', 'mlp_up', 'mlp_down',
         'lm_head']
    """
    sequence = ["embed"]
    for _ in range(num_layers):
        sequence.extend(LAYER_NODE_TYPES)
    sequence.append("lm_head")
    return sequence


def get_total_nodes(num_layers: int) -> int:
    """
    Get total number of nodes in the model.
    
    Args:
        num_layers: Number of transformer layers
        
    Returns:
        Total node count = 1 (embed) + num_layers * 8 + 1 (lm_head)
    """
    return 1 + num_layers * NODES_PER_LAYER + 1


def node_index_to_layer(node_idx: int, num_layers: int = -1) -> int:
    """
    Convert node index to layer index.
    
    Args:
        node_idx: Node index (0 = embed, 1-8 = layer 0, 9-16 = layer 1, ..., last = lm_head)
        num_layers: Number of transformer layers (needed to detect lm_head, -1 to skip check)
        
    Returns:
        Layer index (-1 for embed, 0+ for transformer layers, -2 for lm_head)
    """
    if node_idx == 0:
        return -1  # embed
    
    # Check if this is lm_head (last node)
    if num_layers > 0:
        total_nodes = get_total_nodes(num_layers)
        if node_idx == total_nodes - 1:
            return -2  # lm_head
    
    return (node_idx - 1) // NODES_PER_LAYER


def node_index_to_type(node_idx: int, num_layers: int) -> str:
    """
    Convert node index to node type.
    
    Args:
        node_idx: Node index
        num_layers: Number of transformer layers
        
    Returns:
        Node type string
    """
    if node_idx == 0:
        return "embed"
    
    # Check if this is lm_head (last node)
    total_nodes = get_total_nodes(num_layers)
    if node_idx == total_nodes - 1:
        return "lm_head"
    
    # Position within layer (0-7)
    pos_in_layer = (node_idx - 1) % NODES_PER_LAYER
    return LAYER_NODE_TYPES[pos_in_layer]


# =============================================================================
# Stage Partitioning (Node-based)
# =============================================================================

def get_nodes_per_stage_simple(
    total_nodes: int,
    num_stages: int,
) -> List[Tuple[int, int]]:
    """
    Simple split: distribute nodes evenly across stages.
    
    This simulates IR.py's simple_split logic which divides
    call_module nodes evenly by count.
    
    Args:
        total_nodes: Total number of nodes
        num_stages: Number of pipeline stages (PP degree)
        
    Returns:
        List of (start_node_idx, end_node_idx) tuples for each stage
        
    Example:
        >>> get_nodes_per_stage_simple(641, 8)
        [(0, 79), (80, 159), (160, 239), (240, 319), (320, 399), (400, 479), (480, 559), (560, 640)]
    """
    if num_stages > total_nodes:
        raise ValueError(f"num_stages ({num_stages}) > total_nodes ({total_nodes})")
    
    nodes_per_stage = total_nodes // num_stages
    remainder = total_nodes % num_stages
    
    stage_ranges = []
    current_node = 0
    
    for stage_id in range(num_stages):
        # Distribute remainder to first stages
        stage_size = nodes_per_stage + (1 if stage_id < remainder else 0)
        start_idx = current_node
        end_idx = current_node + stage_size - 1
        stage_ranges.append((start_idx, end_idx))
        current_node += stage_size
    
    return stage_ranges


def get_nodes_per_stage_tp_split(
    num_layers: int,
    num_stages: int,
) -> List[Tuple[int, int]]:
    """
    TP-aware split: distribute nodes by layer boundaries (LLaMA-style).
    
    This simulates IR.py's llama_tp_split logic which:
    1. Treats model as (embedding + layers + lm_head) blocks
    2. Distributes blocks evenly
    3. Adjusts last stage if unbalanced
    4. Converts layer boundaries to node boundaries
    
    Args:
        num_layers: Number of transformer layers
        num_stages: Number of pipeline stages (PP degree)
        
    Returns:
        List of (start_node_idx, end_node_idx) tuples for each stage
    """
    if num_stages > num_layers:
        raise ValueError(f"num_stages ({num_stages}) > num_layers ({num_layers})")
    
    # LLaMA-style block distribution
    # Blocks: embed (block 0) + layers (blocks 1 to num_layers) + lm_head (block num_layers+1)
    num_blocks = num_layers + 2  # embed + layers + lm_head
    last_layer = num_layers - 1
    
    # Compute layer boundaries
    boundaries = [(i * num_blocks) // num_stages for i in range(num_stages + 1)]
    
    # Convert to layer lists
    stage_layers = []
    for stage_id in range(num_stages):
        start_block = boundaries[stage_id]
        end_block = boundaries[stage_id + 1] - 1
        
        layers = []
        for block in range(start_block, end_block + 1):
            layer_idx = block - 1  # Block 1 -> Layer 0
            if 0 <= layer_idx < num_layers:
                layers.append(layer_idx)
        
        stage_layers.append(layers)
    
    # Adjust last stage if unbalanced
    if num_stages > 2:
        if len(stage_layers[-1]) > len(stage_layers[-2]):
            if stage_layers[-1] and stage_layers[-1][0] != last_layer:
                stage_layers[-2].append(stage_layers[-1][0])
                stage_layers[-1] = stage_layers[-1][1:]
    
    # Total nodes including lm_head
    total_nodes = get_total_nodes(num_layers)
    
    # Convert layer indices to node indices
    # embed is node 0, layer L starts at node (1 + L * 8), lm_head is last node
    stage_ranges = []
    
    for stage_id, layers in enumerate(stage_layers):
        if stage_id == 0:
            # First stage always includes embed (node 0)
            start_node = 0
        else:
            if layers:
                # First node of first layer in this stage
                start_node = 1 + layers[0] * NODES_PER_LAYER
            else:
                # Empty stage (edge case)
                prev_end = stage_ranges[-1][1] if stage_ranges else -1
                start_node = prev_end + 1
        
        if stage_id == num_stages - 1:
            # Last stage always includes lm_head
            end_node = total_nodes - 1
        elif layers:
            # Last node of last layer in this stage
            last_layer_idx = layers[-1]
            end_node = 1 + last_layer_idx * NODES_PER_LAYER + NODES_PER_LAYER - 1
        else:
            if stage_id == 0:
                # Only embed
                end_node = 0
            else:
                # Empty stage
                end_node = start_node - 1
        
        stage_ranges.append((start_node, end_node))
    
    return stage_ranges


def get_nodes_per_stage(
    num_layers: int,
    num_stages: int,
    split_method: SplitMethod = SplitMethod.SIMPLE,
) -> List[Tuple[int, int]]:
    """
    Get node distribution across stages based on split method.
    
    Args:
        num_layers: Total number of transformer layers
        num_stages: Number of pipeline stages (PP degree)
        split_method: Split method to use
        
    Returns:
        List of (start_node_idx, end_node_idx) tuples for each stage
    """
    total_nodes = get_total_nodes(num_layers)
    
    if split_method == SplitMethod.SIMPLE:
        return get_nodes_per_stage_simple(total_nodes, num_stages)
    elif split_method == SplitMethod.TP_SPLIT:
        return get_nodes_per_stage_tp_split(num_layers, num_stages)
    else:
        raise ValueError(f"Unknown split method: {split_method}")


# =============================================================================
# Stage Configuration Builder (Node-based)
# =============================================================================

def build_stage_configs(
    num_layers: int,
    num_stages: int,
    split_method: SplitMethod = SplitMethod.SIMPLE,
    gpu_types: Optional[List[str]] = None,
    node_costs: Optional[np.ndarray] = None,
) -> List[StageConfig]:
    """
    Build StageConfig objects for all pipeline stages (node-based).
    
    Args:
        num_layers: Total number of transformer layers
        num_stages: Number of pipeline stages (PP degree)
        split_method: Split method to use
        gpu_types: Optional list of GPU types for each stage
        node_costs: Optional node cost array (for validation)
        
    Returns:
        List of StageConfig objects
    """
    stage_ranges = get_nodes_per_stage(num_layers, num_stages, split_method)
    node_type_sequence = get_node_type_sequence(num_layers)
    
    configs = []
    for stage_id, (start_idx, end_idx) in enumerate(stage_ranges):
        # Get node types for this stage
        if start_idx <= end_idx:
            node_indices = list(range(start_idx, end_idx + 1))
            node_types = [node_type_sequence[i] for i in node_indices]
            last_node_type = node_types[-1] if node_types else "embed"
        else:
            # Empty stage
            node_indices = []
            node_types = []
            last_node_type = "embed"
        
        # Compute layer info
        if node_indices:
            start_layer = node_index_to_layer(node_indices[0])
            end_layer = node_index_to_layer(node_indices[-1])
            # Count complete layers
            layers_in_stage = set()
            for ni in node_indices:
                layer = node_index_to_layer(ni)
                if layer >= 0:
                    layers_in_stage.add(layer)
            num_layers_in_stage = len(layers_in_stage)
        else:
            start_layer = -1
            end_layer = -1
            num_layers_in_stage = 0
        
        gpu_type = gpu_types[stage_id] if gpu_types and stage_id < len(gpu_types) else None
        
        config = StageConfig(
            stage_id=stage_id,
            start_node_idx=start_idx,
            end_node_idx=end_idx,
            num_nodes=len(node_indices),
            node_indices=node_indices,
            node_types=node_types,
            last_node_type=last_node_type,
            gpu_type=gpu_type,
            start_layer=start_layer,
            end_layer=end_layer,
            num_layers=num_layers_in_stage,
        )
        configs.append(config)
    
    return configs


def build_partition_result(
    num_layers: int,
    num_stages: int,
    split_method: SplitMethod = SplitMethod.SIMPLE,
    algorithm: PartitionAlgorithm = PartitionAlgorithm.MINMAX,
    gpu_types: Optional[List[str]] = None,
    profile_data: Optional[np.ndarray] = None,
    mbs_index: int = 0,
) -> PartitionResult:
    """
    Build a complete PartitionResult (node-based).
    
    Args:
        num_layers: Total number of transformer layers
        num_stages: Number of pipeline stages (PP degree)
        split_method: Split method to use
        algorithm: Partition algorithm used
        gpu_types: Optional list of GPU types for each stage
        profile_data: Optional NPZ profile data for cost computation
        mbs_index: Micro-batch size index for profile data
        
    Returns:
        PartitionResult object
    """
    # Expand node costs if profile data provided
    node_costs = None
    if profile_data is not None:
        node_costs = expand_node_costs(profile_data, num_layers, mbs_index)
    
    stages = build_stage_configs(num_layers, num_stages, split_method, gpu_types, node_costs)
    partition = [stage.num_nodes for stage in stages]
    total_nodes = get_total_nodes(num_layers)
    
    return PartitionResult(
        stages=stages,
        partition=partition,
        node_costs=node_costs,
        split_method=split_method,
        algorithm=algorithm,
        total_nodes=total_nodes,
    )


# =============================================================================
# Stage Latency Computation
# =============================================================================

def compute_stage_latencies(
    partition_result: PartitionResult,
) -> List[float]:
    """
    Compute latency for each stage based on node costs.
    
    Args:
        partition_result: PartitionResult with node_costs
        
    Returns:
        List of stage latencies (sum of node costs in each stage)
    """
    if partition_result.node_costs is None:
        raise ValueError("PartitionResult has no node_costs")
    
    latencies = []
    for stage in partition_result.stages:
        if stage.num_nodes > 0:
            stage_costs = partition_result.node_costs[stage.start_node_idx:stage.end_node_idx + 1]
            latencies.append(float(np.sum(stage_costs)))
        else:
            latencies.append(0.0)
    
    return latencies


def compute_total_latency(partition_result: PartitionResult) -> float:
    """
    Compute total latency (sum of all node costs).
    
    Args:
        partition_result: PartitionResult with node_costs
        
    Returns:
        Total latency
    """
    if partition_result.node_costs is None:
        return 0.0
    return float(np.sum(partition_result.node_costs))


def compute_max_stage_latency(partition_result: PartitionResult) -> float:
    """
    Compute maximum stage latency (pipeline bottleneck).
    
    Args:
        partition_result: PartitionResult with node_costs
        
    Returns:
        Maximum stage latency
    """
    latencies = compute_stage_latencies(partition_result)
    return max(latencies) if latencies else 0.0


# =============================================================================
# Utility Functions
# =============================================================================

def validate_partition(partition: List[int], total_nodes: int) -> bool:
    """
    Validate that a partition covers all nodes exactly once.
    
    Args:
        partition: List of node counts per stage
        total_nodes: Total number of nodes
        
    Returns:
        True if valid, False otherwise
    """
    return sum(partition) == total_nodes and all(p >= 0 for p in partition)


def partition_to_node_ranges(partition: List[int]) -> List[Tuple[int, int]]:
    """
    Convert partition list to node ranges.
    
    Args:
        partition: List of node counts per stage [80, 80, 80, ...]
        
    Returns:
        List of (start, end) tuples for each stage
    """
    ranges = []
    current = 0
    for count in partition:
        if count > 0:
            ranges.append((current, current + count - 1))
            current += count
        else:
            ranges.append((current, current - 1))  # Empty range
    return ranges


def get_stage_for_node(node_idx: int, partition_result: PartitionResult) -> int:
    """
    Find which stage a node belongs to.
    
    Args:
        node_idx: Node index
        partition_result: PartitionResult object
        
    Returns:
        Stage ID (0-based)
    """
    for stage in partition_result.stages:
        if stage.start_node_idx <= node_idx <= stage.end_node_idx:
            return stage.stage_id
    return -1


# =============================================================================
# Legacy Compatibility (Layer-based functions)
# =============================================================================

def get_layers_per_stage(
    num_layers: int,
    num_stages: int,
    split_method: SplitMethod = SplitMethod.SIMPLE,
) -> List[List[int]]:
    """
    Get layer distribution across stages (legacy compatibility).
    
    Note: This returns layer indices, not node indices.
    For node-based partitioning, use get_nodes_per_stage() instead.
    
    Args:
        num_layers: Total number of transformer layers
        num_stages: Number of pipeline stages
        split_method: Split method to use
        
    Returns:
        List of layer index lists for each stage
    """
    stage_ranges = get_nodes_per_stage(num_layers, num_stages, split_method)
    
    result = []
    for start_node, end_node in stage_ranges:
        layers = set()
        for node_idx in range(start_node, end_node + 1):
            layer = node_index_to_layer(node_idx)
            if layer >= 0:
                layers.add(layer)
        result.append(sorted(list(layers)))
    
    return result
