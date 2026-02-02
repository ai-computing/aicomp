"""
Centralized configuration for FASOP
All hardware specs, model configs, and pricing information in one place.
"""

import torch

# =============================================================================
# GPU Specifications
# bandwidth: bps (bits per second)
# memory: GB
# =============================================================================

# A40 marker value for intra-node bandwidth (used to detect A40 in cluster_info)
A40_INTRA_BW_MARKER = torch.tensor([40]).float()

GPU_SPECS = {
    # A100 GPU (p4d.24xlarge)
    "A100": {
        "inter_node_bandwidth": torch.tensor([40 * 1e9]).float(),  # 40 GB/s
        "intra_node_bandwidth": torch.tensor([1840 * 1e9]).float(),  # NVLink 1840 GB/s
        "memory_gb": 39.59,
    },
    
    # A10 GPU (g5.24xlarge) - default
    "A10": {
        "inter_node_bandwidth": torch.tensor([40 * 1e9]).float(),  # 40 GB/s
        "intra_node_bandwidth": torch.tensor([128 * 1e9]).float(),
        "memory_gb": 22.20,
    },
    
    # A40 GPU (ETRI)
    "A40": {
        "inter_node_bandwidth": torch.tensor([100 * 1e9]).float(),
        "intra_node_bandwidth": torch.tensor([40]).float(),  # marker value
        "memory_gb": 47.0,
        # A40-specific bandwidth settings (Gbps * 8 for bytes)
        "sendrecv_nvlink": torch.tensor([46.89 * 8 * 1e9]).float(),
        "sendrecv_pcie1": torch.tensor([22.8 * 8 * 1e9]).float(),
        "sendrecv_pcie2": torch.tensor([16.52 * 8 * 1e9]).float(),
        "allreduce_nvlink": torch.tensor([43.94 * 8 * 1e9]).float(),
        "allreduce_pcie1": torch.tensor([19.18 * 8 * 1e9]).float(),
        "allreduce_pcie2": torch.tensor([13.8 * 8 * 1e9]).float(),
    },
    
    # A100 for Pareto experiments
    "A100_pareto": {
        "inter_node_bandwidth": torch.tensor([400 * 1e9]).float(),
        "intra_node_bandwidth": torch.tensor([1840 * 1e9]).float(),
        "memory_gb": 39.59,
    },
    
    # A10 for Pareto experiments
    "A10_pareto": {
        "inter_node_bandwidth": torch.tensor([100 * 1e9]).float(),
        "intra_node_bandwidth": torch.tensor([252 * 1e9]).float(),
        "memory_gb": 22.20,
    },
}

# =============================================================================
# GPU Cluster Configuration
# =============================================================================

# Supported GPU types (add new GPU types here)
SUPPORTED_GPU_TYPES = ["A100", "A10", "A40"]

# GPU type to instance type mapping
GPU_TO_INSTANCE = {
    "A100": "p4d.24xlarge",
    "A10": "g5.24xlarge",
    "A40": "A40",
}

# GPU type internal markers (for cluster_info)
# '1' = A100 (high-end), '0' = A10/A40 (other)
GPU_TYPE_MARKERS = {
    "A100": "1",
    "A10": "0",
    "A40": "0",
}


def parse_gpu_cluster(gpu_args: list) -> dict:
    """
    Parse GPU cluster specification from command line arguments.
    
    Args:
        gpu_args: List of strings like ["A40", "8", "A100", "1"]
                  Format: [gpu_type, count, gpu_type, count, ...]
    
    Returns:
        dict: {gpu_type: count} e.g., {"A40": 8, "A100": 1}
    
    Example:
        parse_gpu_cluster(["A40", "8", "A100", "1"])
        -> {"A40": 8, "A100": 1}
    """
    if not gpu_args:
        return {}
    
    cluster = {}
    i = 0
    while i < len(gpu_args):
        gpu_type = gpu_args[i].upper()
        if gpu_type not in SUPPORTED_GPU_TYPES:
            raise ValueError(f"Unsupported GPU type: {gpu_type}. Supported: {SUPPORTED_GPU_TYPES}")
        
        if i + 1 >= len(gpu_args):
            raise ValueError(f"Missing count for GPU type: {gpu_type}")
        
        try:
            count = int(gpu_args[i + 1])
        except ValueError:
            raise ValueError(f"Invalid count for {gpu_type}: {gpu_args[i + 1]}")
        
        cluster[gpu_type] = count
        i += 2
    
    return cluster


def get_total_gpus(cluster: dict) -> int:
    """Get total number of GPUs in cluster."""
    return sum(cluster.values())


def get_cluster_info_from_spec(cluster: dict) -> dict:
    """
    Convert cluster specification to cluster_info format used internally.
    
    Args:
        cluster: {gpu_type: count} e.g., {"A40": 8, "A100": 1}
    
    Returns:
        cluster_info dict with node indices as keys and '0'/'1' markers as values
        '1' = A100 (high-end), '0' = other GPUs
    """
    cluster_info = {}
    node_idx = 0
    
    # Process A100 first (marker '1')
    if "A100" in cluster:
        for _ in range(cluster["A100"]):
            cluster_info[node_idx] = "1"
            node_idx += 1
    
    # Process other GPUs (marker '0')
    for gpu_type, count in cluster.items():
        if gpu_type != "A100":
            for _ in range(count):
                cluster_info[node_idx] = "0"
                node_idx += 1
    
    return cluster_info


def get_gpu_counts_from_cluster(cluster: dict) -> tuple:
    """
    Get counts of each GPU type category.
    
    Returns:
        (num_a100, num_other) - count of A100s and other GPUs
    """
    num_a100 = cluster.get("A100", 0)
    num_other = sum(count for gpu_type, count in cluster.items() if gpu_type != "A100")
    return num_a100, num_other


# =============================================================================
# Instance Pricing (USD per hour)
# =============================================================================

INSTANCE_PRICING = {
    "p4d.24xlarge": 32.7726,   # A100 instance
    "g5.24xlarge": 8.144,      # A10 instance (default)
    "g5.24xlarge_pareto": 9.773,  # A10 for pareto experiments
    "A40": 8.144,              # A40 instance (same as g5 for now)
}

# =============================================================================
# Instance to GPU Mapping
# =============================================================================

INSTANCE_GPU_MAP = {
    "p4d.24xlarge": "A100",
    "g5.12xlarge": "A10",
    "g5.24xlarge": "A10",
    "A40": "A40",
}

# =============================================================================
# Training Configuration
# =============================================================================

TRAINING_CONFIG = {
    "default_precision": 16,  # FP16
    "default_gbs": 32,        # Global batch size
}

# =============================================================================
# Dataset Configuration
# =============================================================================

# Dataset configurations for training
# dataset_len: Number of samples in the dataset
# Used to calculate iterations: iterations = dataset_len // gbs
DATASET_CONFIGS = {
    # Pre-training datasets
    "c4": {
        "dataset_len": 364_868_892,  # ~365M samples (C4 dataset)
        "description": "Colossal Clean Crawled Corpus",
    },
    "pile": {
        "dataset_len": 210_607_728,  # ~211M samples (The Pile)
        "description": "The Pile - 825GB diverse text dataset",
    },
    "redpajama": {
        "dataset_len": 1_000_000_000,  # ~1B samples
        "description": "RedPajama dataset",
    },
    "openwebtext": {
        "dataset_len": 8_013_769,  # ~8M samples
        "description": "OpenWebText corpus",
    },
    # Fine-tuning / QA datasets
    "squad": {
        "dataset_len": 10_334,  # ~10k samples (SQuAD v1.1 train)
        "description": "Stanford Question Answering Dataset",
    },
    # Default for quick testing
    "default": {
        "dataset_len": 32_000,  # 32k samples (~1000 iters at gbs=32)
        "description": "Default test configuration",
    },
}

# Supported dataset names
SUPPORTED_DATASETS = list(DATASET_CONFIGS.keys())


def get_iterations(dataset: str, gbs: int) -> int:
    """
    Calculate number of training iterations based on dataset size.

    Args:
        dataset: Dataset name (must be in DATASET_CONFIGS)
        gbs: Global batch size

    Returns:
        Number of iterations for full training

    Formula:
        iterations = dataset_len // gbs

    Example:
        get_iterations("c4", gbs=32)
        -> 364M // 32 = ~11.4M iterations
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {SUPPORTED_DATASETS}")

    dataset_len = DATASET_CONFIGS[dataset]["dataset_len"]
    iterations = dataset_len // gbs
    return iterations


def get_dataset_info(dataset: str) -> dict:
    """Get dataset configuration info."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {SUPPORTED_DATASETS}")
    return DATASET_CONFIGS[dataset]


# =============================================================================
# FX Graph Node Definitions
# =============================================================================

# FX graph node types for fine-grained cost modeling
# NPZ format: 10 columns = [embed, attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_act_fn, mlp_up, mlp_down, lm_head]
FX_NODE_TYPES = [
    "embed",       # Column 0: Embedding layer (single, model start)
    "attn_q",      # Column 1: Attention Q projection
    "attn_k",      # Column 2: Attention K projection
    "attn_v",      # Column 3: Attention V projection
    "attn_o",      # Column 4: Attention output projection
    "mlp_gate",    # Column 5: MLP gate projection (for gated architectures like LLaMA)
    "mlp_act_fn",  # Column 6: MLP activation function
    "mlp_up",      # Column 7: MLP up projection
    "mlp_down",    # Column 8: MLP down projection
    "lm_head",     # Column 9: Language model head (single, model end)
]

# Node types within a single transformer layer (excluding embed and lm_head)
# These are repeated for each transformer layer
LAYER_NODE_TYPES = FX_NODE_TYPES[1:-1]  # attn_q through mlp_down (8 nodes)

# Number of nodes per transformer layer
NODES_PER_LAYER = len(LAYER_NODE_TYPES)  # 8

# Node index mapping for quick lookup
FX_NODE_INDEX = {node: idx for idx, node in enumerate(FX_NODE_TYPES)}

# Micro-batch sizes for cost profiling (rows in NPZ files)
MBS_OPTIONS = [1, 2, 4, 8, 16, 32]

def get_mbs_index(mbs: int) -> int:
    """
    Returns the index for the given micro-batch size in the cost profile.
    """
    if mbs in MBS_OPTIONS:
        return MBS_OPTIONS.index(mbs)
    # Find the closest larger mbs
    for i, m in enumerate(MBS_OPTIONS):
        if m >= mbs:
            return i
    return len(MBS_OPTIONS) - 1


# Node categories for grouping
FX_NODE_CATEGORIES = {
    "embedding": ["embed", "lm_head"],  # lm_head often shares weights with embed
    "attention": ["attn_q", "attn_k", "attn_v", "attn_o"],
    "mlp": ["mlp_gate", "mlp_act_fn", "mlp_up", "mlp_down"],
}

# Nodes that produce activations for communication (pipeline parallel)
# These are the nodes at stage boundaries that need to send/receive activations
FX_NODES_WITH_ACTIVATION = ["embed", "attn_o", "mlp_down", "lm_head"]

# =============================================================================
# Memory Estimation Constants
# =============================================================================

# Transformer layer parameter count formula: 12 * h^2 (for standard transformer)
# Breakdown: Q, K, V, O projections (4 * h^2) + MLP (2 * 4h * h = 8h^2) = 12h^2
TRANSFORMER_LAYER_PARAM_MULTIPLIER = 12

# Additional parameters per layer (biases, layer norms, etc.)
TRANSFORMER_LAYER_EXTRA_PARAMS = 20800

# Memory multiplier for optimizer states (Adam with FP16 mixed precision)
# param (2 bytes) + grad (2 bytes) + m (4 bytes) + v (4 bytes) + master weights (4 bytes) + buffer = ~18x
OPTIMIZER_STATE_MULTIPLIER = 18

# ZeRO stage memory multiplier: param + grad + (optim_states / dp)
# param (2 bytes) + grad (2 bytes) + overhead = 6 bytes base
ZERO_BASE_MEMORY_MULTIPLIER = 6
ZERO_OPTIM_STATE_MEMORY = 12  # optimizer states that get sharded

# Parameter count formulas per node (as function of hidden_size h, vocab_size v, num_kv_heads, num_heads)
# These are used for memory estimation and dp_cost calculation
def get_node_param_count(node_type: str, h: int, v: int, num_heads: int, num_kv_heads: int, tp: int = 1) -> int:
    """
    Returns parameter count for a given node type.
    
    Args:
        node_type: FX node type
        h: hidden_size
        v: vocab_size
        num_heads: number of attention heads
        num_kv_heads: number of key-value heads (for GQA)
        tp: tensor parallel degree
    
    Returns:
        Parameter count (number of elements, not bytes)
    """
    head_dim = h // num_heads
    
    param_counts = {
        "embed": (h * v) // tp,
        "attn_q": (h * h) // tp,
        "attn_k": (h * num_kv_heads * head_dim) // tp,
        "attn_v": (h * num_kv_heads * head_dim) // tp,
        "attn_o": (h * h) // tp,
        "mlp_gate": (h * 4 * h) // tp,  # Assuming intermediate_size = 4 * h
        "mlp_act_fn": 0,  # No parameters (activation function)
        "mlp_up": (h * 4 * h) // tp,
        "mlp_down": (4 * h * h) // tp,
        "lm_head": (h * v) // tp,  # Often shares weights with embed
    }
    return param_counts.get(node_type, 0)


# Activation memory formulas per node (for peak memory estimation and cost_c calculation)
def get_node_activation_memory(node_type: str, b: int, s: int, h: int, num_heads: int, tp: int = 1, v: int = 0) -> int:
    """
    Returns activation memory in elements for a given node type.
    
    Args:
        node_type: FX node type
        b: micro-batch size
        s: sequence length
        h: hidden_size
        num_heads: number of attention heads
        tp: tensor parallel degree
        v: vocab_size (needed for lm_head)
    
    Returns:
        Activation size in elements (multiply by dtype bytes for memory)
    """
    head_dim = h // num_heads
    
    # Activation sizes (in elements, multiply by dtype size later)
    # These represent the OUTPUT activation volume of each node
    activation_sizes = {
        "embed": b * s * h,                  # Output: bs × s × h
        "attn_q": b * s * h // tp,           # Output: bs × s × h / tp
        "attn_k": b * s * h // tp,           # Output: bs × s × h / tp
        "attn_v": b * s * h // tp,           # Output: bs × s × h / tp
        "attn_o": b * s * h,                 # Output: bs × s × h (after all-gather)
        "mlp_gate": b * s * 4 * h // tp,     # Output: bs × s × 4h / tp
        "mlp_act_fn": b * s * 4 * h // tp,   # Output: bs × s × 4h / tp
        "mlp_up": b * s * 4 * h // tp,       # Output: bs × s × 4h / tp
        "mlp_down": b * s * h,               # Output: bs × s × h (after all-reduce)
        "lm_head": b * s * v if v > 0 else b * s * h,  # Output: bs × s × v (logits)
    }
    return activation_sizes.get(node_type, 0)


# =============================================================================
# Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    "llama1b": {
        "hidden_size": 2048,
        "sequence_length": 1024,  # Can be up to 131072
        "num_layers": 16,
        "vocab_size": 128256,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,  # For GQA
        "gbs": 32,
    },
    "llama70b": {
        "hidden_size": 8192,
        "sequence_length": 1024,
        "num_layers": 80,
        "vocab_size": 128256,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "gbs": 32,
    },   
    "gpt2XL": {
        "hidden_size": 1600,
        "sequence_length": 1024,
        "num_layers": 48,
        "vocab_size": 50257,
        "num_attention_heads": 25,
        "num_key_value_heads": 25,  # MHA (same as num_attention_heads)
        "gbs": 32,
    },
    "bert": {
        "hidden_size": 1024,
        "sequence_length": 512,
        "num_layers": 24,
        "vocab_size": 30522,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "gbs": 32,
    },
    "T5": {
        "hidden_size": 1024,
        "sequence_length": 512,
        "num_layers": 48,  # 24 encoder + 24 decoder
        "vocab_size": 32128,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "gbs": 32,
    },
}

# =============================================================================
# GPUConfig Class - Dynamic GPU Configuration Management
# =============================================================================

class GPUConfig:
    """
    Manages GPU configurations dynamically based on the cluster specification.
    Only creates configurations for GPU types that are actually used.
    
    Usage:
        gpu_config = GPUConfig(gpu_cluster={"A40": 8, "A100": 1}, pareto=False)
        bw = gpu_config.get_bandwidth("A40")  # [inter_bw, intra_bw]
        a40_detail = gpu_config.get_a40_detail("sendrecv_nvlink")  # A40-specific
    """
    
    def __init__(self, gpu_cluster: dict, pareto: bool = False):
        """
        Initialize GPU configuration for the specified cluster.
        
        Args:
            gpu_cluster: dict like {"A40": 8, "A100": 1}
            pareto: Whether to use pareto experiment settings
        """
        self.gpu_cluster = gpu_cluster
        self.pareto = pareto
        self.gpu_types = list(gpu_cluster.keys())
        
        # Build bandwidth configs only for GPUs in cluster
        self._bandwidths = {}
        self._a40_details = None
        
        for gpu_type in self.gpu_types:
            self._bandwidths[gpu_type] = self._load_bandwidth(gpu_type)
            
            # Load A40-specific details if A40 is in cluster
            if gpu_type == "A40":
                self._a40_details = self._load_a40_details()
    
    def _load_bandwidth(self, gpu_type: str) -> list:
        """Load bandwidth config for a GPU type."""
        key = f"{gpu_type}_pareto" if self.pareto and f"{gpu_type}_pareto" in GPU_SPECS else gpu_type
        spec = GPU_SPECS.get(key, GPU_SPECS["A10"])
        return [spec["inter_node_bandwidth"], spec["intra_node_bandwidth"]]
    
    def _load_a40_details(self) -> dict:
        """Load A40-specific bandwidth details."""
        a40 = GPU_SPECS["A40"]
        return {
            "sendrecv_nvlink": a40["sendrecv_nvlink"],
            "sendrecv_pcie1": a40["sendrecv_pcie1"],
            "sendrecv_pcie2": a40["sendrecv_pcie2"],
            "allreduce_nvlink": a40["allreduce_nvlink"],
            "allreduce_pcie1": a40["allreduce_pcie1"],
            "allreduce_pcie2": a40["allreduce_pcie2"],
        }
    
    def get_bandwidth(self, gpu_type: str) -> list:
        """
        Get [inter_node_bandwidth, intra_node_bandwidth] for a GPU type.
        Falls back to A10 if type not in cluster.
        """
        if gpu_type in self._bandwidths:
            return self._bandwidths[gpu_type]
        # Fallback for unknown types
        return self._load_bandwidth(gpu_type)
    
    def get_a40_detail(self, key: str):
        """
        Get A40-specific bandwidth detail.
        
        Args:
            key: One of 'sendrecv_nvlink', 'sendrecv_pcie1', 'sendrecv_pcie2',
                 'allreduce_nvlink', 'allreduce_pcie1', 'allreduce_pcie2'
        
        Returns:
            Bandwidth tensor, or None if A40 not in cluster
        """
        if self._a40_details is None:
            return None
        return self._a40_details.get(key)
    
    def has_a40(self) -> bool:
        """Check if A40 is in the cluster."""
        return "A40" in self.gpu_types
    
    def has_gpu_type(self, gpu_type: str) -> bool:
        """Check if a GPU type is in the cluster."""
        return gpu_type in self.gpu_types
    
    def get_primary_other_gpu(self) -> str:
        """
        Get the primary non-A100 GPU type in the cluster.
        Used for backward compatibility where code expects 'A10' slot.
        """
        for gpu_type in self.gpu_types:
            if gpu_type != "A100":
                return gpu_type
        return "A10"  # Default fallback
    
    def get_memory_gb(self, gpu_type: str) -> float:
        """Get GPU memory in GB."""
        return GPU_SPECS.get(gpu_type, GPU_SPECS["A10"])["memory_gb"]
    
    def get_instance_type(self, gpu_type: str) -> str:
        """Get instance type for a GPU type."""
        return GPU_TO_INSTANCE.get(gpu_type, "g5.24xlarge")
    
    def __repr__(self):
        return f"GPUConfig(gpu_types={self.gpu_types}, pareto={self.pareto})"


# =============================================================================
# Helper Functions (kept for backward compatibility)
# =============================================================================

def get_gpu_bandwidth(gpu_type: str, pareto: bool = False):
    """
    Returns [inter_node_bandwidth, intra_node_bandwidth] for the given GPU type.
    Compatible with existing code that expects a list of two tensors.
    
    Note: Prefer using GPUConfig class for new code.
    """
    key = f"{gpu_type}_pareto" if pareto and f"{gpu_type}_pareto" in GPU_SPECS else gpu_type
    spec = GPU_SPECS.get(key, GPU_SPECS["A10"])
    return [spec["inter_node_bandwidth"], spec["intra_node_bandwidth"]]


def get_a40_bandwidth():
    """
    Returns A40-specific bandwidth dict.
    
    Note: Prefer using GPUConfig.get_a40_detail() for new code.
    """
    a40 = GPU_SPECS["A40"]
    return {
        "sendrecv_nvlink": a40["sendrecv_nvlink"],
        "sendrecv_pcie1": a40["sendrecv_pcie1"],
        "sendrecv_pcie2": a40["sendrecv_pcie2"],
        "allreduce_nvlink": a40["allreduce_nvlink"],
        "allreduce_pcie1": a40["allreduce_pcie1"],
        "allreduce_pcie2": a40["allreduce_pcie2"],
    }


def get_instance_price_per_sec(instance_type: str, pareto: bool = False):
    """
    Returns the price per second for the given instance type.
    """
    key = f"{instance_type}_pareto" if pareto and f"{instance_type}_pareto" in INSTANCE_PRICING else instance_type
    return INSTANCE_PRICING.get(key, INSTANCE_PRICING["g5.24xlarge"]) / 3600


def get_gpu_memory_gb(gpu_type: str):
    """
    Returns GPU memory in GB for OOM checks.
    """
    return GPU_SPECS.get(gpu_type, GPU_SPECS["A10"])["memory_gb"]


def get_model_config_dict(model_type: str, precision: int = 16):
    """
    Returns model config in the format expected by existing code.
    Compatible with model_config.py's get_model_config().
    """
    if model_type not in MODEL_CONFIGS:
        # Handle llama -> llama1b mapping
        if model_type == "llama":
            model_type = "llama1b"
        else:
            raise ValueError(f"Model type '{model_type}' not supported")
    
    cfg = MODEL_CONFIGS[model_type]
    
    model_config = {
        "hidden_size": torch.tensor([cfg["hidden_size"]]).float(),
        "sequence_length": torch.tensor([cfg["sequence_length"]]).float(),
        "num_layers": torch.tensor([cfg["num_layers"]]).float(),
        "vocab_size": torch.tensor([cfg["vocab_size"]]).float(),
        "num_attention_heads": torch.tensor([cfg["num_attention_heads"]]).float(),
        "num_key_value_heads": torch.tensor([cfg["num_key_value_heads"]]).float(),
        "type": model_type,
        "precision": torch.tensor(precision).float(),
    }
    
    return model_config, cfg["gbs"]


def get_memory_max_from_cluster_info(cluster_info_entry):
    """
    Determines GPU memory limit based on cluster_info bandwidth signature.
    Used in EstimatePeakMemory for OOM checks.
    
    Args:
        cluster_info_entry: cluster_info[j] - a list of [inter_bw, intra_bw]
    
    Returns:
        memory_max in GB
    """
    intra_bw = cluster_info_entry[1]
    
    # A100: intra_node_bandwidth = 1840 * 1e9 (NVLink)
    if intra_bw == GPU_SPECS["A100"]["intra_node_bandwidth"]:
        return get_gpu_memory_gb("A100")
    # A40: uses marker value
    elif intra_bw == A40_INTRA_BW_MARKER:
        return get_gpu_memory_gb("A40")
    # A10: default
    else:
        return get_gpu_memory_gb("A10")


# =============================================================================
# ProfileDB Class - Dynamic Profile Cost Management
# =============================================================================

import os
import glob
import numpy as np
import re
from typing import Dict, List, Tuple, Optional


class ProfileDB:
    """
    Manages GPU cost profiles dynamically based on available NPZ/NPY files.
    
    Automatically discovers and loads profile data for GPU types in the cluster.
    Supports both:
      - NPZ format (FX node-level): shape (num_mbs, num_fx_nodes)
      - NPY format (layer-level, legacy): shape (num_layers,)
    
    Usage:
        profile_db = ProfileDB(
            gpu_types=["A40", "A100"],
            model_type="llama70b",
            profile_dir="known_cost"
        )
        
        # Get available TP values for a GPU type
        tp_list = profile_db.get_available_tp("A40")  # [1, 2, 4, 8]
        
        # Get cost data for specific GPU/TP
        cost_data = profile_db.get_profile("A40", tp=2)
        
        # Get cost for specific node/mbs
        cost = profile_db.get_node_cost("A40", tp=2, mbs=4, node_type="attn_q")
    """
    
    # File format patterns
    NPZ_PATTERN = "{model}_{gpu}_{tp}.npz"
    NPY_PATTERN = "{model}_{gpu}_{tp}.npy"
    
    def __init__(
        self,
        gpu_types: List[str],
        model_type: str,
        profile_dir: str = "known_cost",
        forward_backward_multiplier: float = 3.0,
    ):
        """
        Initialize ProfileDB.
        
        Args:
            gpu_types: List of GPU types to load profiles for (e.g., ["A40", "A100"])
            model_type: Model type string (e.g., "llama70b")
            profile_dir: Directory containing profile files
            forward_backward_multiplier: Multiplier for forward+backward pass (default: 3.0)
        """
        self.gpu_types = gpu_types
        self.model_type = model_type
        self.profile_dir = profile_dir
        self.fb_multiplier = forward_backward_multiplier
        
        # Profile storage: {gpu_type: {tp_str: numpy_array}}
        self._profiles: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Track format per GPU/TP: {gpu_type: {tp_str: "npz" or "npy"}}
        self._formats: Dict[str, Dict[str, str]] = {}
        
        # Discover and load available profiles
        self._discover_and_load_profiles()
    
    def _discover_and_load_profiles(self) -> None:
        """
        Scan profile directory and load all available profiles for the specified GPU types.
        """
        for gpu_type in self.gpu_types:
            self._profiles[gpu_type] = {}
            self._formats[gpu_type] = {}
            
            # Discover available profiles for this GPU type
            available = self._discover_profiles_for_gpu(gpu_type)
            
            for tp, file_path, fmt in available:
                self._load_profile(gpu_type, tp, file_path, fmt)
    
    def _discover_profiles_for_gpu(self, gpu_type: str) -> List[Tuple[int, str, str]]:
        """
        Discover available profile files for a GPU type.
        
        Returns:
            List of (tp_value, file_path, format) tuples
        """
        available = []
        
        # Pattern matching for NPZ files (preferred)
        # e.g., llama70b_A40_2.npz, llama_A40_2.npz
        npz_patterns = [
            os.path.join(self.profile_dir, f"{self.model_type}_{gpu_type}_*.npz"),
            # Also try without model version suffix (llama vs llama70b)
            os.path.join(self.profile_dir, f"{self.model_type.rstrip('0123456789b')}_{gpu_type}_*.npz"),
        ]
        
        npy_patterns = [
            os.path.join(self.profile_dir, f"{self.model_type}_{gpu_type}_*.npy"),
            os.path.join(self.profile_dir, f"{self.model_type.rstrip('0123456789b')}_{gpu_type}_*.npy"),
        ]
        
        found_files = set()
        
        # Search NPZ files first (preferred format)
        for pattern in npz_patterns:
            for file_path in glob.glob(pattern):
                if file_path not in found_files:
                    tp = self._extract_tp_from_filename(file_path)
                    if tp is not None:
                        available.append((tp, file_path, "npz"))
                        found_files.add(file_path)
        
        # Search NPY files (legacy format)
        for pattern in npy_patterns:
            for file_path in glob.glob(pattern):
                if file_path not in found_files:
                    tp = self._extract_tp_from_filename(file_path)
                    if tp is not None:
                        # Only add NPY if we don't already have NPZ for this TP
                        tp_str = str(tp)
                        has_npz = any(t == tp and f == "npz" for t, _, f in available)
                        if not has_npz:
                            available.append((tp, file_path, "npy"))
                            found_files.add(file_path)
        
        return sorted(available, key=lambda x: x[0])
    
    def _extract_tp_from_filename(self, file_path: str) -> Optional[int]:
        """
        Extract TP value from filename.
        e.g., "llama70b_A40_2.npz" -> 2
        """
        basename = os.path.basename(file_path)
        # Match pattern: {model}_{gpu}_{tp}.npz or .npy
        match = re.search(r'_(\d+)\.(npz|npy)$', basename)
        if match:
            return int(match.group(1))
        return None
    
    def _load_profile(self, gpu_type: str, tp: int, file_path: str, fmt: str) -> None:
        """
        Load a profile file into memory.
        
        Args:
            gpu_type: GPU type string
            tp: Tensor parallel degree
            file_path: Path to profile file
            fmt: File format ("npz" or "npy")
        """
        tp_str = str(tp)
        
        try:
            if fmt == "npz":
                data = np.load(file_path)['data']
            else:  # npy
                data = np.load(file_path)
            
            # Apply forward+backward multiplier
            self._profiles[gpu_type][tp_str] = self.fb_multiplier * data
            self._formats[gpu_type][tp_str] = fmt
            
        except Exception as e:
            print(f"Warning: Failed to load profile {file_path}: {e}")
    
    def get_available_tp(self, gpu_type: str) -> List[int]:
        """
        Get list of available TP values for a GPU type.
        
        Args:
            gpu_type: GPU type string
            
        Returns:
            Sorted list of available TP values
        """
        if gpu_type not in self._profiles:
            return []
        return sorted([int(tp) for tp in self._profiles[gpu_type].keys()])
    
    def get_profile(self, gpu_type: str, tp: int) -> Optional[np.ndarray]:
        """
        Get profile data for a GPU type and TP value.
        
        Args:
            gpu_type: GPU type string
            tp: Tensor parallel degree
            
        Returns:
            Profile data array, or None if not available
        """
        tp_str = str(tp)
        
        if gpu_type not in self._profiles:
            return None
        
        if tp_str in self._profiles[gpu_type]:
            return self._profiles[gpu_type][tp_str]
        
        # Fallback: try to find closest available TP
        available_tps = self.get_available_tp(gpu_type)
        if not available_tps:
            return None
        
        # Find closest TP that is >= requested
        for avail_tp in available_tps:
            if avail_tp >= tp:
                return self._profiles[gpu_type][str(avail_tp)]
        
        # If none >= requested, use largest available
        return self._profiles[gpu_type][str(available_tps[-1])]
    
    def get_format(self, gpu_type: str, tp: int) -> Optional[str]:
        """
        Get the file format for a GPU type and TP value.
        
        Returns:
            "npz" or "npy", or None if not available
        """
        tp_str = str(tp)
        if gpu_type in self._formats and tp_str in self._formats[gpu_type]:
            return self._formats[gpu_type][tp_str]
        return None
    
    def is_fx_node_format(self, gpu_type: str, tp: int) -> bool:
        """
        Check if profile is in FX node format (2D NPZ) vs layer format (1D NPY).
        
        Returns:
            True if FX node format (2D array), False otherwise
        """
        profile = self.get_profile(gpu_type, tp)
        if profile is None:
            return False
        return profile.ndim == 2
    
    def get_node_cost(
        self,
        gpu_type: str,
        tp: int,
        mbs: int,
        node_type: str,
    ) -> float:
        """
        Get execution cost for a specific FX node.
        
        Args:
            gpu_type: GPU type string
            tp: Tensor parallel degree
            mbs: Micro-batch size
            node_type: FX node type (e.g., "attn_q", "mlp_down")
            
        Returns:
            Execution cost in seconds, or 0.0 if not available
        """
        profile = self.get_profile(gpu_type, tp)
        if profile is None:
            return 0.0
        
        # Handle 1D (layer-level) vs 2D (node-level) profiles
        if profile.ndim == 1:
            # Legacy layer-level profile - can't get node-specific cost
            return 0.0
        
        # 2D NPZ format: (mbs_options, fx_nodes)
        mbs_idx = get_mbs_index(mbs)
        if mbs_idx >= profile.shape[0]:
            mbs_idx = profile.shape[0] - 1
        
        if node_type not in FX_NODE_INDEX:
            return 0.0
        
        node_idx = FX_NODE_INDEX[node_type]
        if node_idx >= profile.shape[1]:
            return 0.0
        
        return float(profile[mbs_idx, node_idx])
    
    def get_all_profiles_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get all loaded profiles as nested dict.
        
        Returns:
            {gpu_type: {tp_str: numpy_array}}
        """
        return self._profiles
    
    def get_legacy_format_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get profiles in legacy format for backward compatibility.
        
        Returns:
            {"profile_cost_a100": {...}, "profile_cost_a10": {...}}
            
        Note: Maps first A100-class GPU to a100, first non-A100 to a10
        """
        result = {
            "profile_cost_a100": {},
            "profile_cost_a10": {},
        }
        
        for gpu_type in self.gpu_types:
            if gpu_type not in self._profiles:
                continue
            
            # Determine which slot to use
            if gpu_type == "A100":
                target_key = "profile_cost_a100"
            else:
                target_key = "profile_cost_a10"
            
            # Copy profiles to appropriate slot
            for tp_str, data in self._profiles[gpu_type].items():
                # Only overwrite if empty (first GPU of type wins)
                if tp_str not in result[target_key]:
                    result[target_key][tp_str] = data
        
        return result
    
    def summary(self) -> str:
        """Get summary string of loaded profiles."""
        lines = ["ProfileDB Summary:"]
        for gpu_type in self.gpu_types:
            tps = self.get_available_tp(gpu_type)
            if tps:
                formats = [self.get_format(gpu_type, tp) for tp in tps]
                tp_info = ", ".join([f"TP{tp}({fmt})" for tp, fmt in zip(tps, formats)])
                lines.append(f"  {gpu_type}: {tp_info}")
            else:
                lines.append(f"  {gpu_type}: No profiles found")
        return "\n".join(lines)
    
    def __repr__(self):
        return f"ProfileDB(model={self.model_type}, gpus={self.gpu_types})"
