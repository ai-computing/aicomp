"""
Portions of this code adapted from the 'AMP' project (https://github.com/DachengLi1/AMP). 
@article{li2022amp,
  title={AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness},
  author={Li, Dacheng and Wang, Hongyi and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2210.07297},
  year={2022}
}
"""

import time
import os
import sys
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import pandas as pd

from utils import (
    enumerate_parallel_strategies,
    count_parallel_strategies,
    draw_pareto_graph,
    TeeLogger,
    JSONLLogger,
)
from estimate import FASOP
from device_placement import device_placement_from_cluster, get_all_cluster_combinations
from model_config import get_model_config
from config import (
    get_instance_price_per_sec,
    TRAINING_CONFIG,
    parse_gpu_cluster,
    get_total_gpus,
    get_gpu_counts_from_cluster,
    SUPPORTED_GPU_TYPES,
    GPUConfig,
    SUPPORTED_DATASETS,
    get_iterations,
    get_dataset_info,
    ProfileDB,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FasopConfig:
    """Configuration container for FASOP execution."""
    gpu_cluster: Dict[str, int]
    model_type: str
    dataset: str
    gbs: int
    gpu_per_node: int
    precision: int
    pareto: bool
    exhaustive: bool
    pp_partition_method: str  # "even", "minmax", "dp", "ilp", "exhaustive"
    verbose: bool
    add_exp_name: str
    heterogeneous: bool
    num_gpus: int
    log_file: Optional[str] = None  # Output log file path
    pareto_gbs_max: Optional[int] = None  # Max GBS for pareto search (powers of 2)
    parsing: bool = False  # Parsing mode flag
    parsing_file: Optional[str] = None  # Output JSONL file path for real-time logging
    save_csv: bool = True  # Whether to save results to CSV (--no-save-csv sets to False)
    make_pareto: bool = False  # Whether to generate Pareto frontier graph
    pareto_title: Optional[str] = None  # Custom title/filename for Pareto graph

    # Exhaustive mode parameters (optional)
    # Note: gpu_type_list for exhaustive mode is derived from --gpus argument
    exhaustive_mbs: Optional[int] = None
    exhaustive_tp: Optional[int] = None
    exhaustive_dp: Optional[int] = None
    exhaustive_pp: Optional[int] = None


@dataclass
class SearchResult:
    """Container for a single strategy search result."""
    gbs: int
    mbs: int
    tp: int
    dp: int
    pp: float
    node_type: List[str]
    gpu_cluster_str: str
    partition: List[int]
    step_time: float
    throughput: float
    price_per_step: float
    is_oom: bool
    oom_gpumem: float
    iterations: int
    total_time_seconds: float
    total_time_hours: float
    cost: float


# =============================================================================
# 1. Argument Parsing
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for FASOP.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="FASOP: Fast yet Accurate Automated Search for Optimal Parallelization of Transformers on Heterogeneous GPU Clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python FASOP.py --gpus A40 8 --model-type llama70b --dataset c4
  python FASOP.py --gpus A40 8 A100 1 --model-type llama70b --dataset pile
  python FASOP.py --gpus A100 4 --model-type gpt2XL --dataset squad --gbs 64
  python FASOP.py --only-make-pareto main_logs/result.csv --pareto-title my_analysis

Supported datasets: {', '.join(SUPPORTED_DATASETS)}
Note: Heterogeneous mode is auto-detected when multiple GPU types are specified.
"""
    )

    # -------------------------------------------------------------------------
    # Model & Training Configuration
    # -------------------------------------------------------------------------
    model_group = parser.add_argument_group("Model & Training")
    model_group.add_argument("--gpus", nargs='+', required=True,
                             help="GPU cluster specification (REQUIRED): --gpus A40 8 or --gpus A40 8 A100 1")
    model_group.add_argument("--model-type", type=str, default="llama70b",
                             help="Model type (default: llama70b)")
    model_group.add_argument("--dataset", type=str, default="squad",
                             help=f"Dataset for training iteration calculation (default: squad). Options: {', '.join(SUPPORTED_DATASETS)}")
    model_group.add_argument("--gbs", type=int, default=32,
                             help="Global batch size (default: 32)")
    model_group.add_argument("--gpu-per-node", type=int, default=4,
                             help="Number of GPUs per node (default: 4)")
    model_group.add_argument("--precision", type=int, default=16,
                             help="Training precision (default: 16)")

    # -------------------------------------------------------------------------
    # Search Options
    # -------------------------------------------------------------------------
    search_group = parser.add_argument_group("Search Options")
    search_group.add_argument("--exhaustive", action='store_true',
                              help="Run exhaustive search for model partitioning (default: False)")
    search_group.add_argument("--pp-partition-method", type=str, default="minmax",
                              choices=["even", "minmax", "dp", "ilp", "bruteforce"],
                              help="PP partition method: even (Optimus Prime style), minmax, dp, ilp, bruteforce (default: minmax)")

    # -------------------------------------------------------------------------
    # Pareto Options
    # -------------------------------------------------------------------------
    pareto_group = parser.add_argument_group("Pareto Options")
    pareto_group.add_argument("--pareto", action='store_true',
                              help="Run pareto experiments (default: False)")
    pareto_group.add_argument("--pareto-gbs-max", type=int, default=None,
                              help="Max GBS for pareto search (powers of 2). Example: --pareto-gbs-max 64 searches GBS=1,2,4,...,64")
    pareto_group.add_argument("--only-make-pareto", type=str, default=None, metavar="CSV_PATH",
                              help="Generate Pareto graph from existing CSV without running search")
    pareto_group.add_argument("--pareto-title", type=str, default=None,
                              help="Custom title and filename for Pareto graph")
    pareto_group.add_argument("--make-pareto", action='store_true',
                              help="Generate Pareto frontier graph when --pareto is set (default: False)")

    # -------------------------------------------------------------------------
    # Output Options
    # -------------------------------------------------------------------------
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--verbose", action='store_true',
                              help="Enable verbose logging for partition search (default: False)")
    output_group.add_argument("--log-file", type=str, default=None,
                              help="Save output log to file. Auto-generates filename when --verbose is set")
    output_group.add_argument("--add-exp-name", type=str, default="",
                              help="Additional experiment name suffix")
    output_group.add_argument("--parsing", action='store_true',
                              help="Enable parsing mode for real-time JSONL output (default: False)")
    output_group.add_argument("--parsing-file", type=str, default=None,
                              help="Custom JSONL output path for parsing mode (e.g., main_logs/my_run.jsonl)")
    output_group.add_argument("--no-save-csv", action='store_true',
                              help="Skip saving results to CSV file (default: False)")

    return parser.parse_args()


def _derive_gpu_type_lst_from_placement(placement: List[str], pp: int, gpu_per_node: int) -> List[str]:
    """
    Derive gpu_type_lst for each pipeline stage from node placement.
    
    For exhaustive mode, we derive gpu_type_lst from the --gpus argument
    instead of asking the user to input it separately.
    
    Args:
        placement: List of GPU types for each node (from --gpus)
        pp: Pipeline parallelism degree
        gpu_per_node: Number of GPUs per node
        
    Returns:
        List of GPU types for each pipeline stage
    
    Example:
        placement = ["A100", "A40", "A40", "A10"]  # 4 nodes
        pp = 4, gpu_per_node = 4
        -> ["A100", "A40", "A40", "A10"]  # 1 node per stage
        
        placement = ["A100", "A40"]  # 2 nodes  
        pp = 4, gpu_per_node = 4
        -> ["A100", "A100", "A40", "A40"]  # 2 stages per node
    """
    num_nodes = len(placement)
    gpu_type_lst = []
    
    if pp <= num_nodes:
        # Each stage covers one or more nodes
        nodes_per_stage = num_nodes / pp
        for stage in range(pp):
            # Take the GPU type from the first node in this stage's range
            node_idx = int(stage * nodes_per_stage)
            gpu_type_lst.append(placement[node_idx])
    else:
        # Multiple stages per node
        stages_per_node = pp / num_nodes
        for stage in range(pp):
            node_idx = int(stage // stages_per_node)
            gpu_type_lst.append(placement[node_idx])
    
    return gpu_type_lst


def get_exhaustive_params() -> Tuple[int, int, int, int]:
    """
    Get exhaustive search parameters from user input.
    
    Note: gpu_type_list is now derived from --gpus argument, not user input.
    
    Returns:
        Tuple of (mbs, tp, dp, pp)
    """
    print("Type parallelization strategy you want to search")
    print("(Note: gpu_type_list is derived from --gpus argument)")
    mbs = int(input("mbs: "))
    tp = int(input("tp: "))
    dp = int(input("dp: "))
    pp = int(input("pp: "))
    return mbs, tp, dp, pp


# =============================================================================
# 2. Configuration Validation and Initialization
# =============================================================================

def validate_and_init_config(args: argparse.Namespace) -> FasopConfig:
    """
    Validate arguments and initialize FASOP configuration.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        FasopConfig: Validated and initialized configuration
        
    Raises:
        SystemExit: If validation fails
    """
    # Parse GPU cluster configuration
    gpu_cluster = parse_gpu_cluster(args.gpus)
    num_gpus = get_total_gpus(gpu_cluster)
    
    if num_gpus == 0:
        raise SystemExit("Error: --gpus must specify at least one GPU. Example: --gpus A40 8")
    
    # Auto-detect heterogeneous mode
    heterogeneous = len(gpu_cluster) > 1
    
    # Validate dataset
    if args.dataset not in SUPPORTED_DATASETS:
        raise SystemExit(f"Error: Unknown dataset: {args.dataset}. Supported: {', '.join(SUPPORTED_DATASETS)}")
    
    # Adjust gpu_per_node for pareto mode
    gpu_per_node = args.gpu_per_node
    if args.pareto and gpu_per_node == 4:
        print("Pareto experiments should use 8 GPUs per node, so we will use 8 GPUs per node")
        gpu_per_node = 8
    
    # Determine log file path
    log_file = args.log_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_str = "_".join([f"{k}{v}" for k, v in gpu_cluster.items()])
    if log_file is None and args.verbose:
        # Auto-generate log filename when verbose mode is enabled
        log_file = f"main_logs/{args.model_type}_{gpu_str}_gbs{args.gbs}_{timestamp}.log"

    # Validate parsing-file usage
    if args.parsing_file and not args.parsing:
        raise SystemExit("Error: --parsing-file requires --parsing")

    # Determine parsing file path (JSONL for real-time logging)
    parsing_file = None
    if args.parsing:
        parsing_file = args.parsing_file or f"main_logs/{args.model_type}_{gpu_str}_gbs{args.gbs}_{timestamp}.jsonl"

    # Create config
    config = FasopConfig(
        gpu_cluster=gpu_cluster,
        model_type=args.model_type,
        dataset=args.dataset,
        gbs=args.gbs,
        gpu_per_node=gpu_per_node,
        precision=args.precision,
        pareto=args.pareto,
        exhaustive=args.exhaustive,
        pp_partition_method=args.pp_partition_method,
        verbose=args.verbose,
        add_exp_name=args.add_exp_name,
        heterogeneous=heterogeneous,
        num_gpus=num_gpus,
        log_file=log_file,
        pareto_gbs_max=args.pareto_gbs_max,
        parsing=args.parsing,
        parsing_file=parsing_file,
        save_csv=not args.no_save_csv,
        make_pareto=args.make_pareto,
        pareto_title=args.pareto_title,
    )
    
    # Get exhaustive parameters if needed
    if args.exhaustive:
        mbs, tp, dp, pp = get_exhaustive_params()
        config.exhaustive_mbs = mbs
        config.exhaustive_tp = tp
        config.exhaustive_dp = dp
        config.exhaustive_pp = pp
    
    return config


def print_config_summary(config: FasopConfig, model_config: dict, num_iterations: int, dataset_info: dict) -> None:
    """
    Print configuration summary to console.
    
    Args:
        config: FASOP configuration
        model_config: Model configuration dictionary
        num_iterations: Number of training iterations
        dataset_info: Dataset information dictionary
    """
    seq_len = int(model_config["sequence_length"].item())
    
    print(f"GPU Cluster: {config.gpu_cluster}, Total GPUs: {config.num_gpus}")
    print(f"Heterogeneous: {config.heterogeneous} (auto-detected from --gpus)")
    print(f"Dataset: {config.dataset}, GBS: {config.gbs}")
    
    print(f"\n{'='*60}")
    print(f"Model: {config.model_type}, Precision: {config.precision}")
    print(f"Dataset: {config.dataset} ({dataset_info['description']})")
    print(f"  Dataset samples: {dataset_info['dataset_len']:,}")
    print(f"  GBS: {config.gbs}, Seq len: {seq_len}")
    print(f"  Iterations: {num_iterations:,}")
    print(f"PP Partition Method: {config.pp_partition_method}")
    print(f"{'='*60}\n")


def setup_environment() -> Tuple[str, str]:
    """
    Setup environment paths and directories.
    
    Returns:
        Tuple of (home_path, output_dir_path)
    """
    home_path = os.environ['HOME']
    pwd_path = os.environ['PWD']
    dir_path = os.path.join(pwd_path, './main_logs')
    
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    # Remove cache directory
    tmp_path = os.path.join(home_path, "tmp")
    if os.path.exists(tmp_path):
        for root, dirs, files in os.walk(tmp_path):
            for f in files:
                os.unlink(os.path.join(root, f))
    
    return home_path, dir_path


# =============================================================================
# 3. Strategy Search
# =============================================================================

def evaluate_single_placement(
    placement: List[str],
    current_cluster: Dict[str, int],
    config: FasopConfig,
    gpu_config: GPUConfig,
    model_config: dict,
    num_iterations: int,
    search_idx_start: int = 0,
    total_count: int = 0,
    available_tps: List[int] = None,
    jsonl_logger: Optional[JSONLLogger] = None,
) -> List[SearchResult]:
    """
    Evaluate all parallel strategies for a single device placement.

    Args:
        placement: List of GPU types for each position
        current_cluster: Current cluster configuration dict
        config: FASOP configuration
        gpu_config: GPU configuration object
        model_config: Model configuration dictionary
        num_iterations: Number of training iterations
        search_idx_start: Starting index for verbose logging
        total_count: Total number of combinations for verbose logging
        available_tps: List of available TP values from NPZ files
        jsonl_logger: Optional JSONLLogger for real-time result logging

    Returns:
        List of SearchResult objects
    """
    results = []
    num_node = sum(current_cluster.values())
    search_idx = search_idx_start

    print(f"Device placement: {placement}")
    
    # Build node_type and cluster_info_for_model
    node_type = []
    cluster_info_for_model = {}
    
    for i, gpu_type in enumerate(placement):
        node_type.append(gpu_config.get_instance_type(gpu_type))
        cluster_info_for_model[i] = gpu_config.get_bandwidth(gpu_type)
    
    # Create FASOP model
    exp_name = get_exp_name(config, model_config)
    model = FASOP(model_config, exp_name, gpu_config, num_node, config.pp_partition_method)

    known = None

    # Enumerate all parallel strategies
    while True:
        parallel_config = enumerate_parallel_strategies(
            M=config.gpu_per_node,
            N=num_node,
            gbs=config.gbs,
            known=known,
            num_layers=model_config["num_layers"],
            available_tps=available_tps
        )
        
        if parallel_config is None:
            break
        
        tp_deg, dp_deg, mbs, known = parallel_config
        search_idx += 1
        pp_deg = int(config.gpu_per_node * num_node / (tp_deg * dp_deg))

        # Verbose logging - search start
        if config.verbose:
            idx_prefix = f"[{search_idx:3d}/{total_count}] "
            verbose_prefix = f"{idx_prefix}MBS={mbs:2d} PP={pp_deg:2d} DP={dp_deg:2d} TP={tp_deg}"
            print(f"{verbose_prefix} | searching...", end="", flush=True)
            search_start_time = time.time()

        # Prepare model arguments
        parallel_dim = {
            "tp_deg": torch.ones(1,) * tp_deg,
            "dp_deg": torch.ones(1,) * dp_deg,
            "pp_deg": torch.ones(1,) * (config.gpu_per_node * num_node / (tp_deg * dp_deg))
        }
        fake_config = np.ones((config.gpu_per_node, num_node)) * (-1)
        model_args = (fake_config, config.gbs, mbs, cluster_info_for_model, model_config, parallel_dim)
        
        # Handle exhaustive mode
        if config.exhaustive:
            mbs = config.exhaustive_mbs
            pp = int(config.exhaustive_pp)
            parallel_dim = {
                "tp_deg": torch.ones(1,) * int(config.exhaustive_tp),
                "dp_deg": torch.ones(1,) * int(config.exhaustive_dp),
                "pp_deg": torch.ones(1,) * pp
            }
            model_args = (fake_config, config.gbs, mbs, cluster_info_for_model, model_config, parallel_dim)
            # Derive gpu_type_lst from placement (from --gpus argument)
            # gpu_type_lst is per pipeline stage, placement is per node
            exhaustive_gpu_type_lst = _derive_gpu_type_lst_from_placement(placement, pp, config.gpu_per_node)
            exhaustive_args = {"exhaustive": True, "gpu_type_lst": exhaustive_gpu_type_lst, "verbose": config.verbose}
        else:
            exhaustive_args = {"exhaustive": False, "gpu_type_lst": None, "verbose": config.verbose}
        
        # Run estimation
        with torch.no_grad():
            (rank_map, partition, cost, pipecost, dp_side_cost,
             all_reduce_embedding_cost, is_oom, oom_gpumem,
             is_zero_oom, zerooom_gpumem) = model(model_args, node_type, exhaustive_args)

        # Verbose logging - search complete
        if config.verbose:
            search_elapsed = time.time() - search_start_time
            print(f"\r{verbose_prefix} | complete! {search_elapsed:.2f}s")

        # Calculate costs
        price_per_sec = calculate_price_per_sec(current_cluster, gpu_config, config.pareto)
        step_time = cost.item()  # cost tensor -> scalar
        price_per_step = price_per_sec * step_time
        total_cost = price_per_step * num_iterations
        total_time_seconds = step_time * num_iterations
        total_time_hours = total_time_seconds / 3600
        
        # Format cluster info
        cluster_str = ", ".join([f"{k}:{v}" for k, v in current_cluster.items()])

        # Calculate throughput (samples per second)
        throughput = config.gbs / step_time if step_time > 0 else 0.0

        # Create result
        result = SearchResult(
            gbs=config.gbs,
            mbs=mbs,
            tp=tp_deg,
            dp=dp_deg,
            pp=config.gpu_per_node * num_node / (tp_deg * dp_deg),
            node_type=node_type,
            gpu_cluster_str=cluster_str,
            partition=partition,
            step_time=step_time,
            throughput=throughput,
            price_per_step=price_per_step,
            is_oom=is_oom,
            oom_gpumem=oom_gpumem,
            iterations=num_iterations,
            total_time_seconds=total_time_seconds,
            total_time_hours=total_time_hours,
            cost=total_cost,
        )

        # Verbose logging - search result
        if config.verbose:
            # Calculate partition imbalance
            if partition and len(partition) > 0:
                avg_p = sum(partition) / len(partition)
                max_p = max(partition)
                min_p = min(partition)
                imbal = ((max_p - min_p) / avg_p * 100) if avg_p > 0 else 0
                bottleneck = partition.index(max_p)
                # Format partition string with brackets
                if len(partition) > 5:
                    part_str = f"[{partition[0]},{partition[1]},...,{partition[-1]}]"
                else:
                    part_str = str(partition)
                # Align with verbose_prefix (use idx_prefix length for padding)
                print(f"{' ' * len(idx_prefix)}{part_str:<23} | imbal={imbal:5.2f}% b={bottleneck} | step={cost.item():.4f}s")
            else:
                print(f"{' ' * len(idx_prefix)}{'partition=None':<23} | step={cost.item():.4f}s")

        # Real-time JSONL logging (--parsing mode)
        if jsonl_logger:
            jsonl_logger.log(result)

        results.append(result)

    return results


def get_pareto_gbs_values(max_gbs: int) -> List[int]:
    """
    Generate GBS values for pareto search (powers of 2 from 1 to max_gbs).

    Args:
        max_gbs: Maximum GBS value

    Returns:
        List of GBS values (powers of 2)

    Example:
        get_pareto_gbs_values(64) -> [1, 2, 4, 8, 16, 32, 64]
    """
    gbs_values = []
    gbs = 1
    while gbs <= max_gbs:
        gbs_values.append(gbs)
        gbs *= 2
    return gbs_values


def run_strategy_search(
    config: FasopConfig,
    gpu_config: GPUConfig,
    model_config: dict,
    num_iterations: int,
    jsonl_logger: Optional[JSONLLogger] = None,
) -> List[SearchResult]:
    """
    Run the main strategy search across all cluster combinations and placements.

    Args:
        config: FASOP configuration
        gpu_config: GPU configuration object
        model_config: Model configuration dictionary
        num_iterations: Number of training iterations
        jsonl_logger: Optional JSONLLogger for real-time result logging

    Returns:
        List of all SearchResult objects
    """
    from dataclasses import replace

    all_results = []

    # Create ProfileDB to get available TPs from NPZ files
    # Use all GPU types from config to discover available TPs
    all_gpu_types = list(config.gpu_cluster.keys())
    profile_db = ProfileDB(
        gpu_types=all_gpu_types,
        model_type=config.model_type,
        profile_dir=os.path.join(os.path.dirname(__file__), "known_cost"),
    )

    # Generate GBS values for pareto search if pareto_gbs_max is set
    if config.pareto_gbs_max:
        gbs_values = get_pareto_gbs_values(config.pareto_gbs_max)
        print(f"[PARETO] GBS pivoting enabled: {gbs_values}")
    else:
        gbs_values = [config.gbs]  # Single GBS value

    # Pre-calculate total combinations for pareto mode with parsing
    if config.pareto and jsonl_logger:
        pareto_total_count = 0
        pareto_breakdown = []  # List of {gbs, cluster, count} for each combination

        for gbs in gbs_values:
            current_config = replace(config, gbs=gbs)
            cluster_combinations = get_all_cluster_combinations(
                gpu_cluster=current_config.gpu_cluster,
                pareto=current_config.pareto
            )
            print(f"[PARETO]Cluster combinations: {cluster_combinations}")
            
            for current_cluster in cluster_combinations:
                num_node = sum(current_cluster.values())
                cluster_gpu_types = list(current_cluster.keys())
                available_tps_per_gpu = [set(profile_db.get_available_tp(gpu_type)) for gpu_type in cluster_gpu_types]
                if available_tps_per_gpu and all(tps for tps in available_tps_per_gpu):
                    available_tps = sorted(list(set.intersection(*available_tps_per_gpu)))
                else:
                    available_tps = None

                count = count_parallel_strategies(
                    M=current_config.gpu_per_node,
                    N=num_node,
                    gbs=gbs,
                    num_layers=int(model_config["num_layers"].item()),
                    available_tps=available_tps
                )
                pareto_total_count += count
                pareto_breakdown.append({
                    "gbs": gbs,
                    "cluster": dict(current_cluster),
                    "count": count
                })

        # Log pareto meta information at the start
        jsonl_logger.log_meta({
            "type": "pareto_meta",
            "pareto_total_count": pareto_total_count,
            "gbs_values": gbs_values,
            "model_type": config.model_type,
            "gpu_cluster": dict(config.gpu_cluster),
        })
        print(f"[PARETO] Total combinations across all GBS values: {pareto_total_count}")
        

    # Iterate over GBS values
    for gbs in gbs_values:
        # Create config with current GBS
        current_config = replace(config, gbs=gbs)

        if config.pareto_gbs_max and config.verbose:
            print(f"\n{'='*60}")
            print(f"[PARETO] Searching with GBS={gbs}")
            print(f"{'='*60}")

        # Generate cluster combinations
        cluster_combinations = get_all_cluster_combinations(
            gpu_cluster=current_config.gpu_cluster,
            pareto=current_config.pareto
        )

        for current_cluster in cluster_combinations:
            num_node = sum(current_cluster.values())

            # Get available TPs from NPZ files (intersection across all GPU types in cluster)
            cluster_gpu_types = list(current_cluster.keys())
            available_tps_per_gpu = [set(profile_db.get_available_tp(gpu_type)) for gpu_type in cluster_gpu_types]
            if available_tps_per_gpu and all(tps for tps in available_tps_per_gpu):
                available_tps = sorted(list(set.intersection(*available_tps_per_gpu)))
            else:
                available_tps = None  # Fallback to default behavior

            # Calculate total combinations for verbose/parsing logging
            total_count = 0
            if current_config.verbose or jsonl_logger:
                if available_tps and current_config.verbose:
                    print(f"Available TP values from NPZ: {available_tps}")
                total_count = count_parallel_strategies(
                    M=current_config.gpu_per_node,
                    N=num_node,
                    gbs=current_config.gbs,
                    num_layers=int(model_config["num_layers"].item()),
                    available_tps=available_tps
                )
                if current_config.verbose:
                    print(f"[SEARCH] Total combinations: {total_count}")
                    print(f"[SEARCH] PP Partition Method: {current_config.pp_partition_method}")
                # Log total count to JSONL
                if jsonl_logger:
                    jsonl_logger.log_meta({
                        "type": "meta",
                        "total_count": total_count,
                        "gbs": current_config.gbs,
                        "cluster": dict(current_cluster),
                    })

            # Generate device placements for this cluster
            placement_permutations = device_placement_from_cluster(current_cluster)

            for placement in placement_permutations:
                results = evaluate_single_placement(
                    placement=placement,
                    current_cluster=current_cluster,
                    config=current_config,
                    gpu_config=gpu_config,
                    model_config=model_config,
                    num_iterations=num_iterations,
                    search_idx_start=0,
                    total_count=total_count,
                    available_tps=available_tps,
                    jsonl_logger=jsonl_logger,
                )
                all_results.extend(results)

    return all_results


def calculate_price_per_sec(cluster: Dict[str, int], gpu_config: GPUConfig, pareto: bool) -> float:
    """
    Calculate the price per second for a cluster configuration.
    
    Args:
        cluster: Cluster configuration dict {gpu_type: count}
        gpu_config: GPU configuration object
        pareto: Whether pareto mode is enabled
        
    Returns:
        Price per second in dollars
    """
    price_per_sec = 0.0
    for gpu_type, count in cluster.items():
        instance_type = gpu_config.get_instance_type(gpu_type)
        price_per_sec += get_instance_price_per_sec(instance_type, pareto=pareto) * count
    return price_per_sec


def get_exp_name(config: FasopConfig, model_config: dict) -> str:
    """
    Generate experiment name based on configuration.
    
    Args:
        config: FASOP configuration
        model_config: Model configuration dictionary
        
    Returns:
        Experiment name string
    """
    _, _, exp_name = get_model_config(
        config.model_type,
        config.precision,
        config.heterogeneous,
        config.pareto
    )
    return exp_name + config.add_exp_name


# =============================================================================
# 4. Results Output
# =============================================================================

def results_to_dataframe(results: List[SearchResult]) -> pd.DataFrame:
    """
    Convert search results to a pandas DataFrame.
    
    Args:
        results: List of SearchResult objects
        
    Returns:
        DataFrame with all results
    """
    columns = [
        "index", "gbs", "mbs", "tp", "dp", "pp", "node_type", "gpu_cluster", "partition",
        "step_time(s)", "throughput(samples/s)", "price_per_step($)", "is_oom", "oom_gpumem(GB)",
        "iterations", "total_time(s)", "total_time(h)", "cost($)"
    ]

    data = [
        (
            idx, r.gbs, r.mbs, r.tp, r.dp, r.pp,
            r.node_type, r.gpu_cluster_str, r.partition,
            r.step_time, r.throughput, r.price_per_step, r.is_oom, r.oom_gpumem,
            r.iterations, r.total_time_seconds, r.total_time_hours, r.cost
        )
        for idx, r in enumerate(results)
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Add rank column
    df['rank'] = df['step_time(s)'].rank(method='min', ascending=True)
    df['rank'] = df['rank'].astype(int)
    df = df.sort_values(by=['rank'])
    
    return df


def save_results(df: pd.DataFrame, output_dir: str, exp_name: str) -> str:
    """
    Save results DataFrame to CSV file.
    
    Args:
        df: Results DataFrame
        output_dir: Output directory path
        exp_name: Experiment name
        
    Returns:
        Path to saved CSV file
    """
    output_path = os.path.join(output_dir, f"{exp_name}.csv")
    
    # Remove existing file if present
    if os.path.exists(output_path):
        os.remove(output_path)
    
    df.to_csv(output_path, index=False)
    
    return output_path


def print_results_summary(df: pd.DataFrame, output_path: Optional[str], elapsed_time: float) -> None:
    """
    Print results summary to console.

    Args:
        df: Results DataFrame
        output_path: Path to saved CSV file (None if not saved)
        elapsed_time: Total execution time in seconds
    """
    print(f"Finished in {elapsed_time:.2f} seconds")
    print(f"Total configurations evaluated: {len(df)}")
    if output_path:
        print(f"CSV file saved at: {output_path}")
    
    if len(df) > 0:
        # Filter to non-OOM configurations for best selection
        valid_df = df[df['is_oom'] == False]
        if len(valid_df) > 0:
            best = valid_df.iloc[0]
            print(f"\nBest configuration (non-OOM):")
            print(f"  TP={int(best['tp'])}, DP={int(best['dp'])}, PP={int(best['pp'])}, MBS={int(best['mbs'])}")
            print(f"  Step time: {best['step_time(s)']:.4f}s")
            print(f"  Total training time: {best['total_time(s)']:.2f}s ({best['total_time(h)']:.2f}h)")
            print(f"  Peak memory: {best['oom_gpumem(GB)']:.2f}GB")
            print(f"  Estimated cost: ${best['cost($)']:.2f}")
        else:
            print(f"\nNo valid (non-OOM) configurations found.")


# =============================================================================
# 5. Main Function
# =============================================================================

def main() -> None:
    """
    Main entry point for FASOP.
    """
    time_start = time.time()
    tee_logger = None
    jsonl_logger = None

    # 1. Parse arguments
    args = parse_arguments()

    # Handle --only-make-pareto: generate graph from existing CSV and exit
    if args.only_make_pareto:
        csv_path = args.only_make_pareto
        if not os.path.exists(csv_path):
            raise SystemExit(f"Error: CSV file not found: {csv_path}")

        # Determine output path and title from --pareto-title
        if args.pareto_title:
            csv_dir = os.path.dirname(csv_path) or "."
            output_path = os.path.join(csv_dir, f"{args.pareto_title}.png")
            title = f"{args.pareto_title} pareto frontier analysis"
        else:
            output_path = None  # Will use default (csv filename + .png)
            title = "Pareto frontier analysis"

        print(f"Generating Pareto graph from: {csv_path}")
        graph_path = draw_pareto_graph(csv_path=csv_path, output_path=output_path, title=title)
        if graph_path:
            print(f"Pareto graph saved at: {graph_path}")
            elapsed_time = time.time() - time_start
            print(f"Finished in {elapsed_time:.2f} seconds")
        else:
            print("Failed to generate Pareto graph (no valid data)")
        return

    # 2. Validate and initialize configuration
    config = validate_and_init_config(args)

    # Setup logging if log_file is specified
    if config.log_file:
        tee_logger = TeeLogger(config.log_file)
        sys.stdout = tee_logger
        print(f"[LOG] Output will be saved to: {config.log_file}")

    # Setup JSONL logger for parsing mode
    if config.parsing and config.parsing_file:
        jsonl_logger = JSONLLogger(config.parsing_file)
        print(f"[PARSING] Real-time results will be saved to: {config.parsing_file}")

    try:
        # Setup environment
        home_path, output_dir = setup_environment()

        # Create GPU configuration
        gpu_config = GPUConfig(config.gpu_cluster, pareto=config.pareto)
        print(f"GPUConfig: {gpu_config}")

        # Load model configuration
        model_config, _, exp_name = get_model_config(
            config.model_type,
            config.precision,
            config.heterogeneous,
            config.pareto
        )
        exp_name = exp_name + config.add_exp_name

        # Calculate iterations
        num_iterations = get_iterations(config.dataset, config.gbs)
        dataset_info = get_dataset_info(config.dataset)

        # Print configuration summary
        print_config_summary(config, model_config, num_iterations, dataset_info)

        # 3. Run strategy search
        results = run_strategy_search(config, gpu_config, model_config, num_iterations, jsonl_logger)

        # 4. Output results
        df = results_to_dataframe(results)

        if config.save_csv:
            output_path = save_results(df, output_dir, exp_name)
        else:
            output_path = None
            print("[INFO] CSV save skipped (--no-save-csv)")

        # Generate Pareto graph if requested
        if config.pareto and config.make_pareto and output_path:
            if config.pareto_title:
                graph_output = os.path.join(output_dir, f"{config.pareto_title}.png")
                graph_title = f"{config.pareto_title} pareto frontier analysis"
            else:
                graph_output = None  # Use default (same as CSV with .png)
                graph_title = f"{config.model_type.upper()} training pareto frontier analysis"

            graph_path = draw_pareto_graph(
                csv_path=output_path,
                output_path=graph_output,
                title=graph_title
            )
            if graph_path:
                print(f"Pareto graph saved at: {graph_path}")

        elapsed_time = time.time() - time_start
        print_results_summary(df, output_path, elapsed_time)

    finally:
        # Close JSONL logger
        if jsonl_logger:
            jsonl_logger.close()
            print(f"[PARSING] Results saved to: {config.parsing_file} ({jsonl_logger.count} entries)")

        # Restore stdout and close log file
        if tee_logger:
            sys.stdout = tee_logger.terminal
            tee_logger.close()
            print(f"[LOG] Log saved to: {config.log_file}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
