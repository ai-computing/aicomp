#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# Usage: torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> 
#                 --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama_70b.py [options]
#
# *** This program was tested with torch 2.5.0 and transformers 4.46.2.
#     The version of transformers used must be consistent across all machines used for testing ***
#
# Mode: Layer Profiling
#   Measures layer-wise execution time for pipeline parallelism optimization
#
import torch
import torch.nn as nn
import torch.distributed as dist
import datetime
import logging
import os
import sys
import math
import time
import json
from collections import defaultdict
from packaging import version

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers

# Import from local optimus_p module
from optimus_p.opti_pri import Optimus_p
from optimus_p.IR import IR_Anal
from optimus_p.utils import ts, log

logging.basicConfig(level=logging.ERROR)


# ============================================================================
# Argument Parser
# ============================================================================
import argparse

parser = argparse.ArgumentParser(description="Llama Layer Profiling with Pipeline Parallelism")

# Model settings
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                    help="Model name: meta-llama/Llama-3.3-70B-Instruct or meta-llama/Llama-3.2-1B")
parser.add_argument("--num_hidden_layers", type=int, default=None, 
                    help="Number of transformer layers (None=use model default, or specify for lightweight)")
parser.add_argument("--use_cache", type=bool, default=False)
parser.add_argument("--llama_access_token", type=str, default=None)

# Batch settings
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--micro_batch_size", type=int, default=1)
parser.add_argument("--pp_size", type=int, default=2)
parser.add_argument("--tp_size", type=int, default=1)
parser.add_argument("--dp_size", type=int, default=1)
parser.add_argument("--run_id", type=str, default="default")

# Profile settings
parser.add_argument("--profile_steps", type=int, default=15, help="Total profiling steps")
parser.add_argument("--profile_warmup_steps", type=int, default=10, help="Warmup steps before measurement")

args, unknown = parser.parse_known_args()


# ============================================================================
# Directories & Result Files
# ============================================================================
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)


# ============================================================================
# Model Configurations (from HuggingFace official configs)
# ============================================================================
# These configs match the official HuggingFace model configurations exactly

MODEL_CONFIGS = {
    # Llama-3.3-70B-Instruct (https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
    "meta-llama/Llama-3.3-70B-Instruct": {
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 8,  # Full model has 80 layers
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "head_dim": 128,  # hidden_size / num_attention_heads = 8192/64
    },
    # Llama-3.2-1B (https://huggingface.co/meta-llama/Llama-3.2-1B)
    "meta-llama/Llama-3.2-1B": {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 16,  # Full model has 16 layers
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "head_dim": 64,  # hidden_size / num_attention_heads = 2048/32
    },
    # Llama-3.2-3B (https://huggingface.co/meta-llama/Llama-3.2-3B)
    "meta-llama/Llama-3.2-3B": {
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "num_hidden_layers": 28,  # Full model has 28 layers
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "head_dim": 128,  # hidden_size / num_attention_heads = 3072/24
    },
}

# Aliases for convenience
MODEL_CONFIGS["70B"] = MODEL_CONFIGS["meta-llama/Llama-3.3-70B-Instruct"]
MODEL_CONFIGS["1B"] = MODEL_CONFIGS["meta-llama/Llama-3.2-1B"]
MODEL_CONFIGS["3B"] = MODEL_CONFIGS["meta-llama/Llama-3.2-3B"]


def get_model_config(model_name: str) -> dict:
    """Get model configuration by name or alias"""
    # Check direct match
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    
    # Check if it's a partial match (e.g., "70B" in "Llama-3.3-70B-Instruct")
    model_name_lower = model_name.lower()
    for key in MODEL_CONFIGS:
        if model_name_lower in key.lower() or key.lower() in model_name_lower:
            return MODEL_CONFIGS[key]
    
    # Default to 70B if not found
    print(f"[WARNING] Unknown model '{model_name}', defaulting to Llama-3.3-70B-Instruct config")
    return MODEL_CONFIGS["meta-llama/Llama-3.3-70B-Instruct"]


# ============================================================================
# Utility Functions
# ============================================================================
def get_total_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def save_exit_code(exit_code: int, run_id: str, elapsed_time: float = None):
    """Save exit code (rank 0 only)"""
    if os.environ.get("RANK", "0") != "0":
        return
    try:
        log_path = f"tmp/exitcode_{run_id}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            if exit_code == 0 and elapsed_time is not None:
                f.write(f"{exit_code},{elapsed_time:.3f}")
            else:
                f.write(str(exit_code))
        log(f"[rank:0] EXIT_CODE {exit_code} saved to {log_path}")
    except Exception as e:
        log(f"[rank:0] Failed to save EXIT_CODE: {e}")


def save_profile_result(result: dict, output_path: str = ""):
    """Save profile results to JSON file"""
    if os.environ.get("RANK", "0") != "0":
        return
    
    if not output_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{RESULT_DIR}/profile_{timestamp}.json"
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        log(f"[rank:0] Profile results saved to {output_path}")
    except Exception as e:
        log(f"[rank:0] Failed to save profile results: {e}")


def _build_lookup_row(profile_result: dict):
    import math
    import re
    import numpy as np

    combined = profile_result.get('combined_timing') or {}
    embed_mean = (combined.get('embedding') or {}).get('mean_ms', math.nan)
    lm_head_mean = (combined.get('lm_head') or {}).get('mean_ms', math.nan)
    modules = combined.get('modules', []) or []

    cols = [
        'embed',
        'attn_q', 'attn_k', 'attn_v', 'attn_o',
        'mlp_gate', 'mlp_act', 'mlp_up', 'mlp_down', 'lm_head'
    ]
    suffix_map = {
        'attn_q': 'self_attn_q_proj',
        'attn_k': 'self_attn_k_proj',
        'attn_v': 'self_attn_v_proj',
        'attn_o': 'self_attn_o_proj',
        'mlp_gate': 'mlp_gate_proj',
        'mlp_act': 'mlp_act_fn',
        'mlp_up': 'mlp_up_proj',
        'mlp_down': 'mlp_down_proj',
        
    }

    suffix_vals = {k: [] for k in suffix_map}
    name_re = re.compile(r'^model_layers_(\d+)_([a-z0-9_]+)$')
    for m in modules:
        name = m.get('name', '')
        mean = m.get('mean_ms', None)
        if mean is None:
            continue
        mm = name_re.match(name)
        if not mm:
            continue
        suffix = mm.group(2)
        for key, suf in suffix_map.items():
            if suffix == suf:
                suffix_vals[key].append(mean)
                break

    row_vals = [embed_mean]
    for key in cols[1:]:
        if key == 'lm_head':
            row_vals.append(float(lm_head_mean) if lm_head_mean is not None else math.nan)
            continue
        vals = suffix_vals.get(key, [])
        row_vals.append(float(np.mean(vals)) if vals else math.nan)

    return np.array([row_vals], dtype=float)


def _get_gpu_type():
    try:
        import torch
        name = torch.cuda.get_device_name(0)
        # Normalize common NVIDIA names (A40/A100/H100/...)
        for tag in ["A40", "A100", "H100", "V100", "L40", "L4", "T4"]:
            if tag in name:
                return tag
        # Fallback: first token without spaces
        return name.split()[0]
    except Exception:
        return "unknown"


def append_profile_npz(profile_result: dict):
    if os.environ.get("RANK", "0") != "0":
        return

    try:
        import numpy as np
        cfg = profile_result.get('config', {})
        model_size_raw = str(cfg.get('model_size', '')).lower()
        model_size = model_size_raw
        import re
        m = re.search(r'(\d+)\s*b', model_size_raw)
        if m:
            model_size = f"{m.group(1)}b"
        tp = cfg.get('tp_size', 'unknown')
        gpu_type = _get_gpu_type()
        if not model_size:
            model_size = "unknown"
        npz_name = f"llama{model_size}_{gpu_type}_{tp}.npz"
        npz_path = os.path.join(RESULT_DIR, npz_name)
        row = _build_lookup_row(profile_result)

        if os.path.exists(npz_path):
            existing = np.load(npz_path)
            data = existing.get('data')
            if data is None:
                keys = list(existing.keys())
                data = existing[keys[0]] if keys else None
            if data is None:
                data = row
            else:
                data = np.vstack([data, row])
        else:
            data = row

        np.savez(npz_path, data=data)
        log(f"[rank:0] Profile lookup saved to {npz_path}")
    except Exception as e:
        log(f"[rank:0] Failed to save profile lookup npz: {e}")


def gather_and_print_combined_profile(block_profiler, warmup_steps: int, world_size: int, rank: int, gloo_group=None):
    """Gather profile results from all ranks and print combined summary."""
    import pickle
    
    # Get local summary
    local_summary = block_profiler.get_summary(warmup_steps)
    
    # Use dist.all_gather_object for simple object gathering (works with gloo)
    all_summaries_list = [None] * world_size
    local_data = {
        'rank': rank,
        'stage': block_profiler.stage,
        'summary': local_summary
    }
    
    # all_gather_object uses gloo backend internally if available
    try:
        dist.all_gather_object(all_summaries_list, local_data, group=gloo_group)
    except Exception as e:
        print(f"[rank:{rank}] gather error: {e}")
        # Fallback: just use local data
        if rank == 0:
            all_summaries_list = [local_data]
        else:
            return None
    
    if rank == 0:
        # Process gathered data
        all_summaries = {}
        for data in all_summaries_list:
            if data is not None:
                all_summaries[data['rank']] = {
                    'stage': data['stage'],
                    'summary': data['summary']
                }
        
        # Combine results
        combined = {
            'embedding': {'mean_ms': 0, 'min_ms': float('inf'), 'max_ms': 0, 'count': 0},
            'modules': [],
            'lm_head': {'mean_ms': 0, 'min_ms': float('inf'), 'max_ms': 0, 'count': 0}
        }
        
        for rank_id, rank_data in all_summaries.items():
            summary = rank_data['summary']
            for key, stats in summary.items():
                if key == 'embedding':
                    combined['embedding']['mean_ms'] = stats['mean_ms']
                    combined['embedding']['median_ms'] = stats.get('median_ms', stats['mean_ms'])
                    combined['embedding']['min_ms'] = stats['min_ms']
                    combined['embedding']['max_ms'] = stats['max_ms']
                    combined['embedding']['count'] = 1
                elif key == 'lm_head':
                    combined['lm_head']['mean_ms'] = stats['mean_ms']
                    combined['lm_head']['median_ms'] = stats.get('median_ms', stats['mean_ms'])
                    combined['lm_head']['min_ms'] = stats['min_ms']
                    combined['lm_head']['max_ms'] = stats['max_ms']
                    combined['lm_head']['count'] = 1
                else:
                    combined['modules'].append({
                        'name': key,
                        'mean_ms': stats['mean_ms'],
                        'median_ms': stats.get('median_ms', stats['mean_ms']),
                        'min_ms': stats['min_ms'],
                        'max_ms': stats['max_ms'],
                        'std_ms': stats.get('std_ms', 0)
                    })

        combined['modules'].sort(key=lambda x: x['name'])

        if combined['modules']:
            module_median_total = sum(m.get('median_ms', m['mean_ms']) for m in combined['modules'])
            module_mean_total = sum(m['mean_ms'] for m in combined['modules'])
        else:
            module_median_total = 0
            module_mean_total = 0
        
        # Use median values for total (more robust to outliers)
        emb_val = combined['embedding'].get('median_ms', combined['embedding']['mean_ms'])
        lm_val = combined['lm_head'].get('median_ms', combined['lm_head']['mean_ms'])
        total_time = emb_val + module_median_total + lm_val
        
        # Print combined summary
        print(f"\n{'='*90}")
        print(f" COMBINED PROFILE (All Stages/Ranks)")
        print(f" PP={world_size} stages, {len(combined['modules'])} modules")
        print(f" Values: per-forward-pass average")
        print(f"{'='*90}")
        
        print(f"\n[Combined Summary - Per Forward Pass (Median values for stability)]")
        print(f"{'Component':<25} {'Median(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'%':<8}")
        print(f"{'-'*75}")
        
        # Embedding
        if combined['embedding']['count'] > 0:
            emb = combined['embedding']
            median_val = emb.get('median_ms', emb['mean_ms'])
            pct = (median_val / total_time * 100) if total_time > 0 else 0
            print(f"{'Embedding':<25} {median_val:<12.4f} {emb['min_ms']:<12.4f} {emb['max_ms']:<12.4f} {pct:<8.2f}")
        
        # Modules (total)
        if combined['modules']:
            pct = (module_median_total / total_time * 100) if total_time > 0 else 0
            print(f"{'Modules (total)':<25} {module_median_total:<12.4f} {'-':<12} {'-':<12} {pct:<8.2f}")
        
        # LM Head
        if combined['lm_head']['count'] > 0:
            lm = combined['lm_head']
            median_val = lm.get('median_ms', lm['mean_ms'])
            pct = (median_val / total_time * 100) if total_time > 0 else 0
            print(f"{'LM Head':<25} {median_val:<12.4f} {lm['min_ms']:<12.4f} {lm['max_ms']:<12.4f} {pct:<8.2f}")
        
        print(f"{'-'*75}")
        print(f"{'TOTAL':<25} {total_time:<12.4f}")
        
        # Per-module breakdown
        if combined['modules']:
            print(f"\n[Per-Module Breakdown]")
            print(f"{'Module':<45} {'Median(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'Mean(ms)':<12} {'%':<8}")
            print(f"{'-'*95}")
            for module in combined['modules']:
                median_val = module.get('median_ms', module['mean_ms'])
                pct = (median_val / total_time * 100) if total_time > 0 else 0
                print(f"{module['name'][:45]:<45} {median_val:<12.4f} {module['min_ms']:<12.4f} {module['max_ms']:<12.4f} {module['mean_ms']:<12.4f} {pct:<8.2f}")

        print(f"\n{'='*90}\n")
        
        return combined
    
    return None


# ============================================================================
# FX Interpreter-based Layer Profiler
# ============================================================================
# NOTE: FX 노드 레벨 프로파일링이 필요한 경우 opt_prime/IR.py의
#       LayerProfileInterpreter를 사용하세요.
#       - regex 기반 안정적인 레이어 경계 탐지
#       - get_node_range_time(): 특정 노드 범위 시간 측정
#       - get_layer_node_breakdown(): 레이어별 노드 분해
#
# Usage:
#   from opt_prime.IR import LayerProfileInterpreter
#   profiler = LayerProfileInterpreter(submod, use_cuda=True)
#   output = profiler.run(input_tensor)
#   profiler.print_layer_profile()
# ============================================================================


# ============================================================================
# Module Profiler (module end boundary-based, low overhead)
# ============================================================================
class ModuleProfiler:
    """
    Measures module boundary time using end-to-end timestamps.
    
    FX Graph에서 pow/add는 call_method/call_function이므로 직접 hook 불가.
    대신 module end hook을 경계로 사용하여 시간을 측정:
    
    Embedding:
      START: stage_start
      END:   embed_tokens end
    
    Module N:
      START: module_(N-1) end hook
      END:   module_N end hook
      (first module uses embed_tokens post hook if available, else stage start)
      포함: 이전 module 종료 이후 ~ 현재 module 종료 직전의 모든 연산 (call_function/method 포함)
    
    Embedding:
      START: stage_start
      END:   embed_tokens end

    LM Head:
      START: mlp_down_end if exists, else stage_start
      END:   lm_head end
    
    NOTE: Times are accumulated per-step.
          With num_mb micro-batches per step, this measures total time across all micro-batches.
    """
    
    def __init__(self, submod, device, rank, stage, num_mb: int = 1):
        self.submod = submod
        self.device = device
        self.rank = rank
        self.stage = stage
        self.num_mb = num_mb  # micro-batches per step
        
        # Per-step accumulated times (list of step totals)
        self.step_times = defaultdict(list)  # component -> list of step totals
        
        # Current step accumulator (sum of all micro-batches in current step)
        self.current_step_times = defaultdict(float)
        
        # Timing state for boundary-based measurement
        self.embedding_start = None
        self.embedding_end_event = None
        self.stage_start_event = None
        self.module_prev_end_event = None
        self.last_mlp_down_end_event = None
        self.lm_head_start_event = None   # LM head start
        
        # Track module execution order (filled at runtime)
        self.module_seen = set()
        self.module_order = []
        
        # Deferred event pairs (will be processed at step_end)
        self.pending_events = []  # [(key, start_event, end_event), ...]
        
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for boundary-based module timing"""

        # Check if lm_head exists in this stage
        self.has_lm_head = False
        for name, module in self.submod.named_modules():
            if 'lm_head' in name.lower():
                self.has_lm_head = True
                break

        # Stage start boundary (for first module when embed_tokens is absent)
        self._register_stage_start_hook()

        # Second pass: register hooks (module end boundaries)
        for name, module in self.submod.named_modules():
            if name == "":
                continue
            if len(list(module.children())) > 0:
                continue

            name_lower = name.lower()

            # Embedding: record start/end for embedding time
            if 'embed_tokens' in name_lower or name_lower == 'model_embed_tokens':
                self._register_embedding_hooks(name, module)
                continue

            # LM Head: records lm_head timing only
            if 'lm_head' in name_lower:
                self._register_lm_head_hooks(name, module)
                continue

            # All other leaf modules: module end boundary
            self._register_module_end_hook(name, module)

        if self.rank == 0:
            print(f"[ModuleProfiler] MODULE PROFILING (module end boundary-based)")
            print(f"[ModuleProfiler] Embedding: stage_start → embed end")
            print(f"[ModuleProfiler] Module N: module_(N-1) end → module_N end")
            print(f"[ModuleProfiler] First Module: embed end → module end (else stage start)")
            print(f"[ModuleProfiler] LM Head: mlp_down end → lm_head end (else stage start)")
            print(f"[ModuleProfiler] has_lm_head={self.has_lm_head}")
            print(f"[ModuleProfiler] num_mb={self.num_mb} (times will be per-forward-pass average)")

    def _register_stage_start_hook(self):
        """Register pre_hook for stage start boundary"""
        profiler = self

        def pre_hook(mod, inp):
            profiler.stage_start_event = torch.cuda.Event(enable_timing=True)
            profiler.stage_start_event.record()
            profiler.module_prev_end_event = None
            profiler.embedding_end_event = None
            profiler.last_mlp_down_end_event = None

        self.hooks.append(self.submod.register_forward_pre_hook(pre_hook))

    def _register_embedding_hooks(self, name, module):
        """Register pre/post hooks for embedding"""
        profiler = self
        
        def pre_hook(mod, inp):
            profiler.embedding_start = torch.cuda.Event(enable_timing=True)
            profiler.embedding_start.record()
        
        def post_hook(mod, inp, out):
            if profiler.embedding_start:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                if profiler.stage_start_event is not None:
                    profiler.pending_events.append(('embedding', profiler.stage_start_event, end_event))
                profiler.embedding_end_event = end_event
                profiler.module_prev_end_event = end_event
                profiler.embedding_start = None
        
        self.hooks.append(module.register_forward_pre_hook(pre_hook))
        self.hooks.append(module.register_forward_hook(post_hook))
        if self.rank == 0:
            print(f"[ModuleProfiler] Embedding hooks: {name}")
    
    def _register_lm_head_hooks(self, name, module):
        """Register pre/post hooks for lm_head timing"""
        profiler = self
        
        def pre_hook(mod, inp):
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            profiler.lm_head_start_event = start_event
        
        def post_hook(mod, inp, out):
            if profiler.lm_head_start_event:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                start_event = profiler.last_mlp_down_end_event or profiler.stage_start_event
                if start_event is not None:
                    profiler.pending_events.append(('lm_head', start_event, end_event))
                profiler.lm_head_start_event = None
                profiler.module_prev_end_event = end_event
        
        self.hooks.append(module.register_forward_pre_hook(pre_hook))
        self.hooks.append(module.register_forward_hook(post_hook))
        if self.rank == 0:
            print(f"[ModuleProfiler] LM Head hooks: {name}")

    def _register_module_end_hook(self, name, module):
        """Register post_hook for module end boundary timing"""
        profiler = self
        name_lower = name.lower()
        is_mlp_down = 'mlp_down_proj' in name_lower or 'mlp.down_proj' in name_lower

        def post_hook(mod, inp, out):
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()

            if name not in profiler.module_seen:
                profiler.module_seen.add(name)
                profiler.module_order.append(name)

            start_event = profiler.module_prev_end_event
            if start_event is None:
                start_event = profiler.embedding_end_event or profiler.stage_start_event

            if start_event is not None:
                profiler.pending_events.append((name, start_event, end_event))

            profiler.module_prev_end_event = end_event
            if is_mlp_down:
                profiler.last_mlp_down_end_event = end_event

        self.hooks.append(module.register_forward_hook(post_hook))
        if self.rank == 0:
            print(f"[ModuleProfiler] Module end hook: {name}")

    def step_end(self):
        """Called at end of each training step to finalize measurements"""
        torch.cuda.synchronize()
        
        for key, start_event, end_event in self.pending_events:
            elapsed = start_event.elapsed_time(end_event)
            self.current_step_times[key] += elapsed
        self.pending_events.clear()
        
        self.embedding_end_event = None
        self.stage_start_event = None
        self.lm_head_start_event = None
        self.module_prev_end_event = None
        self.last_mlp_down_end_event = None
        
        for key, value in self.current_step_times.items():
            self.step_times[key].append(value)
        self.current_step_times.clear()
    
    def get_summary(self, warmup_steps: int = 0) -> dict:
        """Get timing summary (excluding warmup, per-forward-pass averages)"""
        summary = {}
        
        for key, times in self.step_times.items():
            measured = times[warmup_steps:] if len(times) > warmup_steps else times
            if measured:
                per_fwd_times = [t / self.num_mb for t in measured]
                sorted_times = sorted(per_fwd_times)
                mean_val = sum(per_fwd_times) / len(per_fwd_times)
                median_val = sorted_times[len(sorted_times) // 2]
                summary[key] = {
                    'mean_ms': mean_val,
                    'median_ms': median_val,
                    'min_ms': min(per_fwd_times),
                    'max_ms': max(per_fwd_times),
                    'std_ms': (sum((t - mean_val)**2 for t in per_fwd_times) / len(per_fwd_times)) ** 0.5 if len(per_fwd_times) > 1 else 0,
                    'measured_steps': len(measured)
                }
        
        return summary
    
    def print_summary(self, warmup_steps: int = 0):
        """Print module timing summary"""
        if self.rank != 0:
            return
        
        summary = self.get_summary(warmup_steps)
        total_time = sum(v.get('median_ms', v['mean_ms']) for v in summary.values())
        modules = {k: v for k, v in summary.items() if k not in ['embedding', 'lm_head']}
        module_total = sum(v['mean_ms'] for v in modules.values())
        measured_steps = list(summary.values())[0].get('measured_steps', 0) if summary else 0
        
        print(f"\n{'='*90}")
        print(f" Module Profile (Stage {self.stage}, Rank {self.rank})")
        print(f" Measures: module_(N-1) end → module_N end (first uses embed end or stage start)")
        print(f" Includes: prev module 이후 ~ current module 종료 직전 연산 (call_function/method 포함)")
        print(f" Warmup: {warmup_steps} steps, Measured: {measured_steps} steps")
        print(f" Values: per-forward-pass average (num_mb={self.num_mb})")
        print(f"{'='*90}")
        
        print(f"\n[Summary - Per Forward Pass]")
        print(f"{'Component':<20} {'Median(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'Mean(ms)':<12} {'%':<8}")
        print(f"{'-'*80}")
        
        if 'embedding' in summary:
            s = summary['embedding']
            pct = (s['median_ms'] / total_time * 100) if total_time > 0 else 0
            print(f"{'Embedding':<20} {s['median_ms']:<12.4f} {s['min_ms']:<12.4f} {s['max_ms']:<12.4f} {s['mean_ms']:<12.4f} {pct:<8.2f}")
        
        if modules:
            module_median_total = sum(v.get('median_ms', v['mean_ms']) for v in modules.values())
            module_pct = (module_median_total / total_time * 100) if total_time > 0 else 0
            print(f"{'Modules (total)':<20} {module_median_total:<12.4f} {'-':<12} {'-':<12} {module_total:<12.4f} {module_pct:<8.2f}")
        
        if 'lm_head' in summary:
            s = summary['lm_head']
            pct = (s['median_ms'] / total_time * 100) if total_time > 0 else 0
            print(f"{'LM Head':<20} {s['median_ms']:<12.4f} {s['min_ms']:<12.4f} {s['max_ms']:<12.4f} {s['mean_ms']:<12.4f} {pct:<8.2f}")
        
        print(f"{'-'*80}")
        print(f"{'TOTAL':<20} {total_time:<12.4f}")
        
        if modules:
            print(f"\n[Per-Module Breakdown (prev_module_end → module_end)]")
            print(f"{'Module':<45} {'Mean(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'Std(ms)':<12} {'%':<8}")
            print(f"{'-'*90}")

            ordered_keys = [key for key in self.module_order if key in modules]
            remaining_keys = [key for key in modules.keys() if key not in self.module_order]
            ordered_keys.extend(sorted(remaining_keys))

            for module_key in ordered_keys:
                stats = modules[module_key]
                pct = (stats['mean_ms'] / total_time * 100) if total_time > 0 else 0
                print(f"{module_key[:45]:<45} {stats['mean_ms']:<12.4f} {stats['min_ms']:<12.4f} {stats['max_ms']:<12.4f} {stats['std_ms']:<12.4f} {pct:<8.2f}")

        print(f"\n{'='*90}\n")
    
    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


# ============================================================================
# Main Execution
# ============================================================================
EXIT_CODE = 0
ELAPSED_TIME = None

# Initialize distributed
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
master_addr = os.getenv("MASTER_ADDR")
master_port = os.getenv("MASTER_PORT")

timeout = datetime.timedelta(hours=1)
init_method = f"tcp://{master_addr}:{master_port}"

log(f" rank:{rank}, world_size:{world_size}, init_method:{init_method}")

dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=init_method, timeout=timeout)
group_gloo = dist.new_group(backend="gloo", timeout=timeout)
store = dist.distributed_c10d._get_default_store()
if store is not None:
    store.set_timeout(timeout)


try:
    # Version checks
    required_torch = "2.3.1"
    required_tf = "4.46.2"
    
    if version.parse(torch.__version__) < version.parse(required_torch):
        raise ValueError(f'Requires torch >= {required_torch}, got {torch.__version__}')
    if version.parse(transformers.__version__) < version.parse(required_tf):
        raise ValueError(f'Requires transformers >= {required_tf}, got {transformers.__version__}')
    
    log(f"[rank:{rank}] torch={torch.__version__}, transformers={transformers.__version__} ✓")

    # ========================================================================
    # Tokenizer Setup
    # ========================================================================
    # All Llama 3.x models share the same tokenizer (vocab_size=128256)
    # So we can use 70B tokenizer for 1B/3B models as well
    use_cache = args.use_cache
    access_token = args.llama_access_token or os.environ.get('LLAMA_ACCESS_TOKEN') or os.environ.get('HF_TOKEN')

    # Tokenizer cache paths to try (in order of preference)
    TOKENIZER_CACHE_PATHS = [
        # 70B tokenizer (most likely to be cached)
        "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/",
        # 1B tokenizer
        "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/",
        # 3B tokenizer
        "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/",
    ]

    tokenizer = None
    tokenizer_source = None

    if use_cache:
        # Try each cache path
        for cache_path in TOKENIZER_CACHE_PATHS:
            if os.path.exists(cache_path):
                try:
                    snapshot_id = os.listdir(cache_path)
                    if snapshot_id:
                        full_path = os.path.join(cache_path, snapshot_id[0])
                        tokenizer = AutoTokenizer.from_pretrained(full_path, local_files_only=True)
                        tokenizer_source = full_path
                        break
                except Exception as e:
                    if rank == 0:
                        log(f'> Failed to load tokenizer from {cache_path}: {e}')
                    continue

        if tokenizer is None:
            # No cache found, try to download (only for 70B or if token provided)
            if access_token is not None:
                if rank == 0:
                    log(f'> No tokenizer cache found, downloading from HuggingFace...')
                os.environ.pop('HF_HUB_OFFLINE', None)
                os.environ.pop('HF_DATASETS_OFFLINE', None)
                tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=access_token)
                tokenizer_source = f"HuggingFace ({args.model_name})"
            else:
                raise ValueError("No tokenizer cache found and LLAMA_ACCESS_TOKEN not provided")
    else:
        # use_cache=False: download from HuggingFace
        if access_token is None:
            raise ValueError("LLAMA_ACCESS_TOKEN required when use_cache=False")
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('HF_DATASETS_OFFLINE', None)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=access_token)
        tokenizer_source = f"HuggingFace ({args.model_name})"

    if rank == 0:
        log(f'> Tokenizer loaded from: {tokenizer_source}')
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # ========================================================================
    # Model Setup (Dynamic config based on model_name)
    # ========================================================================
    batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size
    num_mb = batch_size // (micro_batch_size * args.dp_size)
    
    # Get model configuration based on model_name
    model_cfg = get_model_config(args.model_name)
    
    # Determine number of layers (use args if specified, else model default)
    if args.num_hidden_layers is not None:
        num_layers = args.num_hidden_layers
    else:
        num_layers = model_cfg["num_hidden_layers"]
    
    # Determine model size string for logging
    if "70B" in args.model_name or "70b" in args.model_name:
        model_size_str = "70B"
    elif "3B" in args.model_name or "3b" in args.model_name:
        model_size_str = "3B"
    elif "1B" in args.model_name or "1b" in args.model_name:
        model_size_str = "1B"
    else:
        model_size_str = "Unknown"
    
    log('===> Model loading...') if local_rank == 0 else None
    
    if local_rank == 0:
        log(f'> Model: {args.model_name} ({model_size_str} architecture)')
        log(f'> Config: hidden_size={model_cfg["hidden_size"]}, '
            f'intermediate_size={model_cfg["intermediate_size"]}, '
            f'num_attention_heads={model_cfg["num_attention_heads"]}, '
            f'num_key_value_heads={model_cfg["num_key_value_heads"]}')
        log(f'> Using {num_layers} layers (full model: {model_cfg["num_hidden_layers"]} layers)')

    for i in range(local_world_size):
        if local_rank == i:            
            # Llama 3.x vocab_size is 128256
            # Using tokenizer.vocab_size can cause mismatch with actual token IDs
            actual_vocab_size = model_cfg["vocab_size"]
            tokenizer_vocab_size = getattr(tokenizer, 'vocab_size', actual_vocab_size)
            if tokenizer_vocab_size > actual_vocab_size:
                actual_vocab_size = tokenizer_vocab_size
            
            if local_rank == 0:
                log(f'> Tokenizer vocab_size: {tokenizer_vocab_size}, Using: {actual_vocab_size}')
            
            # Create LlamaConfig from model configuration
            config = LlamaConfig(
                vocab_size=actual_vocab_size,
                hidden_size=model_cfg["hidden_size"],
                intermediate_size=model_cfg["intermediate_size"],
                num_hidden_layers=num_layers,
                num_attention_heads=model_cfg["num_attention_heads"],
                num_key_value_heads=model_cfg["num_key_value_heads"],
                max_position_embeddings=model_cfg["max_position_embeddings"],
                rms_norm_eps=model_cfg["rms_norm_eps"],
                rope_theta=model_cfg["rope_theta"],
                use_cache=False,
                tie_word_embeddings=False,
            )
            model = LlamaForCausalLM(config)

            if local_rank == 0:
                log(f'> [PROFILE MODE] {num_layers}-layer Llama {model_size_str} model')
                log(f'> Total parameters: {get_total_params(model):,}')
                log(f'> PP={args.pp_size}, TP={args.tp_size}, DP={args.dp_size}')
                log(f'> GBS={batch_size}, MBS={micro_batch_size}, #MB={num_mb}')

            optimus_p = Optimus_p(
                model, num_mb,
                use_gpu=True,
                pp_size=args.pp_size, tp_size=args.tp_size, dp_size=args.dp_size,
                activation_ckpt=False, force_free_mem=True, display_mem=True,
                swap_opt_in_fwdbwd=False, swap_model_in_optstep=False,
                ir_analyze=IR_Anal.PARALLEL, pre_barrier=group_gloo
            )
            log(f"[rank:{optimus_p.get_rank()}] Optimus_p initialized")

        if local_rank > i:
            log(f"[local_rank:{local_rank}] Waiting for rank {i}...")
            dist.barrier(group=group_gloo)
            log(f"[local_rank:{local_rank}] Rank {i} finished")
    
    log('===> Model loading completed') if local_rank == 0 else None
    optimus_p.train()

    # ========================================================================
    # PROFILE MODE
    # ========================================================================
    log(f"[{ts()}] ========== PROFILE MODE ==========") if rank == 0 else None

    # ================================================================
# ModuleProfiler: module end boundary-based timing
# Measures:
#   - Embedding: stage_start → embed_tokens end
#   - Module N: module_(N-1) end → module_N end
#   - LM Head: mlp_down end → lm_head end (else stage_start)
# ================================================================
    module_profiler = ModuleProfiler(
        optimus_p.run_info.submod,
        optimus_p.run_info.device,
        rank,
        optimus_p.tpl.stage,
        num_mb=num_mb
    )

    # Use optimizer for realistic timing
    if args.tp_size > 1:
        optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5, foreach=False)
    else:
        optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5)

    # Load dataset
    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(record) for record in datasets if len(str(record)) < 500]
    dataloader = optimus_p.prepare_dataloader(datasets, batch_size)

    log(f"[rank:{rank}] Profile: {args.profile_steps} steps ({args.profile_warmup_steps} warmup)")

    # Profile loop
    tick = time.time()
    step_count = 0

    for batch in dataloader:
        if step_count >= args.profile_steps:
            break

        data, labels = None, None
        if optimus_p.is_first_stage():
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            data, labels = tokens.input_ids, tokens.input_ids

        labels = optimus_p.move_labels2last_stage(labels)
        optimus_p.optimizer.zero_grad()

        # Forward + Backward
        optimus_p.run(data, labels, mode="1f1b")

        if args.tp_size == 1:
            torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)

        optimus_p.optimizer.step()
        module_profiler.step_end()

        step_count += 1
        if rank == 0 and step_count % 5 == 0:
            log(f"[rank:0] Profile step {step_count}/{args.profile_steps}")

    tock = time.time()

    # Synchronize all ranks before printing
    dist.barrier()

    # Print per-rank results
    module_profiler.print_summary(warmup_steps=args.profile_warmup_steps)

    # Synchronize again
    dist.barrier()

    # Gather and print combined results from all ranks
    combined_result = gather_and_print_combined_profile(
        module_profiler,
        args.profile_warmup_steps,
        world_size,
        rank,
        gloo_group=group_gloo
    )

    # Save results (rank 0 only)
    if rank == 0:
        profile_result = {
            'config': {
                'model_name': args.model_name,
                'model_size': model_size_str,
                'hidden_size': model_cfg["hidden_size"],
                'intermediate_size': model_cfg["intermediate_size"],
                'num_attention_heads': model_cfg["num_attention_heads"],
                'num_key_value_heads': model_cfg["num_key_value_heads"],
                'num_layers': num_layers,
                'num_layers_full': model_cfg["num_hidden_layers"],
                'batch_size': batch_size,
                'micro_batch_size': micro_batch_size,
                'pp_size': args.pp_size,
                'tp_size': args.tp_size,
                'dp_size': args.dp_size,
                'profile_steps': args.profile_steps,
                'warmup_steps': args.profile_warmup_steps,
                'total_time_sec': tock - tick,
            },
            'combined_timing': combined_result,
            'per_rank_timing': {
                f'rank_{rank}': module_profiler.get_summary(args.profile_warmup_steps)
            }
        }

        output_path = f"{RESULT_DIR}/profile_{args.run_id}.json"
        save_profile_result(profile_result, output_path)
        append_profile_npz(profile_result)
        ELAPSED_TIME = tock - tick

    module_profiler.remove_hooks()
    EXIT_CODE = 0

except torch.cuda.OutOfMemoryError as e:
    log(f" ERROR: OOM - {e}")
    EXIT_CODE = 10

except dist.DistBackendError as e:
    log(f" ERROR: Distributed communication failed - {e}")
    EXIT_CODE = 20

except Exception as e:
    log(f" ERROR: {e}")
    import traceback
    traceback.print_exc()
    EXIT_CODE = 30

finally:
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            print(f"[rank:{os.environ.get('RANK','?')}] process group destroyed.")
    except Exception as e:
        print(f"[rank:{os.environ.get('RANK','?')}] cleanup failed: {e}")
        if EXIT_CODE == 0:
            EXIT_CODE = 40

print(f">>> EXIT_CODE: {EXIT_CODE}, ELAPSED_TIME: {ELAPSED_TIME}")
save_exit_code(EXIT_CODE, args.run_id, ELAPSED_TIME)
sys.exit(EXIT_CODE)
