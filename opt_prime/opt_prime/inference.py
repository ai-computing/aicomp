#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#

"""
Pipeline Parallel Inference Engine for OptimusPrime

This module provides the main inference engine that enables efficient
pipeline parallel inference for large language models. It reuses the
topology, IR transformation, and communication infrastructure from
the training framework while removing training-specific components.
"""

import torch
import torch.nn as nn
import torch.distributed as dist

import logging
import os
import sys
import gc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch.fx.graph_module import GraphModule
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel
)

from opt_prime.comm import Comm
from opt_prime.IR import IR, IR_Anal
from opt_prime.opti_pri import Topology, Run_Info
from opt_prime.schedule_infer import ScheduleInference, ScheduleGeneration
from opt_prime.kv_cache import KVCacheManager, CachedScaledDotProductAttention


logging.basicConfig(level=logging.ERROR)


class Optimus_Inference:
    """
    Pipeline Parallel Inference Engine.

    This class provides efficient inference for large language models using
    pipeline parallelism. It reuses the topology management, IR transformation,
    and communication layers from the training framework while removing
    training-specific components like backward pass, loss computation, and
    optimizer management.

    Args:
        module: PyTorch model to run inference on
        use_gpu: Whether to use GPU (default: True)
        pp_size: Pipeline parallel size (default: auto-calculated from world_size)
        dp_size: Data parallel size (default: 1, not typically used for inference)
        tp_size: Tensor parallel size (default: 1)
        max_batch_size: Maximum batch size for KV cache allocation (default: 32)
        max_seq_len: Maximum sequence length for KV cache allocation (default: 2048)
        dtype: Data type for inference (default: torch.bfloat16)
        ir_analyze: IR analysis mode (default: IR_Anal.PARALLEL)

    Example:
        >>> from opt_prime.inference import Optimus_Inference
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        >>>
        >>> engine = Optimus_Inference(model, use_gpu=True, pp_size=4)
        >>> engine.eval()
        >>>
        >>> if engine.is_first_stage():
        >>>     input_ids = tokenizer("Hello", return_tensors="pt").input_ids.cuda()
        >>> else:
        >>>     input_ids = None
        >>>
        >>> output = engine.generate(input_ids, max_new_tokens=50)
        >>>
        >>> if engine.is_last_stage():
        >>>     print(tokenizer.decode(output[0]))
    """

    def __init__(
        self,
        module: nn.Module,
        use_gpu: bool = True,
        pp_size: int = 1,
        dp_size: int = 1,
        tp_size: int = 1,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        dtype: torch.dtype = torch.bfloat16,
        ir_analyze: IR_Anal = IR_Anal.PARALLEL,
        use_kv_cache: bool = False,
        serving_mode: bool = False,
    ):
        self.use_gpu = use_gpu
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.use_kv_cache = use_kv_cache
        self.serving_mode = serving_mode

        # Model type mapping (reused from training)
        self.model2type = {"hf": 50, "sy": 51, "vt": 52}
        self.model_type = None

        # Initialize communication
        self.comm = Comm(use_gpu=use_gpu, ir_analyze=ir_analyze)

        rank = self.comm.rank
        world_size = self.comm.world_size
        local_rank = self.comm.local_rank

        # Validate parallelism configuration
        assert tp_size >= 1 and world_size % tp_size == 0, \
            f"world size({world_size}) must be divisible by tp size({tp_size})"
        assert dp_size >= 1 and world_size % dp_size == 0, \
            f"world size({world_size}) must be divisible by dp size({dp_size})"

        if pp_size == 1:
            pp_size = world_size // tp_size // dp_size

        assert pp_size >= 1 and world_size == pp_size * dp_size * tp_size, \
            f"world size({world_size}) == pp_size({pp_size}) * dp_size({dp_size}) * tp_size({tp_size})"

        if tp_size > 1:
            assert "llama" in module.__class__.__name__.lower(), \
                f"Tensor parallel (size={tp_size}) is only supported for Llama models."

        if rank == 0:
            print(f"> [Inference] World Size: {world_size}")
            print(f"> [Inference] Pipeline Parallel Size: {pp_size}")
            if dp_size > 1:
                print(f"> [Inference] Data Parallel Size: {dp_size}")
            if tp_size > 1:
                print(f"> [Inference] Tensor Parallel Size: {tp_size}")

        # Initialize topology
        self.tpl = Topology(rank, local_rank, world_size, pp_size, dp_size, tp_size)

        # Set device
        if use_gpu:
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
            print(f">>> [Inference] Using GPU cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
            print(f">>> [Inference] Using CPU")

        # Initialize run info (simplified for inference - no gradients/loss)
        self.run_info = Run_Info(device=self.device, num_mb=1, num_classes=-100)

        # Determine split method
        split_method = "llama-tp-split" if module.__class__.__name__.startswith("Llama") and tp_size > 1 else "simple"
        print(f">> [Inference] model class: {module.__class__.__name__}")
        print(f">> [Inference] split method: {split_method}")

        # Initialize IR and split model
        self.ir = IR(module, self)
        self.model_type = self.ir.retrieve_IR(module, use_kv_cache=use_kv_cache)
        self.ir.split_IR(module, split_method, num_stage=self.tpl.get_num_stage())
        self.ir.setup_submod(self.tpl.stage, rank)
        self.ir.build_getitem_dic()

        # Move submodule to device
        self.run_info.submod.to(self.run_info.device)
        print(f" ### [Inference] Rank:{rank}, name:{self.run_info.node.name}, "
              f"move {self.run_info.name} to {self.run_info.device}")

        self.run_info.output_node = self.ir.get_output_node()

        if rank == 0:
            self.ir.print_graph(rank)
            self.run_info.print_getitem_dic()

        # Cross-reference analysis
        for stage in reversed(range(1, self.tpl.get_num_stage())):
            self.ir.cross_reference_analyze(stage, self.ir.model_ir[0].graph)

        self.run_info.special_nodes = self.ir.special_nodes
        self.run_info.metadata_range = self.ir.metadata_range
        # Note: getitem_dic is already populated by build_getitem_dic()

        print(f" *********** [Inference] rank:{rank} cross-referenced nodes *****************")
        print(f"   special_nodes: {self.run_info.special_nodes}")
        print(f" *************************************************************************")

        # Inject KV cache into FX graph (must happen before clean_module_memory)
        self._cached_sdpa_modules: List = []
        if self.use_kv_cache:
            self._inject_kv_cache_into_graph()

        # Clean up unused modules to save memory
        self.ir.clean_module_memory()
        print(f" ### [Inference] Rank:{rank}, clean_module_memory ...")

        # Apply tensor parallelism if needed
        if tp_size > 1:
            self._prepare_tp_group()

        # Apply data parallelism if needed (uncommon for inference)
        if dp_size > 1:
            self._prepare_dp_group()

        # Initialize KV cache manager (for generation)
        self.kv_cache: Optional[KVCacheManager] = None

        # Initialize schedulers
        self._schedule_infer: Optional[ScheduleInference] = None
        self._schedule_gen: Optional[ScheduleGeneration] = None

    def _prepare_tp_group(self) -> None:
        """Prepare tensor parallel group for LLaMA models."""
        tp_plan = {}
        for name, module in self.run_info.submod.named_modules():
            parts = name.split('_')
            if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                layer_id = int(parts[2])
                if parts[3] == "self" and parts[4] == "attn":
                    if parts[5] == "q" and parts[6] == "proj":
                        tp_plan[f"model_layers_{layer_id}_self_attn_q_proj"] = ColwiseParallel()
                    elif parts[5] == "k" and parts[6] == "proj":
                        tp_plan[f"model_layers_{layer_id}_self_attn_k_proj"] = ColwiseParallel()
                    elif parts[5] == "v" and parts[6] == "proj":
                        tp_plan[f"model_layers_{layer_id}_self_attn_v_proj"] = ColwiseParallel()
                    elif parts[5] == "o" and parts[6] == "proj":
                        tp_plan[f"model_layers_{layer_id}_self_attn_o_proj"] = RowwiseParallel()
                elif parts[3] == "mlp":
                    if parts[4] == "gate" and parts[5] == "proj":
                        tp_plan[f"model_layers_{layer_id}_mlp_gate_proj"] = ColwiseParallel()
                    elif parts[4] == "down" and parts[5] == "proj":
                        tp_plan[f"model_layers_{layer_id}_mlp_down_proj"] = RowwiseParallel()
                    elif parts[4] == "up" and parts[5] == "proj":
                        tp_plan[f"model_layers_{layer_id}_mlp_up_proj"] = ColwiseParallel()

        # Adjust view operations for TP
        for node in self.run_info.submod.graph.nodes:
            if node.op == 'call_method' and node.name.startswith("view"):
                result, layer_id = self._check_tp_post(node.args[0])
                if result in [0, 1, 2]:  # Q, K, V projections
                    new_args = list(node.args)
                    new_args[3] = new_args[3] // self.tpl.tp_mesh.size()
                    node.args = tuple(new_args)

        self.run_info.submod.recompile()
        self.run_info.submod = parallelize_module(
            module=self.run_info.submod,
            device_mesh=self.tpl.tp_mesh,
            parallelize_plan=tp_plan
        )

    def _check_tp_post(self, arg0) -> Tuple[int, int]:
        """Check if a node is a Q/K/V projection for TP adjustment."""
        if isinstance(arg0, torch.fx.Node):
            arg0 = arg0.name
        parts = arg0.split('_')

        if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
            layer_id = int(parts[2])
            if parts[3] == "self" and parts[4] == "attn":
                if parts[5] == "q" and parts[6] == "proj":
                    return 0, layer_id
                elif parts[5] == "k" and parts[6] == "proj":
                    return 1, layer_id
                elif parts[5] == "v" and parts[6] == "proj":
                    return 2, layer_id

        return -1, -1

    def _prepare_dp_group(self) -> None:
        """Prepare data parallel group (uncommon for inference)."""
        self.run_info.submod = DistributedDataParallel(
            self.run_info.submod,
            find_unused_parameters=True,
            device_mesh=self.tpl.dp_mesh
        )

    def eval(self) -> 'Optimus_Inference':
        """
        Set model to evaluation mode.

        Returns:
            self for method chaining
        """
        self.run_info.submod.eval()
        return self

    def train(self, mode: bool = True) -> 'Optimus_Inference':
        """
        Set model training mode (typically False for inference).

        Args:
            mode: Training mode flag

        Returns:
            self for method chaining
        """
        self.run_info.submod.train(mode)
        return self

    def parameters(self):
        """Get model parameters."""
        return self.run_info.submod.parameters()

    def is_first_stage(self) -> bool:
        """Check if this rank is the first pipeline stage."""
        return self.tpl.is_first_stage()

    def is_last_stage(self) -> bool:
        """Check if this rank is the last pipeline stage."""
        return self.tpl.is_last_stage()

    def is_output_rank(self) -> bool:
        """Check if this rank should handle user-facing output.

        Returns True for the first TP rank of the last PP stage.
        With PP=2, TP=2 on ranks [0,1,2,3]: returns True only for rank 2.
        With PP=1, TP=2 on ranks [0,1]: returns True only for rank 0.
        """
        if not self.tpl.is_last_stage():
            return False
        return self.tpl.rank == self.tpl.stage2rank[self.tpl.stage][0]

    def is_input_rank(self) -> bool:
        """Check if this rank should handle user-facing input display.

        Returns True for the first TP rank of the first PP stage.
        """
        if not self.tpl.is_first_stage():
            return False
        return self.tpl.rank == self.tpl.stage2rank[self.tpl.stage][0]

    def get_rank(self) -> int:
        """Get current rank."""
        return self.tpl.rank

    def get_local_rank(self) -> int:
        """Get local rank."""
        return self.tpl.local_rank

    def get_world_size(self) -> int:
        """Get world size."""
        return self.tpl.world_size

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Run a single forward pass through the pipeline.

        Args:
            input_ids: Input token IDs (only needed on first stage)
            attention_mask: Attention mask (optional, only needed on first stage)

        Returns:
            Output logits if this is the last stage, None otherwise
        """
        if self._schedule_infer is None:
            self._schedule_infer = ScheduleInference(self)

        return self._schedule_infer.run(input_ids)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        streamer: Optional[Any] = None,
        verbose: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Generate tokens autoregressively using pipeline parallelism.

        Uses full-sequence recomputation at each decode step to ensure
        correct context is available for attention computation, since the
        FX-traced graph does not support incremental KV cache.

        Communication uses only P2P (send/recv) operations to avoid mixing
        NCCL collectives with P2P, which can cause hangs.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len] (first stage only)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to sample (True) or use greedy decoding (False)
            eos_token_id: End-of-sequence token ID for early stopping
            pad_token_id: Padding token ID
            streamer: Optional streamer for real-time token output

        Returns:
            Generated token IDs if this is the last stage, None otherwise
        """
        # Dispatch to KV cache mode if enabled
        if self.use_kv_cache:
            return self._generate_with_kv_cache(
                input_ids=input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, eos_token_id=eos_token_id,
                pad_token_id=pad_token_id, streamer=streamer, verbose=verbose,
            )

        # Initialize inference scheduler (full forward pass each step)
        if self._schedule_infer is None:
            self._schedule_infer = ScheduleInference(self)

        # Get batch size and initial sequence length from first stage
        # Use P2P communication only (no collectives) for compatibility
        if self.tpl.is_first_stage():
            assert input_ids is not None, "input_ids required on first stage"
            batch_size = input_ids.size(0)
            input_seq_len = input_ids.size(1) if input_ids.dim() > 1 else 1
            input_ids = input_ids.to(self.device)
        else:
            batch_size = 0
            input_seq_len = 0

        # Send dims from first stage to last stage via P2P
        if self.comm.world_size > 1:
            if self.tpl.is_first_stage() and not self.tpl.is_last_stage():
                self.comm.send_data(batch_size, self.tpl.get_last_rank(), self.device)
                self.comm.send_data(input_seq_len, self.tpl.get_last_rank(), self.device)
                # Also send input_ids for output assembly
                self.comm.send_data(input_ids, self.tpl.get_last_rank(), self.device)
            if self.tpl.is_last_stage() and not self.tpl.is_first_stage():
                batch_size = self.comm.receive_data(self.tpl.get_first_rank(), self.device)
                input_seq_len = self.comm.receive_data(self.tpl.get_first_rank(), self.device)
                input_ids = self.comm.receive_data(self.tpl.get_first_rank(), self.device)

        # Initialize output storage on last stage
        if self.tpl.is_last_stage():
            generated_ids = torch.zeros(
                (batch_size, input_seq_len + max_new_tokens),
                dtype=torch.long,
                device=self.device,
            )
            generated_ids[:, :input_seq_len] = input_ids
        else:
            generated_ids = None

        # Track finished sequences for batch EOS handling (last stage only)
        if self.tpl.is_last_stage():
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # First stage maintains the growing sequence for full-sequence forward
        if self.tpl.is_first_stage():
            current_ids = input_ids.clone()  # [batch, current_len]

        # Generation loop: always runs all iterations to avoid pipeline deadlock.
        # Completed sequences are masked with pad_token_id.
        import time as _time
        gen_start_time = _time.time()

        num_generated = 0
        for i in range(max_new_tokens):
            # Full-sequence forward pass through the pipeline
            if self.tpl.is_first_stage():
                full_output = self._schedule_infer.run(current_ids)
            else:
                full_output = self._schedule_infer.run(None)

            # On last stage: extract last-position logits and sample
            if self.tpl.is_last_stage():
                # full_output shape: [batch, seq_len, vocab_size]
                if isinstance(full_output, torch.Tensor) and full_output.dim() >= 2:
                    logits = full_output[:, -1, :]
                else:
                    logits = full_output

                next_token = self._sample_token(
                    logits, temperature, top_k, top_p, do_sample
                )

                # Sync sampled token across TP ranks (sampling may diverge)
                if self.tpl.tp_size > 1:
                    dist.broadcast(next_token, src=self.tpl.stage2rank[self.tpl.stage][0],
                                   group=self.tpl.tp_group)

                # Track EOS and mask finished sequences
                if eos_token_id is not None:
                    finished = finished | (next_token == eos_token_id)
                if pad_token_id is not None:
                    next_token[finished] = pad_token_id

                generated_ids[:, input_seq_len + i] = next_token
                num_generated = i + 1

                if streamer is not None:
                    streamer.put(next_token)

                # Print progress on last stage
                if verbose and (i + 1) % 5 == 0:
                    elapsed = _time.time() - gen_start_time
                    cur_seq_len = input_seq_len + i + 1
                    print(f"  [Token {i+1}/{max_new_tokens}] "
                          f"elapsed={elapsed:.1f}s, "
                          f"seq_len={cur_seq_len}, "
                          f"tok/s={num_generated/elapsed:.2f}",
                          flush=True)
            else:
                next_token = None

            # Send generated token from last stage to first stage (P2P only)
            # Skip when first == last stage (e.g., PP=1 with TP only)
            if self.comm.world_size > 1:
                if self.tpl.is_last_stage() and not self.tpl.is_first_stage():
                    self.comm.send_data(next_token, self.tpl.get_first_rank(), self.device)
                if self.tpl.is_first_stage() and not self.tpl.is_last_stage():
                    next_token = self.comm.receive_data(self.tpl.get_last_rank(), self.device)

            # Append new token to the growing sequence on first stage
            if self.tpl.is_first_stage():
                current_ids = torch.cat(
                    [current_ids, next_token.unsqueeze(1)], dim=1
                )

        # Trim generated_ids to actual length
        if self.tpl.is_last_stage():
            generated_ids = generated_ids[:, :input_seq_len + num_generated]

        # Finalize streamer
        if self.tpl.is_last_stage() and streamer is not None:
            streamer.end()

        return generated_ids

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> torch.Tensor:
        """
        Sample a token from logits using the specified sampling strategy.

        Args:
            logits: Logits tensor of shape [batch_size, vocab_size]
            temperature: Sampling temperature
            top_k: Top-k value
            top_p: Top-p value
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Sampled token IDs of shape [batch_size]
        """
        if not do_sample:
            # Greedy decoding
            return logits.argmax(dim=-1)

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return next_token

    def _inject_kv_cache_into_graph(self) -> None:
        """
        Replace scaled_dot_product_attention call_function nodes with
        CachedScaledDotProductAttention call_module nodes in this stage's submod.

        This performs FX graph surgery: each SDPA call_function is replaced with
        a call_module pointing to a CachedScaledDotProductAttention instance.
        The CachedSDPA module caches K, V tensors across decode steps internally.

        Must be called BEFORE clean_module_memory() so we can access other
        stages' graphs for counting the layer offset.
        """
        import torch.nn.functional as F

        parent_gm = self.ir.model_ir[0]

        # Count SDPA nodes in prior stages to get the layer offset
        layer_offset = 0
        for stage_idx in range(self.tpl.stage):
            submod_name = f"submod_{stage_idx}"
            stage_submod = getattr(parent_gm, submod_name, None)
            if stage_submod is not None:
                for node in stage_submod.graph.nodes:
                    if node.op == 'call_function':
                        target_name = getattr(node.target, '__name__', str(node.target))
                        if 'scaled_dot_product_attention' in target_name:
                            layer_offset += 1

        # Find SDPA nodes in this stage's submod
        submod = self.run_info.submod
        sdpa_nodes = []
        for node in submod.graph.nodes:
            if node.op == 'call_function':
                target_name = getattr(node.target, '__name__', str(node.target))
                if 'scaled_dot_product_attention' in target_name:
                    sdpa_nodes.append(node)

        # Replace each SDPA call_function with a CachedSDPA call_module
        sdpa_count = 0
        for node in sdpa_nodes:
            layer_idx = layer_offset + sdpa_count
            module_name = f"_cached_sdpa_{layer_idx}"

            cached_sdpa = CachedScaledDotProductAttention(
                layer_idx=layer_idx,
                max_seq_len=self.max_seq_len,
                dtype=self.dtype,
            )

            # Register module on the submod and track it
            submod.add_module(module_name, cached_sdpa)
            self._cached_sdpa_modules.append(cached_sdpa)

            # Create call_module node with same args/kwargs, then replace
            with submod.graph.inserting_before(node):
                new_node = submod.graph.call_module(
                    module_name, node.args, node.kwargs
                )

            node.replace_all_uses_with(new_node)
            submod.graph.erase_node(node)
            sdpa_count += 1

        if sdpa_count > 0:
            submod.graph.lint()
            submod.recompile()

        print(f" ### [Inference] Rank:{self.tpl.rank}, injected {sdpa_count} "
              f"CachedSDPA modules (layer_offset={layer_offset})")

    @torch.no_grad()
    def _generate_with_kv_cache(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        streamer: Optional[Any] = None,
        verbose: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Generate tokens using KV cache for O(n) decode performance.

        Prefill phase: forward entire prompt, CachedSDPA stores all K,V.
        Decode phase: forward single token per step, CachedSDPA appends K,V
        and attends to full cached context.

        Args:
            Same as generate()

        Returns:
            Generated token IDs if this is the last stage, None otherwise
        """
        if self._schedule_infer is None:
            self._schedule_infer = ScheduleInference(self)

        # Enable/reset all CachedSDPA modules
        for m in self._cached_sdpa_modules:
            if self.serving_mode and m._enabled:
                m.clear()    # Already allocated — reset position only
            else:
                m.enable()   # First call or batch mode — activate

        # Get batch size and initial sequence length from first stage
        if self.tpl.is_first_stage():
            assert input_ids is not None, "input_ids required on first stage"
            batch_size = input_ids.size(0)
            input_seq_len = input_ids.size(1) if input_ids.dim() > 1 else 1
            input_ids = input_ids.to(self.device)
        else:
            batch_size = 0
            input_seq_len = 0

        # Send dims from first stage to last stage via P2P
        if self.comm.world_size > 1:
            if self.tpl.is_first_stage() and not self.tpl.is_last_stage():
                self.comm.send_data(batch_size, self.tpl.get_last_rank(), self.device)
                self.comm.send_data(input_seq_len, self.tpl.get_last_rank(), self.device)
                self.comm.send_data(input_ids, self.tpl.get_last_rank(), self.device)
            if self.tpl.is_last_stage() and not self.tpl.is_first_stage():
                batch_size = self.comm.receive_data(self.tpl.get_first_rank(), self.device)
                input_seq_len = self.comm.receive_data(self.tpl.get_first_rank(), self.device)
                input_ids = self.comm.receive_data(self.tpl.get_first_rank(), self.device)

        # Initialize output storage on last stage
        if self.tpl.is_last_stage():
            generated_ids = torch.zeros(
                (batch_size, input_seq_len + max_new_tokens),
                dtype=torch.long, device=self.device,
            )
            generated_ids[:, :input_seq_len] = input_ids
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        else:
            generated_ids = None

        import time as _time
        gen_start_time = _time.time()
        num_generated = 0

        # === PREFILL: forward the entire prompt ===
        if self.tpl.is_first_stage():
            position_ids = torch.arange(
                input_seq_len, device=self.device
            ).unsqueeze(0).expand(batch_size, -1)
            full_output = self._schedule_infer.run(input_ids, position_ids=position_ids)
        else:
            full_output = self._schedule_infer.run(None)

        # Sample first token on last stage
        if self.tpl.is_last_stage():
            if isinstance(full_output, torch.Tensor) and full_output.dim() >= 2:
                logits = full_output[:, -1, :]
            else:
                logits = full_output

            next_token = self._sample_token(logits, temperature, top_k, top_p, do_sample)

            # Sync sampled token across TP ranks (sampling may diverge)
            if self.tpl.tp_size > 1:
                dist.broadcast(next_token, src=self.tpl.stage2rank[self.tpl.stage][0],
                               group=self.tpl.tp_group)

            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
            if pad_token_id is not None:
                next_token[finished] = pad_token_id

            generated_ids[:, input_seq_len] = next_token
            num_generated = 1

            if streamer is not None:
                streamer.put(next_token)
        else:
            next_token = None

        # Send first token from last → first stage
        if self.comm.world_size > 1:
            if self.tpl.is_last_stage() and not self.tpl.is_first_stage():
                self.comm.send_data(next_token, self.tpl.get_first_rank(), self.device)
            if self.tpl.is_first_stage() and not self.tpl.is_last_stage():
                next_token = self.comm.receive_data(self.tpl.get_last_rank(), self.device)

        # === DECODE LOOP: one token at a time with KV cache ===
        for i in range(1, max_new_tokens):
            # Forward single token with correct position
            if self.tpl.is_first_stage():
                cur_pos = input_seq_len + i - 1
                position_ids = torch.tensor(
                    [[cur_pos]], device=self.device
                ).expand(batch_size, -1)
                full_output = self._schedule_infer.run(
                    next_token.unsqueeze(1), position_ids=position_ids
                )
            else:
                full_output = self._schedule_infer.run(None)

            # Sample on last stage
            if self.tpl.is_last_stage():
                if isinstance(full_output, torch.Tensor) and full_output.dim() >= 2:
                    logits = full_output[:, -1, :]
                else:
                    logits = full_output

                next_token = self._sample_token(
                    logits, temperature, top_k, top_p, do_sample
                )

                # Sync sampled token across TP ranks (sampling may diverge)
                if self.tpl.tp_size > 1:
                    dist.broadcast(next_token, src=self.tpl.stage2rank[self.tpl.stage][0],
                                   group=self.tpl.tp_group)

                if eos_token_id is not None:
                    finished = finished | (next_token == eos_token_id)
                if pad_token_id is not None:
                    next_token[finished] = pad_token_id

                generated_ids[:, input_seq_len + i] = next_token
                num_generated = i + 1

                if streamer is not None:
                    streamer.put(next_token)

                if verbose and (i + 1) % 5 == 0:
                    elapsed = _time.time() - gen_start_time
                    print(f"  [Token {i+1}/{max_new_tokens}] "
                          f"elapsed={elapsed:.1f}s, "
                          f"tok/s={num_generated/elapsed:.2f}",
                          flush=True)
            else:
                next_token = None

            # Send token from last → first stage
            if self.comm.world_size > 1:
                if self.tpl.is_last_stage() and not self.tpl.is_first_stage():
                    self.comm.send_data(
                        next_token, self.tpl.get_first_rank(), self.device
                    )
                if self.tpl.is_first_stage() and not self.tpl.is_last_stage():
                    next_token = self.comm.receive_data(
                        self.tpl.get_last_rank(), self.device
                    )

        # Serving mode: keep cache allocated for next request (clear position only)
        # Batch mode: free cache memory completely
        for m in self._cached_sdpa_modules:
            if self.serving_mode:
                m.clear()
            else:
                m.disable()

        # Trim generated_ids to actual length
        if self.tpl.is_last_stage():
            generated_ids = generated_ids[:, :input_seq_len + num_generated]

        if self.tpl.is_last_stage() and streamer is not None:
            streamer.end()

        return generated_ids

    def init_kv_cache(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        """
        Initialize KV cache for generation.

        Args:
            num_layers: Total number of transformer layers
            num_heads: Number of attention heads (per TP rank if using TP)
            head_dim: Dimension of each attention head
            batch_size: Batch size (defaults to max_batch_size)
            max_seq_len: Maximum sequence length (defaults to max_seq_len)
        """
        batch_size = batch_size or self.max_batch_size
        max_seq_len = max_seq_len or self.max_seq_len

        # Calculate layer range for this stage
        layers_per_stage = num_layers // self.tpl.pp_size
        layer_start = self.tpl.stage * layers_per_stage
        layer_end = (self.tpl.stage + 1) * layers_per_stage
        if self.tpl.stage == self.tpl.pp_size - 1:
            layer_end = num_layers  # Last stage gets remaining layers

        # Adjust num_heads for tensor parallelism
        if self.tpl.tp_size > 1:
            num_heads = num_heads // self.tpl.tp_size

        self.kv_cache = KVCacheManager(
            num_layers=num_layers,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
            layer_start=layer_start,
            layer_end=layer_end,
        )

        self.kv_cache.allocate(batch_size)
        print(f"[Inference] Rank:{self.tpl.rank} KV cache initialized: {self.kv_cache}")

    def clear_kv_cache(self) -> None:
        """Clear KV cache for new generation."""
        if self.kv_cache is not None:
            self.kv_cache.clear()

    def free_kv_cache(self) -> None:
        """Free KV cache memory."""
        if self.kv_cache is not None:
            self.kv_cache.free()
            self.kv_cache = None

    def release_kv_cache(self) -> None:
        """Explicitly free CachedSDPA KV cache memory.

        Use this in serving mode to release GPU memory when the serving
        session ends. In batch mode this is unnecessary since generate()
        already calls disable() after each request.

        Example (serving mode):
            engine = Optimus_Inference(model, use_kv_cache=True, serving_mode=True)
            for prompt in requests:
                engine.generate(prompt, ...)  # cache kept between calls
            engine.release_kv_cache()         # free when done serving
        """
        for m in self._cached_sdpa_modules:
            m.disable()

    def __repr__(self) -> str:
        return (
            f"Optimus_Inference("
            f"rank={self.tpl.rank}, "
            f"stage={self.tpl.stage}/{self.tpl.pp_size}, "
            f"tp={self.tpl.tp_size}, "
            f"dp={self.tpl.dp_size})"
        )
