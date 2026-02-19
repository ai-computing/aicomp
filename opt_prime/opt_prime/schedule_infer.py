#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#

"""
Inference Schedulers for Pipeline Parallel Inference

This module provides inference-specific schedulers that handle forward-only
execution without backward passes, gradients, or loss computation.
"""

import torch
import torch.nn as nn
from torch import fx
from torch.nn.parallel import DistributedDataParallel

import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

from opt_prime.kv_cache import KVCacheManager


logging.basicConfig(level=logging.ERROR)


class ScheduleInference:
    """
    Inference scheduler for single forward pass (classification, encoding, etc.)

    This scheduler executes forward pass only without gradient tracking,
    backward passes, or loss computation. It reuses the communication
    infrastructure from training schedulers.
    """

    def __init__(self, optimus_infer):
        """
        Initialize inference scheduler.

        Args:
            optimus_infer: Optimus_Inference instance containing model and topology
        """
        self.optimus = optimus_infer
        self.placeholder_name = self._get_placeholder_name()

    def _get_placeholder_name(self) -> str:
        """Get the placeholder name based on model type."""
        if self.optimus.model_type == self.optimus.model2type["hf"]:
            return "input_ids"
        elif self.optimus.model_type == self.optimus.model2type["vt"]:
            return "pixel_values"
        elif self.optimus.model_type == self.optimus.model2type["sy"]:
            return "x"
        return "input_ids"

    def init_env(self) -> None:
        """Initialize environment for a single forward pass."""
        self.optimus.run_info.env[0] = {}
        self.optimus.run_info.env_recv_mark[0] = {}
        self.optimus.run_info.env_send_mark[0] = {}

        self.optimus.run_info.env_recv_mark[0][self.placeholder_name] = None
        self.optimus.run_info.env_send_mark[0][self.placeholder_name] = None

        for i in range(len(self.optimus.run_info.metadata_range)):
            node_name = self.optimus.run_info.metadata_range[i][1]
            self.optimus.run_info.env_recv_mark[0][node_name] = None
            self.optimus.run_info.env_send_mark[0][node_name] = None

        # Initialize marks for special_nodes (e.g., position_ids forwarded cross-stage)
        for node_name in self.optimus.run_info.special_nodes:
            if node_name in self.optimus.run_info.getitem_dic:
                submod_name = self.optimus.run_info.getitem_dic[node_name][0]
                if submod_name not in self.optimus.run_info.env_recv_mark[0]:
                    self.optimus.run_info.env_recv_mark[0][submod_name] = None
                    self.optimus.run_info.env_send_mark[0][submod_name] = None
            else:
                if node_name not in self.optimus.run_info.env_recv_mark[0]:
                    self.optimus.run_info.env_recv_mark[0][node_name] = None
                    self.optimus.run_info.env_send_mark[0][node_name] = None

    def get_input(self, data: torch.Tensor) -> None:
        """
        Prepare input data for the first stage.

        Args:
            data: Input tensor (e.g., input_ids for language models)
        """
        if self.optimus.tpl.is_first_stage():
            if isinstance(data, torch.Tensor):
                data = data.to(self.optimus.run_info.device)
                self.optimus.run_info.env[0]["placeholder"] = data
            else:
                logging.critical(f"Input {data} is not a Tensor - not supported")
                sys.exit(1)

    def pre_forward(self) -> None:
        """Prepare inputs by receiving from previous stage if needed."""
        if self.optimus.tpl.is_first_stage():
            # Move placeholder to the correct input name
            self.optimus.run_info.env[0][self.placeholder_name] = \
                self.optimus.run_info.env[0]["placeholder"]

        if self.optimus.tpl.get_stage() > self.optimus.tpl.get_first_stage():
            prev_rank = self.optimus.tpl.get_prev_rank()

            for node_name, range_ in self.optimus.run_info.special_nodes.items():
                src_stage, needed_by_stage = range_
                if (self.optimus.tpl.stage > src_stage and
                        self.optimus.tpl.stage <= needed_by_stage):

                    if node_name in self.optimus.run_info.getitem_dic:
                        submod_name = self.optimus.run_info.getitem_dic[node_name][0]
                        if self.optimus.run_info.env_recv_mark[0][submod_name] is None:
                            self.optimus.run_info.env[0][submod_name] = \
                                self.optimus.comm.receive_data(
                                    prev_rank, self.optimus.run_info.device
                                )
                            self.optimus.run_info.env_recv_mark[0][submod_name] = 1
                    else:
                        if self.optimus.run_info.env_recv_mark[0][node_name] is None:
                            self.optimus.run_info.env[0][node_name] = \
                                self.optimus.comm.receive_data(
                                    prev_rank, self.optimus.run_info.device
                                )
                            self.optimus.run_info.env_recv_mark[0][node_name] = 1

    def forward_core(self) -> Any:
        """
        Execute forward pass for this stage's submodule.

        Returns:
            Forward pass output
        """
        def extract_tensor_args(b):
            if b.name in self.optimus.run_info.getitem_dic:
                submod_name = self.optimus.run_info.getitem_dic[b.name][0]
                idx = self.optimus.run_info.getitem_dic[b.name][1]
                return self.optimus.run_info.env[0][submod_name][idx]
            else:
                return self.optimus.run_info.env[0][b.name]

        args = fx.graph.map_arg(self.optimus.run_info.node.args, extract_tensor_args)
        kwargs = fx.graph.map_arg(self.optimus.run_info.node.kwargs, extract_tensor_args)

        # Handle DDP wrapped modules
        if isinstance(self.optimus.run_info.submod, DistributedDataParallel):
            result = self.optimus.run_info.submod.module(*args, **kwargs)
        else:
            result = self.optimus.run_info.submod(*args, **kwargs)

        self.optimus.run_info.env[0][self.optimus.run_info.name] = result
        return result

    def post_forward(self) -> None:
        """Send outputs to next stage if needed."""
        if self.optimus.tpl.stage < self.optimus.tpl.get_last_stage():
            next_rank = self.optimus.tpl.get_next_rank()

            for node_name, range_ in self.optimus.run_info.special_nodes.items():
                src_stage, needed_by_stage = range_

                if (self.optimus.tpl.stage >= src_stage and
                        self.optimus.tpl.stage < needed_by_stage):

                    if node_name in self.optimus.run_info.getitem_dic:
                        submod_name = self.optimus.run_info.getitem_dic[node_name][0]
                        if self.optimus.run_info.env_send_mark[0][submod_name] is None:
                            obj = self.optimus.run_info.env[0][submod_name]
                            self.optimus.comm.send_data(
                                obj, next_rank, self.optimus.run_info.device
                            )
                            self.optimus.run_info.env_send_mark[0][submod_name] = 1
                    else:
                        if self.optimus.run_info.env_send_mark[0][node_name] is None:
                            obj = self.optimus.run_info.env[0][node_name]
                            self.optimus.comm.send_data(
                                obj, next_rank, self.optimus.run_info.device
                            )
                            self.optimus.run_info.env_send_mark[0][node_name] = 1

    def get_output(self) -> Optional[Any]:
        """
        Get output from the last stage.

        Returns:
            Model output if this is the last stage, None otherwise
        """
        if not self.optimus.tpl.is_last_stage():
            return None

        output_node = self.optimus.run_info.output_node

        # For HuggingFace models, extract logits
        if self.optimus.model_type == self.optimus.model2type["hf"]:
            key_ = output_node.args[0].get('logits')
            if key_ is None:
                key_ = output_node.args[0]
        elif self.optimus.model_type == self.optimus.model2type["vt"]:
            key_ = output_node.args[0].get('logits')
            if key_ is None:
                key_ = output_node.args[0]
        else:
            key_ = output_node.args[0]

        if str(key_) in self.optimus.run_info.getitem_dic:
            submod_name = self.optimus.run_info.getitem_dic[str(key_)][0]
            idx = self.optimus.run_info.getitem_dic[str(key_)][1]
            return self.optimus.run_info.env[0][submod_name][idx]
        else:
            return self.optimus.run_info.env[0][str(key_)]

    @torch.no_grad()
    def run(self, data: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Optional[Any]:
        """
        Run inference for a single forward pass.

        Args:
            data: Input data tensor
            position_ids: Optional position IDs tensor (for KV cache mode)

        Returns:
            Model output if this is the last stage, None otherwise
        """
        self.init_env()

        if self.optimus.tpl.is_first_stage():
            self.get_input(data)
            if position_ids is not None:
                self.optimus.run_info.env[0]["position_ids"] = \
                    position_ids.to(self.optimus.run_info.device)

        self.pre_forward()
        self.forward_core()
        self.post_forward()

        return self.get_output()


class ScheduleGeneration(ScheduleInference):
    """
    Generation scheduler for autoregressive token-by-token generation.

    Extends ScheduleInference with KV cache support for efficient
    autoregressive generation.
    """

    def __init__(self, optimus_infer, kv_cache: Optional[KVCacheManager] = None):
        """
        Initialize generation scheduler.

        Args:
            optimus_infer: Optimus_Inference instance
            kv_cache: Optional KVCacheManager instance
        """
        super().__init__(optimus_infer)
        self.kv_cache = kv_cache

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Process the prompt and initialize KV cache (prefill phase).

        Args:
            input_ids: Input token IDs tensor of shape [batch_size, seq_len]

        Returns:
            Logits for the last position if this is the last stage, None otherwise
        """
        self.init_env()

        if self.optimus.tpl.is_first_stage():
            self.get_input(input_ids)

        self.pre_forward()
        self.forward_core()
        self.post_forward()

        # Update KV cache sequence length after prefill
        if self.kv_cache is not None and input_ids is not None:
            seq_len = input_ids.size(1) if input_ids.dim() > 1 else input_ids.size(0)
            self.kv_cache.set_seq_len(seq_len)

        output = self.get_output()
        if output is not None:
            # Return logits for the last position only
            if isinstance(output, torch.Tensor) and output.dim() >= 2:
                return output[:, -1, :]
        return output

    @torch.no_grad()
    def decode_step(
        self,
        input_token: torch.Tensor,
        position: int,
    ) -> Optional[torch.Tensor]:
        """
        Generate a single token using KV cache (decode phase).

        Args:
            input_token: Single token tensor of shape [batch_size, 1]
            position: Current position in the sequence

        Returns:
            Logits for the generated position if this is the last stage, None otherwise
        """
        self.init_env()

        if self.optimus.tpl.is_first_stage():
            self.get_input(input_token)

        self.pre_forward()
        self.forward_core()
        self.post_forward()

        # Update KV cache position after decode step
        if self.kv_cache is not None:
            self.kv_cache.set_seq_len(position + 1)

        output = self.get_output()
        if output is not None:
            # Return logits for the generated position
            if isinstance(output, torch.Tensor) and output.dim() >= 2:
                return output[:, -1, :]
        return output

    def clear_cache(self) -> None:
        """Clear KV cache for new generation."""
        if self.kv_cache is not None:
            self.kv_cache.clear()
