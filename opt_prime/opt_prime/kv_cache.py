#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#

"""
KV Cache Manager for Pipeline Parallel Inference

This module provides KV cache management for autoregressive text generation
in pipeline parallel settings. Each stage manages its own KV cache for the
layers it owns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class KVCacheManager:
    """
    KV Cache Manager for pipeline parallel inference.

    Each pipeline stage owns a subset of model layers and maintains
    KV cache only for those layers. The cache stores key and value
    tensors for each layer to enable efficient autoregressive generation.

    Args:
        num_layers: Total number of transformer layers in the model
        max_batch_size: Maximum batch size for cache allocation
        max_seq_len: Maximum sequence length for cache allocation
        num_heads: Number of attention heads (after TP split if applicable)
        head_dim: Dimension of each attention head
        dtype: Data type for cache tensors (default: torch.bfloat16)
        device: Device to allocate cache on
        layer_start: First layer index owned by this stage
        layer_end: Last layer index (exclusive) owned by this stage
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        layer_start: int = 0,
        layer_end: int = None,
    ):
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device or torch.device("cuda")

        # Layer range for this stage
        self.layer_start = layer_start
        self.layer_end = layer_end if layer_end is not None else num_layers
        self.num_local_layers = self.layer_end - self.layer_start

        # Cache storage: Dict[layer_idx, Tuple[key_cache, value_cache]]
        # Each cache has shape [batch_size, num_heads, seq_len, head_dim]
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Current sequence length for each batch item
        self._seq_len: int = 0
        self._batch_size: int = 0
        self._allocated: bool = False

    def allocate(self, batch_size: int, seq_len: Optional[int] = None) -> None:
        """
        Pre-allocate KV cache for the specified batch size.

        Args:
            batch_size: Batch size to allocate for
            seq_len: Optional initial sequence length (defaults to max_seq_len)
        """
        if seq_len is None:
            seq_len = self.max_seq_len

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Requested batch size {batch_size} exceeds max {self.max_batch_size}"
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Requested seq_len {seq_len} exceeds max {self.max_seq_len}"
            )

        self._batch_size = batch_size
        self._seq_len = 0  # Start with empty cache

        # Allocate cache for each local layer
        for layer_idx in range(self.layer_start, self.layer_end):
            key_cache = torch.zeros(
                (batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            value_cache = torch.zeros(
                (batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self._cache[layer_idx] = (key_cache, value_cache)

        self._allocated = True

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        position: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache with new key/value tensors and return the full cache.

        Args:
            layer_idx: Layer index to update
            new_k: New key tensor of shape [batch, num_heads, new_seq_len, head_dim]
            new_v: New value tensor of shape [batch, num_heads, new_seq_len, head_dim]
            position: Starting position for the update (defaults to current seq_len)

        Returns:
            Tuple of (full_key_cache, full_value_cache) up to current position
        """
        if layer_idx not in self._cache:
            raise ValueError(
                f"Layer {layer_idx} not in this stage's cache "
                f"(range: {self.layer_start}-{self.layer_end})"
            )

        key_cache, value_cache = self._cache[layer_idx]

        if position is None:
            position = self._seq_len

        new_seq_len = new_k.size(2)
        end_position = position + new_seq_len

        if end_position > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: position {position} + new_seq_len {new_seq_len} "
                f"> max_seq_len {self.max_seq_len}"
            )

        # Update cache at the specified position
        key_cache[:, :, position:end_position, :] = new_k
        value_cache[:, :, position:end_position, :] = new_v

        # Update sequence length if we're appending
        if layer_idx == self.layer_start:  # Only update once per position
            self._seq_len = max(self._seq_len, end_position)

        # Return the full cache up to current position
        return (
            key_cache[:, :, :end_position, :],
            value_cache[:, :, :end_position, :],
        )

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached key/value tensors for a layer.

        Args:
            layer_idx: Layer index to retrieve cache for

        Returns:
            Tuple of (key_cache, value_cache) up to current sequence length,
            or None if layer not in this stage's cache
        """
        if layer_idx not in self._cache:
            return None

        key_cache, value_cache = self._cache[layer_idx]

        if self._seq_len == 0:
            return None

        return (
            key_cache[:, :, :self._seq_len, :],
            value_cache[:, :, :self._seq_len, :],
        )

    def get_seq_len(self) -> int:
        """Get current sequence length in cache."""
        return self._seq_len

    def set_seq_len(self, seq_len: int) -> None:
        """Set current sequence length (used after prefill)."""
        self._seq_len = seq_len

    def clear(self) -> None:
        """Clear all cached values for new generation."""
        self._seq_len = 0
        for layer_idx in self._cache:
            key_cache, value_cache = self._cache[layer_idx]
            key_cache.zero_()
            value_cache.zero_()

    def free(self) -> None:
        """Free all cache memory."""
        self._cache.clear()
        self._allocated = False
        self._seq_len = 0
        self._batch_size = 0

    @property
    def is_allocated(self) -> bool:
        """Check if cache is allocated."""
        return self._allocated

    @property
    def batch_size(self) -> int:
        """Get current batch size."""
        return self._batch_size

    def get_memory_usage(self) -> int:
        """
        Get total memory usage of the cache in bytes.

        Returns:
            Memory usage in bytes
        """
        if not self._allocated:
            return 0

        # Each layer has 2 caches (K and V)
        # Shape: [batch_size, num_heads, max_seq_len, head_dim]
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        per_cache_elements = (
            self._batch_size * self.num_heads * self.max_seq_len * self.head_dim
        )
        return self.num_local_layers * 2 * per_cache_elements * element_size

    def __repr__(self) -> str:
        return (
            f"KVCacheManager("
            f"layers={self.layer_start}-{self.layer_end}, "
            f"batch_size={self._batch_size}, "
            f"seq_len={self._seq_len}/{self.max_seq_len}, "
            f"allocated={self._allocated})"
        )


class CachedScaledDotProductAttention(nn.Module):
    """
    Drop-in replacement for scaled_dot_product_attention that caches K, V
    across autoregressive decode steps.

    When disabled (_enabled=False), passes through to the standard SDPA.
    When enabled, it lazily allocates a KV cache on the first call and
    accumulates key/value tensors across calls.

    - Prefill (cache_pos==0 on entry): stores K,V and uses original causal masking.
    - Decode (cache_pos>0 on entry): appends new K,V, uses full cached context
      with no masking (all past positions are valid for the current query).

    Args:
        layer_idx: Global transformer layer index (for debugging).
        max_seq_len: Maximum sequence length for cache pre-allocation.
        dtype: Data type for cache tensors.
    """

    def __init__(
        self,
        layer_idx: int,
        max_seq_len: int = 2048,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None
        self.cache_pos: int = 0
        self._enabled: bool = False

    def enable(self) -> None:
        """Enable KV caching for a generation session."""
        self._enabled = True
        self.cache_pos = 0

    def disable(self) -> None:
        """Disable KV caching and free memory."""
        self._enabled = False
        self.cache_pos = 0
        self.cache_k = None
        self.cache_v = None

    def clear(self) -> None:
        """Reset cache position for a new generation (keeps allocation)."""
        self.cache_pos = 0
        if self.cache_k is not None:
            self.cache_k.zero_()
            self.cache_v.zero_()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if not self._enabled:
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask,
                dropout_p=dropout_p, is_causal=is_causal, **kwargs,
            )

        # Lazy cache allocation on first call
        if self.cache_k is None:
            batch_size, num_heads, _, head_dim = key.shape
            self.cache_k = torch.zeros(
                batch_size, num_heads, self.max_seq_len, head_dim,
                dtype=key.dtype, device=key.device,
            )
            self.cache_v = torch.zeros(
                batch_size, num_heads, self.max_seq_len, head_dim,
                dtype=value.dtype, device=value.device,
            )

        new_seq_len = key.size(2)
        end_pos = self.cache_pos + new_seq_len

        # Store new K, V in cache
        self.cache_k[:, :, self.cache_pos:end_pos, :] = key
        self.cache_v[:, :, self.cache_pos:end_pos, :] = value

        # Extract full cached K, V (contiguous for SDPA backend compatibility)
        full_key = self.cache_k[:, :, :end_pos, :].contiguous()
        full_value = self.cache_v[:, :, :end_pos, :].contiguous()

        is_decode = self.cache_pos > 0
        self.cache_pos = end_pos

        if is_decode:
            # Decode: query at current position attends to all cached positions.
            # No masking needed since all positions 0..P are valid for query at P.
            return F.scaled_dot_product_attention(
                query, full_key, full_value, attn_mask=None,
                dropout_p=dropout_p, is_causal=False, **kwargs,
            )
        else:
            # Prefill: standard causal attention with original mask
            return F.scaled_dot_product_attention(
                query, full_key, full_value, attn_mask=attn_mask,
                dropout_p=dropout_p, is_causal=is_causal, **kwargs,
            )

    def __repr__(self) -> str:
        return (
            f"CachedScaledDotProductAttention("
            f"layer={self.layer_idx}, "
            f"cache_pos={self.cache_pos}/{self.max_seq_len}, "
            f"enabled={self._enabled})"
        )
