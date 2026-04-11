"""Forward hook utilities for activation extraction and patching.

Supports both layer-level and per-head activation capture/patching.
Per-head operations reshape the o_proj output from (batch, seq, d_model)
to (batch, seq, n_heads, d_head) to isolate individual head contributions.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn as nn


class ActivationCache:
    """Stores activations captured by forward hooks."""

    def __init__(self):
        self._cache: dict[str, torch.Tensor] = {}

    def store(self, name: str, tensor: torch.Tensor):
        self._cache[name] = tensor.detach().clone()

    def get(self, name: str) -> torch.Tensor:
        return self._cache[name]

    def keys(self):
        return self._cache.keys()

    def clear(self):
        self._cache.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._cache

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._cache[name]


def _get_attn_module(model: nn.Module, layer_idx: int) -> nn.Module:
    """Get the attention output projection module for a given layer.

    Supports Qwen2 and LLaMA-style architectures.
    """
    if hasattr(model, "model"):
        # HuggingFace wrapper
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        raise ValueError("Unsupported model architecture for hook attachment")

    layer = layers[layer_idx]
    # Qwen2 / LLaMA style
    if hasattr(layer, "self_attn"):
        return layer.self_attn.o_proj
    # GPT-style
    if hasattr(layer, "attn"):
        return layer.attn.c_proj
    raise ValueError(f"Cannot find attention output projection in layer {layer_idx}")


def _get_head_config(model: nn.Module) -> tuple[int, int]:
    """Extract (n_heads, d_head) from the model config.

    Returns the number of attention heads and dimension per head.
    """
    config = model.config
    n_heads = getattr(config, "num_attention_heads", None)
    d_head = getattr(config, "head_dim", None)
    if d_head is None and n_heads is not None:
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            d_head = hidden_size // n_heads
    if n_heads is None or d_head is None:
        raise ValueError(
            "Cannot determine n_heads/d_head from model config. "
            f"Available attrs: {[a for a in dir(config) if not a.startswith('_')]}"
        )
    return n_heads, d_head


def make_capture_hook(
    cache: ActivationCache, name: str
) -> Callable:
    """Create a forward hook that captures the module's output."""

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        cache.store(name, output)

    return hook_fn


def make_per_head_capture_hook(
    cache: ActivationCache,
    layer_idx: int,
    n_heads: int,
    d_head: int,
) -> Callable:
    """Create a forward hook that captures per-head activations.

    Reshapes o_proj output from (batch, seq, d_model) to per-head tensors
    and stores each as 'layer_{l}_head_{h}_attn_out'.
    """

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
        # act shape: (batch, seq, d_model) where d_model = n_heads * d_head
        batch, seq, d_model = act.shape
        # Reshape to (batch, seq, n_heads, d_head)
        act_heads = act.view(batch, seq, n_heads, d_head)
        for h in range(n_heads):
            name = f"layer_{layer_idx}_head_{h}_attn_out"
            cache.store(name, act_heads[:, :, h, :])  # (batch, seq, d_head)

    return hook_fn


def make_patch_hook(
    patch_values: dict[str, torch.Tensor],
    name: str,
) -> Callable:
    """Create a forward hook that replaces the module's output with patched values."""

    def hook_fn(module, input, output):
        if name in patch_values:
            patched = patch_values[name]
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched
        return output

    return hook_fn


def make_per_head_patch_hook(
    layer_idx: int,
    head_idx: int,
    patch_value: torch.Tensor,
    n_heads: int,
    d_head: int,
) -> Callable:
    """Create a forward hook that patches a single head's contribution.

    Replaces only head head_idx's slice of the o_proj output while leaving
    all other heads untouched.

    Args:
        patch_value: Tensor of shape (batch, seq, d_head) or broadcastable.
    """

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            act = output[0]
            is_tuple = True
        else:
            act = output
            is_tuple = False

        act = act.clone()
        batch, seq, d_model = act.shape
        # Reshape to (batch, seq, n_heads, d_head)
        act_heads = act.view(batch, seq, n_heads, d_head)
        # Patch only the target head
        pv = patch_value.to(act.device, act.dtype)
        if pv.dim() == 1:
            # (d_head,) -> broadcast to (batch, seq, d_head)
            act_heads[:, :, head_idx, :] = pv.unsqueeze(0).unsqueeze(0).expand(batch, seq, -1)
        else:
            act_heads[:, :, head_idx, :] = pv.expand_as(act_heads[:, :, head_idx, :])
        act = act_heads.reshape(batch, seq, d_model)

        if is_tuple:
            return (act,) + output[1:]
        return act

    return hook_fn


@contextmanager
def capture_activations(
    model: nn.Module,
    layer_indices: list[int],
    cache: ActivationCache | None = None,
):
    """Context manager to capture attention output activations for specified layers."""
    if cache is None:
        cache = ActivationCache()

    handles = []
    for layer_idx in layer_indices:
        name = f"layer_{layer_idx}_attn_out"
        module = _get_attn_module(model, layer_idx)
        handle = module.register_forward_hook(make_capture_hook(cache, name))
        handles.append(handle)

    try:
        yield cache
    finally:
        for h in handles:
            h.remove()


@contextmanager
def capture_per_head_activations(
    model: nn.Module,
    layer_indices: list[int],
    n_heads: int,
    d_head: int,
    cache: ActivationCache | None = None,
):
    """Context manager to capture per-head activations for specified layers.

    Stores activations as 'layer_{l}_head_{h}_attn_out' with shape
    (batch, seq, d_head) for each head in each layer.
    """
    if cache is None:
        cache = ActivationCache()

    handles = []
    for layer_idx in layer_indices:
        module = _get_attn_module(model, layer_idx)
        handle = module.register_forward_hook(
            make_per_head_capture_hook(cache, layer_idx, n_heads, d_head)
        )
        handles.append(handle)

    try:
        yield cache
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_activations(
    model: nn.Module,
    layer_indices: list[int],
    patch_values: dict[str, torch.Tensor],
):
    """Context manager to patch attention output activations during forward pass."""
    handles = []
    for layer_idx in layer_indices:
        name = f"layer_{layer_idx}_attn_out"
        module = _get_attn_module(model, layer_idx)
        handle = module.register_forward_hook(make_patch_hook(patch_values, name))
        handles.append(handle)

    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_single_head(
    model: nn.Module,
    layer_idx: int,
    head_idx: int,
    patch_value: torch.Tensor,
    n_heads: int,
    d_head: int,
):
    """Context manager to patch a single attention head's output.

    Only modifies head_idx in the specified layer; all other heads are untouched.
    """
    module = _get_attn_module(model, layer_idx)
    handle = module.register_forward_hook(
        make_per_head_patch_hook(layer_idx, head_idx, patch_value, n_heads, d_head)
    )
    try:
        yield
    finally:
        handle.remove()
