"""Forward hook utilities for activation extraction and patching."""

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


def make_capture_hook(
    cache: ActivationCache, name: str
) -> Callable:
    """Create a forward hook that captures the module's output."""

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        cache.store(name, output)

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
