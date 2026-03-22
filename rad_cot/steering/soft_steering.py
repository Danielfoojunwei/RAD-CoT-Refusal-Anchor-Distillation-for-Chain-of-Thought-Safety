"""Phase 2: Soft Steering Maintenance — per-token activation correction.

Implements the RAD-CoT safety invariant:
  I_t = for all (l,h) in C_refusal: ||proj_{v_refusal(l,h)}(a(l,h,t))|| >= delta

When the invariant is violated, applies a correction:
  activation(l, h, t) += alpha * deficit * v_refusal[l,h]
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

from rad_cot.steering.dms import DMSResult
from rad_cot.models.hooks import _get_attn_module
from rad_cot.utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class SteeringStats:
    """Statistics from a steering-augmented generation."""

    n_tokens_generated: int = 0
    n_corrections: int = 0
    total_deficit: float = 0.0
    invariant_violations: int = 0
    wall_time_ms: float = 0.0
    correction_time_ms: float = 0.0


class SoftSteeringHook:
    """Forward hook that maintains the refusal safety invariant during generation.

    For each token at each circuit layer:
    1. Compute scalar projection onto refusal direction
    2. If projection < delta, compute deficit and apply correction
    3. Track statistics for monitoring
    """

    def __init__(
        self,
        layer_idx: int,
        refusal_direction: np.ndarray,
        delta: float,
        alpha: float = 0.3,
    ):
        self.layer_idx = layer_idx
        self.delta = delta
        self.alpha = alpha
        self.stats = SteeringStats()

        # Store as tensor (will be moved to device on first use)
        self._v_refusal_np = refusal_direction
        self._v_refusal: torch.Tensor | None = None

    def _ensure_device(self, device: torch.device, dtype: torch.dtype):
        if self._v_refusal is None or self._v_refusal.device != device:
            self._v_refusal = torch.tensor(
                self._v_refusal_np, device=device, dtype=dtype
            )

    def __call__(self, module, input, output):
        t_start = time.perf_counter()

        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output

        self._ensure_device(activation.device, activation.dtype)
        v = self._v_refusal

        # activation shape: (batch, seq_len, d_model)
        # Only steer the last token position (during autoregressive generation)
        last_act = activation[:, -1, :]  # (batch, d_model)

        # Scalar projection onto refusal direction
        norm_t = torch.matmul(last_act, v)  # (batch,)

        # Check invariant: ||proj|| >= delta
        violations = norm_t < self.delta  # (batch,) boolean

        if violations.any():
            deficit = self.delta - norm_t  # (batch,)
            deficit = deficit * violations.float()  # zero out non-violations

            # Apply correction: a += alpha * deficit * v_refusal
            correction = self.alpha * deficit.unsqueeze(-1) * v.unsqueeze(0)  # (batch, d_model)
            activation = activation.clone()
            activation[:, -1, :] = activation[:, -1, :] + correction

            self.stats.n_corrections += violations.sum().item()
            self.stats.total_deficit += deficit.sum().item()
            self.stats.invariant_violations += violations.sum().item()

        self.stats.n_tokens_generated += activation.shape[0]  # batch tokens
        self.stats.correction_time_ms += (time.perf_counter() - t_start) * 1000

        if isinstance(output, tuple):
            return (activation,) + output[1:]
        return activation

    def reset_stats(self):
        self.stats = SteeringStats()


class RADCoTSteering:
    """Manages soft steering hooks for all circuit layers during generation."""

    def __init__(
        self,
        model: nn.Module,
        dms_result: DMSResult,
        alpha: float = 0.3,
    ):
        self.model = model
        self.dms_result = dms_result
        self.alpha = alpha
        self._hooks: list[SoftSteeringHook] = []
        self._handles: list = []

    def attach(self):
        """Attach steering hooks to all circuit layers."""
        if self._handles:
            self.detach()

        for layer_idx, head_idx in self.dms_result.circuit_indices:
            v_refusal = self.dms_result.refusal_directions[(layer_idx, head_idx)]

            hook = SoftSteeringHook(
                layer_idx=layer_idx,
                refusal_direction=v_refusal,
                delta=self.dms_result.delta_threshold,
                alpha=self.alpha,
            )

            module = _get_attn_module(self.model, layer_idx)
            handle = module.register_forward_hook(hook)

            self._hooks.append(hook)
            self._handles.append(handle)

        logger.info(
            f"Attached {len(self._hooks)} steering hooks "
            f"(alpha={self.alpha}, delta={self.dms_result.delta_threshold:.6f})"
        )

    def detach(self):
        """Remove all steering hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hooks.clear()
        logger.info("Detached all steering hooks")

    def get_stats(self) -> SteeringStats:
        """Aggregate statistics from all hooks."""
        combined = SteeringStats()
        for hook in self._hooks:
            combined.n_tokens_generated += hook.stats.n_tokens_generated
            combined.n_corrections += hook.stats.n_corrections
            combined.total_deficit += hook.stats.total_deficit
            combined.invariant_violations += hook.stats.invariant_violations
            combined.correction_time_ms += hook.stats.correction_time_ms
        return combined

    def reset_stats(self):
        for hook in self._hooks:
            hook.reset_stats()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()


def generate_with_steering(
    model: nn.Module,
    tokenizer,
    prompt: str,
    dms_result: DMSResult,
    alpha: float = 0.3,
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> tuple[str, SteeringStats]:
    """Generate text with RAD-CoT soft steering active.

    Returns (generated_text, steering_statistics).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    steering = RADCoTSteering(model, dms_result, alpha=alpha)

    with steering:
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
        wall_time = (time.perf_counter() - t_start) * 1000

    stats = steering.get_stats()
    stats.wall_time_ms = wall_time

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, stats
