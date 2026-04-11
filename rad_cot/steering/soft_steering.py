"""Phase 2: Soft Steering Maintenance — per-token, per-head activation correction.

Implements the RAD-CoT safety invariant at true per-head granularity:
  I_t = for all (l,h) in C_refusal: |v_refusal(l,h)^T @ a(l,h,t)| >= delta

When the invariant is violated for a specific head, applies a correction
only to that head's d_head-dimensional slice of the o_proj output:
  a_head(l, h, t) += alpha * (delta - pi_t) * v_refusal[l,h]

The orthogonal complement within each head (d_head - 1 dimensions) and
all non-circuit heads are left completely untouched.
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


class PerHeadSteeringHook:
    """Forward hook that maintains the refusal safety invariant per-head.

    Attached to a layer's o_proj module. For each token at each circuit head
    in this layer:
    1. Reshape o_proj output to (batch, seq, n_heads, d_head)
    2. For each circuit head in this layer, compute scalar projection onto
       its d_head-dimensional refusal direction
    3. If projection < delta, compute deficit and apply correction to that
       head's slice only
    4. Reshape back to (batch, seq, d_model)
    """

    def __init__(
        self,
        layer_idx: int,
        head_configs: list[tuple[int, np.ndarray]],  # list of (head_idx, refusal_direction)
        n_heads: int,
        d_head: int,
        delta: float,
        alpha: float = 0.3,
    ):
        self.layer_idx = layer_idx
        self.head_configs = head_configs
        self.n_heads = n_heads
        self.d_head = d_head
        self.delta = delta
        self.alpha = alpha
        self.stats = SteeringStats()

        # Pre-convert refusal directions to tensors (moved to device on first use)
        self._v_refusals: list[tuple[int, torch.Tensor]] | None = None
        self._head_configs_np = head_configs

    def _ensure_device(self, device: torch.device, dtype: torch.dtype):
        if self._v_refusals is None:
            self._v_refusals = [
                (head_idx, torch.tensor(v, device=device, dtype=dtype))
                for head_idx, v in self._head_configs_np
            ]
        elif self._v_refusals[0][1].device != device:
            self._v_refusals = [
                (head_idx, v.to(device=device, dtype=dtype))
                for head_idx, v in self._v_refusals
            ]

    def __call__(self, module, input, output):
        t_start = time.perf_counter()

        if isinstance(output, tuple):
            activation = output[0]
            is_tuple = True
        else:
            activation = output
            is_tuple = False

        self._ensure_device(activation.device, activation.dtype)

        batch, seq, d_model = activation.shape

        # Reshape to per-head: (batch, seq, n_heads, d_head)
        act_heads = activation.view(batch, seq, self.n_heads, self.d_head)

        # Only steer the last token position (autoregressive generation)
        needs_clone = False

        for head_idx, v in self._v_refusals:
            # Extract this head's activation at the last position
            last_head_act = act_heads[:, -1, head_idx, :]  # (batch, d_head)

            # Scalar projection onto refusal direction
            pi_t = torch.matmul(last_head_act, v)  # (batch,)

            # Check invariant: pi_t >= delta
            violations = pi_t < self.delta  # (batch,) boolean

            if violations.any():
                if not needs_clone:
                    act_heads = act_heads.clone()
                    needs_clone = True

                deficit = self.delta - pi_t  # (batch,)
                deficit = deficit * violations.float()  # zero out non-violations

                # Apply correction: a_head += alpha * deficit * v_refusal
                correction = self.alpha * deficit.unsqueeze(-1) * v.unsqueeze(0)  # (batch, d_head)
                act_heads[:, -1, head_idx, :] = act_heads[:, -1, head_idx, :] + correction

                self.stats.n_corrections += violations.sum().item()
                self.stats.total_deficit += deficit.sum().item()
                self.stats.invariant_violations += violations.sum().item()

        self.stats.n_tokens_generated += batch
        self.stats.correction_time_ms += (time.perf_counter() - t_start) * 1000

        if needs_clone:
            activation = act_heads.reshape(batch, seq, d_model)

        if is_tuple:
            return (activation,) + output[1:]
        return activation

    def reset_stats(self):
        self.stats = SteeringStats()


class RADCoTSteering:
    """Manages per-head soft steering hooks for all circuit layers during generation."""

    def __init__(
        self,
        model: nn.Module,
        dms_result: DMSResult,
        alpha: float = 0.3,
    ):
        self.model = model
        self.dms_result = dms_result
        self.alpha = alpha
        self._hooks: list[PerHeadSteeringHook] = []
        self._handles: list = []

    def attach(self):
        """Attach per-head steering hooks to all circuit layers."""
        if self._handles:
            self.detach()

        n_heads = self.dms_result.n_heads
        d_head = self.dms_result.d_head

        if n_heads == 0 or d_head == 0:
            raise ValueError(
                "DMSResult missing n_heads/d_head. Re-run DMS identification "
                "with the updated per-head pipeline."
            )

        # Group circuit indices by layer
        layer_to_heads: dict[int, list[tuple[int, np.ndarray]]] = {}
        for layer_idx, head_idx in self.dms_result.circuit_indices:
            v = self.dms_result.refusal_directions.get((layer_idx, head_idx))
            if v is None:
                logger.warning(
                    f"No refusal direction for ({layer_idx}, {head_idx}), skipping"
                )
                continue
            layer_to_heads.setdefault(layer_idx, []).append((head_idx, v))

        # One hook per layer (handles all circuit heads in that layer)
        for layer_idx, head_configs in layer_to_heads.items():
            hook = PerHeadSteeringHook(
                layer_idx=layer_idx,
                head_configs=head_configs,
                n_heads=n_heads,
                d_head=d_head,
                delta=self.dms_result.delta_threshold,
                alpha=self.alpha,
            )

            module = _get_attn_module(self.model, layer_idx)
            handle = module.register_forward_hook(hook)

            self._hooks.append(hook)
            self._handles.append(handle)

        total_heads = sum(len(hc) for hc in layer_to_heads.values())
        logger.info(
            f"Attached {len(self._hooks)} layer hooks covering {total_heads} "
            f"circuit heads (alpha={self.alpha}, "
            f"delta={self.dms_result.delta_threshold:.6f})"
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
    """Generate text with RAD-CoT per-head soft steering active.

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
