"""DMS (Differential Mean Score) computation for refusal circuit identification.

Implements Phase 1 of RAD-CoT: identifying causal refusal circuits via
DMS scoring of attention heads across reasoning layers.

DMS(l, h) = delta_lh * CE_lh

where:
  delta_lh = || mean_activations(D_refuse, l, h) - mean_activations(D_comply, l, h) ||_2
  CE_lh = | d P(refusal_token) / d activation(l, h) |  via activation patching
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from rad_cot.models.hooks import (
    ActivationCache,
    capture_activations,
    patch_activations,
    _get_attn_module,
)
from rad_cot.utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class DMSResult:
    """Results from DMS circuit identification."""

    # DMS scores for all (layer, head) pairs
    dms_scores: np.ndarray  # shape: (n_layers, n_heads)
    delta_scores: np.ndarray  # context sensitivity component
    ce_scores: np.ndarray  # causal effect component

    # Selected circuit
    circuit_indices: list[tuple[int, int]]  # list of (layer, head)
    k: int  # number of heads in circuit

    # Refusal directions per head in circuit
    refusal_directions: dict[tuple[int, int], np.ndarray]  # (l,h) -> v_refusal

    # Threshold
    delta_threshold: float


def compute_mean_activations(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    layer_indices: list[int],
    n_heads: int,
    d_head: int,
    batch_size: int = 4,
    max_seq_len: int = 4096,
) -> dict[str, torch.Tensor]:
    """Compute mean attention output activations over a set of prompts.

    Returns dict mapping "layer_{l}_attn_out" -> mean activation tensor.
    """
    device = next(model.parameters()).device
    accumulators: dict[str, torch.Tensor] = {}
    count = 0

    for i in tqdm(range(0, len(prompts), batch_size), desc="Computing mean activations"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)

        cache = ActivationCache()
        with torch.no_grad(), capture_activations(model, layer_indices, cache):
            model(**inputs)

        for key in cache.keys():
            # Average over batch and sequence dimensions -> (d_model,)
            act = cache.get(key).float()
            # Mask padding tokens
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            masked_act = (act * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp(min=1)

            if key not in accumulators:
                accumulators[key] = masked_act
            else:
                accumulators[key] += masked_act
            count = i + len(batch)

        cache.clear()

    # Average over all prompts
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for key in accumulators:
        accumulators[key] /= n_batches

    return accumulators


def compute_delta_scores(
    mean_refuse: dict[str, torch.Tensor],
    mean_comply: dict[str, torch.Tensor],
    layer_indices: list[int],
) -> np.ndarray:
    """Compute context sensitivity delta for each layer.

    delta_lh = || mean_activations(D_refuse, l, h) - mean_activations(D_comply, l, h) ||_2

    Note: Since we work with the full attention output (not per-head),
    we compute the L2 norm of the difference for each layer.
    Per-head decomposition is done by reshaping.
    """
    deltas = []
    for layer_idx in layer_indices:
        key = f"layer_{layer_idx}_attn_out"
        diff = mean_refuse[key] - mean_comply[key]
        delta = torch.norm(diff, p=2).item()
        deltas.append(delta)
    return np.array(deltas)


def compute_causal_effects(
    model: nn.Module,
    tokenizer,
    refuse_prompts: list[str],
    comply_activations: dict[str, torch.Tensor],
    layer_indices: list[int],
    refusal_token_ids: list[int],
    n_prompts: int = 100,
    max_seq_len: int = 4096,
) -> np.ndarray:
    """Compute causal effect of each layer's attention on refusal probability.

    For each layer, patch refuse-run activations with comply-run activations
    and measure change in P(refusal_token).

    CE_lh = | d P(refusal_token) / d activation(l, h) |
    """
    device = next(model.parameters()).device
    prompts = refuse_prompts[:n_prompts]
    causal_effects = np.zeros(len(layer_indices))

    for prompt in tqdm(prompts, desc="Computing causal effects"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        ).to(device)

        # Clean run: get baseline refusal probability
        with torch.no_grad():
            clean_logits = model(**inputs).logits[0, -1]  # last token logits
            clean_probs = torch.softmax(clean_logits, dim=-1)
            p_refusal_clean = clean_probs[refusal_token_ids].sum().item()

        # Patched runs: one per layer
        for idx, layer_idx in enumerate(layer_indices):
            key = f"layer_{layer_idx}_attn_out"

            # Build patch: replace this layer's activations with comply-set means
            # Expand to match input shape
            patch_val = comply_activations[key].unsqueeze(0).unsqueeze(0)
            patch_val = patch_val.expand(1, inputs["input_ids"].shape[1], -1)
            patch_dict = {key: patch_val.to(device)}

            with torch.no_grad(), patch_activations(model, [layer_idx], patch_dict):
                patched_logits = model(**inputs).logits[0, -1]
                patched_probs = torch.softmax(patched_logits, dim=-1)
                p_refusal_patched = patched_probs[refusal_token_ids].sum().item()

            causal_effects[idx] += abs(p_refusal_clean - p_refusal_patched)

    causal_effects /= len(prompts)
    return causal_effects


def select_circuit(
    dms_scores: np.ndarray,
    dms_mass_threshold: float = 0.90,
) -> tuple[list[int], int]:
    """Select minimal set of layers covering the specified fraction of total DMS mass.

    Returns (selected_layer_indices_sorted_by_dms, K).
    """
    total_mass = dms_scores.sum()
    if total_mass == 0:
        logger.warning("Total DMS mass is 0 — no refusal circuit detected")
        return [], 0

    sorted_indices = np.argsort(dms_scores)[::-1]
    cumulative = 0.0
    selected = []

    for idx in sorted_indices:
        selected.append(int(idx))
        cumulative += dms_scores[idx]
        if cumulative / total_mass >= dms_mass_threshold:
            break

    return selected, len(selected)


def compute_refusal_directions(
    model: nn.Module,
    tokenizer,
    refuse_prompts: list[str],
    comply_prompts: list[str],
    circuit_layer_indices: list[int],
    batch_size: int = 4,
    max_seq_len: int = 4096,
) -> dict[int, np.ndarray]:
    """Compute refusal direction for each layer in the circuit via PCA.

    v_refusal(l) = PCA_component_1(activations(D_refuse, l) - activations(D_comply, l))
    """
    device = next(model.parameters()).device
    directions = {}

    for layer_idx in tqdm(circuit_layer_indices, desc="Computing refusal directions"):
        diffs = []

        for refuse_p, comply_p in zip(refuse_prompts, comply_prompts):
            # Get refuse activation
            refuse_inputs = tokenizer(
                refuse_p, return_tensors="pt", truncation=True, max_length=max_seq_len
            ).to(device)
            cache_r = ActivationCache()
            with torch.no_grad(), capture_activations(model, [layer_idx], cache_r):
                model(**refuse_inputs)
            act_r = cache_r[f"layer_{layer_idx}_attn_out"][0].mean(dim=0).cpu().numpy()

            # Get comply activation
            comply_inputs = tokenizer(
                comply_p, return_tensors="pt", truncation=True, max_length=max_seq_len
            ).to(device)
            cache_c = ActivationCache()
            with torch.no_grad(), capture_activations(model, [layer_idx], cache_c):
                model(**comply_inputs)
            act_c = cache_c[f"layer_{layer_idx}_attn_out"][0].mean(dim=0).cpu().numpy()

            diffs.append(act_r - act_c)

        diff_matrix = np.stack(diffs)
        pca = PCA(n_components=1)
        pca.fit(diff_matrix)
        directions[layer_idx] = pca.components_[0]
        directions[layer_idx] /= np.linalg.norm(directions[layer_idx])

    return directions


def compute_delta_threshold(
    model: nn.Module,
    tokenizer,
    refuse_prompts: list[str],
    circuit_layer_indices: list[int],
    refusal_directions: dict[int, np.ndarray],
    delta_fraction: float = 0.80,
    max_seq_len: int = 4096,
) -> float:
    """Compute the delta threshold for the safety invariant.

    delta = delta_fraction * min over D_refuse prompts of min over CoT steps of
            ||proj_{v_refusal}(activation(l,h,t))||
    """
    device = next(model.parameters()).device
    min_norms = []

    for prompt in tqdm(refuse_prompts[:100], desc="Calibrating delta threshold"):
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)

        cache = ActivationCache()
        with torch.no_grad(), capture_activations(model, circuit_layer_indices, cache):
            model(**inputs)

        prompt_min = float("inf")
        for layer_idx in circuit_layer_indices:
            act = cache[f"layer_{layer_idx}_attn_out"][0]  # (seq_len, d_model)
            v = torch.tensor(refusal_directions[layer_idx], device=device, dtype=act.dtype)
            projections = torch.matmul(act, v)  # (seq_len,)
            norms = projections.abs()
            layer_min = norms.min().item()
            prompt_min = min(prompt_min, layer_min)

        if prompt_min < float("inf"):
            min_norms.append(prompt_min)

    if not min_norms:
        logger.warning("No valid norms computed for delta threshold")
        return 0.0

    delta = delta_fraction * min(min_norms)
    logger.info(f"Delta threshold: {delta:.6f} (fraction={delta_fraction}, min_norm={min(min_norms):.6f})")
    return delta


def run_dms_identification(
    model: nn.Module,
    tokenizer,
    refuse_prompts: list[str],
    comply_prompts: list[str],
    refusal_token_ids: list[int],
    n_layers: int,
    n_heads: int,
    d_head: int,
    dms_mass_threshold: float = 0.90,
    delta_fraction: float = 0.80,
    n_patching_prompts: int = 100,
    batch_size: int = 4,
    max_seq_len: int = 4096,
) -> DMSResult:
    """Run the full DMS circuit identification pipeline (Phase 1).

    Steps:
    1. Compute mean activations for refuse and comply datasets
    2. Compute delta scores (context sensitivity)
    3. Compute causal effects via activation patching
    4. Compute DMS = delta * CE
    5. Select circuit (minimal heads covering 90% DMS mass)
    6. Compute refusal directions via PCA
    7. Calibrate delta threshold
    """
    layer_indices = list(range(n_layers))

    logger.info("Step 1/7: Computing mean activations for D_refuse...")
    mean_refuse = compute_mean_activations(
        model, tokenizer, refuse_prompts, layer_indices, n_heads, d_head, batch_size, max_seq_len
    )

    logger.info("Step 2/7: Computing mean activations for D_comply...")
    mean_comply = compute_mean_activations(
        model, tokenizer, comply_prompts, layer_indices, n_heads, d_head, batch_size, max_seq_len
    )

    logger.info("Step 3/7: Computing delta scores...")
    delta_scores = compute_delta_scores(mean_refuse, mean_comply, layer_indices)

    logger.info("Step 4/7: Computing causal effects via activation patching...")
    ce_scores = compute_causal_effects(
        model, tokenizer, refuse_prompts, mean_comply, layer_indices,
        refusal_token_ids, n_patching_prompts, max_seq_len
    )

    # DMS = delta * CE (element-wise for layers)
    dms_scores_flat = delta_scores * ce_scores
    # Reshape to (n_layers, 1) — per-head decomposition is a future refinement
    dms_scores = dms_scores_flat.reshape(-1, 1)

    logger.info("Step 5/7: Selecting refusal circuit...")
    selected_layers, k = select_circuit(dms_scores_flat, dms_mass_threshold)
    circuit_indices = [(l, 0) for l in selected_layers]  # (layer, head=0) placeholder

    logger.info(f"Selected K={k} layers covering {dms_mass_threshold*100:.0f}% DMS mass")
    logger.info(f"Circuit layers: {selected_layers}")

    logger.info("Step 6/7: Computing refusal directions...")
    refusal_directions_np = compute_refusal_directions(
        model, tokenizer, refuse_prompts, comply_prompts, selected_layers, batch_size, max_seq_len
    )
    refusal_directions = {(l, 0): refusal_directions_np[l] for l in selected_layers}

    logger.info("Step 7/7: Calibrating delta threshold...")
    delta_threshold = compute_delta_threshold(
        model, tokenizer, refuse_prompts, selected_layers,
        refusal_directions_np, delta_fraction, max_seq_len
    )

    return DMSResult(
        dms_scores=dms_scores,
        delta_scores=delta_scores,
        ce_scores=ce_scores,
        circuit_indices=circuit_indices,
        k=k,
        refusal_directions=refusal_directions,
        delta_threshold=delta_threshold,
    )
