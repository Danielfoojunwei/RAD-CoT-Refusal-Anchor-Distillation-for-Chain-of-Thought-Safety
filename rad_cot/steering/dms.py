"""DMS (Differential Mechanism Saliency) computation for refusal circuit identification.

Implements Phase 1 of RAD-CoT: identifying causal refusal circuits via
per-head DMS scoring of attention heads across all layers.

DMS(l, h) = delta_lh * CE_lh

where:
  delta_lh = || mean_activations(D_refuse, l, h) - mean_activations(D_comply, l, h) ||_2
  CE_lh = | d P(refusal_token) / d activation(l, h) |  via activation patching

Each score is computed at true per-head granularity by reshaping the o_proj
output from (batch, seq, d_model) to (batch, seq, n_heads, d_head).
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
    capture_per_head_activations,
    patch_single_head,
    _get_attn_module,
)
from rad_cot.utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class DMSResult:
    """Results from DMS circuit identification."""

    # DMS scores for all (layer, head) pairs
    dms_scores: np.ndarray  # shape: (n_layers, n_heads)
    delta_scores: np.ndarray  # context sensitivity component, shape: (n_layers, n_heads)
    ce_scores: np.ndarray  # causal effect component, shape: (n_layers, n_heads)

    # Selected circuit — true (layer, head) pairs
    circuit_indices: list[tuple[int, int]]  # list of (layer, head)
    k: int  # number of heads in circuit

    # Refusal directions per head in circuit — each is d_head-dimensional
    refusal_directions: dict[tuple[int, int], np.ndarray]  # (l,h) -> v_refusal

    # Threshold
    delta_threshold: float

    # Architecture info needed for steering
    n_heads: int = 0
    d_head: int = 0


def compute_per_head_mean_activations(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    layer_indices: list[int],
    n_heads: int,
    d_head: int,
    batch_size: int = 4,
    max_seq_len: int = 4096,
) -> dict[str, torch.Tensor]:
    """Compute mean per-head attention output activations over a set of prompts.

    Returns dict mapping "layer_{l}_head_{h}_attn_out" -> mean activation tensor
    of shape (d_head,).
    """
    device = next(model.parameters()).device
    accumulators: dict[str, torch.Tensor] = {}
    total_count = 0

    for i in tqdm(range(0, len(prompts), batch_size), desc="Computing per-head mean activations"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)

        cache = ActivationCache()
        with torch.no_grad(), capture_per_head_activations(
            model, layer_indices, n_heads, d_head, cache
        ):
            model(**inputs)

        for key in cache.keys():
            # act shape: (batch, seq, d_head)
            act = cache.get(key).float()
            # Mask padding tokens
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # (batch, seq, 1)
            # Average over sequence positions (masked), then over batch
            masked_sum = (act * mask).sum(dim=1)  # (batch, d_head)
            seq_counts = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
            per_sample_mean = masked_sum / seq_counts  # (batch, d_head)
            batch_mean = per_sample_mean.mean(dim=0)  # (d_head,)

            if key not in accumulators:
                accumulators[key] = batch_mean
            else:
                accumulators[key] += batch_mean

        total_count += 1
        cache.clear()

    # Average over all batches
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for key in accumulators:
        accumulators[key] /= n_batches

    return accumulators


def compute_delta_scores(
    mean_refuse: dict[str, torch.Tensor],
    mean_comply: dict[str, torch.Tensor],
    layer_indices: list[int],
    n_heads: int,
) -> np.ndarray:
    """Compute per-head context sensitivity delta.

    delta_lh = || mean_activations(D_refuse, l, h) - mean_activations(D_comply, l, h) ||_2

    Returns array of shape (n_layers, n_heads).
    """
    n_layers = len(layer_indices)
    deltas = np.zeros((n_layers, n_heads))

    for li, layer_idx in enumerate(layer_indices):
        for h in range(n_heads):
            key = f"layer_{layer_idx}_head_{h}_attn_out"
            if key in mean_refuse and key in mean_comply:
                diff = mean_refuse[key] - mean_comply[key]
                deltas[li, h] = torch.norm(diff, p=2).item()
            else:
                logger.warning(f"Missing activation for {key}")
                deltas[li, h] = 0.0

    return deltas


def compute_causal_effects(
    model: nn.Module,
    tokenizer,
    refuse_prompts: list[str],
    comply_mean_activations: dict[str, torch.Tensor],
    layer_indices: list[int],
    n_heads: int,
    d_head: int,
    refusal_token_ids: list[int],
    n_prompts: int = 100,
    max_seq_len: int = 4096,
) -> np.ndarray:
    """Compute per-head causal effect on refusal probability via activation patching.

    For each (layer, head), patch the refuse-run activation for that single head
    with the comply-run mean activation, then measure change in P(refusal_token).

    Returns array of shape (n_layers, n_heads).
    """
    device = next(model.parameters()).device
    prompts = refuse_prompts[:n_prompts]
    n_layers = len(layer_indices)
    causal_effects = np.zeros((n_layers, n_heads))

    for prompt in tqdm(prompts, desc="Computing per-head causal effects"):
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

        # Patched runs: one per (layer, head)
        for li, layer_idx in enumerate(layer_indices):
            for h in range(n_heads):
                key = f"layer_{layer_idx}_head_{h}_attn_out"
                if key not in comply_mean_activations:
                    continue

                # Patch this single head with comply-set mean activation
                patch_val = comply_mean_activations[key].to(device)  # (d_head,)

                with torch.no_grad(), patch_single_head(
                    model, layer_idx, h, patch_val, n_heads, d_head
                ):
                    patched_logits = model(**inputs).logits[0, -1]
                    patched_probs = torch.softmax(patched_logits, dim=-1)
                    p_refusal_patched = patched_probs[refusal_token_ids].sum().item()

                causal_effects[li, h] += abs(p_refusal_clean - p_refusal_patched)

    causal_effects /= len(prompts)
    return causal_effects


def select_circuit(
    dms_scores: np.ndarray,
    layer_indices: list[int],
    dms_mass_threshold: float = 0.90,
) -> tuple[list[tuple[int, int]], int]:
    """Select minimal set of (layer, head) pairs covering the specified
    fraction of total DMS mass.

    Args:
        dms_scores: Array of shape (n_layers, n_heads).
        layer_indices: Mapping from array row index to actual layer index.
        dms_mass_threshold: Fraction of total DMS mass to cover.

    Returns:
        (selected (layer, head) tuples sorted by DMS descending, K).
    """
    total_mass = dms_scores.sum()
    if total_mass == 0:
        logger.warning("Total DMS mass is 0 — no refusal circuit detected")
        return [], 0

    # Flatten and sort all (layer, head) pairs by DMS score
    n_layers, n_heads = dms_scores.shape
    flat_indices = np.argsort(dms_scores.ravel())[::-1]
    cumulative = 0.0
    selected = []

    for flat_idx in flat_indices:
        li = flat_idx // n_heads
        h = flat_idx % n_heads
        layer_idx = layer_indices[li]
        selected.append((layer_idx, int(h)))
        cumulative += dms_scores[li, h]
        if cumulative / total_mass >= dms_mass_threshold:
            break

    return selected, len(selected)


def compute_refusal_directions(
    model: nn.Module,
    tokenizer,
    refuse_prompts: list[str],
    comply_prompts: list[str],
    circuit_indices: list[tuple[int, int]],
    n_heads: int,
    d_head: int,
    max_seq_len: int = 4096,
) -> dict[tuple[int, int], np.ndarray]:
    """Compute per-head refusal direction via PCA on contrastive activation differences.

    v_refusal(l, h) = PCA_component_1(
        activations(D_refuse, l, h) - activations(D_comply, l, h)
    )

    Each direction is d_head-dimensional (not d_model-dimensional).
    """
    device = next(model.parameters()).device
    directions = {}

    # Group circuit indices by layer to minimize forward passes
    layer_to_heads: dict[int, list[int]] = {}
    for layer_idx, head_idx in circuit_indices:
        layer_to_heads.setdefault(layer_idx, []).append(head_idx)

    all_layers = sorted(layer_to_heads.keys())

    # Collect per-head activation differences
    head_diffs: dict[tuple[int, int], list[np.ndarray]] = {
        (l, h): [] for l, h in circuit_indices
    }

    for refuse_p, comply_p in tqdm(
        zip(refuse_prompts, comply_prompts), desc="Computing refusal directions",
        total=min(len(refuse_prompts), len(comply_prompts)),
    ):
        # Get refuse activations for all circuit layers
        refuse_inputs = tokenizer(
            refuse_p, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)
        cache_r = ActivationCache()
        with torch.no_grad(), capture_per_head_activations(
            model, all_layers, n_heads, d_head, cache_r
        ):
            model(**refuse_inputs)

        # Get comply activations
        comply_inputs = tokenizer(
            comply_p, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)
        cache_c = ActivationCache()
        with torch.no_grad(), capture_per_head_activations(
            model, all_layers, n_heads, d_head, cache_c
        ):
            model(**comply_inputs)

        for layer_idx, head_idx in circuit_indices:
            key = f"layer_{layer_idx}_head_{head_idx}_attn_out"
            # Mean over sequence positions -> (d_head,)
            act_r = cache_r[key][0].mean(dim=0).cpu().numpy()
            act_c = cache_c[key][0].mean(dim=0).cpu().numpy()
            head_diffs[(layer_idx, head_idx)].append(act_r - act_c)

    # PCA on the difference vectors for each head
    for (layer_idx, head_idx), diffs in head_diffs.items():
        if len(diffs) < 2:
            logger.warning(
                f"Only {len(diffs)} samples for layer {layer_idx} head {head_idx}, "
                "using mean difference as refusal direction"
            )
            v = np.mean(diffs, axis=0) if diffs else np.zeros(d_head)
            norm = np.linalg.norm(v)
            directions[(layer_idx, head_idx)] = v / norm if norm > 0 else v
            continue

        diff_matrix = np.stack(diffs)
        pca = PCA(n_components=1)
        pca.fit(diff_matrix)
        v = pca.components_[0]
        v /= np.linalg.norm(v)
        directions[(layer_idx, head_idx)] = v

    return directions


def compute_delta_threshold(
    model: nn.Module,
    tokenizer,
    refuse_prompts: list[str],
    circuit_indices: list[tuple[int, int]],
    refusal_directions: dict[tuple[int, int], np.ndarray],
    n_heads: int,
    d_head: int,
    delta_fraction: float = 0.80,
    max_seq_len: int = 4096,
) -> float:
    """Compute the delta threshold for the safety invariant.

    delta = delta_fraction * min over D_refuse prompts of min over circuit heads of
            |proj_{v_refusal}(activation(l, h, t))|
    """
    device = next(model.parameters()).device
    all_layers = sorted(set(l for l, h in circuit_indices))
    min_norms = []

    for prompt in tqdm(refuse_prompts[:100], desc="Calibrating delta threshold"):
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)

        cache = ActivationCache()
        with torch.no_grad(), capture_per_head_activations(
            model, all_layers, n_heads, d_head, cache
        ):
            model(**inputs)

        prompt_min = float("inf")
        for layer_idx, head_idx in circuit_indices:
            key = f"layer_{layer_idx}_head_{head_idx}_attn_out"
            act = cache[key][0]  # (seq_len, d_head)
            v = torch.tensor(
                refusal_directions[(layer_idx, head_idx)],
                device=device, dtype=act.dtype,
            )
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
    logger.info(
        f"Delta threshold: {delta:.6f} "
        f"(fraction={delta_fraction}, min_norm={min(min_norms):.6f})"
    )
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

    All computations are performed at true per-head granularity:
    1. Compute per-head mean activations for refuse and comply datasets
    2. Compute per-head delta scores (context sensitivity)
    3. Compute per-head causal effects via single-head activation patching
    4. Compute DMS = delta * CE per (layer, head)
    5. Select circuit (minimal heads covering 90% DMS mass)
    6. Compute per-head refusal directions via PCA (d_head-dimensional)
    7. Calibrate delta threshold
    """
    layer_indices = list(range(n_layers))

    logger.info("Step 1/7: Computing per-head mean activations for D_refuse...")
    mean_refuse = compute_per_head_mean_activations(
        model, tokenizer, refuse_prompts, layer_indices,
        n_heads, d_head, batch_size, max_seq_len,
    )

    logger.info("Step 2/7: Computing per-head mean activations for D_comply...")
    mean_comply = compute_per_head_mean_activations(
        model, tokenizer, comply_prompts, layer_indices,
        n_heads, d_head, batch_size, max_seq_len,
    )

    logger.info("Step 3/7: Computing per-head delta scores...")
    delta_scores = compute_delta_scores(mean_refuse, mean_comply, layer_indices, n_heads)

    logger.info("Step 4/7: Computing per-head causal effects via activation patching...")
    ce_scores = compute_causal_effects(
        model, tokenizer, refuse_prompts, mean_comply, layer_indices,
        n_heads, d_head, refusal_token_ids, n_patching_prompts, max_seq_len,
    )

    # DMS = delta * CE (element-wise for all (layer, head) pairs)
    dms_scores = delta_scores * ce_scores  # shape: (n_layers, n_heads)

    logger.info("Step 5/7: Selecting refusal circuit...")
    circuit_indices, k = select_circuit(dms_scores, layer_indices, dms_mass_threshold)

    logger.info(f"Selected K={k} heads covering {dms_mass_threshold*100:.0f}% DMS mass")
    logger.info(f"Circuit (layer, head) pairs: {circuit_indices}")

    logger.info("Step 6/7: Computing per-head refusal directions...")
    refusal_directions = compute_refusal_directions(
        model, tokenizer, refuse_prompts, comply_prompts,
        circuit_indices, n_heads, d_head, max_seq_len,
    )

    logger.info("Step 7/7: Calibrating delta threshold...")
    delta_threshold = compute_delta_threshold(
        model, tokenizer, refuse_prompts, circuit_indices,
        refusal_directions, n_heads, d_head, delta_fraction, max_seq_len,
    )

    return DMSResult(
        dms_scores=dms_scores,
        delta_scores=delta_scores,
        ce_scores=ce_scores,
        circuit_indices=circuit_indices,
        k=k,
        refusal_directions=refusal_directions,
        delta_threshold=delta_threshold,
        n_heads=n_heads,
        d_head=d_head,
    )
