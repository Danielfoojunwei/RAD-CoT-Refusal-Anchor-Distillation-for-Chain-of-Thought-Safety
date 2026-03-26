# RAD-CoT: Refusal-Anchor Distillation for Chain-of-Thought Safety

Preventing Safety Signal Dilution in Large Reasoning Models via Causal Circuit Maintenance.

**Target Venue:** NeurIPS 2026

## Overview

RAD-CoT identifies and maintains the internal attention circuits that causally mediate refusal behavior during chain-of-thought generation. By applying per-token soft steering to prevent dilution of these "refusal anchor circuits," RAD-CoT aims to reduce CoT-Hijacking attack success rate from 99% to below 5% with less than 3% degradation on reasoning benchmarks.

## Method

**Phase 1 — DMS Circuit Identification (Offline)**
- Compute DMS scores: `DMS(l,h) = delta_lh * CE_lh` for all attention heads
- Select minimal circuit `C_refusal` covering 90% of DMS mass
- Extract refusal directions via PCA and calibrate threshold `delta`

**Phase 2 — Soft Steering (Inference)**
- For each token at each circuit layer, check safety invariant:
  `I_t = forall (l,h) in C_refusal: ||proj_{v_refusal}(a(l,h,t))|| >= delta`
- When violated, apply correction: `a(l,h,t) += alpha * deficit * v_refusal`

## Project Structure

```
rad_cot/
  data/           # Calibration dataset construction
  models/         # Model loading and hook utilities
  steering/       # DMS scoring and soft steering
  evaluation/     # Safety judges and benchmark wrappers
  experiments/    # Experiment module
  utils/          # Config and logging
scripts/
  run_dms_identification.py   # Phase 1: DMS circuit identification
  exp1_causal_validation.py   # Exp 1: Causal validation of C_refusal
  exp2_safety_eval.py         # Exp 2: Primary safety evaluation (ASR)
  exp3_reasoning_quality.py   # Exp 3: Reasoning quality preservation
  exp4_generalisation.py      # Exp 4: Generalisation to H-CoT
  exp5_ablations.py           # Exp 5: Ablation studies (K, alpha, delta, depth)
  benchmark_latency.py        # Latency overhead benchmarking
  download_datasets.py        # Download all required datasets
configs/
  default.yaml                # Default configuration
data/                         # Downloaded datasets
```

## Setup

```bash
pip install -e ".[dev]" datasets
python scripts/download_datasets.py --output-dir data
```

Note: AdvBench and HarmBench are gated datasets requiring HuggingFace authentication. Set `HF_TOKEN` or place the JSON files in `data/` manually.

## Running Experiments

```bash
# Phase 1: DMS Circuit Identification
python scripts/run_dms_identification.py \
    --model Qwen/Qwen3-14B \
    --advbench-path data/advbench.json \
    --benign-path data/flan_benign.json

# Experiment 1: Causal Validation
python scripts/exp1_causal_validation.py \
    --dms-result outputs/dms/dms_result.pkl \
    --refuse-data outputs/dms/calibration/d_refuse.json

# Experiment 2: Primary Safety Evaluation
python scripts/exp2_safety_eval.py \
    --dms-result outputs/dms/dms_result.pkl \
    --attack-prompts data/advbench.json

# Experiment 3: Reasoning Quality
python scripts/exp3_reasoning_quality.py \
    --dms-result outputs/dms/dms_result.pkl

# Experiment 4: H-CoT Generalisation
python scripts/exp4_generalisation.py \
    --dms-result outputs/dms/dms_result.pkl \
    --hcot-prompts data/harmbench.json

# Experiment 5: Ablation Studies
python scripts/exp5_ablations.py \
    --dms-result outputs/dms/dms_result.pkl \
    --attack-prompts data/advbench.json
```

## Primary Models

- **Qwen3-14B** (40 layers, 40 heads, d_head=128)
- **DeepSeek-R1-Distill-Qwen-7B** (28 layers, 28 heads, d_head=128)

## KPI Targets

| KPI | Baseline | Target |
|-----|----------|--------|
| CoT-Hijacking ASR | 99% | < 5% |
| H-CoT ASR Reduction | ~80% | 60%+ relative |
| GSM8K drop | — | < 3% absolute |
| MATH L4-5 drop | — | < 3% absolute |
| HumanEval drop | — | < 3% absolute |
| Benign refusal increase | < 2% | < 5% absolute |
| Inference latency overhead | — | < 8% |

## Dependencies

- torch >= 2.2.0
- transformers >= 4.40.0
- scikit-learn >= 1.3.0
- wandb >= 0.16.0
- datasets (HuggingFace)
