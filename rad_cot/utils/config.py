"""Configuration loading and management."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-14B"
    dtype: str = "bfloat16"
    device_map: str = "auto"


@dataclass
class DMSConfig:
    n_refuse: int = 500
    n_comply: int = 500
    n_benign: int = 500
    n_patching_prompts: int = 100
    dms_mass_threshold: float = 0.90
    delta_fraction: float = 0.80


@dataclass
class SteeringConfig:
    alpha: float = 0.3
    alpha_sweep: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.7, 1.0])


@dataclass
class AttackConfig:
    min_padding_tokens: int = 100
    max_padding_tokens: int = 500
    min_cot_length: int = 200


@dataclass
class EvalConfig:
    judge_model: str = "gpt-4o"
    n_attack_attempts: int = 100
    gsm8k_shots: int = 8
    math_shots: int = 4
    humaneval_samples: int = 10


@dataclass
class ComputeConfig:
    seed: int = 42
    max_seq_len: int = 4096
    batch_size: int = 4


@dataclass
class RADCoTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dms: DMSConfig = field(default_factory=DMSConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    wandb_project: str = "rad-cot"
    wandb_entity: str | None = None


def load_config(path: str | Path) -> RADCoTConfig:
    """Load configuration from a YAML file, merging with defaults."""
    path = Path(path)
    if not path.exists():
        return RADCoTConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = RADCoTConfig()

    if "model" in raw:
        cfg.model = ModelConfig(**raw["model"])
    if "dms" in raw:
        cfg.dms = DMSConfig(**raw["dms"])
    if "steering" in raw:
        cfg.steering = SteeringConfig(**raw["steering"])
    if "attack" in raw:
        cfg.attack = AttackConfig(**raw["attack"])
    if "eval" in raw:
        cfg.eval = EvalConfig(**raw["eval"])
    if "compute" in raw:
        cfg.compute = ComputeConfig(**raw["compute"])
    if "wandb" in raw:
        cfg.wandb_project = raw["wandb"].get("project", cfg.wandb_project)
        cfg.wandb_entity = raw["wandb"].get("entity", cfg.wandb_entity)

    return cfg
