"""Model loading utilities for RAD-CoT."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rad_cot.utils.config import ModelConfig
from rad_cot.utils.logging import setup_logger

logger = setup_logger(__name__)

# Models supported for DMS analysis
SUPPORTED_MODELS = {
    "Qwen/Qwen3-14B": {"n_layers": 40, "n_heads": 40, "d_head": 128},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {"n_layers": 28, "n_heads": 28, "d_head": 128},
    "Qwen/Qwen2.5-0.5B-Instruct": {"n_layers": 24, "n_heads": 14, "d_head": 64},
    "Qwen/Qwen2.5-1.5B-Instruct": {"n_layers": 28, "n_heads": 12, "d_head": 128},
}


def load_model_and_tokenizer(
    config: ModelConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a HuggingFace model and tokenizer."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.dtype, torch.bfloat16)

    logger.info(f"Loading model: {config.name} (dtype={config.dtype})")

    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if config.device_map is not None:
        load_kwargs["device_map"] = config.device_map

    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        **load_kwargs,
    )
    model.eval()

    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
    return model, tokenizer


def get_model_info(model_name: str) -> dict:
    """Return architecture info for supported models."""
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]
    raise ValueError(
        f"Model {model_name} not in SUPPORTED_MODELS. "
        f"Add its architecture info to SUPPORTED_MODELS dict."
    )
