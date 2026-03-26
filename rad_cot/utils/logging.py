"""Logging and experiment tracking utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "rad_cot",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def init_wandb(project: str, entity: str | None = None, config: dict | None = None):
    """Initialize Weights & Biases run."""
    try:
        import wandb

        wandb.init(project=project, entity=entity, config=config)
        return wandb.run
    except ImportError:
        logging.getLogger("rad_cot").warning("wandb not installed, skipping tracking")
        return None
