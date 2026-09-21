from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def strict_determinism_enabled() -> bool:
    return os.environ.get("RQSAIL_DETERMINISTIC", "1").strip().lower() in {
        "1", "true", "yes", "on"
    }


def set_seed(seed: int = 42) -> None:
    deterministic = strict_determinism_enabled()
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def load_public_config(config_or_path) -> dict:
    if isinstance(config_or_path, dict):
        return copy.deepcopy(config_or_path)
    path = Path(config_or_path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def embedded_model_config(config: dict) -> dict:
    definition = config.get("model_definition")
    if not isinstance(definition, dict) or "model" not in definition:
        raise ValueError("model_definition.model is required")
    return copy.deepcopy(definition)


def get_device(prefer_gpu: bool = True) -> torch.device:
    return torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")

