"""Deterministic seeding. Every experiment routes through here so that a run is
reproducible on the same machine and the JSON artifacts are stable."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU-only toy; keep the matmul path deterministic.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gen_noise(shape, std: float, seed: int) -> "torch.Tensor":
    """Deterministic Gaussian embedding noise. Used to push the model into an
    uncertainty regime where test-time experimentation has something to do."""
    if std <= 0.0:
        return None
    g = torch.Generator().manual_seed(seed)
    return std * torch.randn(*shape, generator=g)
