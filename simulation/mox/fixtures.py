"""Shared fixtures: train-or-load the model and build every derived object the
experiments share (steering directions, SAE, correctness probe, manifold model,
calibrated steering scale). Deterministic given the configs and seed.

The model learns both sub-tasks cleanly. Embedding noise (NOISE_STD) then pushes
it into an uncertainty regime where some answers are wrong but the latent
structure to recover them is still present. This is the regime in which test-time
experimentation is supposed to matter: high uncertainty, recoverable signal.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from mox import data, probes, sae as sae_mod
from mox.model import ModelConfig, TinyTransformer
from mox.paths import CHECKPOINTS
from mox.scoring import ManifoldModel
from mox.seeds import gen_noise, seed_everything
from mox.train import TrainConfig, train_model, evaluate

# Single task / training configuration used across all experiments.
TASK = data.TaskConfig(n_values=6, n_pairs=3, m_bits=7, max_len=10)
TRAIN = TrainConfig(n_layers=2, d_model=64, n_heads=4, d_mlp=256, steps=4000,
                    lr=3e-3, frac_retrieval=0.5)
LAYER = 1            # resid_post of the final block
NOISE_STD = 0.9      # embedding-noise level for the deployment regime


def _key():
    blob = json.dumps({"task": TASK.__dict__, "train": TRAIN.__dict__,
                       "layer": LAYER}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def eval_noise(n: int, seed: int, max_len: int | None = None):
    return gen_noise((n, max_len or TASK.max_len, TRAIN.d_model), NOISE_STD, seed)


def load_model(seed: int = 0):
    ckpt = CHECKPOINTS / f"model_{_key()}_s{seed}.pt"
    mc = ModelConfig(vocab_size=TASK.vocab_size, max_len=TASK.max_len,
                     d_model=TRAIN.d_model, n_layers=TRAIN.n_layers,
                     n_heads=TRAIN.n_heads, d_mlp=TRAIN.d_mlp)
    model = TinyTransformer(mc)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt))
        model.eval()
        return model
    model, _ = train_model(TASK, TRAIN, seed=seed)
    torch.save(model.state_dict(), ckpt)
    return model


def steer_scale(model, mult: float = 6.0, layer: int = LAYER):
    """A single global steering coefficient, a multiple of the mean
    answer-position residual norm (ActAdd uses large fixed coefficients)."""
    c = probes.collect_ans_residuals(model, TASK, layer, n=2000, seed=3,
                                     noise_std=NOISE_STD)
    return mult * c["h"].norm(dim=-1).mean().item()


def build_bundle(seed: int = 0):
    seed_everything(seed)
    model = load_model(seed)
    ev_clean = evaluate(model, TASK, noise_std=0.0)
    ev = evaluate(model, TASK, noise_std=NOISE_STD)
    dirs, _ = probes.steering_directions(model, TASK, LAYER, n=6000, seed=7,
                                         noise_std=NOISE_STD)
    sae, sae_stats = sae_mod.train_sae(model, TASK, LAYER, n_features=256,
                                       l1=0.3, steps=3000, seed=5,
                                       noise_std=NOISE_STD)
    probe = probes.correctness_probe(model, TASK, LAYER, n=8000, seed=11,
                                     noise_std=NOISE_STD)
    c = probes.collect_ans_residuals(model, TASK, LAYER, n=4000, seed=13,
                                     noise_std=NOISE_STD)
    manifold = ManifoldModel(c["h"])
    alpha = steer_scale(model)
    return {
        "model": model, "dirs": dirs, "sae": sae, "sae_stats": sae_stats,
        "probe": probe, "manifold": manifold, "alpha": alpha,
        "eval": ev, "eval_clean": ev_clean, "task": TASK, "layer": LAYER,
        "noise_std": NOISE_STD,
    }
