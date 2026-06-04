"""Steering directions and a correctness probe, built from cached residuals.

Difference-of-means steering directions are the standard construction in the
inference-time-intervention and activation-addition literature (Li et al. 2023;
Turner et al. 2023): the direction that, added to the residual stream, pushes
the model toward a target answer. The correctness probe is a logistic readout of
"is the current state going to answer correctly", used as a *deployable* scorer
that does not see ground truth at selection time.
"""
from __future__ import annotations

import numpy as np
import torch

from mox import data
from mox.model import answer_logits
from mox.seeds import gen_noise


@torch.no_grad()
def collect_ans_residuals(model, task, layer, n=6000, seed=7, frac_retrieval=0.5,
                          noise_std=0.0):
    """Residual at the answer position, with metadata, for probe building."""
    toks, ans_pos, target, is_retr = data.make_batch(
        task, n, seed=seed, frac_retrieval=frac_retrieval
    )
    noise = gen_noise((n, task.max_len, model.cfg.d_model), noise_std, seed)
    resid = model.residual_at(toks, layer, input_noise=noise)  # [n, L, d]
    B = resid.shape[0]
    h = resid[torch.arange(B), ans_pos]  # [n, d]
    logits = model(toks, input_noise=noise)
    al = answer_logits(logits, ans_pos)
    pred = al.argmax(-1)
    return {
        "h": h, "target": target, "is_retr": is_retr,
        "pred": pred, "correct": (pred == target),
    }


def steering_directions(model, task, layer, noise_std=0.0, **kw):
    """For every legal answer token, the difference-of-means direction toward it.
    Returns dict token -> unit vector (torch [d])."""
    c = collect_ans_residuals(model, task, layer, noise_std=noise_std, **kw)
    h, target = c["h"], c["target"]
    dirs = {}
    answer_tokens = list(range(*task.value_band)) + [data.EVEN, data.ODD]
    for a in answer_tokens:
        mask = (target == a)
        if mask.sum() < 5 or (~mask).sum() < 5:
            continue
        u = h[mask].mean(0) - h[~mask].mean(0)
        nrm = u.norm()
        if nrm > 1e-8:
            dirs[a] = u / nrm
    return dirs, c


def correctness_probe(model, task, layer, n=8000, seed=11, frac_retrieval=0.5,
                      l2=1e-2, iters=300, lr=0.5, noise_std=0.0):
    """Logistic probe predicting P(correct) from the answer-position residual.
    Trained by full-batch gradient descent for determinism. Returns (w, b, mu, sd)
    operating on standardized residuals."""
    c = collect_ans_residuals(model, task, layer, n=n, seed=seed,
                              frac_retrieval=frac_retrieval, noise_std=noise_std)
    X = c["h"].double()
    y = c["correct"].double()
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    Xs = (X - mu) / sd
    w = torch.zeros(Xs.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=lr)
    for _ in range(iters):
        z = Xs @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) \
            + l2 * (w @ w)
        opt.zero_grad(); loss.backward(); opt.step()
    return {
        "w": w.detach(), "b": b.detach(), "mu": mu, "sd": sd,
    }


def probe_score(probe, h: torch.Tensor) -> torch.Tensor:
    """P(correct) for a batch of answer-position residuals [B, d]."""
    Xs = (h.double() - probe["mu"]) / probe["sd"]
    z = Xs @ probe["w"] + probe["b"]
    return torch.sigmoid(z)
