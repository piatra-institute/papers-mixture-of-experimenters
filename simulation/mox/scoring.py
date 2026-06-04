"""Scorers for selecting among candidate interventions.

Each scorer maps candidate outputs to a score per (candidate, example); the
runtime selects the argmax candidate per example. The distinction the paper
turns on:

- ``oracle``  : the correct-answer logit. Needs ground truth. It is the ceiling,
                not a deployable method.
- ``entropy`` : negative predictive entropy. Deployable, sees no labels.
- ``margin``  : top-1 minus top-2 logit. Deployable.
- ``probe``   : a learned correctness readout. Deployable, label-free at test.

A manifold penalty (Mahalanobis distance from the natural activation
distribution) can be subtracted from any scorer (H5).
"""
from __future__ import annotations

import torch

from mox.probes import probe_score


def _logsoftmax(logits):
    return torch.log_softmax(logits, dim=-1)


def entropy_of(logits):
    p = torch.softmax(logits, dim=-1)
    return -(p * _logsoftmax(logits)).sum(-1)


def score_entropy(cand_logits, cand_h=None, **kw):
    # minimise entropy -> maximise (-entropy)
    return -entropy_of(cand_logits)


def score_margin(cand_logits, cand_h=None, **kw):
    top2 = cand_logits.topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def score_oracle(cand_logits, cand_h=None, target=None, **kw):
    # gather the correct-token logit for each example across candidates
    C, B, V = cand_logits.shape
    tgt = target.view(1, B, 1).expand(C, B, 1)
    return cand_logits.gather(-1, tgt).squeeze(-1)


def make_probe_scorer(probe):
    def _score(cand_logits, cand_h=None, **kw):
        C, B, d = cand_h.shape
        return probe_score(probe, cand_h.reshape(C * B, d)).reshape(C, B)
    return _score


# --- manifold model -------------------------------------------------------
class ManifoldModel:
    """Gaussian fit to natural answer-position residuals; Mahalanobis distance
    is the off-manifold score."""

    def __init__(self, H: torch.Tensor, ridge: float = 1e-1):
        H = H.double()
        self.mu = H.mean(0)
        cov = torch.cov(H.T)
        cov = cov + ridge * torch.eye(cov.shape[0], dtype=torch.float64)
        self.prec = torch.linalg.inv(cov)

    def mahalanobis(self, h: torch.Tensor) -> torch.Tensor:
        x = h.double() - self.mu
        m = (x @ self.prec * x).sum(-1)
        return torch.sqrt(torch.clamp(m, min=0.0))


def penalize(base_scorer, manifold: ManifoldModel, lam: float):
    def _score(cand_logits, cand_h=None, **kw):
        base = base_scorer(cand_logits, cand_h=cand_h, **kw)
        C, B, d = cand_h.shape
        pen = manifold.mahalanobis(cand_h.reshape(C * B, d)).reshape(C, B).float()
        return base - lam * pen
    return _score
