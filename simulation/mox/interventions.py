"""Residual-stream intervention operators.

Three operators, all producing a per-example delta added to ``resid_post`` of a
layer at a chosen position:

- ``dense``  : add alpha * u_a, the full difference-of-means direction (ActAdd).
- ``sparse`` : move along a few SAE decoder columns aligned with u_a (a factored,
               more surgical substrate).
- ``blunt``  : add a random unit direction (the "latent vandalism" baseline).

All three are calibrated to a matched on-target logit increase by the caller, so
that collateral can be compared at equal effect.
"""
from __future__ import annotations

import torch


def edit_at_positions(positions: torch.Tensor, deltas: torch.Tensor):
    """Return an edit(resid[B,L,d]) that adds deltas[B,d] at positions[B]."""
    def _edit(resid):
        B = resid.shape[0]
        out = resid.clone()
        out[torch.arange(B), positions] = out[torch.arange(B), positions] + deltas
        return out
    return _edit


def edit_window(positions: torch.Tensor, deltas: torch.Tensor):
    """Like edit_at_positions but persists the delta at every position <= the
    source position, modelling a steering vector written into the cache that
    later tokens then attend over (the invasive case)."""
    def _edit(resid):
        B, L, d = resid.shape
        out = resid.clone()
        for i in range(B):
            out[i, positions[i]] = out[i, positions[i]] + deltas[i]
        return out
    return _edit


def dense_delta(u: torch.Tensor, alpha: float) -> torch.Tensor:
    """alpha * unit(u). u is [d] or [B, d]."""
    un = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
    return alpha * un


def sparse_delta_from_sae(sae, u: torch.Tensor, alpha: float, k: int = 4):
    """A delta lying in the span of the k SAE decoder columns most aligned with
    direction u, scaled to norm alpha. This is the factored-substrate analogue
    of dense_delta: movement expressible as a few feature edits."""
    dec = sae._unit_dec()                      # [F, d], unit rows
    align = dec @ (u / (u.norm() + 1e-8))      # [F]
    top = torch.topk(align.abs(), k).indices
    coef = align[top]                          # signed alignment
    delta = coef @ dec[top]                     # [d]
    return alpha * delta / (delta.norm() + 1e-8)


def blunt_delta(d: int, alpha: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return alpha * v / (v.norm() + 1e-8)


@torch.no_grad()
def calibrate_alpha(model, toks, ans_pos, layer, h, u_builder, target_token,
                    target_gain=2.0, lo=0.0, hi=20.0, iters=18, input_noise=None):
    """Bisection on alpha so the intended target_token logit rises by
    ~target_gain at the answer position. u_builder(alpha) -> delta[B,d].
    Returns the alpha (scalar) achieving the gain for the batch on average."""
    B = toks.shape[0]
    base = model(toks, input_noise=input_noise)
    base_al = base[torch.arange(B), ans_pos]
    base_t = base_al[torch.arange(B), target_token]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        delta = u_builder(mid)
        edit = edit_at_positions(ans_pos, delta)
        al = model(toks, edit=edit, edit_layer=layer,
                   input_noise=input_noise)[torch.arange(B), ans_pos]
        gain = (al[torch.arange(B), target_token] - base_t).mean().item()
        if gain < target_gain:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
