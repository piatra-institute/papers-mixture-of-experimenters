"""A tiny sparse autoencoder over residual-stream activations.

The substrate experiment (H2) compares interventions in the raw dense residual
basis against interventions in this learned sparse-feature basis. The SAE is the
standard tied-bias, unit-norm-decoder, L1-penalised construction (Bricken et al.
2023). It is small and trained on CPU in seconds. The point is not a perfect
dictionary; the point is to measure how much surgical locality a realistic
factored substrate buys, and where it stops (Heap et al. 2025).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mox import data
from mox.seeds import gen_noise, seed_everything


class SAE(nn.Module):
    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.randn(d_model, n_features) * 0.1)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.randn(n_features, d_model) * 0.1)

    def encode(self, h):
        return F.relu((h - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, a):
        return a @ self._unit_dec() + self.b_dec

    def _unit_dec(self):
        return self.W_dec / (self.W_dec.norm(dim=1, keepdim=True) + 1e-8)

    def forward(self, h):
        a = self.encode(h)
        return self.decode(a), a


def collect_residuals_all_positions(model, task, layer, n=4000, seed=23,
                                    frac_retrieval=0.5, noise_std=0.0):
    toks, ans_pos, target, is_retr = data.make_batch(
        task, n, seed=seed, frac_retrieval=frac_retrieval
    )
    noise = gen_noise((n, task.max_len, model.cfg.d_model), noise_std, seed)
    resid = model.residual_at(toks, layer, input_noise=noise)  # [n, L, d]
    mask = toks != data.PAD
    H = resid[mask]  # [tokens, d]
    return H


def train_sae(model, task, layer, n_features=256, l1=1.5e-2, steps=3000,
              lr=2e-3, seed=5, n=4000, noise_std=0.0):
    seed_everything(seed)
    H = collect_residuals_all_positions(model, task, layer, n=n, seed=seed,
                                        noise_std=noise_std)
    d = H.shape[1]
    sae = SAE(d, n_features)
    # tied init: decoder bias at the data mean improves conditioning.
    with torch.no_grad():
        sae.b_dec.copy_(H.mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    idx = torch.arange(H.shape[0])
    for step in range(steps):
        sel = idx[torch.randint(0, H.shape[0], (512,))]
        h = H[sel]
        recon, a = sae(h)
        mse = ((recon - h) ** 2).sum(-1).mean()
        l1pen = a.abs().sum(-1).mean()
        loss = mse + l1 * l1pen
        opt.zero_grad(); loss.backward(); opt.step()
    sae.eval()
    with torch.no_grad():
        recon, a = sae(H)
        var = ((H - H.mean(0)) ** 2).sum(-1).mean()
        mse = ((recon - H) ** 2).sum(-1).mean()
        stats = {
            "fvu": (mse / var).item(),                 # fraction of variance unexplained
            "l0": (a > 1e-6).float().sum(-1).mean().item(),
            "n_features": n_features,
        }
    return sae, stats
