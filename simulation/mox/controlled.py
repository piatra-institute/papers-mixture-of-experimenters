"""Controlled internal experiments.

A controlled experiment pairs a *targeted* intervention on the evidence an answer
should depend on with a *matched control* intervention on evidence that should be
irrelevant. The control cancels the model's generic sensitivity to perturbation,
which is what an uncontrolled probe (a single intervention, as in perturbation-
stability confidence methods) cannot do. The control-corrected differential is a
label-free signal of whether the answer is causally grounded in its evidence.

Two experiment families on the in-context retrieval task:

- swap: replace the value bound to the queried key with a fresh token v'. Under
  the hypothesis "the model is correctly retrieving", the answer should become
  v'. The control swaps a non-queried pair's value, where the answer should not
  move. Signal: how much more the answer follows the queried swap than the
  control swap.
- ablate: corrupt the value bound to the queried key. A grounded answer's own
  probability should collapse. The control corrupts a non-queried value, which a
  grounded answer should ignore. Signal: the control-corrected probability drop.

All signals are computed without the ground-truth answer; the labels are used
only to score the signals afterward.
"""
from __future__ import annotations

import numpy as np
import torch

from mox import data
from mox.model import answer_logits


def _probs(model, toks, ans_pos, noise):
    logits = model(toks, input_noise=noise)
    al = answer_logits(logits, ans_pos)
    return torch.softmax(al, dim=-1)


def _set_token(toks, pos, value):
    out = toks.clone()
    B = toks.shape[0]
    out[torch.arange(B), pos] = value
    return out


def _permute_pairs(toks, task, seed):
    """Reorder the key-value pair blocks within each retrieval sequence, keeping
    every binding intact. A grounded answer is invariant to pair order; a
    position- or recency-biased guess is not. Layout: [TASK_R, (k v) x n, QUERY,
    q, ANS]; pair p occupies indices 1+2p and 2+2p."""
    rng = np.random.default_rng(seed)
    out = toks.clone()
    n = task.n_pairs
    B = toks.shape[0]
    for i in range(B):
        perm = rng.permutation(n)
        block = toks[i, 1:1 + 2 * n].view(n, 2)
        out[i, 1:1 + 2 * n] = block[perm].reshape(-1)
    return out


@torch.no_grad()
def run_experiments(model, task, noise_std, layer, n=4000, seed=42):
    from mox.seeds import gen_noise
    meta = data.make_retrieval_with_meta(task, n, seed)
    toks, ans_pos = meta["tokens"], meta["ans_pos"]
    target, qval_pos, ctrl_pos = meta["target"], meta["qval_pos"], meta["ctrl_pos"]
    B = toks.shape[0]
    noise = gen_noise((n, task.max_len, model.cfg.d_model), noise_std, seed)
    h0 = model.residual_at(toks, layer, input_noise=noise)[torch.arange(B), ans_pos]

    p0 = _probs(model, toks, ans_pos, noise)              # [B, V]
    yhat = p0.argmax(-1)                                   # predicted answer
    correct = (yhat == target)
    ar = torch.arange(B)
    p0_yhat = p0[ar, yhat]

    vlo, vhi = task.value_band
    # a fresh counterfactual value v', distinct from the current answer
    vprime = vlo + ((yhat - vlo + 1) % (vhi - vlo))        # next value token, cyclic

    # --- swap experiment ---
    toks_st = _set_token(toks, qval_pos, vprime)           # swap the queried binding
    toks_sc = _set_token(toks, ctrl_pos, vprime)           # swap a control binding
    p_st = _probs(model, toks_st, ans_pos, noise)
    p_sc = _probs(model, toks_sc, ans_pos, noise)
    follow_t = p_st[ar, vprime] - p0[ar, vprime]           # answer follows queried swap
    follow_c = p_sc[ar, vprime] - p0[ar, vprime]           # answer follows control swap
    g_swap = (follow_t - follow_c)                         # control-corrected
    g_swap_uncontrolled = follow_t                         # the single-intervention probe

    # --- ablation experiment ---
    pad = torch.full((B,), data.PAD)
    toks_at = _set_token(toks, qval_pos, pad)
    toks_ac = _set_token(toks, ctrl_pos, pad)
    p_at = _probs(model, toks_at, ans_pos, noise)
    p_ac = _probs(model, toks_ac, ans_pos, noise)
    drop_t = p0_yhat - p_at[ar, yhat]                      # answer collapses when its evidence is gone
    drop_c = p0_yhat - p_ac[ar, yhat]
    g_abl = (drop_t - drop_c)
    g_abl_uncontrolled = drop_t

    # --- invariance experiment ---
    # A grounded answer is invariant to reordering the (intact) pair blocks. The
    # signal is high when the answer's probability barely moves under the
    # permutation; it is a "should-not-move" test, complementary to swap/ablate.
    toks_perm = _permute_pairs(toks, task, seed + 1)
    p_perm = _probs(model, toks_perm, ans_pos, noise)
    g_inv = -(p0_yhat - p_perm[ar, yhat]).abs()        # higher (closer to 0) = invariant

    # mixtures of the controlled experiments (z-scored sums, label-free)
    def _z(x):
        x = x.double()
        return (x - x.mean()) / (x.std() + 1e-8)
    g_mix = _z(g_swap) + _z(g_abl)
    g_mix3 = _z(g_swap) + _z(g_abl) + _z(g_inv)

    # label-free baselines on the unperturbed distribution
    logp0 = torch.log(p0 + 1e-12)
    entropy = -(p0 * logp0).sum(-1)
    top2 = p0.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]

    return {
        "correct": correct.numpy().astype(int),
        "n": int(B),
        "baseline_acc": float(correct.float().mean()),
        "h0": h0,
        "signals": {
            # controlled experiments (higher = more grounded = more trustworthy)
            "g_swap": g_swap.numpy(),
            "g_ablate": g_abl.numpy(),
            "g_invariance": g_inv.numpy(),
            "g_mixture": g_mix.numpy(),
            "g_mixture3": g_mix3.numpy(),
            # uncontrolled single-intervention probes (no control)
            "swap_uncontrolled": g_swap_uncontrolled.numpy(),
            "ablate_uncontrolled": g_abl_uncontrolled.numpy(),
            # label-free confidence baselines (higher conf = lower entropy)
            "neg_entropy": (-entropy).numpy(),
            "margin": margin.numpy(),
        },
    }
