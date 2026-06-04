"""The test-time experimentation loop: fork the residual, run candidate
interventions, score them, select per example, report.

This is the operational core of "mixture of experimenters" stripped to its
mathematics: a set of candidate latent interventions, a scorer, and a selection
rule. Everything the paper measures is an instance of this loop with a different
candidate set or a different scorer.
"""
from __future__ import annotations

import numpy as np
import torch

from mox.interventions import edit_at_positions
from mox.model import answer_logits


@torch.no_grad()
def candidate_outputs(model, toks, ans_pos, layer, deltas_list, input_noise=None):
    """deltas_list: list of C tensors [B, d] (candidate deltas; include a zero
    delta for the null). Returns (cand_logits [C,B,V], cand_h [C,B,d]).
    cand_h is the answer-position residual after the edit (= h + delta). The same
    ``input_noise`` is shared across all candidates so they differ only by the
    intervention."""
    B = toks.shape[0]
    base_h = model.residual_at(toks, layer, input_noise=input_noise)[torch.arange(B), ans_pos]
    cand_logits, cand_h = [], []
    for delta in deltas_list:
        edit = edit_at_positions(ans_pos, delta)
        logits = model(toks, edit=edit, edit_layer=layer, input_noise=input_noise)
        cand_logits.append(answer_logits(logits, ans_pos))
        cand_h.append(base_h + delta)
    return torch.stack(cand_logits), torch.stack(cand_h)


@torch.no_grad()
def select(cand_logits, cand_h, scorer, target):
    """Pick the argmax-scoring candidate per example. Returns a dict of
    per-example arrays and summary metrics."""
    scores = scorer(cand_logits, cand_h=cand_h, target=target)  # [C,B]
    pick = scores.argmax(0)                                     # [B]
    B = cand_logits.shape[1]
    chosen_logits = cand_logits[pick, torch.arange(B)]          # [B,V]
    p = torch.softmax(chosen_logits, -1)
    pred = chosen_logits.argmax(-1)
    correct = (pred == target)
    conf = p.max(-1).values
    ent = -(p * torch.log_softmax(chosen_logits, -1)).sum(-1)
    return {
        "pick": pick.cpu().numpy(),
        "correct": correct.cpu().numpy().astype(float),
        "conf": conf.cpu().numpy(),
        "entropy": ent.cpu().numpy(),
        "acc": float(correct.float().mean()),
        "mean_conf": float(conf.mean()),
        "mean_entropy": float(ent.mean()),
    }


def candidate_deltas_toward_answers(dirs, answer_tokens, alpha, B, d):
    """Build the candidate set {null} U {alpha * u_a : a in answer_tokens}.
    Returns (names, deltas_list). Missing directions are skipped."""
    names = ["null"]
    deltas = [torch.zeros(B, d)]
    for a in answer_tokens:
        if a in dirs:
            names.append(int(a))
            deltas.append((alpha * dirs[a]).expand(B, d).clone())
    return names, deltas
