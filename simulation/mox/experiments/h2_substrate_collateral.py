"""H2 — the substrate decides collateral.

Preregistered hypothesis: at a matched on-target logit gain, an intervention
expressed in the sparse SAE-feature substrate disturbs the rest of the output
distribution less than the same-effect dense raw-residual intervention, and
lands closer to the natural-activation manifold. A blunt random direction cannot
achieve the targeted gain at comparable selectivity and is reported as the
floor.

PASS criterion: sparse off-target L1 CI95 strictly below dense off-target L1
(paired, sparse - dense < 0).
"""
from __future__ import annotations

import numpy as np
import torch

from mox import data
from mox.experiments.common import fmt_ci, write_result
from mox.fixtures import LAYER, TASK, build_bundle, eval_noise
from mox.interventions import (blunt_delta, calibrate_alpha, dense_delta,
                               edit_at_positions, sparse_delta_from_sae)
from mox.model import answer_logits
from mox.stats import mean_ci, paired_bootstrap_ci

N_EVAL = 2500
EVAL_SEED = 20260602
TARGET_GAIN = 1.2     # a gain both operators can reach, so collateral is matched
K_SPARSE = 8
CAL_HI = 40.0


@torch.no_grad()
def _delta_logits(model, toks, ans_pos, deltas, noise):
    base = answer_logits(model(toks, input_noise=noise), ans_pos)
    edit = edit_at_positions(ans_pos, deltas)
    inter = answer_logits(model(toks, edit=edit, edit_layer=LAYER,
                                input_noise=noise), ans_pos)
    return base, inter


def run():
    b = build_bundle(seed=0)
    model, dirs, sae, manifold = b["model"], b["dirs"], b["sae"], b["manifold"]
    d = model.cfg.d_model

    toks, ans_pos, target, _ = data.make_batch(TASK, N_EVAL, seed=EVAL_SEED,
                                               frac_retrieval=1.0)
    noise_full = eval_noise(N_EVAL, EVAL_SEED)
    keep = torch.tensor([int(t) in dirs for t in target.tolist()])
    toks, ans_pos, target = toks[keep], ans_pos[keep], target[keep]
    noise = noise_full[keep]
    B = toks.shape[0]

    # per-example direction toward the correct token
    U = torch.stack([dirs[int(t)] for t in target.tolist()]).detach()  # [B,d], unit
    # unit sparse direction per example (detach: SAE params carry grad)
    with torch.no_grad():
        S = torch.stack([sparse_delta_from_sae(sae, dirs[int(t)], 1.0, k=K_SPARSE)
                         for t in target.tolist()])
        S = S / (S.norm(dim=-1, keepdim=True) + 1e-8)

    def dense_builder(alpha):
        return alpha * U

    def sparse_builder(alpha):
        return alpha * S

    a_dense = calibrate_alpha(model, toks, ans_pos, LAYER, None, dense_builder,
                              target, target_gain=TARGET_GAIN, hi=CAL_HI,
                              iters=24, input_noise=noise)
    a_sparse = calibrate_alpha(model, toks, ans_pos, LAYER, None, sparse_builder,
                               target, target_gain=TARGET_GAIN, hi=CAL_HI,
                               iters=24, input_noise=noise)

    out = {}
    rows = {}
    base_h = model.residual_at(toks, LAYER, input_noise=noise)[torch.arange(B), ans_pos]
    for nm, delta in [("dense", dense_builder(a_dense)),
                      ("sparse", sparse_builder(a_sparse))]:
        base, inter = _delta_logits(model, toks, ans_pos, delta, noise)
        dlog = inter - base
        on_target = dlog[torch.arange(B), target]
        off_l1 = dlog.abs().sum(-1) - on_target.abs()
        selectivity = on_target / (off_l1 + 1e-6)
        mahal = manifold.mahalanobis(base_h + delta) - manifold.mahalanobis(base_h)
        rows[nm] = {
            "on_target": on_target.numpy(),
            "off_l1": off_l1.numpy(),
            "selectivity": selectivity.numpy(),
            "mahal_increase": mahal.numpy(),
        }
        out[nm] = {
            "alpha": float(a_dense if nm == "dense" else a_sparse),
            "on_target_gain": mean_ci(on_target.numpy()),
            "off_target_l1": mean_ci(off_l1.numpy()),
            "selectivity": mean_ci(selectivity.numpy()),
            "mahal_increase": mean_ci(mahal.numpy()),
        }

    # blunt floor: a random direction calibrated to the same norm as dense,
    # reporting how little on-target it achieves.
    g = blunt_delta(d, a_dense, seed=777).expand(B, d)
    bbase, binter = _delta_logits(model, toks, ans_pos, g, noise)
    bdl = binter - bbase
    out["blunt"] = {
        "on_target_gain": mean_ci(bdl[torch.arange(B), target].numpy()),
        "off_target_l1": mean_ci((bdl.abs().sum(-1)
                                  - bdl[torch.arange(B), target].abs()).numpy()),
        "note": "random direction at dense's norm; reported as the non-targeted floor",
    }

    d_offl1 = paired_bootstrap_ci(rows["sparse"]["off_l1"],
                                  rows["dense"]["off_l1"], seed=3)
    d_mahal = paired_bootstrap_ci(rows["sparse"]["mahal_increase"],
                                  rows["dense"]["mahal_increase"], seed=4)
    passed = bool(d_offl1["ci_hi"] < 0)

    payload = {
        "experiment": "H2_substrate_collateral",
        "hypothesis": "At matched on-target gain, sparse-substrate interventions "
                      "have lower off-target collateral and smaller off-manifold "
                      "drift than dense raw-residual interventions.",
        "pass_criterion": "paired (sparse - dense) off-target L1 CI95 upper < 0",
        "passed": passed,
        "target_gain": TARGET_GAIN,
        "k_sparse": K_SPARSE,
        "sae_stats": b["sae_stats"],
        "n_eval": int(B),
        "by_operator": out,
        "sparse_minus_dense_off_l1": d_offl1,
        "sparse_minus_dense_mahal": d_mahal,
        "interpretation": "If PASS: a factored substrate buys surgical locality, "
                          "supporting the 'do not experiment in the raw residual "
                          "stream' design. If FAIL: the SAE is not clean enough to "
                          "beat the raw direction, which is itself the Heap et al. "
                          "(2025) caution about over-trusting SAE locality.",
    }
    for nm in ("dense", "sparse", "blunt"):
        o = out[nm]
        print(f"  {nm:7s} on_target={o['on_target_gain']['mean']:.3f} "
              f"off_l1={o['off_target_l1']['mean']:.3f}")
    print(f"  sparse-dense off_l1 {fmt_ci(d_offl1)}")
    print(f"  sparse-dense mahal  {fmt_ci(d_mahal)}")
    write_result("h2_substrate_collateral", payload)
    return payload


if __name__ == "__main__":
    run()
