"""H5 — even a correct scorer drifts off-manifold; a manifold penalty pays for
itself.

Using the same retrieval-only setting as H1 with the oracle scorer (which does
lift accuracy), measure how far the selected intervened states sit from the
natural-activation distribution (Mahalanobis). Then subtract a manifold penalty
from the scorer and sweep its weight.

Preregistered hypothesis:
  (a) the unpenalised oracle selection drifts off-manifold: mean Mahalanobis of
      selected states CI95 strictly above the baseline's;
  (b) there is a penalty weight lambda* > 0 at which the accuracy gain over
      baseline is retained (Δacc CI95 still > 0) while the drift is reduced
      relative to lambda = 0 (Δmahal CI95 < 0).
"""
from __future__ import annotations

import numpy as np
import torch

from mox import data, scoring
from mox.experiments.common import fmt_ci, write_result
from mox.fixtures import LAYER, TASK, build_bundle, eval_noise
from mox.runtime import candidate_deltas_toward_answers, candidate_outputs
from mox.stats import paired_bootstrap_ci

N_EVAL = 4000
EVAL_SEED = 20260605
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def _selected(cand_logits, cand_h, scorer, target, manifold):
    scores = scorer(cand_logits, cand_h=cand_h, target=target)
    pick = scores.argmax(0)
    B = cand_logits.shape[1]
    chosen_logits = cand_logits[pick, torch.arange(B)]
    chosen_h = cand_h[pick, torch.arange(B)]
    correct = (chosen_logits.argmax(-1) == target).float().numpy()
    mahal = manifold.mahalanobis(chosen_h).numpy()
    return correct, mahal, pick.numpy()


def run():
    b = build_bundle(seed=0)
    model, dirs, manifold, alpha = b["model"], b["dirs"], b["manifold"], b["alpha"]
    d = model.cfg.d_model

    toks, ans_pos, target, _ = data.make_batch(TASK, N_EVAL, seed=EVAL_SEED,
                                               frac_retrieval=1.0)
    answer_tokens = list(range(*TASK.value_band))
    B = toks.shape[0]
    names, deltas = candidate_deltas_toward_answers(dirs, answer_tokens, alpha, B, d)
    noise = eval_noise(B, EVAL_SEED)
    cand_logits, cand_h = candidate_outputs(model, toks, ans_pos, LAYER, deltas,
                                            input_noise=noise)

    base_h = cand_h[0]                       # null candidate residual
    base_mahal = manifold.mahalanobis(base_h).numpy()
    base_correct = (cand_logits[0].argmax(-1) == target).float().numpy()

    sweep = {}
    arrays = {}
    for lam in LAMBDAS:
        sc = (scoring.score_oracle if lam == 0.0
              else scoring.penalize(scoring.score_oracle, manifold, lam))
        correct, mahal, pick = _selected(cand_logits, cand_h, sc, target, manifold)
        arrays[lam] = {"correct": correct, "mahal": mahal}
        sweep[f"lambda_{lam}"] = {
            "acc": float(correct.mean()),
            "mean_mahal": float(mahal.mean()),
            "d_acc_vs_base": paired_bootstrap_ci(correct, base_correct, seed=1),
            "frac_null": float((pick == 0).mean()),
        }

    # (a) drift real at lambda=0
    drift_real = paired_bootstrap_ci(arrays[0.0]["mahal"], base_mahal, seed=2)
    a_pass = drift_real["ci_lo"] > 0

    # (b) find lambda* retaining accuracy while cutting drift vs lambda=0
    best = None
    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        d_acc = sweep[f"lambda_{lam}"]["d_acc_vs_base"]
        d_mahal = paired_bootstrap_ci(arrays[lam]["mahal"], arrays[0.0]["mahal"], seed=3)
        if d_acc["ci_lo"] > 0 and d_mahal["ci_hi"] < 0:
            best = {"lambda": lam, "d_acc_vs_base": d_acc,
                    "d_mahal_vs_lam0": d_mahal}
            break
    b_pass = best is not None
    passed = bool(a_pass and b_pass)

    payload = {
        "experiment": "H5_offmanifold",
        "hypothesis": "Oracle-selected interventions drift off the natural "
                      "activation manifold; a Mahalanobis penalty cuts the drift "
                      "while retaining the accuracy gain.",
        "pass_criterion": "(a) lambda=0 selected mean Mahalanobis CI95 above "
                          "baseline; (b) some lambda*>0 with Δacc-vs-base CI95>0 "
                          "and Δmahal-vs-lambda0 CI95<0",
        "passed": passed,
        "baseline_acc": float(base_correct.mean()),
        "baseline_mean_mahal": float(base_mahal.mean()),
        "alpha": alpha,
        "n_eval": N_EVAL,
        "drift_real_vs_baseline": drift_real,
        "lambda_star": best,
        "sweep": sweep,
        "interpretation": "If PASS: off-manifold drift is a real cost even when "
                          "the scorer is correct, and a cheap manifold penalty "
                          "buys most of the gain back at lower drift. This is the "
                          "concrete form of the 'stay on the model's manifold' "
                          "constraint.",
    }
    print(f"  baseline acc={payload['baseline_acc']:.3f} mahal={payload['baseline_mean_mahal']:.3f}")
    for lam in LAMBDAS:
        s = sweep[f"lambda_{lam}"]
        print(f"  lam={lam:<4} acc={s['acc']:.3f} mahal={s['mean_mahal']:.3f} "
              f"dacc={fmt_ci(s['d_acc_vs_base'])}")
    print(f"  drift_real {fmt_ci(drift_real)}  lambda* = {best['lambda'] if best else None}")
    write_result("h5_offmanifold", payload)
    return payload


if __name__ == "__main__":
    run()
