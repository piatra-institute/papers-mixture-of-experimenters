"""H1 — the scorer, not the intervention, is the bottleneck.

Preregistered hypothesis: an oracle scorer (correct-answer logit) selecting
among latent steering interventions raises accuracy well above baseline, while
the deployable label-free scorers (entropy, margin) raise *confidence* without
raising accuracy, the Goodhart signature. A learned correctness probe lands
between the two.

PASS criterion (all must hold):
  (a) oracle Δacc CI95 strictly above 0;
  (b) entropy Δconf CI95 strictly above 0 AND entropy Δacc CI95 not strictly
      above 0 (upper bound <= 0.005);
  (c) margin behaves like entropy on accuracy (Δacc CI95 not strictly above 0).
"""
from __future__ import annotations

import numpy as np
import torch

from mox import data, scoring
from mox.experiments.common import fmt_ci, write_result
from mox.fixtures import LAYER, TASK, build_bundle, eval_noise
from mox.runtime import (candidate_deltas_toward_answers, candidate_outputs,
                         select)
from mox.stats import paired_bootstrap_ci

N_EVAL = 4000
EVAL_SEED = 20260601


def run():
    b = build_bundle(seed=0)
    model, dirs, probe = b["model"], b["dirs"], b["probe"]
    alpha = b["alpha"]

    # retrieval-only evaluation: a single, clean candidate answer band.
    toks, ans_pos, target, _ = data.make_batch(TASK, N_EVAL, seed=EVAL_SEED,
                                               frac_retrieval=1.0)
    answer_tokens = list(range(*TASK.value_band))
    B = toks.shape[0]
    d = model.cfg.d_model
    names, deltas = candidate_deltas_toward_answers(dirs, answer_tokens, alpha, B, d)
    noise = eval_noise(B, EVAL_SEED)
    cand_logits, cand_h = candidate_outputs(model, toks, ans_pos, LAYER, deltas,
                                            input_noise=noise)

    # baseline = null candidate only (index 0)
    base = select(cand_logits[:1], cand_h[:1], scoring.score_oracle, target)

    scorers = {
        "oracle": scoring.score_oracle,
        "entropy": scoring.score_entropy,
        "margin": scoring.score_margin,
        "probe": scoring.make_probe_scorer(probe),
    }
    results = {}
    for nm, sc in scorers.items():
        r = select(cand_logits, cand_h, sc, target)
        dacc = paired_bootstrap_ci(r["correct"], base["correct"], seed=1)
        dconf = paired_bootstrap_ci(r["conf"], base["conf"], seed=2)
        # fraction of selections that were the null (no intervention)
        frac_null = float((r["pick"] == 0).mean())
        results[nm] = {
            "acc": r["acc"], "mean_conf": r["mean_conf"],
            "mean_entropy": r["mean_entropy"], "frac_null_selected": frac_null,
            "d_acc": dacc, "d_conf": dconf,
        }

    oracle_helps = results["oracle"]["d_acc"]["ci_lo"] > 0
    entropy_goodhart = (results["entropy"]["d_conf"]["ci_lo"] > 0
                        and results["entropy"]["d_acc"]["ci_hi"] <= 0.005)
    margin_no_acc = results["margin"]["d_acc"]["ci_hi"] <= 0.005
    passed = bool(oracle_helps and entropy_goodhart and margin_no_acc)

    payload = {
        "experiment": "H1_scorer_goodhart",
        "hypothesis": "An oracle scorer over latent steering interventions lifts "
                      "accuracy; entropy/margin scorers lift confidence without "
                      "accuracy (Goodhart); a learned probe sits between.",
        "pass_criterion": "oracle d_acc CI95>0; entropy d_conf CI95>0 and "
                          "d_acc CI95 upper<=0.005; margin d_acc CI95 upper<=0.005",
        "passed": passed,
        "baseline_acc": base["acc"],
        "baseline_mean_conf": base["mean_conf"],
        "n_candidates": len(names),
        "alpha": alpha,
        "n_eval": N_EVAL,
        "model_eval": b["eval"],
        "by_scorer": results,
        "checks": {
            "oracle_helps": oracle_helps,
            "entropy_goodhart": entropy_goodhart,
            "margin_no_acc_gain": margin_no_acc,
        },
        "interpretation": "If PASS: the bottleneck of test-time residual "
                          "experimentation is the scoring function, not the "
                          "intervention. The achievable gain (oracle) is real but "
                          "unreachable by label-free proxies, which buy confidence "
                          "at no accuracy and so mis-rank candidate interventions.",
    }
    for nm in scorers:
        r = results[nm]
        print(f"  {nm:8s} acc={r['acc']:.3f} conf={r['mean_conf']:.3f} "
              f"H={r['mean_entropy']:.3f} null%={r['frac_null_selected']:.2f} "
              f"dacc={fmt_ci(r['d_acc'])} dconf={fmt_ci(r['d_conf'])}")
    write_result("h1_scorer_goodhart", payload)
    return payload


if __name__ == "__main__":
    run()
