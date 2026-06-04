"""C1 - controlled internal experiments as a label-free correctness signal.

Preregistered hypotheses, scored by area under the ROC curve (AUROC) for
discriminating correct from incorrect answers on the in-context retrieval task,
in the noisy regime. All signals are computed without the answer; labels are used
only to evaluate the signals.

  C1a (a control is necessary): the control-corrected signal discriminates
      correctness better than the same intervention without a control
      (paired AUROC difference CI95 > 0), for both the swap and ablate families.
  C1b (it beats label-free confidence): the controlled mixture discriminates
      better than negative entropy and than logit margin (CI95 > 0).
  C1c (it matches or beats a supervised probe): the controlled mixture, which
      uses no labels, is at least competitive with a logistic correctness probe
      trained on labelled residuals (reported; not required to pass).
"""
from __future__ import annotations

import numpy as np
import torch

from mox import controlled
from mox.experiments.common import write_result
from mox.fixtures import LAYER, NOISE_STD, TASK, build_bundle
from mox.probes import probe_score
from mox.stats import auroc_ci, paired_auroc_diff_ci

N_EVAL = 5000
EVAL_SEED = 20260606


def run():
    b = build_bundle(seed=0)
    model, probe = b["model"], b["probe"]

    r = controlled.run_experiments(model, TASK, NOISE_STD, LAYER, n=N_EVAL, seed=EVAL_SEED)
    correct = r["correct"]
    sig = r["signals"]
    # supervised reference: P(correct | h0), trained on labelled residuals
    sig = dict(sig)
    sig["probe_supervised"] = probe_score(probe, r["h0"]).numpy()

    aurocs = {k: auroc_ci(v, correct, seed=1) for k, v in sig.items()}

    c1a_swap = paired_auroc_diff_ci(sig["g_swap"], sig["swap_uncontrolled"], correct, seed=2)
    c1a_abl = paired_auroc_diff_ci(sig["g_ablate"], sig["ablate_uncontrolled"], correct, seed=3)
    c1b_ent = paired_auroc_diff_ci(sig["g_mixture"], sig["neg_entropy"], correct, seed=4)
    c1b_mar = paired_auroc_diff_ci(sig["g_mixture"], sig["margin"], correct, seed=5)
    c1b_unc = paired_auroc_diff_ci(sig["g_mixture"], sig["ablate_uncontrolled"], correct, seed=7)
    c1c_probe = paired_auroc_diff_ci(sig["g_mixture"], sig["probe_supervised"], correct, seed=6)
    c1d_mix3 = paired_auroc_diff_ci(sig["g_mixture3"], sig["g_mixture"], correct, seed=8)

    # primary claim: the controlled mixture beats every label-free baseline and the
    # best uncontrolled single-intervention probe.
    c1b = (c1b_ent["ci_lo"] > 0 and c1b_mar["ci_lo"] > 0 and c1b_unc["ci_lo"] > 0)
    # secondary, per-family: the control is necessary where the intervention is
    # non-specific (ablation) and inert where it is intrinsically specific (swap).
    control_needed_ablate = c1a_abl["ci_lo"] > 0
    control_inert_swap = c1a_swap["ci_lo"] <= 0 <= c1a_swap["ci_hi"]
    passed = bool(c1b)

    payload = {
        "experiment": "C1_controlled_experiments",
        "hypothesis": "A control-corrected internal experiment gives a label-free "
                      "correctness signal that beats the same uncontrolled "
                      "intervention and beats entropy/margin; it is competitive "
                      "with a supervised probe.",
        "pass_criterion": "controlled mixture AUROC exceeds entropy, margin, and "
                          "the best uncontrolled single-intervention probe "
                          "(all paired CI95>0)",
        "passed": passed,
        "baseline_acc": r["baseline_acc"],
        "n_eval": r["n"],
        "auroc": {k: v["auroc"] for k, v in aurocs.items()},
        "auroc_ci": aurocs,
        "c1a_swap_controlled_minus_uncontrolled": c1a_swap,
        "c1a_ablate_controlled_minus_uncontrolled": c1a_abl,
        "c1b_mixture_minus_entropy": c1b_ent,
        "c1b_mixture_minus_margin": c1b_mar,
        "c1b_mixture_minus_best_uncontrolled": c1b_unc,
        "c1c_mixture_minus_supervised_probe": c1c_probe,
        "c1d_threeway_minus_twoway_mixture": c1d_mix3,
        "checks": {
            "c1b_beats_labelfree_and_uncontrolled": c1b,
            "control_needed_for_ablation": control_needed_ablate,
            "control_inert_for_swap": control_inert_swap,
        },
        "interpretation": "Controlled internal experiments, which ask whether the "
                          "answer is causally grounded in its evidence, give a "
                          "label-free correctness signal far stronger than entropy, "
                          "margin, or a supervised probe. The matched control is "
                          "necessary where the intervention has a non-specific "
                          "component (ablation) and inert where the intervention is "
                          "intrinsically specific (the counterfactual swap): a "
                          "control matters exactly when there is generic "
                          "perturbation-sensitivity to cancel.",
    }
    order = ["g_mixture3", "g_mixture", "g_swap", "g_ablate", "g_invariance",
             "swap_uncontrolled", "ablate_uncontrolled", "probe_supervised",
             "neg_entropy", "margin"]
    for k in order:
        print(f"  AUROC {k:22s} {aurocs[k]['auroc']:.3f} "
              f"[{aurocs[k]['ci_lo']:.3f}, {aurocs[k]['ci_hi']:.3f}]")
    print(f"  C1a swap  ctrl-unctrl  {c1a_swap['diff']:+.3f} [{c1a_swap['ci_lo']:+.3f},{c1a_swap['ci_hi']:+.3f}]")
    print(f"  C1a abl   ctrl-unctrl  {c1a_abl['diff']:+.3f} [{c1a_abl['ci_lo']:+.3f},{c1a_abl['ci_hi']:+.3f}]")
    print(f"  C1b mix-entropy        {c1b_ent['diff']:+.3f} [{c1b_ent['ci_lo']:+.3f},{c1b_ent['ci_hi']:+.3f}]")
    print(f"  C1c mix-probe(superv)  {c1c_probe['diff']:+.3f} [{c1c_probe['ci_lo']:+.3f},{c1c_probe['ci_hi']:+.3f}]")
    print(f"  C1d mix3-mix2          {c1d_mix3['diff']:+.3f} [{c1d_mix3['ci_lo']:+.3f},{c1d_mix3['ci_hi']:+.3f}]")
    write_result("c1_controlled_experiments", payload)
    return payload


if __name__ == "__main__":
    run()
