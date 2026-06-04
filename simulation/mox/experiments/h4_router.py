"""H4 — a mixture of method-experimenters pays off only on mixed failure types.

Two experimenters defined by method: one steers among value tokens (repairs
retrieval), one steers among EVEN/ODD (repairs parity). Selection within an
experimenter is by oracle throughout, to isolate the routing/mixture question
from the scoring question of H1.

Preregistered hypotheses:
  H4a: on the mixed population the union mixture beats the better single
       experimenter (Δacc CI95 > 0).
  H4b: on a retrieval-only population the entropy router sends almost every
       example to the retrieval experimenter (frac > 0.9) and the mixture gives
       no gain over that single experimenter (Δacc CI95 contains 0) — no free
       lunch without mixed failure types.
  H4c: the deployable entropy router recovers most of the mixture: routed acc
       is above the better single experimenter on the mixed population.
"""
from __future__ import annotations

import numpy as np
import torch

from mox import data, scoring
from mox.experiments.common import fmt_ci, write_result
from mox.fixtures import LAYER, TASK, build_bundle, eval_noise
from mox.runtime import candidate_outputs, candidate_deltas_toward_answers, select
from mox.stats import paired_bootstrap_ci

N_EVAL = 6000
EVAL_SEED = 20260604


def _expert_candidates(dirs, tokens_band, alpha, B, d):
    return candidate_deltas_toward_answers(dirs, tokens_band, alpha, B, d)


@torch.no_grad()
def _oracle_outcome(model, toks, ans_pos, deltas, target, noise):
    cl, ch = candidate_outputs(model, toks, ans_pos, LAYER, deltas, input_noise=noise)
    r = select(cl, ch, scoring.score_oracle, target)
    # also record, per example, the best achievable min-entropy (router signal)
    ent = scoring.entropy_of(cl)              # [C,B]
    best_ent = ent.min(0).values.cpu().numpy()
    return r["correct"], best_ent, r["acc"]


def _run_population(model, dirs, alpha, d, seed, frac_retrieval, tag):
    toks, ans_pos, target, is_retr = data.make_batch(
        TASK, N_EVAL, seed=seed, frac_retrieval=frac_retrieval)
    B = toks.shape[0]
    noise = eval_noise(B, seed)
    val_band = list(range(*TASK.value_band))
    par_band = [data.EVEN, data.ODD]

    _, d_null = candidate_deltas_toward_answers({}, [], alpha, B, d)
    _, d_retr = _expert_candidates(dirs, val_band, alpha, B, d)
    _, d_par = _expert_candidates(dirs, par_band, alpha, B, d)
    _, d_mix = _expert_candidates(dirs, val_band + par_band, alpha, B, d)

    base_c, _, base_acc = _oracle_outcome(model, toks, ans_pos, d_null, target, noise)
    retr_c, retr_ent, retr_acc = _oracle_outcome(model, toks, ans_pos, d_retr, target, noise)
    par_c, par_ent, par_acc = _oracle_outcome(model, toks, ans_pos, d_par, target, noise)
    mix_c, _, mix_acc = _oracle_outcome(model, toks, ans_pos, d_mix, target, noise)

    # deployable entropy router: send each example to the expert whose best
    # candidate reaches lower predictive entropy.
    route_to_retr = retr_ent <= par_ent
    routed_c = np.where(route_to_retr, retr_c, par_c)
    # oracle router: route by true subtask
    is_r = is_retr.numpy()
    oracle_routed_c = np.where(is_r, retr_c, par_c)

    better_single = retr_c if retr_acc >= par_acc else par_c
    return {
        "tag": tag,
        "n": int(B),
        "frac_retrieval_actual": float(is_r.mean()),
        "acc": {"base": base_acc, "expert_retr": retr_acc,
                "expert_par": par_acc, "mixture": mix_acc,
                "routed_entropy": float(routed_c.mean()),
                "routed_oracle": float(oracle_routed_c.mean())},
        "router_frac_to_retr": float(route_to_retr.mean()),
        "router_accuracy_vs_subtask": float((route_to_retr == is_r).mean()),
        "_arrays": {"base": base_c, "retr": retr_c, "par": par_c, "mix": mix_c,
                    "routed": routed_c, "better_single": better_single},
    }


def run():
    b = build_bundle(seed=0)
    model, dirs, alpha = b["model"], b["dirs"], b["alpha"]
    d = model.cfg.d_model

    mixed = _run_population(model, dirs, alpha, d, EVAL_SEED, 0.5, "mixed")
    pure = _run_population(model, dirs, alpha, d, EVAL_SEED + 1, 1.0, "retrieval_only")

    # H4a: mixture > better single on mixed
    a = mixed["_arrays"]
    h4a = paired_bootstrap_ci(a["mix"], a["better_single"], seed=1)
    # H4c: routed > better single on mixed
    h4c = paired_bootstrap_ci(a["routed"], a["better_single"], seed=2)
    # H4b: on pure retrieval, mixture vs single retrieval expert
    p = pure["_arrays"]
    h4b = paired_bootstrap_ci(p["mix"], p["retr"], seed=3)

    h4a_pass = h4a["ci_lo"] > 0
    h4b_pass = (pure["router_frac_to_retr"] > 0.9
                and h4b["ci_lo"] <= 0 <= h4b["ci_hi"])
    h4c_pass = h4c["ci_lo"] > 0
    passed = bool(h4a_pass and h4b_pass and h4c_pass)

    for k in ("_arrays",):
        mixed.pop(k); pure.pop(k)

    payload = {
        "experiment": "H4_router",
        "hypothesis": "A method-diverse mixture beats the best single experimenter "
                      "only on a mixed-failure population; on a pure population the "
                      "router collapses to one experimenter and the mixture adds "
                      "nothing. A deployable entropy router recovers most of the "
                      "mixture gain.",
        "pass_criterion": "H4a mixture-minus-best-single CI95>0 (mixed); "
                          "H4b router_frac_to_retr>0.9 and mixture-minus-retr "
                          "CI95 contains 0 (pure); H4c routed-minus-best-single "
                          "CI95>0 (mixed)",
        "passed": passed,
        "alpha": alpha,
        "mixed": mixed,
        "retrieval_only": pure,
        "h4a_mixture_vs_best_single_mixed": h4a,
        "h4b_mixture_vs_retr_pure": h4b,
        "h4c_routed_vs_best_single_mixed": h4c,
        "checks": {"h4a": h4a_pass, "h4b": h4b_pass, "h4c": h4c_pass},
        "interpretation": "If PASS: routing buys epistemic value only when the "
                          "input population genuinely mixes failure types; a "
                          "single well-matched experimenter is otherwise as good, "
                          "and the router correctly collapses to it.",
    }
    print(f"  mixed acc: {mixed['acc']}")
    print(f"  pure  acc: {pure['acc']}  router->retr {pure['router_frac_to_retr']:.3f}")
    print(f"  H4a mix-vs-best {fmt_ci(h4a)} | H4b mix-vs-retr(pure) {fmt_ci(h4b)} | H4c routed-vs-best {fmt_ci(h4c)}")
    write_result("h4_router", payload)
    return payload


if __name__ == "__main__":
    run()
