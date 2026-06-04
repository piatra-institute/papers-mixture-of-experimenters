# mox-sim — Mixture of Experimenters simulation

A small trained transformer plus a test-time residual-stream experimentation
toolkit, used to measure the failure modes of the "mixture of experimenters"
idea on a model whose ground truth is known by construction.

## Quick start

```bash
cd simulation
uv sync                              # or: uv venv && uv pip install numpy torch scipy
uv run python -m mox.cli.train_model # train + cache the toy model, print eval
uv run python -m mox.cli.verify_all  # run every preregistered hypothesis test
```

`verify_all` writes one `output/experiments/<name>_verification.json` per
hypothesis and a `output/CLAIMS_SUMMARY.md` table of verdicts. Individual tests:

```bash
uv run python -m mox.experiments.h1_scorer_goodhart
uv run python -m mox.experiments.h2_substrate_collateral
uv run python -m mox.experiments.h3_invasive_vs_fork
uv run python -m mox.experiments.h4_router
uv run python -m mox.experiments.h5_offmanifold
```

CPU only. The shared model trains in well under a minute; the full suite runs in
a few minutes. Determinism is enforced through `mox.seeds`.

## The task

A synthetic in-context task with two failure types (`mox/data.py`):

- **retrieval**: `[TASK_R] k1 v1 ... kn vn [QUERY] q [ANS]`; the answer is the
  value paired with the queried key (a binding failure mode).
- **parity**: `[TASK_P] b1 ... bm [ANS]`; the answer is EVEN/ODD by 1-bit count
  (an aggregation failure mode).

The model learns both cleanly. Gaussian **embedding noise** (`NOISE_STD` in
`mox/fixtures.py`) then puts it in an uncertainty regime where answers are often
wrong but the latent structure to recover them is still present. That is the
regime in which test-time experimentation is supposed to help.

## The toolkit

| module | what it provides |
|---|---|
| `model.py` | tiny decoder-only transformer with residual-stream read/patch hooks |
| `probes.py` | difference-of-means steering directions; a logistic correctness probe |
| `sae.py` | a small L1 sparse autoencoder over residual activations |
| `interventions.py` | dense / sparse / blunt residual deltas; on-target calibration |
| `scoring.py` | oracle / entropy / margin / probe scorers; a Mahalanobis manifold penalty |
| `runtime.py` | the fork-score-select loop |
| `stats.py` | paired bootstrap confidence intervals |

## The hypotheses

| # | claim | file |
|---|---|---|
| H1 | the scorer, not the intervention, is the bottleneck (oracle helps; entropy/margin Goodhart) | `experiments/h1_scorer_goodhart.py` |
| H2 | a sparse substrate is more surgical than the raw residual at matched on-target effect | `experiments/h2_substrate_collateral.py` |
| H3 | invasive edits help the near token and contaminate the horizon; forks do not | `experiments/h3_invasive_vs_fork.py` |
| H4 | a method-diverse mixture pays off only on mixed failure types; else the router collapses | `experiments/h4_router.py` |
| H5 | even a correct scorer drifts off-manifold; a manifold penalty buys it back | `experiments/h5_offmanifold.py` |

Each `run()` returns a payload with a `passed` boolean against a preregistered
`pass_criterion`. The paper cites these JSON artifacts directly.

## Status

This is a toy. It is at its variance limit and the results are directional
signals about *which design choices decide whether test-time residual
experimentation can work*, not quantitative rankings for any production model.
The limits are stated in the paper (§7).
