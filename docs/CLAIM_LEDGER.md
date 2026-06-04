# Claim Ledger

The durable reviewer-facing map for the paper's executable claims. Every numeric
claim in `paper/PAPER.md` traces to a row here and to a committed
`simulation/output/experiments/*_verification.json`. Regenerate with the paper's
suite (`mox.cli.verify_all`, which runs C1 and, if `transformers` is present, C2).

The earlier steering failure-mode experiments (`h1`..`h5`) remain in the
repository as exploratory background and are not part of the paper's claims.

## C1 — controlled experiments on the toy (oracle ground truth), PASS

Source: `c1_controlled_experiments_verification.json`. Retrieval-only, embedding
noise $\sigma = 0.9$, $n = 5000$, baseline accuracy $0.834$. Metric: AUROC for
correct-vs-incorrect discrimination; signals computed without labels.

- [x] two-experiment mixture AUROC $0.956$ $[0.949, 0.962]$; three-experiment mixture (with invariance) $0.967$ $[0.961, 0.972]$.
- [x] ablate (controlled) $0.944$; ablate (uncontrolled) $0.928$; swap $0.904$; invariance $0.755$ $[0.733, 0.776]$.
- [x] negative entropy $0.786$ $[0.768, 0.803]$; logit margin $0.784$; supervised probe $0.602$ $[0.580, 0.623]$.
- [x] mixture minus entropy $[+0.150, +0.191]$; minus margin $[+0.152, +0.192]$; minus supervised probe $[+0.332, +0.377]$; minus best uncontrolled probe $[+0.021, +0.036]$.
- [x] three-experiment mixture minus two-experiment mixture AUROC $[+0.007, +0.015]$ (a differently-shaped experiment adds discrimination).
- [x] control necessary for ablation: controlled minus uncontrolled AUROC $[+0.010, +0.022]$.
- [x] control inert for swap: controlled minus uncontrolled AUROC $[-0.004, +0.005]$.

## C2 — controlled experiments on Pythia-160M (real model), PASS

Source: `c2_realmodel_pythia160m_verification.json`. In-context retrieval, six
key-value pairs, $n = 400$, baseline accuracy $0.403$. AUROC, label-free.

- [x] controlled mixture AUROC $0.841$ $[0.800, 0.879]$.
- [x] ablate (controlled) $0.816$; ablate (uncontrolled) $0.811$; swap $0.757$.
- [x] negative entropy $0.719$ $[0.668, 0.770]$; logit margin $0.719$.
- [x] mixture minus entropy AUROC $[+0.067, +0.176]$; minus margin $[+0.069, +0.177]$.
- [x] control marginal over best uncontrolled probe not significant at this $n$: $[-0.011, +0.070]$ (reported, matching the toy's per-family nuance).

## C3 — generalization by attribution on Pythia-160M, PASS (honest)

Source: `c3_attribution_pythia160m_verification.json`. Evidence found by causal
leave-one-out instead of given; swap test on the attributed token.

- [x] attribution accuracy $0.488$ (chance $1/6$).
- [x] attributed swap AUROC $0.683$ $[0.625, 0.739]$ (CI95 lower $> 0.5$: the signal survives self-attribution).
- [x] known-evidence swap $0.757$ $[0.708, 0.805]$; entropy $0.719$.
- [x] attributed minus known swap $[-0.129, -0.020]$ (significant cost of imperfect attribution); attributed minus entropy $[-0.105, +0.030]$ (no longer beats entropy at this attribution accuracy).
- Finding: the bottleneck for generalization is evidence attribution, not the experiment.

## Standing limits

The method assumes the evidence an answer should depend on is identifiable from
the input (true for structured retrieval; for unstructured questions an
attribution step is required first). The oracle in C1 is an analysis instrument
used only to score signals. The real-model demonstration is a single small model
on one task family at intermediate accuracy. No claim is made about thresholding,
calibration, or a benchmark across models, tasks, or scales.
