# Brief

Written before research begins. See docs/research-pipeline.md §1.

> **Update 2026-06-04.** The original brief below framed the contribution as
> test-time residual experimentation and its failure modes. Independent research
> showed that mechanism is already an active 2025-2026 area (ATLAS, CAST, sparse
> steering; CCPS, CausalGaze, neighbourhood consistency). The paper was rebuilt
> around a new primitive: the **controlled internal experiment**. The current
> claim: an answer-agnostic intervention on an answer's evidence, paired with a
> matched control, gives a label-free correctness signal (AUROC 0.956 on a toy
> with ground truth, 0.841 on Pythia-160M) far above entropy, margin, and a
> supervised probe; grounding tracks correctness where confidence does not. The
> original brief is kept below for provenance.

## Question

The phrase "mixture of experts" is a fossil: an MoE expert is a routed
feed-forward shard, not a scientist who runs experiments. Take the original
word seriously. An *experimenter* is something that probes, perturbs, and tests
a hypothesis before contributing. What would a "mixture of experimenters" be at
the level of mathematics rather than language, and where does it break? The
sharpest version of the idea: at inference time, fork or perturb the residual
stream, evaluate the counterfactual downstream effect, then select or aggregate
the intervention before generation continues.

## Claim

Test-time residual-stream experimentation is implementable and its behaviour is
governed by two design choices that the framing tends to hide: **the scorer**
and **the substrate**. Concretely, on a small trained transformer with a known
ground truth:

1. An oracle scorer (correctness) shows a large achievable gain from selecting
   among latent interventions, but the deployable proxy scorers (entropy,
   logit-margin) do not capture it and can backfire by raising confidence on
   wrong answers. The bottleneck is the scorer, not the intervention.
2. Interventions chosen in a sparse/factored substrate cause less collateral
   change at matched on-target effect than dense raw-residual interventions, but
   the gap is bounded by autoencoder reconstruction quality.
3. Writing the intervention into the live stream (invasive) helps the immediate
   token and degrades over the horizon; a non-invasive fork-and-select does not.
4. Routing experimenters by expected information-gain-per-cost beats uniform
   activation only when the input population mixes failure types; otherwise the
   router collapses to one experimenter and the mixture buys nothing.

The contribution is a typology and a priority order, in the manner of the P-JEPA
revision: not "a new architecture works" but "here is exactly which design
choices decide whether it can work, with toy evidence for each."

## Kind

formal-model (ships a simulation) — sets `has_simulation: true` and
`claims_target: claim-ledger`.

## Cornerstone literature

- Sparse MoE / routing: Shazeer et al. (2017); Fedus, Zoph & Shazeer (2022,
  Switch).
- Activation intervention at inference: Li et al. (2023, ITI); Turner et al.
  (2023, ActAdd / activation engineering).
- Activation patching / causal tracing: Meng et al. (2022, ROME); Zhang &
  Nanda (2024, patching best practices); Nanda (attribution patching).
- Interpretable features / SAEs: Bricken et al. (2023, monosemanticity);
  Templeton et al. (2024, scaling monosemanticity); Heap et al. (2025, SAEs on
  random transformers, the caution).
- Lenses: nostalgebraist (logit lens); Belrose et al. (2023, tuned lens).
- Latent reasoning: Hao et al. (2024, Coconut); Geiping et al. (2025,
  recurrent-depth).
- Off-manifold / steering limits: the 2026 off-manifold steering result and the
  representation-engineering survey (verify exact cites in research).
- Linear representation hypothesis: Park, Choe & Veitch (2023/2024).
- Toolformer (Schick et al., 2023) and Tree-of-Thoughts (Yao et al., 2023) as
  the outer-loop ancestors the paper distinguishes itself from.
