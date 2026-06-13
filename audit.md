# Audit

Dated log of editorial passes and verification runs. Newest first.

## 2026-06-13 — voice reform

Voice-reform pass for AI-writing tells. No number, equation, table value, or citation touched.

Syntax warns fixed (all four inline-contrastives rewritten with "rather than" / positive form):
- Abstract: "selected for conditional compute, not an agent that runs experiments" → "…a compute path rather than an agent that runs experiments."
- §2: "it intervenes to test a claim, not to impose one" → "…to test a claim rather than to impose one."
- §5: "evidence attribution, not the experiment: where…" → "evidence attribution rather than the experiment itself: where…"
- §7 (former Limitations): "that identification step, not the experiment, is the main obstacle" → "that identification step is the main obstacle…, and the experiment itself is not."

Structure: folded the bolt-on "7. Limitations" into "6. Discussion" (two paragraphs appended under a "Two boundaries follow from how the construction is built" lead) and renumbered "8. Conclusion" → "7. Conclusion". The formulaic-skeleton advisory now clears. Cross-references to Section 3 and Section 5 are unaffected (those sections kept their numbers).

Density (closing section): replaced pet-vocab "it earns its cost exactly when" → "it repays its cost when". The remaining "exactly when …" uses are substantive scope-conditions the paper proves (a control is required exactly when the intervention is non-specific); left intact.

Lexical-density advisory: before exactly 6, precisely 1, genuinely 1, earns 1; tricolon proxy 17. After exactly 5, precisely 1, genuinely 1, earns 0; tricolon proxy 16.

Verify: voice 0 errors / 0 warns; structure advisory cleared; build OK, 0 missing-character warnings; claim-ledger intact (8 verification files); check => PASS.

## 2026-06-04 — publish

Scope: publication to the institute web surface (writing-pipeline §6-7).

Changes:
  - metadata.yaml: `date: "June 2026"`, `status: published`.
  - `papers sync`: copied `paper/PAPER.pdf` to the web app at
    `public/papers/mixture-of-experimenters.pdf`.
  - `papers web-entry`: added the `ownPapers` object to the web app
    `app/papers/page.tsx` (topics `['computer-science']`, kinds
    `['simulation','formal']`), as the June 2026 entry.
  - tooling `README.md` index: added the paper under June 2026; count 12 papers,
    9 published.

Verification: `papers check` => PASS, including `web OK` (deployed PDF hash
matches the local build). 9 pages, 0 missing-char warnings, voice 0 errors,
claim-ledger 8 verification files.

## 2026-06-04 — scope extensions (invariance experiment + attribution generalization)

Scope: two additions requested after the pivot.

Changes:
  - Third experiment type, **invariance** (reorder the intact pair blocks; a
    grounded answer is stable): `controlled._permute_pairs`, `g_invariance`,
    `g_mixture3`. C1 now reports it.
  - **Attribution generalization** (C3): on Pythia-160M the evidence is found by
    causal leave-one-out (`realmodel.attribute_evidence`) instead of given, then
    the swap test runs on the attributed token. New experiment `c3_attribution`;
    `verify_all` now runs C1, C2, C3. (Attention-based attribution was tried first
    and abandoned: loading Pythia with `attn_implementation="eager"` collapsed CPU
    inference to chance on this build, so attribution is by ablation, forward
    passes only.)

Results:
  - Invariance alone AUROC $0.755$; three-experiment mixture $0.967$ beats the
    two-experiment $0.956$ by $[+0.007, +0.015]$ (the mixture grows with the panel).
  - C3 PASS (honest criterion): attribution accuracy $0.488$ (chance $1/6$);
    attributed swap AUROC $0.683$ $[0.625, 0.739]$ survives self-attribution but
    no longer beats entropy $0.719$ and trails known-evidence swap $0.757$ by
    $[-0.129, -0.020]$. Finding: generalization is bottlenecked by evidence
    attribution, not the experiment. §7 now quantifies the limitation.

Verification: voice 0 errors; build 9 pages, 0 missing-char; claim-ledger 8
verification files; check => PASS. Status remains `draft`.

## 2026-06-04 — pivot to controlled internal experiments

Scope: independent literature research (June 2026) showed the original framing,
test-time residual-stream experimentation as a novel mechanism, is stale. A
2025-2026 wave already does verifier-selected test-time latent steering (ATLAS,
arXiv:2601.03093), conditional steering (CAST), sparse-space steering (Bayat et
al. 2025), and single-shot perturbation/causal/consistency confidence (CCPS
2505.21772, CausalGaze 2604.11087, neighbourhood consistency 2601.05905). On the
user's direction ("find a new idea of experiment"), the paper was rebuilt around
a genuinely new primitive that survived a novelty check: the *controlled internal
experiment*, an answer-agnostic intervention on an answer's evidence paired with
a matched control, yielding a label-free correctness signal.

Changes:
  - New simulation: `mox/controlled.py` (swap and ablate experiments with matched
    controls on the toy), `mox/realmodel.py` (Pythia-160M in-context retrieval),
    AUROC + paired-AUROC bootstrap in `mox/stats.py`,
    `data.make_retrieval_with_meta`. Experiments `c1_controlled_experiments` (toy,
    oracle ground truth) and `c2_realmodel` (real model). `cli/verify_all` now
    runs C1 and C2 as the paper's claims.
  - Paper fully rewritten: controlled experiments as the lead contribution; the
    earlier steering failure-mode results (h1..h5) retained in the repo as
    background but dropped from the paper body. New title and abstract.
  - References rebuilt around the 2025-2026 literature; first authors verified
    against arXiv (Khanmohammadi, Bayat, Kong, Xu, Nguyen) after the earlier draft
    had guessed surnames. CLAIM_LEDGER rewritten for C1/C2.

Results (AUROC, correct-vs-incorrect, label-free; paired bootstrap CI95):
  - C1 PASS (toy, n=5000, acc 0.834): controlled mixture 0.956; entropy 0.786,
    margin 0.784, supervised probe 0.602. Mixture beats entropy [+0.150,+0.191],
    probe [+0.332,+0.377], best uncontrolled probe [+0.021,+0.036]. Control needed
    for ablation [+0.010,+0.022], inert for swap [-0.004,+0.005].
  - C2 PASS (Pythia-160M, n=400, acc 0.403): controlled mixture 0.841; entropy
    0.719. Mixture beats entropy [+0.067,+0.176]; control marginal not significant
    at this n [-0.011,+0.070].

Verification: voice 0 errors (2 review warns); refs advisory (dash bibliography;
all 17 entries cited, none unused, verified by hand); claims claim-ledger present
(7 verification files, incl. legacy h1..h5); build 9 pages, 0 missing-char
warnings; check => PASS. Status remains `draft`.

## 2026-06-04 — initial draft + simulation

Scope: built the paper from `chat.md` (the originating dialogue on "mixture of
experimenters"). Turned the dialogue's seven informal failure modes into five
preregistered, executable hypotheses on a purpose-built toy.

Changes:
  - `simulation/mox/`: a two-layer, 64-d, 4-head decoder-only transformer
    (`model.py`) with residual-stream read/patch hooks; a synthetic two-failure
    in-context task (`data.py`, retrieval + parity); difference-of-means steering
    directions and a logistic correctness probe (`probes.py`); a small L1 SAE
    (`sae.py`); dense/sparse/blunt intervention operators with on-target
    calibration (`interventions.py`); oracle/entropy/margin/probe scorers and a
    Mahalanobis manifold penalty (`scoring.py`); the fork-score-select runtime
    (`runtime.py`); paired bootstrap CIs (`stats.py`); shared fixtures with an
    embedding-noise uncertainty regime (`fixtures.py`).
  - Five experiments H1-H5 (`mox/experiments/`), each emitting a
    `*_verification.json` with a preregistered criterion and a binary verdict;
    `mox.cli.verify_all` aggregates and writes `output/CLAIMS_SUMMARY.md`.
  - `paper/PAPER.md`: abstract, motivating question, the experimenter formalism
    (§2), operators and scorers (§3), the toy (§4), the five tests (§5), the
    typology and priority order (§6), limits (§7), reproducibility (§8), what it
    is not (§9), 17-entry References.
  - `docs/CLAIM_LEDGER.md`, `docs/HYPOTHESIS_RESULTS.md`; `brief.md`,
    `research.md`, `sources.md`; `README.md`, `.gitignore`, vendored `build.py`.

Results (paired bootstrap CI95):
  - H1 PASS: oracle selection lifts retrieval $0.845\to1.000$ ($[+0.144,+0.167]$);
    entropy $0.504$, margin $0.599$, probe $0.167$, all raising confidence
    ($[+0.036,+0.042]$). The scorer is the bottleneck.
  - H2 FAIL (rejected, opposite direction): at matched on-target gain $1.200$,
    sparse off-target $L_1$ exceeds dense by $[+1.62,+1.81]$; SAE $L_0\approx115$
    of $256$ at FVU $\approx0.001$ (no sparse code at $d=64$).
  - H3 PASS: invasive edit fixes answer-1 ($[+0.197,+0.222]$), contaminates
    answer-2 ($[-0.0068,-0.0018]$); fork holds answer-2 at baseline.
  - H4 FAIL (H4a, H4b confirmed; H4c rejected): mixture beats best single on
    mixed inputs ($[+0.080,+0.094]$), collapses on pure; entropy router
    underperforms best single ($[-0.115,-0.088]$); oracle router recovers the gain.
  - H5 PASS: oracle drift $8.1\to22.2$; manifold penalty $\lambda=0.25$ keeps full
    gain at $0.68\times$ drift.

Verification:
  - voice: 0 errors, 8 review-candidates (inline-contrastive + one negate-pivot;
    all are foundational contrasts the paper develops or §9 factual negations,
    triaged keep).
  - refs: advisory (dash-prefixed bibliography not auto-detected, as for
    humanities papers). Manually reconciled: all 17 bib entries are cited in
    prose; no MISSING, no unused.
  - claims: claim-ledger present, 5 verification files, no unchecked rows.
  - build: 11 pages, 0 missing-character warnings; math, tables, code spans,
    running header all render.
  - check => PASS.

Notes: H2 and H4 came out as informative negatives. H2's reversal is a fact
about this model's representation geometry (no sparse code at $d=64$), not a
general claim; the prediction is that it reverses at a width where features live
in superposition. H4's router fails for the same reason H1's scorers do, which
localizes the failure to the routing signal rather than the mixture. Status
remains `draft`; not yet built into the web app or assigned a `date`.
