# Hypothesis Results

Durable record of the five preregistered tests: the criterion fixed before the
run, the verdict, and the one-line reason. The machine-readable source is
`simulation/output/experiments/*_verification.json`; this file is the human
summary the next editorial pass starts from.

| # | preregistered criterion | verdict | reason |
|---|---|---|---|
| H1 | oracle Δacc CI95>0; entropy Δconf CI95>0 and Δacc upper ≤0.005; margin Δacc upper ≤0.005 | **PASS** | oracle +0.16; entropy/margin/probe all lose accuracy while gaining confidence |
| H2 | sparse minus dense off-target $L_1$ CI95 upper < 0 | **FAIL** (rejected, opposite direction) | sparse is *less* surgical (+1.72); the $d=64$ stream admits no sparse code ($L_0\approx115$) |
| H3 | invasive Δacc1 CI95>0; invasive Δacc2 CI95<0; fork Δacc2 ≥ invasive | **PASS** | invasive fixes answer-1 (+0.21), contaminates answer-2 (CI<0); fork does not |
| H4 | H4a mixture−best single CI95>0; H4b router→retr>0.9 and mixture−retr CI95∋0; H4c routed−best single CI95>0 | **FAIL** (H4a, H4b confirmed; H4c rejected) | mixture beats best single on mixed inputs and collapses on pure; the entropy router underperforms the best single expert |
| H5 | $\lambda{=}0$ drift CI95>baseline; some $\lambda^\star$ with Δacc CI95>0 and Δdrift CI95<0 | **PASS** | drift $8.1\to22.2$; $\lambda^\star=0.25$ keeps full gain at $0.68\times$ drift |

## Notes for the next pass

- H1 came out stronger than preregistered: the deployable scorers do not merely
  fail to find the gain, they move accuracy below baseline. The learned probe is
  the worst because every strong steer reads as "correct" to it.
- H2 reversed. The honest finding is that a trained dictionary is not
  automatically a sparser, more causal basis; on this model it is not sparse at
  all. The prediction (untested here) is that the comparison reverses at a width
  where features live in superposition (Templeton et al. 2024).
- H3 is a genuine but small effect at this scale (two-query horizon, a few tenths
  of a percent). The mechanism is what matters; the magnitude is a scale artifact.
- H4 is the most informative compound result: the mixture story (H4a, H4b) holds,
  but the router is a scorer in disguise and inherits the H1 pathology (H4c). The
  oracle router recovers the gain, which localizes the failure to the routing
  signal, not the mixture.
- H5 is clean and the most directly actionable: a manifold penalty is cheap
  insurance.
