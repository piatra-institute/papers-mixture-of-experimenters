# mixture-of-experimenters

*Mixture of Experimenters: Test-Time Residual-Stream Experimentation and Its
Failure Modes.* Takes the original sense of "expert" in mixture-of-experts
seriously (one who runs experiments) and asks what a mixture of *experimenters*
would be at the level of mathematics: fork or perturb the residual stream at
inference, score the counterfactual, then select. The paper shows on a small
trained transformer that whether this works is decided by two choices the
framing tends to hide, the scorer and the substrate, and reports five
preregistered tests of its failure modes.

## Build

```bash
uv run build.py          # -> paper/PAPER.pdf  (vendored canonical recipe)
```

Requires `pandoc` and `xelatex` on PATH. From the workspace you can also run
`papers build mixture-of-experimenters`.

## Simulation

```bash
cd simulation
uv run python -m mox.cli.verify_all   # runs H1-H5, writes verification JSON
```

Every numeric claim in the paper traces to a
`simulation/output/experiments/*_verification.json` file. See
`simulation/README.md` and `docs/CLAIM_LEDGER.md`.

Part of [piatra-papers](https://github.com/piatra-institute). See the workspace
docs for the research and writing pipelines.
