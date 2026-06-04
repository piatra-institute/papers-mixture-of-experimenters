"""Run the paper's preregistered experiments and write a CLAIMS_SUMMARY.

C1 is the toy with known ground truth (an oracle on correctness). C2 is the
real-model confirmation on Pythia-160M and requires `transformers`.
The earlier steering failure-mode experiments (h1..h5) remain in
``mox.experiments`` as background and are not part of the paper's claims.
"""
from __future__ import annotations

from mox.experiments import c1_controlled_experiments
from mox.paths import OUTPUT

CLAIMS = [("C1", c1_controlled_experiments)]


def main():
    rows = []
    for tag, mod in CLAIMS:
        print(f"\n=== {tag}: {mod.__name__} ===")
        p = mod.run()
        rows.append((tag, p["experiment"], p["passed"], p["pass_criterion"]))

    # C2 and C3 need transformers; run them if available.
    try:
        from mox.experiments import c2_realmodel, c3_attribution
        print("\n=== C2: mox.experiments.c2_realmodel ===")
        p = c2_realmodel.run()
        rows.append(("C2", p["experiment"], p["passed"], p["pass_criterion"]))
        print("\n=== C3: mox.experiments.c3_attribution ===")
        p = c3_attribution.run()
        rows.append(("C3", p["experiment"], p["passed"], p["pass_criterion"]))
    except Exception as e:  # missing transformers or model download
        print(f"  [C2/C3 skipped: {type(e).__name__}: {e}]")

    lines = ["# Claims summary (generated)\n",
             "| # | experiment | verdict | preregistered criterion |",
             "|---|---|---|---|"]
    for tag, name, passed, crit in rows:
        lines.append(f"| {tag} | {name} | {'PASS' if passed else 'FAIL'} | {crit} |")
    (OUTPUT / "CLAIMS_SUMMARY.md").write_text("\n".join(lines) + "\n")

    print("\n=== summary ===")
    for tag, name, passed, _ in rows:
        print(f"  {tag}: {'PASS' if passed else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()
