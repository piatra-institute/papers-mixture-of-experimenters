"""Filesystem anchors for the simulation."""
from __future__ import annotations

import pathlib

SIM_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = SIM_ROOT / "output"
EXPERIMENTS = OUTPUT / "experiments"
CHECKPOINTS = OUTPUT / "checkpoints"

for _d in (OUTPUT, EXPERIMENTS, CHECKPOINTS):
    _d.mkdir(parents=True, exist_ok=True)
