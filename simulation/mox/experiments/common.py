"""Shared helpers for writing experiment verification artifacts."""
from __future__ import annotations

import json

import numpy as np

from mox.paths import EXPERIMENTS


def _clean(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def write_result(name: str, payload: dict) -> str:
    path = EXPERIMENTS / f"{name}_verification.json"
    path.write_text(json.dumps(_clean(payload), indent=2))
    verdict = "PASS" if payload.get("passed") else "FAIL"
    print(f"[{name}] {verdict}  -> {path}")
    return str(path)


def fmt_ci(d, places=4):
    return f"{d['mean']:+.{places}f} CI95 [{d['ci_lo']:+.{places}f}, {d['ci_hi']:+.{places}f}]"
