"""Train (or reuse) the shared toy model and print its evaluation."""
from __future__ import annotations

import json

from mox.fixtures import LAYER, TASK, TRAIN, build_bundle


def main():
    b = build_bundle(seed=0)
    print("task:", TASK.__dict__)
    print("train:", TRAIN.__dict__, "layer:", LAYER)
    print("eval:", json.dumps({k: round(v, 4) if isinstance(v, float) else v
                               for k, v in b["eval"].items()}))
    print("sae:", json.dumps({k: round(v, 4) if isinstance(v, float) else v
                              for k, v in b["sae_stats"].items()}))
    print("steer_alpha:", round(b["alpha"], 4),
          "n_steering_dirs:", len(b["dirs"]))


if __name__ == "__main__":
    main()
