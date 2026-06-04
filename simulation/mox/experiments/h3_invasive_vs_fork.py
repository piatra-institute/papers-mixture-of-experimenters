"""H3 — invasive edits contaminate the horizon; forks do not.

Two-query retrieval. An oracle steering edit toward the correct first answer is
applied at an early layer of the first answer slot. The second answer slot
attends over that slot, so a *persisted* edit (invasive, the live-stream / KV
case) can degrade the second answer even as it fixes the first. A *fork*
(non-invasive) uses the edit only to read off the first answer and leaves the
stream the second answer sees untouched.

PASS criterion:
  (a) invasive Δacc on answer-1 CI95 strictly above 0 (it fixes the near token);
  (b) invasive Δacc on answer-2 CI95 strictly below 0 (it contaminates the
      horizon);
  (c) non-invasive Δacc on answer-2 CI95 contains 0 / >= invasive (the fork does
      not contaminate).
"""
from __future__ import annotations

import numpy as np
import torch

from mox import data
from mox.experiments.common import fmt_ci, write_result
from mox.interventions import dense_delta, edit_at_positions
from mox.model import ModelConfig, TinyTransformer, answer_logits
from mox.paths import CHECKPOINTS
from mox.seeds import gen_noise, seed_everything

TASK = data.TaskConfig(n_values=6, n_pairs=3, m_bits=8, max_len=13)
EDIT_LAYER = 0            # leave a downstream block to carry contamination
N_TRAIN = 6000
N_EVAL = 4000
EVAL_SEED = 20260603
NOISE_STD = 0.9          # same uncertainty regime as the main model
SCALE_MULT = 8.0
MQ_LEN = 2 * TASK.n_pairs + 7


def _noise(n, seed):
    return gen_noise((n, MQ_LEN, 64), NOISE_STD, seed)


def _train_mq(seed=0, steps=5000, d_model=64):
    seed_everything(seed)
    L = 2 * TASK.n_pairs + 7
    mc = ModelConfig(vocab_size=TASK.vocab_size, max_len=L, d_model=d_model,
                     n_layers=2, n_heads=4, d_mlp=256)
    model = TinyTransformer(mc)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    model.train()
    for step in range(steps):
        toks, p1, p2, t1, t2, _ = data.make_multiquery_batch(TASK, 256, seed=5_000 + step)
        logits = model(toks)
        loss = (torch.nn.functional.cross_entropy(answer_logits(logits, p1), t1)
                + torch.nn.functional.cross_entropy(answer_logits(logits, p2), t2))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def _dirs_at(model, layer, pos_kind):
    """Difference-of-means steering directions at the first answer slot, built in
    the same noisy regime the evaluation uses."""
    toks, p1, p2, t1, t2, _ = data.make_multiquery_batch(TASK, N_TRAIN, seed=31)
    resid = model.residual_at(toks, layer, input_noise=_noise(N_TRAIN, 31))
    B = toks.shape[0]
    pos = p1 if pos_kind == 1 else p2
    tgt = t1 if pos_kind == 1 else t2
    h = resid[torch.arange(B), pos]
    dirs = {}
    for a in range(*TASK.value_band):
        m = (tgt == a)
        if m.sum() < 5 or (~m).sum() < 5:
            continue
        u = h[m].mean(0) - h[~m].mean(0)
        dirs[a] = u / (u.norm() + 1e-8)
    return dirs


def run():
    ckpt = CHECKPOINTS / "h3_mq_model.pt"
    L = 2 * TASK.n_pairs + 7
    mc = ModelConfig(vocab_size=TASK.vocab_size, max_len=L, d_model=64,
                     n_layers=2, n_heads=4, d_mlp=256)
    model = TinyTransformer(mc)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt)); model.eval()
    else:
        model = _train_mq(); torch.save(model.state_dict(), ckpt)

    with torch.no_grad():
        return _evaluate(model)


@torch.no_grad()
def _evaluate(model):
    dirs = _dirs_at(model, EDIT_LAYER, pos_kind=1)

    toks, p1, p2, t1, t2, _ = data.make_multiquery_batch(TASK, N_EVAL, seed=EVAL_SEED)
    noise_full = _noise(N_EVAL, EVAL_SEED)
    keep = torch.tensor([int(t) in dirs for t in t1.tolist()])
    toks, p1, p2, t1, t2 = toks[keep], p1[keep], p2[keep], t1[keep], t2[keep]
    noise = noise_full[keep]
    B = toks.shape[0]

    # steering scale: multiple of mean residual norm at the edit layer
    base_resid = model.residual_at(toks, EDIT_LAYER, input_noise=noise)
    scale = SCALE_MULT * base_resid[torch.arange(B), p1].norm(dim=-1).mean().item()

    # baseline (noisy regime)
    base_logits = model(toks, input_noise=noise)
    a1_base = (answer_logits(base_logits, p1).argmax(-1) == t1).float()
    a2_base = (answer_logits(base_logits, p2).argmax(-1) == t2).float()

    # oracle steering toward the correct first answer, persisted at slot 1
    delta = torch.stack([scale * dirs[int(t)] for t in t1.tolist()])
    edit = edit_at_positions(p1, delta)
    inv_logits = model(toks, edit=edit, edit_layer=EDIT_LAYER, input_noise=noise)
    a1_inv = (answer_logits(inv_logits, p1).argmax(-1) == t1).float()
    a2_inv = (answer_logits(inv_logits, p2).argmax(-1) == t2).float()

    # non-invasive fork: first answer read under the edit, second answer read
    # from the untouched stream.
    a1_fork = a1_inv
    a2_fork = a2_base

    from mox.stats import paired_bootstrap_ci
    d_a1_inv = paired_bootstrap_ci(a1_inv.numpy(), a1_base.numpy(), seed=1)
    d_a2_inv = paired_bootstrap_ci(a2_inv.numpy(), a2_base.numpy(), seed=2)
    d_a2_fork = paired_bootstrap_ci(a2_fork.numpy(), a2_base.numpy(), seed=3)

    passed = bool(d_a1_inv["ci_lo"] > 0 and d_a2_inv["ci_hi"] < 0
                  and d_a2_fork["ci_hi"] >= d_a2_inv["ci_hi"])

    payload = {
        "experiment": "H3_invasive_vs_fork",
        "hypothesis": "A persisted (invasive) edit fixes the immediate answer and "
                      "degrades the next answer that attends over it; a "
                      "non-invasive fork fixes the immediate answer without "
                      "degrading the horizon.",
        "pass_criterion": "invasive d_acc1 CI95>0; invasive d_acc2 CI95<0; "
                          "fork d_acc2 CI95 upper >= invasive d_acc2 upper",
        "passed": passed,
        "edit_layer": EDIT_LAYER,
        "scale": scale,
        "n_eval": int(B),
        "acc": {
            "baseline_a1": float(a1_base.mean()),
            "baseline_a2": float(a2_base.mean()),
            "invasive_a1": float(a1_inv.mean()),
            "invasive_a2": float(a2_inv.mean()),
            "fork_a1": float(a1_fork.mean()),
            "fork_a2": float(a2_fork.mean()),
        },
        "invasive_d_acc1": d_a1_inv,
        "invasive_d_acc2": d_a2_inv,
        "fork_d_acc2": d_a2_fork,
        "interpretation": "If PASS: writing an intervention into the live stream "
                          "buys the near token at the cost of the horizon, the "
                          "KV-contamination failure; a fork-and-select avoids it. "
                          "This is the case for non-invasive experimentation.",
    }
    print(f"  baseline a1={payload['acc']['baseline_a1']:.3f} a2={payload['acc']['baseline_a2']:.3f}")
    print(f"  invasive a1={payload['acc']['invasive_a1']:.3f} a2={payload['acc']['invasive_a2']:.3f}")
    print(f"  inv d_a1 {fmt_ci(d_a1_inv)} | inv d_a2 {fmt_ci(d_a2_inv)} | fork d_a2 {fmt_ci(d_a2_fork)}")
    write_result("h3_invasive_vs_fork", payload)
    return payload


if __name__ == "__main__":
    run()
