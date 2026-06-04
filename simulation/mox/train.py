"""Train the toy transformer on the mixed task and cache the checkpoint.

The model is deliberately under-capacity so that it lands at intermediate
accuracy: an error population is required for the intervention experiments to
have anything to repair.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mox import data
from mox.model import ModelConfig, TinyTransformer, answer_logits
from mox.seeds import seed_everything


@dataclass
class TrainConfig:
    steps: int = 4000
    batch: int = 256
    lr: float = 3e-3
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 256
    frac_retrieval: float = 0.5


def train_model(task: data.TaskConfig, tc: TrainConfig, seed: int = 0):
    seed_everything(seed)
    mc = ModelConfig(
        vocab_size=task.vocab_size,
        max_len=task.max_len,
        d_model=tc.d_model,
        n_layers=tc.n_layers,
        n_heads=tc.n_heads,
        d_mlp=tc.d_mlp,
    )
    model = TinyTransformer(mc)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=1e-3)
    model.train()
    for step in range(tc.steps):
        toks, ans_pos, target, _ = data.make_batch(
            task, tc.batch, seed=10_000 + step, frac_retrieval=tc.frac_retrieval
        )
        logits = model(toks)
        al = answer_logits(logits, ans_pos)
        loss = F.cross_entropy(al, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model, mc


@torch.no_grad()
def evaluate(model, task: data.TaskConfig, n: int = 4000, seed: int = 99,
             frac_retrieval: float = 0.5, noise_std: float = 0.0):
    from mox.seeds import gen_noise
    toks, ans_pos, target, is_retr = data.make_batch(
        task, n, seed=seed, frac_retrieval=frac_retrieval
    )
    noise = gen_noise((n, task.max_len, model.cfg.d_model), noise_std, seed)
    logits = model(toks, input_noise=noise)
    al = answer_logits(logits, ans_pos)
    pred = al.argmax(-1)
    correct = (pred == target)
    out = {
        "acc": correct.float().mean().item(),
        "acc_retrieval": correct[is_retr].float().mean().item(),
        "acc_parity": correct[~is_retr].float().mean().item(),
        "n_retrieval": int(is_retr.sum()),
        "n_parity": int((~is_retr).sum()),
    }
    return out
