"""Synthetic in-context task with two sub-tasks of different failure type.

RETRIEVAL: ``[TASK_R] k1 v1 k2 v2 ... kn vn [QUERY] q [ANS]``. The keys in a
sequence are distinct; the query equals one of them; the target is the value
paired with the queried key. The failure mode is attention/binding: the model
must route the right value to the answer slot.

PARITY: ``[TASK_P] b1 ... bm [ANS]``. The target is EVEN or ODD by the count of
1-bits. The failure mode is aggregation/counting, which a shallow transformer
does poorly.

Both ground truths are known by construction, which is the whole point: it lets
an oracle scorer exist, so the gap between the oracle and the deployable proxy
scorers can be measured (the central result of the paper).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# --- vocabulary -----------------------------------------------------------
PAD = 0
TASK_R = 1
TASK_P = 2
QUERY = 3
ANS = 4
EVEN = 5
ODD = 6
BIT0 = 7
BIT1 = 8
# keys and values occupy disjoint contiguous bands after the specials.
KEY0 = 9


@dataclass(frozen=True)
class TaskConfig:
    n_pairs: int = 4          # retrieval key-value pairs
    m_bits: int = 6           # parity bit count
    n_keys: int = 6           # key alphabet size (>= n_pairs)
    n_values: int = 12        # value alphabet size
    max_len: int = 12         # padded sequence length

    @property
    def key_band(self) -> tuple[int, int]:
        return KEY0, KEY0 + self.n_keys

    @property
    def value_band(self) -> tuple[int, int]:
        lo = KEY0 + self.n_keys
        return lo, lo + self.n_values

    @property
    def vocab_size(self) -> int:
        return KEY0 + self.n_keys + self.n_values

    @property
    def retrieval_len(self) -> int:
        return 2 * self.n_pairs + 4  # TASK_R + 2n + QUERY + q + ANS

    @property
    def parity_len(self) -> int:
        return self.m_bits + 2       # TASK_P + m + ANS


def _make_retrieval(cfg: TaskConfig, rng: np.random.Generator):
    klo, khi = cfg.key_band
    vlo, vhi = cfg.value_band
    keys = rng.choice(np.arange(klo, khi), size=cfg.n_pairs, replace=False)
    values = rng.integers(vlo, vhi, size=cfg.n_pairs)
    j = int(rng.integers(0, cfg.n_pairs))
    q = keys[j]
    target = int(values[j])
    seq = [TASK_R]
    for k, v in zip(keys, values):
        seq += [int(k), int(v)]
    seq += [QUERY, int(q), ANS]
    return seq, target


def _make_parity(cfg: TaskConfig, rng: np.random.Generator):
    bits = rng.integers(0, 2, size=cfg.m_bits)
    target = EVEN if int(bits.sum()) % 2 == 0 else ODD
    seq = [TASK_P] + [BIT1 if b else BIT0 for b in bits] + [ANS]
    return seq, target


def make_batch(cfg: TaskConfig, n: int, seed: int, frac_retrieval: float = 0.5):
    """Return (tokens [n, L] long, ans_pos [n] long, target [n] long,
    is_retrieval [n] bool). Sequences are right-padded with PAD to max_len."""
    rng = np.random.default_rng(seed)
    toks = np.full((n, cfg.max_len), PAD, dtype=np.int64)
    ans_pos = np.zeros(n, dtype=np.int64)
    target = np.zeros(n, dtype=np.int64)
    is_retr = np.zeros(n, dtype=bool)
    for i in range(n):
        if rng.random() < frac_retrieval:
            seq, tgt = _make_retrieval(cfg, rng)
            is_retr[i] = True
        else:
            seq, tgt = _make_parity(cfg, rng)
        L = len(seq)
        toks[i, :L] = seq
        ans_pos[i] = L - 1
        target[i] = tgt
    return (
        torch.from_numpy(toks),
        torch.from_numpy(ans_pos),
        torch.from_numpy(target),
        torch.from_numpy(is_retr),
    )


def make_multiquery_batch(cfg: TaskConfig, n: int, seed: int):
    """Two-query retrieval, for the horizon experiment (H3):
    ``[TASK_R] k1 v1 ... kn vn [QUERY] q1 [ANS] [QUERY] q2 [ANS]``.
    Returns tokens, (pos1,pos2), (tgt1,tgt2). The second answer slot attends
    over the first, so a persisted edit at the first slot can contaminate it."""
    rng = np.random.default_rng(seed)
    L = 2 * cfg.n_pairs + 7  # TASK_R + 2n + (QUERY q ANS) x2
    toks = np.full((n, L), PAD, dtype=np.int64)
    pos1 = np.zeros(n, dtype=np.int64)
    pos2 = np.zeros(n, dtype=np.int64)
    tgt1 = np.zeros(n, dtype=np.int64)
    tgt2 = np.zeros(n, dtype=np.int64)
    klo, khi = cfg.key_band
    vlo, vhi = cfg.value_band
    for i in range(n):
        keys = rng.choice(np.arange(klo, khi), size=cfg.n_pairs, replace=False)
        values = rng.integers(vlo, vhi, size=cfg.n_pairs)
        j1, j2 = rng.integers(0, cfg.n_pairs, size=2)
        seq = [TASK_R]
        for k, v in zip(keys, values):
            seq += [int(k), int(v)]
        seq += [QUERY, int(keys[j1]), ANS, QUERY, int(keys[j2]), ANS]
        toks[i, :len(seq)] = seq
        pos1[i] = len(seq) - 4
        pos2[i] = len(seq) - 1
        tgt1[i] = int(values[j1])
        tgt2[i] = int(values[j2])
    return (torch.from_numpy(toks), torch.from_numpy(pos1),
            torch.from_numpy(pos2), torch.from_numpy(tgt1),
            torch.from_numpy(tgt2), L)


def make_retrieval_with_meta(cfg: TaskConfig, n: int, seed: int):
    """Retrieval batch exposing the evidence layout, for controlled experiments.

    Returns tokens [n, max_len], ans_pos [n], target [n], and per example: the
    index of the queried value token (``qval_pos``, the evidence the answer
    should depend on), a control value token index of a non-queried pair
    (``ctrl_pos``), and the queried key token index (``qkey_pos``)."""
    rng = np.random.default_rng(seed)
    toks = np.full((n, cfg.max_len), PAD, dtype=np.int64)
    ans_pos = np.zeros(n, dtype=np.int64)
    target = np.zeros(n, dtype=np.int64)
    qval_pos = np.zeros(n, dtype=np.int64)
    ctrl_pos = np.zeros(n, dtype=np.int64)
    qkey_pos = np.zeros(n, dtype=np.int64)
    klo, khi = cfg.key_band
    vlo, vhi = cfg.value_band
    for i in range(n):
        keys = rng.choice(np.arange(klo, khi), size=cfg.n_pairs, replace=False)
        values = rng.integers(vlo, vhi, size=cfg.n_pairs)
        j = int(rng.integers(0, cfg.n_pairs))
        jc = int(rng.choice([x for x in range(cfg.n_pairs) if x != j]))
        seq = [TASK_R]
        for k, v in zip(keys, values):
            seq += [int(k), int(v)]
        seq += [QUERY, int(keys[j]), ANS]
        L = len(seq)
        toks[i, :L] = seq
        ans_pos[i] = L - 1
        target[i] = int(values[j])
        qkey_pos[i] = 1 + 2 * j
        qval_pos[i] = 2 + 2 * j
        ctrl_pos[i] = 2 + 2 * jc
    return {
        "tokens": torch.from_numpy(toks),
        "ans_pos": torch.from_numpy(ans_pos),
        "target": torch.from_numpy(target),
        "qval_pos": torch.from_numpy(qval_pos),
        "ctrl_pos": torch.from_numpy(ctrl_pos),
        "qkey_pos": torch.from_numpy(qkey_pos),
    }


def answer_band(cfg: TaskConfig, is_retrieval: bool) -> np.ndarray:
    """The set of legal answer tokens for a sub-task. Retrieval answers are
    value tokens; parity answers are EVEN/ODD. Used to build the candidate
    intervention set."""
    if is_retrieval:
        lo, hi = cfg.value_band
        return np.arange(lo, hi)
    return np.array([EVEN, ODD])
