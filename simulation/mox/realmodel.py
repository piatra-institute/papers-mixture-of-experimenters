"""Controlled internal experiments on a real pretrained model (Pythia-160M).

The toy establishes the mechanism with an oracle and known ground truth; this
module confirms that the same controlled experiment, run on a real model on an
in-context retrieval task it solves imperfectly, yields a label-free correctness
signal that beats entropy. The model is used only for inference (CPU).

Task: a list of single-token key/value pairs followed by a repeated query key,
e.g. " key: river val: red ; key: stone val: blue ; ... ; key: river val:". The
model completes the value bound to the queried key. The controlled experiment
edits the queried binding (target) versus a non-queried binding (control) at the
token level and measures whether the completion tracks the edit.
"""
from __future__ import annotations

import numpy as np
import torch

_MODEL = None
_TOK = None
MODEL_NAME = "EleutherAI/pythia-160m"

KEY_WORDS = ["river", "stone", "apple", "tiger", "ocean", "forest", "planet",
             "copper", "silver", "garden", "window", "engine", "castle",
             "meadow", "harbor", "desert", "valley", "bridge", "anchor", "candle"]
VALUE_WORDS = ["red", "blue", "green", "gold", "black", "white", "pink", "gray"]


def load():
    global _MODEL, _TOK
    if _MODEL is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        _MODEL.eval()
    return _MODEL, _TOK


def _single_token_ids(tok, words):
    out = []
    for w in words:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            out.append(ids[0])
    return out


def build_dataset(n_examples=400, n_pairs=6, seed=0):
    _, tok = load()
    rng = np.random.default_rng(seed)
    key_ids = _single_token_ids(tok, KEY_WORDS)
    val_ids = _single_token_ids(tok, VALUE_WORDS)
    kmark = tok.encode(" key:", add_special_tokens=False)
    vmark = tok.encode(" val:", add_special_tokens=False)
    sep = tok.encode(" ;", add_special_tokens=False)

    seqs, qpos, cpos, targets = [], [], [], []
    for _ in range(n_examples):
        ks = rng.choice(len(key_ids), size=n_pairs, replace=False)
        vs = rng.integers(0, len(val_ids), size=n_pairs)
        j = int(rng.integers(0, n_pairs))
        jc = int(rng.choice([x for x in range(n_pairs) if x != j]))
        ids, vpos = [], []
        for p in range(n_pairs):
            ids += kmark + [key_ids[ks[p]]] + vmark
            vpos.append(len(ids)); ids += [val_ids[vs[p]]]
            ids += sep
        ids += kmark + [key_ids[ks[j]]] + vmark   # the query
        seqs.append(ids)
        qpos.append(vpos[j])
        cpos.append(vpos[jc])
        targets.append(val_ids[vs[j]])
    return {
        "ids": torch.tensor(seqs),                 # [N, L], equal length
        "qpos": torch.tensor(qpos),                # queried value-token index
        "cpos": torch.tensor(cpos),                # control value-token index
        "targets": torch.tensor(targets),          # correct value token id
        "val_ids": torch.tensor(val_ids),          # the value vocabulary
        "all_valpos": torch.tensor(vpos),          # all value-token indices (fixed layout)
    }


@torch.no_grad()
def attribute_evidence(ids, all_valpos):
    """Identify the value token the answer depends on, with no knowledge of the
    queried binding, by causal leave-one-out: ablate each value token in turn
    (overwriting it with its own key id) and pick the one whose removal most
    reduces the answer's probability. Forward passes only; no attention weights.
    This is the label-free stand-in for the known evidence position. Attribution
    is by ablation; the generalization experiment then tests with a *swap*, a
    different operator, so the test is not circular."""
    model, _ = load()
    N = ids.shape[0]
    ar = torch.arange(N)
    # baseline answer prob
    logits0, _ = _model_value_probs(ids)
    yhat = logits0.argmax(-1)
    p0 = logits0[ar, yhat]
    drops = []
    for p in all_valpos.tolist():
        edit = ids.clone()
        edit[ar, p] = ids[ar, p - 2]                   # overwrite value with its key id
        lp, _ = _model_value_probs(edit)
        drops.append((p0 - lp[ar, yhat]).numpy())
    drops = np.stack(drops, 1)                          # [N, n_pairs]
    best = drops.argmax(1)
    return all_valpos[torch.tensor(best)]


@torch.no_grad()
def _model_value_probs(ids, batch=64):
    """Full next-token probabilities at the final position, batched."""
    model, _ = load()
    outs = []
    for s in range(0, ids.shape[0], batch):
        outs.append(torch.log_softmax(model(ids[s:s + batch]).logits[:, -1, :], dim=-1))
    return torch.cat(outs, 0).exp(), None


@torch.no_grad()
def _answer_probs(ids, val_ids, batch=64):
    """P over the value vocabulary at the final position, for each example."""
    model, _ = load()
    outs = []
    for s in range(0, ids.shape[0], batch):
        chunk = ids[s:s + batch]
        logits = model(chunk).logits[:, -1, :]      # next-token logits
        outs.append(torch.log_softmax(logits, dim=-1))
    logp = torch.cat(outs, 0)
    full = logp.exp()
    val_p = full[:, val_ids]                          # [N, |V|] over value tokens
    return full, val_p


def _edit(ids, pos, new_val):
    out = ids.clone()
    out[torch.arange(ids.shape[0]), pos] = new_val
    return out


@torch.no_grad()
def run_experiments(n_examples=400, n_pairs=6, seed=0, evidence="oracle"):
    """evidence='oracle' uses the known queried-binding position; 'attribution'
    identifies the evidence by causal leave-one-out (the generalization setting),
    using no knowledge of which binding was queried."""
    data = build_dataset(n_examples, n_pairs, seed)
    ids, qpos, cpos = data["ids"], data["qpos"], data["cpos"]
    targets, val_ids = data["targets"], data["val_ids"]
    N = ids.shape[0]
    ar = torch.arange(N)

    attribution_acc = None
    if evidence == "attribution":
        valpos = data["all_valpos"]
        qpos = attribute_evidence(ids, valpos)
        attribution_acc = float((qpos == data["qpos"]).float().mean())
        # control: the attended position's neighbour in the value list
        order = {int(p): k for k, p in enumerate(valpos.tolist())}
        nxt = torch.tensor([int(valpos[(order[int(p)] + 1) % len(valpos)]) for p in qpos])
        cpos = nxt

    full0, valp0 = _answer_probs(ids, val_ids)
    yhat_idx = valp0.argmax(-1)                        # index into val_ids
    yhat = val_ids[yhat_idx]
    correct = (yhat == targets).numpy().astype(int)
    p0_yhat = full0[ar, yhat]

    # counterfactual value v': the value token after the current answer, cyclic
    vprime_idx = (yhat_idx + 1) % len(val_ids)
    vprime = val_ids[vprime_idx]

    # swap experiment
    full_st, _ = _answer_probs(_edit(ids, qpos, vprime), val_ids)
    full_sc, _ = _answer_probs(_edit(ids, cpos, vprime), val_ids)
    follow_t = (full_st[ar, vprime] - full0[ar, vprime])
    follow_c = (full_sc[ar, vprime] - full0[ar, vprime])
    g_swap = (follow_t - follow_c).numpy()
    swap_unc = follow_t.numpy()

    # ablation experiment: overwrite the value with the query-key id (a content-free
    # corruption); the answer should collapse if it depends on that binding
    qkey = ids[ar, qpos - 2]                           # the key token sits two before its value
    full_at, _ = _answer_probs(_edit(ids, qpos, qkey), val_ids)
    full_ac, _ = _answer_probs(_edit(ids, cpos, ids[ar, cpos - 2]), val_ids)
    drop_t = (p0_yhat - full_at[ar, yhat])
    drop_c = (p0_yhat - full_ac[ar, yhat])
    g_abl = (drop_t - drop_c).numpy()
    abl_unc = drop_t.numpy()

    def _z(x):
        x = np.asarray(x, float)
        return (x - x.mean()) / (x.std() + 1e-8)
    g_mix = _z(g_swap) + _z(g_abl)

    logp0 = torch.log(valp0 + 1e-12)
    pv = valp0 / valp0.sum(-1, keepdim=True)
    entropy = -(pv * torch.log(pv + 1e-12)).sum(-1)
    top2 = valp0.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1])

    return {
        "correct": correct,
        "n": int(N),
        "n_pairs": n_pairs,
        "evidence": evidence,
        "attribution_acc": attribution_acc,
        "baseline_acc": float(correct.mean()),
        "signals": {
            "g_swap": g_swap, "g_ablate": g_abl, "g_mixture": g_mix,
            "swap_uncontrolled": swap_unc, "ablate_uncontrolled": abl_unc,
            "neg_entropy": (-entropy).numpy(), "margin": margin.numpy(),
        },
    }
