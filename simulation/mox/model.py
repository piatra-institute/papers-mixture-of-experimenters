"""A small decoder-only transformer with explicit residual-stream access.

Written from scratch (not nn.TransformerEncoder) so that the residual stream at
each layer can be read, forked, and patched cleanly. The intervention surface is
``resid_post`` of a chosen layer: ``h_{l}`` in the paper's notation. A forward
pass accepts an optional ``edit`` callback applied to that residual, which is how
every experiment perturbs the stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    max_len: int
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 256
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.GELU(),
            nn.Linear(cfg.d_mlp, cfg.d_model),
        )

    def forward(self, x, attn_mask):
        a = self.ln1(x)
        a, _ = self.attn(a, a, a, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.lnf = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def _causal_mask(self, L, device):
        return torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

    def forward(
        self,
        tokens: torch.Tensor,
        edit: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        edit_layer: int = 0,
        return_resid: Optional[int] = None,
        input_noise: Optional[torch.Tensor] = None,
    ):
        """Run the model. If ``edit`` is given, it is applied to resid_post of
        ``edit_layer``. If ``return_resid`` is set to a layer index, the residual
        post that layer is also returned (before any later edit). ``input_noise``
        ([B, L, d]) is added to the embeddings to create an uncertainty regime."""
        B, L = tokens.shape
        device = tokens.device
        pos = torch.arange(L, device=device).unsqueeze(0)
        x = self.tok(tokens) + self.pos(pos)
        if input_noise is not None:
            x = x + input_noise
        mask = self._causal_mask(L, device)
        captured = None
        for li, blk in enumerate(self.blocks):
            x = blk(x, mask)
            if return_resid is not None and li == return_resid:
                captured = x.clone()
            if edit is not None and li == edit_layer:
                x = edit(x)
        logits = self.unembed(self.lnf(x))
        if return_resid is not None:
            return logits, captured
        return logits

    @torch.no_grad()
    def residual_at(self, tokens: torch.Tensor, layer: int,
                    input_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, resid = self.forward(tokens, return_resid=layer, input_noise=input_noise)
        return resid

    @torch.no_grad()
    def logits_from_edit(
        self, tokens: torch.Tensor, edit: Callable, edit_layer: int
    ) -> torch.Tensor:
        return self.forward(tokens, edit=edit, edit_layer=edit_layer)


def answer_logits(logits: torch.Tensor, ans_pos: torch.Tensor) -> torch.Tensor:
    """Gather the logits at each example's answer position -> [B, V]."""
    B = logits.shape[0]
    return logits[torch.arange(B), ans_pos]
