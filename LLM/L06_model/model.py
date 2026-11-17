"""L06 Block, model, loss (docs/specification.md, S-5).

Embedding -> n_layers blocks -> RMSNorm -> logits. The head uses the weight of the
embedding (tied). Each block: x + attention(norm(x)), then x + feed-forward(norm(x)).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from course.py.io import load, save
from LLM.L03_rmsnorm_swiglu.rmsnorm_swiglu import FeedForward, RMSNorm
from LLM.L04_rope.rope import rope_tables
from LLM.L05_attention.attention import Attention, LayerCache


@dataclass
class Config:
    vocab_size: int = 6400
    dim: int = 768
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 96
    hidden: int = 2432        # feed-forward width: ceil(768 * pi / 64) * 64
    eps: float = 1e-6
    rope_base: float = 1e6
    max_len: int = 2048       # rows of the RoPE tables: the longest sequence


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.input_layernorm = RMSNorm(c.dim, c.eps)
        self.self_attn = Attention(c.dim, c.n_heads, c.n_kv_heads, c.head_dim, c.eps)
        self.post_attention_layernorm = RMSNorm(c.dim, c.eps)
        self.mlp = FeedForward(c.dim, c.hidden)

    def forward(self, x, cos, sin, cache: LayerCache | None = None) -> torch.Tensor:   # x: [B, T, dim]
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache)              # [B, T, dim]
        return x + self.mlp(self.post_attention_layernorm(x))                         # [B, T, dim]


class Model(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.dim)                          # [V, dim]
        self.layers = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.dim, c.eps)
        cos, sin = rope_tables(c.head_dim, c.max_len, c.rope_base)                     # [max_len, D] each
        self.register_buffer("cos", cos, persistent=False)                             # not in the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids: torch.Tensor, cache: list[LayerCache] | None = None) -> torch.Tensor:
        """ids: [B, T] -> logits [B, T, V]. With a cache (L07), ids continue the cached positions."""
        start = 0 if cache is None or cache[0].k is None else cache[0].k.size(2)
        T = ids.size(1)
        cos, sin = self.cos[start:start + T], self.sin[start:start + T]                # [T, D] each
        x = self.embed_tokens(ids)                                                     # [B, T, dim]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, None if cache is None else cache[i])
        return F.linear(self.norm(x), self.embed_tokens.weight)                        # [B, T, V]


def lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of position t against the label of t+1. Labels of -100 do not count."""
    V = logits.size(-1)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)


def init_weights(model: Model, std: float = 0.02) -> None:
    """Each matrix (embedding, projections) from N(0, std^2). The RMSNorm weights stay 1."""
    for p in model.parameters():
        if p.dim() == 2:
            nn.init.normal_(p, mean=0.0, std=std)


def save_checkpoint(model: Model, path) -> None:
    """A dictionary: parameter name -> tensor (docs/specification.md, S-4)."""
    save(path, dict(model.state_dict()))


def load_checkpoint(model: Model, path) -> None:
    model.load_state_dict(load(path))


if __name__ == "__main__":
    model = Model(Config())
    init_weights(model)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")      # 63,912,192
    ids = torch.randint(0, 6400, (2, 16))
    logits = model(ids)
    print(f"logits {tuple(logits.shape)}; loss {lm_loss(logits, ids):.3f} (random: ln 6400 = 8.764)")
