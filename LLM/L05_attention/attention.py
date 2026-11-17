"""L05 Attention: causal grouped-query attention (docs/specification.md, S-5).

8 query heads share 4 key-value heads: query head h reads key-value head h // 2. RMSNorm
on each query and key head, then RoPE. Query t of T sees keys 0 .. S-T+t of S (S >= T).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from LLM.L03_rmsnorm_swiglu.rmsnorm_swiglu import RMSNorm
from LLM.L04_rope.rope import apply_rope, rope_tables


def causal_mask(T: int, S: int, device=None) -> torch.Tensor:
    """[T, S], True where query t can see key s: s <= S - T + t."""
    return torch.ones(T, S, dtype=torch.bool, device=device).tril(S - T)


def attend_manual(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q: [B, H, T, D]; k, v: [B, KV, S, D] -> [B, H, T, D]. Each step written out."""
    rep = q.size(1) // k.size(1)                                         # query heads for each key-value head
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)  # [B, H, S, D]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))             # [B, H, T, S]
    scores = scores.masked_fill(~causal_mask(q.size(2), k.size(2), q.device), float("-inf"))
    return scores.softmax(dim=-1) @ v                                    # [B, H, T, D]


def attend_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Same result as attend_manual, with the fused kernel of PyTorch."""
    T, S = q.size(2), k.size(2)
    if T == S:                                                           # training and the first step
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(T, S, q.device), enable_gqa=True)


@dataclass
class LayerCache:
    """L07: keys and values of the positions before, for one layer. [B, KV, S, D] each."""
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(head_dim, eps), RMSNorm(head_dim, eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: LayerCache | None = None) -> torch.Tensor:
        """x: [B, T, dim]; cos, sin: [T, D], the rows of the positions of x -> [B, T, dim]."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)     # [B, H, T, D]
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, KV, T, D]
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, KV, T, D]
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        if cache is not None:                    # L07: the new keys and values go after the old ones
            if cache.k is not None:
                k, v = torch.cat([cache.k, k], dim=2), torch.cat([cache.v, v], dim=2)  # [B, KV, S, D]
            cache.k, cache.v = k, v
        y = attend_sdpa(q, k, v)                                                        # [B, H, T, D]
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))  # [B, T, dim]


if __name__ == "__main__":
    q, k, v = torch.randn(1, 8, 6, 96), torch.randn(1, 4, 6, 96), torch.randn(1, 4, 6, 96)
    diff = (attend_manual(q, k, v) - attend_sdpa(q, k, v)).abs().max()
    print(f"manual and fused attention: largest difference {diff:.1e}")
    attn = Attention(768, 8, 4, 96)
    cos, sin = rope_tables(96, 2048, 1e6)
    x = torch.randn(2, 6, 768)
    print(f"Attention: {tuple(attn(x, cos[:6], sin[:6]).shape)}, {sum(p.numel() for p in attn.parameters())} parameters")
