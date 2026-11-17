"""Golden file of L05 for the C++ test (attention_test.cpp). Run all: uv run tools/make_goldens.py"""

from pathlib import Path

import torch
from torch import nn

from course.py.io import save, seed
from LLM.L04_rope.rope import rope_tables
from LLM.L05_attention.attention import Attention, attend_manual, attend_sdpa, causal_mask

DIM, HEADS, KV_HEADS, HEAD_DIM, T = 64, 4, 2, 16, 7


def make_golden(out: Path) -> None:
    seed(5)
    q, k, v = torch.randn(2, HEADS, T, HEAD_DIM), torch.randn(2, KV_HEADS, T, HEAD_DIM), torch.randn(2, KV_HEADS, T, HEAD_DIM)
    q3 = torch.randn(2, HEADS, 3, HEAD_DIM)                       # 3 queries, 7 keys: the KV cache case
    attn = Attention(DIM, HEADS, KV_HEADS, HEAD_DIM)
    for norm in (attn.q_norm, attn.k_norm):
        nn.init.uniform_(norm.weight, 0.5, 1.5)
    x = torch.randn(2, T, DIM)                                    # [B, T, dim]
    cos, sin = rope_tables(HEAD_DIM, 32, 1e6)
    with torch.no_grad():
        save(out / "golden.pt", {
            "dim": DIM, "n_heads": HEADS, "n_kv_heads": KV_HEADS, "head_dim": HEAD_DIM, "eps": 1e-6,
            "mask_3_7": causal_mask(3, 7), "q": q, "k": k, "v": v, "q3": q3,
            "manual_out": attend_manual(q, k, v), "manual_out_3": attend_manual(q3, k, v),
            "sdpa_out": attend_sdpa(q, k, v), "sdpa_out_3": attend_sdpa(q3, k, v),
            "weights": dict(attn.state_dict()), "x": x, "cos": cos[:T], "sin": sin[:T],
            "attention_out": attn(x, cos[:T], sin[:T]),
        })
