"""Golden file of L03 for the C++ test (rmsnorm_swiglu_test.cpp). Run all: uv run tools/make_goldens.py"""

from pathlib import Path

import torch
from torch import nn

from course.py.io import save, seed
from LLM.L03_rmsnorm_swiglu.rmsnorm_swiglu import FeedForward, RMSNorm

DIM, HIDDEN, HEAD_DIM = 64, 172, 16


def make_golden(out: Path) -> None:
    seed(3)
    x = torch.randn(2, 5, DIM) * 2 + 0.5                         # [B, T, dim]
    x4 = torch.randn(2, 4, 5, HEAD_DIM)                          # [B, H, T, head_dim]: q_norm in L05 has this shape
    norm, norm4, ffn = RMSNorm(DIM), RMSNorm(HEAD_DIM), FeedForward(DIM, HIDDEN)
    nn.init.uniform_(norm.weight, 0.5, 1.5)                      # not ones, so that the weight counts
    nn.init.uniform_(norm4.weight, 0.5, 1.5)
    with torch.no_grad():
        save(out / "golden.pt", {
            "dim": DIM, "hidden": HIDDEN, "head_dim": HEAD_DIM, "eps": 1e-6, "x": x, "x4": x4,
            "norm_weights": dict(norm.state_dict()), "norm_out": norm(x),
            "norm4_weights": dict(norm4.state_dict()), "norm4_out": norm4(x4),
            "ffn_weights": dict(ffn.state_dict()), "ffn_out": ffn(x),
        })
