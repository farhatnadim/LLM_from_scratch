"""Golden file of L04 for the C++ test (rope_test.cpp). Run all: uv run tools/make_goldens.py"""

from pathlib import Path

import torch

from course.py.io import save, seed
from LLM.L04_rope.rope import apply_rope, rope_tables, rotate_half

HEAD_DIM, MAX_LEN, BASE = 96, 64, 1e6


def make_golden(out: Path) -> None:
    seed(4)
    cos, sin = rope_tables(HEAD_DIM, MAX_LEN, BASE)                       # [64, 96] each
    x = torch.randn(2, 4, 10, HEAD_DIM)                                   # [B, H, T, head_dim]
    save(out / "golden.pt", {
        "head_dim": HEAD_DIM, "max_len": MAX_LEN, "base": BASE, "cos": cos, "sin": sin, "x": x,
        "rotated_half": rotate_half(x),
        "rope_out": apply_rope(x, cos[:10], sin[:10]),                    # positions 0 .. 9
        "offset": 37, "rope_out_offset": apply_rope(x, cos[37:47], sin[37:47]),  # positions 37 .. 46 (KV cache)
    })
