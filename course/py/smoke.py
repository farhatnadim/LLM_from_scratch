"""Smoke test of milestone M0, Python side. The C++ side is course/cpp/tests/smoke_test.cpp.

make_golden(): Python writes py_to_cpp.pt; the C++ test reads it.
main:          Python reads cpp_to_py.pt, which the C++ test wrote, and checks it.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from course.py.io import golden_dir, load, save, seed


def make_golden(out: Path) -> None:
    seed(0)
    x = torch.randn(2, 5, 16)                                                # [B, T, C]
    weight = torch.rand(16) + 0.5                                            # [C]
    rmsnorm = weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    q = torch.randn(2, 4, 5, 8)                                              # [B, H, T, D]
    attention = F.scaled_dot_product_attention(q, q, q, is_causal=True)      # [B, H, T, D]
    save(out / "py_to_cpp.pt", {"x": x, "weight": weight, "rmsnorm": rmsnorm, "q": q, "attention": attention})


def check(out: Path) -> None:
    expected = load(out / "py_to_cpp.pt")
    got = load(out / "cpp_to_py.pt")
    for key in ("rmsnorm", "attention"):
        diff = (got[key] - expected[key]).abs().max().item()
        assert diff <= 1e-6, f"{key}: largest difference {diff:.3e} > 1e-6"
        print(f"{key}: C++ result read by Python, largest difference {diff:.3e}")
    assert int(got["step"]) == 1234, got["step"]


if __name__ == "__main__":
    check(golden_dir("smoke"))
    sys.exit(0)
