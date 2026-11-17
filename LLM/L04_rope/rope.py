"""L04 RoPE: rotary position embedding (docs/specification.md, S-5).

The two halves of a head vector are the two coordinates of head_dim/2 points. RoPE turns
point i of the vector at position p by the angle p * base^(-2i/head_dim). Then the dot
product of a query and a key depends only on the distance between their positions.
"""

import torch


def rope_tables(head_dim: int, max_len: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of the angles of positions 0 .. max_len-1: two tensors [max_len, head_dim]."""
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)   # [head_dim/2]
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)                # [max_len, head_dim/2]
    angles = torch.cat([angles, angles], dim=-1)                                 # [max_len, head_dim]
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1), with x1 and x2 the two halves of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)                                                  # [..., head_dim/2] each
    return torch.cat([-x2, x1], dim=-1)                                          # [..., head_dim]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, head_dim]; cos, sin: [T, head_dim], the rows of the positions of x."""
    return x * cos + rotate_half(x) * sin                                        # [B, H, T, head_dim]


if __name__ == "__main__":
    cos, sin = rope_tables(96, 2048, 1e6)
    q, k = torch.randn(1, 1, 1, 96), torch.randn(1, 1, 1, 96)
    for m, n in [(3, 1), (103, 101), (1003, 1001)]:   # same distance 2: same dot product
        qm, kn = apply_rope(q, cos[m:m + 1], sin[m:m + 1]), apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        print(f"positions {m:4d} and {n:4d}: q.k = {(qm * kn).sum():.5f}")
