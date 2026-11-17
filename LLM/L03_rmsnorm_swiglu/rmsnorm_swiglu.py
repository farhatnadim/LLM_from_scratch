"""L03 RMSNorm and SwiGLU (docs/specification.md, S-5).

RMSNorm comes before each sub-layer of a block. SwiGLU is the feed-forward sub-layer.
"""

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """y = weight * x / sqrt(mean(x^2) + eps), over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))                          # [dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:                      # x: [..., dim]
        x32 = x.float()                                                      # float32, also under autocast
        y = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # [..., dim]
        return (self.weight * y).type_as(x)                                  # [..., dim]


class FeedForward(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x)). No bias."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)                  # weight [hidden, dim]
        self.up_proj = nn.Linear(dim, hidden, bias=False)                    # weight [hidden, dim]
        self.down_proj = nn.Linear(hidden, dim, bias=False)                  # weight [dim, hidden]

    def forward(self, x: torch.Tensor) -> torch.Tensor:                      # x: [..., dim]
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))   # [..., dim]


if __name__ == "__main__":
    x = torch.randn(2, 5, 768) * 3 + 1                                       # [B, T, dim]
    y = RMSNorm(768)(x)
    print(f"RMS of x {x.pow(2).mean(-1).sqrt().mean():.2f}, RMS of RMSNorm(x) {y.pow(2).mean(-1).sqrt().mean():.2f}")
    ffn = FeedForward(768, 2432)
    print(f"FeedForward: {tuple(ffn(y).shape)}, {sum(p.numel() for p in ffn.parameters())} parameters")
