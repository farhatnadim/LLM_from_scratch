"""Golden file of L07 for the C++ test (generate_test.cpp). Run all: uv run tools/make_goldens.py"""

from dataclasses import asdict
from pathlib import Path

import torch

from course.py.io import save, seed
from LLM.L06_model.golden import SMALL, small_model
from LLM.L06_model.model import save_checkpoint
from LLM.L07_generation.generate import filter_probs, generate

CASES = [(0.7, 0, 1.0), (1.0, 20, 1.0), (1.0, 0, 0.9), (0.8, 50, 0.95)]  # (temperature, top_k, top_p)


def make_golden(out: Path) -> None:
    model = small_model(7).eval()
    save_checkpoint(model, out / "model.pt")
    seed(7)
    prompt = torch.randint(0, SMALL.vocab_size, (2, 5))                   # [B, T]
    logits = torch.randn(3, SMALL.vocab_size) * 3                        # [B, V]
    golden = {**asdict(SMALL), "prompt": prompt, "max_new": 12, "greedy": generate(model, prompt, 12),
              "logits": logits, "n_cases": len(CASES)}
    eos = int(golden["greedy"][0, 4])                                     # row 0 writes it at step 5
    golden |= {"eos": eos, "greedy_eos": generate(model, prompt, 12, eos=eos)}
    for i, (t, k, p) in enumerate(CASES):
        golden |= {f"case{i}_temperature": t, f"case{i}_top_k": k, f"case{i}_top_p": p,
                   f"case{i}_probs": filter_probs(logits, t, k, p)}
    save(out / "golden.pt", golden)
