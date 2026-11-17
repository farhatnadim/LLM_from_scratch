"""Golden file of L06 for the C++ test (model_test.cpp). Run all: uv run tools/make_goldens.py

Run as a program, it checks the checkpoint that the C++ test wrote with save_checkpoint (CTest: L06_model_py)."""

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from course.py.io import golden_dir, load, save, seed
from LLM.L06_model.model import Block, Config, Model, init_weights, lm_loss, save_checkpoint

SMALL = Config(vocab_size=600, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, hidden=172, max_len=64)


def small_model(seed_value: int) -> Model:
    """The small model of the golden files: weights from init_weights, RMSNorm weights not 1."""
    seed(seed_value)
    model = Model(SMALL)
    init_weights(model, std=0.1)
    for name, p in model.named_parameters():
        if p.dim() == 1:
            nn.init.uniform_(p, 0.5, 1.5)
    return model


def make_golden(out: Path) -> None:
    model = small_model(6)
    save_checkpoint(model, out / "model.pt")
    ids = torch.randint(0, SMALL.vocab_size, (2, 9))                              # [B, T]
    labels = ids.clone()
    labels[1, 6:] = -100                                                          # padding of sample 1
    block = Block(SMALL)
    x = torch.randn(2, 9, SMALL.dim)
    with torch.no_grad():
        logits = model(ids)
        save(out / "golden.pt", {
            **asdict(SMALL), "ids": ids, "labels": labels,                        # config keys at the top
            "logits": logits, "loss": lm_loss(logits, labels),
            "names": sorted(model.state_dict()), "full_parameters": 63_912_192,
            "block_weights": dict(block.state_dict()), "block_x": x, "block_cos": model.cos[:9],
            "block_sin": model.sin[:9], "block_out": block(x, model.cos[:9], model.sin[:9]),
        })


def check_cpp(out: Path) -> None:
    """C++ to Python: the checkpoint of the C++ test has the names and the values of model.pt."""
    got, expected = load(out / "cpp_model.pt"), load(out / "model.pt")
    assert sorted(got) == sorted(expected), f"names differ: {sorted(set(got) ^ set(expected))}"
    for name, tensor in expected.items():
        assert torch.equal(got[name], tensor), f"{name}: values differ"
    print(f"cpp_model.pt: {len(got)} tensors, equal to model.pt")


if __name__ == "__main__":
    check_cpp(golden_dir("LLM/L06_model"))
