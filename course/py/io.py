"""Tensor exchange between Python and C++ (docs/specification.md, S-1).

Python to C++: save() writes a dictionary; C++ reads it with torch::pickle_load
(course/cpp/course/golden.hpp). C++ to Python: C++ writes with torch::pickle_save;
load() reads it. torch.load needs one allowed name for files of C++.
"""

import random
from pathlib import Path

import torch
import torch.jit._pickle

ROOT = Path(__file__).resolve().parents[2]  # repository root
GOLDENS = ROOT / "data" / "goldens"


def golden_dir(lesson: str) -> Path:
    """Folder of the golden files of one lesson, for example golden_dir("LLM/L03_rmsnorm_swiglu")."""
    return GOLDENS / lesson


def save(path: str | Path, values: dict) -> None:
    """Writes a dictionary for C++. Values: tensors, int, float, str, lists, and dictionaries of these."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(values, path)


def load(path: str | Path) -> dict:
    """Reads a dictionary that C++ (torch::pickle_save) or save() wrote."""
    with torch.serialization.safe_globals([torch.jit._pickle.restore_type_tag]):
        return torch.load(path)


def seed(value: int) -> None:
    """Sets the seed of the random generators that the references use."""
    random.seed(value)
    torch.manual_seed(value)
