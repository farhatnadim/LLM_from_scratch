"""Golden file of L02 for the C++ test (token_data_test.cpp). Run all: uv run tools/make_goldens.py"""

from pathlib import Path

import torch

from course.py.io import save
from course.py.jsonl import write_jsonl
from LLM.L01_tokenizer.golden import CORPUS
from LLM.L01_tokenizer.tokenizer import Tokenizer, train_bpe
from LLM.L02_token_data.token_data import TokenFile, make_batch, pretrain_sample, tokenize_jsonl

LONG = " ".join(["The little dog ran to the park and saw a big red ball."] * 4)
TEXTS = [*CORPUS, "", LONG, "Max  ran\tto the café 🙂!"]  # an empty text gives a sample of length 0


def make_golden(out: Path) -> None:
    tok = Tokenizer(train_bpe(CORPUS, 600))
    tok.save(out / "merges.txt")
    write_jsonl(out / "corpus.jsonl", ({"text": t} for t in TEXTS))
    count = tokenize_jsonl(out / "corpus.jsonl", tok, out / "tokens")
    data = TokenFile(out / "tokens")
    short_in, short_labels = pretrain_sample(data.sample(0), 32)             # padding
    long_in, long_labels = pretrain_sample(data.sample(len(CORPUS) + 1), 16)  # cut
    indices = [3, 0, len(CORPUS), 2]
    batch_in, batch_labels = make_batch(data, indices, 24)                   # [4, 24]
    save(out / "golden.pt", {
        "count": count, "tokens": torch.tensor(data.tokens.tolist()), "offsets": torch.tensor(data.offsets.tolist()),
        "short_index": 0, "short_max_len": 32, "short_input_ids": short_in, "short_labels": short_labels,
        "long_index": len(CORPUS) + 1, "long_max_len": 16, "long_input_ids": long_in, "long_labels": long_labels,
        "batch_indices": torch.tensor(indices), "batch_max_len": 24, "batch_input_ids": batch_in,
        "batch_labels": batch_labels,
    })
