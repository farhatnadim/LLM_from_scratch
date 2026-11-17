"""Golden file of L01 for the C++ test (tokenizer_test.cpp). Run all: uv run tools/make_goldens.py"""

from pathlib import Path

import torch

from course.py.io import save
from LLM.L01_tokenizer.tokenizer import (FIRST_MERGE, FIRST_SPECIAL, Tokenizer, pretokenize, special_text,
                                         split_special, train_bpe)

CORPUS = [  # repeated words, punctuation, tabs, Unicode, a control text
    "Once upon a time, there was a little dog named Max. Max liked to run in the park.",
    "One day, Max saw a big red ball. He ran to the ball and   pushed it with his nose.",
    'The little girl laughed. "Max, you are a funny dog!" she said.\nMax wagged his tail.',
    "Tom and Lily went to the café.\tThey ate a crêpe, and Lily smiled 🙂.",
    "<|endoftext|>The sun was warm. The birds sang in the trees, and the little dog slept.",
    'Mom said: "It is time for bed." Lily said: "One more story, please!"\r\n',
    "The end.   The end.\n\nThe end!",
]
PLAIN = "Max  ran\tto the café 🙂!\n\nThe end. "
SAMPLE = ("<|im_start|>user\nTell me a story about Max.<|im_end|>\n<|im_start|>assistant\n<think>\nA dog."
          "\n</think>\n\nOnce upon a time, Max  ran\tto the café 🙂!<|im_end|>\n<|act_7|><|act_255|>")


def make_golden(out: Path) -> None:
    tok = Tokenizer(train_bpe(CORPUS, 600))
    pieces = split_special(SAMPLE)
    tok.save(out / "merges.txt")
    save(out / "golden.pt", {
        "corpus": CORPUS, "vocab_size": 600, "merges": torch.tensor(tok.merges),      # [72, 2]
        "special_texts": [special_text(i) for i in range(FIRST_SPECIAL, FIRST_MERGE)],
        "plain": PLAIN, "pretokens": [w.decode() for w in pretokenize(PLAIN)],
        "sample": SAMPLE, "piece_texts": [p if isinstance(p, str) else "" for p in pieces],
        "piece_ids": torch.tensor([p if isinstance(p, int) else -1 for p in pieces]),
        "ids": torch.tensor(tok.encode(SAMPLE)), "decoded": tok.decode(tok.encode(SAMPLE)),
    })
