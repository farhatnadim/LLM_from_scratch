"""L01 BPE tokenizer: byte-level byte-pair encoding (docs/specification.md, S-5).

Ids 0-255 are the bytes. Ids 256-271 are control tokens and ids 272-527 are action tokens:
together, the special tokens. Ids from 528 are the merges, in the order of training.
"""

import re
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from course.py.io import ROOT
from course.py.jsonl import read_jsonl

FIRST_SPECIAL, FIRST_ACTION, FIRST_MERGE, VOCAB_SIZE = 256, 272, 528, 6400
CONTROL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|image_pad|>"] + [
    f"<|reserved_{i}|>" for i in range(10)]
ENDOFTEXT, IM_START, IM_END, THINK, END_THINK, IMAGE_PAD = range(256, 262)


def special_text(i: int) -> str:
    """Text of special id i, 256 <= i < 528: a control text, or <|act_k|> for action k."""
    return CONTROL[i - FIRST_SPECIAL] if i < FIRST_ACTION else f"<|act_{i - FIRST_ACTION}|>"


SPECIAL_ID = {special_text(i): i for i in range(FIRST_SPECIAL, FIRST_MERGE)}
SPECIAL_RE = re.compile("(" + "|".join(map(re.escape, SPECIAL_ID)) + ")")  # no text is the start of another
PRETOKEN_RE = re.compile(rb" ?[^\s]+|\s")  # \s of a bytes pattern: space \t \n \r \f \v


def split_special(text: str) -> list[str | int]:
    """Cuts the special texts out of text. Returns plain pieces (str) and special ids (int), in order."""
    return [SPECIAL_ID.get(p, p) for p in SPECIAL_RE.split(text) if p]


def pretokenize(text: str) -> list[bytes]:
    """Pre-tokens of a plain piece: an optional space and a run of non-space bytes, or one space byte."""
    return PRETOKEN_RE.findall(text.encode("utf-8"))


def base_vocab() -> list[bytes]:
    """Bytes of ids 0-527: one byte each, then the special texts."""
    return [bytes([i]) for i in range(256)] + [special_text(i).encode() for i in range(256, FIRST_MERGE)]


def merge(seq: list[int], pair: tuple[int, int], new: int) -> list[int]:
    """Replaces each occurrence of pair in seq by new, from left to right."""
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_bpe(texts: Iterable[str], vocab_size: int) -> list[tuple[int, int]]:
    """Learns vocab_size - 528 merges. A tie in the count goes to the pair with the larger bytes."""
    words = Counter()                                            # pre-token -> count
    for text in texts:
        for piece in split_special(text):
            if isinstance(piece, str):
                words.update(pretokenize(piece))
    work = [([*w], c) for w, c in words.items() if len(w) > 1]   # (byte ids, count)
    vocab, merges = base_vocab(), []
    while len(vocab) < vocab_size:
        pairs = Counter()
        for seq, c in work:
            for pair in zip(seq, seq[1:]):
                pairs[pair] += c
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], vocab[p[0]], vocab[p[1]]))
        merges.append(best)
        vocab.append(vocab[best[0]] + vocab[best[1]])
        work = [(s, c) for s, c in ((merge(s, best, len(vocab) - 1), c) for s, c in work) if len(s) > 1]
    return merges


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(m) for m in merges]
        self.rank = {pair: i for i, pair in enumerate(self.merges)}   # pair -> rank; its id is 528 + rank
        self.vocab = base_vocab()                                      # id -> bytes
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.cache: dict[bytes, list[int]] = {}                        # pre-token -> ids

    def encode_pretoken(self, word: bytes) -> list[int]:
        """Applies the merges: always the pair of the lowest rank first."""
        if word not in self.cache:
            seq = [*word]
            while len(seq) > 1:
                pair = min(zip(seq, seq[1:]), key=lambda p: self.rank.get(p, len(self.merges)))
                if pair not in self.rank:
                    break
                seq = merge(seq, pair, FIRST_MERGE + self.rank[pair])
            self.cache[word] = seq
        return self.cache[word]

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in split_special(text):
            if isinstance(piece, int):
                ids.append(piece)
            else:
                for word in pretokenize(piece):
                    ids += self.encode_pretoken(word)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """One merge on each line: the two ids, separated by a space."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(f"{a} {b}\n" for a, b in self.merges))

    @staticmethod
    def load(path: str | Path) -> "Tokenizer":
        return Tokenizer([tuple(map(int, line.split())) for line in Path(path).read_text().splitlines()])


if __name__ == "__main__":  # trains the reference tokenizer on TinyStories (valid)
    data = ROOT / "data/llm/tinystories/valid.jsonl"
    if not data.exists():
        raise SystemExit("First get the data: uv run tools/fetch_data.py tinystories-valid")
    texts = [row["text"] for row in read_jsonl(data)]
    start = time.perf_counter()
    tok = Tokenizer(train_bpe(texts, VOCAB_SIZE))
    print(f"{len(tok.merges)} merges from {len(texts)} texts in {time.perf_counter() - start:.0f} s")
    tok.save(ROOT / "runs/reference/tokenizer.txt")
    n_bytes = sum(len(t.encode()) for t in texts[:1000])
    print(f"{n_bytes / sum(len(tok.encode(t)) for t in texts[:1000]):.2f} bytes per token")
    ids = tok.encode("Once upon a time, there was a little dog.")
    print(ids, [tok.vocab[i].decode(errors="replace") for i in ids])
