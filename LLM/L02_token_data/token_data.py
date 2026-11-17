"""L02 Token data: token files, pre-training samples, batches (docs/specification.md, S-5, S-10).

tokens.bin   all token ids of all samples, uint16, little-endian
offsets.bin  start of each sample in tokens.bin, uint64, little-endian; the last entry is the total
"""

import sys
from array import array
from pathlib import Path

import torch

from course.py.io import ROOT
from course.py.jsonl import read_jsonl
from LLM.L01_tokenizer.tokenizer import ENDOFTEXT, IM_END, IM_START, Tokenizer

assert sys.byteorder == "little", "array writes the byte order of the machine; the files are little-endian"
PAD, IGNORE = ENDOFTEXT, -100  # padding id; label that the loss ignores


def write_token_files(folder: str | Path, samples: list[list[int]]) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tokens, offsets = array("H"), array("Q", [0])
    for ids in samples:
        tokens.extend(ids)
        offsets.append(len(tokens))
    (folder / "tokens.bin").write_bytes(tokens.tobytes())
    (folder / "offsets.bin").write_bytes(offsets.tobytes())


def tokenize_jsonl(jsonl: str | Path, tok: Tokenizer, folder: str | Path) -> int:
    """Encodes the "text" of each line of a JSON Lines file. Returns the count of samples."""
    samples = [tok.encode(row["text"]) for row in read_jsonl(jsonl)]
    write_token_files(folder, samples)
    return len(samples)


class TokenFile:
    def __init__(self, folder: str | Path):
        folder = Path(folder)
        self.tokens = array("H", (folder / "tokens.bin").read_bytes())
        self.offsets = array("Q", (folder / "offsets.bin").read_bytes())

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def sample(self, i: int) -> list[int]:
        return self.tokens[self.offsets[i]:self.offsets[i + 1]].tolist()


def pretrain_sample(ids: list[int], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[IM_START] + ids + [IM_END], cut and padded to max_len. Labels: the ids; IGNORE at the padding."""
    ids = [IM_START] + ids[:max_len - 2] + [IM_END]
    input_ids = torch.full((max_len,), PAD, dtype=torch.long)          # [max_len]
    input_ids[:len(ids)] = torch.tensor(ids)
    labels = input_ids.clone()                                          # [max_len]
    labels[len(ids):] = IGNORE
    return input_ids, labels


def make_batch(data: TokenFile, indices: list[int], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The samples at indices, as two tensors [len(indices), max_len]: input ids and labels."""
    pairs = [pretrain_sample(data.sample(i), max_len) for i in indices]
    return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])


if __name__ == "__main__":  # tokenizes TinyStories (valid) with the tokenizer of L01
    if not (ROOT / "runs/reference/tokenizer.txt").exists():
        raise SystemExit("First train the tokenizer: uv run python -m LLM.L01_tokenizer.tokenizer")
    tok = Tokenizer.load(ROOT / "runs/reference/tokenizer.txt")
    folder = ROOT / "runs/reference/tokens/tinystories-valid"
    n = tokenize_jsonl(ROOT / "data/llm/tinystories/valid.jsonl", tok, folder)
    data = TokenFile(folder)
    lengths = torch.tensor([len(data.sample(i)) for i in range(len(data))]).float()
    print(f"{n} samples, {len(data.tokens)} tokens; length mean {lengths.mean():.0f}, "
          f"95th percentile {lengths.quantile(0.95):.0f}, max {lengths.max():.0f}")
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(0))[:4].tolist()
    input_ids, labels = make_batch(data, order, 256)
    print(f"batch {tuple(input_ids.shape)}; first sample: {tok.decode(input_ids[0, :24].tolist())!r}")
