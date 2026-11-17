"""Download the data and weights of the course at pinned revisions (docs/specification.md, S-10).

Run:  uv run tools/fetch_data.py --list
      uv run tools/fetch_data.py tinystories-valid
      uv run tools/fetch_data.py llm          (all items of block 1)

The files go into data/, which is not in git. Each item has a fixed revision
(a commit of the Hugging Face repository), so a run gets the same bytes each time.
Some items are also changed into a course format (JSON Lines), see `convert`.
The items of block 3 come with milestone M3a.
"""

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from course.py.jsonl import write_jsonl  # noqa: E402

DATA = ROOT / "data"


@dataclass
class Item:
    repo: str
    kind: str                    # "dataset" or "model"
    revision: str                # commit of the repository
    files: list[str]
    dest: str                    # folder below data/
    size: str
    licence: str
    convert: Callable[[Path], None] | None = None
    note: str = ""
    groups: list[str] = field(default_factory=list)


def tinystories_to_jsonl(folder: Path) -> None:
    """TinyStoriesV2-GPT4-<split>.txt -> <split>.jsonl, one {"text": story} for each story."""
    for txt in sorted(folder.glob("TinyStoriesV2-GPT4-*.txt")):
        split = txt.stem.rsplit("-", 1)[1]                       # train or valid
        out = folder / f"{split}.jsonl"
        if out.exists() and out.stat().st_mtime >= txt.stat().st_mtime:
            continue
        n = write_jsonl(out, ({"text": s} for s in read_stories(txt)))
        print(f"  wrote {out.relative_to(ROOT)}: {n} stories")


def read_stories(txt: Path, chunk: int = 1 << 24):
    """Yields the stories of a TinyStories file. The separator is <|endoftext|>."""
    rest = ""
    with open(txt, encoding="utf-8", errors="replace") as f:
        while block := f.read(chunk):
            parts = (rest + block).split("<|endoftext|>")
            rest = parts.pop()
            yield from (p.strip() for p in parts if p.strip())
    if rest.strip():
        yield rest.strip()


ITEMS = {
    "tinystories-valid": Item(
        "roneneldan/TinyStories", "dataset", "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        ["TinyStoriesV2-GPT4-valid.txt"], "llm/tinystories", "22.5 MB", "CDLA-Sharing-1.0",
        convert=tinystories_to_jsonl, note="27 630 stories; tokenizer training and validation",
        groups=["llm"]),
    "tinystories-train": Item(
        "roneneldan/TinyStories", "dataset", "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        ["TinyStoriesV2-GPT4-train.txt"], "llm/tinystories", "2.23 GB", "CDLA-Sharing-1.0",
        convert=tinystories_to_jsonl, note="2 717 699 stories; pre-training", groups=["llm"]),
    "smol-smoltalk": Item(
        "HuggingFaceTB/smol-smoltalk", "dataset", "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
        [f"data/train-0000{i}-of-00004.parquet" for i in range(4)] + ["data/test-00000-of-00001.parquet"],
        "llm/smol-smoltalk", "0.97 GB", "Apache-2.0",
        note="460 341 chat conversations; fine-tune (lesson L10)", groups=["llm"]),
    "smoltalk2-think": Item(
        "HuggingFaceTB/smoltalk2", "dataset", "fc6cc2103c066455aade5d7fbb346039ae36ca5e",
        ["SFT/smoltalk_everyday_convs_reasoning_Qwen3_32B_think-00000-of-00001.parquet",
         "SFT/smoltalk_systemchats_Qwen3_32B_think-00000-of-00001.parquet"],
        "llm/smoltalk2-think", "71 MB", "Apache-2.0",
        note="29 493 conversations with a think block; fine-tune (lesson L11)", groups=["llm"]),
    "minimind-v": Item(
        "jingyaogong/minimind-v_dataset", "dataset", "1e279a8b665cb10383451a6af6fd62b9f35bdd79",
        ["pretrain_i2t.parquet", "sft_i2t.parquet"], "vlm/minimind-v", "9.3 GB",
        "tag Apache-2.0; its source ALLaVA-4V is CC BY-NC 4.0: study only",
        note="images 256 x 256 with captions and questions; block 2", groups=["vlm"]),
    "siglip2": Item(
        "google/siglip2-base-patch32-256", "model", "94dffa8cb1179de3e03f091dbc3917e5d5a9ae84",
        ["config.json", "preprocessor_config.json", "model.safetensors"],
        "weights/siglip2-base-patch32-256", "1.5 GB", "Apache-2.0",
        note="vision encoder of block 2 (the file also holds the text tower)", groups=["vlm"]),
}


def fetch(name: str, item: Item) -> None:
    from huggingface_hub import hf_hub_download

    folder = DATA / item.dest
    print(f"{name}: {item.repo}@{item.revision[:10]} -> {folder.relative_to(ROOT)} ({item.size}, {item.licence})")
    for file in item.files:
        hf_hub_download(item.repo, file, repo_type=item.kind, revision=item.revision, local_dir=folder)
    if item.convert:
        item.convert(folder)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="items or groups (llm, vlm)")
    parser.add_argument("--list", action="store_true", help="show the items and stop")
    args = parser.parse_args()

    if args.list or not args.names:
        for name, it in ITEMS.items():
            print(f"{name:18s} {it.size:>8s}  {','.join(it.groups):4s} {it.repo}  [{it.licence}]  {it.note}")
        return 0

    chosen = []
    for name in args.names:
        found = [n for n, it in ITEMS.items() if n == name or name in it.groups]
        if not found:
            parser.error(f"unknown item or group '{name}'")
        chosen += [n for n in found if n not in chosen]
    for name in chosen:
        fetch(name, ITEMS[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
