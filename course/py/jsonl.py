"""JSON Lines: one JSON object on each line (docs/specification.md, S-10)."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yields the objects of a JSON Lines file, one for each line that is not empty."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Writes the rows, one JSON object on each line. Returns the count of rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
