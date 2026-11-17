"""Make the golden files of the smoke test and of all lessons (docs/specification.md, S-3).

Run:  uv run tools/make_goldens.py [--force]

Each reference with a function make_golden(out) writes its files into
data/goldens/<block>/<lesson>/. The C++ tests read them. CTest runs this
program first (fixture "goldens"). It does no work when no reference changed.
"""

import argparse
import hashlib
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from course.py.io import GOLDENS  # noqa: E402

STAMP = GOLDENS / "stamp.txt"


def references() -> list[tuple[str, str]]:
    """(golden folder, module) for the smoke test and for each lesson reference."""
    found = [("smoke", "course.py.smoke")]
    for py in sorted(ROOT.glob("*/L[0-9][0-9]_*/*.py")):
        lesson = py.parent.relative_to(ROOT)                     # LLM/L03_rmsnorm_swiglu
        found.append((lesson.as_posix(), ".".join((*lesson.parts, py.stem))))
    return found


def stamp() -> str:
    """Hash of every Python source that a golden file can depend on, and of the torch version."""
    h = hashlib.sha256(torch.__version__.encode())
    files = sorted([*ROOT.glob("course/py/*.py"), *ROOT.glob("*/L[0-9][0-9]_*/*.py"), Path(__file__)])
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="make all golden files again")
    args = parser.parse_args()

    refs = references()
    current = stamp()
    complete = all((GOLDENS / folder).is_dir() for folder, _ in refs)
    if not args.force and complete and STAMP.exists() and STAMP.read_text().strip() == current:
        print(f"golden files are up to date ({len(refs)} references)")
        return 0

    for folder, module_name in refs:
        module = importlib.import_module(module_name)
        if not hasattr(module, "make_golden"):
            continue
        out = GOLDENS / folder
        out.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        module.make_golden(out)
        print(f"{folder:32s} {time.perf_counter() - start:6.2f} s")
    STAMP.write_text(current + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
