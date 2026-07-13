#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path


def iter_filenames(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_filenames(value)
        return

    if hasattr(obj, "file") and hasattr(obj.file, "name"):
        yield obj.file.name

    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from iter_filenames(item)


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "tokenized.p"

    if not path.exists():
        print(f"Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    print(text)
        return

    seen = set()
    for name in iter_filenames(data):
        if name and name not in seen:
            seen.add(name)
            print(name)


if __name__ == "__main__":
    main()
