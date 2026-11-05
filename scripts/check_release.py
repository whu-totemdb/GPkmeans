#!/usr/bin/env python3
"""Lightweight release hygiene checks."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".gitignore"}
BANNED_PATTERNS = [
    re.compile(r"[\u4e00-\u9fff]"),
    re.compile(r"/data\d*/"),
    re.compile("/" + "home/"),
    re.compile(r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    re.compile(r"bearer\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE),
]
BANNED_WORDS = ["TODO: Author", "rebuttal"]


def main() -> None:
    failures = []
    for path in ROOT.rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if ".git" in path.parts or not path.is_file():
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in BANNED_PATTERNS:
            if pattern.search(text):
                failures.append(f"{path.relative_to(ROOT)} matches {pattern.pattern}")
        for word in BANNED_WORDS:
            if word.lower() in text.lower():
                failures.append(f"{path.relative_to(ROOT)} contains {word!r}")
    if failures:
        print("Release hygiene check failed:")
        for item in failures:
            print("-", item)
        raise SystemExit(1)
    print("Release hygiene check passed.")


if __name__ == "__main__":
    main()
