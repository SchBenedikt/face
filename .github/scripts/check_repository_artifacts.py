#!/usr/bin/env python3
"""Fail CI when generated, private, or oversized artifacts are tracked."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_FILE_SIZE = 10 * 1024 * 1024
BANNED_NAMES = {
    ".ds_store",
    "duplicates.db",
    "bilder_alles.txt",
    "vector_store.py.backup",
}
BANNED_SUFFIXES = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".db-journal",
    ".db-wal",
    ".db-shm",
    ".log",
    ".backup",
    ".pem",
    ".key",
}


def tracked_paths() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / entry.decode("utf-8") for entry in output.split(b"\0") if entry]


def main() -> int:
    violations: list[str] = []
    for path in tracked_paths():
        relative = path.relative_to(ROOT)
        name = relative.name.lower()
        suffix = path.suffix.lower()

        if name in BANNED_NAMES or suffix in BANNED_SUFFIXES:
            violations.append(f"generated or sensitive artifact is tracked: {relative}")
            continue
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            violations.append(f"environment file is tracked: {relative}")
            continue
        if path.is_file() and path.stat().st_size > MAX_FILE_SIZE:
            size = path.stat().st_size / (1024 * 1024)
            violations.append(f"file exceeds 10 MiB ({size:.1f} MiB): {relative}")

    if violations:
        print("Repository artifact check failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print("Repository artifact check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
