"""Duplicate detection must also look inside per-domain scraped folders."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from duplicate_detector import SimpleDuplicateDetector


def test_find_duplicates_inside_subdirectories(tmp_path):
    root = tmp_path / "scraped"
    domain = root / "example_com"
    domain.mkdir(parents=True)
    original = domain / "foto.jpg"
    original.write_bytes(b"identical-bytes")
    (domain / "foto_kopie.jpg").write_bytes(b"identical-bytes")
    (domain / "einzigartig.jpg").write_bytes(b"different-bytes")

    detector = SimpleDuplicateDetector(db_path=str(tmp_path / "duplicates.db"))
    duplicates = detector.find_duplicates_in_directory(root)

    assert len(duplicates) == 1, duplicates
    names = {path.name for path in next(iter(duplicates.values()))}
    assert names == {"foto.jpg", "foto_kopie.jpg"}


def test_quick_scan_matches_name_and_size_but_not_content(tmp_path):
    root = tmp_path / "images"
    first = root / "a" / "foto.jpg"
    second = root / "b" / "foto.jpg"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"aaaa")
    second.write_bytes(b"bbbb")  # same name and size, different bytes

    detector = SimpleDuplicateDetector(db_path=str(tmp_path / "duplicates.db"))

    quick = detector.find_duplicates_in_directory(root, use_content_hash=False)
    full = detector.find_duplicates_in_directory(root, use_content_hash=True)

    assert len(quick) == 1, quick
    assert full == {}, full
