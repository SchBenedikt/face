"""Duplicate resolution must delete only the files it was told to delete."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from duplicate_detector import delete_duplicate_files


def test_deletes_every_file_except_the_kept_one(tmp_path):
    files = [tmp_path / name for name in ("a.jpg", "b.jpg", "c.jpg")]
    for path in files:
        path.write_bytes(b"same")

    result = delete_duplicate_files(files, files[0])

    assert [path.name for path in result["deleted"]] == ["b.jpg", "c.jpg"]
    assert result["missing"] == []
    assert result["errors"] == []
    assert files[0].exists()
    assert not files[1].exists()
    assert not files[2].exists()


def test_already_deleted_files_are_reported_instead_of_raising(tmp_path):
    keep = tmp_path / "keep.jpg"
    keep.write_bytes(b"same")
    already_gone = tmp_path / "gone.jpg"

    result = delete_duplicate_files([keep, already_gone], keep)

    assert result["deleted"] == []
    assert [path.name for path in result["missing"]] == ["gone.jpg"]
    assert result["errors"] == []
    assert keep.exists()


def test_nothing_is_deleted_when_the_keep_file_is_not_in_the_set(tmp_path):
    files = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for path in files:
        path.write_bytes(b"same")

    result = delete_duplicate_files(files, tmp_path / "not-in-set.jpg")

    assert result["deleted"] == []
    assert result["errors"], "a foreign keep file must be rejected"
    assert all(path.exists() for path in files)
