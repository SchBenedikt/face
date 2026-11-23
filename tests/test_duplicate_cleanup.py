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


def test_false_positive_candidate_is_never_deleted(tmp_path):
    keep = tmp_path / "portrait.jpg"
    candidate = tmp_path / "portrait-300x200.jpg"
    keep.write_bytes(b"original image")
    candidate.write_bytes(b"different image with the same size")

    result = delete_duplicate_files([keep, candidate], keep)

    assert result["deleted"] == []
    assert result["errors"]
    assert keep.is_file() and candidate.is_file()


def test_partial_unlink_failure_restores_files_already_removed(tmp_path, monkeypatch):
    keep, first, second = [tmp_path / name for name in ("a.jpg", "b.jpg", "c.jpg")]
    for path in (keep, first, second):
        path.write_bytes(b"identical image bytes")
    real_unlink = Path.unlink

    def fail_on_second(path, *args, **kwargs):
        if path == second:
            raise OSError("injected unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_on_second)
    result = delete_duplicate_files([keep, first, second], keep)

    assert result["deleted"] == []
    assert result["rolled_back"] == [first]
    assert result["errors"]
    assert keep.read_bytes() == first.read_bytes() == second.read_bytes()


def test_failed_rollback_reports_only_files_still_missing(tmp_path, monkeypatch):
    import duplicate_detector as module

    keep, first, second = [tmp_path / name for name in ("a.jpg", "b.jpg", "c.jpg")]
    for path in (keep, first, second):
        path.write_bytes(b"identical image bytes")
    real_unlink = Path.unlink
    real_replace = module.os.replace

    def fail_on_second(path, *args, **kwargs):
        if path == second:
            raise OSError("injected unlink failure")
        return real_unlink(path, *args, **kwargs)

    def fail_restore(source, destination):
        if Path(destination) == first:
            raise OSError("injected rollback failure")
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "unlink", fail_on_second)
    monkeypatch.setattr(module.os, "replace", fail_restore)
    result = delete_duplicate_files([keep, first, second], keep)

    assert result["deleted"] == [first]
    assert result["rolled_back"] == []
    assert not first.exists()
    assert second.exists()
