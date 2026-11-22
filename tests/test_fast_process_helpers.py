from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def fast_process_module(monkeypatch, tmp_path):
    # fast_process configures a file logger at import time; keep its log out of
    # the shared repository while exercising the update decision helper.
    monkeypatch.chdir(tmp_path)
    import fast_process

    return fast_process


def test_update_keeps_old_faces_when_processing_failed(fast_process_module):
    result = {
        "image_path": "/images/portrait.jpg",
        "error": "Could not extract embeddings",
        "preserve_existing": True,
    }

    assert not fast_process_module._should_remove_existing_faces(result)


def test_update_removes_old_faces_after_successful_no_face_result(fast_process_module):
    result = {"image_path": "/images/portrait.jpg", "face_count": 0}

    assert fast_process_module._should_remove_existing_faces(result)


def test_update_removes_old_faces_for_unreadable_source(fast_process_module):
    result = {
        "image_path": "/images/broken.jpg",
        "error": "Failed to load image",
        "source_unreadable": True,
    }

    assert fast_process_module._should_remove_existing_faces(result)
