"""The gallery controls must actually filter the database rows."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_FILE = Path(__file__).resolve().parents[1] / "app.py"
GALLERY = "👥 Face Gallery"


def _app() -> AppTest:
    return AppTest.from_file(APP_FILE, default_timeout=180)


def _goto(at: AppTest, page: str) -> AppTest:
    at.sidebar.selectbox[0].set_value(page).run()
    return at


def _all_text(at: AppTest) -> str:
    parts = []
    for name in ("markdown", "caption", "subheader", "header", "title", "info", "error", "warning"):
        collection = getattr(at, name, None)
        if collection is None:
            continue
        try:
            parts.extend(str(element.value) for element in collection)
        except TypeError:
            continue
    return " ".join(parts)


def _search_box(at: AppTest):
    for widget in at.text_input:
        if "Suche nach Dateiname" in widget.label:
            return widget
    raise AssertionError("gallery search box not found")


def _assert_no_exception(at: AppTest) -> None:
    assert not at.exception, [str(element.value) for element in at.exception]


def test_gallery_search_filters_by_file_name(seeded_data):
    at = _app()
    at.run()
    _goto(at, GALLERY)
    _assert_no_exception(at)

    _search_box(at).set_value("person_1").run()

    _assert_no_exception(at)
    text = _all_text(at)
    assert "person_1.jpg" in text, text[:1200]
    assert "person_0.jpg" not in text, text[:1200]
    assert "Zeige 2 von 2 Gesichtern" in text, text[:1200]


def test_gallery_search_filters_by_face_id(seeded_data):
    at = _app()
    at.run()
    _goto(at, GALLERY)
    _assert_no_exception(at)

    _search_box(at).set_value("face_1_0").run()

    _assert_no_exception(at)
    text = _all_text(at)
    assert "face_1_0" in text, text[:1200]
    assert "Zeige 1 von 1 Gesichtern" in text, text[:1200]


def test_gallery_minimum_size_filters_out_small_faces(seeded_data):
    at = _app()
    at.run()
    _goto(at, GALLERY)
    _assert_no_exception(at)

    assert at.slider, "gallery size slider not found"
    # The smaller side of every seeded face is 120px, so 121px filters all.
    at.slider[0].set_value(121).run()

    _assert_no_exception(at)
    text = _all_text(at)
    assert "Keine Gesichter entsprechen der eingestellten Mindestgröße" in text, text[:1200]


def test_gallery_metadata_scan_returns_every_seeded_row(seeded_data):
    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    faces = store.get_all_faces_metadata()

    assert len(faces) == len(seeded_data["ids"])
    assert {face["face_id"] for face in faces} == set(seeded_data["ids"])
    assert all(face["metadata"].get("image_path") for face in faces)
