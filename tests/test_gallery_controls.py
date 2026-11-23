"""The gallery controls must actually filter the database rows."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_FILE = Path(__file__).resolve().parents[1] / "app.py"
GALLERY = "face_gallery"


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
        if "Search by file name" in widget.label:
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
    assert "Showing 2 of 2 faces" in text, text[:1200]


def test_gallery_search_filters_by_face_id(seeded_data):
    at = _app()
    at.run()
    _goto(at, GALLERY)
    _assert_no_exception(at)

    _search_box(at).set_value("face_1_0").run()

    _assert_no_exception(at)
    text = _all_text(at)
    assert "face_1_0" in text, text[:1200]
    assert "Showing 1 of 1 faces" in text, text[:1200]


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
    assert "No faces match the selected minimum size" in text, text[:1200]


def test_gallery_metadata_scan_returns_every_seeded_row(seeded_data):
    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    faces = store.get_all_faces_metadata()

    assert len(faces) == len(seeded_data["ids"])
    assert {face["face_id"] for face in faces} == set(seeded_data["ids"])
    assert all(face["metadata"].get("image_path") for face in faces)


def test_gallery_page_searches_people_and_returns_only_requested_page(seeded_data):
    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    first = store.get_face_gallery_page(
        search_term="erika mustermann", min_face_size=50, page=1, page_size=1
    )
    second = store.get_face_gallery_page(
        search_term="erika mustermann", min_face_size=50, page=2, page_size=1
    )

    assert first["total"] == 4
    assert first["total_pages"] == 4
    assert len(first["faces"]) == len(second["faces"]) == 1
    assert first["faces"][0]["face_id"] != second["faces"][0]["face_id"]
    assert all("Erika" in face["metadata"]["full_name"] for face in first["faces"] + second["faces"])


def test_gallery_page_clamps_out_of_range_page_and_sorts_by_image_name(seeded_data):
    from vector_store import FaceVectorStore

    result = FaceVectorStore().get_face_gallery_page(
        sort_by="Image name", page=999, page_size=3
    )

    assert result["total"] == len(seeded_data["ids"])
    assert result["page"] == result["total_pages"]
    assert len(result["faces"]) == result["total"] - 3 * (result["total_pages"] - 1)
