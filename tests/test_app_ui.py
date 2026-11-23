"""
Headless UI tests: these execute app.py through Streamlit's AppTest harness.

They cover the pages that were rewritten in this change set - face gallery,
search results, name gallery, duplicate manager, scraped grid and settings -
so the thumbnail cache, pagination, similarity scoring and stats cache are
executed instead of only parsed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

APP_FILE = Path(__file__).resolve().parents[1] / "app.py"


def _app(default_timeout: int = 180) -> AppTest:
    return AppTest.from_file(APP_FILE, default_timeout=default_timeout)


def _goto(at: AppTest, page: str) -> AppTest:
    page_ids = {
        "🔍 Face Search": "face_search",
        "📥 Image Upload & Processing": "image_import",
        "🧠 Batch Face Processing": "batch_processing",
        "🌐 Web Scraping": "web_import",
        "👥 Face Gallery": "face_gallery",
        "🏷️ Name Gallery": "people",
        "🧩 Person Groups": "person_groups",
        "🔧 Duplicate Manager": "duplicates",
        "📊 Database Statistics": "database_statistics",
        "⚙️ Settings": "settings",
    }
    at.sidebar.selectbox[0].set_value(page_ids.get(page, page)).run()
    return at


def _assert_no_exception(at: AppTest) -> None:
    assert not at.exception, [str(element.value) for element in at.exception]


def _iter_elements(node):
    """Walk the element tree; AppTest.get() only sees the top level."""
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, dict):
        iterable = children.values()
    elif isinstance(children, (list, tuple)):
        iterable = children
    else:
        return
    for child in iterable:
        yield from _iter_elements(child)


def _images(at: AppTest):
    return [
        element for element in _iter_elements(at.main)
        if type(element).__name__ == "Image"
    ]


def _all_text(at: AppTest) -> str:
    parts = []
    for name in ("markdown", "caption", "subheader", "header", "title", "info", "error"):
        collection = getattr(at, name, None)
        if collection is None:
            continue
        try:
            parts.extend(str(element.value) for element in collection)
        except TypeError:
            continue
    try:
        parts.extend(f"{element.label} {element.value}" for element in at.metric)
    except TypeError:
        pass
    return " ".join(parts)


def test_app_boots(seeded_data):
    at = _app()
    at.run()
    _assert_no_exception(at)
    assert any("Face Search" in title.value for title in at.title)


@pytest.mark.parametrize(
    "page, needle",
    [
        ("👥 Face Gallery", "Showing"),
        ("🏷️ Name Gallery", "Erika"),
        ("🔧 Duplicate Manager", "Duplicate"),
        ("🌐 Web Scraping", "Scraped"),
        ("⚙️ Settings", "Settings"),
        ("📊 Database Statistics", "Total Faces"),
    ],
)
def test_pages_render_without_exception(seeded_data, page, needle):
    at = _app()
    at.run()
    _goto(at, page)
    _assert_no_exception(at)
    joined = _all_text(at)
    assert needle.lower() in joined.lower(), joined[:800]


def test_settings_report_the_effective_face_detector(seeded_data):
    at = _app()
    at.run()
    _goto(at, "⚙️ Settings")
    _assert_no_exception(at)

    joined = _all_text(at)
    assert "Active detection" in joined
    assert "RetinaFace" in joined
    assert "MTCNN" in joined


def test_face_gallery_renders_cached_thumbnails(seeded_data):
    at = _app()
    at.run()
    _goto(at, "👥 Face Gallery")
    _assert_no_exception(at)
    images = _images(at)
    assert len(images) >= len(seeded_data["metadatas"][:50]), len(images)
    assert not at.error, [str(element.value) for element in at.error]
    assert any("Include biometric embeddings" in checkbox.label for checkbox in at.checkbox)
    assert any("Include original images" in checkbox.label for checkbox in at.checkbox)


def test_search_results_render(seeded_data):
    """Feed real search results (real ChromaDB query) into the render path."""
    import numpy as np

    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    source = np.asarray(seeded_data["embeddings"][0], dtype=np.float32)
    results = store.search_similar_faces(source, n_results=6, min_similarity=0.0)
    assert results, "similarity search returned no results"
    similarities = [result["similarity"] for result in results]
    assert similarities == sorted(similarities, reverse=True)
    assert similarities[0] == pytest.approx(1.0, abs=1e-4)

    at = _app()
    at.session_state["search_results"] = results
    at.run()
    _goto(at, "🔍 Face Search")
    _assert_no_exception(at)
    assert len(_images(at)) > 0, "search results rendered no images"
    assert "%" in _all_text(at)
    # Low-similarity hits are intentionally rendered as st.error badges, so only
    # look for real rendering failures here.
    real_errors = [
        str(element.value) for element in at.error
        if "Bild" in str(element.value) or "Image" in str(element.value)
    ]
    assert not real_errors, real_errors


def test_analysis_modal_path_is_safe(seeded_data):
    """The face-info modal is rendered from session state; it must not crash."""
    metadata = seeded_data["metadatas"][0]
    at = _app()
    at.session_state["info_face_id"] = metadata["face_id"]
    at.session_state["info_image_path"] = metadata["image_path"]
    at.session_state["info_face_location"] = metadata["location"]
    at.run()
    _assert_no_exception(at)


def test_batch_processing_lists_configured_directories(seeded_data):
    """The batch page must use FACE_DATA_DIR instead of a hardcoded data/ path."""
    at = _app()
    at.run()
    _goto(at, "🧠 Batch Face Processing")
    _assert_no_exception(at)

    warnings = [str(element.value) for element in at.warning]
    assert not any("No image directories found" in value for value in warnings), warnings

    options = [str(option) for box in at.selectbox for option in box.options]
    image_count = len({meta["image_path"] for meta in seeded_data["metadatas"]})
    assert any(f"{image_count} images" in option for option in options), options
    assert any("3 images" in option for option in options), options


def test_scraping_rejects_invalid_urls(seeded_data):
    """Invalid input is rejected with a clear message instead of a scraper run."""
    at = _app()
    at.run()
    _goto(at, "🌐 Web Scraping")
    at.text_area[0].set_value("not a url")
    next(button for button in at.button if "Start Scraping" in button.label).click().run()
    _assert_no_exception(at)

    errors = [str(element.value) for element in at.error]
    assert any("Invalid URL" in value for value in errors), errors


def test_run_coroutine_cancels_pending_tasks():
    """Closing the loop while Playwright's transport task is pending must not happen."""
    import asyncio

    from async_utils import run_coroutine

    state = {}

    async def coroutine_with_background_task():
        async def worker():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise

        asyncio.create_task(worker())
        return "done"

    assert run_coroutine(coroutine_with_background_task()) == "done"
    assert state.get("cancelled") is True


def test_search_rejects_unreadable_upload(seeded_data):
    """An empty upload must yield a message instead of a PIL traceback."""
    at = _app()
    at.run()
    _goto(at, "🔍 Face Search")
    at.file_uploader[0].set_value(("leer.jpg", b"", "image/jpeg"))
    at.run()
    _assert_no_exception(at)

    errors = [str(element.value) for element in at.error]
    assert any("not a readable image" in value for value in errors), errors


def test_uploaded_image_uses_safe_data_path_without_shared_tmp_file(seeded_data):
    from io import BytesIO

    from config import IMAGES_DIR

    buffer = BytesIO()
    Image.new("RGB", (640, 480), (120, 80, 40)).save(buffer, format="JPEG")
    at = _app()
    at.run()
    _goto(at, "image_import")
    at.file_uploader[0].set_value(("portrait.jpg", buffer.getvalue(), "image/jpeg")).run()
    next(button for button in at.button if "Process and Add to Database" in button.label).click().run()

    _assert_no_exception(at)
    imported_path = IMAGES_DIR / "portrait.jpg"
    assert imported_path.is_file()
    assert not Path("/tmp/uploaded_0_../portrait.jpg").exists()
    from vector_store import FaceVectorStore
    FaceVectorStore().delete_faces_by_image_path(str(imported_path))
    imported_path.unlink(missing_ok=True)


@pytest.fixture
def duplicate_pair(seeded_data):
    """A byte-identical copy of a seeded image with its own database row."""
    import shutil

    import numpy as np

    from config import IMAGES_DIR
    from vector_store import FaceVectorStore

    source = sorted(IMAGES_DIR.glob("person_*.jpg"))[0]
    copy = IMAGES_DIR / "person_0_kopie.jpg"
    shutil.copyfile(source, copy)

    store = FaceVectorStore()
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1.0
    store.add_face_embedding("face_duplicate_copy", vector, {
        "face_id": "face_duplicate_copy",
        "image_path": str(copy),
        "location": "40,160,160,40",
        "embedding_dimension": 512,
        "added_at": "2026-01-01T00:00:00",
    })

    try:
        yield source, copy, store
    finally:
        copy.unlink(missing_ok=True)
        store.delete_face("face_duplicate_copy")
        store.invalidate_stats_cache()


def test_duplicate_manager_deletes_copy_and_reconciles_database_rows(duplicate_pair, monkeypatch):
    source, copy, store = duplicate_pair
    from vector_store import FaceVectorStore

    # Simulate a failed direct deletion; the reconciliation pass must remove
    # stale vectors for files that were actually deleted from disk.
    monkeypatch.setattr(FaceVectorStore, "delete_faces_by_image_path", lambda self, image_path: 0)

    at = _app()
    at.run()
    _goto(at, "🔧 Duplicate Manager")
    next(button for button in at.button if "Find Duplicates in images" in button.label).click().run()
    _assert_no_exception(at)

    radios = [element for element in at.radio if "Keep file" in getattr(element, "label", "")]
    assert radios, "no keep-selection was rendered for the duplicate set"
    radios[0].set_value(str(source))
    next(button for button in at.button if "Review deletion" in button.label).click().run()
    _assert_no_exception(at)
    assert any("permanently delete" in str(item.value) for item in at.warning)
    next(button for button in at.button if "Confirm deletion" in button.label).click().run()
    _assert_no_exception(at)

    assert source.exists(), "the kept file must survive"
    assert not copy.exists(), "the other file must be deleted from disk"
    remaining = store.collection.get(where={"image_path": str(copy)}, include=[])["ids"]
    assert remaining == [], "reconciliation must remove database rows for deleted files"
    assert any("deleted" in str(message.value) for message in at.success)


@pytest.fixture
def isolated_settings_file():
    from settings_store import settings_path

    path = settings_path()
    backup = path.read_bytes() if path.exists() else None
    path.unlink(missing_ok=True)
    yield path
    if backup is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(backup)


def test_settings_save_persists_and_is_used_by_the_search(seeded_data, isolated_settings_file):
    at = _app()
    at.run()
    _goto(at, "⚙️ Settings")

    at.slider[0].set_value(0.9)
    at.number_input[0].set_value(7)
    at.number_input[1].set_value(16)
    next(button for button in at.button if "Save settings" in button.label).click().run()
    _assert_no_exception(at)

    assert any("Settings saved" in str(message.value) for message in at.success)
    from settings_store import load_settings

    assert load_settings() == {
        "embedding_model": "buffalo_l",
        "similarity_threshold": 0.9,
        "max_results": 7,
        "batch_size": 16,
        "detection_model": "cnn",
    }

    # A fresh AppTest is a new session: the values must come from disk.
    image_path = Path(seeded_data["metadatas"][0]["image_path"])
    restarted = _app()
    restarted.run()
    _goto(restarted, "🔍 Face Search")
    restarted.file_uploader[0].set_value(("bild.jpg", image_path.read_bytes(), "image/jpeg"))
    restarted.run()
    _assert_no_exception(restarted)

    max_results_slider = next(slider for slider in restarted.slider if "Maximum results" in slider.label)
    threshold_slider = next(slider for slider in restarted.slider if "Minimum cosine similarity" in slider.label)
    min_face_size_slider = next(slider for slider in restarted.slider if "Minimum face size" in slider.label)
    assert max_results_slider.value == 7
    assert threshold_slider.value == pytest.approx(0.9)
    assert min_face_size_slider.value == 30
    assert not any("Additional similarity filter" in item.label for item in restarted.selectbox)


def test_pending_navigation_switches_the_sidebar_page(seeded_data):
    at = _app()
    at.run()

    at.session_state["pending_page"] = "👥 Face Gallery"
    at.run()

    _assert_no_exception(at)
    assert at.session_state["nav_page"] == "face_gallery"
    assert any("Face library" in header.value for header in at.header), [
        header.value for header in at.header
    ]


def test_stable_navigation_switches_between_sections(seeded_data):
    at = _app()
    at.run()
    _goto(at, "duplicates")
    assert at.session_state["nav_page"] == "duplicates"
    assert any("Duplicate manager" in header.value for header in at.header)

    _goto(at, "face_gallery")
    assert at.session_state["nav_page"] == "face_gallery"
    assert any("Face library" in header.value for header in at.header)
    _assert_no_exception(at)


def test_settings_can_change_the_detection_model(seeded_data, isolated_settings_file):
    at = _app()
    at.run()
    _goto(at, "⚙️ Settings")

    model_select = next(
        element for element in at.selectbox if "Fallback face detector" in element.label
    )
    model_select.set_value("hog")
    next(button for button in at.button if "Save settings" in button.label).click().run()
    _assert_no_exception(at)

    from settings_store import load_settings

    assert any("Settings saved" in str(message.value) for message in at.success)
    assert load_settings()["detection_model"] == "hog"
