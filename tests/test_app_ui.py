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
from streamlit.testing.v1 import AppTest

APP_FILE = Path(__file__).resolve().parents[1] / "app.py"


def _app(default_timeout: int = 180) -> AppTest:
    return AppTest.from_file(APP_FILE, default_timeout=default_timeout)


def _goto(at: AppTest, page: str) -> AppTest:
    at.sidebar.selectbox[0].set_value(page).run()
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
    assert any("Face Recognition Search" in title.value for title in at.title)


@pytest.mark.parametrize(
    "page, needle",
    [
        ("👥 Face Gallery", "Zeige"),
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


def test_face_gallery_renders_cached_thumbnails(seeded_data):
    at = _app()
    at.run()
    _goto(at, "👥 Face Gallery")
    _assert_no_exception(at)
    images = _images(at)
    assert len(images) >= len(seeded_data["metadatas"][:50]), len(images)
    assert not at.error, [str(element.value) for element in at.error]


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
    assert any("Ungültige URL" in value for value in errors), errors


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
