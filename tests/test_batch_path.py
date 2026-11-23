"""End-to-end regression tests for the batch-processing data path.

The batch page used to pass 0-100 progress values to ``st.progress`` (which
expects 0-1), so every batch failed immediately. It also imported
``fast_config``, whose hardcoded paths created an unused ``<repo>/data`` tree
and ignored ``FACE_DATA_DIR``. These tests run the real batch button with an
empty data directory and verify the resulting database, gallery, names, search
and sidebar state.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUBS_DIR = PROJECT_ROOT / "tests" / "stubs"
APP_FILE = PROJECT_ROOT / "app.py"


def test_fast_config_uses_the_configured_data_directory():
    """fast_config must not grow a second hardcoded data tree."""
    import config
    import fast_config

    assert Path(fast_config.DATA_DIR) == Path(config.DATA_DIR)
    assert Path(fast_config.IMAGES_DIR) == Path(config.IMAGES_DIR)
    assert Path(fast_config.EMBEDDINGS_DIR) == Path(config.EMBEDDINGS_DIR)
    assert Path(fast_config.SCRAPED_DIR) == Path(config.SCRAPED_DIR)
    assert Path(fast_config.VECTOR_DB_PATH) == Path(config.VECTOR_DB_PATH)


def test_batch_path_processes_gallery_names_and_search(tmp_path):
    data_dir = tmp_path / "batch-data"
    first_image = tmp_path / "batch-a.jpg"
    second_image = tmp_path / "batch-b.jpg"
    Image.new("RGB", (640, 480), (180, 140, 120)).save(first_image, format="JPEG")
    Image.new("RGB", (640, 480), (70, 120, 180)).save(second_image, format="JPEG")
    assert not data_dir.exists()

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(STUBS_DIR)!r})

from pathlib import Path
from streamlit.testing.v1 import AppTest
from config import IMAGES_DIR, VECTOR_DB_PATH
from vector_store import FaceVectorStore

APP = {str(APP_FILE)!r}
FIRST = Path({str(first_image)!r})
SECOND = Path({str(second_image)!r})
BATCH = "batch_processing"
GALLERY = "face_gallery"
NAMES = "people"
SEARCH = "face_search"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
(IMAGES_DIR / FIRST.name).write_bytes(FIRST.read_bytes())
(IMAGES_DIR / SECOND.name).write_bytes(SECOND.read_bytes())


def _iter(node):
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, dict):
        iterable = children.values()
    elif isinstance(children, (list, tuple)):
        iterable = children
    else:
        return
    for child in iterable:
        yield from _iter(child)


def images(at):
    return [element for element in _iter(at.main) if type(element).__name__ == "Image"]


def exceptions(at):
    return [str(element.value) for element in at.exception]


def texts(at, *names):
    result = []
    for name in names:
        collection = getattr(at, name, None)
        if collection is not None:
            result.extend(str(element.value) for element in collection)
    return result


at = AppTest.from_file(APP, default_timeout=300)
at.run()
assert not exceptions(at), exceptions(at)
at.sidebar.selectbox[0].set_value(BATCH).run()
assert not exceptions(at), exceptions(at)
next(button for button in at.button if "Process Images" in button.label).click().run()
assert not exceptions(at), exceptions(at)

success = texts(at, "success")
assert any("Batch processing completed successfully" in value for value in success), success
infos = texts(at, "info")
assert any("2 images processed" in value and "8 faces added" in value for value in infos), infos
assert not texts(at, "error"), texts(at, "error")
assert any(
    metric.label == "Total Faces" and str(metric.value) == "8"
    for metric in at.metric
), [(metric.label, metric.value) for metric in at.metric]
print("BATCH_SUCCESS:", success)
print("BATCH_INFO:", infos[:3])
print("BATCH_SIDEBAR:", [(metric.label, metric.value) for metric in at.metric])

store = FaceVectorStore()
faces = store.get_all_faces_metadata()
assert len(faces) == 8, len(faces)
assert len({{face["metadata"]["image_path"] for face in faces}}) == 2
assert all(Path(face["metadata"]["image_path"]).is_file() for face in faces)
print("DB_PATH:", VECTOR_DB_PATH)
print("DB_FACES:", len(faces))
print("DB_STATS:", store.get_collection_stats(force_refresh=True))

person_id = store.assign_person_name(faces[0]["face_id"], {{
    "first_name": "Batch",
    "middle_names": "",
    "last_name": "Person",
    "birth_date": "",
    "birth_place": "",
    "notes": "synthetic batch",
}})
assert person_id, "batch name assignment failed"
print("PERSON_ID:", person_id)

at.sidebar.selectbox[0].set_value(GALLERY).run()
assert not exceptions(at), exceptions(at)
assert any(
    metric.label == "Faces" and str(metric.value) == "8"
    for metric in at.metric
), [(metric.label, metric.value) for metric in at.metric]
assert len(images(at)) >= 8, len(images(at))
print("GALLERY_IMAGES:", len(images(at)))

at.session_state["selected_person"] = person_id
at.sidebar.selectbox[0].set_value(NAMES).run()
assert not exceptions(at), exceptions(at)
expander_labels = [getattr(element, "label", "") for element in at.expander]
assert any("Batch Person" in label for label in expander_labels), expander_labels
assert len(images(at)) >= 1, len(images(at))
print("NAME_GALLERY:", expander_labels)

at.sidebar.selectbox[0].set_value(SEARCH).run()
at.file_uploader[0].set_value((FIRST.name, FIRST.read_bytes(), "image/jpeg")).run()
assert not exceptions(at), exceptions(at)
next(button for button in at.button if "Detect faces" in button.label).click().run()
assert not exceptions(at), exceptions(at)
next(button for button in at.button if "Search face 1" in button.label).click().run()
assert not exceptions(at), exceptions(at)

results = at.session_state["search_results"]
assert results, "batch search returned no results"
assert results[0]["similarity"] > 0.99, results[0]["similarity"]
assert any("Enhanced Search Results" in value for value in texts(at, "subheader")), texts(at, "subheader")
print("SEARCH_RESULTS:", len(results))
print("SEARCH_TOP:", round(results[0]["similarity"], 4))
print("BATCH_E2E_OK")
"""
    env = {
        "FACE_DATA_DIR": str(data_dir),
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "ANONYMIZED_TELEMETRY": "False",
        "PYTHONPATH": f"{STUBS_DIR}:{PROJECT_ROOT}",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output[-6000:]
    assert "BATCH_E2E_OK" in output, output[-6000:]
    assert "DB_FACES: 8" in output, output[-6000:]
    assert "SEARCH_TOP: 1.0" in output, output[-6000:]
