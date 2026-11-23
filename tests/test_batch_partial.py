"""Regression test for partial batch failures and the update mode.

The valid images must still be written while corrupt images are reported by
name, without aborting the run. A second pass in "Update All Images" mode must
keep the database and UI consistent. The test runs in a subprocess with an
empty FACE_DATA_DIR so the shared UI fixtures cannot pre-populate the store.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUBS_DIR = PROJECT_ROOT / "tests" / "stubs"
APP_FILE = PROJECT_ROOT / "app.py"


def test_partial_batch_reports_files_and_update_mode_stays_consistent(tmp_path):
    data_dir = tmp_path / "partial-data"
    first_image = tmp_path / "partial-a.jpg"
    second_image = tmp_path / "partial-b.jpg"
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
UPDATE_MODE = "🔄 Update All Images (Replace Existing)"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
(IMAGES_DIR / FIRST.name).write_bytes(FIRST.read_bytes())
(IMAGES_DIR / SECOND.name).write_bytes(SECOND.read_bytes())
(IMAGES_DIR / "broken_zero.jpg").write_bytes(b"")
(IMAGES_DIR / "broken_truncated.jpg").write_bytes(FIRST.read_bytes()[:200])


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


def assert_failure_report(at):
    messages = texts(at, "warning", "info", "markdown")
    assert any("2 errors" in value for value in messages), messages
    assert any("broken_zero.jpg" in value for value in messages), messages
    assert any("broken_truncated.jpg" in value for value in messages), messages
    assert not exceptions(at), exceptions(at)


at = AppTest.from_file(APP, default_timeout=300)
at.run()
assert not exceptions(at), exceptions(at)
at.sidebar.selectbox[0].set_value(BATCH).run()
assert not exceptions(at), exceptions(at)

next(button for button in at.button if "Process Images" in button.label).click().run()
assert_failure_report(at)
assert any(
    metric.label == "Total Faces" and str(metric.value) == "8"
    for metric in at.metric
), [(metric.label, metric.value) for metric in at.metric]
print("MIXED_WARNING:", texts(at, "warning"))
print("MIXED_INFO:", texts(at, "info")[:4])
print("MIXED_SIDEBAR:", [(metric.label, metric.value) for metric in at.metric])

store = FaceVectorStore()
faces = store.get_all_faces_metadata()
assert len(faces) == 8, len(faces)
assert len({{face["metadata"]["image_path"] for face in faces}}) == 2
print("DB_FACES_MIXED:", len(faces))
print("DB_STATS_MIXED:", store.get_collection_stats(force_refresh=True))

# The update path must reprocess every image, not skip the already-written ones.
at.radio[0].set_value(UPDATE_MODE).run()
next(button for button in at.button if "Update Images" in button.label).click().run()
assert_failure_report(at)
assert any(
    metric.label == "Total Faces" and str(metric.value) == "8"
    for metric in at.metric
), [(metric.label, metric.value) for metric in at.metric]
print("UPDATE_WARNING:", texts(at, "warning"))
print("UPDATE_INFO:", texts(at, "info")[:4])
print("UPDATE_SIDEBAR:", [(metric.label, metric.value) for metric in at.metric])

faces = store.get_all_faces_metadata()
assert len(faces) == 8, len(faces)
print("DB_FACES_UPDATE:", len(faces))
print("DB_STATS_UPDATE:", store.get_collection_stats(force_refresh=True))

at.sidebar.selectbox[0].set_value(GALLERY).run()
assert not exceptions(at), exceptions(at)
assert any(
    metric.label == "Faces" and str(metric.value) == "8"
    for metric in at.metric
), [(metric.label, metric.value) for metric in at.metric]
assert len(images(at)) >= 8, len(images(at))
print("GALLERY_IMAGES:", len(images(at)))

person_id = store.assign_person_name(faces[0]["face_id"], {{
    "first_name": "Partial",
    "middle_names": "",
    "last_name": "Person",
    "birth_date": "",
    "birth_place": "",
    "notes": "partial batch",
}})
assert person_id
at.session_state["selected_person"] = person_id
at.sidebar.selectbox[0].set_value(NAMES).run()
assert not exceptions(at), exceptions(at)
expander_labels = [getattr(element, "label", "") for element in at.expander]
assert any("Partial Person" in label for label in expander_labels), expander_labels
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
assert results, "search returned no results"
assert results[0]["similarity"] > 0.99, results[0]["similarity"]
print("SEARCH_RESULTS:", len(results))
print("SEARCH_TOP:", round(results[0]["similarity"], 4))
print("PARTIAL_E2E_OK")
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
    assert "PARTIAL_E2E_OK" in output, output[-6000:]
    assert "DB_FACES_MIXED: 8" in output, output[-6000:]
    assert "DB_FACES_UPDATE: 8" in output, output[-6000:]
    assert "SEARCH_TOP: 1.0" in output, output[-6000:]
    assert "broken_zero.jpg" in output, output[-6000:]
    assert "broken_truncated.jpg" in output, output[-6000:]
