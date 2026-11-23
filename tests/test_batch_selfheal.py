"""Regression test for self-healing Update All Images runs.

A valid batch is written first. One source file is then corrupted and another
is removed before an Update All Images run. The update must delete both stale
vectors, so database, sidebar, gallery and search agree on the remaining valid
image. Runs in a subprocess with an isolated FACE_DATA_DIR.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUBS_DIR = PROJECT_ROOT / "tests" / "stubs"
APP_FILE = PROJECT_ROOT / "app.py"


def test_update_all_prunes_corrupt_and_removed_images(tmp_path):
    data_dir = tmp_path / "selfheal-data"
    sources = []
    for index, colour in enumerate(((180, 140, 120), (70, 120, 180), (120, 180, 90))):
        path = tmp_path / f"selfheal-{index}.jpg"
        Image.new("RGB", (640, 480), colour).save(path, format="JPEG")
        sources.append(path)
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
SOURCE_A = Path({str(sources[0])!r})
SOURCE_B = Path({str(sources[1])!r})
SOURCE_C = Path({str(sources[2])!r})
BATCH = "batch_processing"
UPDATE = "🔄 Update All Images (Replace Existing)"
GALLERY = "face_gallery"
NAMES = "people"
SEARCH = "face_search"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
for source in (SOURCE_A, SOURCE_B, SOURCE_C):
    (IMAGES_DIR / source.name).write_bytes(source.read_bytes())


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


store = FaceVectorStore()

# A normal first batch on three valid images.
at = AppTest.from_file(APP, default_timeout=300)
at.run()
at.sidebar.selectbox[0].set_value(BATCH).run()
next(button for button in at.button if "Process Images" in button.label).click().run()
assert not exceptions(at), exceptions(at)
assert len(store.get_all_faces_metadata()) == 12
print("DB_BEFORE:", len(store.get_all_faces_metadata()))

# Change the data directory between runs: corrupt B, remove C.
(IMAGES_DIR / SOURCE_B.name).write_bytes(b"")
(IMAGES_DIR / SOURCE_C.name).unlink(missing_ok=True)
print("REMAINING_FILES:", sorted(path.name for path in IMAGES_DIR.iterdir()))

# Update All Images must prune the old B and C vectors.
at2 = AppTest.from_file(APP, default_timeout=300)
at2.run()
at2.sidebar.selectbox[0].set_value(BATCH).run()
at2.radio[0].set_value(UPDATE).run()
next(button for button in at2.button if "Update Images" in button.label).click().run()
assert not exceptions(at2), exceptions(at2)

infos = texts(at2, "info")
assert any("Stale entries removed: 8" in value for value in infos), infos
assert any("selfheal-1.jpg" in value for value in infos), infos
assert any("Batch Summary:" in value for value in infos), infos
assert any(
    metric.label == "Total Faces" and str(metric.value) == "4"
    for metric in at2.metric
), [(metric.label, metric.value) for metric in at2.metric]
print("UPDATE_INFO:", infos[:4])
print("UPDATE_METRICS:", [(metric.label, metric.value) for metric in at2.metric])

faces = store.get_all_faces_metadata()
assert len(faces) == 4, [face["metadata"]["image_path"] for face in faces]
assert Path(faces[0]["metadata"]["image_path"]).name == SOURCE_A.name
stats = store.get_collection_stats(force_refresh=True)
assert stats["total_faces"] == 4 and stats["unique_images"] == 1, stats
print("DB_AFTER:", len(faces))
print("DB_STATS_AFTER:", stats)
print("DB_PATH:", VECTOR_DB_PATH)
print("STALE_REMOVED: 8")

at2.sidebar.selectbox[0].set_value(GALLERY).run()
assert not exceptions(at2), exceptions(at2)
assert any(
    metric.label == "Faces" and str(metric.value) == "4"
    for metric in at2.metric
), [(metric.label, metric.value) for metric in at2.metric]
assert len(images(at2)) == 4, len(images(at2))
assert not any(
    "Image not found" in value or "Could not load image" in value
    for value in texts(at2, "error", "warning")
), texts(at2, "error", "warning")
print("GALLERY_IMAGES:", len(images(at2)))

at2.sidebar.selectbox[0].set_value(NAMES).run()
assert not exceptions(at2), exceptions(at2)
assert any("No names assigned yet" in value for value in texts(at2, "info")), texts(at2, "info")
print("NAME_GALLERY_EMPTY:", True)

at2.sidebar.selectbox[0].set_value(SEARCH).run()
at2.file_uploader[0].set_value((SOURCE_A.name, SOURCE_A.read_bytes(), "image/jpeg")).run()
assert not exceptions(at2), exceptions(at2)
next(button for button in at2.button if "Detect faces" in button.label).click().run()
assert not exceptions(at2), exceptions(at2)
next(button for button in at2.button if "Search face 1" in button.label).click().run()
assert not exceptions(at2), exceptions(at2)

results = at2.session_state["search_results"]
assert results, "search returned no results"
assert results[0]["similarity"] > 0.99, results[0]["similarity"]
assert all(
    Path(result["metadata"]["image_path"]).name == SOURCE_A.name
    for result in results
), [result["metadata"]["image_path"] for result in results]
print("SEARCH_RESULTS:", len(results))
print("SEARCH_TOP:", round(results[0]["similarity"], 4))
print("SELFHEAL_E2E_OK")
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

    assert completed.returncode == 0, output[-7000:]
    assert "SELFHEAL_E2E_OK" in output, output[-7000:]
    assert "DB_BEFORE: 12" in output, output[-7000:]
    assert "DB_AFTER: 4" in output, output[-7000:]
    assert "SEARCH_TOP: 1.0" in output, output[-7000:]
    assert "STALE_REMOVED: 8" in output, output[-7000:]
