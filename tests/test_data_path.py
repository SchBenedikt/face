"""End-to-end data path in an isolated, empty FACE_DATA_DIR.

The test uploads a synthetic image through the real Streamlit upload page,
processes it with the test stubs, reads the resulting rows from ChromaDB and
renders the gallery, a named gallery and a real similarity search. It runs in
a subprocess so none of the shared UI-test fixtures can pre-populate the data
directory, and so the stale "no faces" banner regression is observable.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUBS_DIR = PROJECT_ROOT / "tests" / "stubs"
APP_FILE = PROJECT_ROOT / "app.py"


def test_upload_process_gallery_names_and_search_in_empty_data_dir(tmp_path):
    data_dir = tmp_path / "fresh-data"
    source_image = tmp_path / "synthetic-portrait.jpg"
    Image.new("RGB", (640, 480), (180, 140, 120)).save(source_image, format="JPEG")
    assert not data_dir.exists()

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(STUBS_DIR)!r})

from pathlib import Path
from streamlit.testing.v1 import AppTest
from config import VECTOR_DB_PATH
from vector_store import FaceVectorStore

APP = {str(APP_FILE)!r}
SOURCE = Path({str(source_image)!r})
UPLOAD = "📥 Image Upload & Processing"
GALLERY = "👥 Face Gallery"
NAMES = "🏷️ Name Gallery"
SEARCH = "🔍 Face Search"


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


at = AppTest.from_file(APP, default_timeout=240)
at.run()
assert not exceptions(at), exceptions(at)

# Upload and process one image through the actual page.
at.sidebar.selectbox[0].set_value(UPLOAD).run()
at.file_uploader[0].set_value((SOURCE.name, SOURCE.read_bytes(), "image/jpeg")).run()
assert not exceptions(at), exceptions(at)
next(button for button in at.button if "Process and Add" in button.label).click().run()
assert not exceptions(at), exceptions(at)

success = texts(at, "success")
assert any("Successfully processed" in value for value in success), success
infos = texts(at, "info")
assert not any("Noch keine Gesichter in der Datenbank" in value for value in infos), infos
assert any(
    metric.label == "Total Faces" and str(metric.value) == "4"
    for metric in at.metric
), [(metric.label, metric.value) for metric in at.metric]
print("PROCESS_SUCCESS:", success)
print("PROCESS_SIDEBAR:", [(metric.label, metric.value) for metric in at.metric])

store = FaceVectorStore()
faces = store.get_all_faces_metadata()
assert len(faces) == 4, [(face["face_id"], face["metadata"]) for face in faces]
assert all(Path(face["metadata"]["image_path"]).is_file() for face in faces)
print("DB_FACES:", len(faces))
print("DB_PATH:", VECTOR_DB_PATH)
print("DB_STATS:", store.get_collection_stats(force_refresh=True))

person_id = store.assign_person_name(faces[0]["face_id"], {{
    "first_name": "Test",
    "middle_names": "",
    "last_name": "Person",
    "birth_date": "",
    "birth_place": "",
    "notes": "synthetic",
}})
assert person_id, "name assignment failed"
print("PERSON_ID:", person_id)

# Gallery and name gallery render the newly written rows.
at.sidebar.selectbox[0].set_value(GALLERY).run()
assert not exceptions(at), exceptions(at)
assert any(metric.label == "👥 Total Faces" and str(metric.value) == "4" for metric in at.metric)
gallery_images = images(at)
assert len(gallery_images) >= 4, len(gallery_images)
print("GALLERY_IMAGES:", len(gallery_images))

at.session_state["selected_person"] = person_id
at.sidebar.selectbox[0].set_value(NAMES).run()
assert not exceptions(at), exceptions(at)
expander_labels = [getattr(element, "label", "") for element in at.expander]
assert any("Test Person" in label for label in expander_labels), expander_labels
assert len(images(at)) >= 1, "named faces did not render"
print("NAME_GALLERY:", expander_labels)

# Upload the same image as a query and run the real search path.
at.sidebar.selectbox[0].set_value(SEARCH).run()
at.file_uploader[0].set_value((SOURCE.name, SOURCE.read_bytes(), "image/jpeg")).run()
assert not exceptions(at), exceptions(at)
next(button for button in at.button if "Analyze Image for Faces" in button.label).click().run()
assert not exceptions(at), exceptions(at)
assert at.session_state["detected_faces"], "query image yielded no faces"
next(button for button in at.button if "Nach Gesicht 1 suchen" in button.label).click().run()
assert not exceptions(at), exceptions(at)

results = at.session_state["search_results"]
assert results, "similarity search returned no results"
assert results[0]["similarity"] > 0.99, results[0]["similarity"]
assert any("Enhanced Search Results" in value for value in texts(at, "subheader")), texts(at, "subheader")
print("SEARCH_RESULTS:", len(results))
print("SEARCH_TOP:", round(results[0]["similarity"], 4))
print("SEARCH_IMAGES:", len(images(at)))
print("E2E_OK")
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
        timeout=240,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output[-5000:]
    assert "E2E_OK" in output, output[-5000:]
    assert "DB_FACES: 4" in output, output[-5000:]
    assert "SEARCH_TOP: 1.0" in output, output[-5000:]
