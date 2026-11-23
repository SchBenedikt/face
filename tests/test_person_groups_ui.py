"""End-to-end test for the reviewed identity-grouping page.

Eight synthetic faces belong to two identities. The page must show two
four-face groups, and assigning the first group must name exactly those four
faces while leaving the other identity untouched.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUBS_DIR = PROJECT_ROOT / "tests" / "stubs"
APP_FILE = PROJECT_ROOT / "app.py"


def test_person_groups_page_assigns_one_reviewed_identity(tmp_path):
    data_dir = tmp_path / "groups-data"

    script = f"""
import os
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(STUBS_DIR)!r})

from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from config import IMAGES_DIR
from vector_store import FaceVectorStore

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261121)
basis, _ = np.linalg.qr(rng.normal(size=(512, 12)))
basis = basis.T.astype(np.float32)


def variation(base, cosine, orthogonal):
    return (
        np.float32(cosine) * base
        + np.sqrt(1.0 - cosine * cosine) * orthogonal
    ).astype(np.float32)


a0, b0 = basis[0], basis[1]
embeddings = {{
    "A0": a0,
    "A1": variation(a0, 0.74, basis[2]),
    "A2": variation(a0, 0.78, basis[3]),
    "A3": variation(a0, 0.86, basis[4]),
    "B0": b0,
    "B1": variation(b0, 0.72, basis[5]),
    "B2": variation(b0, 0.81, basis[6]),
    "B3": variation(b0, 0.85, basis[7]),
}}
FACES_A = {{"A0", "A1", "A2", "A3"}}
FACES_B = {{"B0", "B1", "B2", "B3"}}

store = FaceVectorStore()
for index, (face_id, embedding) in enumerate(embeddings.items()):
    image_path = IMAGES_DIR / f"{{face_id}}.jpg"
    Image.new("RGB", (400, 400), (40 + index * 20, 80, 160)).save(
        image_path, format="JPEG"
    )
    store.add_face_embedding(face_id, embedding, {{
        "face_id": face_id,
        "image_path": str(image_path),
        "location": "10,210,210,10",
    }})

at = AppTest.from_file({str(APP_FILE)!r}, default_timeout=180)
at.run()
assert not at.exception, [str(element.value) for element in at.exception]
at.sidebar.selectbox[0].set_value("person_groups").run()
assert not at.exception, [str(element.value) for element in at.exception]

metrics = {{metric.label: str(metric.value) for metric in at.metric}}
assert metrics.get("Unassigned faces") == "8", metrics
assert metrics.get("Groups") == "2", metrics
assert metrics.get("Faces in groups") == "8", metrics

expanders = [getattr(element, "label", "") for element in at.expander]
assert len(expanders) == 2, expanders
assert all("4 faces" in label for label in expanders), expanders

cache = at.session_state["person_groups_cache"]
grouped = [set(group["face_ids"]) for group in cache["groups"]]
assert FACES_A in grouped, grouped
assert FACES_B in grouped, grouped
print("GROUPS:", [(group["size"], sorted(group["face_ids"])) for group in cache["groups"]])

first_name = next(element for element in at.text_input if element.label == "First name")
first_name.set_value("Anna").run()
last_name = next(element for element in at.text_input if element.label == "Last name")
last_name.set_value("Test").run()
assign_buttons = [button for button in at.button if "Assign group" in button.label]
assert len(assign_buttons) == 2, [button.label for button in at.button]
assign_buttons[0].click().run()
assert not at.exception, [str(element.value) for element in at.exception]
assert any(
    "Assigned 4 faces to Anna Test" in str(message.value)
    for message in at.success
), [str(message.value) for message in at.success]

faces = store.get_all_faces_metadata()
named = [
    face for face in faces
    if (face["metadata"].get("person_id") or "").strip()
]
named_ids = {{face["face_id"] for face in named}}
assert len(named) == 4, sorted(named_ids)
assert named_ids in (FACES_A, FACES_B), sorted(named_ids)
assert {{face["metadata"].get("full_name") for face in named}} == {{"Anna Test"}}
unnamed = [
    face for face in faces
    if not (face["metadata"].get("person_id") or "").strip()
]
assert len(unnamed) == 4, [face["face_id"] for face in unnamed]
print("NAMED_IDS:", sorted(named_ids))
print("UNNAMED_COUNT:", len(unnamed))

at.sidebar.selectbox[0].set_value("people").run()
assert not at.exception, [str(element.value) for element in at.exception]
text = " ".join(str(element.value) for element in at.markdown)
assert "Anna" in text and "Test" in text, text[:500]
assert "1 people found" in text, text[:500]
print("NAME_GALLERY_OK")

at.sidebar.selectbox[0].set_value("person_groups").run()
assert not at.exception, [str(element.value) for element in at.exception]
undo = next(button for button in at.button if "Undo last group assignment" in button.label)
undo.click().run()
assert not at.exception, [str(element.value) for element in at.exception]
assert any("Undid 4 face assignment(s)" in str(item.value) for item in at.success)
assert all(not (face["metadata"].get("person_id") or "").strip() for face in store.get_all_faces_metadata())
print("GROUP_UNDO_OK")
print("PERSON_GROUPS_E2E_OK")
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
    assert "PERSON_GROUPS_E2E_OK" in output, output[-7000:]
    assert "GROUP_UNDO_OK" in output, output[-7000:]
    assert "NAMED_IDS: ['A0', 'A1', 'A2', 'A3']" in output or (
        "NAMED_IDS: ['B0', 'B1', 'B2', 'B3']" in output
    ), output[-7000:]
    assert "UNNAMED_COUNT: 4" in output, output[-7000:]
