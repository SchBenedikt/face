from __future__ import annotations

import numpy as np
import subprocess
import sys
from pathlib import Path

from privacy_data import delete_local_biometric_data
from duplicate_detector import SimpleDuplicateDetector
from vector_store import FaceVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUBS_DIR = PROJECT_ROOT / "tests" / "stubs"
APP_FILE = PROJECT_ROOT / "app.py"


def _store(tmp_path, image_path):
    store = FaceVectorStore(
        db_path=str(tmp_path / "vectors"), collection_name="privacy_delete"
    )
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    assert store.add_face_embedding(
        "private-face",
        embedding,
        {
            "face_id": "private-face",
            "image_path": str(image_path),
            "person_id": "private-person",
            "full_name": "Private Person",
        },
    )
    return store


def test_full_privacy_delete_removes_index_images_archives_and_derived_data(tmp_path):
    source = tmp_path / "external-library" / "portrait.jpg"
    source.parent.mkdir()
    source.write_bytes(b"private image")
    sidecar = source.with_name(source.name + ".provenance.json")
    sidecar.write_text("{\"source_url\": \"https://example.test/image.jpg\"}")
    managed = tmp_path / "data" / "images"
    managed.mkdir(parents=True)
    (managed / "unindexed.jpg").write_bytes(b"unindexed image")
    backup = tmp_path / "data" / "duplicate_backups"
    backup.mkdir(parents=True)
    archive = backup / "recovery.zip"
    archive.write_bytes(b"biometric backup")
    derived = tmp_path / "data" / "embeddings"
    derived.mkdir(parents=True)
    (derived / "cache.npy").write_bytes(b"derived data")
    unrelated = tmp_path / "outside.txt"
    unrelated.write_text("keep", encoding="utf-8")
    store = _store(tmp_path, source)
    metadata_store = SimpleDuplicateDetector(str(tmp_path / "hashes.db"))
    metadata_store.add_file_hash(source)
    legacy_hash_db = tmp_path / "legacy-duplicates.db"
    legacy_hash_db.write_bytes(b"legacy metadata")

    outcome = delete_local_biometric_data(
        store,
        delete_source_images=True,
        managed_image_directories=[managed],
        derived_directories=[derived],
        backup_directories=[backup],
        metadata_stores=[metadata_store],
        auxiliary_files=[legacy_hash_db],
    )

    assert outcome["success"], outcome
    assert outcome["faces_removed"] == 1
    assert outcome["images_removed"] == 2
    assert outcome["derived_entries_removed"] == 1
    assert outcome["backup_files_removed"] == 1
    assert outcome["auxiliary_files_removed"] == 1
    assert store.collection.count() == 0
    assert not source.exists()
    assert not sidecar.exists()
    assert not (managed / "unindexed.jpg").exists()
    assert not archive.exists()
    assert not (derived / "cache.npy").exists()
    assert metadata_store.get_stats()["total_files"] == 0
    assert not legacy_hash_db.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_index_only_privacy_delete_keeps_original_source_images(tmp_path):
    source = tmp_path / "external-library" / "portrait.jpg"
    source.parent.mkdir()
    source.write_bytes(b"private image")
    sidecar = source.with_name(source.name + ".provenance.json")
    sidecar.write_text("{\"source_url\": \"https://example.test/image.jpg\"}")
    backup = tmp_path / "data" / "pipeline_backups"
    backup.mkdir(parents=True)
    (backup / "index.zip").write_bytes(b"biometric backup")
    store = _store(tmp_path, source)

    outcome = delete_local_biometric_data(
        store,
        delete_source_images=False,
        managed_image_directories=[],
        derived_directories=[],
        backup_directories=[backup],
    )

    assert outcome["success"], outcome
    assert outcome["faces_removed"] == 1
    assert outcome["images_removed"] == 0
    assert store.collection.count() == 0
    assert source.read_bytes() == b"private image"
    assert not sidecar.exists()
    assert not (backup / "index.zip").exists()


def test_settings_privacy_control_deletes_local_data_end_to_end(tmp_path):
    data_dir = tmp_path / "app-data"
    source = data_dir / "images" / "portrait.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"private image")
    script = f"""
import os, sys
from pathlib import Path
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(STUBS_DIR)!r})
os.environ["FACE_DATA_DIR"] = {str(data_dir)!r}
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import numpy as np
from config import DATA_DIR, IMAGES_DIR
from vector_store import FaceVectorStore
store = FaceVectorStore()
embedding = np.zeros(512, dtype=np.float32)
embedding[0] = 1.0
assert store.add_face_embedding("private-ui-face", embedding, {{
    "face_id": "private-ui-face", "image_path": {str(source)!r},
    "person_id": "private-ui-person", "full_name": "Private Person",
}})
backup = DATA_DIR / "pipeline_backups" / "recovery.zip"
backup.parent.mkdir(parents=True, exist_ok=True)
backup.write_bytes(b"biometric backup")
derived = DATA_DIR / "embeddings" / "derived.npy"
derived.parent.mkdir(parents=True, exist_ok=True)
derived.write_bytes(b"derived data")

from duplicate_detector import duplicate_detector
duplicate_detector.add_file_hash(Path({str(source)!r}))
from streamlit.testing.v1 import AppTest
at = AppTest.from_file({str(APP_FILE)!r}, default_timeout=180)
at.run()
assert not at.exception, [str(element.value) for element in at.exception]
at.sidebar.selectbox[0].set_value("settings").run()
assert not at.exception, [str(element.value) for element in at.exception]
next(element for element in at.checkbox if element.key == "delete_biometric_source_images").set_value(True).run()
at.text_input[0].set_value("DELETE").run()
next(button for button in at.button if button.label == "Permanently delete local biometric data").click().run()
assert not at.exception, [str(element.value) for element in at.exception]
assert FaceVectorStore().collection.count() == 0
assert not Path({str(source)!r}).exists()
assert not backup.exists()
assert not derived.exists()
assert duplicate_detector.get_stats()["total_files"] == 0
print("PRIVACY_UI_DELETE_OK")
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
        timeout=180,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output[-5000:]
    assert "PRIVACY_UI_DELETE_OK" in output, output[-5000:]
