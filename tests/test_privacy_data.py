from __future__ import annotations

import numpy as np

from privacy_data import delete_local_biometric_data
from vector_store import FaceVectorStore


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

    outcome = delete_local_biometric_data(
        store,
        delete_source_images=True,
        managed_image_directories=[managed],
        derived_directories=[derived],
        backup_directories=[backup],
    )

    assert outcome["success"], outcome
    assert outcome["faces_removed"] == 1
    assert outcome["images_removed"] == 2
    assert outcome["derived_entries_removed"] == 1
    assert outcome["backup_files_removed"] == 1
    assert store.collection.count() == 0
    assert not source.exists()
    assert not (managed / "unindexed.jpg").exists()
    assert not archive.exists()
    assert not (derived / "cache.npy").exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_index_only_privacy_delete_keeps_original_source_images(tmp_path):
    source = tmp_path / "external-library" / "portrait.jpg"
    source.parent.mkdir()
    source.write_bytes(b"private image")
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
    assert not (backup / "index.zip").exists()
