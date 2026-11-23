from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path

from duplicate_backup import create_duplicate_backup, restore_duplicate_backup
from vector_store import FaceVectorStore


def test_duplicate_cleanup_backup_restores_faces_and_image(tmp_path):
    store = FaceVectorStore(
        db_path=str(tmp_path / "vectors"),
        collection_name="duplicate_backup_roundtrip",
    )
    image_path = tmp_path / "images" / "duplicate.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (80, 80), (120, 90, 60)).save(image_path)
    vector = np.zeros(512, dtype=np.float32)
    vector[3] = 1.0
    face_id = "duplicate-backup-face"
    assert store.add_face_embedding(
        face_id,
        vector,
        {"face_id": face_id, "image_path": str(image_path), "location": "10,70,70,10"},
    )

    created = create_duplicate_backup(store, [image_path], tmp_path / "backups")
    assert created["success"] is True
    archive_path = created["backup_path"]
    assert archive_path.is_file()
    assert created["face_count"] == 1

    assert store.delete_face(face_id) is True
    image_path.unlink()
    restored = restore_duplicate_backup(archive_path, store, tmp_path / "restored-images")

    assert restored["success"] is True, restored
    assert restored["restored_faces"] == 1
    assert restored["restored_images"] == 1
    assert not archive_path.exists()
    record = store.get_face_by_id(face_id)
    assert record is not None
    assert (tmp_path / "restored-images" / "restored").is_dir()
    assert Path(record["metadata"]["image_path"]).is_file()


def test_duplicate_backup_without_indexed_faces_needs_no_archive(tmp_path):
    store = FaceVectorStore(
        db_path=str(tmp_path / "vectors"),
        collection_name="duplicate_backup_empty",
    )
    result = create_duplicate_backup(store, [tmp_path / "no-face.jpg"], tmp_path / "backups")

    assert result["success"] is True
    assert result["backup_path"] is None
    assert result["face_count"] == 0


def test_duplicate_recovery_skips_faces_left_by_a_partial_cleanup(tmp_path):
    store = FaceVectorStore(
        db_path=str(tmp_path / "vectors"),
        collection_name="duplicate_backup_partial",
    )
    image_paths = [tmp_path / "images" / f"duplicate-{index}.jpg" for index in range(2)]
    for index, image_path in enumerate(image_paths):
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 80), (120 + index, 90, 60)).save(image_path)
        vector = np.zeros(512, dtype=np.float32)
        vector[index] = 1.0
        assert store.add_face_embedding(
            f"partial-{index}",
            vector,
            {"face_id": f"partial-{index}", "image_path": str(image_path), "location": "10,70,70,10"},
        )

    created = create_duplicate_backup(store, image_paths, tmp_path / "backups")
    assert created["success"]
    assert store.delete_face("partial-0")

    restored = restore_duplicate_backup(created["backup_path"], store, tmp_path / "restored-images")

    assert restored["success"] is True, restored
    assert restored["restored_faces"] == 1
    assert restored["skipped_existing"] == 1
    assert store.get_face_by_id("partial-0") is not None
    assert store.get_face_by_id("partial-1") is not None


def test_duplicate_recovery_repairs_a_stale_vector_after_a_crash(tmp_path):
    store = FaceVectorStore(
        db_path=str(tmp_path / "vectors"),
        collection_name="duplicate_backup_stale_row",
    )
    image_path = tmp_path / "images" / "crash.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (80, 80), (12, 90, 60)).save(image_path)
    vector = np.zeros(512, dtype=np.float32)
    vector[4] = 1.0
    assert store.add_face_embedding(
        "stale-after-crash",
        vector,
        {"face_id": "stale-after-crash", "image_path": str(image_path), "location": "10,70,70,10"},
    )
    created = create_duplicate_backup(store, [image_path], tmp_path / "backups")
    image_path.unlink()  # Simulate a crash before the vector row was removed.

    restored = restore_duplicate_backup(created["backup_path"], store, tmp_path / "recovered")

    assert restored["success"] is True, restored
    assert restored["restored_faces"] == 1
    record = store.get_face_by_id("stale-after-crash")
    assert record is not None
    assert Path(record["metadata"]["image_path"]).is_file()
