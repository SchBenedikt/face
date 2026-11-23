"""Portable face archives round-trip through the real temporary Chroma store."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from portable_export import create_face_export, restore_face_export
from settings_store import active_embedding_pipeline_version
from vector_store import FaceVectorStore


def _store(path, name):
    return FaceVectorStore(db_path=str(path), collection_name=name)


def test_export_and_restore_preserve_embeddings_metadata_and_optional_image(tmp_path):
    source_image = tmp_path / "source.jpg"
    source_image.write_bytes(b"synthetic image payload")
    source = _store(tmp_path / "source-db", "export_source")
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1
    assert source.add_face_embedding("face-1", vector, {
        "face_id": "face-1",
        "image_path": str(source_image),
        "full_name": "Example Person",
        "source_url": "https://example.test/portrait.jpg",
        "location": "10,90,90,10",
    })

    archive = create_face_export(
        source.collection,
        source._embedding_pipeline_version,
        face_ids=["face-1"],
        include_embeddings=True,
        include_images=True,
    )
    restored = _store(tmp_path / "restored-db", "export_restored")
    outcome = restore_face_export(archive, restored, tmp_path / "restored-images")

    assert outcome == {"success": True, "restored_faces": 1, "restored_images": 1}
    face = restored.get_face_by_id("face-1")
    assert face["metadata"]["full_name"] == "Example Person"
    assert face["metadata"]["source_url"] == "https://example.test/portrait.jpg"
    assert face["metadata"]["embedding_pipeline_version"] == active_embedding_pipeline_version()
    np.testing.assert_allclose(face["embedding"], vector, atol=1e-6)
    assert open(face["metadata"]["image_path"], "rb").read() == source_image.read_bytes()


def test_restore_rejects_integrity_tampering_without_writing_rows(tmp_path):
    source = _store(tmp_path / "source-db", "tamper_source")
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1
    source.add_face_embedding("tamper-face", vector, {"face_id": "tamper-face"})
    valid = create_face_export(
        source.collection, source._embedding_pipeline_version, include_embeddings=True
    )
    replacement = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(valid), "r") as original, zipfile.ZipFile(replacement, "w") as edited:
        for member in original.namelist():
            content = original.read(member)
            if member.startswith("records/"):
                content = content.replace(b"tamper-face", b"changed-face")
            edited.writestr(member, content)

    target = _store(tmp_path / "target-db", "tamper_target")
    result = restore_face_export(replacement.getvalue(), target, tmp_path / "images")

    assert not result["success"]
    assert "Integrity check failed" in result["error"]
    assert target.collection.count() == 0


def test_restore_rejects_duplicate_ids_and_metadata_only_archives(tmp_path):
    source = _store(tmp_path / "source-db", "duplicate_source")
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1
    source.add_face_embedding("same-face", vector, {"face_id": "same-face"})
    archive = create_face_export(
        source.collection, source._embedding_pipeline_version, include_embeddings=True
    )
    target = _store(tmp_path / "target-db", "duplicate_target")
    target.add_face_embedding("same-face", vector, {"face_id": "same-face"})

    duplicate = restore_face_export(archive, target, tmp_path / "images")
    metadata_only = create_face_export(source.collection, source._embedding_pipeline_version)
    metadata_result = restore_face_export(metadata_only, _store(tmp_path / "empty-db", "metadata_target"), tmp_path / "images")

    assert not duplicate["success"]
    assert "already present" in duplicate["error"]
    assert target.collection.count() == 1
    assert not metadata_result["success"]
    assert "metadata only" in metadata_result["error"]
