from __future__ import annotations

import numpy as np

from settings_store import active_embedding_pipeline_metadata
from image_provenance import write_image_provenance
from vector_store import FaceVectorStore


def test_pipeline_identity_fields_persist_in_collection_and_face_rows(tmp_path):
    store = FaceVectorStore(
        db_path=str(tmp_path / "vectors"),
        collection_name="pipeline_metadata",
    )
    expected = active_embedding_pipeline_metadata()
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1.0
    assert store.add_face_embedding(
        "metadata-face",
        vector,
        {"face_id": "metadata-face", "image_path": "/synthetic/face.jpg"},
    )

    collection_metadata = dict(store.collection.metadata)
    row = store.collection.get(ids=["metadata-face"], include=["metadatas"])
    row_metadata = row["metadatas"][0]
    for key, value in expected.items():
        assert collection_metadata[key] == value
        assert row_metadata[key] == value
    assert collection_metadata["embedding_pipeline_version"] == store._embedding_pipeline_version
    assert row_metadata["embedding_pipeline_version"] == store._embedding_pipeline_version

    reopened = FaceVectorStore(
        db_path=str(tmp_path / "vectors"),
        collection_name="pipeline_metadata",
    )
    assert dict(reopened.collection.metadata)["index_schema_version"] == "face-index-v1"
    assert reopened.collection.get(
        ids=["metadata-face"], include=["metadatas"]
    )["metadatas"][0]["embedding_model"] == expected["embedding_model"]


def test_web_source_provenance_is_saved_with_indexed_face_metadata(tmp_path):
    image = tmp_path / "portrait.jpg"
    image.write_bytes(b"synthetic image bytes")
    record = write_image_provenance(
        image,
        source_url="https://example.test/photos/portrait.jpg",
        authorization_confirmed=True,
    )
    store = FaceVectorStore(
        db_path=str(tmp_path / "vectors"),
        collection_name="provenance_metadata",
    )
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1.0

    assert store.add_face_embedding(
        "provenance-face",
        vector,
        {"face_id": "provenance-face", "image_path": str(image)},
    )
    metadata = store.get_face_by_id("provenance-face")["metadata"]
    assert metadata["source_type"] == "web_import"
    assert metadata["source_url"] == record["source_url"]
    assert metadata["collected_at"] == record["collected_at"]
    assert metadata["authorization_confirmed"] is True
