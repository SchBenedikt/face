"""Face identity and batch writes must be stable across reprocessing.

The face id used to be derived from ``image_path.stem`` only. Two different
files called ``foto.jpg`` therefore produced the same ids, and processing an
image a second time tried to add those ids again instead of replacing the old
rows. These tests pin the corrected behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_recognition_engine import FaceRecognitionEngine, build_face_id
from vector_store import FaceVectorStore
from settings_store import active_embedding_pipeline_version

EMBEDDING_PIPELINE_VERSION = active_embedding_pipeline_version()


def _embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=512).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _write_image(path: Path, colour=(120, 80, 40)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (480, 360), colour).save(path, format="JPEG")
    return path


def test_face_id_depends_on_the_full_path_and_face_index(tmp_path):
    first = tmp_path / "a" / "same.jpg"
    second = tmp_path / "b" / "same.jpg"

    assert build_face_id(first, 0) != build_face_id(second, 0)
    assert build_face_id(first, 0) != build_face_id(first, 1)
    assert build_face_id(first, 0) == build_face_id(first, 0)


def test_processed_face_ids_are_unique_for_equal_file_names(tmp_path, monkeypatch):
    engine = FaceRecognitionEngine()
    # Do not download/run the real DeepFace weights in a unit test.
    monkeypatch.setattr(
        engine,
        "_extract_single_model_embedding",
        lambda face_image, model_name: np.linspace(0.1, 0.9, 512, dtype=np.float32),
    )

    first = _write_image(tmp_path / "a" / "foto.jpg", (120, 80, 40))
    second = _write_image(tmp_path / "b" / "foto.jpg", (40, 80, 120))

    first_result = engine.process_image(first)
    second_result = engine.process_image(second)

    first_ids = {face["face_id"] for face in first_result["faces"]}
    second_ids = {face["face_id"] for face in second_result["faces"]}

    assert first_ids, first_result
    assert second_ids, second_result
    assert first_ids.isdisjoint(second_ids)


def test_batch_add_replaces_legacy_rows_for_the_same_image(tmp_path):
    store = FaceVectorStore(
        db_path=str(tmp_path / "vectors"), collection_name="index_integrity"
    )
    image = tmp_path / "foto.jpg"
    image.write_bytes(b"placeholder")
    image_path = str(image)

    store.collection.add(
        ids=["legacy_0", "legacy_1"],
        embeddings=[_embedding(1).tolist(), _embedding(2).tolist()],
        metadatas=[
            {"image_path": image_path, "location": "0,10,10,0", "face_id": "legacy_0"},
            {"image_path": image_path, "location": "20,30,30,20", "face_id": "legacy_1"},
        ],
    )

    new_ids = [build_face_id(image, 0), build_face_id(image, 1)]
    metadatas = [
        {"image_path": image_path, "location": "0,20,20,0", "face_id": new_ids[0]},
        {"image_path": image_path, "location": "20,40,40,20", "face_id": new_ids[1]},
    ]

    added = store.add_face_embeddings_batch(
        new_ids, [_embedding(3), _embedding(4)], metadatas
    )

    assert added == 2
    assert store.collection.count() == 2
    stored = store.collection.get(include=["metadatas"])
    assert set(stored["ids"]) == set(new_ids)
    assert all(metadata["image_path"] == image_path for metadata in stored["metadatas"])


def test_search_excludes_legacy_vectors_until_reprocessed(tmp_path):
    store = FaceVectorStore(
        db_path=str(tmp_path / "vectors"), collection_name="pipeline_migration"
    )
    query = np.zeros(512, dtype=np.float32)
    query[0] = 1.0
    current = np.zeros(512, dtype=np.float32)
    current[1] = 1.0
    store.collection.modify(metadata={"description": "legacy index"})
    store.collection.add(
        ids=["legacy"],
        embeddings=[query.tolist()],
        metadatas=[{"image_path": "/legacy.jpg", "face_id": "legacy"}],
    )

    assert store.add_face_embedding(
        "current", current, {"image_path": "/current.jpg", "face_id": "current"}
    )
    results = store.search_similar_faces(current, min_similarity=0.5)
    assert [result["face_id"] for result in results] == ["current"]
    status = store.get_embedding_pipeline_status()
    assert status["reindex_in_progress"]
    assert status["pending_faces"] == 1

    assert store.add_face_embedding(
        "legacy", current, {"image_path": "/legacy.jpg", "face_id": "legacy"}
    )
    status = store.get_embedding_pipeline_status()
    assert status["complete"]
    assert not status["reindex_in_progress"]


@pytest.mark.parametrize("space", ["l2", "cosine", "ip"])
def test_search_scores_follow_collection_distance_space(tmp_path, space):
    store = FaceVectorStore(
        db_path=str(tmp_path / f"vectors_{space}"),
        collection_name=f"metric_{space}",
    )
    store.client.delete_collection(name=store.collection_name)
    store.collection = store.client.create_collection(
        name=store.collection_name,
        configuration={"hnsw": {"space": space}},
        metadata={
            "description": "metric scoring regression",
            "embedding_pipeline_version": EMBEDDING_PIPELINE_VERSION,
            "embedding_pipeline_complete": True,
        },
    )
    store._pipeline_status_cache = None

    target = np.array([1.0, 0.0], dtype=np.float32)
    query = np.array([0.74, np.sqrt(1.0 - 0.74**2)], dtype=np.float32)
    assert store.add_face_embedding(
        "match",
        target,
        {"image_path": "/match.jpg", "face_id": "match"},
    )

    results = store.search_similar_faces(query, n_results=1, min_similarity=0.73)

    assert len(results) == 1
    assert results[0]["cosine_similarity"] == pytest.approx(0.74, abs=0.01)
    assert results[0]["similarity"] == pytest.approx(0.74, abs=0.01)
    assert results[0]["euclidean_distance"] == pytest.approx(np.sqrt(0.52), abs=0.01)
    assert results[0]["chroma_distance"] == pytest.approx(
        0.52 if space == "l2" else 0.26, abs=0.01
    )


def test_large_search_request_raises_hnsw_search_breadth(tmp_path):
    store = FaceVectorStore(
        db_path=str(tmp_path / "vectors_ef_search"),
        collection_name="ef_search_test",
    )
    store.client.delete_collection(name=store.collection_name)
    store.collection = store.client.create_collection(
        name=store.collection_name,
        configuration={"hnsw": {"space": "l2", "ef_search": 10}},
        metadata={
            "embedding_pipeline_version": EMBEDDING_PIPELINE_VERSION,
            "embedding_pipeline_complete": True,
        },
    )
    store._pipeline_status_cache = None

    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(80, 512)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    store.collection.add(
        ids=[f"face_{index}" for index in range(len(embeddings))],
        embeddings=embeddings.tolist(),
        metadatas=[
            {"face_id": f"face_{index}", "embedding_pipeline_version": EMBEDDING_PIPELINE_VERSION}
            for index in range(len(embeddings))
        ],
    )

    store.search_similar_faces(embeddings[0], n_results=20, min_similarity=0.0)

    assert store.collection.configuration["hnsw"]["ef_search"] >= 60


def test_reprocessing_the_same_image_is_idempotent(tmp_path):
    store = FaceVectorStore(
        db_path=str(tmp_path / "vectors"), collection_name="idempotent"
    )
    image = _write_image(tmp_path / "foto.jpg")
    image_path = str(image)
    ids = [build_face_id(image, 0), build_face_id(image, 1)]
    metadatas = [
        {"image_path": image_path, "location": "0,20,20,0", "face_id": ids[0]},
        {"image_path": image_path, "location": "20,40,40,20", "face_id": ids[1]},
    ]

    assert store.add_face_embeddings_batch(ids, [_embedding(5), _embedding(6)], metadatas) == 2
    assert store.add_face_embeddings_batch(ids, [_embedding(7), _embedding(8)], metadatas) == 2

    assert store.collection.count() == 2


def test_pruning_removes_only_faces_whose_file_is_gone(tmp_path):
    store = FaceVectorStore(
        db_path=str(tmp_path / "vectors"), collection_name="pruning"
    )
    present = _write_image(tmp_path / "present.jpg")
    missing = tmp_path / "missing.jpg"

    store.collection.add(
        ids=["present_0", "present_1", "missing_0"],
        embeddings=[_embedding(9).tolist(), _embedding(10).tolist(), _embedding(11).tolist()],
        metadatas=[
            {"image_path": str(present), "location": "0,20,20,0", "face_id": "present_0"},
            {"image_path": str(present), "location": "20,40,40,20", "face_id": "present_1"},
            {"image_path": str(missing), "location": "0,20,20,0", "face_id": "missing_0"},
        ],
    )

    removed = store.prune_missing_faces()

    assert removed == 1
    stored = store.collection.get(include=["metadatas"])
    assert set(stored["ids"]) == {"present_0", "present_1"}


def test_fast_process_filters_on_the_full_path_not_the_file_name(tmp_path):
    from fast_process import filter_unprocessed_images

    first = tmp_path / "a" / "same.jpg"
    second = tmp_path / "b" / "same.jpg"
    processed = {str(first.resolve())}

    remaining = filter_unprocessed_images([first, second], processed)

    assert remaining == [second]
