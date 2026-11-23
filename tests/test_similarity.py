import numpy as np
import pytest
from settings_store import active_embedding_pipeline_version

from similarity import (
    cosine_from_chroma_l2_distance,
    cosine_from_l2_distance,
    cosine_similarity,
    normalise_embedding,
    similarity_from_chroma_l2_distance,
    similarity_from_l2_distance,
)


def test_zero_distance_is_a_perfect_match():
    assert cosine_from_l2_distance(0.0) == 1.0
    assert similarity_from_l2_distance(0.0) == 1.0


def test_orthogonal_vectors_have_no_similarity():
    # distance between orthogonal unit vectors is sqrt(2)
    assert cosine_from_l2_distance(np.sqrt(2)) == pytest.approx(0.0, abs=1e-9)
    assert similarity_from_l2_distance(np.sqrt(2)) == pytest.approx(0.0, abs=1e-9)


def test_distance_to_similarity_is_monotonic():
    values = [similarity_from_l2_distance(distance / 10) for distance in range(0, 20)]
    assert values == sorted(values, reverse=True)


def test_similarity_stays_within_bounds():
    assert similarity_from_l2_distance(2.0) == 0.0
    assert similarity_from_l2_distance(-1.0) == 1.0


def test_chroma_squared_distance_is_converted_back_to_cosine():
    # ChromaDB's L2 space returns squared distances. For unit vectors a
    # squared distance of 0.52 means cosine 1 - 0.52/2 = 0.74.
    assert cosine_from_chroma_l2_distance(0.52) == pytest.approx(0.74, abs=1e-6)
    assert similarity_from_chroma_l2_distance(0.52) == pytest.approx(0.74, abs=1e-6)
    assert cosine_from_chroma_l2_distance(2.0) == pytest.approx(0.0, abs=1e-9)


def test_normalise_embedding_returns_unit_length():
    normalised = normalise_embedding(np.array([3.0, 4.0, 0.0]))
    assert np.linalg.norm(normalised) == pytest.approx(1.0)
    assert np.allclose(normalised, np.array([0.6, 0.8, 0.0], dtype=np.float32))


def test_normalise_embedding_handles_zero_vector():
    assert not np.any(normalise_embedding(np.zeros(512)))


def test_cosine_similarity_of_identical_vectors():
    vector = np.array([0.2, 0.5, -0.3, 0.8])
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_search_keeps_candidates_above_user_threshold_below_75_percent(tmp_path):
    from vector_store import FaceVectorStore

    store = FaceVectorStore(
        db_path=str(tmp_path / "low-score-vectors"),
        collection_name="low_score_search",
    )
    version = active_embedding_pipeline_version()
    anchor = np.zeros(512, dtype=np.float32)
    anchor[0] = 1.0
    candidate = np.zeros(512, dtype=np.float32)
    candidate[0] = 0.60
    candidate[1] = np.sqrt(1.0 - 0.60**2)
    candidate = candidate.astype(np.float32)
    store.collection.add(
        ids=["candidate-60"],
        embeddings=[candidate.tolist()],
        metadatas=[{
            "face_id": "candidate-60",
            "image_path": str(tmp_path / "candidate.jpg"),
            "embedding_pipeline_version": version,
        }],
    )

    matches = store.search_similar_faces(anchor, n_results=10, min_similarity=0.4)

    assert [match["face_id"] for match in matches] == ["candidate-60"]
    assert matches[0]["similarity"] == pytest.approx(0.60, abs=1e-5)
    assert matches[0]["ranking_score"] == pytest.approx(0.60, abs=1e-5)
    assert "confidence_score" not in matches[0]
