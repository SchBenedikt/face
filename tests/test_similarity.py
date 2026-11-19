import numpy as np
import pytest

from similarity import (
    cosine_from_l2_distance,
    cosine_similarity,
    normalise_embedding,
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


def test_normalise_embedding_returns_unit_length():
    normalised = normalise_embedding(np.array([3.0, 4.0, 0.0]))
    assert np.linalg.norm(normalised) == pytest.approx(1.0)
    assert np.allclose(normalised, np.array([0.6, 0.8, 0.0], dtype=np.float32))


def test_normalise_embedding_handles_zero_vector():
    assert not np.any(normalise_embedding(np.zeros(512)))


def test_cosine_similarity_of_identical_vectors():
    vector = np.array([0.2, 0.5, -0.3, 0.8])
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)
