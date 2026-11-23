import numpy as np
import pytest

from face_utils import FaceSimilarityCalculator


@pytest.mark.parametrize("algorithm", ["basic", "enhanced", "premium"])
def test_all_legacy_algorithms_use_the_same_cosine_score(algorithm):
    calculator = FaceSimilarityCalculator()
    result = calculator.calculate_comprehensive_similarity(
        np.array([3.0, 0.0, 0.0]),
        np.array([4.0, 3.0, 0.0]),
        algorithm,
    )

    assert result["cosine_similarity"] == pytest.approx(0.8)
    assert result["final_similarity"] == pytest.approx(0.8)
    assert 0.0 <= result["final_similarity"] <= 1.0


def test_identical_and_orthogonal_vectors_have_expected_scores():
    calculator = FaceSimilarityCalculator()
    vector = np.array([1.0, -2.0, 3.0])

    identical = calculator.calculate_comprehensive_similarity(vector, vector)
    orthogonal = calculator.calculate_comprehensive_similarity(
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    )

    assert identical["final_similarity"] == pytest.approx(1.0)
    assert orthogonal["final_similarity"] == pytest.approx(0.0)


def test_negative_cosine_is_clipped_only_for_the_nonnegative_ranking_score():
    result = FaceSimilarityCalculator().calculate_comprehensive_similarity(
        np.array([1.0, 0.0]), np.array([-1.0, 0.0])
    )

    assert result["cosine_similarity"] == pytest.approx(-1.0)
    assert result["final_similarity"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        (np.zeros(3), np.ones(3), "non-zero norm"),
        (np.array([1.0, np.nan]), np.ones(2), "finite"),
        (np.ones(2), np.ones(3), "same dimension"),
    ],
)
def test_invalid_vectors_return_a_clear_error(first, second, message):
    result = FaceSimilarityCalculator().calculate_comprehensive_similarity(first, second)

    assert result["final_similarity"] == 0.0
    assert message in result["error"]
