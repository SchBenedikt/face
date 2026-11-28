import pytest

from search_scoring import top_score_gap


def test_top_score_gap_uses_the_two_highest_scores():
    results = [
        {"similarity": 0.62},
        {"similarity": 0.91},
        {"similarity": 0.74},
    ]

    assert top_score_gap(results) == pytest.approx(0.17)


def test_top_score_gap_reports_tied_leaders_as_zero():
    assert top_score_gap([{"similarity": 0.8}, {"similarity": 0.8}]) == 0.0


def test_top_score_gap_needs_two_valid_scores():
    assert top_score_gap([{"similarity": 0.8}]) is None
    assert top_score_gap([{"similarity": 0.8}, {"similarity": float("nan")}]) is None


def test_top_score_gap_ignores_invalid_scores():
    results = [
        {"similarity": 0.9},
        {"similarity": "invalid"},
        {"similarity": 10**10000},
        {"similarity": 0.7},
        {},
    ]

    assert top_score_gap(results) == pytest.approx(0.2)