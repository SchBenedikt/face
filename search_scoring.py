"""Small helpers for explaining relative face-search rankings."""
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any


def top_score_gap(results: Sequence[Mapping[str, Any]]) -> float | None:
    """Return the gap between the two highest valid cosine-ranking scores."""
    scores = []
    for result in results:
        try:
            score = float(result["similarity"])
        except (KeyError, OverflowError, TypeError, ValueError):
            continue
        if isfinite(score) and 0.0 <= score <= 1.0:
            scores.append(score)

    if len(scores) < 2:
        return None

    scores.sort(reverse=True)
    return scores[0] - scores[1]