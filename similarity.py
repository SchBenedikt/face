"""
Similarity helpers used by the vector store.

Deliberately free of heavy imports (numpy only) so the scoring can be unit
tested without ChromaDB, TensorFlow or dlib installed.
"""
from __future__ import annotations

from typing import Union

import numpy as np


def cosine_from_l2_distance(distance: float) -> float:
    """
    Convert an L2 distance between two unit vectors into cosine similarity.

    For unit vectors ``||a - b||^2 = 2 - 2 * cos``, therefore
    ``cos = 1 - d^2 / 2``. Embeddings are stored normalised, so this is exact
    and avoids transferring the stored vectors back into Python.
    """
    d = float(distance)
    if d < 0.0:
        d = 0.0
    return 1.0 - (d * d) / 2.0


def similarity_from_l2_distance(distance: float) -> float:
    """Like :func:`cosine_from_l2_distance`, clipped to 0..1 for ranking."""
    return float(np.clip(cosine_from_l2_distance(distance), 0.0, 1.0))


def normalise_embedding(embedding: Union[np.ndarray, list]) -> np.ndarray:
    """Return a unit-length float32 copy of an embedding."""
    vector = np.asarray(embedding, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-10:
        return vector
    return vector / norm


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two embeddings, clipped to -1..1."""
    left = normalise_embedding(a)
    right = normalise_embedding(b)
    if not np.any(left) or not np.any(right):
        return 0.0
    return float(np.clip(np.dot(left, right), -1.0, 1.0))
