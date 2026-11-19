import sys
import types

import numpy as np
import pytest

# vector_store imports chromadb at module level. Use the real package when it
# is installed (the UI tests seed a real database) and fall back to a stub only
# when it is genuinely missing.
try:
    import chromadb  # noqa: F401
    import chromadb.config  # noqa: F401
except Exception:  # noqa: BLE001
    chromadb_stub = types.ModuleType("chromadb")
    chromadb_stub.PersistentClient = lambda *args, **kwargs: None
    sys.modules.setdefault("chromadb", chromadb_stub)

    config_stub = types.ModuleType("chromadb.config")
    config_stub.Settings = lambda *args, **kwargs: None
    sys.modules.setdefault("chromadb.config", config_stub)

from vector_store import FaceVectorStore  # noqa: E402


def _store_without_database() -> FaceVectorStore:
    return object.__new__(FaceVectorStore)


def test_identical_embeddings_score_one():
    store = _store_without_database()
    rng = np.random.default_rng(7)
    vector = rng.normal(size=512).astype(np.float32)
    metrics = store._calculate_enhanced_similarity(vector, vector)
    assert metrics["primary_similarity"] == pytest.approx(1.0, abs=1e-5)


def test_unrelated_embeddings_are_not_scored_as_matches():
    store = _store_without_database()
    rng = np.random.default_rng(11)
    left = rng.normal(size=512).astype(np.float32)
    right = rng.normal(size=512).astype(np.float32)
    metrics = store._calculate_enhanced_similarity(left, right)
    # The old piecewise rescaling pushed random pairs to ~0.55.
    assert metrics["primary_similarity"] < 0.3


def test_confidence_matches_similarity():
    store = _store_without_database()
    rng = np.random.default_rng(13)
    left = rng.normal(size=512).astype(np.float32)
    right = left + rng.normal(size=512).astype(np.float32) * 0.1
    metrics = store._calculate_enhanced_similarity(left, right)
    confidence = store._calculate_confidence_score(metrics)
    assert confidence == pytest.approx(metrics["primary_similarity"], abs=1e-6)


def test_second_search_round_is_gone():
    assert not hasattr(FaceVectorStore, "_search_around_high_confidence_matches")
