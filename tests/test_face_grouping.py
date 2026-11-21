"""Regression tests for face identity grouping and similarity calibration.

The search used to treat ChromaDB's squared L2 distance as a Euclidean
distance, so a true cosine of 0.74 was reported as 0.86. That made the old
0.8 auto-assignment threshold behave like a much lower one and hid the real
similarity from the user. These tests pin the corrected scale and the new
reviewed grouping behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

from vector_store import FaceVectorStore

FACES_A = ("A0", "A1", "A2", "A3")
FACES_B = ("B0", "B1", "B2", "B3")


def _basis(size: int = 512, count: int = 12) -> np.ndarray:
    rng = np.random.default_rng(20261121)
    matrix, _ = np.linalg.qr(rng.normal(size=(size, count)))
    return matrix.T.astype(np.float32)


def _variation(base: np.ndarray, cosine: float, orthogonal: np.ndarray) -> np.ndarray:
    return (
        np.float32(cosine) * base
        + np.sqrt(1.0 - cosine * cosine) * orthogonal
    ).astype(np.float32)


def _make_store(tmp_path, named: dict[str, str] | None = None) -> FaceVectorStore:
    store = FaceVectorStore(
        db_path=str(tmp_path / "face-vectors"),
        collection_name="grouping-test",
    )
    basis = _basis()
    a0, b0 = basis[0], basis[1]
    embeddings = {
        "A0": a0,
        "A1": _variation(a0, 0.74, basis[2]),
        "A2": _variation(a0, 0.78, basis[3]),
        "A3": _variation(a0, 0.86, basis[4]),
        "B0": b0,
        "B1": _variation(b0, 0.72, basis[5]),
        "B2": _variation(b0, 0.81, basis[6]),
        "B3": _variation(b0, 0.85, basis[7]),
    }
    for face_id, embedding in embeddings.items():
        store.add_face_embedding(face_id, embedding, {
            "face_id": face_id,
            "image_path": f"/synthetic/{face_id}.jpg",
            "location": "10,210,210,10",
        })
    for face_id, name in (named or {}).items():
        store.assign_person_name(face_id, {
            "first_name": name,
            "middle_names": "",
            "last_name": "Test",
        })
    return store


def test_search_reports_true_cosine_instead_of_inflated_distance(tmp_path):
    store = _make_store(tmp_path)
    hits = store.search_similar_faces(
        store.get_face_by_id("A0")["embedding"],
        n_results=10,
        min_similarity=0.0,
    )
    by_id = {hit["face_id"]: hit for hit in hits}

    assert by_id["A1"]["similarity"] == pytest.approx(0.74, abs=0.01)
    assert by_id["A1"]["similarity"] < 0.80
    assert by_id["A1"]["cosine_similarity"] == pytest.approx(0.74, abs=0.01)
    assert by_id["A0"]["similarity"] == pytest.approx(1.0, abs=1e-6)


def test_grouping_recovers_two_identities_without_mixing(tmp_path):
    store = _make_store(tmp_path)

    result = store.get_face_groups(threshold=0.68, min_size=2)

    assert not result.get("error"), result
    assert result["unnamed_faces"] == 8
    assert result["grouped_faces"] == 8
    assert result["singletons"] == 0
    assert len(result["groups"]) == 2
    grouped = [set(group["face_ids"]) for group in result["groups"]]
    assert set(FACES_A) in grouped
    assert set(FACES_B) in grouped
    assert all(group["size"] == 4 for group in result["groups"])


def test_auto_assignment_groups_the_anchor_identity_only(tmp_path):
    store = _make_store(tmp_path)

    result = store.assign_person_name_with_auto_matching(
        "A0",
        {"first_name": "Anna", "middle_names": "", "last_name": "Test"},
    )

    assert result["success"], result
    assert result["total_assigned"] == 4, result
    named = store.get_faces_by_person_id(result["person_id"])
    assert {face["face_id"] for face in named} == set(FACES_A)
    assert result["skipped_ambiguous"] == 0


def test_ambiguous_face_is_not_auto_assigned(tmp_path):
    store = _make_store(tmp_path)
    basis = _basis()
    # A face halfway between two identities: 0.707 to each anchor. It must not
    # inherit whichever name happened to be processed first.
    mixed = (
        basis[0] + basis[1]
    ) / np.linalg.norm(basis[0] + basis[1])
    store.add_face_embedding("MIXED", mixed.astype(np.float32), {
        "face_id": "MIXED",
        "image_path": "/synthetic/MIXED.jpg",
        "location": "10,210,210,10",
    })
    store.assign_person_name("B0", {
        "first_name": "Berta",
        "middle_names": "",
        "last_name": "Test",
    })

    result = store.assign_person_name_with_auto_matching(
        "A0",
        {"first_name": "Anna", "middle_names": "", "last_name": "Test"},
    )

    assert result["success"], result
    assert result["skipped_ambiguous"] >= 1, result
    mixed_face = store.get_face_by_id("MIXED")
    assert not (mixed_face["metadata"].get("person_id") or "").strip()


def test_group_suggests_a_matching_named_person(tmp_path):
    store = _make_store(tmp_path)
    # The named face is excluded from the clusters but should be suggested.
    store.add_face_embedding(
        "N0",
        _basis()[0],
        {
            "face_id": "N0",
            "image_path": "/synthetic/N0.jpg",
            "location": "10,210,210,10",
        },
    )
    store.assign_person_name("N0", {
        "first_name": "Erika",
        "middle_names": "",
        "last_name": "Test",
    })

    result = store.get_face_groups(threshold=0.68, min_size=2)

    assert not result.get("error"), result
    a_group = next(group for group in result["groups"] if "A0" in group["face_ids"])
    suggestion = a_group["suggested_person"]
    assert suggestion is not None, a_group
    assert suggestion["full_name"] == "Erika Test"
    assert suggestion["similarity"] >= 0.68
