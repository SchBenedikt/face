"""Identity grouping for face embeddings.

The app stores one unit-normalised embedding per detected face. Grouping those
embeddings into identities is a clustering problem, not a simple threshold:
same-person and different-person score ranges depend on the active embedding
model. This module keeps clustering deterministic and exposes proposals for
review before names are assigned.

Only numpy is used; no additional clustering dependency is required. The
assignment step is vectorised over centroid rows, and the optional merge step
is capped so a large collection cannot freeze the Streamlit page.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from similarity import normalise_embedding
from config import FACE_GROUPING_MARGIN, FACE_GROUPING_THRESHOLD

DEFAULT_GROUPING_THRESHOLD = FACE_GROUPING_THRESHOLD
DEFAULT_GROUPING_MARGIN = FACE_GROUPING_MARGIN
# Two clusters are merged only when their centroids are clearly closer than the
# grouping threshold. This repairs order effects without joining look-alikes.
MERGE_MARGIN = 0.05
# Merging is quadratic. Above this many clusters it is not worth the wait and
# the reviewed groups are shown as they are.
MAX_MERGE_CLUSTERS = 300

ClusterMember = Tuple[str, np.ndarray]


def centroid_of(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Return the unit-length mean of the given embeddings."""
    matrix = np.asarray(list(embeddings), dtype=np.float32)
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    return normalise_embedding(matrix.mean(axis=0))


def _merge_close_clusters(
    members: List[List[ClusterMember]],
    centroids: np.ndarray,
    threshold: float,
) -> List[List[ClusterMember]]:
    """Merge clusters whose centroid distance is clearly below the threshold.

    Only called for a bounded number of clusters. Union-find handles chains
    deterministically; the caller keeps the group boundaries reviewable.
    """
    count = len(members)
    if count < 2 or count > MAX_MERGE_CLUSTERS:
        return members

    similarities = centroids @ centroids.T
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    merge_threshold = min(0.99, threshold + MERGE_MARGIN)
    for left in range(count):
        for right in range(left + 1, count):
            if similarities[left, right] >= merge_threshold:
                union(left, right)

    roots: Dict[int, List[int]] = {}
    for index in range(count):
        roots.setdefault(find(index), []).append(index)
    if len(roots) == count:
        return members

    return [
        [member for index in indices for member in members[index]]
        for indices in roots.values()
    ]


def group_embeddings(
    face_ids: Sequence[str],
    embeddings: Sequence[np.ndarray],
    threshold: float = DEFAULT_GROUPING_THRESHOLD,
    margin: float = DEFAULT_GROUPING_MARGIN,
) -> List[Dict[str, Any]]:
    """Group unit-length embeddings into identity clusters.

    A face joins an existing cluster only if its similarity to that centroid
    reaches ``threshold`` *and* beats the second-best cluster by ``margin``.
    The margin prevents a face that looks similar to two different people from
    silently joining one of them. Faces that fit nowhere become a new cluster.

    The deterministic order (face id) and the bounded centroid merge keep the
    result stable across reruns. Returned clusters include singletons; callers
    decide which minimum size they want to display.
    """
    if len(face_ids) != len(embeddings):
        raise ValueError("face_ids and embeddings must have the same length")

    items: List[ClusterMember] = []
    for face_id, embedding in zip(face_ids, embeddings):
        vector = normalise_embedding(embedding)
        if not np.any(vector):
            continue
        items.append((str(face_id), vector))
    items.sort(key=lambda item: item[0])
    if not items:
        return []

    dimension = items[0][1].shape[0]
    max_clusters = len(items)
    centroids = np.zeros((max_clusters, dimension), dtype=np.float32)
    sums = np.zeros((max_clusters, dimension), dtype=np.float32)
    members: List[List[ClusterMember]] = []
    cluster_count = 0

    for face_id, vector in items:
        best_index = -1
        best_similarity = -1.0
        second_similarity = -1.0
        if cluster_count:
            # One BLAS matrix-vector product per face, not one Python dot per
            # cluster. This keeps thousands of faces interactive.
            similarities = centroids[:cluster_count] @ vector
            if cluster_count == 1:
                best_index = 0
                best_similarity = float(similarities[0])
            else:
                top_two = np.argpartition(similarities, -2)[-2:]
                first, second = int(top_two[0]), int(top_two[1])
                if similarities[first] < similarities[second]:
                    first, second = second, first
                best_index = first
                best_similarity = float(similarities[first])
                second_similarity = float(similarities[second])

        unambiguous = (
            second_similarity < 0.0
            or (best_similarity - second_similarity) >= margin
        )
        if best_index >= 0 and best_similarity >= threshold and unambiguous:
            members[best_index].append((face_id, vector))
            sums[best_index] += vector
            centroids[best_index] = normalise_embedding(sums[best_index])
        else:
            members.append([(face_id, vector)])
            sums[cluster_count] = vector
            centroids[cluster_count] = vector
            cluster_count += 1

    members = _merge_close_clusters(
        members, centroids[:cluster_count].copy(), threshold
    )

    groups = []
    for cluster_members in members:
        centroid = centroid_of([vector for _, vector in cluster_members])
        similarities = [
            float(np.dot(vector, centroid)) for _, vector in cluster_members
        ]
        groups.append({
            "face_ids": [face_id for face_id, _ in cluster_members],
            "centroid": centroid,
            "mean_similarity": float(np.mean(similarities)) if similarities else 0.0,
            "min_similarity": float(np.min(similarities)) if similarities else 0.0,
        })

    groups.sort(
        key=lambda group: (
            -len(group["face_ids"]),
            -group["mean_similarity"],
            group["face_ids"][0],
        )
    )
    return groups
