#!/usr/bin/env python3
"""Evaluate face retrieval on a labeled directory tree.

Expected layout: one subdirectory per identity, containing images that each
show exactly one person. The benchmark reports measurements for this dataset;
it does not claim those measurements generalize to other collections.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from face_recognition_engine import FaceRecognitionEngine
from utils import load_and_preprocess_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _labeled_images(root: Path) -> dict[str, list[Path]]:
    """Return image paths grouped by their immediate identity directory."""
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    identities = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        images = sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            identities[directory.name] = images
    if not identities:
        raise ValueError("Expected image files in subdirectories named by identity")
    return identities


def _select_images(
    identities: dict[str, list[Path]], limit: int, seed: int
) -> dict[str, list[Path]]:
    """Select a reproducible, identity-balanced subset when a limit is set."""
    if limit <= 0 or sum(map(len, identities.values())) <= limit:
        return identities
    rng = np.random.default_rng(seed)
    shuffled = {
        identity: list(rng.permutation(paths))
        for identity, paths in identities.items()
    }
    identity_order = list(shuffled)
    rng.shuffle(identity_order)
    selected: dict[str, list[Path]] = defaultdict(list)
    while sum(map(len, selected.values())) < limit:
        added = False
        for identity in identity_order:
            paths = shuffled[identity]
            if paths and sum(map(len, selected.values())) < limit:
                selected[identity].append(paths.pop())
                added = True
        if not added:
            break
    return dict(selected)


def evaluate(
    root: Path,
    max_images: int = 2000,
    seed: int = 42,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7, 0.8),
) -> dict[str, Any]:
    identities = _select_images(_labeled_images(root), max_images, seed)
    engine = FaceRecognitionEngine()
    labels: list[str] = []
    vectors: list[np.ndarray] = []
    counts = {"images_considered": sum(map(len, identities.values())), "unreadable": 0,
              "no_face": 0, "multiple_faces": 0, "embedding_errors": 0}

    started = time.monotonic()
    for identity, paths in identities.items():
        for path in paths:
            image = load_and_preprocess_image(path)
            if image is None:
                counts["unreadable"] += 1
                continue
            locations = engine.detect_faces(image)
            if not locations:
                counts["no_face"] += 1
                continue
            if len(locations) != 1:
                counts["multiple_faces"] += 1
                continue
            embedding = engine.extract_face_embedding(image, locations[0])
            if embedding is None:
                counts["embedding_errors"] += 1
                continue
            vector = np.asarray(embedding, dtype=np.float32).ravel()
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 1e-10:
                counts["embedding_errors"] += 1
                continue
            labels.append(identity)
            vectors.append(vector / norm)

    counts["elapsed_seconds"] = round(time.monotonic() - started, 2)
    counts["images_embedded"] = len(vectors)
    counts["identities_embedded"] = len(set(labels))
    if len(vectors) < 2:
        return {**counts, "queries": 0, "top1": None, "top5": None,
                "pair_metrics": [], "message": "Need at least two usable images."}

    matrix = np.stack(vectors)
    label_array = np.asarray(labels)
    scores = np.clip(matrix @ matrix.T, -1.0, 1.0)
    upper_rows, upper_cols = np.triu_indices(len(labels), k=1)
    same = label_array[upper_rows] == label_array[upper_cols]
    # Search scores clip negative cosine values to zero, matching vector_store.
    pair_scores = np.clip(scores[upper_rows, upper_cols], 0.0, 1.0)
    same_scores = pair_scores[same]
    different_scores = pair_scores[~same]

    queries = 0
    top1_hits = 0
    top5_hits = 0
    for query_index, identity in enumerate(labels):
        candidates = scores[query_index].copy()
        candidates[query_index] = -np.inf
        candidate_indices = np.argpartition(candidates, -min(5, len(labels) - 1))[-min(5, len(labels) - 1):]
        candidate_indices = candidate_indices[np.argsort(candidates[candidate_indices])[::-1]]
        if not np.any(label_array[np.arange(len(labels)) != query_index] == identity):
            continue
        queries += 1
        ranked_labels = label_array[candidate_indices]
        top1_hits += int(ranked_labels[0] == identity)
        top5_hits += int(np.any(ranked_labels == identity))

    pair_metrics = []
    for threshold in thresholds:
        pair_metrics.append({
            "cosine_threshold": threshold,
            "same_identity_recall": (
                float(np.mean(same_scores >= threshold)) if len(same_scores) else None
            ),
            "different_identity_false_accept_rate": (
                float(np.mean(different_scores >= threshold))
                if len(different_scores) else None
            ),
        })

    return {
        **counts,
        "queries": queries,
        "top1": top1_hits / queries if queries else None,
        "top5": top5_hits / queries if queries else None,
        "same_identity_pairs": int(len(same_scores)),
        "different_identity_pairs": int(len(different_scores)),
        "same_identity_median_cosine": float(np.median(same_scores)) if len(same_scores) else None,
        "different_identity_median_cosine": float(np.median(different_scores)) if len(different_scores) else None,
        "pair_metrics": pair_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Root directory with one identity subdirectory per person")
    parser.add_argument("--max-images", type=int, default=2000,
                        help="Balanced image cap for the pairwise comparison (0 means no cap; default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Subset selection seed")
    parser.add_argument("--thresholds", default="0.3,0.5,0.7,0.8",
                        help="Comma-separated cosine thresholds (default: 0.3,0.5,0.7,0.8)")
    args = parser.parse_args()
    try:
        thresholds = tuple(float(value) for value in args.thresholds.split(","))
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("thresholds must be between 0 and 1")
        result = evaluate(args.dataset, args.max_images, args.seed, thresholds)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
