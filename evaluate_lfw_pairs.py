#!/usr/bin/env python3
"""Evaluate the active face pipeline on a reproducible LFW pair sample.

This diagnostic uses LFW's labeled same-person/different-person pairs. It is
not the official cross-validated LFW benchmark and does not predict performance
on an unrelated collection. Install scikit-learn to use its dataset loader.
Downloaded images stay in the external data cache, never in this repository.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha1
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from face_recognition_engine import FaceRecognitionEngine
from settings_store import active_embedding_model, active_embedding_pipeline_version


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert sklearn's normalized float images to RGB pixels safely."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if image.size and np.nanmax(image) <= 1.0:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def evaluate_lfw_pairs(
    pairs_per_class: int = 200,
    seed: int = 42,
    data_home: Path | None = None,
    thresholds: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    model_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate pair scores with the configured detector and embedding model."""
    if pairs_per_class < 1:
        raise ValueError("pairs_per_class must be at least 1")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("thresholds must be finite values between 0 and 1")

    try:
        from sklearn.datasets import fetch_lfw_pairs
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to run this benchmark: pip install scikit-learn") from exc

    cache_path = data_home or Path.home() / ".cache" / "face-search" / "lfw"
    dataset = fetch_lfw_pairs(
        subset="test",
        data_home=str(cache_path),
        funneled=True,
        resize=1.0,
        color=True,
        # Keep the full source frame instead of sklearn's default center crop;
        # this exercises multi-person detection as users' photos do.
        slice_=(slice(0, 250), slice(0, 250)),
        download_if_missing=True,
    )

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    targets = np.asarray(dataset.target, dtype=np.int8)
    for label in (0, 1):
        indices = np.flatnonzero(targets == label)
        count = min(pairs_per_class, len(indices))
        selected.extend(rng.choice(indices, size=count, replace=False).tolist())
    rng.shuffle(selected)

    model_name = model_name or active_embedding_model()
    engine = FaceRecognitionEngine()
    engine.primary_model = model_name
    image_vectors: dict[str, np.ndarray | None] = {}
    image_counts: Counter[str] = Counter()
    started = time.monotonic()
    total_unique = len({
        sha1(_to_uint8(dataset.pairs[index, side]).tobytes()).hexdigest()
        for index in selected
        for side in (0, 1)
    })

    for pair_index in selected:
        for side in (0, 1):
            image = _to_uint8(dataset.pairs[pair_index, side])
            key = sha1(image.tobytes()).hexdigest()
            if key in image_vectors:
                continue

            locations = engine.detect_faces(image)
            if not locations:
                image_counts["no_face"] += 1
                image_vectors[key] = None
                continue

            if len(locations) > 1:
                image_counts["multiple_faces"] += 1
            # LFW pair labels refer to the primary centered person, but the
            # full source frame may contain other visible people. Select the
            # largest box to score that labeled person while retaining the
            # multi-face count as a separate detector observation.
            location = max(
                locations,
                key=lambda box: max(0, box[1] - box[3]) * max(0, box[2] - box[0]),
            )
            embedding = engine.extract_face_embedding(image, location)
            if embedding is None:
                image_counts["embedding_errors"] += 1
                image_vectors[key] = None
                continue

            vector = np.asarray(embedding, dtype=np.float32).ravel()
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 1e-10:
                image_counts["embedding_errors"] += 1
                image_vectors[key] = None
                continue

            image_vectors[key] = vector / norm
            image_counts["embedded"] += 1
            if len(image_vectors) % 50 == 0:
                print(f"Processed {len(image_vectors)}/{total_unique} unique images", flush=True)

    same_scores: list[float] = []
    different_scores: list[float] = []
    skipped_pairs = 0
    for pair_index in selected:
        left = _to_uint8(dataset.pairs[pair_index, 0])
        right = _to_uint8(dataset.pairs[pair_index, 1])
        left_vector = image_vectors[sha1(left.tobytes()).hexdigest()]
        right_vector = image_vectors[sha1(right.tobytes()).hexdigest()]
        if left_vector is None or right_vector is None:
            skipped_pairs += 1
            continue
        score = float(np.clip(np.dot(left_vector, right_vector), 0.0, 1.0))
        (same_scores if targets[pair_index] == 1 else different_scores).append(score)

    metrics = []
    for threshold in thresholds:
        metrics.append({
            "cosine_threshold": threshold,
            "same_person_recall": (
                float(np.mean(np.asarray(same_scores) >= threshold)) if same_scores else None
            ),
            "different_person_false_accept_rate": (
                float(np.mean(np.asarray(different_scores) >= threshold))
                if different_scores else None
            ),
        })

    return {
        "dataset": "LFW test pairs sample",
        "model": engine.primary_model,
        "configured_pipeline_version": active_embedding_pipeline_version(model_name),
        "seed": seed,
        "requested_pairs_per_class": pairs_per_class,
        "sampled_pairs": len(selected),
        "valid_pairs": len(same_scores) + len(different_scores),
        "skipped_pairs": skipped_pairs,
        "unique_images": len(image_vectors),
        "image_counts": dict(image_counts),
        "same_person_pairs": len(same_scores),
        "different_person_pairs": len(different_scores),
        "same_person_median_cosine": float(np.median(same_scores)) if same_scores else None,
        "different_person_median_cosine": float(np.median(different_scores)) if different_scores else None,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "threshold_metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-class", type=int, default=200,
                        help="Number of same-person and different-person pairs (default: 200 each)")
    parser.add_argument("--seed", type=int, default=42, help="Random sample seed (default: 42)")
    parser.add_argument("--data-home", type=Path, help="External cache directory for LFW images")
    parser.add_argument(
        "--model", default=None,
        help="DeepFace embedding model to evaluate (default: active Settings model)",
    )
    parser.add_argument("--thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70",
                        help="Comma-separated cosine thresholds")
    args = parser.parse_args()
    try:
        thresholds = tuple(float(value) for value in args.thresholds.split(","))
        result = evaluate_lfw_pairs(
            args.pairs_per_class, args.seed, args.data_home, thresholds, args.model
        )
    except (RuntimeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
