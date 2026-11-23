#!/usr/bin/env python3
"""Compare face embeddings on CFP frontal-profile verification pairs.

The evaluator follows CFP's ten official FP splits. A fixed, seeded sample can
be selected from each split for a quicker diagnostic; passing 0 evaluates all
official pairs. One detection/alignment pass is shared across the requested
models, while each embedding is produced by the application's engine.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from config import FACE_EMBEDDING_MODEL
from face_recognition_engine import FaceRecognitionEngine
from utils import load_and_preprocess_image


def _image_map(dataset_root: Path, list_name: str) -> dict[int, Path]:
    protocol_root = dataset_root / "Protocol"
    result: dict[int, Path] = {}
    for line in (protocol_root / list_name).read_text(encoding="utf-8").splitlines():
        index, relative_path = line.split(maxsplit=1)
        result[int(index)] = (protocol_root / relative_path).resolve()
    return result


def _selected_pairs(
    dataset_root: Path, pairs_per_class: int, seed: int
) -> list[tuple[Path, Path, bool, int]]:
    frontal = _image_map(dataset_root, "Pair_list_F.txt")
    profile = _image_map(dataset_root, "Pair_list_P.txt")
    pairs: list[tuple[Path, Path, bool, int]] = []

    for fold in range(1, 11):
        for is_same, file_name in ((True, "same.txt"), (False, "diff.txt")):
            pair_file = dataset_root / "Protocol" / "Split" / "FP" / f"{fold:02d}" / file_name
            lines = pair_file.read_text(encoding="utf-8").splitlines()
            if pairs_per_class > 0 and len(lines) > pairs_per_class:
                rng = random.Random(seed + fold * 1009 + int(is_same))
                lines = rng.sample(lines, pairs_per_class)
            for line in lines:
                frontal_id, profile_id = map(int, line.split(","))
                pairs.append((frontal[frontal_id], profile[profile_id], is_same, fold))
    return pairs


def _unit_vector(embedding: Any) -> np.ndarray | None:
    vector = np.asarray(embedding, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-10:
        return None
    return vector / norm


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, roc_thresholds = roc_curve(labels, scores, pos_label=True)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    accepted = scores >= threshold
    same = labels
    different = ~labels
    return {
        "pairs_scored": int(len(scores)),
        "same_pairs_scored": int(np.count_nonzero(same)),
        "different_pairs_scored": int(np.count_nonzero(different)),
        "auc": float(roc_auc_score(labels, scores)),
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "eer_threshold": float(roc_thresholds[eer_index]),
        "threshold": float(threshold),
        "accuracy_at_threshold": float(np.mean(accepted == labels)),
        "true_accept_rate_at_threshold": float(np.mean(accepted[same])) if np.any(same) else None,
        "false_accept_rate_at_threshold": float(np.mean(accepted[different])) if np.any(different) else None,
    }


def evaluate(
    dataset_root: Path,
    models: tuple[str, ...],
    pairs_per_class: int,
    seed: int,
    threshold: float,
) -> dict[str, Any]:
    pairs = _selected_pairs(dataset_root, pairs_per_class, seed)
    image_paths = list(dict.fromkeys(path for frontal, profile, _, _ in pairs for path in (frontal, profile)))
    engine = FaceRecognitionEngine()
    counts: dict[str, Any] = {
        "images_considered": len(image_paths),
        "unreadable": 0,
        "no_face": 0,
        "multiple_faces": 0,
        "embedding_errors_by_model": {model: 0 for model in models},
    }
    vectors: dict[Path, dict[str, np.ndarray]] = {}
    started = time.monotonic()

    for index, path in enumerate(image_paths, start=1):
        image = load_and_preprocess_image(path)
        if image is None:
            counts["unreadable"] += 1
            continue
        locations = engine.detect_faces(image)
        if not locations:
            counts["no_face"] += 1
            continue
        if len(locations) > 1:
            counts["multiple_faces"] += 1
        location = max(
            locations,
            key=lambda box: max(0, box[1] - box[3]) * max(0, box[2] - box[0]),
        )
        image_vectors: dict[str, np.ndarray] = {}
        for model in models:
            engine.primary_model = model
            embedding = engine.extract_face_embedding(image, location)
            vector = _unit_vector(embedding) if embedding is not None else None
            if vector is None:
                counts["embedding_errors_by_model"][model] += 1
            else:
                image_vectors[model] = vector
        if image_vectors:
            vectors[path] = image_vectors
        if index % 100 == 0:
            elapsed = round(time.monotonic() - started, 1)
            print(f"Embedded {index}/{len(image_paths)} images ({elapsed}s)", flush=True)

    model_results: dict[str, Any] = {}
    for model in models:
        scored: list[tuple[bool, float, int]] = []
        for frontal, profile, same, fold in pairs:
            left = vectors.get(frontal, {}).get(model)
            right = vectors.get(profile, {}).get(model)
            if left is None or right is None or left.shape != right.shape:
                continue
            score = float(np.clip(np.dot(left, right), 0.0, 1.0))
            scored.append((same, score, fold))

        if not scored:
            model_results[model] = {"pairs_scored": 0, "message": "No protocol pairs could be scored."}
            continue
        labels = np.asarray([same for same, _, _ in scored], dtype=bool)
        scores = np.asarray([score for _, score, _ in scored], dtype=np.float64)
        fold_ids = np.asarray([fold for _, _, fold in scored], dtype=np.int8)
        result: dict[str, Any] = _metrics(labels, scores, threshold)
        result["folds"] = {}
        for fold in range(1, 11):
            mask = fold_ids == fold
            fold_labels = labels[mask]
            fold_scores = scores[mask]
            if len(fold_scores) and np.any(fold_labels) and np.any(~fold_labels):
                result["folds"][str(fold)] = _metrics(fold_labels, fold_scores, threshold)
        model_results[model] = result

    counts["images_embedded_by_any_model"] = len(vectors)
    counts["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return {
        "dataset": "CFP frontal-profile (FP) verification protocol",
        "pairs_per_class_per_fold": pairs_per_class or "all",
        "seed": seed,
        "threshold": threshold,
        "models": model_results,
        "processing": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Extracted cfp-dataset directory")
    parser.add_argument(
        "--models", nargs="+", default=[FACE_EMBEDDING_MODEL, "ArcFace"],
        help="DeepFace embedding models to compare (default: active model and ArcFace)",
    )
    parser.add_argument(
        "--pairs-per-class", type=int, default=50,
        help="Same and different pairs sampled from each fold; 0 uses all 350 per class (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--threshold", type=float, default=0.40)
    args = parser.parse_args()
    if not args.dataset.is_dir():
        parser.error(f"Dataset directory not found: {args.dataset}")
    if args.pairs_per_class < 0:
        parser.error("--pairs-per-class must be zero or greater")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if len(set(args.models)) != len(args.models):
        parser.error("Model names must be unique")

    print(json.dumps(evaluate(
        args.dataset, tuple(args.models), args.pairs_per_class, args.seed, args.threshold
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
