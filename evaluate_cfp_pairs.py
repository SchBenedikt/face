#!/usr/bin/env python3
"""Compare face embeddings on CFP frontal-profile verification pairs.

The evaluator follows CFP's ten official FP splits. A fixed, seeded sample can
be selected from each split for a quicker diagnostic; passing 0 evaluates all
official pairs. One detection/alignment pass is shared across the requested
models, while each embedding is produced by the application's engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from config import FACE_EMBEDDING_MODEL
from face_recognition_engine import FaceRecognitionEngine
from utils import load_and_preprocess_image


def _cache_signature(dataset_root: Path, models: tuple[str, ...]) -> str:
    from settings_store import active_embedding_pipeline_version

    values = {
        "dataset_root": str(dataset_root.resolve()),
        "models": list(models),
        "pipeline": active_embedding_pipeline_version(),
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _file_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_cache(cache_path: Path | None, signature: str):
    if cache_path is None or not cache_path.is_file():
        return {"signature": signature, "files": {}}, {}
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["__metadata__"].item()))
            if metadata.get("signature") != signature:
                print("Embedding cache does not match this dataset/model pipeline; rebuilding it.")
                return {"signature": signature, "files": {}}, {}
            arrays = {
                key: np.asarray(archive[key], dtype=np.float32)
                for key in archive.files if key != "__metadata__"
            }
        return metadata, arrays
    except Exception as exc:  # noqa: BLE001 - a bad optional cache must not block evaluation
        print(f"Could not read embedding cache ({exc}); rebuilding it.")
        return {"signature": signature, "files": {}}, {}


def _write_cache(cache_path: Path | None, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=cache_path.parent, prefix=f".{cache_path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                __metadata__=np.asarray(json.dumps(metadata, sort_keys=True)),
                **arrays,
            )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


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


def _metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
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
        "threshold_sweep": {
            f"{candidate:.2f}": {
                "accuracy": float(np.mean((scores >= candidate) == labels)),
                "true_accept_rate": float(np.mean(scores[same] >= candidate)) if np.any(same) else None,
                "false_accept_rate": float(np.mean(scores[different] >= candidate)) if np.any(different) else None,
            }
            for candidate in thresholds
        },
    }


def evaluate(
    dataset_root: Path,
    models: tuple[str, ...],
    pairs_per_class: int,
    seed: int,
    threshold: float,
    thresholds: tuple[float, ...],
    cache_path: Path | None = None,
) -> dict[str, Any]:
    pairs = _selected_pairs(dataset_root, pairs_per_class, seed)
    image_paths = list(dict.fromkeys(path for frontal, profile, _, _ in pairs for path in (frontal, profile)))
    engine = FaceRecognitionEngine()
    signature = _cache_signature(dataset_root, models)
    cache_metadata, cache_arrays = _read_cache(cache_path, signature)
    cached_files = cache_metadata.setdefault("files", {})
    vectors: dict[Path, dict[str, np.ndarray]] = {}
    cached_paths = set()
    for path in image_paths:
        key = str(path)
        cached = cached_files.get(key)
        if not cached:
            continue
        try:
            if cached.get("fingerprint") != _file_fingerprint(path):
                continue
        except OSError:
            continue
        cached_paths.add(path)
        image_vectors = {}
        for model, array_key in cached.get("vectors", {}).items():
            if model in models and array_key in cache_arrays:
                image_vectors[model] = cache_arrays[array_key]
        if image_vectors:
            vectors[path] = image_vectors

    print(f"Loaded embeddings for {len(vectors)} cached images; {len(image_paths) - len(cached_paths)} images need processing.", flush=True)
    counts: dict[str, Any] = {
        "images_considered": len(image_paths),
        "unreadable": 0,
        "no_face": 0,
        "multiple_faces": 0,
        "embedding_errors_by_model": {model: 0 for model in models},
        "images_loaded_from_cache": len(vectors),
    }
    started = time.monotonic()

    for index, path in enumerate(image_paths, start=1):
        key = str(path)
        if path in cached_paths:
            cached = cached_files[key]
            status = cached.get("status")
            counts["unreadable"] += int(status == "unreadable")
            counts["no_face"] += int(status == "no_face")
            counts["multiple_faces"] += int(bool(cached.get("multiple_faces")))
            for model in cached.get("embedding_errors", []):
                if model in counts["embedding_errors_by_model"]:
                    counts["embedding_errors_by_model"][model] += 1
            continue

        try:
            fingerprint = _file_fingerprint(path)
        except OSError:
            fingerprint = None
        file_data: dict[str, Any] = {
            "fingerprint": fingerprint,
            "status": "processed",
            "vectors": {},
            "embedding_errors": [],
            "multiple_faces": False,
        }
        image = load_and_preprocess_image(path)
        if fingerprint is None or image is None:
            counts["unreadable"] += 1
            file_data["status"] = "unreadable"
            cached_files[key] = file_data
            continue
        locations = engine.detect_faces(image)
        if not locations:
            counts["no_face"] += 1
            file_data["status"] = "no_face"
            cached_files[key] = file_data
            continue
        if len(locations) > 1:
            counts["multiple_faces"] += 1
            file_data["multiple_faces"] = True
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
                file_data["embedding_errors"].append(model)
            else:
                image_vectors[model] = vector
                array_key = f"v_{len(cache_arrays):08d}"
                cache_arrays[array_key] = vector
                file_data["vectors"][model] = array_key
        if image_vectors:
            vectors[path] = image_vectors
        cached_files[key] = file_data
        if index % 100 == 0:
            elapsed = round(time.monotonic() - started, 1)
            print(f"Embedded {index}/{len(image_paths)} images ({elapsed}s)", flush=True)
            _write_cache(cache_path, cache_metadata, cache_arrays)

    _write_cache(cache_path, cache_metadata, cache_arrays)

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
        result: dict[str, Any] = _metrics(labels, scores, threshold, thresholds)
        result["folds"] = {}
        for fold in range(1, 11):
            mask = fold_ids == fold
            fold_labels = labels[mask]
            fold_scores = scores[mask]
            if len(fold_scores) and np.any(fold_labels) and np.any(~fold_labels):
                result["folds"][str(fold)] = _metrics(
                    fold_labels, fold_scores, threshold, thresholds
                )
        model_results[model] = result

    counts["images_embedded_by_any_model"] = len(vectors)
    counts["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return {
        "dataset": "CFP frontal-profile (FP) verification protocol",
        "pairs_per_class_per_fold": pairs_per_class or "all",
        "seed": seed,
        "threshold": threshold,
        "thresholds_evaluated": list(thresholds),
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
    parser.add_argument(
        "--thresholds", default="0.30,0.35,0.40,0.45,0.50",
        help="Comma-separated operating points to report for every model",
    )
    parser.add_argument(
        "--embedding-cache", type=Path,
        help="Optional local NPZ cache for derived face embeddings; allows fast metric reruns",
    )
    parser.add_argument("--output", type=Path, help="Write the full JSON report to this file")
    args = parser.parse_args()
    if not args.dataset.is_dir():
        parser.error(f"Dataset directory not found: {args.dataset}")
    if args.pairs_per_class < 0:
        parser.error("--pairs-per-class must be zero or greater")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if len(set(args.models)) != len(args.models):
        parser.error("Model names must be unique")
    try:
        thresholds = tuple(float(value) for value in args.thresholds.split(","))
    except ValueError:
        parser.error("--thresholds must be comma-separated numbers")
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        parser.error("--thresholds must contain values between 0 and 1")

    report = evaluate(
        args.dataset, tuple(args.models), args.pairs_per_class, args.seed,
        args.threshold, thresholds, args.embedding_cache,
    )
    report_text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text + "\n", encoding="utf-8")
        summary = {
            "report_file": str(args.output),
            "models": {
                model: {
                    key: result.get(key)
                    for key in ("pairs_scored", "auc", "eer", "accuracy_at_threshold",
                                "true_accept_rate_at_threshold", "false_accept_rate_at_threshold")
                }
                for model, result in report["models"].items()
            },
            "processing": report["processing"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(report_text)


if __name__ == "__main__":
    main()
