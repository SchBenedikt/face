#!/usr/bin/env python3
"""Evaluate face retrieval on a labeled directory tree.

Expected layout: one subdirectory per identity, containing images whose main
face belongs to that identity. When an image has multiple detections, the
largest face is evaluated. Results describe this dataset, not other collections.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from face_recognition_engine import FaceRecognitionEngine
from config import FACE_EMBEDDING_MODEL
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
    max_error_examples: int = 50,
    model_name: str = FACE_EMBEDDING_MODEL,
) -> dict[str, Any]:
    identities = _select_images(_labeled_images(root), max_images, seed)
    engine = FaceRecognitionEngine()
    engine.primary_model = model_name
    labels: list[str] = []
    vector_paths: list[str] = []
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
            if len(locations) > 1:
                counts["multiple_faces"] += 1
            # Each directory is labeled as one identity, so measure retrieval
            # for its most prominent detected face. Keep multi-face coverage
            # visible in the report instead of silently dropping those images.
            location = max(
                locations,
                key=lambda box: max(0, box[1] - box[3]) * max(0, box[2] - box[0]),
            )
            embedding = engine.extract_face_embedding(image, location)
            if embedding is None:
                counts["embedding_errors"] += 1
                continue
            vector = np.asarray(embedding, dtype=np.float32).ravel()
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 1e-10:
                counts["embedding_errors"] += 1
                continue
            labels.append(identity)
            vector_paths.append(str(path))
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
    top1_error_count = 0
    top1_errors: list[dict[str, Any]] = []
    ann_top1_hits = 0
    ann_top5_hits = 0
    ann_queries = 0
    ann_top1_error_count = 0
    ann_top1_errors: list[dict[str, Any]] = []
    ann_index_seconds = 0.0
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
        if ranked_labels[0] != identity:
            top1_error_count += 1
            if len(top1_errors) < max_error_examples:
                same_identity_indices = np.flatnonzero(label_array == identity)
                same_identity_indices = same_identity_indices[same_identity_indices != query_index]
                top1_errors.append({
                    "query_path": vector_paths[query_index],
                    "true_identity": identity,
                    "top1_path": vector_paths[int(candidate_indices[0])],
                    "top1_identity": str(ranked_labels[0]),
                    "top1_cosine": float(scores[query_index, candidate_indices[0]]),
                    "best_same_identity_cosine": float(
                        np.max(scores[query_index, same_identity_indices])
                    ),
                })

    # Check the application's actual ChromaDB/HNSW retrieval path as well as
    # exhaustive cosine ranking. Exclude each query's own indexed vector and
    # compare the next five returned neighbors with its identity label.
    from vector_store import FaceVectorStore

    ann_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="face-search-index-") as temporary_index:
        store = FaceVectorStore(db_path=temporary_index)
        face_ids = [f"benchmark_face_{index}" for index in range(len(vectors))]
        metadatas = [
            {"image_path": image_path, "identity": identity}
            for image_path, identity in zip(vector_paths, labels)
        ]
        written = store.add_face_embeddings_batch(face_ids, vectors, metadatas)
        if written != len(vectors):
            raise RuntimeError(f"Could not index benchmark vectors: wrote {written}/{len(vectors)}")

        for query_index, identity in enumerate(labels):
            hits = store.search_similar_faces(
                vectors[query_index], n_results=6, min_similarity=0.0
            )
            neighbors = [hit for hit in hits if hit["face_id"] != face_ids[query_index]][:5]
            if not neighbors:
                continue
            ann_queries += 1
            neighbor_labels = [hit["metadata"].get("identity") for hit in neighbors]
            ann_top1_hits += int(neighbor_labels[0] == identity)
            ann_top5_hits += int(identity in neighbor_labels)
            if neighbor_labels[0] != identity:
                ann_top1_error_count += 1
                if len(ann_top1_errors) < max_error_examples:
                    ann_top1_errors.append({
                        "query_path": vector_paths[query_index],
                        "true_identity": identity,
                        "top1_path": neighbors[0]["metadata"].get("image_path"),
                        "top1_identity": neighbor_labels[0],
                        "top1_cosine": float(neighbors[0]["cosine_similarity"]),
                    })
        del store
    ann_index_seconds = round(time.monotonic() - ann_started, 2)

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
        "model": engine.primary_model,
        "queries": queries,
        "top1": top1_hits / queries if queries else None,
        "top5": top5_hits / queries if queries else None,
        "top1_error_count": top1_error_count,
        "top1_errors": top1_errors,
        "ann_queries": ann_queries,
        "chroma_top1": ann_top1_hits / ann_queries if ann_queries else None,
        "chroma_top5": ann_top5_hits / ann_queries if ann_queries else None,
        "chroma_top1_error_count": ann_top1_error_count,
        "chroma_top1_errors": ann_top1_errors,
        "chroma_index_and_query_seconds": ann_index_seconds,
        "same_identity_pairs": int(len(same_scores)),
        "different_identity_pairs": int(len(different_scores)),
        "same_identity_median_cosine": float(np.median(same_scores)) if len(same_scores) else None,
        "different_identity_median_cosine": float(np.median(different_scores)) if len(different_scores) else None,
        "pair_metrics": pair_metrics,
    }


def evaluate_lfw_people(
    max_images: int = 300,
    seed: int = 42,
    data_home: Path | None = None,
    min_faces_per_person: int = 20,
    thresholds: tuple[float, ...] = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7),
    model_name: str = FACE_EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Run closed-set retrieval on a reproducible LFW people subset."""
    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to run this benchmark: pip install scikit-learn") from exc

    from PIL import Image

    cache_path = data_home or Path.home() / ".cache" / "face-search" / "lfw"
    data = fetch_lfw_people(
        data_home=str(cache_path),
        funneled=True,
        resize=1.0,
        color=True,
        min_faces_per_person=min_faces_per_person,
    )
    by_identity = {
        name: np.flatnonzero(data.target == index).tolist()
        for index, name in enumerate(data.target_names)
    }
    selected = _select_images(by_identity, max_images, seed)

    # Reuse the directory evaluator as the single implementation of face
    # detection, ranking, and pair metrics. Generated images are temporary.
    source_root = cache_path / "lfw_home" / "lfw_funneled"
    source_files: list[Path] = []
    for person_directory in sorted(path for path in source_root.iterdir() if path.is_dir()):
        person_files = sorted(path for path in person_directory.iterdir() if path.is_file())
        if len(person_files) >= min_faces_per_person:
            source_files.extend(person_files)
    if len(source_files) != len(data.images):
        raise RuntimeError("Could not map sampled images to the LFW source cache")
    # sklearn deterministically shuffles LFW images with RandomState(42).
    source_order = np.arange(len(source_files))
    np.random.RandomState(42).shuffle(source_order)
    source_path_for_index = {
        index: str(source_files[int(source_position)].resolve())
        for index, source_position in enumerate(source_order)
    }
    source_path_for_image: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="face-lfw-retrieval-") as temporary:
        root = Path(temporary)
        sampled_count = 0
        for identity, indices in selected.items():
            identity_dir = root / identity
            identity_dir.mkdir(parents=True, exist_ok=True)
            for image_index, dataset_index in enumerate(indices):
                image = np.asarray(data.images[dataset_index])
                if np.issubdtype(image.dtype, np.floating) and image.size and np.nanmax(image) <= 1.0:
                    image = image * 255.0
                image = np.clip(image, 0, 255).astype(np.uint8)
                temporary_image = identity_dir / f"{image_index:04d}.jpg"
                Image.fromarray(image).save(temporary_image, quality=95)
                source_path_for_image[str(temporary_image.resolve())] = (
                    source_path_for_index[int(dataset_index)]
                )
                sampled_count += 1

        result = evaluate(
            root, max_images=0, seed=seed, thresholds=thresholds,
            model_name=model_name,
        )

    for field in ("top1_errors", "chroma_top1_errors"):
        for error in result.get(field, []):
            for path_field in ("query_path", "top1_path"):
                source_path = source_path_for_image.get(error.get(path_field))
                if source_path is not None:
                    error[path_field] = source_path

    result.update({
        "dataset": "LFW people retrieval sample",
        "evaluation_crop": "LFW default center crop (y=70:195, x=78:172)",
        "seed": seed,
        "min_faces_per_person": min_faces_per_person,
        "sampled_images": sampled_count,
        "sampled_identities": len(selected),
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?",
                        help="Root directory with one identity subdirectory per person")
    parser.add_argument("--lfw-people", action="store_true",
                        help="Download and evaluate a reproducible LFW people sample")
    parser.add_argument("--data-home", type=Path,
                        help="External LFW download cache (default: ~/.cache/face-search/lfw)")
    parser.add_argument("--min-faces-per-person", type=int, default=20,
                        help="LFW identity inclusion minimum when using --lfw-people (default: 20)")
    parser.add_argument("--max-images", type=int, default=2000,
                        help="Balanced image cap for retrieval and pair metrics (0 means no cap; default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Subset selection seed")
    parser.add_argument(
        "--model", default=FACE_EMBEDDING_MODEL,
        help=f"DeepFace embedding model to evaluate (default: {FACE_EMBEDDING_MODEL})",
    )
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7",
                        help="Comma-separated cosine thresholds (default: 0.3,0.4,0.5,0.6,0.7)")
    args = parser.parse_args()
    try:
        thresholds = tuple(float(value) for value in args.thresholds.split(","))
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("thresholds must be between 0 and 1")
        if args.lfw_people:
            if args.dataset is not None:
                raise ValueError("Do not pass a dataset directory together with --lfw-people")
            if args.min_faces_per_person < 1:
                raise ValueError("--min-faces-per-person must be at least 1")
            result = evaluate_lfw_people(
                args.max_images, args.seed, args.data_home,
                args.min_faces_per_person, thresholds, args.model,
            )
        else:
            if args.dataset is None:
                raise ValueError("Provide a labeled dataset directory or use --lfw-people")
            result = evaluate(
                args.dataset, args.max_images, args.seed, thresholds,
                model_name=args.model,
            )
    except (RuntimeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
