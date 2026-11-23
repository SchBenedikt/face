#!/usr/bin/env python3
"""Measure local Chroma index-build and query latency with synthetic vectors.

This isolates vector-store overhead. It does not measure face detection,
embedding quality, or identity retrieval accuracy; use evaluate_face_search.py
for a labeled retrieval benchmark.
"""
from __future__ import annotations

import argparse
import gc
import logging
import statistics
import tempfile
import time

import numpy as np

from vector_store import FaceVectorStore

logging.getLogger("vector_store").setLevel(logging.WARNING)


def benchmark(size: int, query_count: int, batch_size: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(size, 512)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = vectors[: min(size, query_count)].copy()

    with tempfile.TemporaryDirectory(prefix=f"face-vector-bench-{size}-") as db_path:
        store = FaceVectorStore(db_path=db_path)
        written = 0
        build_started = time.perf_counter()
        for start in range(0, size, batch_size):
            stop = min(start + batch_size, size)
            face_ids = [f"synthetic_{index}" for index in range(start, stop)]
            metadata = [
                {"image_path": f"/synthetic/{index}.jpg"}
                for index in range(start, stop)
            ]
            written += store.add_face_embeddings_batch(
                face_ids, vectors[start:stop], metadata
            )
        build_seconds = time.perf_counter() - build_started

        latencies_ms = []
        for query in queries:
            query_started = time.perf_counter()
            store.search_similar_faces(query, n_results=50, min_similarity=0.0)
            latencies_ms.append((time.perf_counter() - query_started) * 1000)

        result = {
            "index_size": size,
            "written": written,
            "queries": len(queries),
            "build_seconds": round(build_seconds, 2),
            "search_p50_ms": round(statistics.median(latencies_ms), 2),
            "search_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
            "search_max_ms": round(max(latencies_ms), 2),
        }
        del store

    del vectors, queries
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", default="1000,10000,100000",
        help="Comma-separated vector counts (default: 1000,10000,100000)",
    )
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if any(size < 1 for size in sizes) or args.queries < 1 or args.batch_size < 1:
        parser.error("sizes, queries and batch-size must be positive")

    for size in sizes:
        print(benchmark(size, args.queries, args.batch_size, args.seed))


if __name__ == "__main__":
    main()
