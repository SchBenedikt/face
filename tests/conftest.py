"""
Shared setup for the headless Streamlit UI tests.

The app imports ``face_recognition`` (needs dlib) and ``deepface`` (needs
TensorFlow) at import time. Those two packages cannot be built here, so the
real package is used when it is installed and a minimal stub from
``tests/stubs`` is used otherwise. Everything else (Streamlit, ChromaDB,
OpenCV, Pillow) is the real thing.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUBS_DIR = Path(__file__).resolve().parent / "stubs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for module_name in ("face_recognition", "deepface"):
    try:
        __import__(module_name)
    except Exception:  # noqa: BLE001 - any import-time failure means "use the stub"
        if str(STUBS_DIR) not in sys.path:
            sys.path.insert(0, str(STUBS_DIR))

# Keep user data (vector DB, images) out of the repository while testing.
SESSION_DATA_DIR = Path(tempfile.mkdtemp(prefix="face-ui-data-"))
os.environ["FACE_DATA_DIR"] = str(SESSION_DATA_DIR)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def _face_embedding(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=512).astype(np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture(scope="session")
def seeded_data():
    """Create synthetic images, scraped images and a real ChromaDB collection."""
    from PIL import Image

    from config import (
        COLLECTION_NAME,
        EMBEDDING_PIPELINE_VERSION,
        IMAGES_DIR,
        SCRAPED_DIR,
        VECTOR_DB_PATH,
    )

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    SCRAPED_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2024)
    ids = []
    embeddings = []
    metadatas = []

    for image_index in range(4):
        colour = tuple(int(value) for value in rng.integers(40, 220, 3))
        image_path = IMAGES_DIR / f"person_{image_index}.jpg"
        Image.new("RGB", (480, 360), colour).save(image_path, format="JPEG")

        for face_index in range(2):
            top = 40 + face_index * 160
            face_id = f"face_{image_index}_{face_index}"
            metadata = {
                "face_id": face_id,
                "image_path": str(image_path),
                "location": f"{top},{top + 120},{top + 120},40",
                "embedding_dimension": 512,
                "embedding_pipeline_version": EMBEDDING_PIPELINE_VERSION,
                "added_at": "2026-01-01T00:00:00",
            }
            # Two of the four images belong to a known person.
            if image_index < 2:
                metadata.update({
                    "person_id": "person-erika",
                    "first_name": "Erika",
                    "middle_names": "",
                    "last_name": "Mustermann",
                    "full_name": "Erika Mustermann",
                    "birth_date": "",
                    "birth_place": "",
                    "notes": "",
                })
            ids.append(face_id)
            embeddings.append(_face_embedding(rng).tolist())
            metadatas.append(metadata)

    for scraped_index in range(3):
        Image.new(
            "RGB",
            (320, 240),
            tuple(int(value) for value in rng.integers(40, 220, 3)),
        ).save(SCRAPED_DIR / f"scraped_{scraped_index}.jpg", format="JPEG")

    import chromadb

    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - first run has no collection yet
        pass
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={
            "description": "Headless UI test fixtures",
            "embedding_pipeline_version": EMBEDDING_PIPELINE_VERSION,
            "embedding_pipeline_complete": True,
        },
    )
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    return {
        "ids": ids,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "data_dir": SESSION_DATA_DIR,
    }
