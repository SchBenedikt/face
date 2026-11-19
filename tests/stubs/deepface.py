"""
Minimal stand-in for the ``deepface`` package (which needs TensorFlow).

Only used by the UI tests when the real package is not installed. Embeddings
are derived deterministically from the pixels, so repeated runs produce the
same vectors and the vector store can be exercised for real.
"""
from __future__ import annotations

import numpy as np

EMBEDDING_SIZE = 512


def _vector_from_image(image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.size == 0:
        return np.zeros(EMBEDDING_SIZE, dtype=np.float32)
    flat = array.reshape(-1)
    indices = np.linspace(0, flat.size - 1, EMBEDDING_SIZE).astype(np.int64)
    vector = np.array(flat[indices], dtype=np.float32)
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    if norm > 1e-10:
        vector = vector / norm
    return vector


class DeepFace:
    """The small subset of the DeepFace API that the app calls."""

    @staticmethod
    def represent(
        img,
        model_name="Facenet512",
        enforce_detection=False,
        detector_backend="skip",
        normalization="ArcFace",
        **kwargs,
    ):
        return [{"embedding": _vector_from_image(img).tolist(), "face_confidence": 1.0}]

    @staticmethod
    def analyze(
        img_path=None,
        actions=("age", "gender", "race", "emotion"),
        enforce_detection=False,
        silent=True,
        **kwargs,
    ):
        return {
            "age": 34,
            "gender": "Man",
            "dominant_gender": "Man",
            "emotion": {
                "angry": 0.4,
                "disgust": 0.1,
                "fear": 0.2,
                "happy": 2.1,
                "sad": 0.5,
                "surprise": 0.6,
                "neutral": 96.1,
            },
            "dominant_emotion": "neutral",
            "race": {"white": 96.0, "black": 2.0, "asian": 1.5, "latino hispanic": 0.5},
            "dominant_race": "white",
            "region": {"x": 0, "y": 0, "w": 64, "h": 64},
        }

    @staticmethod
    def build_model(model_name="Facenet512"):
        return None
