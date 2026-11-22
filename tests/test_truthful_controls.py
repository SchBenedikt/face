"""Settings must reach the engine, not just the settings file."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import face_recognition_engine
import settings_store
from face_recognition_engine import FaceRecognitionEngine


@pytest.fixture
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_store.config, "DATA_DIR", tmp_path)
    return tmp_path / settings_store.SETTINGS_FILENAME


def test_detection_model_is_validated_and_persisted(isolated_settings):
    saved, errors = settings_store.save_settings({"detection_model": "hog"})

    assert saved, errors
    assert settings_store.load_settings()["detection_model"] == "hog"

    saved, errors = settings_store.save_settings({"detection_model": "retinaface"})

    assert not saved
    assert "detection_model" in errors
    assert settings_store.load_settings()["detection_model"] == "hog"


def test_engine_reads_the_saved_detection_model(isolated_settings, monkeypatch):
    settings_store.save_settings({"detection_model": "hog"})
    engine = FaceRecognitionEngine()

    captured = {}

    class FakeFaceRecognition:
        @staticmethod
        def face_locations(image, number_of_times_to_upsample=0, model="hog"):
            captured["model"] = model
            return []

    monkeypatch.setattr(face_recognition_engine, "face_recognition", FakeFaceRecognition)
    engine.detect_faces(np.zeros((32, 32, 3), dtype=np.uint8))

    assert engine.detection_model == "hog"
    assert captured["model"] == "hog"


def test_deepface_aligned_crop_is_used_for_embedding(monkeypatch):
    engine = FaceRecognitionEngine()
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    aligned_crop = np.full((64, 64, 3), 173, dtype=np.uint8)
    captured = {}

    class FakeDeepFace:
        @staticmethod
        def extract_faces(**kwargs):
            captured["extract_kwargs"] = kwargs
            return [{
                "facial_area": {"x": 30, "y": 25, "w": 55, "h": 58},
                "confidence": 0.99,
                "face": aligned_crop,
            }]

    monkeypatch.setattr(face_recognition_engine, "DeepFace", FakeDeepFace)
    monkeypatch.setattr(face_recognition_engine, "face_recognition", None)

    def capture_embedding(face_image, model_name):
        captured["face_image"] = face_image.copy()
        captured["model_name"] = model_name
        return np.ones(512, dtype=np.float32)

    monkeypatch.setattr(engine, "_extract_single_model_embedding", capture_embedding)
    locations = engine.detect_faces(image)
    assert locations == [(25, 85, 83, 30)]

    embedding = engine.extract_face_embedding(image, locations[0])

    assert embedding is not None
    np.testing.assert_array_equal(captured["face_image"], aligned_crop)
    assert captured["model_name"] == engine.primary_model
    assert captured["extract_kwargs"]["align"] is True
    assert captured["extract_kwargs"]["color_face"] == "rgb"
    assert captured["extract_kwargs"]["normalize_face"] is False


def test_invalid_primary_detection_does_not_block_deepface_fallback(monkeypatch):
    engine = FaceRecognitionEngine()
    image = np.zeros((120, 120, 3), dtype=np.uint8)

    class FakeFaceRecognition:
        @staticmethod
        def face_locations(image, number_of_times_to_upsample=0, model="hog"):
            return [(0, 10, 10, 0)]  # Below the configured minimum size.

    class FakeDeepFace:
        @staticmethod
        def extract_faces(**kwargs):
            return [{
                "facial_area": {"x": 30, "y": 25, "w": 55, "h": 58},
                "confidence": 0.99,
                "face": np.full((64, 64, 3), 173, dtype=np.uint8),
            }]

    monkeypatch.setattr(face_recognition_engine, "face_recognition", FakeFaceRecognition)
    monkeypatch.setattr(face_recognition_engine, "DeepFace", FakeDeepFace)

    assert engine.detect_faces(image) == [(25, 85, 83, 30)]
