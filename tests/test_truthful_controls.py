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
