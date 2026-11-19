"""Settings must be validated, persisted and reloadable."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import settings_store


@pytest.fixture
def isolated_settings(monkeypatch, tmp_path):
    """Point the settings file at a temporary data directory."""
    monkeypatch.setattr(settings_store.config, "DATA_DIR", tmp_path)
    return tmp_path / settings_store.SETTINGS_FILENAME


def test_defaults_are_used_without_a_file(isolated_settings):
    values = settings_store.load_settings()

    assert values["max_results"] == 50
    assert 0.0 <= values["similarity_threshold"] <= 1.0
    assert values["batch_size"] >= 1


def test_saved_values_survive_a_reload(isolated_settings):
    saved, errors = settings_store.save_settings(
        {"similarity_threshold": 0.85, "max_results": 7, "batch_size": 16}
    )

    assert saved and errors == {}
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert stored["similarity_threshold"] == 0.85

    # load_settings keeps no state: this is exactly what a restart reads.
    assert settings_store.load_settings() == {
        "similarity_threshold": 0.85,
        "max_results": 7,
        "batch_size": 16,
        "detection_model": "cnn",
    }


def test_invalid_values_are_rejected_without_touching_the_file(isolated_settings):
    assert settings_store.save_settings({"similarity_threshold": 0.5})[0]

    saved, errors = settings_store.save_settings(
        {"similarity_threshold": 1.5, "max_results": 0, "batch_size": 999}
    )

    assert not saved
    assert set(errors) == {"similarity_threshold", "max_results", "batch_size"}
    assert json.loads(isolated_settings.read_text(encoding="utf-8"))["similarity_threshold"] == 0.5
