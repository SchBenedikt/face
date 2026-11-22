"""Editable, persistent application settings.

Defaults come from ``config.py``; overrides are stored as JSON inside the data
directory (``FACE_DATA_DIR`` aware) so they survive a server restart.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import config

SETTINGS_FILENAME = "settings.json"

# name: (default, type, minimum, maximum)
SCHEMA: Dict[str, Tuple[Any, type, float, float]] = {
    "similarity_threshold": (float(config.SIMILARITY_THRESHOLD), float, 0.0, 1.0),
    "max_results": (50, int, 1, 200),
    "batch_size": (int(config.BATCH_SIZE), int, 1, 256),
    "detection_model": (
        str(getattr(config, "FACE_RECOGNITION_MODEL", "cnn")), str, 0, 0
    ),
}

CHOICES: Dict[str, Tuple[str, ...]] = {
    "detection_model": ("hog", "cnn"),
}


def settings_path() -> Path:
    """Location of the persisted settings file (inside the data directory)."""
    return Path(config.DATA_DIR) / SETTINGS_FILENAME


def defaults() -> Dict[str, Any]:
    return {name: spec[0] for name, spec in SCHEMA.items()}


def _coerce(name: str, raw: Any) -> Any:
    default, kind, minimum, maximum = SCHEMA[name]
    if isinstance(raw, bool):
        raise ValueError("Wahrheitswerte sind hier nicht erlaubt")
    if kind is str:
        value = str(raw).strip().lower()
        allowed = CHOICES.get(name)
        if allowed and value not in allowed:
            raise ValueError(f"muss einer von {', '.join(allowed)} sein")
        return value
    if kind is float:
        value = float(str(raw).strip().replace(",", "."))
    else:
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError("muss eine ganze Zahl sein")
        value = int(str(raw).strip())
    if not minimum <= value <= maximum:
        raise ValueError(f"muss zwischen {minimum:g} und {maximum:g} liegen")
    return value


def validate_settings(values: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Return ``(clean_values, errors)``; unknown keys are ignored."""
    clean: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    if not isinstance(values, dict):
        return clean, {"_": "Einstellungen müssen ein Objekt sein"}

    for name in SCHEMA:
        if name not in values:
            continue
        try:
            clean[name] = _coerce(name, values[name])
        except (TypeError, ValueError) as exc:
            errors[name] = str(exc)

    return clean, errors


def load_settings() -> Dict[str, Any]:
    """Return the effective settings (defaults merged with the saved file).

    A corrupt or invalid file never breaks the app: valid entries win, invalid
    ones fall back to their defaults.
    """
    values = defaults()
    path = settings_path()
    try:
        if not path.exists():
            return values
        data = json.loads(path.read_text(encoding="utf-8"))
        clean, _errors = validate_settings(data if isinstance(data, dict) else {})
        values.update(clean)
    except Exception:  # noqa: BLE001 - unreadable settings must not crash the app
        pass
    return values


def active_embedding_pipeline_version() -> str:
    """Return the pipeline id for the detector this installation will use.

    The optional ``face_recognition`` package changes the active detector to
    dlib and its HOG/CNN setting changes face boxes and crops. Those vectors
    must not share an index version with DeepFace-aligned embeddings or with
    the other dlib model.
    """
    try:
        import face_recognition  # noqa: F401 - availability selects the path
    except Exception:  # noqa: BLE001 - optional dlib stack may be unavailable
        return str(config.EMBEDDING_PIPELINE_VERSION)

    model = str(load_settings().get("detection_model", "cnn")).lower()
    if model not in CHOICES["detection_model"]:
        model = "cnn"
    return f"{config.EMBEDDING_PIPELINE_VERSION}-dlib-{model}"


def save_settings(values: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """Validate and persist ``values``; nothing is written when they are invalid."""
    clean, errors = validate_settings(values)
    if errors:
        return False, errors

    merged = load_settings()
    merged.update(clean)

    path = settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle_fd, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                json.dump(merged, handle, indent=2, ensure_ascii=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except OSError as exc:
        return False, {"_": f"Konnte die Einstellungen nicht speichern: {exc}"}

    return True, {}
