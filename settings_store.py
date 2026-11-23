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
    "embedding_model": (str(config.FACE_EMBEDDING_MODEL).lower(), str, 0, 0),
    "similarity_threshold": (float(config.SIMILARITY_THRESHOLD), float, 0.0, 1.0),
    "max_results": (50, int, 1, 200),
    "batch_size": (int(config.BATCH_SIZE), int, 1, 256),
    "detection_model": (
        str(getattr(config, "FACE_RECOGNITION_MODEL", "cnn")), str, 0, 0
    ),
}

CHOICES: Dict[str, Tuple[str, ...]] = {
    "embedding_model": ("buffalo_l", "arcface"),
    "detection_model": ("hog", "cnn"),
}

_MODEL_DEFAULT_THRESHOLDS = {
    "facenet512": 0.65,
    "buffalo_l": 0.40,
    "arcface": 0.40,
}

_MODEL_GROUPING_THRESHOLDS = {
    "buffalo_l": 0.35,
    "arcface": 0.55,
}


def settings_path() -> Path:
    """Location of the persisted settings file (inside the data directory)."""
    return Path(config.DATA_DIR) / SETTINGS_FILENAME


def defaults() -> Dict[str, Any]:
    return {name: spec[0] for name, spec in SCHEMA.items()}


def _coerce(name: str, raw: Any) -> Any:
    default, kind, minimum, maximum = SCHEMA[name]
    if isinstance(raw, bool):
        raise ValueError("Boolean values are not allowed")
    if kind is str:
        value = str(raw).strip().lower()
        allowed = CHOICES.get(name)
        if allowed and value not in allowed:
            raise ValueError(f"must be one of {', '.join(allowed)}")
        return value
    if kind is float:
        value = float(str(raw).strip().replace(",", "."))
    else:
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError("must be a whole number")
        value = int(str(raw).strip())
    if not minimum <= value <= maximum:
        raise ValueError(f"must be between {minimum:g} and {maximum:g}")
    return value


def validate_settings(values: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Return ``(clean_values, errors)``; unknown keys are ignored."""
    clean: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    if not isinstance(values, dict):
        return clean, {"_": "Settings must be an object"}

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
        if isinstance(data, dict):
            previous_model = str(data.get("similarity_model") or "Facenet512").lower()
            current_model = str(
                clean.get("embedding_model", config.FACE_EMBEDDING_MODEL)
            ).lower()
            previous_default = _MODEL_DEFAULT_THRESHOLDS.get(previous_model)
            saved_threshold = clean.get("similarity_threshold")
            if (
                previous_model != current_model
                and previous_default is not None
                and saved_threshold == previous_default
            ):
                # Migrate an unchanged old default when the app changes encoder.
                # User-selected thresholds remain intact unless they equal the
                # previous model's shipped default.
                clean["similarity_threshold"] = float(config.SIMILARITY_THRESHOLD)
        values.update(clean)
    except Exception:  # noqa: BLE001 - unreadable settings must not crash the app
        pass
    return values


def active_embedding_pipeline_version(model_name: str | None = None) -> str:
    """Return the pipeline id for the detector this installation will use.

    The embedding model is part of the index identity. DeepFace alignment is
    the primary detector path. If its detectors cannot find a face, optional
    dlib HOG/CNN or Haar fallbacks may be used; their availability and setting
    therefore remain part of the index version.
    """
    model = (model_name or active_embedding_model()).lower()
    base_version = str(config.EMBEDDING_PIPELINE_VERSION)
    if model != str(config.FACE_EMBEDDING_MODEL).lower():
        base_version = f"{base_version}-{model}"
    try:
        import face_recognition  # noqa: F401 - availability selects the path
    except Exception:  # noqa: BLE001 - optional dlib stack may be unavailable
        return base_version

    model = str(load_settings().get("detection_model", "cnn")).lower()
    if model not in CHOICES["detection_model"]:
        model = "cnn"
    return f"{base_version}-dlib-{model}"


def active_embedding_model() -> str:
    """Return the selected DeepFace embedding model in its display spelling."""
    selected = str(load_settings().get(
        "embedding_model", config.FACE_EMBEDDING_MODEL
    )).lower()
    return {"buffalo_l": "Buffalo_L", "arcface": "ArcFace"}.get(
        selected, str(config.FACE_EMBEDDING_MODEL)
    )


def active_grouping_threshold() -> float:
    """Return an evidence-based starting threshold for the selected model."""
    override = os.environ.get("FACE_GROUPING_THRESHOLD")
    if override is not None:
        try:
            value = float(override)
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            pass
    model = active_embedding_model().lower()
    return _MODEL_GROUPING_THRESHOLDS.get(model, float(config.FACE_GROUPING_THRESHOLD))


def save_settings(values: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """Validate and persist ``values``; nothing is written when they are invalid."""
    clean, errors = validate_settings(values)
    if errors:
        return False, errors

    merged = load_settings()
    merged.update(clean)
    # Keep the model associated with its last selected score threshold without
    # exposing this internal migration marker as an editable UI setting.
    merged["similarity_model"] = str(
        clean.get("embedding_model", active_embedding_model())
    )

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
        return False, {"_": f"Could not save settings: {exc}"}

    return True, {}
