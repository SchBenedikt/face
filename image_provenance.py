"""Small sidecar records for images collected through the web importer."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def provenance_path(image_path: str | Path) -> Path:
    """Return the app-owned provenance sidecar path for an image."""
    image = Path(image_path)
    return image.with_name(image.name + ".provenance.json")


def write_image_provenance(
    image_path: str | Path,
    *,
    source_url: str,
    authorization_confirmed: bool,
) -> dict[str, Any]:
    """Record the source URL and the user's authorization confirmation."""
    record = {
        "source_type": "web_import",
        "source_url": str(source_url),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "authorization_confirmed": bool(authorization_confirmed),
    }
    target = provenance_path(image_path)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(target)
    return record


def read_image_provenance(image_path: str | Path) -> dict[str, Any]:
    """Read a valid provenance record or return an empty mapping."""
    raw_path = str(image_path or "").strip()
    if not raw_path or not Path(raw_path).name:
        return {}
    path = provenance_path(raw_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}
