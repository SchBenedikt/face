"""Local recovery archives for duplicate cleanup."""
from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from portable_export import MAX_ARCHIVE_BYTES, create_face_export, restore_face_export


def _face_ids_for_paths(collection: Any, paths: list[str]) -> list[str]:
    """Find face IDs for exact source paths, with compatibility fallback."""
    if not paths:
        return []
    try:
        rows = collection.get(where={"image_path": {"$in": paths}}, include=[])
        return list(dict.fromkeys(str(value) for value in rows.get("ids") or []))
    except Exception:  # noqa: BLE001 - older Chroma versions may lack $in
        found: list[str] = []
        for path in paths:
            try:
                rows = collection.get(where={"image_path": path}, include=[])
                found.extend(str(value) for value in rows.get("ids") or [])
            except Exception:  # noqa: BLE001 - keep scanning other source paths
                continue
        return list(dict.fromkeys(found))


def create_duplicate_backup(
    vector_store: Any,
    image_paths: Iterable[str | Path],
    backup_directory: str | Path,
) -> dict[str, Any]:
    """Persist a searchable, restorable copy of affected faces and source images."""
    paths = list(dict.fromkeys(str(Path(path)) for path in image_paths))
    face_ids = _face_ids_for_paths(vector_store.collection, paths)
    if not face_ids:
        return {
            "success": True,
            "backup_path": None,
            "face_count": 0,
            "message": "No indexed faces are attached to the files being removed.",
        }

    try:
        archive_data = create_face_export(
            vector_store.collection,
            vector_store._embedding_pipeline_version,
            face_ids=face_ids,
            include_embeddings=True,
            include_images=True,
        )
    except Exception as exc:  # noqa: BLE001 - cleanup must stop if backup fails
        return {
            "success": False,
            "backup_path": None,
            "face_count": len(face_ids),
            "message": f"Could not prepare the recovery archive: {exc}",
        }
    if len(archive_data) > MAX_ARCHIVE_BYTES:
        return {
            "success": False,
            "backup_path": None,
            "face_count": len(face_ids),
            "message": "The recovery archive would exceed the 128 MiB limit. Nothing was removed.",
        }

    directory = Path(backup_directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = directory / f"duplicate-recovery-{stamp}-{uuid.uuid4().hex[:8]}.zip"
    fd, temporary_path = tempfile.mkstemp(prefix=".duplicate-recovery-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as destination:
            destination.write(archive_data)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, target)
    finally:
        Path(temporary_path).unlink(missing_ok=True)

    return {
        "success": True,
        "backup_path": target,
        "face_count": len(face_ids),
        "message": "A local recovery archive is ready.",
    }


def restore_duplicate_backup(
    archive_path: str | Path,
    vector_store: Any,
    image_directory: str | Path,
) -> dict[str, Any]:
    """Restore a local duplicate cleanup archive and remove it after success."""
    path = Path(archive_path)
    result = restore_face_export(
        path.read_bytes(),
        vector_store,
        image_directory,
        skip_existing=True,
    )
    if result.get("success"):
        path.unlink(missing_ok=True)
    return result
