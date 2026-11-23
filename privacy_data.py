"""Explicit local deletion for indexed biometric data and recovery archives."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable

CHUNK_SIZE = 250


def _remove_directory_contents(directory: Path) -> int:
    """Remove files below a configured app-owned directory, keeping the root."""
    if not directory.exists() or directory.is_symlink():
        return 0
    removed = 0
    for child in directory.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink(missing_ok=True)
            removed += 1
        elif child.is_dir():
            shutil.rmtree(child)
            removed += 1
    return removed


def delete_local_biometric_data(
    store: Any,
    *,
    delete_source_images: bool,
    managed_image_directories: Iterable[str | Path],
    derived_directories: Iterable[str | Path],
    backup_directories: Iterable[str | Path],
    metadata_stores: Iterable[Any] = (),
    auxiliary_files: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Clear all face rows and biometric archives, optionally deleting images.

    The caller must obtain explicit user confirmation before invoking this
    irreversible operation. Referenced files outside the app's managed image
    directories are unlinked only when ``delete_source_images`` is true.
    """
    errors: list[str] = []
    total = 0
    removed_images: set[Path] = set()
    removed_derived = 0
    removed_backups = 0
    removed_auxiliary_files = 0
    index_cleared = False
    try:
        if store is None:
            raise RuntimeError("The face index is not initialized.")
        total = int(store.collection.count())
        source_paths: set[Path] = set()
        for offset in range(0, total, CHUNK_SIZE):
            rows = store.collection.get(
                limit=min(CHUNK_SIZE, total - offset),
                offset=offset,
                include=["metadatas"],
            )
            for metadata in rows.get("metadatas") or []:
                raw_path = str((metadata or {}).get("image_path") or "").strip()
                if raw_path:
                    source_paths.add(Path(raw_path).expanduser())

        index_cleared = bool(store.clear_collection())
        index_cleared = index_cleared and int(store.collection.count()) == 0
        if not index_cleared:
            errors.append("The face index could not be fully cleared.")

        def record_file_error(path: Path, action: str) -> None:
            try:
                if path.is_symlink() or path.is_file():
                    path.unlink(missing_ok=True)
                    removed_images.add(path)
            except OSError as exc:
                errors.append(f"Could not {action} {path}: {exc}")

        if delete_source_images:
            for source_path in source_paths:
                record_file_error(source_path, "delete")
            for raw_directory in managed_image_directories:
                directory = Path(raw_directory).expanduser()
                try:
                    if not directory.exists() or directory.is_symlink():
                        continue
                    for child in directory.iterdir():
                        try:
                            if child.is_symlink() or child.is_file():
                                child.unlink(missing_ok=True)
                                removed_images.add(child)
                            elif child.is_dir():
                                shutil.rmtree(child)
                                removed_images.add(child)
                        except OSError as exc:
                            errors.append(f"Could not delete {child}: {exc}")
                except OSError as exc:
                    errors.append(f"Could not inspect {directory}: {exc}")

        for raw_directory in derived_directories:
            try:
                removed_derived += _remove_directory_contents(
                    Path(raw_directory).expanduser()
                )
            except OSError as exc:
                errors.append(f"Could not clear derived data: {exc}")

        for raw_directory in backup_directories:
            try:
                removed_backups += _remove_directory_contents(
                    Path(raw_directory).expanduser()
                )
            except OSError as exc:
                errors.append(f"Could not clear recovery archives: {exc}")

        for metadata_store in metadata_stores:
            try:
                if not metadata_store.clear_hash_database():
                    errors.append("Could not clear duplicate image metadata.")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Could not clear duplicate image metadata: {exc}")

        for raw_file in auxiliary_files:
            path = Path(raw_file).expanduser()
            try:
                if path.is_symlink() or path.is_file():
                    path.unlink(missing_ok=True)
                    removed_auxiliary_files += 1
                for suffix in ("-journal", "-wal", "-shm"):
                    Path(f"{path}{suffix}").unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"Could not remove legacy metadata file {path}: {exc}")
    except Exception as exc:  # noqa: BLE001 - return a useful UI error
        errors.append(str(exc))
    finally:
        # Clear in-memory copies even when one filesystem removal fails.
        try:
            from ui_cache import clear_thumbnail_cache

            clear_thumbnail_cache()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Could not clear thumbnail cache: {exc}")
        if store is not None:
            try:
                store.invalidate_stats_cache()
                store._pipeline_status_cache = None
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Could not clear index status cache: {exc}")

    result = {
        "success": index_cleared and not errors,
        "faces_removed": total if index_cleared else 0,
        "images_removed": len(removed_images),
        "derived_entries_removed": removed_derived,
        "backup_files_removed": removed_backups,
        "auxiliary_files_removed": removed_auxiliary_files,
    }
    if errors:
        result["error"] = "Local data deletion was incomplete: " + "; ".join(errors)
    return result
