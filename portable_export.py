"""Versioned, integrity-checked face archive export and restore helpers."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable

import numpy as np

ARCHIVE_SCHEMA = 1
CHUNK_SIZE = 250
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def create_face_export(
    collection: Any,
    pipeline_version: str,
    *,
    face_ids: Iterable[str] | None = None,
    include_embeddings: bool = False,
    include_images: bool = False,
) -> bytes:
    """Create an archive in a temporary file and return its bytes.

    Chroma rows are fetched in bounded chunks. Images are copied directly into
    the ZIP writer, and duplicate source files are stored once by content hash.
    """
    selected_ids = list(dict.fromkeys(str(value) for value in face_ids or []))
    total = len(selected_ids) if face_ids is not None else int(collection.count())
    manifest_files: dict[str, str] = {}
    image_members: dict[str, str] = {}
    image_hashes: dict[str, str] = {}
    record_count = 0

    with tempfile.TemporaryDirectory(prefix="face-export-") as temp_dir:
        archive_path = Path(temp_dir) / "faces.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            offsets = range(0, total, CHUNK_SIZE)
            for chunk_number, offset in enumerate(offsets):
                options: dict[str, Any] = {"include": ["metadatas"]}
                if include_embeddings:
                    options["include"].append("embeddings")
                if face_ids is None:
                    options.update({"limit": min(CHUNK_SIZE, total - offset), "offset": offset})
                    rows = collection.get(**options)
                else:
                    batch_ids = selected_ids[offset:offset + CHUNK_SIZE]
                    if not batch_ids:
                        continue
                    rows = collection.get(ids=batch_ids, **options)

                ids = rows.get("ids") or []
                metadata_rows = rows.get("metadatas") or []
                embedding_rows = rows.get("embeddings") if include_embeddings else None
                records = []
                for index, face_id in enumerate(ids):
                    metadata = dict(metadata_rows[index] or {}) if index < len(metadata_rows) else {}
                    record: dict[str, Any] = {"face_id": str(face_id), "metadata": metadata}
                    if include_embeddings and embedding_rows is not None and index < len(embedding_rows):
                        record["embedding"] = np.asarray(embedding_rows[index], dtype=np.float32).tolist()
                    image_path = Path(str(metadata.get("image_path") or ""))
                    if include_images and image_path.is_file():
                        image_hash = _sha256_file(image_path)
                        member = image_members.get(image_hash)
                        if member is None:
                            member = f"images/{image_hash}{image_path.suffix.lower()}"
                            archive.write(image_path, member)
                            image_members[image_hash] = member
                            image_hashes[member] = image_hash
                        record["image_member"] = member
                    records.append(record)

                if not records:
                    continue
                member = f"records/{chunk_number:06d}.jsonl"
                content = "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in records).encode("utf-8")
                archive.writestr(member, content)
                manifest_files[member] = hashlib.sha256(content).hexdigest()
                record_count += len(records)

            for member, digest in image_hashes.items():
                manifest_files[member] = digest

            if record_count != total:
                raise ValueError(
                    f"The export found {record_count} of {total} requested faces. Retry after the index stops changing."
                )

            manifest = {
                "schema_version": ARCHIVE_SCHEMA,
                "embedding_pipeline_version": pipeline_version,
                "face_count": record_count,
                "includes_embeddings": include_embeddings,
                "includes_images": include_images,
                "files": manifest_files,
            }
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))

        if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("The export exceeds the 128 MiB download limit. Export fewer faces or omit images.")
        return archive_path.read_bytes()


def restore_face_export(
    archive_data: bytes | BinaryIO,
    vector_store: Any,
    image_dir: str | Path,
) -> dict[str, Any]:
    """Validate and restore an archive into a compatible local vector store."""
    image_dir = Path(image_dir).expanduser().resolve()
    created_images: list[Path] = []
    inserted_ids: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="face-restore-") as temp_dir:
            archive_path = Path(temp_dir) / "restore.zip"
            if hasattr(archive_data, "read"):
                with archive_path.open("wb") as destination:
                    shutil.copyfileobj(archive_data, destination)
            else:
                if len(archive_data) > MAX_ARCHIVE_BYTES:
                    raise ValueError("The archive exceeds the 128 MiB limit.")
                archive_path.write_bytes(archive_data)

            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ValueError("The archive contains duplicate file entries.")
                if any(not _is_safe_member(name) for name in names):
                    raise ValueError("The archive contains an unsafe file path.")
                if sum(info.file_size for info in archive.infolist()) > 1024 * 1024 * 1024:
                    raise ValueError("The uncompressed archive exceeds the 1 GiB restore limit.")
                manifest = json.loads(archive.read("manifest.json"))
                if manifest.get("schema_version") != ARCHIVE_SCHEMA:
                    raise ValueError("Unsupported face archive schema version.")
                if manifest.get("embedding_pipeline_version") != vector_store._embedding_pipeline_version:
                    raise ValueError("The archive uses a different embedding pipeline. Reindex or switch to a compatible model first.")

                expected_files = manifest.get("files") or {}
                if set(names) - {"manifest.json"} != set(expected_files):
                    raise ValueError("The archive contains files that are not covered by its integrity manifest.")
                for member, expected_hash in expected_files.items():
                    if member not in names or not _is_safe_member(member):
                        raise ValueError("The archive manifest references a missing or unsafe file.")
                    digest = hashlib.sha256()
                    with archive.open(member) as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != expected_hash:
                        raise ValueError(f"Integrity check failed for {member}.")

                if not manifest.get("includes_embeddings"):
                    raise ValueError("This archive contains metadata only and cannot restore the searchable index.")

                restore_id = uuid.uuid4().hex

                batch_ids: list[str] = []
                batch_vectors: list[list[float]] = []
                batch_metadata: list[dict[str, Any]] = []
                restored_records = 0
                for member in sorted(name for name in names if name.startswith("records/") and name.endswith(".jsonl")):
                    with archive.open(member) as source:
                        for raw_line in source:
                            record = json.loads(raw_line)
                            face_id = str(record["face_id"])
                            vector = np.asarray(record["embedding"], dtype=np.float32)
                            if (
                                vector.ndim != 1
                                or vector.shape[0] != 512
                                or not np.all(np.isfinite(vector))
                                or not np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-3)
                            ):
                                raise ValueError(f"Invalid embedding in {member}.")
                            metadata = dict(record.get("metadata") or {})
                            metadata.update({
                                "face_id": face_id,
                                "embedding_pipeline_version": vector_store._embedding_pipeline_version,
                            })
                            image_member = record.get("image_member")
                            if image_member:
                                suffix = Path(image_member).suffix.lower()
                                if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                                    raise ValueError("The archive contains an unsupported image type.")
                                restored_path = image_dir / "restored" / restore_id / f"{hashlib.sha256(face_id.encode()).hexdigest()}{suffix}"
                                restored_path.parent.mkdir(parents=True, exist_ok=True)
                                with archive.open(image_member) as source, restored_path.open("wb") as destination:
                                    shutil.copyfileobj(source, destination)
                                created_images.append(restored_path)
                                metadata["image_path"] = str(restored_path)
                            batch_ids.append(face_id)
                            batch_vectors.append(vector.tolist())
                            batch_metadata.append(metadata)
                            restored_records += 1
                            if len(batch_ids) >= CHUNK_SIZE:
                                existing = vector_store.collection.get(ids=batch_ids, include=[]).get("ids") or []
                                if existing:
                                    raise ValueError("The archive contains face IDs already present in this index.")
                                inserted_ids.extend(batch_ids)
                                vector_store.collection.upsert(ids=batch_ids, embeddings=batch_vectors, metadatas=batch_metadata)
                                batch_ids, batch_vectors, batch_metadata = [], [], []
                if batch_ids:
                    existing = vector_store.collection.get(ids=batch_ids, include=[]).get("ids") or []
                    if existing:
                        raise ValueError("The archive contains face IDs already present in this index.")
                    inserted_ids.extend(batch_ids)
                    vector_store.collection.upsert(ids=batch_ids, embeddings=batch_vectors, metadatas=batch_metadata)
                if restored_records != int(manifest.get("face_count", -1)):
                    raise ValueError("The archive face count does not match its manifest.")
                vector_store.invalidate_stats_cache()
                vector_store._pipeline_status_cache = None
                vector_store.mark_embedding_pipeline_complete()
                return {"success": True, "restored_faces": len(inserted_ids), "restored_images": len(created_images)}
    except Exception as exc:  # noqa: BLE001 - report a clean rollback result
        if inserted_ids:
            try:
                vector_store.collection.delete(ids=inserted_ids)
                vector_store.invalidate_stats_cache()
                vector_store._pipeline_status_cache = None
            except Exception:
                pass
        for path in created_images:
            path.unlink(missing_ok=True)
        return {"success": False, "error": str(exc), "restored_faces": 0, "restored_images": 0}
