"""
Vector Store for fast face similarity search using ChromaDB
"""
# Disable ChromaDB telemetry before any imports
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_CLIENT_DISABLE_TELEMETRY'] = 'True'
os.environ['CHROMA_DISABLE_TELEMETRY'] = '1'

import chromadb
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import sqlite3
import tempfile
from datetime import datetime
import uuid

from config import (
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    SIMILARITY_THRESHOLD,
    FACE_GROUPING_MARGIN,
    FACE_AUTO_ASSIGN_THRESHOLD,
    FACE_AUTO_ASSIGN_AMBIGUITY_MARGIN,
)
from face_grouping import centroid_of, group_embeddings
from settings_store import (
    active_embedding_pipeline_metadata,
    active_embedding_pipeline_version,
    active_grouping_threshold,
)
from similarity import (
    cosine_from_chroma_distance,
    normalise_embedding,
    similarity_from_chroma_distance,
)

# Additional telemetry suppression after imports
try:
    import chromadb.telemetry.product.posthog as posthog_telemetry
    # Monkey patch the capture method to do nothing
    def no_op_capture(*args, **kwargs):
        pass
    posthog_telemetry.Posthog.capture = no_op_capture
except (ImportError, AttributeError):
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PERSON_METADATA_FIELDS = (
    "person_id",
    "first_name",
    "middle_names",
    "last_name",
    "full_name",
    "birth_date",
    "birth_place",
    "notes",
    "name_assigned_at",
)


def _face_box(value: Any) -> Optional[Tuple[int, int, int, int]]:
    """Parse a face box stored as ``top,right,bottom,left``."""
    try:
        if isinstance(value, str):
            coordinates = [int(part.strip()) for part in value.split(",")]
        elif isinstance(value, (list, tuple)):
            coordinates = [int(part) for part in value]
        else:
            return None
        if len(coordinates) != 4:
            return None
        top, right, bottom, left = coordinates
        return (top, right, bottom, left) if right > left and bottom > top else None
    except (TypeError, ValueError):
        return None


def _face_box_iou(
    first: Optional[Tuple[int, int, int, int]],
    second: Optional[Tuple[int, int, int, int]],
) -> float:
    """Return intersection-over-union for two ``top,right,bottom,left`` boxes."""
    if first is None or second is None:
        return 0.0
    top = max(first[0], second[0])
    right = min(first[1], second[1])
    bottom = min(first[2], second[2])
    left = max(first[3], second[3])
    intersection = max(0, bottom - top) * max(0, right - left)
    if not intersection:
        return 0.0
    first_area = (first[2] - first[0]) * (first[1] - first[3])
    second_area = (second[2] - second[0]) * (second[1] - second[3])
    return intersection / float(first_area + second_area - intersection)


def _preserve_face_metadata(
    existing: Dict[str, Any], incoming: Dict[str, Any]
) -> Dict[str, Any]:
    """Keep a saved name only when the refreshed detection is the same face."""
    preserved = dict(existing)
    if not any(preserved.get(field) for field in _PERSON_METADATA_FIELDS):
        return preserved

    overlap = _face_box_iou(
        _face_box(existing.get("location")),
        _face_box(incoming.get("location")),
    )
    if overlap < 0.35:
        for field in _PERSON_METADATA_FIELDS:
            preserved.pop(field, None)
        logger.warning(
            "Discarded stale person metadata for a reused face ID because its "
            "new box did not match the old box (IoU %.3f)",
            overlap,
        )
    return preserved


def _inherit_matched_person_metadata(
    incoming_metadatas: List[Dict[str, Any]],
    existing_metadatas: List[Dict[str, Any]],
    minimum_iou: float = 0.35,
) -> List[Dict[str, Any]]:
    """Carry names across detector reordering using one-to-one box matches."""
    updated = [dict(metadata or {}) for metadata in incoming_metadatas]
    incoming_by_image: Dict[str, List[int]] = {}
    existing_by_image: Dict[str, List[int]] = {}
    for index, metadata in enumerate(updated):
        image_path = str(metadata.get("image_path") or "")
        if image_path:
            incoming_by_image.setdefault(image_path, []).append(index)
    for index, metadata in enumerate(existing_metadatas):
        image_path = str(metadata.get("image_path") or "")
        if image_path and any(metadata.get(field) for field in _PERSON_METADATA_FIELDS):
            existing_by_image.setdefault(image_path, []).append(index)

    for image_path, incoming_indices in incoming_by_image.items():
        existing_indices = existing_by_image.get(image_path, [])
        pairs = []
        for old_index in existing_indices:
            old_box = _face_box(existing_metadatas[old_index].get("location"))
            for new_index in incoming_indices:
                if any(updated[new_index].get(field) for field in _PERSON_METADATA_FIELDS):
                    continue
                new_box = _face_box(updated[new_index].get("location"))
                overlap = _face_box_iou(old_box, new_box)
                if overlap >= minimum_iou:
                    pairs.append((overlap, old_index, new_index))

        old_scores: Dict[int, List[float]] = {}
        new_scores: Dict[int, List[float]] = {}
        for overlap, old_index, new_index in pairs:
            old_scores.setdefault(old_index, []).append(overlap)
            new_scores.setdefault(new_index, []).append(overlap)

        def is_ambiguous(scores: List[float]) -> bool:
            ranked = sorted(scores, reverse=True)
            return len(ranked) > 1 and ranked[0] - ranked[1] < 0.10

        ambiguous_old = {
            index for index, scores in old_scores.items() if is_ambiguous(scores)
        }
        ambiguous_new = {
            index for index, scores in new_scores.items() if is_ambiguous(scores)
        }
        used_old = set()
        used_new = set()
        for _overlap, old_index, new_index in sorted(pairs, reverse=True):
            if old_index in ambiguous_old or new_index in ambiguous_new:
                continue
            if old_index in used_old or new_index in used_new:
                continue
            old_metadata = existing_metadatas[old_index]
            for field in _PERSON_METADATA_FIELDS:
                if old_metadata.get(field) and not updated[new_index].get(field):
                    updated[new_index][field] = old_metadata[field]
            used_old.add(old_index)
            used_new.add(new_index)

    return updated

class FaceVectorStore:
    """
    Vector database for storing and searching face embeddings
    """
    
    def __init__(self, db_path: str = VECTOR_DB_PATH, collection_name: str = COLLECTION_NAME):
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Cache for expensive stats calculation
        self._stats_cache = None
        self._last_face_count = None
        self._cache_timestamp = None
        self._pipeline_status_cache = None
        self._embedding_pipeline_version = active_embedding_pipeline_version()
        self._embedding_pipeline_metadata = active_embedding_pipeline_metadata()
        
        # Check for cache reset marker
        self._check_cache_reset_marker()
        
        # Initialize ChromaDB client (telemetry already disabled globally)
        try:
            from chromadb.config import Settings
            settings = Settings(anonymized_telemetry=False)
            self.client = chromadb.PersistentClient(path=db_path, settings=settings)
        except (ImportError, AttributeError):
            # Fallback to basic client if Settings not available
            self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            existing_metadata = dict(self.collection.metadata or {})
            if existing_metadata.get("embedding_pipeline_version") == self._embedding_pipeline_version:
                existing_metadata.update(self._embedding_pipeline_metadata)
                self.collection.modify(metadata=existing_metadata)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                configuration={"hnsw": {"space": "l2"}},
                metadata={
                    "description": "Face embeddings for similarity search",
                    **self._embedding_pipeline_metadata,
                    "embedding_pipeline_version": self._embedding_pipeline_version,
                    "embedding_pipeline_complete": True,
                }
            )
            logger.info(f"Created new collection: {collection_name}")

    def refresh_embedding_pipeline_version(self) -> None:
        """Refresh the active pipeline id after runtime settings are saved."""
        version = active_embedding_pipeline_version()
        if version != self._embedding_pipeline_version:
            self._embedding_pipeline_version = version
            self._embedding_pipeline_metadata = active_embedding_pipeline_metadata()
            self._pipeline_status_cache = None

    def get_embedding_pipeline_status(self) -> Dict[str, Any]:
        """Describe whether the stored index matches the active embedding pipeline."""
        if (
            self._pipeline_status_cache is not None
            and self._pipeline_status_cache.get("current_version") == self._embedding_pipeline_version
        ):
            return dict(self._pipeline_status_cache)
        try:
            metadata = dict(self.collection.metadata or {})
            stored_version = metadata.get("embedding_pipeline_version")
            total_faces = int(self.collection.count())
            compatible = stored_version == self._embedding_pipeline_version
            complete = compatible and bool(metadata.get("embedding_pipeline_complete"))
            pending_faces = 0
            if compatible and not complete and total_faces:
                current_rows = self.collection.get(
                    where={"embedding_pipeline_version": self._embedding_pipeline_version},
                    include=[],
                )
                pending_faces = max(0, total_faces - len(current_rows.get("ids") or []))
                if pending_faces == 0:
                    self.mark_embedding_pipeline_complete()
                    complete = True
            status = {
                "total_faces": total_faces,
                "stored_version": stored_version,
                "current_version": self._embedding_pipeline_version,
                "compatible": compatible,
                "complete": complete,
                "reindex_required": total_faces > 0 and not compatible,
                "reindex_in_progress": compatible and not complete,
                "pending_faces": pending_faces,
            }
            self._pipeline_status_cache = dict(status)
            return status
        except Exception as exc:  # noqa: BLE001 - status must not prevent app startup
            logger.warning("Could not read embedding pipeline status: %s", exc)
            status = {
                "total_faces": 0,
                "stored_version": None,
                "current_version": self._embedding_pipeline_version,
                "compatible": False,
                "complete": False,
                "reindex_required": False,
                "reindex_in_progress": False,
                "error": str(exc),
            }
            self._pipeline_status_cache = dict(status)
            return status

    def _mark_pipeline_write(self) -> None:
        """Tag new rows and make legacy rows invisible to current searches."""
        metadata = dict(self.collection.metadata or {})
        old_version = metadata.get("embedding_pipeline_version")
        was_complete = (
            old_version == self._embedding_pipeline_version
            and bool(metadata.get("embedding_pipeline_complete"))
        )
        metadata.update({
            **self._embedding_pipeline_metadata,
            "embedding_pipeline_version": self._embedding_pipeline_version,
            "embedding_pipeline_complete": was_complete,
        })
        self.collection.modify(metadata=metadata)
        self._pipeline_status_cache = None

    def mark_embedding_pipeline_complete(self) -> None:
        """Mark the index complete after every available source image is reprocessed."""
        metadata = dict(self.collection.metadata or {})
        metadata.update({
            **self._embedding_pipeline_metadata,
            "embedding_pipeline_version": self._embedding_pipeline_version,
            "embedding_pipeline_complete": True,
        })
        self.collection.modify(metadata=metadata)
        self._pipeline_status_cache = None

    def _metadata_for_existing_ids(self, face_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Read metadata that should survive an embedding refresh for stable IDs."""
        if not face_ids:
            return {}
        try:
            rows = self.collection.get(ids=face_ids, include=['metadatas'])
            return {
                str(face_id): dict(metadata or {})
                for face_id, metadata in zip(
                    rows.get('ids') or [], rows.get('metadatas') or []
                )
            }
        except Exception as exc:  # noqa: BLE001 - a metadata read must not block indexing
            logger.warning("Could not preserve metadata for refreshed faces: %s", exc)
            return {}

    def _metadata_for_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Read old face metadata for images being refreshed, for box matching."""
        paths = sorted({str(path) for path in image_paths if path})
        if not paths:
            return []
        try:
            rows = self.collection.get(
                where={"image_path": {"$in": paths}}, include=["metadatas"]
            )
            return [dict(metadata or {}) for metadata in rows.get("metadatas") or []]
        except Exception as exc:  # noqa: BLE001 - old labels are optional
            logger.warning("Could not load old face boxes for label matching: %s", exc)
            return []
    
    def add_face_embedding(
        self, 
        face_id: str, 
        embedding: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Add a face embedding to the vector store
        
        Args:
            face_id: Unique identifier for the face
            embedding: Face embedding vector
            metadata: Additional metadata (image_path, location, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Chroma uses squared L2 distance by default, so all stored and
            # queried vectors must be unit length for cosine scoring to hold.
            vector = normalise_embedding(embedding)
            if vector.size == 0 or not np.all(np.isfinite(vector)) or not np.any(vector):
                raise ValueError("Embedding must be a finite, non-zero vector")
            embedding_list = vector.tolist()
            
            existing_metadata = self._metadata_for_existing_ids([face_id])
            preserved_metadata = _preserve_face_metadata(
                existing_metadata.get(face_id, {}), metadata
            )
            full_metadata = {
                **preserved_metadata,
                **metadata,
                **self._embedding_pipeline_metadata,
                "added_at": datetime.now().isoformat(),
                "embedding_dimension": len(vector),
                "embedding_pipeline_version": self._embedding_pipeline_version,
            }
            
            # Add or replace the row; re-adding a face must not fail.
            self.collection.upsert(
                embeddings=[embedding_list],
                metadatas=[full_metadata],
                ids=[face_id]
            )
            self._mark_pipeline_write()
            self.invalidate_stats_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding face embedding {face_id}: {e}")
            return False
    
    def add_face_embeddings_batch(
        self, 
        face_ids: List[str], 
        embeddings: List[np.ndarray], 
        metadatas: List[Dict[str, Any]]
    ) -> int:
        """Write a batch of faces, replacing earlier rows for the same images.

        ``upsert`` makes a second run update the existing ids instead of
        failing. Rows that still point at the same image path but have a
        different id (written by the old stem-based scheme) are removed, so a
        reprocessed image never appears twice.

        Returns the number of written embeddings, or 0 on failure.
        """
        try:
            if len(face_ids) != len(embeddings) or len(face_ids) != len(metadatas):
                raise ValueError("Lists must have the same length")
            
            vectors = [normalise_embedding(embedding) for embedding in embeddings]
            if any(
                vector.size == 0 or not np.all(np.isfinite(vector)) or not np.any(vector)
                for vector in vectors
            ):
                raise ValueError("Embeddings must be finite, non-zero vectors")
            dimensions = {int(vector.size) for vector in vectors}
            if len(dimensions) != 1:
                raise ValueError("All embeddings in a batch must have the same dimension")
            embedding_lists = [vector.tolist() for vector in vectors]
            
            image_paths = {
                str(metadata.get("image_path"))
                for metadata in metadatas
                if metadata.get("image_path")
            }
            existing_metadata = self._metadata_for_existing_ids(face_ids)
            existing_image_metadata = self._metadata_for_images(list(image_paths))
            refreshed_metadatas = _inherit_matched_person_metadata(
                metadatas, existing_image_metadata
            )
            full_metadatas = []
            for face_id, metadata, vector in zip(face_ids, refreshed_metadatas, vectors):
                preserved_metadata = _preserve_face_metadata(
                    existing_metadata.get(str(face_id), {}), metadata
                )
                full_metadata = {
                    **preserved_metadata,
                    **metadata,
                    **self._embedding_pipeline_metadata,
                    "added_at": datetime.now().isoformat(),
                    "embedding_dimension": int(vector.size),
                    "embedding_pipeline_version": self._embedding_pipeline_version,
                }
                full_metadatas.append(full_metadata)
            
            # Add or update the new rows first. Only delete stale rows after a
            # successful write, so a failed reprocess keeps the old data.
            self.collection.upsert(
                embeddings=embedding_lists,
                metadatas=full_metadatas,
                ids=face_ids
            )
            self._mark_pipeline_write()

            self._delete_stale_faces_for_images(image_paths, set(face_ids))
            self.invalidate_stats_cache()
            
            logger.info(f"Wrote {len(face_ids)} face embeddings to vector store")
            return len(face_ids)
            
        except Exception as e:
            logger.error(f"Error adding batch embeddings: {e}")
            return 0

    def _delete_stale_faces_for_images(
        self, image_paths, keep_ids: set
    ) -> int:
        """Remove rows for ``image_paths`` that are not part of ``keep_ids``."""
        paths = sorted({str(path) for path in image_paths if path})
        if not paths:
            return 0

        try:
            # A batch can contain hundreds of source images. Querying Chroma
            # once per path makes indexing scale with the number of images in
            # Python/DB round trips. `$in` keeps replacement cleanup bounded.
            existing = self.collection.get(
                where={"image_path": {"$in": paths}},
                include=[],
            )
            stale = [
                face_id
                for face_id in (existing.get("ids") or [])
                if face_id not in keep_ids
            ]
        except Exception as exc:  # noqa: BLE001 - cleanup must not fail the write
            # Retain compatibility with Chroma versions that lack `$in` filters.
            logger.debug("Batch stale-face lookup failed; using path lookups: %s", exc)
            stale = []
            for image_path in paths:
                try:
                    existing = self.collection.get(
                        where={"image_path": image_path},
                        include=[],
                    )
                    stale.extend(
                        face_id
                        for face_id in (existing.get("ids") or [])
                        if face_id not in keep_ids
                    )
                except Exception as path_exc:  # noqa: BLE001 - cleanup must not fail the write
                    logger.warning("Could not clean old faces for %s: %s", image_path, path_exc)

        for start in range(0, len(stale), 500):
            self.collection.delete(ids=stale[start:start + 500])
        if stale:
            self._pipeline_status_cache = None
        return len(stale)

    def prune_missing_faces(self, root_directory: Optional[str | Path] = None) -> int:
        """Delete embeddings whose source image no longer exists.

        Returns the number of removed rows. This makes the database usable
        again after files were removed outside the app. When a root directory
        is supplied, rows outside it are left untouched.
        """
        try:
            total = self.collection.count()
            if total == 0:
                return 0

            root = (
                Path(root_directory).expanduser().resolve()
                if root_directory is not None
                else None
            )

            stale: List[str] = []
            offset = 0
            chunk_size = 2000
            while offset < total:
                chunk = self.collection.get(
                    limit=min(chunk_size, total - offset),
                    offset=offset,
                    include=['metadatas'],
                )
                ids = chunk.get("ids") or []
                for index, metadata in enumerate(chunk.get("metadatas") or []):
                    image_path = (metadata or {}).get("image_path")
                    if not image_path:
                        if root is not None:
                            # A directory-scoped job cannot identify ownership
                            # of rows that have no recorded source path.
                            continue
                        is_stale = True
                    else:
                        source_path = Path(str(image_path)).expanduser().resolve()
                        if root is not None:
                            try:
                                source_path.relative_to(root)
                            except ValueError:
                                continue
                        is_stale = not source_path.is_file()
                    if is_stale and index < len(ids):
                        stale.append(ids[index])
                offset += chunk_size

            for start in range(0, len(stale), 500):
                self.collection.delete(ids=stale[start:start + 500])

            self.invalidate_stats_cache()
            self._pipeline_status_cache = None
            logger.info(f"Pruned {len(stale)} face(s) with missing image files")
            return len(stale)
        except Exception as e:
            logger.error(f"Error pruning missing faces: {e}")
            return 0
    
    def search_similar_faces(
        self, 
        query_embedding: np.ndarray, 
        n_results: int = 10,
        min_similarity: float = SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Find faces similar to a query embedding.

        Embeddings are stored unit-normalised, so ChromaDB's L2 distance maps
        directly onto cosine similarity. Only the nearest candidates are
        requested and no embeddings are transferred back, which keeps a search
        in the low milliseconds instead of several seconds.
        """
        try:
            query_list = normalise_embedding(query_embedding)
            if not np.any(query_list):
                logger.warning("Query embedding has zero norm")
                return []

            try:
                collection_size = self.collection.count()
            except Exception:
                collection_size = 0
            if collection_size == 0:
                return []

            status = self.get_embedding_pipeline_status()
            if not status["compatible"]:
                logger.warning(
                    "Search blocked: index pipeline %r does not match active pipeline %r",
                    status.get("stored_version"),
                    self._embedding_pipeline_version,
                )
                return []

            collection_metadata = dict(self.collection.metadata or {})
            collection_configuration = getattr(self.collection, "configuration", {}) or {}
            index_configuration = (
                collection_configuration.get("hnsw")
                or collection_configuration.get("spann")
                or {}
            )
            index_type = (
                "hnsw" if collection_configuration.get("hnsw")
                else "spann" if collection_configuration.get("spann")
                else None
            )
            distance_space = str(
                index_configuration.get("space")
                or collection_metadata.get("hnsw:space")
                or "l2"
            ).lower()
            if distance_space not in {"l2", "cosine", "ip"}:
                logger.error("Unsupported Chroma distance space %r; refusing to mis-score results", distance_space)
                return []

            # Trade a little recall for a lot of speed: the old code asked for
            # up to 20,000 neighbours including their embeddings.
            search_n = int(min(max(n_results * 3, 50), 500, collection_size))

            # HNSW's default ef_search is 100. Requests for larger candidate
            # sets can otherwise return fewer useful neighbors than requested,
            # lowering recall on large indexes. Increase the per-collection
            # search breadth only when needed, and never lower an existing value.
            target_search_ef = min(search_n, collection_size)
            current_search_ef = int(index_configuration.get("ef_search", 100))
            if index_type and target_search_ef > current_search_ef:
                updates = {"ef_search": target_search_ef}
                if index_type == "spann":
                    current_probe = int(index_configuration.get("search_nprobe", 1))
                    if target_search_ef > current_probe:
                        updates["search_nprobe"] = target_search_ef
                try:
                    self.collection.modify(configuration={index_type: updates})
                except Exception as exc:  # noqa: BLE001 - still run search with persisted settings
                    logger.warning(
                        "Could not raise Chroma %s search breadth to %d: %s",
                        index_type,
                        target_search_ef,
                        exc,
                    )

            results = self.collection.query(
                query_embeddings=[query_list.tolist()],
                n_results=search_n,
                include=['metadatas', 'distances'],
                where={"embedding_pipeline_version": self._embedding_pipeline_version},
            )

            ids = results.get('ids') or [[]]
            distances = results.get('distances') or [[]]
            metadatas = results.get('metadatas') or [[]]

            similar_faces: List[Dict[str, Any]] = []
            for index, face_id in enumerate(ids[0]):
                try:
                    distance = float(distances[0][index])
                    cosine_similarity = cosine_from_chroma_distance(distance, distance_space)
                    similarity = similarity_from_chroma_distance(distance, distance_space)
                    if similarity < min_similarity:
                        continue

                    euclidean_distance = float(np.sqrt(max(0.0, 2.0 - 2.0 * cosine_similarity)))

                    similar_faces.append({
                        'face_id': face_id,
                        'similarity': similarity,
                        'similarity_percentage': similarity * 100,
                        'cosine_similarity': cosine_similarity,
                        'euclidean_distance': euclidean_distance,
                        'chroma_distance': distance,
                        'confidence_score': similarity,
                        'metadata': metadatas[0][index] or {},
                        'similarity_metrics': {
                            'cosine_similarity': cosine_similarity,
                            'euclidean_distance': euclidean_distance,
                            'chroma_distance': distance,
                            'primary_similarity': similarity,
                        },
                    })
                except Exception as e:
                    logger.debug(f"Skipping malformed search hit: {e}")
                    continue

            similar_faces.sort(key=lambda face: face['similarity'], reverse=True)
            top_score = similar_faces[0]['similarity_percentage'] if similar_faces else 0.0
            logger.debug(
                f"Found {len(similar_faces)} similar faces above {min_similarity:.2f} "
                f"(top: {top_score:.1f}%)"
            )
            return similar_faces[:n_results]

        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            return []
    def _calculate_enhanced_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Dict[str, float]:
        """
        Calculate similarity metrics for two embeddings.

        ``primary_similarity`` is the raw cosine similarity clipped to 0..1.
        Earlier versions pushed cosine through a piecewise curve that lifted
        unrelated faces to roughly 0.55, so a threshold of 0.4 matched almost
        everything. The additional metrics are kept because existing callers and
        tests read them, but they no longer influence ranking.
        """
        try:
            left = normalise_embedding(embedding1)
            right = normalise_embedding(embedding2)
            if not np.any(left) or not np.any(right):
                return self._get_zero_similarity_metrics()

            cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
            euclidean_distance = float(np.linalg.norm(left - right))
            euclidean_similarity = max(0.0, 1.0 - (euclidean_distance / 2.0))
            manhattan_distance = float(np.sum(np.abs(left - right)))
            manhattan_similarity = max(0.0, 1.0 - (manhattan_distance / (2.0 * len(left))))

            try:
                correlation = float(np.corrcoef(left, right)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0
            except Exception:
                correlation = 0.0

            angular_similarity = 1.0 - float(np.arccos(cosine)) / np.pi
            primary_similarity = max(0.0, cosine)

            return {
                'cosine_similarity': cosine,
                'euclidean_distance': euclidean_distance,
                'euclidean_similarity': euclidean_similarity,
                'dot_product': cosine,
                'correlation_similarity': correlation,
                'angular_similarity': angular_similarity,
                'manhattan_similarity': manhattan_similarity,
                'primary_similarity': primary_similarity,
                'ensemble_score': primary_similarity,
                'normalized_embedding1': left,
                'normalized_embedding2': right,
            }

        except Exception as e:
            logger.error(f"Error calculating enhanced similarity: {e}")
            return self._get_zero_similarity_metrics()
    def _get_zero_similarity_metrics(self) -> Dict[str, float]:
        """Return zero similarity metrics for error cases"""
        return {
            'cosine_similarity': 0.0,
            'euclidean_distance': 2.0,
            'euclidean_similarity': 0.0,
            'dot_product': 0.0,
            'correlation_similarity': 0.0,
            'angular_similarity': 0.0,
            'manhattan_similarity': 0.0,
            'primary_similarity': 0.0,
            'ensemble_score': 0.0
        }
    
    def _calculate_confidence_score(self, similarity_metrics: Dict[str, float]) -> float:
        """
        Return the confidence for a match.

        With honest cosine scoring the similarity *is* the confidence. The
        method remains for API compatibility with existing callers and tests.
        """
        try:
            similarity = float(similarity_metrics.get('primary_similarity', 0.0))
            return float(max(0.0, min(1.0, similarity)))
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding (should be normalized)
            embedding2: Second embedding
            
        Returns:
            Cosine similarity (-1 to 1, where 1 is most similar)
        """
        try:
            # Normalize both embeddings to unit length
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            embedding1_normalized = embedding1 / norm1
            embedding2_normalized = embedding2 / norm2
            
            # Calculate cosine similarity (dot product of normalized vectors)
            similarity = np.dot(embedding1_normalized, embedding2_normalized)
            
            # Ensure result is in [-1, 1] range
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific face by ID
        
        Args:
            face_id: Face identifier
            
        Returns:
            Face data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[face_id],
                include=['embeddings', 'metadatas']
            )
            
            if results['ids'] and len(results['ids']) > 0:
                return {
                    'face_id': results['ids'][0],
                    'embedding': np.array(results['embeddings'][0]),
                    'metadata': results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting face {face_id}: {e}")
            return None
    
    def update_face_metadata(self, face_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a face
        
        Args:
            face_id: Face identifier
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing face
            existing_face = self.get_face_by_id(face_id)
            if not existing_face:
                return False
            
            # Update metadata
            updated_metadata = {
                **existing_face['metadata'],
                **metadata,
                "updated_at": datetime.now().isoformat()
            }
            
            # Update in collection
            self.collection.update(
                ids=[face_id],
                metadatas=[updated_metadata]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating face metadata {face_id}: {e}")
            return False
    
    def delete_face(self, face_id: str) -> bool:
        """
        Delete a face from the vector store
        
        Args:
            face_id: Face identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[face_id])
            self.invalidate_stats_cache()
            self._pipeline_status_cache = None
            logger.info(f"Deleted face {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting face {face_id}: {e}")
            return False
    
    def delete_faces_by_image_path(self, image_path: str) -> int:
        """Delete every face whose metadata points at ``image_path``.

        Returns the number of removed embeddings. The statistics cache is
        invalidated so counters do not keep the old values.
        """
        try:
            existing = self.collection.get(
                where={"image_path": str(image_path)},
                include=[],
            )
            ids = list(existing.get("ids") or [])
            if ids:
                self.collection.delete(ids=ids)
                self._pipeline_status_cache = None
                logger.info(f"Deleted {len(ids)} face(s) for image {image_path}")
            self.invalidate_stats_cache()
            return len(ids)
        except Exception as e:
            logger.error(f"Error deleting faces for {image_path}: {e}")
            return 0

    def invalidate_stats_cache(self) -> None:
        """Drop the in-memory and on-disk statistics cache."""
        self._stats_cache = None
        self._last_face_count = None
        self._cache_timestamp = None
        try:
            self._stats_cache_file().unlink(missing_ok=True)
        except OSError as exc:
            logger.debug(f"Could not remove stats cache: {exc}")

    def _stats_cache_file(self) -> Path:
        """Location of the on-disk statistics cache (next to the vector DB)."""
        return Path(self.db_path).parent / ".stats_cache.json"

    def _load_stats_disk_cache(self) -> Optional[Dict[str, Any]]:
        try:
            cache_file = self._stats_cache_file()
            if not cache_file.exists():
                return None
            with open(cache_file, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as e:
            logger.debug(f"Could not read stats cache: {e}")
            return None

    def _save_stats_disk_cache(self, stats: Dict[str, Any]) -> None:
        try:
            cache_file = self._stats_cache_file()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as handle:
                json.dump({
                    "total_faces": stats.get("total_faces", 0),
                    "unique_images": stats.get("unique_images", 0),
                    "avg_embedding_dimension": stats.get("avg_embedding_dimension", 0),
                    "cached_at": datetime.now().isoformat(),
                }, handle)
        except Exception as e:
            logger.debug(f"Could not write stats cache: {e}")

    def get_collection_stats(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get statistics about the collection with intelligent caching.

        Counting unique images means walking the metadata of every face, so the
        result is cached in memory and on disk (``data/.stats_cache.json``).
        Restarting the app therefore no longer re-scans the whole collection,
        which used to stall the first page load on large databases.
        """
        try:
            collection_count = self.collection.count()

            cache_valid = (
                not force_refresh
                and self._stats_cache is not None
                and self._last_face_count == collection_count
                and self._cache_timestamp is not None
                and (datetime.now() - self._cache_timestamp).total_seconds() < 600
            )
            if cache_valid:
                return self._stats_cache

            stats: Dict[str, Any] = {
                "total_faces": collection_count,
                "collection_name": self.collection_name,
                "db_path": self.db_path,
            }

            if collection_count == 0:
                stats.update({"unique_images": 0, "avg_embedding_dimension": 0})
            else:
                cached = None if force_refresh else self._load_stats_disk_cache()
                if cached and cached.get("total_faces") == collection_count:
                    stats.update({
                        "unique_images": int(cached.get("unique_images", 0)),
                        "avg_embedding_dimension": float(cached.get("avg_embedding_dimension", 0) or 0),
                    })
                    logger.info(f"Using stored stats for {collection_count} faces")
                else:
                    unique_images = set()
                    dimension_sum = 0
                    dimension_count = 0
                    chunk_size = 2000
                    offset = 0

                    while offset < collection_count:
                        chunk = self.collection.get(
                            limit=min(chunk_size, collection_count - offset),
                            offset=offset,
                            include=['metadatas'],
                        )
                        for metadata in chunk.get('metadatas') or []:
                            if not metadata:
                                continue
                            image_path = metadata.get('image_path')
                            if image_path:
                                unique_images.add(image_path)
                            dimension = metadata.get('embedding_dimension')
                            if dimension:
                                dimension_sum += int(dimension)
                                dimension_count += 1
                        offset += chunk_size

                    stats.update({
                        "unique_images": len(unique_images),
                        "avg_embedding_dimension": (dimension_sum / dimension_count) if dimension_count else 0,
                    })
                    self._save_stats_disk_cache(stats)
                    logger.info(f"Stats calculated: {collection_count} faces, {len(unique_images)} unique images")

            self._stats_cache = stats
            self._last_face_count = collection_count
            self._cache_timestamp = datetime.now()
            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def _check_cache_reset_marker(self):
        """Check if cache should be reset due to cleanup operations"""
        try:
            cache_reset_file = Path(self.db_path).parent / ".cache_reset_marker"
            if cache_reset_file.exists():
                logger.info("Cache reset marker found - clearing stats cache")
                self._stats_cache = None
                self._last_face_count = None 
                self._cache_timestamp = None
                
                # Remove the marker file
                cache_reset_file.unlink()
                logger.info("Stats cache cleared and reset marker removed")
        except Exception as e:
            logger.debug(f"Cache reset marker check failed: {e}")
    
    def clear_collection(self) -> bool:
        """
        Clear all data from the collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection
            self.client.delete_collection(self.collection_name)
            self._pipeline_status_cache = None
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                configuration={"hnsw": {"space": "l2"}},
                metadata={
                    "description": "Face embeddings for similarity search",
                    "embedding_pipeline_version": self._embedding_pipeline_version,
                    "embedding_pipeline_complete": True,
                }
            )
            self.invalidate_stats_cache()
            
            logger.info(f"Cleared collection {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def get_all_faces_metadata(self) -> List[Dict[str, Any]]:
        """Return every face as ``{"face_id", "metadata"}`` with one scan.

        The gallery used to paginate in the database, which made filtering and
        sorting impossible: it can only sort and slice what it has loaded. A
        single chunked read is then cached by the UI, so a page change does not
        touch the database again.
        """
        faces: List[Dict[str, Any]] = []
        try:
            total = self.collection.count()
            offset = 0
            chunk_size = 2000
            while offset < total:
                chunk = self.collection.get(
                    limit=min(chunk_size, total - offset),
                    offset=offset,
                    include=['metadatas'],
                )
                ids = chunk.get("ids") or []
                for index, metadata in enumerate(chunk.get("metadatas") or []):
                    if not metadata:
                        continue
                    face_id = metadata.get("face_id") or (ids[index] if index < len(ids) else "")
                    if not face_id:
                        continue
                    faces.append({"face_id": str(face_id), "metadata": dict(metadata)})
                offset += chunk_size
            logger.info(f"Loaded metadata for {len(faces)} faces")
            return faces
        except Exception as e:
            logger.error(f"Error loading all face metadata: {e}")
            return faces

    def get_face_gallery_page(
        self,
        search_term: str = "",
        min_face_size: int = 0,
        sort_by: str = "Newest",
        page: int = 1,
        page_size: int = 50,
        chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """Stream face metadata to a temporary SQLite index and return one page.

        The gallery needs substring search and global sorting, which Chroma's
        metadata filters do not provide. Scanning metadata in chunks and
        sorting it in a temporary disk database keeps the full collection out
        of Streamlit session memory. No embeddings or image bytes are copied.
        """
        page_size = max(1, min(int(page_size), 500))
        chunk_size = max(1, min(int(chunk_size), 5000))
        requested_page = max(1, int(page))
        search = str(search_term or "").casefold().strip()
        sort_key = sort_by if sort_by in {"Newest", "Image name", "Face ID"} else "Newest"
        sort_direction = "DESC" if sort_key == "Newest" else "ASC"

        try:
            with tempfile.TemporaryDirectory(prefix="face-gallery-") as temp_dir:
                database_path = Path(temp_dir) / "gallery.sqlite3"
                with sqlite3.connect(database_path) as connection:
                    connection.execute(
                        "CREATE TABLE matches ("
                        "face_id TEXT NOT NULL, sort_value TEXT NOT NULL, metadata TEXT NOT NULL)"
                    )
                    total = int(self.collection.count())
                    offset = 0
                    while offset < total:
                        chunk = self.collection.get(
                            limit=min(chunk_size, total - offset),
                            offset=offset,
                            include=["metadatas"],
                        )
                        ids = chunk.get("ids") or []
                        metadatas = chunk.get("metadatas") or []
                        rows = []
                        for index, raw_metadata in enumerate(metadatas):
                            metadata = dict(raw_metadata or {})
                            face_id = str(
                                metadata.get("face_id")
                                or (ids[index] if index < len(ids) else "")
                            )
                            if not face_id:
                                continue

                            image_path = str(metadata.get("image_path") or "")
                            if search and search not in " ".join((
                                face_id,
                                image_path,
                                Path(image_path).name,
                                str(metadata.get("full_name") or ""),
                            )).casefold():
                                continue

                            if min_face_size:
                                box = _face_box(metadata.get("location"))
                                if box is None:
                                    continue
                                top, right, bottom, left = box
                                if min(right - left, bottom - top) < min_face_size:
                                    continue

                            if sort_key == "Newest":
                                sort_value = str(metadata.get("added_at") or "")
                            elif sort_key == "Image name":
                                sort_value = Path(image_path).name.casefold()
                            else:
                                sort_value = face_id.casefold()
                            rows.append((face_id, sort_value, json.dumps(metadata)))

                        if rows:
                            connection.executemany(
                                "INSERT INTO matches(face_id, sort_value, metadata) VALUES (?, ?, ?)",
                                rows,
                            )
                        offset += max(1, len(ids))

                    connection.execute(
                        "CREATE INDEX matches_sort_idx ON matches(sort_value, face_id)"
                    )
                    filtered_total = int(
                        connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
                    )
                    total_pages = max(1, (filtered_total + page_size - 1) // page_size)
                    current_page = min(requested_page, total_pages)
                    page_offset = (current_page - 1) * page_size
                    result_rows = connection.execute(
                        "SELECT face_id, metadata FROM matches "
                        f"ORDER BY sort_value {sort_direction}, face_id ASC LIMIT ? OFFSET ?",
                        (page_size, page_offset),
                    ).fetchall()

                return {
                    "faces": [
                        {"face_id": face_id, "metadata": json.loads(metadata)}
                        for face_id, metadata in result_rows
                    ],
                    "total": filtered_total,
                    "total_faces": total,
                    "total_pages": total_pages,
                    "page": current_page,
                }
        except Exception as exc:  # noqa: BLE001 - keep the gallery usable on read failures
            logger.exception("Could not load a face gallery page")
            return {"faces": [], "total": 0, "total_faces": 0, "total_pages": 1, "page": 1, "error": str(exc)}

    def get_faces_paginated(self, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
        """
        Robust paginated retrieval of faces with error handling
        
        Args:
            offset: Number of faces to skip
            limit: Maximum number of faces to return
            
        Returns:
            Dictionary with faces data or empty result on error
        """
        try:
            # First try normal pagination
            try:
                results = self.collection.get(
                    limit=limit,
                    offset=offset,
                    include=['metadatas']
                )
                
                if results and results.get('metadatas'):
                    return results
                    
            except Exception as direct_error:
                logger.warning(f"Direct pagination failed: {direct_error}")
                
                # Fallback approach: get a larger chunk and slice
                try:
                    chunk_size = min(offset + limit + 100, 2000)  # Get reasonable chunk
                    all_results = self.collection.get(
                        limit=chunk_size,
                        include=['metadatas']
                    )
                    
                    if all_results and all_results.get('metadatas'):
                        # Manually slice the results
                        metadatas = all_results.get('metadatas', [])
                        embeddings = all_results.get('embeddings', [])
                        documents = all_results.get('documents', [])
                        ids = all_results.get('ids', [])
                        
                        if len(metadatas) > offset:
                            end_idx = min(offset + limit, len(metadatas))
                            
                            return {
                                'ids': ids[offset:end_idx] if ids else [],
                                'metadatas': metadatas[offset:end_idx],
                                'embeddings': embeddings[offset:end_idx] if embeddings else [],
                                'documents': documents[offset:end_idx] if documents else []
                            }
                            
                except Exception as fallback_error:
                    logger.error(f"Fallback pagination failed: {fallback_error}")
            
            # Return empty result if all methods fail
            return {
                'ids': [],
                'metadatas': [],
                'embeddings': [],
                'documents': []
            }
            
        except Exception as e:
            logger.error(f"Error in get_faces_paginated: {e}")
            return {
                'ids': [],
                'metadatas': [],
                'embeddings': [],
                'documents': []
            }
    
    # Person Name Assignment Functions
    
    def assign_person_name(
        self,
        face_id: str,
        person_data: Dict[str, str],
        person_id: Optional[str] = None,
        *,
        allow_reassignment: bool = False,
        assignment_source: str = "manual_review",
        suggested_similarity: Optional[float] = None,
        actor: str = "local-user",
    ) -> Optional[str]:
        """
        Assign extended person data to a face
        
        Args:
            face_id: Face identifier
            person_data: Dictionary with person information:
                - first_name: First name (required)
                - middle_names: Middle names (optional)
                - last_name: Last name (required)
                - birth_date: Birth date (optional)
                - birth_place: Birth place (optional)
                - notes: Additional notes (optional)
            person_id: Unique person ID (if None, creates new one)
            
        Returns:
            person_id if successful, None if failed
        """
        try:
            # Generate person_id if not provided
            if not person_id:
                person_id = str(uuid.uuid4())
            
            # Extract and validate person data
            first_name = person_data.get('first_name', '').strip()
            middle_names = person_data.get('middle_names', '').strip()
            last_name = person_data.get('last_name', '').strip()
            birth_date = person_data.get('birth_date', '').strip()
            birth_place = person_data.get('birth_place', '').strip()
            notes = person_data.get('notes', '').strip()
            
            # Build full name
            name_parts = [first_name]
            if middle_names:
                name_parts.append(middle_names)
            if last_name:
                name_parts.append(last_name)
            full_name = ' '.join(name_parts)
            
            # Keep a per-face, append-only record of identity edits. Chroma
            # metadata accepts strings but not nested lists, so serialize the
            # event list as JSON. This records the previous value as well as
            # the user-reviewed replacement and supports later manual undo.
            existing_face = self.get_face_by_id(face_id)
            if not existing_face:
                return None
            existing_metadata = existing_face.get("metadata") or {}
            current_person_id = str(existing_metadata.get("person_id", "") or "")
            if current_person_id and current_person_id != person_id and not allow_reassignment:
                logger.warning(
                    "Refused person reassignment for %s without explicit confirmation",
                    face_id,
                )
                return None
            try:
                history = json.loads(existing_metadata.get("assignment_history", "[]"))
            except (TypeError, ValueError):
                history = []
            if not isinstance(history, list):
                history = []
            now = datetime.now().isoformat()
            tracked_person_fields = (
                "first_name", "middle_names", "last_name", "full_name",
                "birth_date", "birth_place", "notes",
            )
            from_fields = {
                key: str(existing_metadata.get(key, "") or "")
                for key in tracked_person_fields
            }
            to_fields = {
                "first_name": first_name,
                "middle_names": middle_names,
                "last_name": last_name,
                "full_name": full_name,
                "birth_date": birth_date,
                "birth_place": birth_place,
                "notes": notes,
            }
            # Prevent accidentally creating a second person record when the
            # same identifying name and birth details are entered again. A
            # shared name by itself is not sufficient: different people may
            # legitimately have the same name.
            if not self.get_faces_by_person_id(str(person_id)):
                duplicate_rows = self.collection.get(
                    where={"full_name": full_name},
                    include=["metadatas"],
                )
                has_identifying_details = bool(birth_date or birth_place)
                for duplicate_metadata in duplicate_rows.get("metadatas") or []:
                    duplicate_metadata = duplicate_metadata or {}
                    if (
                        has_identifying_details
                        and str(duplicate_metadata.get("birth_date", "") or "") == birth_date
                        and str(duplicate_metadata.get("birth_place", "") or "") == birth_place
                    ):
                        logger.warning(
                            "Refused duplicate person identity for face %s", face_id
                        )
                        return None
            # A person ID is a shared identity key. Reject edits that would
            # silently split one person's name or details across face rows.
            for person_face in self.get_faces_by_person_id(str(person_id)):
                other_metadata = person_face.get("metadata") or {}
                if any(
                    str(other_metadata.get(key, "") or "") != value
                    for key, value in to_fields.items()
                ):
                    logger.warning(
                        "Refused inconsistent details for person %s on face %s",
                        person_id,
                        face_id,
                    )
                    return None

            if suggested_similarity is not None:
                suggested_similarity = float(suggested_similarity)
                if not np.isfinite(suggested_similarity):
                    suggested_similarity = None
            history.append({
                "action": "assigned",
                "actor": str(actor or "local-user")[:254],
                "at": now,
                "source": str(assignment_source),
                "suggested_similarity": suggested_similarity,
                "from_person_id": existing_metadata.get("person_id", ""),
                "from_name": existing_metadata.get("full_name", ""),
                "from_fields": from_fields,
                "to_person_id": person_id,
                "to_name": full_name,
                "to_fields": to_fields,
                "reviewed": True,
            })

            # Update face metadata with extended person info
            person_metadata = {
                'person_id': person_id,
                'first_name': first_name,
                'middle_names': middle_names,
                'last_name': last_name,
                'full_name': full_name,
                'birth_date': birth_date,
                'birth_place': birth_place,
                'notes': notes,
                'name_assigned_at': now,
                'assignment_history': json.dumps(history, ensure_ascii=False),
            }
            
            success = self.update_face_metadata(face_id, person_metadata)
            return person_id if success else None
            
        except Exception as e:
            logger.error(f"Error assigning person data to {face_id}: {e}")
            return None

    def assign_person_name_with_auto_matching(
        self,
        face_id: str,
        person_data: Dict[str, str],
        person_id: Optional[str] = None,
        similarity_threshold: float = FACE_AUTO_ASSIGN_THRESHOLD,
        ambiguity_margin: float = FACE_AUTO_ASSIGN_AMBIGUITY_MARGIN,
    ) -> Dict[str, Any]:
        """Assign one reviewed face and return similar faces as suggestions.

        The legacy method name is retained for callers, but similarity alone
        never propagates a person's name. Every other face must be reviewed and
        assigned separately by an explicit user action.
        """
        try:
            assigned_person_id = self.assign_person_name(face_id, person_data, person_id)
            if not assigned_person_id:
                return {
                    "success": False,
                    "error": "Failed to assign name to the selected face",
                    "total_assigned": 0,
                    "similar_faces": [],
                    "skipped_ambiguous": 0,
                }

            original = self.get_face_by_id(face_id)
            if not original or original.get("embedding") is None:
                return {
                    "success": True,
                    "person_id": assigned_person_id,
                    "total_assigned": 1,
                    "similar_faces": [],
                    "skipped_ambiguous": 0,
                    "warning": "No embedding was available to suggest similar faces.",
                }

            matches = self.search_similar_faces(
                original["embedding"], n_results=100, min_similarity=similarity_threshold
            )
            other_person_centroids = self._person_centroids(exclude_person_id=assigned_person_id)
            suggestions = []
            skipped_ambiguous = 0
            for match in matches:
                candidate_id = str(match.get("id", match.get("face_id", "")))
                if not candidate_id or candidate_id == face_id:
                    continue
                metadata = match.get("metadata") or {}
                if metadata.get("person_id") and metadata.get("full_name"):
                    continue
                candidate_embedding = None
                try:
                    row = self.collection.get(ids=[candidate_id], include=["embeddings"])
                    vectors = row.get("embeddings") or []
                    if vectors:
                        candidate_embedding = normalise_embedding(vectors[0])
                except Exception as exc:  # noqa: BLE001 - suggestions remain available
                    logger.debug("Could not inspect suggestion %s: %s", candidate_id, exc)
                ambiguous = False
                if candidate_embedding is not None and other_person_centroids:
                    other_best = max(
                        float(np.dot(candidate_embedding, centroid))
                        for centroid in other_person_centroids.values()
                    )
                    ambiguous = other_best >= float(match.get("similarity", 0)) - ambiguity_margin
                if ambiguous:
                    skipped_ambiguous += 1
                suggestions.append({
                    "face_id": candidate_id,
                    "similarity": float(match.get("similarity", 0)),
                    "ambiguous": ambiguous,
                    "metadata": metadata,
                })

            return {
                "success": True,
                "person_id": assigned_person_id,
                "total_assigned": 1,
                "similar_faces": suggestions,
                "suggested_faces": len(suggestions),
                "skipped_ambiguous": skipped_ambiguous,
                "similarity_threshold": similarity_threshold,
                "requires_review": bool(suggestions),
            }
        except Exception as exc:  # noqa: BLE001 - keep storage API errors structured
            logger.exception("Error generating person assignment suggestions")
            return {
                "success": False,
                "error": str(exc),
                "total_assigned": 0,
                "similar_faces": [],
                "skipped_ambiguous": 0,
            }

    def remove_person_name(self, face_id: str, *, actor: str = "local-user") -> bool:
        """
        Remove person name assignment from a face
        
        Args:
            face_id: Face identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            existing_face = self.get_face_by_id(face_id)
            if not existing_face:
                return False
            metadata = existing_face.get("metadata") or {}
            try:
                history = json.loads(metadata.get("assignment_history", "[]"))
            except (TypeError, ValueError):
                history = []
            if not isinstance(history, list):
                history = []
            history.append({
                "action": "removed",
                "actor": str(actor or "local-user")[:254],
                "at": datetime.now().isoformat(),
                "from_person_id": metadata.get("person_id", ""),
                "from_name": metadata.get("full_name", ""),
                "from_fields": {
                    key: str(metadata.get(key, "") or "")
                    for key in (
                        "first_name", "middle_names", "last_name", "full_name",
                        "birth_date", "birth_place", "notes",
                    )
                },
                "to_person_id": "",
                "to_name": "",
                "to_fields": {},
                "reviewed": True,
            })
            # Remove all person-related metadata (use empty strings instead of None for ChromaDB compatibility)
            removal_metadata = {
                'person_id': '',
                'first_name': '',
                'middle_names': '',
                'last_name': '',
                'full_name': '',
                'birth_date': '',
                'birth_place': '',
                'notes': '',
                'name_assigned_at': '',
                'name_removed_at': datetime.now().isoformat(),
                'assignment_history': json.dumps(history, ensure_ascii=False),
            }
            
            return self.update_face_metadata(face_id, removal_metadata)
            
        except Exception as e:
            logger.error(f"Error removing person name from {face_id}: {e}")
            return False

    def undo_person_assignments(
        self,
        expected_events: Dict[str, str],
        *,
        actor: str = "local-user",
    ) -> Dict[str, Any]:
        """Undo only the exact reviewed assignment events supplied by the UI.

        Event timestamps prevent an old group-level undo action from reverting
        a later edit to the same face. Conflicts are skipped and reported.
        """
        result: Dict[str, Any] = {
            "undone": [],
            "skipped": [],
            "errors": [],
            "remaining": {},
        }
        for face_id, expected_at in expected_events.items():
            face = self.get_face_by_id(face_id)
            if not face:
                result["skipped"].append(face_id)
                continue
            metadata = face.get("metadata") or {}
            try:
                history = json.loads(metadata.get("assignment_history", "[]"))
            except (TypeError, ValueError):
                history = []
            if not isinstance(history, list) or not history:
                result["skipped"].append(face_id)
                continue
            latest = history[-1]
            if (
                latest.get("action") != "assigned"
                or str(latest.get("at", "")) != str(expected_at)
                or str(metadata.get("person_id", "")) != str(latest.get("to_person_id", ""))
            ):
                result["skipped"].append(face_id)
                # Preserve the originally requested event. Never promote a
                # newer assignment to the target of an older undo action.
                result["remaining"][face_id] = str(expected_at)
                continue

            from_fields = latest.get("from_fields")
            if not isinstance(from_fields, dict):
                result["skipped"].append(face_id)
                result["remaining"][face_id] = str(latest.get("at", ""))
                continue

            now = datetime.now().isoformat()
            restored_person_id = str(latest.get("from_person_id", "") or "")
            restored_name = str(latest.get("from_name", "") or "")
            history.append({
                "action": "undo",
                "actor": str(actor or "local-user")[:254],
                "at": now,
                "undo_of": str(latest.get("at", "")),
                "from_person_id": str(metadata.get("person_id", "") or ""),
                "from_name": str(metadata.get("full_name", "") or ""),
                "from_fields": {
                    key: str(metadata.get(key, "") or "")
                    for key in (
                        "first_name", "middle_names", "last_name", "full_name",
                        "birth_date", "birth_place", "notes",
                    )
                },
                "to_person_id": restored_person_id,
                "to_name": restored_name,
                "to_fields": from_fields,
                "reviewed": True,
            })
            update = {
                key: str(from_fields.get(key, "") or "")
                for key in (
                    "first_name", "middle_names", "last_name", "full_name",
                    "birth_date", "birth_place", "notes",
                )
            }
            update.update({
                "person_id": restored_person_id,
                "name_assigned_at": now if restored_person_id else "",
                "name_removed_at": "" if restored_person_id else now,
                "assignment_history": json.dumps(history, ensure_ascii=False),
            })
            try:
                if self.update_face_metadata(face_id, update):
                    result["undone"].append(face_id)
                else:
                    result["errors"].append(face_id)
                    result["remaining"][face_id] = str(latest.get("at", ""))
            except Exception as exc:  # noqa: BLE001 - keep processing other faces
                logger.error("Could not undo person assignment for %s: %s", face_id, exc)
                result["errors"].append(face_id)
                result["remaining"][face_id] = str(latest.get("at", ""))
        return result
    
    def get_faces_by_person_id(self, person_id: str) -> List[Dict[str, Any]]:
        """
        Get all faces assigned to a specific person
        
        Args:
            person_id: Person identifier
            
        Returns:
            List of face dictionaries
        """
        try:
            logger.info(f"Getting faces for person {person_id}")
            
            # Use ChromaDB's where clause to filter by person_id directly
            # This is much more efficient than manual iteration
            target_person_id = str(person_id).strip()
            
            results = self.collection.get(
                where={"person_id": target_person_id},
                include=['metadatas']
            )
            
            faces = []
            if results and results.get('metadatas'):
                for i, metadata in enumerate(results['metadatas']):
                    try:
                        # Safely get embedding - check if embeddings exist and has correct index
                        embedding = None
                        embeddings_data = results.get('embeddings')
                        if embeddings_data is not None and isinstance(embeddings_data, (list, np.ndarray)) and i < len(embeddings_data):
                            embedding = embeddings_data[i]
                        
                        face_data = {
                            'face_id': results['ids'][i],
                            'metadata': metadata,
                            'embedding': embedding
                        }
                        faces.append(face_data)
                    except Exception as e:
                        logger.warning(f"Skipping face at index {i}: {e}")
                        continue
                        
            logger.info(f"Found {len(faces)} faces for person {person_id}")
            return faces
            
        except Exception as e:
            logger.error(f"Error getting faces for person {person_id}: {e}")
            return []
    
    def get_all_persons(self) -> List[Dict[str, Any]]:
        """
        Get all unique persons in the database with enhanced debugging
        
        Returns:
            List of person dictionaries with names and face counts
        """
        try:
            logger.info("Getting all persons from database")
            
            # Get all faces with person_id metadata
            results = self.collection.get(
                include=['metadatas']
            )
            
            persons = {}
            person_ids_found = []  # Debug list
            
            if results and results.get('ids') and results.get('metadatas'):
                total_faces = len(results['metadatas'])
                logger.info(f"Processing {total_faces} faces for person extraction")
                
                for i, metadata in enumerate(results['metadatas']):
                    try:
                        # Debug: Show sample metadata for first few faces
                        if i < 3:
                            logger.debug(f"Face {i} metadata keys: {list(metadata.keys())}")
                            if 'person_id' in metadata:
                                logger.debug(f"Face {i} person_id raw: {metadata['person_id']} (type: {type(metadata['person_id'])})")
                        
                        # Check if this face has a person assigned
                        if 'person_id' not in metadata:
                            continue
                            
                        raw_person_id = metadata['person_id']
                        
                        # Skip None values
                        if raw_person_id is None:
                            continue
                        
                        # Convert to string and clean
                        person_id = str(raw_person_id).strip()
                        
                        # Handle array-like strings (from ChromaDB serialization issues)
                        if person_id.startswith('[') and person_id.endswith(']'):
                            inner_content = person_id[1:-1].strip()
                            if inner_content:
                                # Remove quotes and take first value if comma separated
                                if ',' in inner_content:
                                    person_id = inner_content.split(',')[0].strip().strip("'\"")
                                else:
                                    person_id = inner_content.strip("'\"")
                        
                        # Skip invalid values
                        if not person_id or person_id in ["None", "null", "", "nan"]:
                            continue
                        
                        # Track found person IDs for debugging
                        if person_id not in person_ids_found:
                            person_ids_found.append(person_id)
                            logger.debug(f"Found person ID: {person_id}")
                        
                        # Add or update person
                        if person_id not in persons:
                            # Build full name from components
                            first_name = metadata.get('first_name', '')
                            middle_names = metadata.get('middle_names', '')
                            last_name = metadata.get('last_name', '')
                            
                            # Create full name
                            name_parts = [p.strip() for p in [first_name, middle_names, last_name] if p.strip()]
                            full_name = ' '.join(name_parts) if name_parts else 'Unknown person'
                            
                            persons[person_id] = {
                                'person_id': person_id,
                                'first_name': first_name,
                                'middle_names': middle_names,
                                'last_name': last_name,
                                'full_name': full_name,
                                'birth_date': metadata.get('birth_date', ''),
                                'birth_place': metadata.get('birth_place', ''),
                                'notes': metadata.get('notes', ''),
                                'face_count': 0,
                                'face_ids': []
                            }
                        
                        persons[person_id]['face_count'] += 1
                        persons[person_id]['face_ids'].append(results['ids'][i])
                    
                    except Exception as e:
                        logger.debug(f"Skipping person at index {i}: {e}")
                        continue
                
                # Debug output
                logger.info(f"Person IDs found: {person_ids_found}")
            
            persons_list = list(persons.values())
            logger.info(f"Found {len(persons_list)} unique persons")
            
            # Additional debug for first person
            if persons_list:
                first_person = persons_list[0]
                logger.debug(f"First person: {first_person['full_name']} with {first_person['face_count']} faces")
            
            return persons_list
            
        except Exception as e:
            logger.error(f"Error getting all persons: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    def _person_centroids(
        self, exclude_person_id: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Return one unit-length centroid per named person.

        Used by the auto-assignment ambiguity check and by the group page to
        suggest an existing person for a new identity cluster.
        """
        try:
            if not self.get_embedding_pipeline_status()["compatible"]:
                return {}
            rows = self.collection.get(
                where={"embedding_pipeline_version": self._embedding_pipeline_version},
                include=['embeddings', 'metadatas'],
            )
            embeddings = rows.get('embeddings')
            if embeddings is None:
                embeddings = []
            vectors: Dict[str, List[np.ndarray]] = {}
            for embedding, metadata in zip(
                embeddings, rows.get('metadatas') or []
            ):
                metadata = metadata or {}
                person_id = str(metadata.get('person_id') or '').strip()
                if not person_id or person_id == exclude_person_id:
                    continue
                vector = normalise_embedding(embedding)
                if np.any(vector):
                    vectors.setdefault(person_id, []).append(vector)
            return {
                person_id: centroid_of(person_vectors)
                for person_id, person_vectors in vectors.items()
            }
        except Exception as exc:  # noqa: BLE001 - grouping must stay optional
            logger.warning(f"Could not build person centroids: {exc}")
            return {}

    def get_face_groups(
        self,
        threshold: float | None = None,
        margin: float = FACE_GROUPING_MARGIN,
        min_size: int = 2,
    ) -> Dict[str, Any]:
        """Group unnamed faces into reviewed identity suggestions.

        Named faces are never changed here; they are only used to suggest a
        matching existing person for a group. The review page asks the user
        before assigning anything.
        """
        if threshold is None:
            threshold = active_grouping_threshold()
        empty: Dict[str, Any] = {
            'groups': [],
            'unnamed_faces': 0,
            'grouped_faces': 0,
            'singletons': 0,
            'threshold': threshold,
            'margin': margin,
        }
        try:
            status = self.get_embedding_pipeline_status()
            if not status["compatible"]:
                empty["error"] = "The index must be rebuilt with the current face pipeline."
                return empty
            rows = self.collection.get(
                where={"embedding_pipeline_version": self._embedding_pipeline_version},
                include=['embeddings', 'metadatas'],
            )
            ids = rows.get('ids') or []
            metadatas = rows.get('metadatas') or []
            embeddings = rows.get('embeddings')
            if embeddings is None:
                embeddings = []

            unnamed_ids: List[str] = []
            unnamed_embeddings: List[np.ndarray] = []
            metadata_by_id: Dict[str, Dict[str, Any]] = {}
            embedding_by_id: Dict[str, np.ndarray] = {}
            person_vectors: Dict[str, List[np.ndarray]] = {}
            person_metadata: Dict[str, Dict[str, Any]] = {}

            for face_id, metadata, embedding in zip(ids, metadatas, embeddings):
                metadata = metadata or {}
                vector = normalise_embedding(embedding)
                if not np.any(vector):
                    continue
                person_id = str(metadata.get('person_id') or '').strip()
                if person_id:
                    person_vectors.setdefault(person_id, []).append(vector)
                    person_metadata.setdefault(person_id, metadata)
                else:
                    key = str(face_id)
                    unnamed_ids.append(key)
                    unnamed_embeddings.append(vector)
                    metadata_by_id[key] = metadata
                    embedding_by_id[key] = vector

            person_centroids = {
                person_id: centroid_of(vectors)
                for person_id, vectors in person_vectors.items()
            }
            clusters = group_embeddings(
                unnamed_ids, unnamed_embeddings, threshold=threshold, margin=margin
            )

            groups = []
            for cluster in clusters:
                if len(cluster['face_ids']) < min_size:
                    continue

                suggestion = None
                if person_centroids:
                    best_person_id, best_similarity = max(
                        (
                            (person_id, float(np.dot(cluster['centroid'], centroid)))
                            for person_id, centroid in person_centroids.items()
                        ),
                        key=lambda item: item[1],
                    )
                    if best_similarity >= threshold:
                        metadata = person_metadata.get(best_person_id, {})
                        name = metadata.get('full_name') or ' '.join(
                            part for part in (
                                metadata.get('first_name', ''),
                                metadata.get('last_name', ''),
                            ) if part
                        )
                        suggestion = {
                            'person_id': best_person_id,
                            'full_name': name or 'Unknown person',
                            'similarity': best_similarity,
                        }

                groups.append({
                    'face_ids': cluster['face_ids'],
                    'faces': [
                        {
                            'face_id': face_id,
                            'metadata': metadata_by_id.get(face_id, {}),
                            'group_similarity': float(np.dot(
                                embedding_by_id[face_id], cluster['centroid']
                            )),
                        }
                        for face_id in cluster['face_ids']
                    ],
                    'size': len(cluster['face_ids']),
                    'mean_similarity': cluster['mean_similarity'],
                    'min_similarity': cluster['min_similarity'],
                    'suggested_person': suggestion,
                })

            grouped_faces = sum(group['size'] for group in groups)
            empty.update({
                'groups': groups,
                'unnamed_faces': len(unnamed_ids),
                'grouped_faces': grouped_faces,
                'singletons': len(unnamed_ids) - grouped_faces,
            })
            return empty

        except Exception as exc:  # noqa: BLE001 - the page shows the error
            logger.error(f"Error grouping faces: {exc}")
            empty['error'] = str(exc)
            return empty

    def find_similar_faces(
        self, 
        query_embedding: np.ndarray, 
        n_results: int = 10,
        min_similarity: float = SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Find similar faces (alias for search_similar_faces for compatibility)
        
        Args:
            query_embedding: Query face embedding
            n_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar faces with metadata
        """
        return self.search_similar_faces(query_embedding, n_results, min_similarity)
    
    def auto_assign_similar_faces(
        self,
        face_id: str,
        similarity_threshold: float = FACE_AUTO_ASSIGN_THRESHOLD,
    ) -> int:
        """Deprecated safe shim: similarity never writes an identity.

        Keep the legacy call signature for integrations, but require users to
        review each proposed match in the app before assigning a person.
        """
        logger.warning(
            "Ignored automatic identity propagation for %s (threshold %.3f); "
            "review candidate faces before assigning a person",
            face_id,
            float(similarity_threshold),
        )
        return 0
    
    def search_faces_by_name(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search faces by person name
        
        Args:
            search_term: Name to search for (first name, last name, or full name)
            
        Returns:
            List of matching faces
        """
        try:
            search_term = search_term.lower().strip()
            logger.info(f"Searching faces by name: '{search_term}'")
            
            # Get all faces with person names - completely safe approach
            results = self.collection.get(
                include=['metadatas']
            )
            
            matching_faces = []
            if results and results.get('ids') and results.get('metadatas'):
                logger.info(f"Searching through {len(results['metadatas'])} faces")
                
                for i, metadata in enumerate(results['metadatas']):
                    try:
                        # Extract person_id with extreme safety measures
                        stored_person_id = None
                        
                        if 'person_id' in metadata and metadata['person_id'] is not None:
                            raw_person_id = metadata['person_id']
                            
                            # Multiple safe extraction attempts
                            try:
                                # Check if it's already a simple string/number
                                if isinstance(raw_person_id, (str, int, float)):
                                    stored_person_id = str(raw_person_id).strip()
                                # Handle list/array types
                                elif hasattr(raw_person_id, '__len__') and hasattr(raw_person_id, '__getitem__'):
                                    if len(raw_person_id) > 0:
                                        stored_person_id = str(raw_person_id[0]).strip()
                                    else:
                                        stored_person_id = None
                                # Handle numpy arrays specifically
                                elif hasattr(raw_person_id, 'item'):
                                    stored_person_id = str(raw_person_id.item()).strip()
                                # Last resort conversion
                                else:
                                    stored_person_id = str(raw_person_id).strip()
                                    
                            except Exception as conversion_error:
                                logger.warning(f"Person ID conversion failed for index {i}: {conversion_error}")
                                stored_person_id = None
                        
                        # Only check faces with valid person data
                        if stored_person_id and stored_person_id not in ["None", "null", ""]:
                            first_name = (metadata.get('first_name') or '').lower()
                            last_name = (metadata.get('last_name') or '').lower()
                            full_name = (metadata.get('full_name') or '').lower()
                            
                            if (search_term in first_name or 
                                search_term in last_name or 
                                search_term in full_name):
                                
                                matching_faces.append({
                                    'face_id': results['ids'][i],
                                    'metadata': metadata,
                                    'embedding': results['embeddings'][i] if results.get('embeddings') and i < len(results['embeddings']) else None
                                })
                    
                    except Exception as search_error:
                        logger.warning(f"Error processing search at index {i}: {search_error}")
                        continue
            
            logger.info(f"Found {len(matching_faces)} faces matching '{search_term}'")
            return matching_faces
            
        except Exception as e:
            logger.error(f"Error searching faces by name '{search_term}': {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
