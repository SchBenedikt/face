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
from datetime import datetime
import uuid

from config import VECTOR_DB_PATH, COLLECTION_NAME, SIMILARITY_THRESHOLD
from similarity import (
    cosine_from_l2_distance,
    normalise_embedding,
    similarity_from_l2_distance,
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
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Face embeddings for similarity search"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
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
            # Convert embedding to list for ChromaDB
            embedding_list = embedding.tolist()
            
            # Add metadata
            full_metadata = {
                **metadata,
                "added_at": datetime.now().isoformat(),
                "embedding_dimension": len(embedding)
            }
            
            # Add or replace the row; re-adding a face must not fail.
            self.collection.upsert(
                embeddings=[embedding_list],
                metadatas=[full_metadata],
                ids=[face_id]
            )
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
            
            # Convert embeddings to lists
            embedding_lists = [emb.tolist() for emb in embeddings]
            
            # Add metadata
            full_metadatas = []
            for metadata in metadatas:
                full_metadata = {
                    **metadata,
                    "added_at": datetime.now().isoformat(),
                    "embedding_dimension": len(embeddings[0])
                }
                full_metadatas.append(full_metadata)
            
            # Add or update the new rows first. Only delete stale rows after a
            # successful write, so a failed reprocess keeps the old data.
            self.collection.upsert(
                embeddings=embedding_lists,
                metadatas=full_metadatas,
                ids=face_ids
            )

            image_paths = {
                str(metadata.get("image_path"))
                for metadata in metadatas
                if metadata.get("image_path")
            }
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
        removed = 0
        for image_path in sorted(image_paths):
            try:
                existing = self.collection.get(
                    where={"image_path": str(image_path)},
                    include=[],
                )
                stale = [
                    face_id
                    for face_id in (existing.get("ids") or [])
                    if face_id not in keep_ids
                ]
                if stale:
                    self.collection.delete(ids=stale)
                    removed += len(stale)
            except Exception as exc:  # noqa: BLE001 - cleanup must not fail the write
                logger.warning(f"Could not clean old faces for {image_path}: {exc}")
        return removed

    def prune_missing_faces(self) -> int:
        """Delete embeddings whose source image no longer exists.

        Returns the number of removed rows. This makes the database usable
        again after files were removed outside the app, without touching rows
        that still have a valid image.
        """
        try:
            total = self.collection.count()
            if total == 0:
                return 0

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
                    if not image_path or not Path(str(image_path)).is_file():
                        if index < len(ids):
                            stale.append(ids[index])
                offset += chunk_size

            for start in range(0, len(stale), 500):
                self.collection.delete(ids=stale[start:start + 500])

            self.invalidate_stats_cache()
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

            # Trade a little recall for a lot of speed: the old code asked for
            # up to 20,000 neighbours including their embeddings.
            search_n = int(min(max(n_results * 3, 50), 500, collection_size))

            results = self.collection.query(
                query_embeddings=[query_list.tolist()],
                n_results=search_n,
                include=['metadatas', 'distances'],
            )

            ids = results.get('ids') or [[]]
            distances = results.get('distances') or [[]]
            metadatas = results.get('metadatas') or [[]]

            similar_faces: List[Dict[str, Any]] = []
            for index, face_id in enumerate(ids[0]):
                try:
                    distance = float(distances[0][index])
                    similarity = similarity_from_l2_distance(distance)
                    if similarity < min_similarity:
                        continue

                    similar_faces.append({
                        'face_id': face_id,
                        'similarity': similarity,
                        'similarity_percentage': similarity * 100,
                        'cosine_similarity': cosine_from_l2_distance(distance),
                        'euclidean_distance': distance,
                        'confidence_score': similarity,
                        'metadata': metadatas[0][index] or {},
                        'similarity_metrics': {
                            'cosine_similarity': cosine_from_l2_distance(distance),
                            'euclidean_distance': distance,
                            'primary_similarity': similarity,
                        },
                    })
                except Exception as e:
                    logger.debug(f"Skipping malformed search hit: {e}")
                    continue

            similar_faces.sort(key=lambda face: face['similarity'], reverse=True)
            top_score = similar_faces[0]['similarity_percentage'] if similar_faces else 0.0
            logger.info(
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
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Face embeddings for similarity search"}
            )
            self.invalidate_stats_cache()
            
            logger.info(f"Cleared collection {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
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
    
    def assign_person_name(self, face_id: str, person_data: Dict[str, str], person_id: Optional[str] = None) -> Optional[str]:
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
                'name_assigned_at': datetime.now().isoformat()
            }
            
            success = self.update_face_metadata(face_id, person_metadata)
            return person_id if success else None
            
        except Exception as e:
            logger.error(f"Error assigning person data to {face_id}: {e}")
            return None

    def assign_person_name_with_auto_matching(self, face_id: str, person_data: Dict[str, str], person_id: Optional[str] = None, similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Assign person name and automatically find and assign similar faces
        
        Args:
            face_id: Face identifier for the initial face
            person_data: Dictionary with person information
            person_id: Unique person ID (if None, creates new one)
            similarity_threshold: Minimum similarity for auto-assignment (default 0.8 = 80%)
            
        Returns:
            Dictionary with results:
            - success: bool
            - person_id: str
            - total_assigned: int (number of faces assigned)
            - similar_faces: list of assigned face IDs
        """
        try:
            # First assign the name to the original face
            assigned_person_id = self.assign_person_name(face_id, person_data, person_id)
            
            if not assigned_person_id:
                return {
                    'success': False,
                    'error': 'Failed to assign name to original face',
                    'total_assigned': 0,
                    'similar_faces': []
                }
            
            # Get the embedding of the original face to find similar ones
            original_face = self.get_face_by_id(face_id)
            if not original_face or 'embedding' not in original_face:
                logger.warning(f"Could not get embedding for face {face_id}, skipping auto-matching")
                return {
                    'success': True,
                    'person_id': assigned_person_id,
                    'total_assigned': 1,
                    'similar_faces': [],
                    'warning': 'Auto-matching skipped - no embedding available'
                }
            
            # Search for similar faces with high similarity
            similar_faces = self.search_similar_faces(
                original_face['embedding'],
                n_results=100,  # Get more results to filter
                min_similarity=similarity_threshold
            )
            
            assigned_similar = []
            
            # Auto-assign names to similar faces that don't already have names
            for similar_face in similar_faces:
                similar_face_id = similar_face.get('id', similar_face.get('face_id'))
                
                # Skip the original face
                if similar_face_id == face_id:
                    continue
                
                # Check if face already has a name assigned
                existing_metadata = similar_face.get('metadata', {})
                if existing_metadata.get('person_id') and existing_metadata.get('full_name'):
                    # Skip faces that already have names
                    continue
                
                # Assign the same person data to this similar face
                if self.assign_person_name(similar_face_id, person_data, assigned_person_id):
                    assigned_similar.append(similar_face_id)
                    logger.info(f"Auto-assigned name to similar face {similar_face_id} (similarity: {similar_face['similarity']:.1%})")
            
            total_assigned = 1 + len(assigned_similar)  # Original + similar faces
            
            logger.info(f"Successfully assigned person '{person_data.get('first_name', '')} {person_data.get('last_name', '')}' to {total_assigned} faces")
            
            return {
                'success': True,
                'person_id': assigned_person_id,
                'total_assigned': total_assigned,
                'similar_faces': assigned_similar,
                'similarity_threshold': similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"Error in assign_person_name_with_auto_matching: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_assigned': 0,
                'similar_faces': []
            }
    
    def remove_person_name(self, face_id: str) -> bool:
        """
        Remove person name assignment from a face
        
        Args:
            face_id: Face identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
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
                'name_removed_at': datetime.now().isoformat()
            }
            
            return self.update_face_metadata(face_id, removal_metadata)
            
        except Exception as e:
            logger.error(f"Error removing person name from {face_id}: {e}")
            return False
    
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
                            full_name = ' '.join(name_parts) if name_parts else 'Unbekannt'
                            
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
    
    def auto_assign_similar_faces(self, face_id: str, similarity_threshold: float = 0.8) -> int:
        """
        Automatically assign person name to faces with high similarity (>threshold)
        
        Args:
            face_id: Reference face with assigned person
            similarity_threshold: Minimum similarity to auto-assign (default 0.8 = 80%)
            
        Returns:
            Number of faces that were auto-assigned
        """
        try:
            # Get the reference face
            reference_face = self.get_face_by_id(face_id)
            if not reference_face or not reference_face['metadata'].get('person_id'):
                return 0
            
            # Extract all person data from reference face
            metadata = reference_face['metadata']
            person_data = {
                'first_name': metadata.get('first_name', ''),
                'middle_names': metadata.get('middle_names', ''),
                'last_name': metadata.get('last_name', ''),
                'birth_date': metadata.get('birth_date', ''),
                'birth_place': metadata.get('birth_place', ''),
                'notes': metadata.get('notes', '')
            }
            person_id = metadata['person_id']
            
            # Find similar faces using the correct method
            similar_faces = self.search_similar_faces(
                reference_face['embedding'],
                n_results=1000,  # Check many faces
                min_similarity=similarity_threshold
            )
            
            assigned_count = 0
            for similar_face in similar_faces:
                # Skip if already has person assigned or is the reference face
                if (similar_face['face_id'] != face_id and 
                    not similar_face['metadata'].get('person_id')):
                    
                    # Auto-assign the same person with all data
                    success = self.assign_person_name(
                        similar_face['face_id'],
                        person_data,
                        person_id
                    )
                    if success:
                        assigned_count += 1
                        
            return assigned_count
            
        except Exception as e:
            logger.error(f"Error auto-assigning similar faces for {face_id}: {e}")
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