"""
Advanced Face Recognition Utilities
Enhanced face processing and embedding utilities for improved accuracy
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path

from similarity import normalise_embedding

logger = logging.getLogger(__name__)

class FaceEmbeddingOptimizer:
    """
    Advanced face embedding optimization for better similarity calculations
    """
    
    def __init__(self):
        self.embedding_cache = {}
        self.quality_threshold = 0.8
        
    def optimize_embedding(self, embedding: np.ndarray, metadata: Dict[str, Any] = None) -> np.ndarray:
        """
        Optimize face embedding for better similarity calculations
        
        Args:
            embedding: Raw face embedding
            metadata: Optional metadata for context-aware optimization
            
        Returns:
            Optimized embedding vector
        """
        try:
            # Step 1: Clean the embedding
            cleaned_embedding = self._clean_embedding(embedding)
            
            # Step 2: Apply normalization
            normalized_embedding = self._advanced_normalize(cleaned_embedding)
            
            # Step 3: Apply quality enhancement if possible
            enhanced_embedding = self._enhance_quality(normalized_embedding, metadata)
            
            return enhanced_embedding
            
        except Exception as e:
            logger.error(f"Error optimizing embedding: {e}")
            return embedding
    
    def _clean_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Clean embedding by removing outliers and invalid values
        """
        try:
            # Remove NaN and infinite values
            embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Clip extreme outliers (beyond 4 standard deviations)
            std_dev = np.std(embedding)
            mean_val = np.mean(embedding)
            embedding = np.clip(embedding, mean_val - 4*std_dev, mean_val + 4*std_dev)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Error cleaning embedding: {e}")
            return embedding
    
    def _advanced_normalize(self, embedding: np.ndarray) -> np.ndarray:
        """
        Advanced normalization with multiple techniques
        """
        try:
            # Convert to higher precision for calculations
            embedding = embedding.astype(np.float64)
            
            # Method 1: L2 normalization (unit vector)
            l2_norm = np.linalg.norm(embedding)
            if l2_norm > 1e-10:
                l2_normalized = embedding / l2_norm
            else:
                l2_normalized = embedding
            
            # Method 2: Standard score normalization (z-score)
            std_dev = np.std(embedding)
            if std_dev > 1e-10:
                z_normalized = (embedding - np.mean(embedding)) / std_dev
            else:
                z_normalized = embedding
            
            # Combine both methods for better stability
            # Use primarily L2 normalization with z-score regularization
            combined = l2_normalized * 0.8 + (z_normalized / np.linalg.norm(z_normalized)) * 0.2
            
            # Final L2 normalization to ensure unit vector
            final_norm = np.linalg.norm(combined)
            if final_norm > 1e-10:
                result = combined / final_norm
            else:
                result = l2_normalized
            
            return result.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Error in advanced normalization: {e}")
            # Fallback to simple L2 normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                return (embedding / norm).astype(np.float32)
            else:
                return embedding.astype(np.float32)
    
    def _enhance_quality(self, embedding: np.ndarray, metadata: Dict[str, Any] = None) -> np.ndarray:
        """
        Apply quality enhancement based on metadata context
        """
        try:
            enhanced = embedding.copy()
            
            # If we have face size information, boost quality for larger faces
            if metadata and 'face_size' in metadata:
                face_width, face_height = metadata['face_size']
                size_factor = min(face_width * face_height, 50000) / 50000  # Normalize to [0,1]
                
                # Apply gentle enhancement for larger faces
                if size_factor > 0.5:
                    enhanced = enhanced * (1.0 + (size_factor - 0.5) * 0.1)
            
            # Re-normalize after enhancement
            norm = np.linalg.norm(enhanced)
            if norm > 0:
                enhanced = enhanced / norm
            
            return enhanced.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Error in quality enhancement: {e}")
            return embedding

class FaceSimilarityCalculator:
    """Backward-compatible wrapper around normalized cosine similarity.

    Historical ``basic``, ``enhanced`` and ``premium`` names are accepted for
    callers that still pass them, but they no longer alter the score. The old
    weighted formulas mixed incompatible scales and included unvalidated
    metrics. ``final_similarity`` is now clipped cosine similarity in [0, 1],
    and must not be interpreted as a probability of shared identity.
    """

    _SUPPORTED_ALGORITHMS = {"basic", "enhanced", "premium"}

    def calculate_comprehensive_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        algorithm: str = "enhanced",
    ) -> Dict[str, float]:
        """Return normalized cosine and distance metrics for two embeddings."""
        try:
            if algorithm not in self._SUPPORTED_ALGORITHMS:
                raise ValueError(f"Unsupported similarity algorithm: {algorithm}")

            first = np.asarray(embedding1, dtype=np.float32).ravel()
            second = np.asarray(embedding2, dtype=np.float32).ravel()
            if first.size != second.size:
                raise ValueError("Embeddings must have the same dimension")
            if first.size == 0 or not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
                raise ValueError("Embeddings must be non-empty and finite")

            first = normalise_embedding(first)
            second = normalise_embedding(second)
            if not np.any(first) or not np.any(second):
                raise ValueError("Embeddings must have non-zero norm")

            metrics = self._calculate_basic_metrics(first, second)
            metrics["final_similarity"] = self._calculate_weighted_similarity(
                metrics, algorithm
            )
            return metrics
        except (TypeError, ValueError, FloatingPointError) as exc:
            logger.warning("Could not calculate face similarity: %s", exc)
            return {"error": str(exc), "final_similarity": 0.0}
        except Exception as exc:  # noqa: BLE001 - preserve the legacy error result
            logger.exception("Unexpected face similarity calculation error")
            return {"error": str(exc), "final_similarity": 0.0}

    def _calculate_basic_metrics(
        self, emb1: np.ndarray, emb2: np.ndarray
    ) -> Dict[str, float]:
        """Calculate cosine [-1, 1] and Euclidean metrics for unit vectors."""
        cosine = float(np.clip(np.dot(emb1, emb2), -1.0, 1.0))
        distance = float(np.linalg.norm(emb1 - emb2))
        return {
            "cosine_similarity": cosine,
            "euclidean_distance": distance,
            "euclidean_similarity": float(np.clip(1.0 - distance / 2.0, 0.0, 1.0)),
        }

    def _calculate_weighted_similarity(
        self, metrics: Dict[str, float], algorithm: str
    ) -> float:
        """Return the canonical score; legacy algorithm names do not reweight it."""
        if algorithm not in self._SUPPORTED_ALGORITHMS:
            return 0.0
        return float(np.clip(metrics.get("cosine_similarity", 0.0), 0.0, 1.0))

def validate_face_embedding(embedding: np.ndarray, strict: bool = False) -> Tuple[bool, str]:
    """
    Validate face embedding quality
    
    Args:
        embedding: Face embedding to validate
        strict: Whether to apply strict validation criteria
        
    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        if embedding is None:
            return False, "Embedding is None"
        
        if len(embedding) == 0:
            return False, "Embedding is empty"
        
        # Check for invalid values
        if np.any(np.isnan(embedding)):
            return False, "Embedding contains NaN values"
            
        if np.any(np.isinf(embedding)):
            return False, "Embedding contains infinite values"
        
        # Check if all zeros
        if np.allclose(embedding, 0, atol=1e-10):
            return False, "Embedding is all zeros"
        
        # Check embedding norm
        norm = np.linalg.norm(embedding)
        if strict:
            if norm < 0.1 or norm > 10.0:
                return False, f"Embedding norm out of range: {norm}"
        else:
            if norm < 1e-6:
                return False, f"Embedding norm too small: {norm}"
        
        # Check variance
        variance = np.var(embedding)
        min_variance = 1e-6 if not strict else 1e-4
        if variance < min_variance:
            return False, f"Embedding variance too low: {variance}"
        
        return True, "Valid embedding"
        
    except Exception as e:
        return False, f"Error validating embedding: {e}"