"""
Advanced Face Recognition Utilities
Enhanced face processing and embedding utilities for improved accuracy
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path

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
    """
    Advanced similarity calculation with multiple algorithms
    """
    
    def __init__(self):
        self.optimizer = FaceEmbeddingOptimizer()
    
    def calculate_comprehensive_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        algorithm: str = "enhanced"
    ) -> Dict[str, float]:
        """
        Calculate comprehensive similarity metrics
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding  
            algorithm: "basic", "enhanced", or "premium"
            
        Returns:
            Dictionary with similarity metrics
        """
        try:
            # Optimize both embeddings
            opt_emb1 = self.optimizer.optimize_embedding(embedding1)
            opt_emb2 = self.optimizer.optimize_embedding(embedding2)
            
            metrics = {}
            
            # Basic metrics
            metrics.update(self._calculate_basic_metrics(opt_emb1, opt_emb2))
            
            if algorithm in ["enhanced", "premium"]:
                # Enhanced metrics
                metrics.update(self._calculate_enhanced_metrics(opt_emb1, opt_emb2))
                
            if algorithm == "premium":
                # Premium metrics (more computationally expensive)
                metrics.update(self._calculate_premium_metrics(opt_emb1, opt_emb2))
            
            # Calculate final similarity score
            metrics['final_similarity'] = self._calculate_weighted_similarity(metrics, algorithm)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive similarity: {e}")
            return {'error': str(e), 'final_similarity': 0.0}
    
    def _calculate_basic_metrics(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        """Calculate basic similarity metrics"""
        metrics = {}
        
        # Cosine similarity
        cosine_sim = float(np.dot(emb1, emb2))
        metrics['cosine_similarity'] = np.clip(cosine_sim, -1.0, 1.0)
        
        # Euclidean distance and similarity
        euclidean_distance = float(np.linalg.norm(emb1 - emb2))
        euclidean_similarity = max(0.0, 1.0 - euclidean_distance / 2.0)
        metrics['euclidean_distance'] = euclidean_distance
        metrics['euclidean_similarity'] = euclidean_similarity
        
        return metrics
    
    def _calculate_enhanced_metrics(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        """Calculate enhanced similarity metrics"""
        metrics = {}
        
        # Manhattan distance (L1 norm)
        manhattan_distance = float(np.sum(np.abs(emb1 - emb2)))
        manhattan_similarity = max(0.0, 1.0 - manhattan_distance / len(emb1))
        metrics['manhattan_similarity'] = manhattan_similarity
        
        # Correlation coefficient
        try:
            correlation_matrix = np.corrcoef(emb1, emb2)
            correlation = correlation_matrix[0, 1]
            if not np.isnan(correlation):
                correlation_similarity = (correlation + 1.0) / 2.0
                metrics['correlation_similarity'] = float(correlation_similarity)
            else:
                metrics['correlation_similarity'] = 0.0
        except:
            metrics['correlation_similarity'] = 0.0
        
        # Angular similarity
        try:
            cosine_val = np.clip(metrics.get('cosine_similarity', 0.0), -1.0, 1.0)
            angular_distance = np.arccos(np.abs(cosine_val)) / (np.pi / 2)
            angular_similarity = 1.0 - angular_distance
            metrics['angular_similarity'] = float(angular_similarity)
        except:
            metrics['angular_similarity'] = 0.0
        
        return metrics
    
    def _calculate_premium_metrics(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        """Calculate premium similarity metrics (more expensive)"""
        metrics = {}
        
        # Earth Mover's Distance approximation
        try:
            # Simple approximation using sorted differences
            sorted_diff = np.sort(np.abs(emb1 - emb2))
            emd_approx = np.mean(sorted_diff[:len(sorted_diff)//2])  # Focus on smaller differences
            emd_similarity = max(0.0, 1.0 - emd_approx)
            metrics['emd_similarity'] = float(emd_similarity)
        except:
            metrics['emd_similarity'] = 0.0
        
        # Jensen-Shannon divergence approximation
        try:
            # Normalize embeddings to pseudo-probabilities
            p = np.abs(emb1) + 1e-10
            q = np.abs(emb2) + 1e-10
            p = p / np.sum(p)
            q = q / np.sum(q)
            m = (p + q) / 2
            
            kl_pm = np.sum(p * np.log(p / m))
            kl_qm = np.sum(q * np.log(q / m))
            js_divergence = (kl_pm + kl_qm) / 2
            js_similarity = max(0.0, 1.0 - js_divergence)
            metrics['js_similarity'] = float(js_similarity)
        except:
            metrics['js_similarity'] = 0.0
        
        return metrics
    
    def _calculate_weighted_similarity(self, metrics: Dict[str, float], algorithm: str) -> float:
        """Calculate final weighted similarity score"""
        try:
            cosine_sim = metrics.get('cosine_similarity', 0.0)
            euclidean_sim = metrics.get('euclidean_similarity', 0.0)
            
            if algorithm == "basic":
                # Simple weighted average
                return float(cosine_sim * 0.7 + euclidean_sim * 0.3)
                
            elif algorithm == "enhanced":
                # Include more metrics with careful weighting
                correlation_sim = metrics.get('correlation_similarity', 0.0)
                angular_sim = metrics.get('angular_similarity', 0.0)
                
                weighted = (
                    cosine_sim * 0.4 +
                    euclidean_sim * 0.3 +
                    correlation_sim * 0.2 +
                    angular_sim * 0.1
                )
                return float(weighted)
                
            elif algorithm == "premium":
                # Include all available metrics
                correlation_sim = metrics.get('correlation_similarity', 0.0)
                angular_sim = metrics.get('angular_similarity', 0.0)
                emd_sim = metrics.get('emd_similarity', 0.0)
                js_sim = metrics.get('js_similarity', 0.0)
                manhattan_sim = metrics.get('manhattan_similarity', 0.0)
                
                weighted = (
                    cosine_sim * 0.3 +
                    euclidean_sim * 0.2 +
                    correlation_sim * 0.15 +
                    angular_sim * 0.15 +
                    manhattan_sim * 0.1 +
                    emd_sim * 0.05 +
                    js_sim * 0.05
                )
                return float(weighted)
            
            else:
                return float(cosine_sim)
                
        except Exception as e:
            logger.error(f"Error calculating weighted similarity: {e}")
            return 0.0

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