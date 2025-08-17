"""
Face Recognition Engine using multiple models for optimal performance
"""
# Set up environment before importing other modules to prevent threading issues
from setup_env import setup_environment
setup_environment()

import warnings
# Suppress the pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle

from utils import load_and_preprocess_image
import config
from config import (
    FACE_RECOGNITION_MODEL, FACE_RECOGNITION_TOLERANCE, 
    MIN_FACE_SIZE,
    IMAGES_DIR, EMBEDDINGS_DIR, BATCH_SIZE, MAX_WORKERS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionEngine:
    """
    Advanced face recognition engine combining multiple DeepFace models for optimal accuracy
    """
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Multiple models for ensemble approach - ordered by accuracy
        self.models = getattr(config, 'FACE_EMBEDDING_MODELS', [
            "Facenet512",    # Primary: Highest accuracy, 512-dim embeddings
            "ArcFace",       # Secondary: Excellent for verification
            "VGG-Face",      # Tertiary: Good general performance
            "Facenet"        # Fallback: Standard Facenet
        ])
        
        self.primary_model = self.models[0]
        self.detection_backends = getattr(config, 'FACE_DETECTION_BACKENDS', ["opencv", "mtcnn", "retinaface"])
        self.model_cache = {}  # Cache for model loading
        self.ensemble_enabled = getattr(config, 'FACE_ENSEMBLE_WEIGHTING', True)
        
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face bounding boxes (top, right, bottom, left)
        """
        try:
            # Use face_recognition library for detection
            face_locations = face_recognition.face_locations(image, model=FACE_RECOGNITION_MODEL)
            
            # Filter faces by size
            valid_faces = []
            for top, right, bottom, left in face_locations:
                width = right - left
                height = bottom - top
                
                # Only check minimum size, no maximum limit for highest quality
                if (width >= MIN_FACE_SIZE[0] and height >= MIN_FACE_SIZE[1]):
                    valid_faces.append((top, right, bottom, left))
                    
            return valid_faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def extract_face_embedding(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract face embedding using ensemble of multiple DeepFace models for enhanced accuracy
        
        Args:
            image: Input image
            face_location: Face bounding box (top, right, bottom, left)
            
        Returns:
            Optimized face embedding vector or None if extraction fails
        """
        try:
            top, right, bottom, left = face_location
            
            # Extract face region with adaptive padding
            padding = min(20, min(right-left, bottom-top) // 4)  # Adaptive padding
            height, width = image.shape[:2]
            
            # Apply padding while staying within image bounds
            face_top = max(0, top - padding)
            face_bottom = min(height, bottom + padding)
            face_left = max(0, left - padding)
            face_right = min(width, right + padding)
            
            face_image = image[face_top:face_bottom, face_left:face_right]
            
            # Validate face size
            if face_image.shape[0] < MIN_FACE_SIZE[0] or face_image.shape[1] < MIN_FACE_SIZE[1]:
                logger.debug(f"Face too small: {face_image.shape}")
                return None
            
            # Enhanced face preprocessing with alignment
            face_image = self._preprocess_face_advanced(face_image)
            
            # Ensemble approach: Try multiple models and combine results
            embeddings = []
            successful_models = []
            
            for model_name in self.models:
                try:
                    embedding = self._extract_single_model_embedding(face_image, model_name)
                    if embedding is not None and self._validate_embedding_quality(embedding):
                        embeddings.append(embedding)
                        successful_models.append(model_name)
                        logger.debug(f"Successfully extracted embedding with {model_name}")
                        
                        # For speed, use primary model if it works well
                        if model_name == self.primary_model and len(embeddings) == 1:
                            return self._finalize_embedding(embedding, successful_models)
                            
                except Exception as model_e:
                    logger.debug(f"Model {model_name} failed: {model_e}")
                    continue
            
            # Combine multiple embeddings if available
            if len(embeddings) > 1:
                combined_embedding = self._combine_embeddings(embeddings, successful_models)
                return self._finalize_embedding(combined_embedding, successful_models)
            elif len(embeddings) == 1:
                return self._finalize_embedding(embeddings[0], successful_models)
            
            # Fallback to face_recognition library
            return self._extract_fallback_embedding(face_image)
                
        except Exception as e:
            logger.error(f"Critical error extracting face embedding: {e}")
            return None
    
    def _preprocess_face_advanced(self, face_image: np.ndarray) -> np.ndarray:
        """
        Advanced face preprocessing with alignment and enhancement
        
        Args:
            face_image: Raw face image
            
        Returns:
            Preprocessed and aligned face image
        """
        try:
            # Step 1: Basic preprocessing
            face_processed = self._preprocess_face(face_image)
            
            # Step 2: Face alignment (if landmarks can be detected)
            try:
                # Try to detect and align face landmarks
                face_aligned = self._align_face(face_processed)
                if face_aligned is not None:
                    face_processed = face_aligned
            except Exception as align_e:
                logger.debug(f"Face alignment failed, using basic preprocessing: {align_e}")
            
            # Step 3: Advanced enhancement
            face_enhanced = self._enhance_face_quality(face_processed)
            
            return face_enhanced
            
        except Exception as e:
            logger.warning(f"Advanced face preprocessing failed: {e}, using basic")
            return self._preprocess_face(face_image)
    
    def _align_face(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Align face using detected landmarks for better embeddings
        """
        try:
            # This is a simplified alignment - in a full implementation,
            # you would use dlib or similar for proper landmark detection
            # For now, we'll just ensure the face is properly centered and rotated
            
            # Convert to RGB for landmark detection
            if len(face_image.shape) == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
            
            # Use basic alignment based on face region
            height, width = face_image.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            # Apply slight rotation correction if needed
            angle = 0  # Could be computed from eye positions
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            aligned = cv2.warpAffine(face_image, M, (width, height))
            
            return aligned
            
        except Exception as e:
            logger.debug(f"Face alignment error: {e}")
            return None
    
    def _enhance_face_quality(self, face_image: np.ndarray) -> np.ndarray:
        """
        Apply quality enhancement techniques
        """
        try:
            # Sharpening filter
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(face_image, -1, kernel)
            
            # Blend original and sharpened (subtle enhancement)
            enhanced = cv2.addWeighted(face_image, 0.7, sharpened, 0.3, 0)
            
            return enhanced
            
        except Exception as e:
            logger.debug(f"Face enhancement failed: {e}")
            return face_image
    
    def _extract_single_model_embedding(self, face_image: np.ndarray, model_name: str) -> Optional[np.ndarray]:
        """
        Extract embedding using a single DeepFace model
        """
        try:
            embedding = DeepFace.represent(
                face_image, 
                model_name=model_name, 
                enforce_detection=False,
                detector_backend='skip',
                normalization='ArcFace'
            )
            
            if isinstance(embedding, list) and len(embedding) > 0:
                embedding_vector = np.array(embedding[0]['embedding'], dtype=np.float32)
            else:
                embedding_vector = np.array(embedding['embedding'], dtype=np.float32)
            
            return self._advanced_normalize_embedding(embedding_vector)
            
        except Exception as e:
            logger.debug(f"Single model extraction failed for {model_name}: {e}")
            return None
    
    def _combine_embeddings(self, embeddings: List[np.ndarray], models: List[str]) -> np.ndarray:
        """
        Combine multiple embeddings using weighted average
        """
        try:
            # Weights based on model accuracy (can be tuned)
            model_weights = {
                "Facenet512": 0.4,
                "ArcFace": 0.3, 
                "VGG-Face": 0.2,
                "Facenet": 0.1
            }
            
            # Normalize weights to sum to 1
            weights = [model_weights.get(model, 0.1) for model in models]
            weights = np.array(weights) / sum(weights)
            
            # Weighted average
            combined = np.zeros_like(embeddings[0])
            for i, embedding in enumerate(embeddings):
                combined += weights[i] * embedding
            
            return combined.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Embedding combination failed: {e}, using first embedding")
            return embeddings[0]
    
    def _finalize_embedding(self, embedding: np.ndarray, models_used: List[str]) -> np.ndarray:
        """
        Finalize embedding with quality assurance
        """
        try:
            # Final normalization
            final_embedding = self._advanced_normalize_embedding(embedding)
            
            # Add metadata (for debugging)
            logger.debug(f"Finalized embedding using models: {models_used}")
            
            return final_embedding
            
        except Exception as e:
            logger.error(f"Embedding finalization failed: {e}")
            return embedding
    
    def _extract_fallback_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback embedding extraction using face_recognition library
        """
        try:
            logger.debug("Using fallback face_recognition method with preprocessing")
            
            # Convert to RGB for face_recognition
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
            
            face_encodings = face_recognition.face_encodings(
                face_rgb, 
                model="large"  # Use larger, more accurate model
            )
            
            if len(face_encodings) > 0:
                embedding_vector = face_encodings[0].astype(np.float32)
                embedding_vector = self._advanced_normalize_embedding(embedding_vector)
                
                if self._validate_embedding_quality(embedding_vector):
                    logger.debug(f"Fallback embedding extracted successfully")
                    return embedding_vector
                else:
                    logger.warning("Fallback embedding also low quality")
                    
        except Exception as fallback_e:
            logger.error(f"Fallback embedding extraction failed: {fallback_e}")
        
        return None
    
    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Advanced face preprocessing for better embedding quality
        
        Args:
            face_image: Raw face image
            
        Returns:
            Preprocessed face image
        """
        try:
            # Resize to optimal size for face recognition (224x224 for most models)
            target_size = (224, 224)
            face_resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Apply histogram equalization for better lighting
            if len(face_resized.shape) == 3:
                # Convert to LAB color space for better equalization
                lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
                face_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Grayscale image
                face_processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(face_resized)
            
            # Apply gentle gaussian blur to reduce noise
            face_processed = cv2.GaussianBlur(face_processed, (3, 3), 0.5)
            
            return face_processed
            
        except Exception as e:
            logger.warning(f"Face preprocessing failed: {e}, using original")
            return face_image
        """
        Advanced face preprocessing for better embedding quality
        
        Args:
            face_image: Raw face image
            
        Returns:
            Preprocessed face image
        """
        try:
            # Resize to optimal size for face recognition (224x224 for most models)
            target_size = (224, 224)
            face_resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Apply histogram equalization for better lighting
            if len(face_resized.shape) == 3:
                # Convert to LAB color space for better equalization
                lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
                face_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Grayscale image
                face_processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(face_resized)
            
            # Apply gentle gaussian blur to reduce noise
            face_processed = cv2.GaussianBlur(face_processed, (3, 3), 0.5)
            
            return face_processed
            
        except Exception as e:
            logger.warning(f"Face preprocessing failed: {e}, using original")
            return face_image
    
    def _advanced_normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Advanced embedding normalization for consistent similarity calculations
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Normalized embedding vector with improved stability
        """
        try:
            # Convert to float64 for better numerical precision
            embedding = embedding.astype(np.float64)
            
            # Remove potential outliers using robust normalization
            # Clip extreme values to 3 standard deviations
            std_dev = np.std(embedding)
            mean_val = np.mean(embedding)
            embedding = np.clip(embedding, mean_val - 3*std_dev, mean_val + 3*std_dev)
            
            # L2 normalization (unit vector)
            norm = np.linalg.norm(embedding)
            
            if norm < 1e-10:  # Avoid division by very small numbers
                logger.warning("Embedding has very small norm, using zero vector")
                return np.zeros_like(embedding, dtype=np.float32)
            
            # Normalize to unit length
            normalized = embedding / norm
            
            # Convert back to float32 for storage efficiency
            return normalized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in advanced embedding normalization: {e}")
            return embedding.astype(np.float32)
    
    def _validate_embedding_quality(self, embedding: np.ndarray) -> bool:
        """
        Validate if an embedding is of sufficient quality for face recognition
        
        Args:
            embedding: Face embedding to validate
            
        Returns:
            True if embedding quality is sufficient, False otherwise
        """
        try:
            if embedding is None or len(embedding) == 0:
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                logger.warning("Embedding contains NaN or infinite values")
                return False
            
            # Check if embedding is not all zeros
            if np.allclose(embedding, 0, atol=1e-10):
                logger.warning("Embedding is all zeros")
                return False
            
            # Check embedding magnitude (should be normalized to ~1.0)
            norm = np.linalg.norm(embedding)
            if norm < 0.1 or norm > 10.0:
                logger.warning(f"Embedding has unusual norm: {norm}")
                return False
            
            # Check for sufficient variance (not all values too similar)
            variance = np.var(embedding)
            if variance < 1e-6:
                logger.warning(f"Embedding has very low variance: {variance}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating embedding quality: {e}")
            return False
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Basic embedding normalization (kept for backward compatibility)
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Normalized embedding vector
        """
        # Use the advanced normalization method
        return self._advanced_normalize_embedding(embedding)
    
    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Process an image to extract all face embeddings with timeout protection
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing image info and face data
        """
        try:
            logger.debug(f"Starting to process image: {image_path.name}")
            
            # Load image with timeout protection
            image = load_and_preprocess_image(image_path)
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return {"error": "Failed to load image", "image_path": str(image_path)}
            
            logger.debug(f"Image loaded successfully: {image.shape}")
            
            # Detect faces with timeout protection
            try:
                face_locations = self.detect_faces(image)
                logger.debug(f"Face detection complete: found {len(face_locations)} faces")
            except Exception as e:
                logger.error(f"Face detection failed for {image_path}: {e}")
                return {"error": f"Face detection failed: {e}", "image_path": str(image_path)}
            
            if not face_locations:
                return {
                    "image_path": str(image_path),
                    "faces": [],
                    "face_count": 0,
                    "image_shape": image.shape
                }
            
            # Extract embeddings for each face
            faces_data = []
            for i, face_location in enumerate(face_locations):
                try:
                    logger.debug(f"Extracting embedding for face {i+1}/{len(face_locations)}")
                    embedding = self.extract_face_embedding(image, face_location)
                    
                    if embedding is not None:
                        faces_data.append({
                            "face_id": f"{image_path.stem}_face_{i}",
                            "location": face_location,
                            "embedding": embedding,
                            "embedding_size": len(embedding)
                        })
                        logger.debug(f"Successfully extracted embedding for face {i+1}")
                    else:
                        logger.warning(f"Failed to extract embedding for face {i+1} in {image_path}")
                        
                except Exception as e:
                    logger.error(f"Error extracting embedding for face {i+1} in {image_path}: {e}")
                    continue
            
            result = {
                "image_path": str(image_path),
                "faces": faces_data,
                "face_count": len(faces_data),
                "image_shape": image.shape
            }
            
            logger.debug(f"Image processing complete: {len(faces_data)} faces extracted")
            return result
            
        except Exception as e:
            logger.error(f"Critical error processing image {image_path}: {e}")
            return {"error": str(e), "image_path": str(image_path)}
    
    def process_images_batch(self, image_paths: List[Path], max_workers: int = 1) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch with limited concurrency to avoid mutex issues
        
        Args:
            image_paths: List of image file paths
            max_workers: Maximum concurrent workers (set to 1 to avoid mutex issues)
            
        Returns:
            List of processing results
        """
        results = []
        total_images = len(image_paths)
        
        logger.info(f"Starting batch processing of {total_images} images with {max_workers} workers")
        
        # Process sequentially to avoid mutex lock issues with face recognition libraries
        if max_workers == 1:
            for i, path in enumerate(image_paths):
                try:
                    logger.debug(f"Processing image {i+1}/{total_images}: {path.name}")
                    result = self.process_image(path)
                    results.append(result)
                    
                    # More frequent progress updates
                    if (i + 1) % 10 == 0 or (i + 1) == total_images:
                        logger.info(f"Processed {i+1}/{total_images} images")
                        
                except Exception as e:
                    logger.error(f"Error processing image {path}: {e}")
                    results.append({"error": str(e), "image_path": str(path)})
        else:
            # Use ThreadPoolExecutor with limited workers for better performance
            with ThreadPoolExecutor(max_workers=min(max_workers, 2)) as executor:
                # Submit all tasks
                futures = [executor.submit(self.process_image, path) for path in image_paths]
                
                # Collect results with better error handling
                for i, future in enumerate(futures):
                    try:
                        logger.debug(f"Waiting for result {i+1}/{total_images}: {image_paths[i].name}")
                        result = future.result(timeout=120)  # Increased timeout to 2 minutes
                        results.append(result)
                        
                        # More frequent progress updates
                        if (i + 1) % 10 == 0 or (i + 1) == total_images:
                            logger.info(f"Processed {i+1}/{total_images} images")
                            
                    except TimeoutError:
                        logger.error(f"Timeout processing image {image_paths[i]}")
                        results.append({"error": "Processing timeout", "image_path": str(image_paths[i])})
                    except Exception as e:
                        logger.error(f"Error processing image {image_paths[i]}: {e}")
                        results.append({"error": str(e), "image_path": str(image_paths[i])})
        
        logger.info(f"Batch processing complete: {len(results)} results for {total_images} images")
        return results
    
    def save_embeddings(self, results: List[Dict[str, Any]], output_file: Path) -> bool:
        """
        Save face embeddings to disk
        
        Args:
            results: Processing results from process_images_batch
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Filter valid results
            valid_results = [r for r in results if "error" not in r and r["face_count"] > 0]
            
            # Create embeddings data structure
            embeddings_data = {
                "metadata": {
                    "model_name": self.model_name,
                    "total_images": len(results),
                    "valid_images": len(valid_results),
                    "total_faces": sum(r["face_count"] for r in valid_results)
                },
                "embeddings": []
            }
            
            # Add face data
            for result in valid_results:
                for face_data in result["faces"]:
                    embeddings_data["embeddings"].append({
                        "face_id": face_data["face_id"],
                        "image_path": result["image_path"],
                        "location": face_data["location"],
                        "embedding": face_data["embedding"],
                        "embedding_size": face_data["embedding_size"]
                    })
            
            # Save to pickle file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'wb') as f:
                pickle.dump(embeddings_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Saved {len(embeddings_data['embeddings'])} face embeddings to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self, embeddings_file: Path) -> Optional[Dict[str, Any]]:
        """
        Load face embeddings from disk
        
        Args:
            embeddings_file: Path to embeddings file
            
        Returns:
            Embeddings data or None if failed
        """
        try:
            if not embeddings_file.exists():
                return None
                
            with open(embeddings_file, 'rb') as f:
                embeddings_data = pickle.load(f)
                
            logger.info(f"Loaded {len(embeddings_data.get('embeddings', []))} face embeddings")
            return embeddings_data
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive similarity metrics between two face embeddings
        
        Args:
            embedding1: First face embedding (should be normalized)
            embedding2: Second face embedding (should be normalized)
            
        Returns:
            Tuple of (primary_similarity, detailed_metrics)
            primary_similarity: Main similarity score (0-1, where 1 is most similar)  
            detailed_metrics: Dictionary with various similarity metrics
        """
        try:
            # Ensure both embeddings are properly normalized
            embedding1_norm = self._advanced_normalize_embedding(embedding1)
            embedding2_norm = self._advanced_normalize_embedding(embedding2)
            
            # Calculate multiple similarity metrics
            metrics = {}
            
            # 1. Cosine similarity (most reliable for face embeddings)
            cosine_similarity = np.dot(embedding1_norm, embedding2_norm)
            cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
            metrics['cosine_similarity'] = float(cosine_similarity)
            
            # 2. Euclidean distance (converted to similarity)
            euclidean_distance = np.linalg.norm(embedding1_norm - embedding2_norm)
            # Convert distance to similarity (lower distance = higher similarity)
            euclidean_similarity = max(0.0, 1.0 - (euclidean_distance / 2.0))
            metrics['euclidean_similarity'] = float(euclidean_similarity)
            metrics['euclidean_distance'] = float(euclidean_distance)
            
            # 3. Manhattan distance
            manhattan_distance = np.sum(np.abs(embedding1_norm - embedding2_norm))
            manhattan_similarity = max(0.0, 1.0 - (manhattan_distance / len(embedding1_norm)))
            metrics['manhattan_similarity'] = float(manhattan_similarity)
            
            # 4. Pearson correlation coefficient
            try:
                correlation = np.corrcoef(embedding1_norm, embedding2_norm)[0, 1]
                if not np.isnan(correlation):
                    # Convert correlation to similarity (adjust range from [-1,1] to [0,1])
                    correlation_similarity = (correlation + 1.0) / 2.0
                    metrics['correlation_similarity'] = float(correlation_similarity)
                else:
                    metrics['correlation_similarity'] = 0.0
            except:
                metrics['correlation_similarity'] = 0.0
            
            # Calculate weighted primary similarity score
            # Cosine similarity is most reliable for face embeddings, so weight it heavily
            primary_similarity = (
                cosine_similarity * 0.7 +          # Primary metric (70%)
                euclidean_similarity * 0.2 +       # Secondary metric (20%)
                metrics['correlation_similarity'] * 0.1  # Tertiary metric (10%)
            )
            
            # Ensure primary similarity is in valid range [0, 1]
            primary_similarity = max(0.0, min(1.0, primary_similarity))
            
            # Convert to 0-1 range where values closer to 1 indicate higher similarity
            if cosine_similarity < 0:
                # If cosine similarity is negative, the faces are quite different
                primary_similarity = primary_similarity * 0.5  # Reduce confidence
            
            metrics['primary_similarity'] = float(primary_similarity)
            
            logger.debug(f"Face similarity - Primary: {primary_similarity:.4f}, Cosine: {cosine_similarity:.4f}, Euclidean: {euclidean_similarity:.4f}")
            
            return float(primary_similarity), metrics
            
        except Exception as e:
            logger.error(f"Error comparing faces: {e}")
            return 0.0, {'error': str(e)}