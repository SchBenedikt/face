"""
Improved Face Recognition Engine using Best AI Models
Integrates with modular FaceDetector and FaceRecognizer
"""
# Set up environment
from setup_env import setup_environment
setup_environment()

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor

from utils import load_and_preprocess_image
from models import FaceDetector, FaceRecognizer
import config
from config import MIN_FACE_SIZE, BATCH_SIZE, MAX_WORKERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognitionEngine:
    """
    Improved face recognition engine using state-of-the-art AI models
    - Face Detection: RetinaFace, MTCNN, or OpenCV
    - Face Recognition: ArcFace, FaceNet512, or VGG-Face
    """
    
    def __init__(
        self, 
        detection_backend: str = "auto",
        recognition_model: str = "auto"
    ):
        """
        Initialize face recognition engine
        
        Args:
            detection_backend: Backend for face detection ('auto', 'retinaface', 'mtcnn', etc.)
            recognition_model: Model for face recognition ('auto', 'arcface', 'facenet512', etc.)
        """
        logger.info("Initializing Face Recognition Engine with best AI models...")
        
        # Initialize face detector
        self.face_detector = FaceDetector(backend=detection_backend)
        logger.info(f"Face Detector ready with backend: {self.face_detector.get_best_backend()}")
        
        # Initialize face recognizer
        self.face_recognizer = FaceRecognizer(model_name=recognition_model)
        logger.info(f"Face Recognizer ready with model: {self.face_recognizer.active_model}")
    
    def detect_faces(
        self, 
        image: np.ndarray,
        min_confidence: float = 0.9
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image
        
        Args:
            image: Input image as numpy array (RGB)
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of face bounding boxes as (top, right, bottom, left)
        """
        try:
            face_locations = self.face_detector.detect_faces(
                image,
                min_confidence=min_confidence,
                min_face_size=MIN_FACE_SIZE
            )
            
            # Filter valid faces
            if face_locations:
                face_locations = self.face_detector.filter_valid_faces(
                    face_locations,
                    image.shape[:2],
                    MIN_FACE_SIZE
                )
            
            logger.debug(f"Detected {len(face_locations)} valid faces")
            return face_locations
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def extract_face_embedding(
        self, 
        image: np.ndarray, 
        face_location: Tuple[int, int, int, int],
        align: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract face embedding using best recognition model
        
        Args:
            image: Input image
            face_location: Face bounding box (top, right, bottom, left)
            align: Whether to align face before extraction
            
        Returns:
            Face embedding vector or None if extraction fails
        """
        try:
            embedding = self.face_recognizer.extract_embedding(
                image,
                face_location,
                align=align
            )
            
            if embedding is not None:
                logger.debug(f"Successfully extracted {len(embedding)}D embedding")
            else:
                logger.warning("Failed to extract embedding")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None
    
    def process_image(
        self, 
        image_path: Path,
        min_confidence: float = 0.9
    ) -> Dict[str, Any]:
        """
        Process a single image: detect faces and extract embeddings
        
        Args:
            image_path: Path to image file
            min_confidence: Minimum confidence for face detection
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Load image
            image = load_and_preprocess_image(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": "Failed to load image",
                    "image_path": str(image_path)
                }
            
            # Detect faces
            face_locations = self.detect_faces(image, min_confidence)
            
            if not face_locations:
                return {
                    "success": True,
                    "image_path": str(image_path),
                    "face_count": 0,
                    "faces": []
                }
            
            # Extract embeddings for each face
            faces = []
            for idx, face_location in enumerate(face_locations):
                embedding = self.extract_face_embedding(image, face_location)
                
                if embedding is not None:
                    import uuid
                    face_data = {
                        "face_id": str(uuid.uuid4()),
                        "location": face_location,
                        "embedding": embedding,
                        "confidence": 1.0  # High confidence from modern models
                    }
                    faces.append(face_data)
            
            return {
                "success": True,
                "image_path": str(image_path),
                "face_count": len(faces),
                "faces": faces
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_path": str(image_path)
            }
    
    def process_images_batch(
        self,
        image_paths: List[Path],
        max_workers: int = MAX_WORKERS,
        min_confidence: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in parallel
        
        Args:
            image_paths: List of image paths
            max_workers: Maximum number of parallel workers
            min_confidence: Minimum confidence for face detection
            
        Returns:
            List of processing results for each image
        """
        results = []
        
        if max_workers == 1:
            # Sequential processing
            for image_path in image_paths:
                result = self.process_image(image_path, min_confidence)
                results.append(result)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {
                    executor.submit(self.process_image, path, min_confidence): path
                    for path in image_paths
                }
                
                for future in future_to_path:
                    try:
                        result = future.result(timeout=60)
                        results.append(result)
                    except Exception as e:
                        path = future_to_path[future]
                        logger.error(f"Error processing {path}: {e}")
                        results.append({
                            "success": False,
                            "error": str(e),
                            "image_path": str(path)
                        })
        
        return results
    
    def compare_faces(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 0.6
    ) -> Tuple[bool, float]:
        """
        Compare two face embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold for match
            
        Returns:
            Tuple of (is_match, similarity_score)
        """
        similarity = self.face_recognizer.compare_faces(
            embedding1,
            embedding2,
            metric="cosine"
        )
        
        is_match = similarity >= threshold
        
        return is_match, similarity
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about active models"""
        
        detector_backend = self.face_detector.get_best_backend()
        recognizer_info = self.face_recognizer.get_model_info()
        
        return {
            "detection": {
                "backend": detector_backend,
                "available_backends": self.face_detector.available_backends
            },
            "recognition": recognizer_info,
            "embedding_dimension": self.face_recognizer.get_embedding_dimension()
        }


# Legacy compatibility - create instance with same interface as old engine
def create_engine(detection_backend: str = "auto", recognition_model: str = "auto"):
    """Create face recognition engine instance"""
    return FaceRecognitionEngine(
        detection_backend=detection_backend,
        recognition_model=recognition_model
    )
