"""
State-of-the-art Face Recognition Module
Uses best available models: ArcFace, AdaFace, or FaceNet512
"""
import numpy as np
import logging
from typing import Optional, List, Tuple
import cv2

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Advanced face recognition using state-of-the-art embedding models.
    Priority: ArcFace > FaceNet512 > VGG-Face > FaceNet
    """
    
    def __init__(self, model_name: str = "auto"):
        """
        Initialize face recognizer with specified model
        
        Args:
            model_name: Model to use ('auto', 'arcface', 'facenet512', 'vggface', 'facenet')
        """
        self.model_name = model_name
        self.available_models = []
        self.active_model = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize and check available models"""
        try:
            from deepface import DeepFace
            
            # Check which models are available
            available = ['ArcFace', 'Facenet512', 'VGG-Face', 'Facenet']
            
            for model in available:
                try:
                    # Try to build model to verify availability
                    DeepFace.build_model(model)
                    self.available_models.append(model)
                    logger.info(f"✅ {model} available")
                except Exception as e:
                    logger.debug(f"{model} not available: {e}")
            
            if not self.available_models:
                raise RuntimeError("No face recognition models available")
            
            # Set active model
            if self.model_name == "auto":
                # Use best available model
                self.active_model = self.available_models[0]
            elif self.model_name.capitalize() in ['Arcface']:
                self.active_model = 'ArcFace'
            elif self.model_name.capitalize() in ['Facenet512']:
                self.active_model = 'Facenet512'
            elif self.model_name.upper() in ['VGG-FACE', 'VGGFACE']:
                self.active_model = 'VGG-Face'
            elif self.model_name.capitalize() in ['Facenet']:
                self.active_model = 'Facenet'
            else:
                self.active_model = self.available_models[0]
            
            logger.info(f"Active recognition model: {self.active_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize face recognition models: {e}")
            raise
    
    def extract_embedding(
        self, 
        image: np.ndarray,
        face_location: Optional[Tuple[int, int, int, int]] = None,
        align: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract face embedding from image
        
        Args:
            image: Input image as numpy array (RGB)
            face_location: Face bounding box as (top, right, bottom, left), None for full image
            align: Whether to align face before extraction
            
        Returns:
            Face embedding as numpy array or None if extraction fails
        """
        try:
            from deepface import DeepFace
            
            # Crop face if location is provided
            if face_location is not None:
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
            else:
                face_image = image
            
            # Extract embedding using DeepFace
            embedding_obj = DeepFace.represent(
                img_path=face_image,
                model_name=self.active_model,
                enforce_detection=False,
                detector_backend='skip',  # Skip detection as we already have the face
                align=align
            )
            
            # Extract embedding array
            if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                embedding = embedding_obj[0].get('embedding', None)
            elif isinstance(embedding_obj, dict):
                embedding = embedding_obj.get('embedding', None)
            else:
                embedding = None
            
            if embedding is not None:
                return np.array(embedding, dtype=np.float32)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None
    
    def extract_embeddings_batch(
        self,
        images: List[np.ndarray],
        face_locations: Optional[List[Tuple[int, int, int, int]]] = None,
        align: bool = True
    ) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings for multiple faces
        
        Args:
            images: List of input images
            face_locations: List of face bounding boxes (same length as images)
            align: Whether to align faces before extraction
            
        Returns:
            List of face embeddings
        """
        embeddings = []
        
        if face_locations is None:
            face_locations = [None] * len(images)
        
        for image, face_loc in zip(images, face_locations):
            embedding = self.extract_embedding(image, face_loc, align)
            embeddings.append(embedding)
        
        return embeddings
    
    def compare_faces(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compare two face embeddings and return similarity score
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            metric: Distance metric ('cosine', 'euclidean', or 'euclidean_l2')
            
        Returns:
            Similarity score (higher = more similar, range 0-1)
        """
        try:
            if metric == "cosine":
                # Cosine similarity
                similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                )
                # Convert to 0-1 range where 1 is most similar
                return (similarity + 1) / 2
                
            elif metric == "euclidean":
                # Euclidean distance
                distance = np.linalg.norm(embedding1 - embedding2)
                # Convert to similarity (inverse relationship)
                # Normalize to 0-1 range
                similarity = 1 / (1 + distance)
                return similarity
                
            elif metric == "euclidean_l2":
                # L2 normalized Euclidean distance
                norm1 = embedding1 / np.linalg.norm(embedding1)
                norm2 = embedding2 / np.linalg.norm(embedding2)
                distance = np.linalg.norm(norm1 - norm2)
                similarity = 1 / (1 + distance)
                return similarity
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
        except Exception as e:
            logger.error(f"Face comparison error: {e}")
            return 0.0
    
    def verify_face(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 0.6
    ) -> bool:
        """
        Verify if two embeddings belong to the same person
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding  
            threshold: Similarity threshold for verification
            
        Returns:
            True if faces match, False otherwise
        """
        similarity = self.compare_faces(embedding1, embedding2, metric="cosine")
        return similarity >= threshold
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for active model"""
        dimensions = {
            'ArcFace': 512,
            'Facenet512': 512,
            'VGG-Face': 4096,
            'Facenet': 128
        }
        return dimensions.get(self.active_model, 512)
    
    def get_model_info(self) -> dict:
        """Get information about the active model"""
        info = {
            'ArcFace': {
                'name': 'ArcFace',
                'dimension': 512,
                'accuracy': 'State-of-the-art',
                'speed': 'Fast',
                'description': 'Angular margin loss for superior face recognition'
            },
            'Facenet512': {
                'name': 'FaceNet512',
                'dimension': 512,
                'accuracy': 'Excellent',
                'speed': 'Fast',
                'description': 'High-quality embeddings with 512 dimensions'
            },
            'VGG-Face': {
                'name': 'VGG-Face',
                'dimension': 4096,
                'accuracy': 'Good',
                'speed': 'Moderate',
                'description': 'Robust face recognition with deep CNN'
            },
            'Facenet': {
                'name': 'FaceNet',
                'dimension': 128,
                'accuracy': 'Good',
                'speed': 'Very Fast',
                'description': 'Compact embeddings for efficient processing'
            }
        }
        
        return info.get(self.active_model, {
            'name': self.active_model,
            'dimension': 512,
            'accuracy': 'Unknown',
            'speed': 'Unknown',
            'description': 'Face recognition model'
        })
