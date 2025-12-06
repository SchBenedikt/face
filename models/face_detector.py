"""
State-of-the-art Face Detection Module
Uses best available models: RetinaFace, MTCNN, or optimized OpenCV
"""
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Advanced face detection using multiple backends with fallback mechanism.
    Priority: RetinaFace > MTCNN > DeepFace > OpenCV
    """
    
    def __init__(self, backend: str = "auto"):
        """
        Initialize face detector with specified backend
        
        Args:
            backend: Detection backend ('auto', 'retinaface', 'mtcnn', 'opencv', 'deepface')
        """
        self.backend = backend
        self.available_backends = []
        self._initialize_backends()
        
    def _initialize_backends(self):
        """Initialize and check available backends"""
        # Check RetinaFace availability
        try:
            from retinaface import RetinaFace
            self.available_backends.append('retinaface')
            logger.info("✅ RetinaFace available (highest accuracy)")
        except ImportError:
            logger.debug("RetinaFace not available")
        
        # Check MTCNN availability (via DeepFace or direct)
        try:
            from deepface.detectors import MtcnnWrapper
            self.available_backends.append('mtcnn')
            logger.info("✅ MTCNN available (excellent accuracy)")
        except ImportError:
            logger.debug("MTCNN not available")
        
        # Check DeepFace availability
        try:
            from deepface import DeepFace
            self.available_backends.append('deepface')
            logger.info("✅ DeepFace available (good accuracy)")
        except ImportError:
            logger.debug("DeepFace not available")
        
        # OpenCV is always available
        self.available_backends.append('opencv')
        logger.info("✅ OpenCV available (fast, moderate accuracy)")
        
        # Initialize OpenCV cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        logger.info(f"Available detection backends: {', '.join(self.available_backends)}")
    
    def detect_faces(
        self, 
        image: np.ndarray,
        min_confidence: float = 0.9,
        min_face_size: Tuple[int, int] = (30, 30)
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image using best available backend
        
        Args:
            image: Input image as numpy array (RGB)
            min_confidence: Minimum confidence threshold for detection
            min_face_size: Minimum face size (width, height)
            
        Returns:
            List of face bounding boxes as (top, right, bottom, left)
        """
        if self.backend == "auto":
            # Try backends in order of accuracy
            for backend in ['retinaface', 'mtcnn', 'deepface', 'opencv']:
                if backend in self.available_backends:
                    try:
                        faces = self._detect_with_backend(
                            image, backend, min_confidence, min_face_size
                        )
                        if faces:
                            logger.debug(f"Successfully detected {len(faces)} faces with {backend}")
                            return faces
                    except Exception as e:
                        logger.debug(f"Backend {backend} failed: {e}")
                        continue
            return []
        else:
            # Use specified backend
            return self._detect_with_backend(
                image, self.backend, min_confidence, min_face_size
            )
    
    def _detect_with_backend(
        self,
        image: np.ndarray,
        backend: str,
        min_confidence: float,
        min_face_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces using specified backend"""
        
        if backend == 'retinaface':
            return self._detect_retinaface(image, min_confidence)
        elif backend == 'mtcnn':
            return self._detect_mtcnn(image, min_confidence)
        elif backend == 'deepface':
            return self._detect_deepface(image, min_confidence)
        elif backend == 'opencv':
            return self._detect_opencv(image, min_face_size)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _detect_retinaface(
        self, 
        image: np.ndarray, 
        min_confidence: float
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces using RetinaFace (best accuracy)"""
        try:
            from retinaface import RetinaFace
            
            # RetinaFace expects BGR
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # Detect faces
            faces = RetinaFace.detect_faces(image_bgr)
            
            if not isinstance(faces, dict):
                return []
            
            face_locations = []
            for key, face_data in faces.items():
                confidence = face_data.get('score', 0.0)
                if confidence >= min_confidence:
                    facial_area = face_data['facial_area']
                    x, y, x2, y2 = facial_area
                    # Convert to (top, right, bottom, left) format
                    face_locations.append((y, x2, y2, x))
            
            return face_locations
            
        except Exception as e:
            logger.error(f"RetinaFace detection error: {e}")
            return []
    
    def _detect_mtcnn(
        self, 
        image: np.ndarray, 
        min_confidence: float
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MTCNN (excellent accuracy)"""
        try:
            from deepface.detectors import MtcnnWrapper
            
            # Initialize MTCNN
            detector = MtcnnWrapper.build_model()
            
            # Detect faces
            faces = MtcnnWrapper.detect_face(detector, detector, image)
            
            if faces is None or len(faces) == 0:
                return []
            
            face_locations = []
            for face in faces:
                if len(face) >= 5:  # confidence is typically the 5th element
                    x, y, w, h, confidence = face[:5]
                    if confidence >= min_confidence:
                        # Convert to (top, right, bottom, left)
                        face_locations.append((int(y), int(x + w), int(y + h), int(x)))
            
            return face_locations
            
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
            return []
    
    def _detect_deepface(
        self, 
        image: np.ndarray, 
        min_confidence: float
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DeepFace backends"""
        try:
            from deepface import DeepFace
            
            # Try different DeepFace backends
            for backend in ['retinaface', 'mtcnn', 'opencv', 'ssd']:
                try:
                    # Use DeepFace's extract_faces function
                    faces = DeepFace.extract_faces(
                        img_path=image,
                        detector_backend=backend,
                        enforce_detection=False,
                        align=True
                    )
                    
                    if faces:
                        face_locations = []
                        for face in faces:
                            confidence = face.get('confidence', 0.0)
                            if confidence >= min_confidence:
                                region = face['facial_area']
                                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                                # Convert to (top, right, bottom, left)
                                face_locations.append((y, x + w, y + h, x))
                        
                        if face_locations:
                            return face_locations
                except:
                    continue
            
            return []
            
        except Exception as e:
            logger.error(f"DeepFace detection error: {e}")
            return []
    
    def _detect_opencv(
        self, 
        image: np.ndarray, 
        min_face_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar Cascades (fast fallback)"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert from (x, y, w, h) to (top, right, bottom, left)
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append((y, x + w, y + h, x))
            
            return face_locations
            
        except Exception as e:
            logger.error(f"OpenCV detection error: {e}")
            return []
    
    def get_best_backend(self) -> str:
        """Return the best available backend"""
        priority = ['retinaface', 'mtcnn', 'deepface', 'opencv']
        for backend in priority:
            if backend in self.available_backends:
                return backend
        return 'opencv'
    
    def filter_valid_faces(
        self,
        face_locations: List[Tuple[int, int, int, int]],
        image_shape: Tuple[int, int],
        min_face_size: Tuple[int, int] = (30, 30)
    ) -> List[Tuple[int, int, int, int]]:
        """
        Filter out invalid or too small faces
        
        Args:
            face_locations: List of face bounding boxes
            image_shape: Image dimensions (height, width)
            min_face_size: Minimum valid face size
            
        Returns:
            Filtered list of face bounding boxes
        """
        height, width = image_shape
        min_w, min_h = min_face_size
        
        valid_faces = []
        for top, right, bottom, left in face_locations:
            # Check if coordinates are valid
            if top < 0 or left < 0 or bottom > height or right > width:
                continue
            
            # Check if face is large enough
            face_width = right - left
            face_height = bottom - top
            
            if face_width >= min_w and face_height >= min_h:
                valid_faces.append((top, right, bottom, left))
        
        return valid_faces
