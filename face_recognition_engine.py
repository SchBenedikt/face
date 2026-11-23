"""Face detection and embedding extraction for local similarity search."""
# Set up environment before importing other modules to prevent threading issues
from setup_env import setup_environment
setup_environment()

import warnings
# Suppress the pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

import cv2
import hashlib
import numpy as np
try:  # optional: needs dlib, which is not part of the default runtime
    import face_recognition
except Exception as _face_recognition_error:  # ImportError, OSError (missing dlib)
    face_recognition = None
    _FACE_RECOGNITION_IMPORT_ERROR = _face_recognition_error
else:
    _FACE_RECOGNITION_IMPORT_ERROR = None
from deepface import DeepFace
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from threading import RLock
import pickle
from weakref import ref as weakref_ref

from utils import load_and_preprocess_image
import config
from config import (
    FACE_RECOGNITION_MODEL, FACE_RECOGNITION_TOLERANCE, 
    MIN_FACE_SIZE,
    IMAGES_DIR, EMBEDDINGS_DIR, BATCH_SIZE, MAX_WORKERS
)
from settings_store import active_embedding_model, load_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_face_id(image_path: Path | str, face_index: int) -> str:
    """Return a stable id that is unique per file path and face position.

    The old scheme used only the file stem, so ``a/foto.jpg`` and
    ``b/foto.jpg`` produced the same ids. Besides mixing unrelated faces up,
    that made a second run collide with the rows written by the first one.
    """
    resolved = str(Path(image_path).expanduser().resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    return f"face_{digest}_{face_index}"


class FaceRecognitionEngine:
    """Detect faces and extract embeddings from the configured model."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        saved_settings = load_settings()
        self.primary_model = active_embedding_model()
        # Keep legacy public attributes for older integrations. Embedding
        # extraction uses only ``primary_model``; no ensemble is produced.
        self.models = list(getattr(config, "FACE_EMBEDDING_MODELS", [self.primary_model]))
        self.ensemble_enabled = False
        self.detection_model = FACE_RECOGNITION_MODEL
        try:
            saved_model = str(saved_settings.get("detection_model", FACE_RECOGNITION_MODEL)).lower()
            if saved_model in ("hog", "cnn"):
                self.detection_model = saved_model
        except Exception as exc:  # noqa: BLE001 - a broken settings file must not block startup
            logger.warning(f"Could not load detection model setting: {exc}")
        self.detection_backends = getattr(config, 'FACE_DETECTION_BACKENDS', ["opencv", "mtcnn", "retinaface"])
        self.model_cache = {}  # Cache for model loading
        self._aligned_face_crops = OrderedDict()
        self._aligned_face_crops_lock = RLock()

    def set_detection_model(self, model: str) -> None:
        """Apply a saved detector choice to this engine instance."""
        model = str(model).strip().lower()
        if model not in ("hog", "cnn"):
            raise ValueError("detection model must be 'hog' or 'cnn'")
        self.detection_model = model
        
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using optimized single method for speed
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face bounding boxes (top, right, bottom, left)
        """
        try:
            # DeepFace's landmark detector is the canonical path for this
            # pipeline: its aligned RGB crop is part of the embedding input.
            # Prefer it consistently even when the optional dlib package is
            # installed; otherwise identical app versions silently produce
            # different embeddings depending on the machine's dependencies.
            deepface_faces = self._detect_faces_with_deepface(image)
            if deepface_faces:
                candidate_locations = [location for location, _ in deepface_faces]
                valid_locations = self._filter_valid_faces(
                    candidate_locations, image.shape[:2], allow_full_frame=True
                )
                if valid_locations:
                    valid_set = set(valid_locations)
                    crops = {}
                    for location, crop in deepface_faces:
                        if crop is None:
                            continue
                        normalized = self._filter_valid_faces(
                            [location], image.shape[:2], allow_full_frame=True
                        )
                        if normalized and normalized[0] in valid_set:
                            crops[normalized[0]] = crop
                    if crops:
                        self._remember_aligned_face_crops(image, crops)
                    return valid_locations
                logger.debug("DeepFace detections were all filtered; trying fallback detectors")

            # Optional dlib and Haar detectors are fallbacks only. Returning
            # their unaligned crops as primary detections made indexing vary
            # with optional package availability and bypassed the configured
            # RetinaFace alignment path.
            try:
                if face_recognition is None:
                    raise RuntimeError(
                        "face_recognition is unavailable "
                        f"({_FACE_RECOGNITION_IMPORT_ERROR})"
                    )
                face_locations = face_recognition.face_locations(
                    image, 
                    # Upsampling recovers smaller faces that are commonly missed
                    # in full-resolution family/group photos. The old value (0)
                    # traded too much recall for a small speed gain.
                    number_of_times_to_upsample=1,
                    model=self.detection_model
                )
                if face_locations:
                    valid_locations = self._filter_valid_faces(face_locations, image.shape[:2])
                    if valid_locations:
                        logger.debug(f"face_recognition detected %d valid faces", len(valid_locations))
                        return valid_locations
                    logger.debug("face_recognition detections were all filtered; trying next detector")
            except Exception as e:
                logger.debug(f"face_recognition detection failed: {e}")

            # Final fallback - OpenCV Haar cascades (fastest, least robust)
            try:
                # load_and_preprocess_image returns RGB arrays (Pillow first).
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    # Haar is only a fallback when the configured detector is
                    # unavailable. Use stricter settings here because loose
                    # cascades frequently label textures and objects as faces.
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=MIN_FACE_SIZE,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces) > 0:
                    # Convert OpenCV format (x, y, w, h) to face_recognition format (top, right, bottom, left)
                    face_locations = []
                    for (x, y, w, h) in faces:
                        face_locations.append((y, x + w, y + h, x))
                    
                    logger.debug(f"OpenCV detected {len(face_locations)} faces")
                    return self._filter_valid_faces(face_locations, image.shape[:2])
            except Exception as e:
                logger.debug(f"OpenCV detection failed: {e}")
            
            # Skip DeepFace detection completely for speed
            logger.debug("No faces detected by fast methods")
            return []
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []

    def _detect_faces_with_deepface(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]]:
        """Return boxes and DeepFace's aligned RGB crops for detected faces."""
        extract_faces = getattr(DeepFace, "extract_faces", None)
        if not callable(extract_faces):
            return []

        try:
            # DeepFace accepts OpenCV arrays in BGR; app images are RGB.
            detector_image = (
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if image.ndim == 3 and image.shape[2] == 3
                else image
            )
            backends = list(dict.fromkeys(self.detection_backends))
        except Exception as exc:  # noqa: BLE001 - detector fallback must be optional
            logger.debug("Could not prepare DeepFace detector input: %s", exc)
            return []

        for backend in backends:
            try:
                detections = extract_faces(
                    img_path=detector_image,
                    detector_backend=backend,
                    enforce_detection=False,
                    align=True,
                    color_face="rgb",
                    normalize_face=False,
                )
                locations = []
                for detection in detections or []:
                    area = detection.get("facial_area") or {}
                    confidence = float(detection.get("confidence") or 0.0)
                    # DeepFace returns the entire image as a fallback box when
                    # a backend finds no face and detection is not enforced.
                    # A missing/zero confidence is not a real detection.
                    if not np.isfinite(confidence) or confidence < 0.65:
                        continue
                    x, y = int(area.get("x", 0)), int(area.get("y", 0))
                    width, height = int(area.get("w", 0)), int(area.get("h", 0))
                    if width > 0 and height > 0:
                        location = (y, x + width, y + height, x)
                        face_crop = detection.get("face")
                        if not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
                            face_crop = None
                        else:
                            face_crop = np.ascontiguousarray(face_crop)
                        locations.append((location, face_crop))
                if locations:
                    logger.debug(
                        "DeepFace/%s detected %d face(s)", backend, len(locations)
                    )
                    return sorted(locations, key=lambda item: (item[0][0], item[0][3]))
            except Exception as exc:  # noqa: BLE001 - try the next available backend
                logger.debug("DeepFace/%s face detection failed: %s", backend, exc)

        return []

    def _remember_aligned_face_crops(
        self,
        image: np.ndarray,
        crops: Dict[Tuple[int, int, int, int], np.ndarray],
    ) -> None:
        """Cache aligned crops without retaining full source images in memory."""
        image_key = id(image)
        with self._aligned_face_crops_lock:
            self._aligned_face_crops[image_key] = (weakref_ref(image), dict(crops))
            self._aligned_face_crops.move_to_end(image_key)
            while len(self._aligned_face_crops) > 8:
                self._aligned_face_crops.popitem(last=False)

    def _get_aligned_face_crop(
        self, image: np.ndarray, location: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Return the reusable aligned crop associated with one image and box."""
        image_key = id(image)
        with self._aligned_face_crops_lock:
            cached = self._aligned_face_crops.get(image_key)
            if cached is None or cached[0]() is not image:
                if cached is not None and cached[0]() is None:
                    self._aligned_face_crops.pop(image_key, None)
                return None
            self._aligned_face_crops.move_to_end(image_key)
            return cached[1].get(tuple(location))
    
    def _filter_valid_faces(
        self,
        face_locations: List[Tuple[int, int, int, int]],
        image_shape: Tuple[int, int],
        allow_full_frame: bool = False,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Filter face detections based on size and position constraints
        """
        try:
            valid_faces = []
            height, width = image_shape
            
            for top, right, bottom, left in face_locations:
                # Ensure coordinates are within image bounds
                # Face boxes use an exclusive bottom/right edge, so the image
                # dimensions themselves are valid upper bounds.
                top = max(0, min(height, top))
                bottom = max(0, min(height, bottom))
                left = max(0, min(width, left))
                right = max(0, min(width, right))
                
                # Calculate face dimensions
                face_width = right - left
                face_height = bottom - top
                
                # Apply size constraints
                if (face_width >= MIN_FACE_SIZE[0] and face_height >= MIN_FACE_SIZE[1]):
                    # Keep close-up portraits; the generic dlib/Haar path
                    # rejects boxes spanning almost the whole image, while
                    # confidence-filtered DeepFace boxes may legitimately do so.
                    if allow_full_frame or (
                        face_width < width * 0.98 and face_height < height * 0.98
                    ):
                        valid_faces.append((top, right, bottom, left))
                    else:
                        logger.debug(f"Filtered out oversized face: {face_width}x{face_height}")
                else:
                    logger.debug(f"Filtered out undersized face: {face_width}x{face_height}")
                    
            return valid_faces
            
        except Exception as e:
            logger.error(f"Error filtering faces: {e}")
            return face_locations

    def get_face_crop(
        self, image: np.ndarray, face_location: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Return the same aligned or padded crop used to build the embedding."""
        try:
            crop = self._get_aligned_face_crop(image, face_location)
            if crop is not None:
                return crop

            top, right, bottom, left = face_location
            height, width = image.shape[:2]
            padding = 10
            face_top = max(0, top - padding)
            face_bottom = min(height, bottom + padding)
            face_left = max(0, left - padding)
            face_right = min(width, right + padding)
            crop = image[face_top:face_bottom, face_left:face_right]
            return crop if crop.size else None
        except Exception as exc:  # noqa: BLE001 - invalid boxes should not crash UI
            logger.debug("Could not extract crop for face %s: %s", face_location, exc)
            return None
    
    def extract_face_embedding(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract face embedding using optimized single model approach for speed
        
        Args:
            image: Input image
            face_location: Face bounding box (top, right, bottom, left)
            
        Returns:
            Face embedding vector or None if extraction fails
        """
        try:
            top, right, bottom, left = face_location
            
            # Use the same crop for UI preview and embedding. DeepFace results
            # provide an eye-aligned RGB crop; dlib/Haar use a padded source box.
            face_image = self.get_face_crop(image, face_location)
            if face_image is None:
                logger.debug("No crop available for face at %s", face_location)
                return None
            
            # Validate face size
            if face_image.shape[0] < MIN_FACE_SIZE[0] or face_image.shape[1] < MIN_FACE_SIZE[1]:
                logger.debug(f"Face too small: {face_image.shape}")
                return None
            
            # Keep the detector crop at its original resolution. DeepFace
            # resizes to the model's actual input size; forcing every crop to
            # 224x224 and applying CLAHE/blur here discards detail and changes
            # the model's expected pixel distribution.
            try:
                embedding = self._extract_single_model_embedding(face_image, self.primary_model)
                if embedding is not None:
                    logger.debug(f"Fast embedding extraction with {self.primary_model}")
                    return self._finalize_embedding(embedding, [self.primary_model])
            except Exception as model_e:
                logger.debug(f"Primary model {self.primary_model} failed: {model_e}")
            
            # Never mix embeddings from different recognition models. The
            # face_recognition fallback produces 128-dimensional dlib vectors,
            # while the configured DeepFace model normally contains 512 values.
            # Mixing them makes Chroma writes/searches fail or compares
            # incompatible feature spaces. Report extraction failure instead.
            logger.warning(
                "Embedding extraction failed for the configured model %s; "
                "skipping this face rather than creating an incompatible vector",
                self.primary_model,
            )
            return None
                
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
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
            # DeepFace's ndarray API accepts BGR arrays. Our image loader and
            # detector pipeline use RGB, so convert explicitly at this
            # boundary instead of silently swapping red and blue channels.
            model_image = (
                cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                if face_image.ndim == 3 and face_image.shape[2] == 3
                else face_image
            )
            embedding = DeepFace.represent(
                model_image,
                model_name=model_name, 
                enforce_detection=False,
                detector_backend='skip',
                # Keep DeepFace's model-compatible default preprocessing.
                # ArcFace-specific normalization must not be applied to other
                # model families such as Buffalo_L or Facenet512.
                normalization='base'
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
        if face_recognition is None:
            return None

        try:
            logger.debug("Using fallback face_recognition method with preprocessing")
            
            face_encodings = face_recognition.face_encodings(
                face_image,
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
                lab = cv2.cvtColor(face_resized, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
                face_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
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
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("Embedding must contain finite values")

        # Do not clip individual coordinates: the embedding model learned the
        # full feature distribution, and clipping outliers changes identity
        # geometry. Unit-normalize only, which is also the metric expected by
        # the vector store's cosine scoring.
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-10:
            logger.warning("Embedding has very small norm")
            return np.zeros_like(vector, dtype=np.float32)
        return (vector / norm).astype(np.float32)
    
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
                return {
                    "error": "Failed to load image",
                    "image_path": str(image_path),
                    "source_unreadable": True,
                }
            
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
            embedding_failures = 0
            for i, face_location in enumerate(face_locations):
                try:
                    logger.debug(f"Extracting embedding for face {i+1}/{len(face_locations)}")
                    embedding = self.extract_face_embedding(image, face_location)
                    
                    if embedding is not None:
                        faces_data.append({
                            "face_id": build_face_id(image_path, i),
                            "location": face_location,
                            "embedding": embedding,
                            "embedding_size": len(embedding)
                        })
                        logger.debug(f"Successfully extracted embedding for face {i+1}")
                    else:
                        embedding_failures += 1
                        logger.warning(f"Failed to extract embedding for face {i+1} in {image_path}")
                        
                except Exception as e:
                    logger.error(f"Error extracting embedding for face {i+1} in {image_path}: {e}")
                    embedding_failures += 1
                    continue

            # Treat the image as a failed unit when any detected face could
            # not be embedded. In update mode, callers then retain the previous
            # complete set instead of replacing it with partial data.
            if embedding_failures:
                return {
                    "error": (
                        f"Could not extract embeddings for {embedding_failures} "
                        f"of {len(face_locations)} detected face(s)"
                    ),
                    "image_path": str(image_path),
                    "preserve_existing": True,
                }
            
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
            return {
                "error": str(e),
                "image_path": str(image_path),
                "preserve_existing": True,
            }
    
    def process_images_batch(self, image_paths: List[Path], max_workers: int = 1) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch with optimized speed and timeout handling
        
        Args:
            image_paths: List of image file paths
            max_workers: Maximum concurrent workers (optimized for speed)
            
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
                    if (i + 1) % 5 == 0 or (i + 1) == total_images:
                        logger.info(f"Processed {i+1}/{total_images} images")
                        
                except Exception as e:
                    logger.error(f"Error processing image {path}: {e}")
                    results.append({"error": str(e), "image_path": str(path)})
        else:
            # Use ThreadPoolExecutor with limited workers for better performance
            with ThreadPoolExecutor(max_workers=min(max_workers, 2)) as executor:
                # Submit all tasks
                futures = [executor.submit(self.process_image, path) for path in image_paths]
                
                # Collect results with much shorter timeout for speed
                for i, future in enumerate(futures):
                    try:
                        logger.debug(f"Waiting for result {i+1}/{total_images}: {image_paths[i].name}")
                        result = future.result(timeout=30)  # Reduced timeout to 30 seconds for speed
                        results.append(result)
                        
                        # More frequent progress updates
                        if (i + 1) % 5 == 0 or (i + 1) == total_images:
                            logger.info(f"Processed {i+1}/{total_images} images")
                            
                    except TimeoutError:
                        logger.error(f"Timeout processing image {image_paths[i]}")
                        results.append({"error": "Processing timeout (30s)", "image_path": str(image_paths[i])})
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
                    "model_name": self.primary_model,
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
