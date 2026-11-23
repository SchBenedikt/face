"""
Utility functions for image processing and face recognition
"""
import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keep image discovery, validation, uploads and batch indexing aligned.
IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif",
})

def load_and_preprocess_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load and preprocess an image for face recognition with enhanced error handling
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image as numpy array or None if failed
    """
    try:
        image_path = Path(image_path)
        
        # Check if file exists and is not empty
        if not image_path.exists():
            logger.debug(f"Image file does not exist: {image_path}")
            return None
        
        if image_path.stat().st_size < 100:  # Less than 100 bytes is likely not a valid image
            logger.error(f"Image file too small ({image_path.stat().st_size} bytes): {image_path}")
            return None
        
        # Try multiple loading methods for robustness
        image = None
        
        # Method 1: Try PIL/Pillow first (handles more formats)
        try:
            with Image.open(image_path) as pil_image:
                # Camera photos commonly store portrait/rotation in EXIF
                # metadata instead of physically rotating the pixel data.
                # Apply it before coordinates and embeddings are computed.
                pil_image = ImageOps.exif_transpose(pil_image)
                # Convert to RGB if necessary
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Convert to numpy array
                image = np.array(pil_image)
                logger.debug(f"Successfully loaded with PIL: {image_path}")
                
        except Exception as pil_error:
            logger.debug(f"PIL loading failed for {image_path}: {pil_error}")
            
            # Method 2: Try OpenCV as fallback
            try:
                image = cv2.imread(str(image_path))
                if image is not None:
                    # Convert from BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    logger.debug(f"Successfully loaded with OpenCV: {image_path}")
                else:
                    logger.error(f"OpenCV failed to load image: {image_path}")
                    
            except Exception as cv_error:
                logger.error(f"Both PIL and OpenCV failed for {image_path}: PIL={pil_error}, CV={cv_error}")
                return None
        
        if image is None:
            logger.error(f"Failed to load image with all methods: {image_path}")
            return None
        
        # Validate image dimensions
        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.error(f"Invalid image dimensions: {image.shape} for {image_path}")
            return None
        
        height, width = image.shape[:2]
        
        # Check minimum size requirements
        if height < 50 or width < 50:
            logger.debug(f"Image too small for face detection: {width}x{height} for {image_path}")
            return None
        
        # Resize logic removed to preserve maximum image quality
        # No resize limit to allow maximum face detection quality
        
        # Final validation
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        logger.debug(f"Successfully preprocessed image: {image_path} -> {image.shape}")
        return image
        
    except Exception as e:
        logger.error(f"Critical error loading image {image_path}: {e}")
        return None


def load_uploaded_image(uploaded_file) -> Optional[Image.Image]:
    """Decode a bounded uploaded image as RGB and apply its EXIF orientation."""
    try:
        raw_size = getattr(uploaded_file, "size", None)
        if raw_size is None and hasattr(uploaded_file, "getbuffer"):
            raw_size = len(uploaded_file.getbuffer())
        if raw_size is not None and int(raw_size) > 25 * 1024 * 1024:
            raise ValueError("Uploaded image exceeds the 25 MiB limit")
        try:
            uploaded_file.seek(0)
        except (AttributeError, OSError):
            pass
        with Image.open(uploaded_file) as image:
            if image.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                raise ValueError(f"Unsupported image format: {image.format}")
            if image.width * image.height > 40_000_000:
                raise ValueError("Uploaded image exceeds the 40 megapixel limit")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - invalid uploads are a normal UI case
        logger.warning("Could not decode uploaded image: %s", exc)
        return None

def is_valid_image_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file is a valid image file with enhanced validation
    
    Args:
        file_path: Path to check
        
    Returns:
        True if valid image file, False otherwise
    """
    try:
        file_path = Path(file_path)
        
        # Check if file exists and has reasonable size
        if not file_path.exists():
            return False
        
        if file_path.stat().st_size < 100:  # Less than 100 bytes is probably not a valid image
            return False
        
        # Check extension
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            return False
        
        # Check magic bytes for proper image format validation
        try:
            with open(file_path, 'rb') as f:
                header = f.read(12)
            
            # Validate magic bytes for common formats
            magic_bytes_valid = False
            
            # JPEG
            if header[:3] == b'\xff\xd8\xff':
                magic_bytes_valid = True
            # PNG
            elif header[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
                magic_bytes_valid = True
            # GIF
            elif header[:6] in (b'GIF87a', b'GIF89a'):
                magic_bytes_valid = True
            # WebP
            elif header[:4] == b'RIFF' and header[8:12] == b'WEBP':
                magic_bytes_valid = True
            # BMP
            elif header[:2] == b'BM':
                magic_bytes_valid = True
            # TIFF
            elif header[:4] in (b'II*\x00', b'MM\x00*'):
                magic_bytes_valid = True
            
            if not magic_bytes_valid:
                logger.debug(f"Invalid magic bytes for {file_path}")
                return False
                
        except Exception as magic_error:
            logger.debug(f"Magic bytes check failed for {file_path}: {magic_error}")
            # Continue with PIL check as fallback
        
        # Final validation: try to open with PIL
        try:
            with Image.open(file_path) as img:
                # Try to load the image data to ensure it's not corrupted
                img.verify()  # This will raise an exception if the file is corrupted
                
                # Re-open for size check (verify() closes the file)
                with Image.open(file_path) as img2:
                    width, height = img2.size
                    
                    # Check minimum dimensions for meaningful processing
                    if width < 20 or height < 20:
                        logger.debug(f"Image too small: {width}x{height} for {file_path}")
                        return False
                    
                    # Check if it has a reasonable number of channels
                    if hasattr(img2, 'mode'):
                        if img2.mode not in ('RGB', 'RGBA', 'L', 'P'):
                            logger.debug(f"Unsupported image mode {img2.mode} for {file_path}")
                            return False
                    
                    return True
                    
        except Exception as pil_error:
            logger.debug(f"PIL validation failed for {file_path}: {pil_error}")
            return False
            
    except Exception as e:
        logger.debug(f"Image validation error for {file_path}: {e}")
        return False

def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (200, 200)) -> np.ndarray:
    """
    Create a high-quality thumbnail from an image
    
    Args:
        image: Input image as numpy array
        size: Thumbnail size (width, height)
        
    Returns:
        High-quality thumbnail image
    """
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Use high-quality resampling for better results
    # LANCZOS provides best quality for downsampling
    pil_image.thumbnail(size, Image.Resampling.LANCZOS)
    
    # Convert back to numpy array
    return np.array(pil_image)

def get_image_files(directory: Union[str, Path], validate: bool = True) -> List[Path]:
    """
    Get all image files from a directory
    
    Args:
        directory: Directory to search
        validate: Decode every file to verify it (slow). Pass False for plain
            file counts in the UI, where the extension check is enough.
        
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []
        
    image_files = []
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            if not validate or is_valid_image_file(file_path):
                image_files.append(file_path)
                
    return sorted(image_files)

def save_image(image: np.ndarray, output_path: Union[str, Path]) -> bool:
    """
    Save an image to disk
    
    Args:
        image: Image as numpy array
        output_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to PIL and save
        pil_image = Image.fromarray(image)
        pil_image.save(output_path, quality=90, optimize=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False

def calculate_image_hash(image: np.ndarray) -> str:
    """
    Calculate a hash for image deduplication
    
    Args:
        image: Image as numpy array
        
    Returns:
        Image hash as string
    """
    import hashlib
    
    # Resize to small size for hashing
    small_image = cv2.resize(image, (64, 64))
    
    # Convert to grayscale
    if len(small_image.shape) == 3:
        small_image = cv2.cvtColor(small_image, cv2.COLOR_RGB2GRAY)
    
    # Calculate hash
    image_bytes = small_image.tobytes()
    return hashlib.md5(image_bytes).hexdigest()

def normalise_box_size(size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Accept ``240`` or ``(240, 240)`` and return a (width, height) tuple."""
    if isinstance(size, (int, float)):
        return (max(1, int(size)), max(1, int(size)))
    try:
        width, height = size
        return (max(1, int(width)), max(1, int(height)))
    except (TypeError, ValueError):
        return (240, 240)

def parse_face_location(location) -> Optional[Tuple[int, int, int, int]]:
    """
    Normalise a stored face location to ``(top, right, bottom, left)``.

    ChromaDB metadata only keeps scalars, so locations are stored as the string
    ``"top,right,bottom,left"``. Older entries may still hold tuples, lists or
    dicts, so all of those shapes are accepted here.
    """
    if location is None:
        return None

    if isinstance(location, dict):
        try:
            return (
                int(location['top']),
                int(location['right']),
                int(location['bottom']),
                int(location['left']),
            )
        except (KeyError, TypeError, ValueError):
            return None

    if isinstance(location, str):
        parts = [part.strip() for part in location.split(',')]
        if len(parts) != 4:
            return None
        try:
            top, right, bottom, left = (int(float(part)) for part in parts)
        except ValueError:
            return None
        return (top, right, bottom, left)

    try:
        top, right, bottom, left = location
        return (int(top), int(right), int(bottom), int(left))
    except (TypeError, ValueError):
        return None

def clamp_face_box(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    padding: int = 0,
) -> Tuple[int, int, int, int]:
    """Add padding to a face box and clamp it to the image bounds."""
    top, right, bottom, left = box
    top = max(0, min(height - 1, top - padding))
    left = max(0, min(width - 1, left - padding))
    bottom = min(height, bottom + padding)
    right = min(width, right + padding)
    bottom = max(top + 1, bottom)
    right = max(left + 1, right)
    return (top, right, bottom, left)

def _open_rgb(image_path: Union[str, Path]) -> Optional[Image.Image]:
    """Open an image, honour EXIF orientation and convert it to RGB."""
    try:
        with Image.open(image_path) as pil_image:
            return ImageOps.exif_transpose(pil_image).convert('RGB')
    except Exception as e:
        logger.debug(f"Could not open image {image_path}: {e}")
        return None

def _encode_jpeg(pil_image: Image.Image, quality: int = 88) -> bytes:
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality, optimize=False)
    return buffer.getvalue()

def encode_image_thumbnail(
    image_path: Union[str, Path],
    size: Union[int, Tuple[int, int]] = (240, 240),
    quality: int = 88,
) -> Optional[bytes]:
    """
    Return a small JPEG thumbnail of an image as bytes.

    Bytes (not numpy arrays) are returned so the UI can cache thumbnails
    cheaply and Streamlit does not have to re-encode the picture every rerun.
    """
    pil_image = _open_rgb(image_path)
    if pil_image is None:
        return None
    try:
        pil_image.thumbnail(normalise_box_size(size), Image.Resampling.LANCZOS)
        return _encode_jpeg(pil_image, quality)
    except Exception as e:
        logger.debug(f"Thumbnail encoding failed for {image_path}: {e}")
        return None

def encode_face_thumbnail(
    image_path: Union[str, Path],
    location,
    size: Union[int, Tuple[int, int]] = (240, 240),
    padding: int = 10,
    quality: int = 88,
) -> Optional[bytes]:
    """Crop a face from an image and return it as a small JPEG thumbnail."""
    box = parse_face_location(location)
    if box is None:
        return encode_image_thumbnail(image_path, size, quality)

    pil_image = _open_rgb(image_path)
    if pil_image is None:
        return None
    try:
        top, right, bottom, left = clamp_face_box(
            box, pil_image.width, pil_image.height, padding
        )
        if right <= left or bottom <= top:
            return None
        crop = pil_image.crop((left, top, right, bottom))
        crop.thumbnail(normalise_box_size(size), Image.Resampling.LANCZOS)
        return _encode_jpeg(crop, quality)
    except Exception as e:
        logger.debug(f"Face thumbnail encoding failed for {image_path}: {e}")
        return None
