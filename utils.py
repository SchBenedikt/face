"""
Utility functions for image processing and face recognition
"""
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            logger.error(f"Image file does not exist: {image_path}")
            return None
        
        if image_path.stat().st_size < 100:  # Less than 100 bytes is likely not a valid image
            logger.error(f"Image file too small ({image_path.stat().st_size} bytes): {image_path}")
            return None
        
        # Try multiple loading methods for robustness
        image = None
        
        # Method 1: Try PIL/Pillow first (handles more formats)
        try:
            with Image.open(image_path) as pil_image:
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
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif', '.gif'}
        if file_path.suffix.lower() not in valid_extensions:
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

def get_image_files(directory: Union[str, Path]) -> List[Path]:
    """
    Get all image files from a directory
    
    Args:
        directory: Directory to search
        
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []
        
    image_files = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            if is_valid_image_file(file_path):
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