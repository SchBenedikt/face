"""
Fast Configuration for Face Recognition - Optimized for Speed
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SCRAPED_DIR = DATA_DIR / "scraped"

# Create directories if they don't exist
for directory in [DATA_DIR, IMAGES_DIR, EMBEDDINGS_DIR, SCRAPED_DIR]:
    directory.mkdir(exist_ok=True)

# SPEED OPTIMIZED Face recognition settings
FACE_RECOGNITION_MODEL = "hog"  # Much faster than CNN, good enough accuracy
FACE_RECOGNITION_TOLERANCE = 0.5  # Slightly more lenient for speed
MIN_FACE_SIZE = (40, 40)  # Larger minimum to skip tiny faces (faster)
MAX_FACE_SIZE = (None, None)

# Vector database settings
VECTOR_DB_PATH = str(DATA_DIR / "face_vectors.db")
COLLECTION_NAME = "face_embeddings"
SIMILARITY_THRESHOLD = 0.4

# FAST Face Recognition Quality Settings - Single model for speed
FACE_EMBEDDING_MODEL = "Facenet"  # Faster than Facenet512
FACE_EMBEDDING_MODELS = ["Facenet"]  # Single model only for speed
FACE_DETECTION_BACKENDS = ["opencv"]  # Only OpenCV for speed
FACE_PREPROCESSING_ENABLED = False   # Disable advanced preprocessing
FACE_QUALITY_VALIDATION = False     # Skip validation for speed
FACE_SIMILARITY_ALGORITHM = "basic"  # Basic algorithm for speed
FACE_ENSEMBLE_WEIGHTING = False     # Disable ensemble
FACE_ALIGNMENT_ENABLED = False      # Disable alignment for speed

# Speed-optimized similarity settings
SIMILARITY_THRESHOLD = 0.45  # Slightly higher for speed
ENSEMBLE_SIMILARITY_WEIGHTS = {
    'cosine': 1.0  # Only cosine similarity for speed
}

# Streamlit settings
PAGE_TITLE = "🚀 Fast Face Recognition Search"
PAGE_ICON = "🚀"
LAYOUT = "wide"

# Results and display settings
RESULTS_PER_PAGE = 20
THUMBNAIL_SIZE = (150, 150)

# Processing settings - optimized for speed
BATCH_SIZE = 10  # Smaller batches for faster feedback
MAX_WORKERS = 1  # Single worker to avoid threading issues
PROCESSING_TIMEOUT = 15  # Much shorter timeout

# Image processing settings - optimized for speed
MAX_IMAGE_SIZE = (800, 800)  # Resize large images for speed
JPEG_QUALITY = 85
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
