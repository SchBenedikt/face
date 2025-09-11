"""
Configuration settings for the Face Recognition Application
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

# Face recognition settings - Optimized for BEST accuracy over speed
FACE_RECOGNITION_MODEL = "cnn"  # Always use CNN for maximum accuracy
FACE_RECOGNITION_TOLERANCE = 0.5  # Lower tolerance for better precision  
MIN_FACE_SIZE = (40, 40)  # Slightly larger minimum for better quality detection
MAX_FACE_SIZE = (None, None)  # No limit for maximum quality

# Best accuracy models prioritized over speed

# Vector database settings
VECTOR_DB_PATH = str(DATA_DIR / "face_vectors.db")
COLLECTION_NAME = "face_embeddings"
SIMILARITY_THRESHOLD = 0.4  # Verbesserte Threshold für bessere Gesichtserkennung (0.4 = 40% Ähnlichkeit)

# Face Recognition Quality Settings - PREMIUM MODE: Best accuracy models only
FACE_EMBEDDING_MODEL = "ArcFace"  # Primary model - best for face verification and recognition
FACE_EMBEDDING_MODELS = ["ArcFace", "Facenet512", "VGG-Face", "SFace"]  # Best models ranked by accuracy
FACE_DETECTION_BACKENDS = ["retinaface", "mtcnn", "opencv"]  # Best detection backends ranked by accuracy
FACE_PREPROCESSING_ENABLED = True   # Advanced face preprocessing for maximum quality
FACE_QUALITY_VALIDATION = True     # Validate embedding quality  
FACE_SIMILARITY_ALGORITHM = "premium"  # Use premium algorithm for best accuracy
FACE_ENSEMBLE_WEIGHTING = True     # Use weighted ensemble of multiple models
FACE_ALIGNMENT_ENABLED = True      # Enable face alignment for better embeddings

# Advanced similarity settings - Optimized for best accuracy
SIMILARITY_THRESHOLD = 0.3  # Lower threshold for better recall with premium ensemble approach
ENSEMBLE_SIMILARITY_WEIGHTS = {
    'cosine': 0.4,        # ArcFace works best with cosine similarity
    'euclidean': 0.3,     # Good secondary measure
    'correlation': 0.2,   # Additional validation
    'angular': 0.1        # Fine-tuning measure
}

# Model performance settings - Prioritize accuracy over speed
EMBEDDING_CACHE_SIZE = 2000  # Larger cache for better model performance
BATCH_PROCESSING_ENABLED = True
PARALLEL_MODEL_EXTRACTION = True  # Enable parallel processing for ensemble accuracy
FACE_DETECTION_UPSAMPLING = 2     # Higher upsampling for better small face detection
FACE_DETECTION_MIN_NEIGHBORS = 6  # More strict detection for accuracy

# Scraping settings
MAX_IMAGES_PER_SITE = 0  # 0 = Unbegrenzt, alle verfügbaren Bilder herunterladen
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
DEFAULT_SCRAPING_SITES = [
    'https://example.com',  # Add safe example sites
]

# Duplikat-Erkennung
ENABLE_DUPLICATE_DETECTION = True
DUPLICATE_HASH_ALGORITHM = "md5"  # md5, sha1, sha256

# Streamlit settings
PAGE_TITLE = "Face Recognition Search"
PAGE_ICON = "🔍"
LAYOUT = "wide"

# Performance settings
BATCH_SIZE = 32
MAX_WORKERS = 4
EMBEDDING_DIMENSION = 128

# UI settings
RESULTS_PER_PAGE = 500  # Zeige viele Bilder auf einmal an
THUMBNAIL_SIZE = (400, 400)  # Größere Thumbnails für bessere Gesichtserkennung