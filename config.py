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

# Face recognition settings - Enhanced for maximum quality
FACE_RECOGNITION_MODEL = "cnn"  # "hog" (schneller) oder "cnn" (genauer) - using CNN for best quality
FACE_RECOGNITION_TOLERANCE = 0.5  # Lowered for stricter matching
MIN_FACE_SIZE = (30, 30)  # Kleinere Gesichter zulassen
MAX_FACE_SIZE = (None, None)  # Keine Begrenzung für Gesichtsgröße - maximale Qualität

# Model comparison:
# HOG: Schneller, weniger Ressourcen, gut für Echtzeit
# CNN: Genauer, mehr Ressourcen, besser für Qualität

# Vector database settings
VECTOR_DB_PATH = str(DATA_DIR / "face_vectors.db")
COLLECTION_NAME = "face_embeddings"

# Face Recognition Quality Settings - Enhanced for better accuracy over speed
FACE_EMBEDDING_MODEL = "Facenet512"  # Primary DeepFace model for highest quality
FACE_EMBEDDING_MODELS = ["Facenet512", "ArcFace", "VGG-Face", "Facenet"]  # Ensemble models for best quality
FACE_DETECTION_BACKENDS = ["opencv", "mtcnn", "retinaface"]  # Multiple detection backends for comprehensive coverage
FACE_PREPROCESSING_ENABLED = True   # Erweiterte Gesichtsvorverarbeitung aktivieren
FACE_QUALITY_VALIDATION = True     # Embedding-Qualität validieren
FACE_SIMILARITY_ALGORITHM = "premium"  # "basic", "enhanced" oder "premium" für beste Genauigkeit
FACE_ENSEMBLE_WEIGHTING = True     # Use weighted ensemble of multiple models for maximum quality
FACE_ALIGNMENT_ENABLED = True      # Enable face alignment for better embeddings

# Advanced similarity settings - Tuned for quality over speed  
SIMILARITY_THRESHOLD = 0.25  # Further lowered for better recall with premium algorithms
ENSEMBLE_SIMILARITY_WEIGHTS = {
    'cosine': 0.35,         # Reduced primary weight to balance ensemble
    'euclidean': 0.25,      # Secondary metric
    'correlation': 0.15,    # Tertiary metric
    'angular': 0.10,        # Quaternary metric
    'manhattan': 0.10,      # Enhanced weight for additional coverage
    'chebyshev': 0.05       # Additional metric for fine-tuning
}

# Model performance settings - Optimized for quality
EMBEDDING_CACHE_SIZE = 2000  # Larger cache for better performance
BATCH_PROCESSING_ENABLED = True
PARALLEL_MODEL_EXTRACTION = True  # Enable parallel processing for ensemble approach

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