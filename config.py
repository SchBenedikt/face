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

# Face recognition settings
FACE_RECOGNITION_MODEL = "cnn"  # "hog" (schneller) oder "cnn" (genauer)
FACE_RECOGNITION_TOLERANCE = 0.6
MIN_FACE_SIZE = (30, 30)  # Kleinere Gesichter zulassen
MAX_FACE_SIZE = (None, None)  # Keine Begrenzung für Gesichtsgröße - maximale Qualität

# Model comparison:
# HOG: Schneller, weniger Ressourcen, gut für Echtzeit
# CNN: Genauer, mehr Ressourcen, besser für Qualität

# Vector database settings
VECTOR_DB_PATH = str(DATA_DIR / "face_vectors.db")
COLLECTION_NAME = "face_embeddings"
SIMILARITY_THRESHOLD = 0.4  # Verbesserte Threshold für bessere Gesichtserkennung (0.4 = 40% Ähnlichkeit)

# Face Recognition Quality Settings - Enhanced for better accuracy
FACE_EMBEDDING_MODEL = "Facenet512"  # Primary DeepFace model for highest quality
FACE_EMBEDDING_MODELS = ["Facenet512", "ArcFace", "VGG-Face", "Facenet"]  # Ensemble models
FACE_DETECTION_BACKENDS = ["opencv", "mtcnn", "retinaface"]  # Multiple detection backends
FACE_PREPROCESSING_ENABLED = True   # Erweiterte Gesichtsvorverarbeitung aktivieren
FACE_QUALITY_VALIDATION = True     # Embedding-Qualität validieren
FACE_SIMILARITY_ALGORITHM = "enhanced"  # "basic", "enhanced" oder "premium" für bessere Genauigkeit
FACE_ENSEMBLE_WEIGHTING = True     # Use weighted ensemble of multiple models
FACE_ALIGNMENT_ENABLED = True      # Enable face alignment for better embeddings

# Advanced similarity settings
SIMILARITY_THRESHOLD = 0.35  # Lowered threshold for better recall with ensemble approach
ENSEMBLE_SIMILARITY_WEIGHTS = {
    'cosine': 0.5,
    'euclidean': 0.25, 
    'correlation': 0.15,
    'angular': 0.1
}

# Model performance settings
EMBEDDING_CACHE_SIZE = 1000  # Cache embeddings for faster processing
BATCH_PROCESSING_ENABLED = True
PARALLEL_MODEL_EXTRACTION = False  # Set to False to avoid memory issues

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