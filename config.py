"""
Configuration settings for the Face Recognition Application
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent

# Optional .env in the project root. Real environment variables take
# precedence, and a missing/unreadable file simply leaves the defaults intact.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except Exception:  # noqa: BLE001 - optional convenience, never a startup blocker
    pass

# FACE_DATA_DIR lets tests (and multi-instance setups) keep user data outside
# the code directory without changing the defaults.
DATA_DIR = Path(os.environ.get("FACE_DATA_DIR", BASE_DIR / "data")).expanduser()
IMAGES_DIR = DATA_DIR / "images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SCRAPED_DIR = DATA_DIR / "scraped"

# Create directories if they don't exist
for directory in [DATA_DIR, IMAGES_DIR, EMBEDDINGS_DIR, SCRAPED_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Warnung: Verzeichnis {directory} konnte nicht angelegt werden: {exc}")

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
SIMILARITY_THRESHOLD = 0.70  # Facenet512 cosine baseline; lower values increase false positives

# Identity grouping. Facenet512 uses a cosine verification threshold around
# 0.70 (DeepFace's 0.30 cosine distance); 0.68 is slightly more permissive and
# is meant for reviewed group suggestions, not blind assignment.
FACE_GROUPING_THRESHOLD = float(os.environ.get("FACE_GROUPING_THRESHOLD", "0.68"))
FACE_GROUPING_MARGIN = float(os.environ.get("FACE_GROUPING_MARGIN", "0.04"))
FACE_AUTO_ASSIGN_THRESHOLD = float(os.environ.get("FACE_AUTO_ASSIGN_THRESHOLD", "0.68"))
FACE_AUTO_ASSIGN_AMBIGUITY_MARGIN = 0.05

# Face Recognition Quality Settings - Enhanced for better accuracy
FACE_EMBEDDING_MODEL = "Facenet512"  # Primary DeepFace model for highest quality
# Retained for compatibility with older integrations. The active pipeline uses
# FACE_EMBEDDING_MODEL only; it does not combine these candidates.
FACE_EMBEDDING_MODELS = ["Facenet512", "ArcFace", "VGG-Face", "Facenet"]
# Bump this whenever the detector, crop or embedding preprocessing changes.
# Similarity comparisons across pipeline versions are not meaningful.
EMBEDDING_PIPELINE_VERSION = "facenet512-retinaface-aligned-rgb-base-exif-v5"
FACE_DETECTION_BACKENDS = ["retinaface", "mtcnn", "opencv"]  # Accuracy-first fallback order
# Deprecated compatibility flags; the active pipeline uses DeepFace-aligned
# crops, base preprocessing, and cosine search regardless of these values.
FACE_PREPROCESSING_ENABLED = False
FACE_QUALITY_VALIDATION = False
FACE_SIMILARITY_ALGORITHM = "cosine"
FACE_ENSEMBLE_WEIGHTING = False
FACE_ALIGNMENT_ENABLED = True

# Advanced similarity settings
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
# The active Facenet512 model produces 512 values per face.
EMBEDDING_DIMENSION = 512

# UI settings
RESULTS_PER_PAGE = 100  # 100 pro Seite rendert schnell, ohne 500 Bilder je Rerun
THUMBNAIL_SIZE = (240, 240)  # 240px reichen für die Übersicht und sind viel schneller
