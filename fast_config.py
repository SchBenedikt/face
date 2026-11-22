"""Batch-processing settings shared with the active face pipeline.

This module no longer selects a different detector or embedding model. Batch
indexing must use the same Facenet512 pipeline as interactive search so vectors
remain comparable across both paths.
"""
from config import (
    DATA_DIR,
    EMBEDDINGS_DIR,
    IMAGES_DIR,
    SCRAPED_DIR,
    VECTOR_DB_PATH,
    COLLECTION_NAME,
)

# A modest batch size limits memory use without changing recognition quality.
BATCH_SIZE = 10
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
