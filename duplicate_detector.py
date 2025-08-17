"""
Simple duplicate detector for image files
"""
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)

class SimpleDuplicateDetector:
    """Simple duplicate detector using file hashes"""
    
    def __init__(self, db_path: str = "duplicates.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the hash database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_hashes (
                    file_path TEXT PRIMARY KEY,
                    hash_value TEXT NOT NULL,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def is_duplicate(self, file_path: Path) -> bool:
        """Check if file is a duplicate"""
        if not file_path.exists():
            return False
        
        file_hash = self.calculate_file_hash(file_path)
        if not file_hash:
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM file_hashes WHERE hash_value = ?", (file_hash,))
            count = cursor.fetchone()[0]
            conn.close()
            
            if count == 0:
                # Add to database
                self.add_file_hash(file_path, file_hash)
                return False
            else:
                return True
        except Exception as e:
            logger.error(f"Error checking duplicate for {file_path}: {e}")
            return False
    
    def add_file_hash(self, file_path: Path, file_hash: str | None = None):
        """Add file hash to database"""
        if file_hash is None:
            file_hash = self.calculate_file_hash(file_path)
        
        if not file_hash:
            return
        
        try:
            file_size = file_path.stat().st_size
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO file_hashes (file_path, hash_value, file_size)
                VALUES (?, ?, ?)
            """, (str(file_path), file_hash, file_size))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error adding hash for {file_path}: {e}")
    
    def find_duplicates_in_directory(self, directory: Path, use_content_hash: bool = True) -> Dict[str, List[Path]]:
        """Find duplicate files in directory"""
        duplicates = {}
        hash_to_files = {}
        
        if not directory.exists():
            return duplicates
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        # Calculate hashes and group files
        for file_path in image_files:
            if file_path.is_file():
                file_hash = self.calculate_file_hash(file_path)
                if file_hash:
                    if file_hash not in hash_to_files:
                        hash_to_files[file_hash] = []
                    hash_to_files[file_hash].append(file_path)
        
        # Find duplicates (hashes with multiple files)
        for file_hash, file_list in hash_to_files.items():
            if len(file_list) > 1:
                duplicates[file_hash] = file_list
        
        return duplicates
    
    def get_stats(self) -> Dict:
        """Get duplicate detector statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM file_hashes")
            total_files = cursor.fetchone()[0]
            conn.close()
            
            db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
            
            return {
                "total_files": total_files,
                "hash_algorithm": "sha256",
                "database_size": db_size
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_files": 0, "hash_algorithm": "sha256", "database_size": 0}
    
    def cleanup_missing_files(self) -> int:
        """Remove entries for files that no longer exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM file_hashes")
            all_paths = cursor.fetchall()
            
            removed_count = 0
            for (file_path,) in all_paths:
                if not Path(file_path).exists():
                    cursor.execute("DELETE FROM file_hashes WHERE file_path = ?", (file_path,))
                    removed_count += 1
            
            conn.commit()
            conn.close()
            return removed_count
        except Exception as e:
            logger.error(f"Error cleaning up missing files: {e}")
            return 0
    
    @property
    def hash_database(self):
        """For compatibility - returns a simple dict-like object"""
        return {"clear": self.clear_database}
    
    def clear_database(self):
        """Clear all hash data"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM file_hashes")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
    
    def save_hash_database(self):
        """For compatibility - database is already saved"""
        pass

# Create singleton instance
duplicate_detector = SimpleDuplicateDetector()