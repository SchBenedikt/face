#!/usr/bin/env python3
"""
Image Quality Manager

Handles duplicate images with different sizes/quality and automatically selects
the best version to keep. This solves the problem where the same image exists
in multiple sizes (e.g., -300x200.jpg, -scaled.jpg, -150x150.jpg) leading to 
duplicate faces in recognition results.

Features:
- Detects images with same base name but different size suffixes
- Prioritizes quality: scaled > highest resolution > original
- Can work as post-processing or integrated into download process
- Flexible pattern recognition for various image size formats
- Preserves metadata and handles file cleanup
"""

import re
import json
import shutil
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

class ImageQualityManager:
    """Manages image quality by detecting and removing lower-quality duplicates"""
    
    # Quality priority rules (higher number = better quality)
    QUALITY_PRIORITIES = {
        'scaled': 1000,      # WordPress "scaled" version - usually best
        'original': 900,     # Original file (without size suffix)
        'full': 850,         # Full size
        '2048x': 800,        # High resolution (2048+ width)
        '1536x': 700,        # Medium-high resolution  
        '1024x': 600,        # Medium resolution
        '768x': 500,         # Lower medium resolution
        '400x': 400,         # Small resolution
        '300x': 300,         # Very small resolution
        '150x': 200,         # Thumbnail size
        'thumbnail': 100,    # WordPress thumbnail
        'thumb': 100,        # Thumbnail
        'medium': 250,       # WordPress medium
        'large': 350,        # WordPress large
    }
    
    def __init__(self, image_dir: Optional[Path] = None, metadata_file: Optional[Path] = None):
        """
        Initialize Image Quality Manager
        
        Args:
            image_dir: Directory containing images
            metadata_file: Path to image metadata JSON file
        """
        self.image_dir = image_dir or Path("data/images")
        self.metadata_file = metadata_file or self.image_dir / "image_metadata.json"
        self.metadata = {}
        self.load_metadata()
    
    def load_metadata(self):
        """Load existing image metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded {len(self.metadata)} image metadata entries")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self.metadata = {}
    
    def save_metadata(self) -> bool:
        """Save updated metadata to file"""
        temporary_name = None
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            # Create backup first
            if self.metadata_file.exists():
                backup_file = self.metadata_file.with_suffix('.json.backup')
                try:
                    shutil.copy2(self.metadata_file, backup_file)
                except OSError as backup_exc:
                    logger.warning("Could not create metadata backup: %s", backup_exc)

            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{self.metadata_file.name}.",
                suffix=".tmp",
                dir=self.metadata_file.parent,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.metadata, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.metadata_file)
            temporary_name = None
            logger.info(f"Metadata saved to {self.metadata_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return False
        finally:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
    
    def extract_base_name(self, filename: str) -> str:
        """
        Extract base name by removing size suffixes and scaling indicators
        
        Examples:
            'IMG_3463-300x200.jpg' -> 'IMG_3463.jpg'
            'IMG_6189-scaled.jpg' -> 'IMG_6189.jpg'
            'photo-1536x1024.jpg' -> 'photo.jpg'
            'banner-2048x1365.jpg' -> 'banner.jpg'
        """
        # Handle various size patterns
        patterns = [
            r'-(\d+x\d+)',           # -300x200, -1024x768, etc.
            r'-scaled',              # -scaled
            r'-thumbnail',           # -thumbnail
            r'-thumb',               # -thumb  
            r'-medium',              # -medium
            r'-large',               # -large
            r'-full',                # -full
            r'-(\d+)x(\d+)',         # Alternative digit pattern
            r'-(\d+w)',              # -300w (width)
            r'-(\d+h)',              # -200h (height)
        ]
        
        base_name = filename
        for pattern in patterns:
            base_name = re.sub(pattern, '', base_name, flags=re.IGNORECASE)
        
        return base_name
    
    def calculate_quality_score(self, filename: str) -> int:
        """
        Calculate quality score for a filename based on size indicators
        
        Args:
            filename: Image filename to analyze
            
        Returns:
            Quality score (higher = better quality)
        """
        filename_lower = filename.lower()
        
        # Check for explicit quality indicators
        for indicator, score in self.QUALITY_PRIORITIES.items():
            if indicator in filename_lower:
                # For resolution-based scores, also consider the actual numbers
                if 'x' in indicator and indicator.replace('x', '').isdigit():
                    continue  # Handle resolution separately
                return score
        
        # Handle resolution-based scoring
        resolution_match = re.search(r'(\d+)x(\d+)', filename)
        if resolution_match:
            width = int(resolution_match.group(1))
            height = int(resolution_match.group(2))
            total_pixels = width * height
            
            # Score based on total pixels
            if total_pixels >= 4000000:    # 2048x1953+ (4MP+)
                return 850
            elif total_pixels >= 2000000:  # 1536x1300+ (2MP+)
                return 750
            elif total_pixels >= 700000:   # 1024x683+ (700K+)
                return 650
            elif total_pixels >= 400000:   # 768x521+ (400K+)  
                return 550
            elif total_pixels >= 120000:   # 400x300+ (120K+)
                return 450
            elif total_pixels >= 60000:    # 300x200+ (60K+)
                return 350
            else:                          # Smaller than 300x200
                return 250
        
        # If no size indicator found, assume it's original quality
        return 900
    
    def group_images_by_base_name(self, file_paths: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Group images by their base name (without size suffixes)
        
        Args:
            file_paths: Optional list of file paths. If None, uses all metadata entries
            
        Returns:
            Dictionary mapping base names to lists of full paths
        """
        if file_paths is None:
            file_paths = list(self.metadata.keys())
        
        groups = {}
        
        for file_path in file_paths:
            # Extract just the filename from the path
            path_obj = Path(file_path)
            filename = path_obj.name
            
            # Remove website prefix if present (e.g., "oxfordhigh_gdst_net_")
            clean_filename = re.sub(r'^[^_]+_[^_]+_net_', '', filename)
            
            base_name = self.extract_base_name(clean_filename)
            
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(file_path)
        
        # Only return groups with multiple files (potential duplicates)
        return {k: v for k, v in groups.items() if len(v) > 1}
    
    def select_best_quality_image(self, image_paths: List[str]) -> Tuple[str, List[str]]:
        """
        Select the best quality image from a group of similar images
        
        Args:
            image_paths: List of image paths with same base name
            
        Returns:
            Tuple of (best_image_path, list_of_lower_quality_paths)
        """
        if len(image_paths) <= 1:
            return image_paths[0] if image_paths else "", []
        
        # Score all images
        scored_images = []
        for path in image_paths:
            filename = Path(path).name
            score = self.calculate_quality_score(filename)
            file_size = 0
            
            # Also consider file size from metadata
            if path in self.metadata and 'file_size' in self.metadata[path]:
                file_size = self.metadata[path]['file_size']
            
            scored_images.append((score, file_size, path))
        
        # Sort by score (descending), then by file size (descending)
        scored_images.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        best_image = scored_images[0][2]
        duplicates = [img[2] for img in scored_images[1:]]
        
        return best_image, duplicates
    
    def find_all_duplicates(self) -> Dict[str, Tuple[str, List[str]]]:
        """
        Find byte-identical duplicate images and determine which to keep/remove
        
        Returns:
            Dictionary mapping base_name to (best_image, duplicates_to_remove)
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        files = sorted(
            path for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in image_extensions
        ) if self.image_dir.exists() else []
        by_hash: Dict[str, List[str]] = {}
        for path in files:
            digest = hashlib.sha256()
            try:
                with path.open("rb") as image_file:
                    for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as exc:
                logger.warning("Could not hash image %s: %s", path.name, exc)
                continue
            by_hash.setdefault(digest.hexdigest(), []).append(str(path))

        duplicates_info = {}

        for digest, image_paths in by_hash.items():
            if len(image_paths) > 1:
                best_image, duplicates = self.select_best_quality_image(image_paths)
                duplicates_info[digest] = (best_image, duplicates)
        
        return duplicates_info
    
    def remove_duplicate_files(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Remove duplicate image files, keeping only the best quality version
        
        Args:
            dry_run: If True, only simulate the removal without actually deleting files
            
        Returns:
            Dictionary with removal statistics and details
        """
        duplicates_info = self.find_all_duplicates()
        
        stats = {
            'total_groups': len(duplicates_info),
            'files_to_remove': 0,
            'files_removed': 0,
            'files_to_keep': 0,
            'bytes_to_free': 0,
            'files_removed': 0,
            'removed_files': [],
            'kept_files': [],
            'errors': []
        }
        vector_store = None
        
        for base_name, (best_image, duplicates) in duplicates_info.items():
            stats['files_to_keep'] += 1
            stats['files_to_remove'] += len(duplicates)
            stats['kept_files'].append(best_image)
            
            for duplicate_path in duplicates:
                if dry_run:
                    stats['removed_files'].append(duplicate_path)
                
                # Calculate bytes to free
                try:
                    stats['bytes_to_free'] += Path(duplicate_path).stat().st_size
                except OSError:
                    pass

            if dry_run:
                continue
            try:
                from config import DATA_DIR
                from duplicate_backup import create_duplicate_backup
                from duplicate_detector import delete_duplicate_files
                from vector_store import FaceVectorStore

                if vector_store is None:
                    vector_store = FaceVectorStore()
                backup = create_duplicate_backup(
                    vector_store,
                    duplicates,
                    DATA_DIR / "duplicate_backups",
                )
                if not backup.get("success"):
                    raise RuntimeError(backup.get("message", "Recovery archive failed."))
                result = delete_duplicate_files([best_image, *duplicates], best_image)
                if (
                    not result["deleted"]
                    and result.get("rolled_back")
                    and backup.get("backup_path")
                ):
                    Path(backup["backup_path"]).unlink(missing_ok=True)
                for removed_path in result["deleted"]:
                    vector_store.delete_faces_by_image_path(str(removed_path))
                    self.metadata.pop(str(removed_path), None)
                stats["files_removed"] += len(result["deleted"])
                stats["removed_files"].extend(str(path) for path in result["deleted"])
                if result["deleted"]:
                    vector_store.prune_missing_faces(root_directory=self.image_dir)
                    stats.setdefault("recovery_archives", []).append(
                        str(backup["backup_path"]) if backup.get("backup_path") else ""
                    )
                # Persist every successfully removed file even when a later
                # group fails, so metadata cannot lag disk state.
                if not self.save_metadata():
                    stats["errors"].append("Could not persist updated image metadata")
                stats["errors"].extend(result["errors"])
            except Exception as exc:  # noqa: BLE001 - report partial cleanup
                error_msg = f"Error removing duplicate group: {exc}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        if not dry_run:
            stats['orphan_cleanup'] = self.cleanup_orphaned_metadata()
            
        return stats
    
    def cleanup_orphaned_metadata(self) -> Dict[str, Any]:
        """
        Remove metadata entries for files that no longer exist physically
        
        Returns:
            Dictionary with cleanup statistics
        """
        orphaned_entries = []
        existing_files = 0
        
        for file_path in list(self.metadata.keys()):
            physical_path = Path(file_path)
            if physical_path.exists():
                existing_files += 1
            else:
                orphaned_entries.append(file_path)
        
        # Remove orphaned entries
        for orphaned_path in orphaned_entries:
            del self.metadata[orphaned_path]
        
        # Save updated metadata
        if orphaned_entries:
            self.save_metadata()
            logger.info(f"Removed {len(orphaned_entries)} orphaned metadata entries")
        
        stats = {
            'total_metadata_entries': existing_files + len(orphaned_entries),
            'existing_files': existing_files,
            'orphaned_entries_removed': len(orphaned_entries),
            'orphaned_files': orphaned_entries[:10]  # Show first 10 for debugging
        }
        
        return stats
    
    def generate_report(self) -> str:
        """Generate a detailed report about byte-identical images."""
        duplicates_info = self.find_all_duplicates()
        
        if not duplicates_info:
            return "No byte-identical duplicate images found."
        
        report = ["Image Duplicate Analysis Report", "=" * 50, ""]
        
        total_duplicates = sum(len(duplicates) for _, duplicates in duplicates_info.values())
        total_groups = len(duplicates_info)
        
        report.append("Summary:")
        report.append(f"   {total_groups} groups of byte-identical images found")
        report.append(f"   {total_duplicates} exact duplicate files can be removed")
        report.append("")
        
        # Calculate potential space savings
        total_bytes_to_free = 0
        for _, duplicates in duplicates_info.values():
            for duplicate_path in duplicates:
                try:
                    total_bytes_to_free += Path(duplicate_path).stat().st_size
                except OSError:
                    pass
        
        if total_bytes_to_free > 0:
            mb_to_free = total_bytes_to_free / (1024 * 1024)
            report.append(f"Estimated space savings: {mb_to_free:.2f} MiB")
            report.append("")
        
        # Show detailed groups (first 10)
        report.append("Details (first 10 groups):")
        report.append("")
        
        for i, (_digest, (best_image, duplicates)) in enumerate(duplicates_info.items()):
            if i >= 10:  # Limit to first 10 groups
                report.append(f"   ... and {len(duplicates_info) - 10} more groups")
                break
                
            report.append(f"{i+1}. Exact content match")
            best_filename = Path(best_image).name
            report.append(f"   Keep: {best_filename}")
            for duplicate_path in duplicates:
                dup_filename = Path(duplicate_path).name
                report.append(f"   Remove: {dup_filename}")
            
            report.append("")
        
        return "\n".join(report)
    
    def clean_downloads_directory(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean the downloads directory by removing duplicate images
        
        This is the main method to call for cleaning up downloaded images.
        
        Args:
            dry_run: If True, only simulate without actually removing files
            
        Returns:
            Dictionary with detailed results
        """
        logger.info(f"Starting image quality cleanup (dry_run={dry_run})")
        
        # Generate report first
        report = self.generate_report()
        logger.info("Generated duplicate analysis report")
        
        # Remove duplicates
        removal_stats = self.remove_duplicate_files(dry_run=dry_run)
        
        results = {
            'report': report,
            'removal_stats': removal_stats,
            'dry_run': dry_run
        }
        
        return results


# Standalone functions for easy integration
def clean_image_duplicates(image_dir: str = "data/images", dry_run: bool = True) -> Dict[str, Any]:
    """
    Clean image duplicates in the specified directory
    
    Args:
        image_dir: Path to image directory
        dry_run: If True, only simulate removal
        
    Returns:
        Dictionary with cleanup results
    """
    manager = ImageQualityManager(
        image_dir=Path(image_dir),
        metadata_file=Path(image_dir) / "image_metadata.json"
    )
    
    return manager.clean_downloads_directory(dry_run=dry_run)


def generate_duplicate_report(image_dir: str = "data/images") -> str:
    """
    Generate a report about duplicate images
    
    Args:
        image_dir: Path to image directory
        
    Returns:
        String report
    """
    manager = ImageQualityManager(
        image_dir=Path(image_dir),
        metadata_file=Path(image_dir) / "image_metadata.json"
    )
    
    return manager.generate_report()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    from config import IMAGES_DIR

    parser = argparse.ArgumentParser(
        description="Report or safely remove byte-identical image files."
    )
    parser.add_argument("--image-dir", default=str(IMAGES_DIR), help="Path to the image directory")
    parser.add_argument("--action", choices=["report", "clean"], default="report")
    parser.add_argument("--no-dry-run", action="store_true", help="Perform cleanup after confirmation")
    parser.add_argument("--yes", action="store_true", help="Confirm cleanup non-interactively")
    args = parser.parse_args()

    image_directory = Path(args.image_dir).expanduser()
    if not image_directory.is_dir():
        parser.error(f"image directory does not exist: {image_directory}")

    if args.action == "clean" and args.no_dry_run and not args.yes:
        try:
            response = input("Type DELETE to remove byte-identical duplicates: ").strip()
        except EOFError:
            response = ""
        if response != "DELETE":
            print("Cleanup cancelled.")
            raise SystemExit(0)

    manager = ImageQualityManager(
        image_dir=image_directory,
        metadata_file=image_directory / "image_metadata.json",
    )

    if args.action == 'report':
        print(manager.generate_report())
    elif args.action == 'clean':
        dry_run = not args.no_dry_run
        results = manager.clean_downloads_directory(dry_run=dry_run)
        print(results["report"])
        print("\nCleanup results")
        print("=" * 50)
        stats = results['removal_stats']
        print(f"Groups processed: {stats['total_groups']}")
        print(f"Files to keep: {stats['files_to_keep']}")
        print(f"Duplicate files found: {stats['files_to_remove']}")
        print(f"Files removed: {stats.get('files_removed', 0)}")
        print(f"Estimated space: {stats['bytes_to_free'] / (1024 * 1024):.2f} MiB")
        for backup in stats.get("recovery_archives", []):
            if backup:
                print(f"Recovery archive: {backup}")
        if stats['errors']:
            for error in stats['errors']:
                print(f"Error: {error}")
        if dry_run:
            print("Dry run only; no files were removed. Use --no-dry-run to clean after confirmation.")
    elif args.action == 'test':
        # Test the base name extraction
        test_files = [
            "IMG_3463-300x200.jpg",
            "IMG_3463-scaled.jpg", 
            "IMG_3463-150x150.jpg",
            "IMG_6189-1024x768.jpg",
            "IMG_6189-2048x1536.jpg",
            "Banner-admissions-scaled.jpg",
            "photo-thumbnail.jpg",
            "image-medium.jpg",
            "picture-large.jpg"
        ]
        
        print("Testing base name extraction:")
        print("="*50)
        for filename in test_files:
            base_name = manager.extract_base_name(filename)
            score = manager.calculate_quality_score(filename)
            print(f"{filename:30} -> {base_name:20} (score: {score})")
