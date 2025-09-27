#!/usr/bin/env python3
"""
Enhanced Image Metadata Utilities

Handles enhanced image metadata including both direct image URLs and source page URLs.
This addresses the issue where images from AWS S3 (or other CDNs) need to track both
the direct image URL and the page where the image was originally found.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse, urljoin


def add_image_metadata(
    local_path: str,
    image_url: str,
    source_page_url: Optional[str] = None,
    website: Optional[str] = None,
    download_date: Optional[str] = None,
    file_size: Optional[int] = None,
    metadata_file: str = "data/images/image_metadata.json"
) -> bool:
    """
    Add enhanced image metadata with both image URL and source page URL.
    
    Args:
        local_path: Local file path where image is stored
        image_url: Direct URL to the image (e.g., amazonaws.com URL)
        source_page_url: URL of the page where the image was found (e.g., oxford high school page)
        website: Website name/identifier
        download_date: When the image was downloaded
        file_size: Size of the downloaded file in bytes
        metadata_file: Path to the metadata JSON file
        
    Returns:
        bool: True if metadata was successfully added/updated
    """
    try:
        # Ensure metadata directory exists
        metadata_path = Path(metadata_file)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                metadata = {}
        
        # Create enhanced metadata entry
        entry = {
            'image_url': image_url,  # Direct URL (e.g., amazonaws.com)
            'source_page_url': source_page_url,  # Page where image was found
            'website': website or _extract_website_from_url(source_page_url or image_url),
            'download_date': download_date or time.strftime('%Y-%m-%d %H:%M:%S'),
            'file_size': file_size,
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Update metadata
        metadata[local_path] = entry
        
        # Save updated metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Error adding image metadata: {e}")
        return False


def get_image_metadata(local_path: str, metadata_file: str = "data/images/image_metadata.json") -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific image.
    
    Args:
        local_path: Local file path of the image
        metadata_file: Path to the metadata JSON file
        
    Returns:
        Dict with image metadata or None if not found
    """
    try:
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        return metadata.get(local_path)
        
    except Exception:
        return None


def get_metadata_stats(metadata_file: str = "data/images/image_metadata.json") -> Dict[str, Any]:
    """
    Get statistics about the metadata collection.
    
    Args:
        metadata_file: Path to the metadata JSON file
        
    Returns:
        Dict with statistics
    """
    try:
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            return {'total_images': 0, 'websites': [], 'has_source_pages': 0}
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        websites = set()
        has_source_pages = 0
        
        for entry in metadata.values():
            if entry.get('website'):
                websites.add(entry['website'])
            if entry.get('source_page_url'):
                has_source_pages += 1
        
        return {
            'total_images': len(metadata),
            'websites': sorted(list(websites)),
            'has_source_pages': has_source_pages,
            'source_page_coverage': (has_source_pages / len(metadata) * 100) if metadata else 0
        }
        
    except Exception:
        return {'total_images': 0, 'websites': [], 'has_source_pages': 0}


def update_source_page_urls(url_mapping: Dict[str, str], metadata_file: str = "data/images/image_metadata.json") -> int:
    """
    Batch update source page URLs for existing metadata entries.
    
    Args:
        url_mapping: Dict mapping image URLs to their source page URLs
        metadata_file: Path to the metadata JSON file
        
    Returns:
        Number of entries updated
    """
    try:
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            return 0
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        updated_count = 0
        
        for local_path, entry in metadata.items():
            image_url = entry.get('image_url')
            if image_url in url_mapping and not entry.get('source_page_url'):
                entry['source_page_url'] = url_mapping[image_url]
                entry['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
                updated_count += 1
        
        # Save updated metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return updated_count
        
    except Exception as e:
        print(f"Error updating source page URLs: {e}")
        return 0


def _extract_website_from_url(url: str) -> str:
    """
    Extract website identifier from URL.
    
    Args:
        url: URL to extract website name from
        
    Returns:
        Website identifier string
    """
    if not url:
        return "unknown"
        
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix and convert to safe identifier
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain.replace('.', '_').replace('-', '_')
    except Exception:
        return "unknown"


def migrate_old_metadata(old_metadata: Dict[str, Any], metadata_file: str = "data/images/image_metadata.json") -> int:
    """
    Migrate old metadata format to new enhanced format.
    
    Args:
        old_metadata: Dictionary with old metadata format
        metadata_file: Path to the new metadata JSON file
        
    Returns:
        Number of entries migrated
    """
    try:
        migrated_count = 0
        
        for local_path, entry in old_metadata.items():
            # Handle both old formats
            if isinstance(entry, dict):
                image_url = entry.get('source_url') or entry.get('url')
                website = entry.get('website')
                download_date = entry.get('download_date')
                file_size = entry.get('file_size')
            else:
                # Very old format where entry was just the URL
                image_url = entry
                website = None
                download_date = None
                file_size = None
            
            if image_url:
                success = add_image_metadata(
                    local_path=local_path,
                    image_url=image_url,
                    source_page_url=None,  # Will be populated later if available
                    website=website,
                    download_date=download_date,
                    file_size=file_size,
                    metadata_file=metadata_file
                )
                if success:
                    migrated_count += 1
        
        return migrated_count
        
    except Exception as e:
        print(f"Error migrating metadata: {e}")
        return 0


def get_images_by_website(website: str, metadata_file: str = "data/images/image_metadata.json") -> Dict[str, Dict[str, Any]]:
    """
    Get all images from a specific website.
    
    Args:
        website: Website identifier to filter by
        metadata_file: Path to the metadata JSON file
        
    Returns:
        Dict mapping local paths to metadata entries for the specified website
    """
    try:
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            return {}
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return {
            local_path: entry 
            for local_path, entry in metadata.items() 
            if entry.get('website') == website
        }
        
    except Exception:
        return {}