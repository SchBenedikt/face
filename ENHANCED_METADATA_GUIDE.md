# Enhanced Image Metadata System

This document explains the new enhanced image metadata functionality that tracks both direct image URLs (like AWS S3) and the source pages where images were found (like Oxford High School pages).

## Problem Solved

Previously, when images were stored on CDNs like Amazon S3 (`amazonaws.com`), the system only stored the direct CDN URL. Users couldn't easily find the original webpage where the image was displayed (e.g., the Oxford High School gallery page).

## Solution: Dual URL Tracking

The enhanced system now stores **both URLs**:

1. **Direct Image URL** - The actual image file location (e.g., `https://s3.amazonaws.com/bucket/image.jpg`)
2. **Source Page URL** - The webpage where the image was found (e.g., `https://oxfordhigh.gdst.net/gallery/`)

## New Features

### 1. Enhanced Metadata Storage (`image_metadata_utils.py`)

```python
from image_metadata_utils import add_image_metadata

# Store both URLs
add_image_metadata(
    local_path="/data/images/school_photo.jpg",
    image_url="https://s3.amazonaws.com/oxford-assets/photo.jpg",  # Direct URL
    source_page_url="https://oxfordhigh.gdst.net/galleries/2023",  # Source page
    website="oxfordhigh_gdst_net"
)
```

### 2. Crawler Improvements

The crawlers (`image.py` and `image-download.py`) now:
- Track which page each image was found on
- Store the page-to-image relationship
- Automatically detect CDN vs direct URLs

### 3. Frontend Enhancements (`app.py`)

Info popups now display:
- **📸 Direct Image URL** with click-to-open functionality
- **🌐 Source Page URL** to navigate back to original content
- Website information and download metadata

## Usage Examples

### For Developers

```python
# Get metadata for an image
from image_metadata_utils import get_image_metadata

metadata = get_image_metadata("/path/to/image.jpg")
print(f"Direct URL: {metadata['image_url']}")
print(f"Found on: {metadata['source_page_url']}")
```

### For End Users

1. **In Face Gallery**: Click "ℹ️ Infos" on any face
2. **View URLs Tab**: See both the direct image URL and source page
3. **Click Links**: Open either the image file or the original webpage

## Migration Support

Existing metadata is automatically supported:

```python
from image_metadata_utils import migrate_old_metadata

# Migrate from old format
old_data = {"/path/image.jpg": {"source_url": "https://example.com/img.jpg"}}
migrate_old_metadata(old_data, "new_metadata.json")
```

## Configuration

### Metadata File Location

By default, metadata is stored in: `data/images/image_metadata.json`

### Website Classification

The system automatically identifies websites and CDNs:
- `amazonaws.com` → AWS S3 CDN
- `cloudfront.net` → AWS CloudFront
- `oxfordhigh.gdst.net` → Oxford High School

## Benefits

1. **Better Attribution**: Users can see both the image and its context
2. **Improved Navigation**: Easy to go back to the original webpage
3. **Compliance**: Better tracking for terms of service compliance
4. **Debugging**: Easier to trace image sources and issues
5. **User Experience**: More informative and useful interface

## Testing

Run the test suite to verify functionality:

```bash
python3 test_enhanced_metadata.py
python3 demo_enhanced_crawler.py
```

## API Reference

### `add_image_metadata()`

Stores enhanced metadata for an image.

**Parameters:**
- `local_path`: Where the image is stored locally
- `image_url`: Direct URL to the image file
- `source_page_url`: URL of the page where image was found
- `website`: Website identifier
- `download_date`: When the image was downloaded
- `file_size`: Size of the image file

### `get_image_metadata()`

Retrieves metadata for a specific image.

**Returns:** Dictionary with image metadata or `None`

### `get_metadata_stats()`

Get statistics about your image collection.

**Returns:**
- `total_images`: Number of images with metadata
- `websites`: List of websites in collection
- `has_source_pages`: Count of images with source page URLs
- `source_page_coverage`: Percentage of images with source pages

## Backward Compatibility

- Existing code continues to work unchanged
- Old metadata formats are automatically supported
- No breaking changes to current functionality
- `fast_process.py` and other tools work normally

## Troubleshooting

### Missing Source Page URLs

If some images don't have source page URLs, they were likely:
1. Downloaded before the enhancement
2. Found through directory brute-force scanning
3. Loaded from existing collections

Use `update_source_page_urls()` to add missing source pages.

### Performance

The enhanced metadata adds minimal overhead:
- ~100 bytes per image for metadata
- No impact on face detection performance
- Metadata loading is cached and optimized

## Future Extensions

The metadata system is designed to be extensible for:
- Additional URL types (thumbnails, variants)
- More detailed page context (titles, descriptions)
- Integration with external metadata services
- Enhanced search and filtering capabilities