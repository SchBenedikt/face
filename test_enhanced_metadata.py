#!/usr/bin/env python3
"""
Test script for enhanced image metadata functionality
Demonstrates the dual URL tracking feature
"""

import tempfile
import json
from pathlib import Path
from image_metadata_utils import (
    add_image_metadata, 
    get_image_metadata, 
    get_metadata_stats,
    update_source_page_urls,
    migrate_old_metadata
)

def test_enhanced_metadata():
    """Test the enhanced metadata functionality"""
    print("🧪 Testing Enhanced Image Metadata System")
    print("="*50)
    
    # Create temporary metadata file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        metadata_file = f.name
    
    try:
        # Test cases simulating Oxford High School example
        test_cases = [
            {
                'local_path': '/data/images/oxfordhigh_gdst_net/oxford_school_photo_001.jpg',
                'image_url': 'https://twk-media-offload.s3.eu-west-1.amazonaws.com/oxfordhigh/images/school-photo-2023.jpg',
                'source_page_url': 'https://oxfordhigh.gdst.net/school-life/galleries/autumn-term-2023',
                'website': 'oxfordhigh_gdst_net'
            },
            {
                'local_path': '/data/images/oxfordhigh_gdst_net/sports_day_002.jpg', 
                'image_url': 'https://s3.amazonaws.com/oxford-assets/sports/sports-day-action.jpg',
                'source_page_url': 'https://oxfordhigh.gdst.net/news/sports-day-2023-results',
                'website': 'oxfordhigh_gdst_net'
            },
            {
                'local_path': '/data/images/example_site/regular_image.jpg',
                'image_url': 'https://example.com/images/regular-image.jpg',
                'source_page_url': 'https://example.com/gallery/page1',
                'website': 'example_com'
            }
        ]
        
        # Test 1: Add enhanced metadata
        print("1️⃣ Testing metadata addition...")
        for i, case in enumerate(test_cases):
            success = add_image_metadata(
                local_path=case['local_path'],
                image_url=case['image_url'],
                source_page_url=case['source_page_url'],
                website=case['website'],
                file_size=1024 * (i + 1),  # Simulate different file sizes
                metadata_file=metadata_file
            )
            print(f"   ✅ Case {i+1}: {'Success' if success else 'Failed'}")
        
        # Test 2: Retrieve metadata
        print("\n2️⃣ Testing metadata retrieval...")
        for i, case in enumerate(test_cases):
            metadata = get_image_metadata(case['local_path'], metadata_file)
            if metadata:
                print(f"   ✅ Case {i+1}: Retrieved metadata")
                print(f"      📸 Image URL: {metadata.get('image_url', 'N/A')}")
                print(f"      🌐 Source Page: {metadata.get('source_page_url', 'N/A')}")
                print(f"      🏠 Website: {metadata.get('website', 'N/A')}")
            else:
                print(f"   ❌ Case {i+1}: No metadata found")
        
        # Test 3: Statistics
        print("\n3️⃣ Testing statistics...")
        stats = get_metadata_stats(metadata_file)
        print(f"   📊 Total images: {stats['total_images']}")
        print(f"   🌐 Websites: {stats['websites']}")
        print(f"   🔗 Images with source pages: {stats['has_source_pages']}")
        print(f"   📈 Source page coverage: {stats['source_page_coverage']:.1f}%")
        
        # Test 4: Batch update source page URLs
        print("\n4️⃣ Testing batch URL updates...")
        url_mapping = {
            'https://new-cdn.example.com/image1.jpg': 'https://example.com/new-page',
            test_cases[0]['image_url']: 'https://oxfordhigh.gdst.net/updated-gallery-page'
        }
        updated_count = update_source_page_urls(url_mapping, metadata_file)
        print(f"   🔄 Updated {updated_count} entries")
        
        # Test 5: Migration from old format
        print("\n5️⃣ Testing migration from old metadata format...")
        old_metadata = {
            '/old/path/image1.jpg': {
                'source_url': 'https://old-site.com/image1.jpg',
                'website': 'old_site',
                'download_date': '2023-01-01 12:00:00'
            },
            '/old/path/image2.jpg': 'https://old-site.com/image2.jpg'  # Very old format
        }
        
        # Create temporary file for migration test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            migration_file = f.name
        
        migrated_count = migrate_old_metadata(old_metadata, migration_file)
        print(f"   📦 Migrated {migrated_count} entries")
        
        # Verify migration
        migrated_stats = get_metadata_stats(migration_file)
        print(f"   📊 Migrated file has {migrated_stats['total_images']} images")
        
        print("\n✅ All tests completed successfully!")
        print("\n📋 Key Features Demonstrated:")
        print("   • Dual URL tracking (direct image + source page)")
        print("   • Website identification and categorization") 
        print("   • Metadata statistics and coverage reporting")
        print("   • Batch URL updates for existing entries")
        print("   • Migration from old metadata formats")
        print("   • AWS S3 and CDN URL handling")
        
        # Show example of what the enhanced popup would display
        print("\n🎯 Example Frontend Display:")
        example_metadata = get_image_metadata(test_cases[0]['local_path'], metadata_file)
        if example_metadata:
            print(f"   📸 Direct Image URL: {example_metadata['image_url']}")
            print(f"   🌐 Found on Page: {example_metadata['source_page_url']}")
            print(f"   🏠 Website: {example_metadata['website']}")
            print(f"   📅 Downloaded: {example_metadata['download_date']}")
            
        # Clean up
        Path(migration_file).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    finally:
        # Clean up
        Path(metadata_file).unlink(missing_ok=True)
    
    return True

if __name__ == "__main__":
    success = test_enhanced_metadata()
    exit(0 if success else 1)