#!/usr/bin/env python3
"""
Demo script showing the enhanced crawler functionality
Simulates crawling Oxford High School site and tracking both direct image URLs and source pages
"""

import tempfile
import json
from pathlib import Path
from image_metadata_utils import add_image_metadata, get_metadata_stats, get_image_metadata

def simulate_enhanced_crawling():
    """Simulate the enhanced crawling process with dual URL tracking"""
    print("🌐 Enhanced Image Crawler Demo")
    print("Simulating crawl of Oxford High School website")
    print("="*60)
    
    # Create temporary metadata file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        metadata_file = f.name
    
    try:
        # Simulate the crawling process
        crawl_results = [
            {
                'page_url': 'https://oxfordhigh.gdst.net/school-life/galleries/autumn-term-2023',
                'images_found': [
                    'https://twk-media-offload.s3.eu-west-1.amazonaws.com/oxfordhigh/images/autumn-concert-2023.jpg',
                    'https://twk-media-offload.s3.eu-west-1.amazonaws.com/oxfordhigh/images/sports-day-group.jpg',
                    'https://oxfordhigh.gdst.net/wp-content/uploads/2023/10/school-logo.png'
                ]
            },
            {
                'page_url': 'https://oxfordhigh.gdst.net/news/academic-achievements-2023', 
                'images_found': [
                    'https://twk-media-offload.s3.eu-west-1.amazonaws.com/oxfordhigh/images/awards-ceremony.jpg',
                    'https://s3.amazonaws.com/oxford-gdst-assets/academic/graduation-2023.jpg'
                ]
            },
            {
                'page_url': 'https://oxfordhigh.gdst.net/admissions/virtual-tour',
                'images_found': [
                    'https://twk-media-offload.s3.eu-west-1.amazonaws.com/oxfordhigh/tour/library-view.jpg',
                    'https://twk-media-offload.s3.eu-west-1.amazonaws.com/oxfordhigh/tour/science-lab.jpg',
                    'https://oxfordhigh.gdst.net/assets/images/campus-aerial.jpg'
                ]
            }
        ]
        
        print("🕷️ Crawling pages and tracking image sources...")
        total_images = 0
        
        for page_result in crawl_results:
            page_url = page_result['page_url']
            images = page_result['images_found']
            
            print(f"\n📄 Processing page: {page_url}")
            print(f"   Found {len(images)} images:")
            
            for i, image_url in enumerate(images):
                # Simulate download and metadata storage
                local_path = f"/data/images/oxfordhigh_gdst_net/image_{total_images + i + 1:03d}.jpg"
                
                # Determine if this is a CDN/S3 URL or direct website URL
                is_cdn = any(cdn in image_url.lower() for cdn in ['s3.', 'amazonaws.com', 'cloudfront.net', 'cdn.'])
                
                if is_cdn:
                    print(f"   🌩️  CDN Image: {image_url}")
                    print(f"      📍 Found on: {page_url}")
                else:
                    print(f"   🖼️  Direct Image: {image_url}")
                    print(f"      📍 Found on: {page_url}")
                
                # Add enhanced metadata
                success = add_image_metadata(
                    local_path=local_path,
                    image_url=image_url,
                    source_page_url=page_url,
                    website='oxfordhigh_gdst_net',
                    file_size=1024 * (100 + i * 50),  # Simulate varying file sizes
                    metadata_file=metadata_file
                )
                
                if success:
                    print(f"      ✅ Metadata saved: {Path(local_path).name}")
                else:
                    print(f"      ❌ Failed to save metadata")
            
            total_images += len(images)
        
        print(f"\n📊 Crawling completed! Processed {total_images} images")
        
        # Show statistics
        print("\n📈 Metadata Statistics:")
        stats = get_metadata_stats(metadata_file)
        print(f"   • Total images: {stats['total_images']}")
        print(f"   • Websites: {', '.join(stats['websites'])}")
        print(f"   • Images with source pages: {stats['has_source_pages']}")
        print(f"   • Source page coverage: {stats['source_page_coverage']:.1f}%")
        
        # Show examples of the enhanced metadata
        print("\n🎯 Example Enhanced Metadata (Frontend would display):")
        print("-" * 50)
        
        # Load the actual metadata file to show examples
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for i, (local_path, entry) in enumerate(list(metadata.items())[:3]):
            print(f"\n📸 Image {i+1}: {Path(local_path).name}")
            print(f"   🔗 Direct URL: {entry['image_url']}")
            print(f"   🌐 Found on Page: {entry['source_page_url']}")
            print(f"   🏠 Website: {entry['website']}")
            print(f"   📅 Downloaded: {entry['download_date']}")
            print(f"   💾 File Size: {entry.get('file_size', 'N/A')} bytes")
            
            # Show what type of URL this is
            if 's3.' in entry['image_url'] or 'amazonaws.com' in entry['image_url']:
                print("   ☁️  Type: AWS S3/CloudFront CDN")
            elif 'cdn.' in entry['image_url']:
                print("   🌐 Type: Content Delivery Network")
            else:
                print("   🏠 Type: Direct website image")
        
        print("\n✨ Key Benefits of Enhanced Metadata:")
        print("   • Users can see both the direct image URL (AWS S3) AND the page where it was found")
        print("   • Better attribution and source tracking")
        print("   • Easier to navigate back to original content")
        print("   • Improved compliance with website terms of service")
        print("   • Enhanced debugging and image source verification")
        
        print("\n🖼️ Frontend Integration:")
        print("   • Info popups now show both URLs with clear labels")
        print("   • Users can click to open either the direct image or source page")
        print("   • Scraped images gallery shows source page information")
        print("   • Enhanced tooltips and expandable sections for metadata")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        return False
    finally:
        # Clean up
        Path(metadata_file).unlink(missing_ok=True)
    
    return True

if __name__ == "__main__":
    success = simulate_enhanced_crawling()
    print(f"\n{'🎉 Demo completed successfully!' if success else '❌ Demo failed!'}")
    exit(0 if success else 1)