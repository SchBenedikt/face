#!/usr/bin/env python3
"""
Quick script to process all scraped images and populate the database
"""
import asyncio
import logging
from pathlib import Path
from face_recognition_engine import FaceRecognitionEngine
from vector_store import FaceVectorStore
from image_scraper import advanced_scraper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_process_all():
    """Quickly process all scraped images and add faces to database"""
    
    print("🚀 Quick Processing All Scraped Images")
    print("=" * 50)
    
    # Initialize systems
    face_engine = FaceRecognitionEngine()
    vector_store = FaceVectorStore()
    
    # Get all scraped images
    scraped_images = advanced_scraper.get_scraped_images()
    print(f"📸 Found {len(scraped_images)} scraped images")
    
    if not scraped_images:
        print("❌ No scraped images found!")
        return
    
    # Get current database stats
    stats_before = vector_store.get_collection_stats()
    faces_before = stats_before.get("total_faces", 0)
    print(f"📊 Database currently contains {faces_before} faces")
    
    # Process in larger batches for speed
    batch_size = 25
    total_faces_added = 0
    processed_images = 0
    
    print(f"⚙️ Processing in batches of {batch_size}...")
    
    for i in range(0, len(scraped_images), batch_size):
        batch = scraped_images[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(scraped_images) - 1) // batch_size + 1
        
        print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} images)...")
        
        try:
            # Process batch
            results = face_engine.process_images_batch(batch, max_workers=1)
            
            # Prepare data for database
            face_ids = []
            embeddings = []
            metadatas = []
            batch_faces = 0
            
            for result in results:
                if "error" not in result and result["face_count"] > 0:
                    processed_images += 1
                    for face_data in result["faces"]:
                        face_ids.append(face_data["face_id"])
                        embeddings.append(face_data["embedding"])
                        
                        # Convert location to string
                        location = face_data["location"]
                        location_str = f"{location[0]},{location[1]},{location[2]},{location[3]}"
                        
                        metadatas.append({
                            "image_path": result["image_path"],
                            "location": location_str,
                            "face_id": face_data["face_id"]
                        })
                        batch_faces += 1
            
            # Add to database
            if face_ids:
                added_count = vector_store.add_face_embeddings_batch(face_ids, embeddings, metadatas)
                total_faces_added += added_count
                print(f"  ✅ Batch {batch_num}: {batch_faces} faces found, {added_count} added to database")
            else:
                print(f"  ℹ️ Batch {batch_num}: No faces found")
                
        except Exception as e:
            print(f"  ❌ Batch {batch_num} error: {e}")
            continue
    
    # Final statistics
    stats_after = vector_store.get_collection_stats()
    faces_after = stats_after.get("total_faces", 0)
    
    print("\n" + "=" * 50)
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"📊 Images processed: {processed_images}/{len(scraped_images)}")
    print(f"👥 Faces found: {total_faces_added}")
    print(f"📈 Database before: {faces_before} faces")
    print(f"📈 Database after: {faces_after} faces")
    print(f"➕ Net increase: {faces_after - faces_before} faces")
    
    if faces_after > faces_before:
        print(f"\n✨ Success! The database now contains {faces_after} faces from {stats_after.get('unique_images', 'unknown')} images")
        print("🔍 You can now use the Face Search and Face Gallery features!")
    else:
        print("\n⚠️ No new faces were added. This might be because:")
        print("   - The images don't contain detectable faces")
        print("   - The faces are too small (< 30x30 pixels)")
        print("   - The faces were already processed")

if __name__ == "__main__":
    quick_process_all()