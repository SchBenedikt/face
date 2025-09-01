#!/usr/bin/env python3
"""
Fast Batch Face Processing Script
Optimized for maximum speed with decent accuracy
"""

import sys
import time
import logging
from pathlib import Path
from typing import List

# Import fast configuration
from fast_config import *

# Set up fast logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fast_processing.log')
    ]
)

logger = logging.getLogger(__name__)

def get_image_files_fast(directory: Path) -> List[Path]:
    """Fast image file discovery"""
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(directory.rglob(f"*{ext}"))
        image_files.extend(directory.rglob(f"*{ext.upper()}"))
    return image_files

def process_images_fast(update_existing=False):
    """Fast batch processing with minimal overhead"""
    
    print("🚀 FAST FACE PROCESSING MODE")
    if update_existing:
        print("🔄 DATABASE UPDATE MODE - Processing all images")
    print("="*50)
    
    # Find images
    image_files = get_image_files_fast(IMAGES_DIR)
    total_images = len(image_files)
    
    if total_images == 0:
        print(f"❌ No images found in {IMAGES_DIR}")
        return
    
    # Import face engine with fast config
    from face_recognition_engine import FaceRecognitionEngine
    from vector_store import FaceVectorStore
    
    # Initialize with fast settings
    face_engine = FaceRecognitionEngine()
    vector_store = FaceVectorStore()
    
    # Filter already processed images if not updating
    if not update_existing:
        print("🔍 Checking for already processed images...")
        processed_images = set()
        try:
            # Get all existing metadatas to find processed images
            existing_data = vector_store.collection.get()
            if existing_data and existing_data.get('metadatas'):
                for metadata in existing_data['metadatas']:
                    if 'image_path' in metadata:
                        processed_images.add(Path(metadata['image_path']).name)
        except Exception as e:
            print(f"⚠️ Could not check existing images: {e}")
        
        # Filter out processed images
        original_count = len(image_files)
        image_files = [img for img in image_files if img.name not in processed_images]
        skipped = original_count - len(image_files)
        
        if skipped > 0:
            print(f"⏭️ Skipping {skipped} already processed images")
        
        total_images = len(image_files)
        if total_images == 0:
            print("✅ All images already processed!")
            return
    
    print(f"📁 Processing {total_images} images")
    print(f"⚙️ Batch size: {BATCH_SIZE}")
    print(f"🔧 Model: {FACE_EMBEDDING_MODEL} (fast mode)")
    print(f"⏱️ Timeout: {PROCESSING_TIMEOUT}s per image")
    if update_existing:
        print("🔄 Update mode: Will replace existing face data")
    print()
    
    # Process in batches
    total_faces = 0
    total_processed = 0
    total_deleted = 0
    total_errors = 0
    start_time = time.time()
    
    num_batches = (total_images - 1) // BATCH_SIZE + 1
    
    for batch_num in range(num_batches):
        batch_start = time.time()
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_images)
        batch_files = image_files[start_idx:end_idx]
        
        print(f"🔄 Batch {batch_num + 1}/{num_batches}: Processing {len(batch_files)} images...")
        
        # Process batch
        results = face_engine.process_images_batch(batch_files, max_workers=1)
        
        # Collect results
        face_ids = []
        embeddings = []
        metadatas = []
        batch_faces = 0
        batch_errors = 0
        
        for result in results:
            image_name = Path(result.get('image_path', 'unknown')).name
            
            if "error" not in result and result.get("face_count", 0) > 0:
                total_processed += 1
                face_count = result["face_count"]
                batch_faces += face_count
                
                print(f"  ✅ {image_name} → {face_count} face(s)")
                
                for face_data in result["faces"]:
                    face_ids.append(face_data["face_id"])
                    embeddings.append(face_data["embedding"])
                    
                    location = face_data["location"]
                    location_str = f"{location[0]},{location[1]},{location[2]},{location[3]}"
                    
                    metadatas.append({
                        "image_path": result["image_path"],
                        "location": location_str,
                        "face_id": face_data["face_id"]
                    })
                    
            elif "error" in result:
                batch_errors += 1
                total_errors += 1
                error_msg = result.get('error', 'Unknown error')
                print(f"  ❌ {image_name} → ERROR: {error_msg}")
                
            else:
                total_processed += 1
                # No faces detected: delete the image file to remove it completely
                image_path_str = result.get('image_path')
                try:
                    img_path = Path(image_path_str) if image_path_str else None
                    if img_path and not img_path.exists():
                        # Try to resolve relative to IMAGES_DIR
                        img_path = IMAGES_DIR / (Path(image_path_str).name if image_path_str else '')
                    if img_path and img_path.exists():
                        img_path.unlink()
                        total_deleted += 1
                        print(f"  🗑️ {image_name} → No faces detected — image deleted")
                    else:
                        print(f"  ⚠️ {image_name} → No faces detected (file not found to delete)")
                except Exception as e:
                    print(f"  ⚠️ {image_name} → No faces detected (failed to delete): {e}")
        
        # Add to database
        if face_ids:
            if update_existing:
                # In update mode, delete existing faces for this image first
                for result in results:
                    if "error" not in result:
                        image_path = result["image_path"]
                        # Find and delete existing faces from this image
                        try:
                            existing_data = vector_store.collection.get()
                            if existing_data and existing_data.get('metadatas'):
                                faces_to_delete = []
                                for i, metadata in enumerate(existing_data['metadatas']):
                                    if metadata.get('image_path') == image_path:
                                        if i < len(existing_data.get('ids', [])):
                                            faces_to_delete.append(existing_data['ids'][i])
                                
                                if faces_to_delete:
                                    vector_store.collection.delete(ids=faces_to_delete)
                                    print(f"  🗑️ Deleted {len(faces_to_delete)} existing faces from {Path(image_path).name}")
                        except Exception as e:
                            print(f"  ⚠️ Could not delete existing faces: {e}")
            
            added_count = vector_store.add_face_embeddings_batch(face_ids, embeddings, metadatas)
            total_faces += added_count
            print(f"  💾 Added {added_count} faces to database")
        
        # Batch timing
        batch_time = time.time() - batch_start
        avg_time_per_image = batch_time / len(batch_files)
        
        print(f"  ⏱️ Batch completed in {batch_time:.1f}s ({avg_time_per_image:.1f}s/image)")
        print()
    
    # Final statistics
    total_time = time.time() - start_time
    avg_time_per_image = total_time / total_images if total_images > 0 else 0
    
    print("🎉 PROCESSING COMPLETE!")
    print("="*50)
    print(f"📊 Total Images: {total_images}")
    print(f"✅ Successfully Processed: {total_processed}")
    print(f"👥 Total Faces Found: {total_faces}")
    print(f"🗑️ Images deleted (no faces): {total_deleted}")
    print(f"❌ Errors: {total_errors}")
    print(f"⏱️ Total Time: {total_time:.1f}s")
    print(f"🚀 Average per Image: {avg_time_per_image:.1f}s")
    
    if total_processed > 0:
        success_rate = (total_processed / total_images) * 100
        avg_faces_per_image = total_faces / total_processed
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"👤 Avg Faces per Image: {avg_faces_per_image:.2f}")


def process_images_streamlit(update_existing=False, progress_callback=None, status_callback=None):
    """Streamlit-compatible version of fast processing with callbacks"""
    import io
    import sys
    
    # Capture print output for Streamlit
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    
    try:
        # Find images
        image_files = get_image_files_fast(IMAGES_DIR)
        total_images = len(image_files)
        
        if total_images == 0:
            return {"success": False, "message": f"No images found in {IMAGES_DIR}"}
        
        # Import face engine with fast config
        from face_recognition_engine import FaceRecognitionEngine
        from vector_store import FaceVectorStore
        
        # Initialize with fast settings
        face_engine = FaceRecognitionEngine()
        vector_store = FaceVectorStore()
        
        if status_callback:
            status_callback(f"🔍 Found {total_images} images to process")
        
        # Filter already processed images if not updating
        if not update_existing:
            processed_images = set()
            try:
                existing_data = vector_store.collection.get()
                if existing_data and existing_data.get('metadatas'):
                    for metadata in existing_data['metadatas']:
                        if 'image_path' in metadata:
                            processed_images.add(Path(metadata['image_path']).name)
            except Exception as e:
                if status_callback:
                    status_callback(f"⚠️ Could not check existing images: {e}")
            
            # Filter out processed images
            original_count = len(image_files)
            image_files = [img for img in image_files if img.name not in processed_images]
            skipped = original_count - len(image_files)
            
            if skipped > 0 and status_callback:
                status_callback(f"⏭️ Skipping {skipped} already processed images")
            
            total_images = len(image_files)
            if total_images == 0:
                return {"success": True, "message": "All images already processed!", "faces_added": 0}
        
        # Process in batches
        total_faces = 0
        total_processed = 0
        total_errors = 0
        start_time = time.time()
        
        num_batches = (total_images - 1) // BATCH_SIZE + 1
        
        for batch_num in range(num_batches):
            start_idx = batch_num * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_images)
            batch_files = image_files[start_idx:end_idx]
            
            if progress_callback:
                progress_percent = ((batch_num + 1) / num_batches) * 100
                progress_callback(min(100, int(progress_percent)))
            
            if status_callback:
                status_callback(f"🔄 Batch {batch_num + 1}/{num_batches}: Processing {len(batch_files)} images...")
            
            # Process batch
            results = face_engine.process_images_batch(batch_files, max_workers=1)
            
            # Collect results
            face_ids = []
            embeddings = []
            metadatas = []
            batch_faces = 0
            batch_errors = 0
            
            for result in results:
                if "error" not in result and result.get("face_count", 0) > 0:
                    total_processed += 1
                    face_count = result["face_count"]
                    batch_faces += face_count
                    
                    for face_data in result["faces"]:
                        face_ids.append(face_data["face_id"])
                        embeddings.append(face_data["embedding"])
                        
                        location = face_data["location"]
                        location_str = f"{location[0]},{location[1]},{location[2]},{location[3]}"
                        
                        metadatas.append({
                            "image_path": result["image_path"],
                            "location": location_str,
                            "face_id": face_data["face_id"]
                        })
                        
                elif "error" in result:
                    batch_errors += 1
                    total_errors += 1
                else:
                    total_processed += 1
            
            # Add to database
            if face_ids:
                if update_existing:
                    # Delete existing faces for processed images
                    for result in results:
                        if "error" not in result:
                            image_path = result["image_path"]
                            try:
                                existing_data = vector_store.collection.get()
                                if existing_data and existing_data.get('metadatas'):
                                    faces_to_delete = []
                                    for i, metadata in enumerate(existing_data['metadatas']):
                                        if metadata.get('image_path') == image_path:
                                            if i < len(existing_data.get('ids', [])):
                                                faces_to_delete.append(existing_data['ids'][i])
                                    
                                    if faces_to_delete:
                                        vector_store.collection.delete(ids=faces_to_delete)
                            except Exception:
                                pass  # Continue processing even if deletion fails
                
                added_count = vector_store.add_face_embeddings_batch(face_ids, embeddings, metadatas)
                total_faces += added_count
        
        # Final statistics
        total_time = time.time() - start_time
        
        result = {
            "success": True,
            "total_images": len(image_files),
            "processed": total_processed,
            "faces_added": total_faces,
            "errors": total_errors,
            "processing_time": total_time,
            "update_mode": update_existing
        }
        
        return result
        
    except Exception as e:
        return {"success": False, "message": f"Error during processing: {str(e)}"}
    finally:
        # Restore stdout
        sys.stdout = old_stdout

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast face processing")
    parser.add_argument("--update", action="store_true", help="Update existing database entries")
    parser.add_argument("--unique", action="store_true", help="Show unique images explanation")
    args = parser.parse_args()
    
    if args.unique:
        print("📋 UNIQUE IMAGES EXPLANATION")
        print("="*50)
        print("'Unique images' bezieht sich auf die Anzahl unterschiedlicher Bilder")
        print("in der Datenbank, basierend auf:")
        print("• Eindeutige Dateipfade")  
        print("• Bildinhalt-Hashing zur Duplikatserkennung")
        print("• Ein Bild kann mehrere Gesichter enthalten")
        print("• Gesichter-Anzahl ≠ Unique Images-Anzahl")
        print()
        print("Beispiel:")
        print("• 100 unique images mit je 2 Gesichtern = 200 Gesichter in DB")
        print("• 50 unique images mit je 1 Gesicht = 50 Gesichter in DB")
        exit(0)
    
    try:
        process_images_fast(update_existing=args.update)
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        logger.error(f"Fast processing failed: {e}")
