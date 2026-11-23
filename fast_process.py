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


def _normalised_path(path) -> str:
    return str(Path(path).expanduser().resolve())


def processed_image_paths(vector_store) -> set:
    """Return the source paths that already have rows in the vector store."""
    processed = set()
    try:
        existing_data = vector_store.collection.get(include=['metadatas'])
        for metadata in existing_data.get('metadatas') or []:
            image_path = (metadata or {}).get('image_path')
            if image_path:
                processed.add(_normalised_path(image_path))
    except Exception as exc:  # noqa: BLE001 - processing can start without this check
        logger.warning(f"Could not read already processed images: {exc}")
    return processed


def filter_unprocessed_images(image_files: List[Path], processed_paths: set) -> List[Path]:
    """Keep files whose full path is new; equal file names in other folders stay."""
    return [path for path in image_files if _normalised_path(path) not in processed_paths]


def _should_remove_existing_faces(result: dict) -> bool:
    """Only replace old rows after a successful no-face result or bad source."""
    if result.get("source_unreadable"):
        return True
    return "error" not in result and result.get("face_count", 0) == 0

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
    
    # Filter already processed images if not updating. Compare full paths:
    # equal file names in different folders are different images.
    if not update_existing:
        print("🔍 Checking for already processed images...")
        processed_images = processed_image_paths(vector_store)
        original_count = len(image_files)
        image_files = filter_unprocessed_images(image_files, processed_images)
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
    total_no_faces = 0
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
                total_no_faces += 1
                print(f"  ℹ️ {image_name} → No faces detected (file kept)")
        
        # Add to database. add_face_embeddings_batch replaces existing rows for
        # the same image, so update mode no longer scans the whole collection.
        if face_ids:
            added_count = vector_store.add_face_embeddings_batch(face_ids, embeddings, metadatas)
            total_faces += added_count
            print(f"  💾 Wrote {added_count} faces to database")

        if update_existing:
            # Preserve old rows when model/runtime processing failed. Remove
            # them only after a valid no-face result or a corrupt source file.
            for result in results:
                if _should_remove_existing_faces(result):
                    vector_store.delete_faces_by_image_path(result["image_path"])
        
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
    print(f"ℹ️ Images without faces kept: {total_no_faces}")
    print(f"❌ Errors: {total_errors}")
    print(f"⏱️ Total Time: {total_time:.1f}s")
    print(f"🚀 Average per Image: {avg_time_per_image:.1f}s")
    
    if total_processed > 0:
        success_rate = (total_processed / total_images) * 100
        avg_faces_per_image = total_faces / total_processed
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"👤 Avg Faces per Image: {avg_faces_per_image:.2f}")


def process_images_streamlit(
    update_existing=False,
    progress_callback=None,
    status_callback=None,
    directory_path=None,
):
    """Process one explicit image directory without mutating shared globals."""
    try:
        target_directory = Path(directory_path or IMAGES_DIR).expanduser().resolve()
        image_files = get_image_files_fast(target_directory)
        total_images = len(image_files)
        
        if total_images == 0:
            return {"success": False, "message": f"No images found in {target_directory}"}
        
        # Import face engine with fast config
        from face_recognition_engine import FaceRecognitionEngine
        from vector_store import FaceVectorStore
        
        # Initialize with fast settings
        face_engine = FaceRecognitionEngine()
        vector_store = FaceVectorStore()
        
        if status_callback:
            status_callback(f"🔍 Found {total_images} images in {target_directory}")
        
        # Filter already processed images if not updating. Compare full paths:
        # equal file names in different folders are different images.
        if not update_existing:
            processed_images = processed_image_paths(vector_store)
            original_count = len(image_files)
            image_files = filter_unprocessed_images(image_files, processed_images)
            skipped = original_count - len(image_files)
            
            if skipped > 0 and status_callback:
                status_callback(f"⏭️ Skipping {skipped} already processed images")
            
            total_images = len(image_files)
            if total_images == 0:
                return {"success": True, "message": "All images already processed!", "faces_added": 0}
        
        # Process in batches; the batch size is editable on the Settings page.
        try:
            from settings_store import load_settings
            batch_size = int(load_settings()["batch_size"])
        except Exception:  # noqa: BLE001 - fall back to the static config
            batch_size = BATCH_SIZE

        total_faces = 0
        total_processed = 0
        total_no_faces = 0
        total_errors = 0
        failed_images = []
        stale_removed = 0
        start_time = time.time()
        
        num_batches = (total_images - 1) // batch_size + 1
        
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_images)
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
                    image_path = result.get("image_path", "unknown")
                    failed_images.append({
                        "image_path": image_path,
                        "name": Path(image_path).name,
                        "error": result.get("error", "Unknown error"),
                    })
                else:
                    total_processed += 1
                    total_no_faces += 1
            
            # Add to database; the vector store replaces rows for the same path.
            if face_ids:
                added_count = vector_store.add_face_embeddings_batch(face_ids, embeddings, metadatas)
                total_faces += added_count

            if update_existing:
                for result in results:
                    if _should_remove_existing_faces(result):
                        # A corrupt or successfully faceless image must not keep
                        # stale rows. Runtime/model failures preserve old rows.
                        stale_removed += vector_store.delete_faces_by_image_path(
                            result["image_path"]
                        )

        if update_existing:
            # Prune only this job's source tree; other indexed directories may
            # be outside it and must not be affected by this operation.
            stale_removed += vector_store.prune_missing_faces(target_directory)
        
        # Final statistics
        total_time = time.time() - start_time
        
        result = {
            "success": True,
            "total_images": len(image_files),
            "processed": total_processed,
            "faces_added": total_faces,
            "no_faces": total_no_faces,
            "errors": total_errors,
            "failed_images": failed_images,
            "stale_removed": stale_removed,
            "processing_time": total_time,
            "update_mode": update_existing
        }
        
        return result
        
    except Exception as e:
        return {"success": False, "message": f"Error during processing: {str(e)}"}

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
