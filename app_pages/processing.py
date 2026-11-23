"""Local image-import and batch-processing Streamlit pages."""
from pathlib import Path
import time

import streamlit as st

from config import IMAGES_DIR, SCRAPED_DIR
from utils import IMAGE_EXTENSIONS, get_image_files

IMAGE_SUFFIXES = tuple(sorted(IMAGE_EXTENSIONS))


def render_batch_processing_page(discover_image_directories, show_embedding_pipeline_status, invalidate_gallery_cache, invalidate_persons_cache):
    """Enhanced batch processing page using fast_process.py for optimal performance."""

    st.header("🚀 Fast Batch Face Processing")
    st.markdown("**High-performance face detection and processing using optimized algorithms**")
    show_embedding_pipeline_status()

    tab1, tab2 = st.tabs(["⚡ Fast Processing", "📊 Processing Stats"])

    with tab1:
        # Processing mode selection
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🎯 Processing Mode")
            processing_mode = st.radio(
                "Select Processing Mode:",
                ["🆕 Process New Images Only", "🔄 Update All Images (Replace Existing)"],
                help="New: Skip already processed images for faster processing\nUpdate: Replace all existing face data with fresh analysis"
            )
            update_existing = processing_mode.startswith("🔄")

        with col2:
            # Current database statistics
            st.subheader("📊 Database Status")
            try:
                from vector_store import FaceVectorStore
                vector_store = FaceVectorStore()
                existing_data = vector_store.collection.get()
                total_faces = len(existing_data.get('ids', []))

                # Count unique images
                unique_images = set()
                if existing_data and existing_data.get('metadatas'):
                    metadatas = existing_data.get('metadatas') or []
                    for metadata in metadatas:
                        if metadata and 'image_path' in metadata and isinstance(metadata['image_path'], str):
                            unique_images.add(Path(metadata['image_path']).name)

                st.metric("Total Faces", f"{total_faces:,}")
                st.metric("Unique Images", f"{len(unique_images):,}")

                if total_faces > 0:
                    avg_faces = total_faces / len(unique_images) if unique_images else 0
                    st.metric("Avg Faces/Image", f"{avg_faces:.1f}")

            except Exception as e:
                st.error(f"Could not load database stats: {e}")

        st.divider()

        # Directory selection
        st.subheader("📁 Directory Selection")

        # Auto-detect available directories with image counts
        available_dirs = discover_image_directories()

        if not available_dirs:
            st.warning("📂 No image directories found. Please download or scrape images first.")
            return

        # Directory selection
        selected_option = st.selectbox(
            "Choose directory to process:",
            options=[option[0] for option in available_dirs],
            index=0,
            help="Most recent downloaded directories are shown first"
        )

        # Get actual directory path
        selected_dir = None
        for option, path in available_dirs:
            if option == selected_option:
                selected_dir = path
                break

        if selected_dir:
            st.success(f"📁 Selected: `{selected_dir}`")

            # Processing button
            st.divider()

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    f"🚀 {'Update' if update_existing else 'Process'} Images",
                    type="primary",
                    width='stretch',
                    help=f"{'Replace existing face data' if update_existing else 'Add new faces only'}"
                ):
                    process_with_fast_engine(selected_dir, update_existing, invalidate_gallery_cache, invalidate_persons_cache)

    with tab2:
        show_processing_statistics()


def process_with_fast_engine(directory_path, update_existing, invalidate_gallery_cache, invalidate_persons_cache):
    """Process images using fast_process.py engine with Streamlit integration."""
    try:
        target_directory = Path(directory_path).expanduser().resolve()
        allowed_roots = [Path(IMAGES_DIR).resolve(), Path(SCRAPED_DIR).resolve()]
        if not target_directory.is_dir() or not any(
            target_directory == root or root in target_directory.parents
            for root in allowed_roots
        ):
            st.error("The selected folder is outside the configured image directories.")
            return

        from fast_process import process_images_streamlit

        # Create progress containers
        progress_container = st.container()
        status_container = st.container()
        results_container = st.container()

        with progress_container:
            progress_bar = st.progress(0)
            progress_text = st.empty()

        with status_container:
            status_text = st.empty()

        # Callback functions for progress updates
        def update_progress(percent):
            # Ensure percent is a regular Python int/float, not numpy type.
            # fast_process reports 0-100, while st.progress expects 0.0-1.0.
            percent = max(0.0, min(100.0, float(percent)))
            progress_bar.progress(percent / 100.0)
            progress_text.text(f"Processing: {int(percent)}%")

        def update_status(message):
            status_text.info(message)

        # Start processing
        start_time = time.time()

        with st.spinner("🧠 Initializing fast face processing engine..."):
            result = process_images_streamlit(
                update_existing=update_existing,
                progress_callback=update_progress,
                status_callback=update_status,
                directory_path=target_directory,
            )

        # Show results
        with results_container:
            processing_time = time.time() - start_time

            if result["success"]:
                invalidate_gallery_cache()
                invalidate_persons_cache()
                failed_images = result.get("failed_images", [])
                if failed_images:
                    st.warning(
                        f"⚠️ Processing completed with {len(failed_images)} error(s). "
                        "The valid images were added to the database."
                    )
                else:
                    st.success("🎉 Processing completed successfully!")

                # Results metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("📁 Images", f"{result.get('total_images', 0):,}")

                with col2:
                    st.metric("✅ Processed", f"{result.get('processed', 0):,}")

                with col3:
                    st.metric("👤 Faces Added", f"{result.get('faces_added', 0):,}")

                with col4:
                    if result.get('processed', 0) > 0:
                        avg_faces = result.get('faces_added', 0) / result.get('processed', 1)
                        st.metric("📊 Faces/Image", f"{avg_faces:.1f}")

                # Performance statistics
                st.subheader("⚡ Performance Statistics")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🕒 Total Time", f"{processing_time:.1f}s")

                with col2:
                    if result.get('total_images', 0) > 0:
                        img_per_sec = result.get('total_images', 0) / processing_time
                        st.metric("📈 Images/sec", f"{img_per_sec:.2f}")

                with col3:
                    if result.get('errors', 0) > 0:
                        error_rate = (result.get('errors', 0) / result.get('total_images', 1)) * 100
                        st.metric("⚠️ Error Rate", f"{error_rate:.1f}%")
                    else:
                        st.metric("✅ Success Rate", "100%")

                # Additional info
                if result.get('update_mode'):
                    st.info("🔄 **Update Mode**: All existing face data was replaced with fresh analysis")
                else:
                    st.info("🆕 **New Mode**: Only new images were processed, existing faces were preserved")

                if failed_images:
                    st.warning("**Images with errors:**")
                    for failure in failed_images:
                        st.write(
                            f"- `{failure.get('name', 'unknown')}`: "
                            f"{failure.get('error', 'Unknown error')}"
                        )

                # Suggest next steps
                st.subheader("🎯 Next Steps")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("🔍 View Face Gallery", width='stretch'):
                        st.session_state.pending_page = "face_gallery"
                        st.rerun()

                with col2:
                    if st.button("📊 Analyze Results", width='stretch'):
                        st.session_state.pending_page = "face_search"
                        st.rerun()

                # The sidebar was rendered before this batch ran. Queue a
                # short summary and let main rerun after the page returns so
                # its face counts and the gallery reflect the new rows.
                if result.get("faces_added", 0) > 0 or result.get("stale_removed", 0) > 0:
                    error_lines = "\n".join(
                        f"- `{failure.get('name', 'unknown')}`: "
                        f"{failure.get('error', 'Unknown error')}"
                        for failure in failed_images
                    )
                    summary = (
                        f"**Batch Summary:** {result.get('processed', 0)} images "
                        f"processed, {result.get('faces_added', 0)} faces added, "
                        f"{result.get('errors', 0)} errors."
                    )
                    if result.get("stale_removed", 0):
                        summary += (
                            f"\n- Stale entries removed: {result.get('stale_removed')}"
                        )
                    if error_lines:
                        summary += f"\n\n**Images with errors:**\n{error_lines}"
                    if failed_images:
                        headline = (
                            "⚠️ Batch processing completed with errors; "
                            "the valid images were added."
                        )
                        kind = "warning"
                    else:
                        headline = "🎉 Batch processing completed successfully!"
                        kind = "success"
                    st.session_state["_flash_messages"] = [
                        (kind, headline),
                        ("info", summary),
                    ]
                    st.session_state["_refresh_after_processing"] = True

            else:
                st.error(f"❌ Processing failed: {result.get('message', 'Unknown error')}")

                if "No images found" in result.get('message', ''):
                    st.info("💡 Make sure the selected directory contains JPG, PNG, or JPEG images")
                elif "All images already processed" in result.get('message', ''):
                    st.success("✅ All images in this directory have already been processed!")
                    st.info("💡 Use 'Update All Images' mode if you want to reprocess everything")

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.exception(e)


def show_processing_statistics():
    """Display comprehensive processing statistics."""
    st.subheader("📊 Database Statistics")

    try:
        from vector_store import FaceVectorStore
        vector_store = FaceVectorStore()
        existing_data = vector_store.collection.get()

        if not existing_data or not existing_data.get('ids'):
            st.info("📭 No faces in database yet. Process some images first!")
            return

        total_faces = len(existing_data['ids'])
        metadatas = existing_data.get('metadatas', [])

        # Collect statistics
        image_stats = {}
        if metadatas:
            for metadata in metadatas:
                if metadata and isinstance(metadata, dict) and 'image_path' in metadata:
                    img_path = metadata['image_path']
                    if isinstance(img_path, str):
                        img_name = Path(img_path).name

                        if img_name not in image_stats:
                            image_stats[img_name] = {
                                'faces': 0,
                                'path': img_path
                            }
                        image_stats[img_name]['faces'] += 1

        # Display overview statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("👤 Total Faces", f"{total_faces:,}")

        with col2:
            st.metric("🖼️ Unique Images", f"{len(image_stats):,}")

        with col3:
            if image_stats:
                avg_faces = total_faces / len(image_stats)
                st.metric("📊 Avg Faces/Image", f"{avg_faces:.1f}")

        with col4:
            if image_stats:
                max_faces = max(img['faces'] for img in image_stats.values())
                st.metric("🎯 Max Faces/Image", max_faces)

        # Face distribution chart
        st.subheader("📈 Face Distribution")

        if len(image_stats) > 0:
            # Create histogram of faces per image
            face_counts = [img['faces'] for img in image_stats.values()]

            import pandas as pd
            df = pd.DataFrame({
                'Faces per Image': face_counts
            })

            st.bar_chart(df)

            # Top images with most faces
            st.subheader("🏆 Top Images by Face Count")

            sorted_images = sorted(image_stats.items(), key=lambda x: x[1]['faces'], reverse=True)[:10]

            for i, (img_name, stats) in enumerate(sorted_images, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{i}. `{img_name}`")
                with col2:
                    st.write(f"👤 {stats['faces']} faces")

    except Exception as e:
        st.error(f"Could not load statistics: {e}")


def render_image_import_page(load_query_image, invalidate_persons_cache, invalidate_gallery_cache):
    """Image upload and processing page"""

    st.header("📥 Image Upload & Processing")

    tab1, tab2 = st.tabs(["Upload Images", "Process Existing Images"])

    with tab1:
        st.subheader("Upload New Images")

        uploaded_files = st.file_uploader(
            "Choose image files:",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            accept_multiple_files=True,
            help="Upload multiple images to add to the face database"
        )

        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files")

            if st.button("📤 Process and Add to Database", type="primary"):
                process_uploaded_images(uploaded_files, load_query_image, invalidate_persons_cache, invalidate_gallery_cache)

    with tab2:
        st.subheader("Process Existing Images")

        # Show existing image directories
        image_dirs = [IMAGES_DIR, SCRAPED_DIR]

        for img_dir in image_dirs:
            if img_dir.exists():
                image_files = get_image_files(img_dir)

                if image_files:
                    st.write(f"**{img_dir.name}**: {len(image_files)} images")

                    if st.button(f"Process {img_dir.name} Images", key=f"process_{img_dir.name}"):
                        process_existing_images(image_files, invalidate_persons_cache, invalidate_gallery_cache)

def process_uploaded_images(uploaded_files, load_query_image, invalidate_persons_cache, invalidate_gallery_cache):
    """Process uploaded images and add to database"""

    with st.spinner("Processing uploaded images..."):
        try:
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            saved_paths = []
            for uploaded_file in uploaded_files:
                safe_name = Path(uploaded_file.name).name
                if Path(safe_name).suffix.lower() not in IMAGE_SUFFIXES:
                    st.warning(f"Skipped {safe_name}: unsupported image extension.")
                    continue
                image, image_error = load_query_image(uploaded_file)
                if image is None:
                    st.warning(f"Skipped {safe_name}: {image_error}")
                    continue
                content = uploaded_file.getvalue()
                final_path = IMAGES_DIR / safe_name

                # Handle duplicate names
                counter = 1
                while final_path.exists():
                    stem = Path(safe_name).stem
                    suffix = Path(safe_name).suffix
                    final_path = IMAGES_DIR / f"{stem}_{counter}{suffix}"
                    counter += 1

                final_path.write_bytes(content)
                saved_paths.append(final_path)

            # Process images
            if saved_paths:
                process_existing_images(saved_paths, invalidate_persons_cache, invalidate_gallery_cache)
            else:
                st.warning("No valid images were selected. Choose readable JPG, PNG, BMP, or WEBP files.")

        except Exception as e:
            st.error(f"Error processing uploaded images: {e}")

def process_existing_images(image_paths, invalidate_persons_cache, invalidate_gallery_cache):
    """Process existing images and add to database"""

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Processing images for face detection...")

        # Process images in batches
        results = st.session_state.face_engine.process_images_batch(image_paths)

        progress_bar.progress(50)
        status_text.text("Extracting face embeddings...")

        # Prepare data for vector store
        face_ids = []
        embeddings = []
        metadatas = []

        # Count processing statistics
        successful_images = 0
        failed_images = 0
        total_faces = 0
        images_without_faces = 0

        for result in results:
            if "error" not in result:
                successful_images += 1
                face_count = result.get("face_count", 0)
                total_faces += face_count

                if face_count > 0:
                    for face_data in result["faces"]:
                        face_ids.append(face_data["face_id"])
                        embeddings.append(face_data["embedding"])
                        # Convert location tuple to string for database storage
                        location = face_data["location"]
                        location_str = f"{location[0]},{location[1]},{location[2]},{location[3]}"
                        metadatas.append({
                            "image_path": result["image_path"],
                            "location": location_str,
                            "face_id": face_data["face_id"]
                        })
                else:
                    images_without_faces += 1
            else:
                failed_images += 1

        progress_bar.progress(75)
        status_text.text("Adding to vector database...")

        # Add to vector store. Existing rows for these paths are replaced, so
        # processing the same image again does not duplicate it.
        added_count = 0
        if face_ids:
            added_count = st.session_state.vector_store.add_face_embeddings_batch(
                face_ids, embeddings, metadatas
            )

        # An image that no longer contains a face should not keep old rows.
        for result in results:
            if "error" not in result and result.get("face_count", 0) == 0:
                st.session_state.vector_store.delete_faces_by_image_path(
                    result["image_path"]
                )
        invalidate_persons_cache()
        invalidate_gallery_cache()

        progress_bar.progress(100)
        status_text.text("Complete!")

        # Provide detailed summary
        success_message = None
        if added_count > 0:
            success_message = f"🎉 Successfully processed {len(image_paths)} images and added {added_count} face embeddings to the database!"
            st.success(success_message)
        elif total_faces > 0:
            st.warning(f"⚠️ Found {total_faces} faces but failed to add them to database. Please check the database connection.")
        elif successful_images > 0:
            st.info(f"✅ Successfully processed {successful_images} images, but no faces were detected in any of them.")
        else:
            st.warning("❌ No images could be processed successfully. Please check the image files.")

        # Show detailed processing summary
        summary_message = (f"**Processing Summary:**\n"
                           f"- Total images: {len(image_paths)}\n"
                           f"- Successfully processed: {successful_images}\n"
                           f"- Failed to process: {failed_images}\n"
                           f"- Total faces detected: {total_faces}\n"
                           f"- Images without faces: {images_without_faces}\n"
                           f"- Face embeddings written to database: {added_count}")
        st.info(summary_message)

    except Exception as e:
        st.error(f"Error processing images: {e}")
        import traceback
        st.code(traceback.format_exc())
    else:
        # The sidebar is rendered before the page content, so its stats still
        # show the pre-processing count in this run. Queue the summary and a
        # refresh; main triggers the rerun after the page function returns (a
        # rerun raised here could be swallowed by callers' error handling).
        if added_count > 0:
            st.session_state["_flash_messages"] = [
                ("success", success_message),
                ("info", summary_message),
            ]
            st.session_state["_refresh_after_processing"] = True
