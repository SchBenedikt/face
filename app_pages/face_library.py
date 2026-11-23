"""Face-library page and its page-specific UI helpers."""
from pathlib import Path
import logging

import streamlit as st

from config import IMAGES_DIR
from portable_export import create_face_export, restore_face_export
from app_pages.components import display_face_from_metadata
from ui_cache import face_thumbnail

logger = logging.getLogger(__name__)


def render_face_library_page(faces_to_csv, invalidate_persons_cache, invalidate_gallery_cache):
    """Enhanced Face gallery to display all faces in the database"""

    st.header("Face library", icon=":material/gallery_thumbnail:")
    st.caption("Browse, review, and manage faces in your local database.")

    try:
        stats = st.session_state.vector_store.get_collection_stats()
        if "error" in stats:
            st.error(f"Could not load gallery statistics: {stats['error']}")
            return
        total_faces = int(stats.get("total_faces", 0))
        unique_images = int(stats.get("unique_images", 0))

        if total_faces == 0:
            st.info("No faces in the database yet. Upload images or import them from websites to get started.")
            return

        # Enhanced stats display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Faces", total_faces)
        with col2:
            st.metric("Images", unique_images)
        with col3:
            avg_faces = total_faces / unique_images if unique_images > 0 else 0
            st.metric("Average faces per image", f"{avg_faces:.1f}")

        st.markdown("---")

        # Gallery options
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            faces_per_page = st.selectbox("Faces per page", [20, 50, 100, 200, 500], index=1)

        with col2:
            show_metadata = st.checkbox("Show metadata", value=True)

        with col3:
            group_by_image = st.checkbox("Group by image", value=False)

        with col4:
            sort_by = st.selectbox("Sort by:", ["Newest", "Image name", "Face ID"])

        # Search and filter options
        st.markdown("---")

        # Show deletion statistics if available
        if 'deleted_faces_session' in st.session_state and st.session_state.deleted_faces_session > 0:
            st.info(f"{st.session_state.deleted_faces_session} faces removed in this session.")

        col1, col2 = st.columns([2, 1])

        with col1:
            search_term = st.text_input(
                "Search by file name, face ID, or person",
                placeholder="For example, 'image_001', 'abc123', or a person's name",
            )

        with col2:
            st.markdown("**Filters**")
            min_face_size = st.slider("Minimum face size (px)", 30, 200, 50)

        # Stream database metadata in bounded chunks and keep the sorted page
        # in a short-lived SQLite file instead of caching every face in session.
        filter_signature = (search_term, min_face_size, sort_by, faces_per_page)
        if st.session_state.get("gallery_filter_signature") != filter_signature:
            st.session_state.gallery_filter_signature = filter_signature
            st.session_state.gallery_page = 0
            st.session_state.pop("page_jump", None)

        if "gallery_page" not in st.session_state:
            st.session_state.gallery_page = 0
        page_result = st.session_state.vector_store.get_face_gallery_page(
            search_term=search_term,
            min_face_size=min_face_size,
            sort_by=sort_by,
            page=int(st.session_state.gallery_page) + 1,
            page_size=faces_per_page,
        )
        if page_result.get("error"):
            st.error(f"Could not load gallery results: {page_result['error']}")
            return
        filtered_total = int(page_result["total"])

        if filtered_total == 0:
            if search_term.strip():
                st.warning(f"No faces found for '{search_term}'.")
            else:
                st.info("No faces match the selected minimum size.")
            return

        # Pagination controls
        st.markdown("---")

        total_pages = int(page_result["total_pages"])
        current_page = int(page_result["page"]) - 1
        st.session_state.gallery_page = current_page

        # Page info and navigation in one row
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])

        with nav_col1:
            if st.button("Previous", disabled=(current_page == 0), icon=":material/chevron_left:"):
                st.session_state.gallery_page = max(0, current_page - 1)
                st.rerun()

        with nav_col2:
            page_input = st.number_input(
                "Go to page",
                min_value=1,
                max_value=total_pages,
                value=current_page + 1,
                key="page_jump",
                label_visibility="collapsed"
            )
            if page_input - 1 != current_page:
                st.session_state.gallery_page = page_input - 1
                st.rerun()

        with nav_col3:
            st.write(f"**Page {current_page + 1} of {total_pages}**")
            remaining_faces = filtered_total - (current_page * faces_per_page)
            faces_on_page = min(faces_per_page, remaining_faces)
            st.write(f"Showing {faces_on_page} of {filtered_total} faces")

        with nav_col4:
            if st.button("Next", disabled=(current_page >= total_pages - 1), icon=":material/chevron_right:"):
                st.session_state.gallery_page = min(total_pages - 1, current_page + 1)
                st.rerun()

        with nav_col5:
            if st.button("Refresh", icon=":material/refresh:"):
                invalidate_gallery_cache()
                st.session_state.gallery_page = 0
                st.rerun()

        st.markdown("---")

        offset = current_page * faces_per_page
        remaining_faces = filtered_total - offset
        limit = min(faces_per_page, remaining_faces)

        if limit <= 0:
            st.warning("No faces on this page. Go to an earlier page.")
            return

        with st.spinner(f"Loading {limit} faces..."):
            # Get face data from database with improved robust pagination
            try:
                page_faces = page_result["faces"]
                results = {"metadatas": [face["metadata"] for face in page_faces]}

                if results['metadatas'] and len(results['metadatas']) > 0:
                    st.subheader(f"Showing {len(results['metadatas'])} faces (page {current_page + 1})")

                    # Debug information for troubleshooting
                    if st.checkbox("Show debug information", value=False):
                        st.info(f"""
                        **Debug information:**
                        - Current page: {current_page + 1}
                        - Offset: {offset}
                        - Limit: {limit}
                        - Faces found: {len(results['metadatas'])}
                        - Total faces in database: {filtered_total}
                        - Embeddings available: {len(results.get('embeddings', [])) > 0}
                        - IDs available: {len(results.get('ids', [[]])[0]) > 0 if results.get('ids') else False}
                        """)

                    # Group faces if requested
                    if group_by_image:
                        display_faces_grouped_by_image(results['metadatas'], show_metadata)
                    else:
                        display_faces_in_grid(results['metadatas'], show_metadata)

                    # Add export/download options
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

                    with col1:
                        st.download_button(
                            "Export faces as CSV",
                            data=lambda: faces_to_csv(
                                st.session_state.vector_store.get_all_faces_metadata()
                            ),
                            file_name="faces.csv",
                            mime="text/csv",
                            key="download_faces_csv",
                            width="stretch",
                        )
                        with st.expander("Restore a portable ZIP"):
                            restore_file = st.file_uploader(
                                "Choose a face-library ZIP",
                                type=["zip"],
                                key="restore_face_archive_upload",
                            )
                            st.caption("Restore requires the same embedding pipeline version and skips archives with face IDs already in this index.")
                            if restore_file is not None and st.button(
                                "Restore faces and optional images",
                                key="restore_face_archive_button",
                                type="primary",
                            ):
                                outcome = restore_face_export(
                                    restore_file.getvalue(),
                                    st.session_state.vector_store,
                                    IMAGES_DIR,
                                )
                                if outcome["success"]:
                                    invalidate_persons_cache()
                                    invalidate_gallery_cache()
                                    st.success(
                                        f"Restored {outcome['restored_faces']} faces and "
                                        f"{outcome['restored_images']} images."
                                    )
                                    st.rerun()
                                else:
                                    st.error(f"Restore failed: {outcome['error']}")
                        st.markdown("**Portable backup**")
                        st.caption("Archives are staged on disk and limited to 128 MiB. Embedding-free exports are metadata copies and cannot restore a searchable index.")
                        export_scope = st.selectbox(
                            "Faces to export",
                            ["Current page", "All indexed faces"],
                            key="portable_export_scope",
                        )
                        include_embeddings = st.checkbox(
                            "Include biometric embeddings",
                            value=False,
                            key="portable_export_embeddings",
                            help="Embeddings are sensitive biometric data. Enable this only when required for a searchable backup.",
                        )
                        include_images = st.checkbox(
                            "Include original images",
                            value=False,
                            key="portable_export_images",
                            help="Original images may contain sensitive or identifying information.",
                        )
                        selected_face_ids = (
                            [str(metadata.get("face_id")) for metadata in results["metadatas"] if metadata.get("face_id")]
                            if export_scope == "Current page"
                            else None
                        )
                        st.download_button(
                            "Download portable ZIP",
                            data=lambda: create_face_export(
                                st.session_state.vector_store.collection,
                                st.session_state.vector_store._embedding_pipeline_version,
                                face_ids=selected_face_ids,
                                include_embeddings=include_embeddings,
                                include_images=include_images,
                            ),
                            file_name="face-library-backup.zip",
                            mime="application/zip",
                            key="download_portable_faces",
                            width="stretch",
                        )

                    with col2:
                        # Batch delete current page faces
                        if st.button("Delete all on this page", help="Delete all faces on the current page."):
                            if st.session_state.get('confirm_batch_delete', False):
                                # Perform batch deletion
                                deleted_count = 0
                                face_ids = [metadata.get('face_id', '') for metadata in results['metadatas'] if metadata.get('face_id')]

                                with st.spinner(f"Deleting {len(face_ids)} faces..."):
                                    for face_id in face_ids:
                                        if delete_face_from_gallery(face_id, invalidate_persons_cache, invalidate_gallery_cache):
                                            deleted_count += 1

                                st.success(f"Deleted {deleted_count} of {len(face_ids)} faces.")
                                st.session_state.confirm_batch_delete = False

                                # Go back to page 1 to avoid pagination errors
                                st.session_state.gallery_page = 0
                                st.rerun()
                            else:
                                st.session_state.confirm_batch_delete = True
                                st.warning("Click again to confirm deleting ALL faces on this page.")
                                st.rerun()

                    with col3:
                        # Delete by similarity threshold
                        if st.button("Delete low-quality faces", help="Delete detections with low image quality."):
                            # Analyze current page faces for quality issues
                            low_quality_faces = identify_low_quality_faces(results['metadatas'])

                            if low_quality_faces:
                                st.warning(f"Found {len(low_quality_faces)} potentially incorrect detections:")

                                # Show preview of faces to be deleted
                                with st.expander("Review low-quality faces", expanded=False):
                                    for i, face_info in enumerate(low_quality_faces[:5]):  # Show first 5
                                        st.write(f"• **{face_info['face_id'][:12]}...**: {face_info['reason']}")

                                    if len(low_quality_faces) > 5:
                                        st.write(f"... and {len(low_quality_faces) - 5} more")

                                if st.button("Delete low-quality faces", key="confirm_quality_delete"):
                                    deleted_count = 0

                                    with st.spinner(f"Deleting {len(low_quality_faces)} low-quality faces..."):
                                        for face_info in low_quality_faces:
                                            if delete_face_from_gallery(face_info['face_id'], invalidate_persons_cache, invalidate_gallery_cache):
                                                deleted_count += 1

                                    st.success(f"Deleted {deleted_count} low-quality faces.")
                                    st.rerun()
                            else:
                                st.info("No low-quality faces found on this page.")

                    with col4:
                        if st.button("🔄 Refresh gallery"):
                            invalidate_gallery_cache()
                            # Reset to first page to avoid pagination errors
                            st.session_state.gallery_page = 0
                            st.rerun()
                else:
                    st.warning("No face data found on this page.")

                    # Helpful suggestions for user
                    st.info("""
                    **Try these steps:**
                    - Go to an earlier page.
                    - Reduce the number of faces per page.
                    - Refresh the gallery.
                    - Return to page 1.
                    """)

                    if st.button("Return to page 1"):
                        st.session_state.gallery_page = 0
                        st.rerun()

            except Exception as db_e:
                st.error(f"Database error while loading faces: {db_e}")
                st.info("Refresh the gallery or choose another page.")

    except Exception as e:
        st.error(f"Error loading the face gallery: {e}")
        logger.error(f"Face gallery error: {e}")

    # The modals are now called directly from button events, no need for session state checks

def display_faces_grouped_by_image(metadatas, show_metadata=True):
    """Display faces grouped by source image"""
    faces_by_image = {}

    # Group faces by image
    for i, metadata in enumerate(metadatas):
        image_path = metadata.get('image_path', 'unknown')
        if image_path not in faces_by_image:
            faces_by_image[image_path] = []
        faces_by_image[image_path].append((i, metadata))

    # Display grouped faces
    for image_path, face_list in faces_by_image.items():
        image_name = Path(image_path).name

        with st.expander(f"📸 {image_name} ({len(face_list)} faces)", expanded=True):
            # Show original image thumbnail
            if Path(image_path).exists():
                try:
                    original_thumbnail = face_thumbnail(image_path, None, (200, 200))
                    if original_thumbnail is not None:

                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(original_thumbnail, caption="Original Image", width='stretch')
                        with col2:
                            st.write(f"**Filename:** {image_name}")
                            st.write(f"**👥 Faces:** {len(face_list)}")
                            if Path(image_path).exists():
                                file_size = Path(image_path).stat().st_size / 1024
                                st.write(f"**💾 File size:** {file_size:.1f} KB")
                except Exception as e:
                    st.warning(f"Could not load the original image: {e}")

            # Display faces in this image
            cols_per_row = 6
            for i in range(0, len(face_list), cols_per_row):
                cols = st.columns(cols_per_row)

                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(face_list):
                        face_idx, metadata = face_list[idx]
                        with col:
                            display_face_from_metadata(metadata, show_metadata, compact=True)

def display_faces_in_grid(metadatas, show_metadata=True):
    """Display faces in a grid layout"""
    cols_per_row = 5  # More faces per row for better overview

    for i in range(0, len(metadatas), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(metadatas):
                with col:
                    metadata = metadatas[idx]
                    display_face_from_metadata(metadata, show_metadata)

def delete_face_from_gallery(face_id: str, invalidate_persons_cache, invalidate_gallery_cache) -> bool:
    """Delete a face from the gallery and vector store"""
    try:
        # Delete from vector store
        if st.session_state.vector_store.delete_face(face_id):
            # Update deletion statistics
            if 'deleted_faces_session' not in st.session_state:
                st.session_state.deleted_faces_session = 0
            st.session_state.deleted_faces_session += 1

            invalidate_gallery_cache()
            invalidate_persons_cache()
            logger.info(f"Successfully deleted face {face_id} from gallery")
            return True
        else:
            logger.error(f"Failed to delete face {face_id} from vector store")
            return False
    except Exception as e:
        logger.error(f"Error deleting face {face_id}: {e}")
        return False

def identify_low_quality_faces(metadatas, min_face_size=30):
    """Identify potentially low-quality or false-positive faces"""
    low_quality_faces = []

    for metadata in metadatas:
        try:
            location_str = metadata.get('location', '0,0,0,0')
            top, right, bottom, left = map(int, location_str.split(','))

            face_width = right - left
            face_height = bottom - top
            face_area = face_width * face_height

            # Criteria for low quality faces
            is_too_small = face_width < min_face_size or face_height < min_face_size
            is_too_narrow = face_width < face_height * 0.6  # Very narrow faces are suspicious
            is_too_wide = face_width > face_height * 2.0    # Very wide faces are suspicious
            is_tiny_area = face_area < 900  # Less than 30x30 pixels

            if is_too_small or is_too_narrow or is_too_wide or is_tiny_area:
                low_quality_faces.append({
                    'face_id': metadata.get('face_id'),
                    'reason': f"Size: {face_width}×{face_height}px, Area: {face_area}px²",
                    'metadata': metadata
                })

        except Exception as e:
            logger.error(f"Error analyzing face quality: {e}")
            continue

    return low_quality_faces
