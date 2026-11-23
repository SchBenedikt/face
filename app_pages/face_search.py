"""Face-query upload, detection, filtering, and local search page."""
import hashlib
import logging

import cv2
import numpy as np
import streamlit as st

from app_pages.search_results import display_search_results
from settings_store import active_embedding_model, load_settings
from utils import create_thumbnail, load_uploaded_image

logger = logging.getLogger(__name__)

SEARCH_MODE_LABELS = {
    "Balanced": "Balanced",
    "High Precision": "High precision",
    "High Recall": "High recall",
}
SORT_LABELS = {
    "Similarity Score": "Similarity",
    "Face Size": "Face size",
    "Date Added": "Date added",
    "File Name": "File name",
}


def render_face_search_page(show_embedding_pipeline_status):
    """Face search functionality with multi-face selection"""

    st.header("Face search", icon=":material/search:")
    st.caption(
        "Search runs only against locally indexed images. Similarity scores are "
        "not match probabilities; review every suggested match."
    )
    show_embedding_pipeline_status()

    # Upload query image
    uploaded_file = st.file_uploader(
        "Upload a search image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Choose a photo with one or more faces from your local collection."
    )

    # Preserve uploaded file in session state for UI consistency
    if uploaded_file is not None:
        st.session_state.current_uploaded_file = uploaded_file
    elif 'current_uploaded_file' in st.session_state:
        uploaded_file = st.session_state.current_uploaded_file

    if uploaded_file is not None and st.session_state.get('detected_faces'):
        detected_data = st.session_state.detected_faces
        current_digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        analyzed_digest = detected_data.get('image_sha256')
        if analyzed_digest is None and detected_data.get('image_data') is not None:
            analyzed_digest = hashlib.sha256(detected_data['image_data']).hexdigest()
        if analyzed_digest != current_digest:
            # Never show faces from an earlier upload beside a new query image.
            del st.session_state.detected_faces

    col1, col2 = st.columns([1, 1])

    with col1:
        if uploaded_file is not None:
            # Display uploaded image
            query_image, image_error = load_query_image(uploaded_file)
            if image_error:
                st.error(f"❌ {image_error}")
                st.stop()
            st.image(
                query_image, caption="Search image", width="stretch",
                alt="Uploaded search image",
            )

            query_settings = load_settings()
            query_model = active_embedding_model()

            # Enhanced Search Parameters
            st.subheader("Search settings", icon=":material/tune:")

            col_a, col_b = st.columns(2)

            with col_a:
                max_results = st.slider(
                    "Maximum results",
                    min_value=1,
                    max_value=200,
                    value=int(query_settings["max_results"]),
                    key="face_search_max_results",
                    help="Maximum number of candidate faces to display."
                )

                similarity_threshold = st.slider(
                    "Minimum cosine similarity",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(query_settings["similarity_threshold"]),
                    step=0.05,
                    key="face_search_similarity_threshold",
                    help=(
                        "Cosine similarity is not a match probability. "
                        f"Active model: {query_model}. Higher thresholds return "
                        "fewer, stricter candidates."
                    )
                )

            with col_b:
                # Face Size Filter
                min_face_size = st.slider(
                    "Minimum face size (px)",
                    min_value=20,
                    max_value=200,
                    value=30,
                    key="face_search_min_face_size",
                    help="Small faces can still be valid matches. Raise this value to hide them."
                )

                # Search Mode
                search_mode = st.selectbox(
                    "Search mode",
                    ["Balanced", "High Precision", "High Recall"],
                    index=0,
                    key="face_search_mode",
                    format_func=SEARCH_MODE_LABELS.__getitem__,
                    help="Balanced uses your threshold. High precision raises it; high recall lowers it."
                )

                # Result Sorting
                sort_by = st.selectbox(
                    "Sort results by",
                    ["Similarity Score", "Face Size", "Date Added", "File Name"],
                    index=0,
                    key="face_search_sort_by",
                    format_func=SORT_LABELS.__getitem__,
                    help="Choose how search results are ordered."
                )

            # Advanced Filters
            with st.expander("Additional filters", expanded=False):
                aspect_ratio_filter = st.checkbox(
                    "Hide unusual face aspect ratios",
                    value=True,
                    key="face_search_aspect_ratio_filter",
                    help="Hide boxes with unusual width-to-height ratios."
                )

            # Facial Attribute Analysis Info
            with st.expander("Facial attribute analysis", expanded=False):
                st.markdown("""
                **Analyze facial attributes**

                View model-estimated attributes for a search result:

                **Available attributes:**
                - **👤 Age**: Approximate model estimate; accuracy varies with the image
                - **⚧️ Gender**: Approximate model category; not a verified demographic attribute
                - **😊 Emotions**: Happy, Neutral, Sad, Angry, Surprised, Fearful, Disgusted
                - **🌍 Ethnicity**: Asian, Caucasian, Middle Eastern, Indian, Latino, African

                **Important:**
                - Based on DeepFace library with neural networks
                - Model outputs may be wrong and are not calibrated for decision-making
                - Category scores are model outputs, not verified probabilities

                Estimates can be wrong and are not verified personal or demographic information.
                """)

            # Step 1: Analyze uploaded image for faces
            if st.button(
                "Detect faces", type="primary", icon=":material/face:"
            ):
                # Store search parameters in session state for enhanced search
                st.session_state.min_face_size = min_face_size
                st.session_state.search_mode = search_mode
                st.session_state.sort_by = sort_by
                st.session_state.aspect_ratio_filter = aspect_ratio_filter

                analyze_uploaded_image_for_faces(uploaded_file, max_results, similarity_threshold)

    with col2:
        # Step 2: Show face selection if faces were detected
        if 'detected_faces' in st.session_state and st.session_state.detected_faces:
            show_face_selection_interface()

    # Step 3: Display search results - ALWAYS show if available (moved outside col2 to persist after popup interactions)
    if st.session_state.search_results:
        st.markdown("---")

        # Add clear results button
        col_results, col_clear = st.columns([4, 1])
        with col_results:
            st.subheader("Search results", icon=":material/manage_search:")
        with col_clear:
            if st.button("🗑️ Clear", help="Clear search results", key="clear_search_results"):
                st.session_state.search_results = []
                if 'detected_faces' in st.session_state:
                    del st.session_state.detected_faces
                if 'current_uploaded_file' in st.session_state:
                    del st.session_state.current_uploaded_file
                st.rerun()

        display_search_results(st.session_state.search_results)

def load_query_image(uploaded_file):
    """Decode an uploaded image and return ``(image, error_message)``.

    ``Image.open`` only reads the header, so empty or truncated files either
    fail immediately or later in ``st.image``. Both cases are turned into a
    readable message instead of a traceback.
    """
    try:
        image = load_uploaded_image(uploaded_file)
        if image is None:
            raise ValueError("Image decoding failed")
        return image, None
    except Exception as exc:  # noqa: BLE001 - any decode failure means "invalid"
        logger.warning(f"Could not decode uploaded image: {exc}")
        return None, (
            "The uploaded file is not a readable image. Choose a valid JPG, PNG, "
            "BMP, or WEBP file. The file may be empty or damaged."
        )


def analyze_uploaded_image_for_faces(uploaded_file, max_results: int, similarity_threshold: float):
    """Analyze uploaded image and detect all faces"""

    # A failed analysis must not leave the previous upload searchable. Without
    # clearing these values first, a no-face/error result could still show the
    # old face-selection cards and results, making users believe they searched
    # the newly selected image.
    st.session_state.pop('detected_faces', None)
    st.session_state.search_results = []
    st.session_state.pop('current_uploaded_file', None)

    with st.spinner("🔍 Analyzing image and detecting faces..."):
        try:
            # Decode each upload directly from its own in-memory stream. A
            # shared /tmp filename allowed concurrent Streamlit sessions to
            # overwrite each other's query image.
            uploaded_image, image_error = load_query_image(uploaded_file)
            if uploaded_image is None:
                st.error(image_error or "❌ Error loading image. Supported formats: JPG, PNG, BMP, WEBP")
                return

            query_image = np.asarray(uploaded_image, dtype=np.uint8)
            if query_image.shape[0] < 50 or query_image.shape[1] < 50:
                st.error("❌ This image is too small. Choose an image that is at least 50 × 50 pixels.")
                return

            # Detect faces
            face_locations = st.session_state.face_engine.detect_faces(query_image)

            if not face_locations:
                st.error("""
                    ❌ **No faces detected.**

                **Tips for better results:**
                    - Use an image with a clearly visible face
                    - Make sure the face is well lit
                    - Avoid very small faces (minimum size: 30 × 30 pixels)
                    - Try a photo with a more frontal face view
                """)
                return

            # Store detected faces in session state
            st.session_state.detected_faces = {
                'image_sha256': hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
                'face_locations': face_locations,
                'query_image': query_image,
                'max_results': max_results,
                'similarity_threshold': similarity_threshold,
            }

            st.success(f"✅ {len(face_locations)} face(s) detected! Select the desired face for search on the right.")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error analyzing image: {str(e)}")

def show_face_selection_interface():
    """Show interface for selecting which face to search for"""

    st.subheader("👥 Face Selection")

    detected_data = st.session_state.detected_faces
    face_locations = detected_data['face_locations']
    query_image = detected_data['query_image']

    st.write(f"**{len(face_locations)} face(s) detected. Select the face for search:**")

    # A crop-only gallery hides where a face came from in group photos. Show
    # the full query with the same indices used by the thumbnails and buttons.
    source_height, source_width = query_image.shape[:2]
    display_scale = min(1.0, 1600.0 / max(source_height, source_width))
    display_size = (
        max(1, int(round(source_width * display_scale))),
        max(1, int(round(source_height * display_scale))),
    )
    overview = (
        query_image.copy()
        if display_scale == 1.0
        else cv2.resize(query_image, display_size, interpolation=cv2.INTER_AREA)
    )
    image_height, image_width = overview.shape[:2]
    font_scale = max(0.5, min(1.5, min(image_height, image_width) / 600.0))
    line_width = max(2, int(font_scale * 2))
    for idx, face_loc in enumerate(face_locations):
        if isinstance(face_loc, dict):
            top, right, bottom, left = (
                face_loc['top'], face_loc['right'], face_loc['bottom'], face_loc['left']
            )
        else:
            top, right, bottom, left = face_loc
        top = int(round(top * display_scale))
        right = int(round(right * display_scale))
        bottom = int(round(bottom * display_scale))
        left = int(round(left * display_scale))
        color = (25, 230, 90)  # RGB image: bright green outline
        cv2.rectangle(overview, (left, top), (right, bottom), color, line_width)
        label = str(idx + 1)
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
        )
        label_top = max(0, top - label_height - baseline - 6)
        cv2.rectangle(
            overview,
            (left, label_top),
            (left + label_width + 8, min(image_height - 1, label_top + label_height + baseline + 6)),
            color,
            thickness=-1,
        )
        cv2.putText(
            overview,
            label,
            (left + 4, min(image_height - baseline - 1, label_top + label_height + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_width,
            cv2.LINE_AA,
        )
    st.image(overview, caption="Numbers match the face previews and search buttons.", width='stretch', alt="Uploaded image with numbered face detections")

    # Display faces in a grid
    cols = st.columns(min(len(face_locations), 3))

    for idx, face_loc in enumerate(face_locations):
        with cols[idx % 3]:
            # Extract face region
            if isinstance(face_loc, dict):
                top, right, bottom, left = face_loc['top'], face_loc['right'], face_loc['bottom'], face_loc['left']
            else:
                top, right, bottom, left = face_loc

            # Preview the exact aligned/padded crop that will be embedded.
            face_image = st.session_state.face_engine.get_face_crop(
                query_image, (top, right, bottom, left)
            )
            if face_image is None:
                face_image = query_image[top:bottom, left:right]
            face_thumbnail = create_thumbnail(face_image, (120, 120))

            st.image(face_thumbnail, caption=f"Face {idx+1}", alt=f"Detected face {idx+1}")

            # Search button for this face with enhanced parameters
            if st.button(f"Search face {idx+1}", key=f"search_face_{idx}", width='stretch', icon=":material/search:"):
                search_with_selected_face_enhanced(idx)

    # Clear faces button - only clear face detection, preserve search results
    if st.button("Reset face detection", key="clear_faces", icon=":material/refresh:"):
        if 'detected_faces' in st.session_state:
            del st.session_state.detected_faces
        st.rerun()

def search_with_selected_face_enhanced(face_index: int):
    """Perform enhanced search using the selected face with advanced filtering"""

    # Never leave results from a previous face visible when this search fails
    # to extract an embedding or returns no matches.
    st.session_state.search_results = []

    detected_data = st.session_state.detected_faces
    face_locations = detected_data['face_locations']
    query_image = detected_data['query_image']
    max_results = int(st.session_state.get(
        'face_search_max_results', detected_data['max_results']
    ))
    similarity_threshold = float(st.session_state.get(
        'face_search_similarity_threshold', detected_data['similarity_threshold']
    ))

    # Get advanced search parameters from session state
    search_params = {
        'min_face_size': int(st.session_state.get(
            'face_search_min_face_size', st.session_state.get('min_face_size', 30)
        )),
        'search_mode': st.session_state.get(
            'face_search_mode', st.session_state.get('search_mode', 'Balanced')
        ),
        'sort_by': st.session_state.get(
            'face_search_sort_by', st.session_state.get('sort_by', 'Similarity Score')
        ),
        'aspect_ratio_filter': st.session_state.get(
            'face_search_aspect_ratio_filter', st.session_state.get('aspect_ratio_filter', True)
        ),
    }

    selected_face_location = face_locations[face_index]

    with st.spinner(f"🔍 Searching indexed faces for face {face_index + 1}..."):
        try:
            # Extract embedding for selected face
            query_embedding = st.session_state.face_engine.extract_face_embedding(query_image, selected_face_location)

            if query_embedding is None:
                st.error("❌ Could not extract face features. Try another face or image.")
                return

            # Adjust search parameters based on mode
            adjusted_threshold = similarity_threshold
            adjusted_max_results = max_results

            if search_params['search_mode'] == "High Precision":
                adjusted_threshold = min(similarity_threshold + 0.15, 0.9)
                adjusted_max_results = min(max_results, 25)
            elif search_params['search_mode'] == "High Recall":
                adjusted_threshold = max(similarity_threshold - 0.1, 0.2)
                adjusted_max_results = max_results * 2

            # Search for similar faces
            # Fetch a wider candidate pool so the UI filters can remove small
            # or weak results without leaving the requested result page empty.
            requested_results = max(1, adjusted_max_results)
            candidate_limit = min(500, max(requested_results, requested_results * 5))
            similar_faces = st.session_state.vector_store.search_similar_faces(
                query_embedding,
                n_results=candidate_limit,
                min_similarity=adjusted_threshold
            )

            # Apply advanced filters
            if similar_faces:
                filtered_faces = apply_advanced_face_filters(similar_faces, search_params)
                eligible_count = len(filtered_faces)

                # Sort results based on selected criteria
                sorted_faces = sort_search_results(
                    filtered_faces, search_params['sort_by']
                )[:requested_results]

                # Store results
                st.session_state.search_results = sorted_faces
                st.session_state.current_page = 1

                # Show enhanced results summary
                if sorted_faces:
                    avg_similarity = sum(face['similarity'] for face in sorted_faces) / len(sorted_faces)
                    top_similarity = max(face['similarity'] for face in sorted_faces)

                    st.success(f"""
                    🎯 **Enhanced Search Results for Face {face_index + 1}**

                    📊 **Search summary:**
                    - Candidates retrieved: {len(similar_faces)}
                    - Candidates after filters: {eligible_count}
                    - Results displayed: {len(sorted_faces)}
                    - Highest displayed similarity: {top_similarity*100:.1f}%
                    - Average displayed similarity: {avg_similarity*100:.1f}%
                    - Search Mode: {search_params['search_mode']}
                    - Minimum similarity: {adjusted_threshold:.0%}
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **No faces match the current filters for face {face_index + 1}**

                    **Try adjusting:**
                    - Lower similarity threshold (currently: {similarity_threshold*100:.0f}%)
                    - Switch to "High Recall" search mode
                    - Reduce minimum face size filter
                    """)
            else:
                st.warning(f"No similar faces found for Face {face_index + 1}")

            st.rerun()

        except Exception as e:
            st.error(f"❌ Enhanced search failed: {str(e)}")

def apply_advanced_face_filters(faces, search_params):
    """Apply score and optional geometry filters without rejecting unknown boxes."""
    filtered_faces = []

    for face in faces:
        try:
            metadata = face.get('metadata', {})
            face_width = face_height = None
            location = metadata.get('location')
            if isinstance(location, str):
                coords = location.split(',')
            elif isinstance(location, (tuple, list)):
                coords = list(location)
            else:
                coords = []
            if len(coords) == 4:
                try:
                    top, right, bottom, left = map(int, coords)
                    if right > left and bottom > top:
                        face_width = right - left
                        face_height = bottom - top
                except (TypeError, ValueError):
                    pass

            if face_width is not None and face_height is not None:
                face_size = min(face_width, face_height)
                if face_size < search_params['min_face_size']:
                    continue
                # Aspect ratio filter
                if search_params['aspect_ratio_filter']:
                    aspect_ratio = face_width / max(face_height, 1)
                    if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # Skip extremely wide/tall faces
                        continue

            # These options are cosine thresholds; face size is controlled by
            # the separate geometry filter above.
            filtered_faces.append(face)

        except Exception:
            # Skip faces with malformed metadata
            continue

    return filtered_faces

def sort_search_results(faces, sort_by):
    """Sort search results by specified criteria"""
    try:
        if sort_by == "Similarity Score":
            return sorted(faces, key=lambda x: x['similarity'], reverse=True)
        elif sort_by == "Face Size":
            def get_face_size(face):
                try:
                    location_str = face.get('metadata', {}).get('location', '0,0,0,0')
                    coords = location_str.split(',')
                    if len(coords) == 4:
                        top, right, bottom, left = map(int, coords)
                        return (right - left) * (bottom - top)
                    return 0
                except:
                    return 0
            return sorted(faces, key=get_face_size, reverse=True)
        elif sort_by == "File Name":
            return sorted(faces, key=lambda x: x.get('metadata', {}).get('image_path', ''))
        elif sort_by == "Date Added":
            return sorted(faces, key=lambda x: x.get('metadata', {}).get('added_at', ''), reverse=True)
        else:
            return faces
    except Exception:
        # Fall back to similarity sorting
        return sorted(faces, key=lambda x: x['similarity'], reverse=True)
