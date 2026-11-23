"""
Face Recognition Search Application - Streamlit Frontend
"""
# Set up environment before importing other modules to prevent threading issues
from setup_env import setup_environment
setup_environment()

import warnings
# Suppress the pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
import os
import csv
import io
import streamlit as st
import asyncio
import hashlib
import json
from urllib.parse import urlparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
import time
import logging
import tempfile
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# Import our modules. The heavy/optional dependencies are guarded so that a
# missing package produces a readable message instead of a traceback.
import_errors = {}

try:
    from face_recognition_engine import FaceRecognitionEngine
except Exception as exc:  # noqa: BLE001 - surface any import-time failure to the UI
    FaceRecognitionEngine = None
    import_errors["Face recognition (face-recognition, deepface)"] = (
        "pip install face-recognition deepface", exc
    )

try:
    from vector_store import FaceVectorStore
except Exception as exc:  # noqa: BLE001
    FaceVectorStore = None
    import_errors["Vector database (chromadb)"] = ("pip install chromadb", exc)

try:
    from old_image_scraper import AdvancedImageScraper, advanced_scraper
except Exception as exc:  # noqa: BLE001
    AdvancedImageScraper = None
    advanced_scraper = None
    import_errors["Web-Scraping (aiohttp, playwright)"] = (
        "pip install aiohttp playwright beautifulsoup4", exc
    )

from duplicate_detector import delete_duplicate_files, duplicate_detector
from portable_export import create_face_export, restore_face_export
from utils import (
    IMAGE_EXTENSIONS, load_and_preprocess_image, load_uploaded_image,
    create_thumbnail, get_image_files, parse_face_location
)
from ui_cache import face_thumbnail
from async_utils import run_coroutine
from settings_store import (
    active_embedding_model,
    active_grouping_threshold,
    load_settings,
    save_settings,
    settings_path,
)
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT,
    RESULTS_PER_PAGE, THUMBNAIL_SIZE,
    IMAGES_DIR, SCRAPED_DIR
)

IMAGE_SUFFIXES = tuple(sorted(IMAGE_EXTENSIONS))

PAGE_LABELS = {
    "face_search": "Face search",
    "image_import": "Import images",
    "batch_processing": "Batch processing",
    "web_import": "Web import",
    "face_gallery": "Face library",
    "people": "People",
    "person_groups": "Review groups",
    "duplicates": "Duplicates",
    "database_statistics": "Database statistics",
    "settings": "Settings",
}
PAGES = tuple(PAGE_LABELS)

LEGACY_PAGE_IDS = {
    "🔍 Face Search": "face_search",
    "📥 Image Upload & Processing": "image_import",
    "🧠 Batch Face Processing": "batch_processing",
    "🌐 Web Scraping": "web_import",
    "👥 Face Gallery": "face_gallery",
    "🏷️ Name Gallery": "people",
    "🧩 Person Groups": "person_groups",
    "🔧 Duplicate Manager": "duplicates",
    "📊 Database Statistics": "database_statistics",
    "⚙️ Settings": "settings",
}


def _stable_page_id(value: str) -> str:
    """Migrate older display-label navigation state to stable route IDs."""
    if value in PAGE_LABELS:
        return value
    return LEGACY_PAGE_IDS.get(value, "face_search")

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

# These pages add faces during their own run. An empty-state notice rendered
# before their upload/process action would still be visible afterwards and
# contradict the success summary until the next rerun.
DATA_ENTRY_PAGES = (
    "image_import",
    "batch_processing",
    "web_import",
)


def count_images(directory: Path, recursive: bool = False) -> int:
    """Count image files in a directory, optionally including subdirectories."""
    directory = Path(directory)
    if not directory.exists():
        return 0

    found = set()
    for suffix in IMAGE_SUFFIXES:
        for pattern in (f"*{suffix}", f"*{suffix.upper()}"):
            globber = directory.rglob if recursive else directory.glob
            found.update(path for path in globber(pattern) if path.is_file())
    return len(found)


def show_embedding_pipeline_status() -> None:
    """Explain when stored vectors need to be rebuilt after model-pipeline changes."""
    store = st.session_state.get("vector_store")
    if store is None:
        return
    status = store.get_embedding_pipeline_status()
    if status.get("reindex_required"):
        st.warning(
            "The saved face index was created with a different detection or embedding pipeline "
            "(for example, after switching between Buffalo_L and ArcFace or changing the detector). "
            "Search is paused until the index is updated so incompatible embeddings are not compared. "
            "Open **Batch processing**, choose **Update All Images**, then update "
            "**Images and subfolders** and **Scraped Images**."
        )
    elif status.get("reindex_in_progress"):
        st.info(
            f"Index update in progress: {status.get('pending_faces', 0):,} entries remain. "
            "Search results include only updated images until the process finishes. "
            "Update all image folders in Batch processing."
        )


def discover_image_directories() -> List[Any]:
    """Return the (label, path) pairs the batch processor can work on.

    Uses the configured data directory instead of a hardcoded ``data/`` path,
    so FACE_DATA_DIR and custom layouts are honoured.
    """
    available: List[Any] = []

    # The root option is the reliable way to reprocess a full library. Include
    # nested downloaded folders as well as files placed directly in images/.
    images_count = count_images(IMAGES_DIR, recursive=True)
    if images_count:
        available.append((f"🖼️ Images and subfolders ({images_count} images)", str(IMAGES_DIR)))

    if Path(IMAGES_DIR).exists():
        downloaded = [
            directory
            for directory in Path(IMAGES_DIR).iterdir()
            if directory.is_dir() and directory.name.startswith("downloaded_images")
        ]
        for directory in sorted(downloaded, key=lambda item: item.stat().st_mtime, reverse=True):
            count = count_images(directory)
            if not count:
                continue
            stamp = datetime.fromtimestamp(directory.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            available.append((f"📥 {directory.name} ({count} images, {stamp})", str(directory)))

    scraped_count = count_images(SCRAPED_DIR, recursive=True)
    if scraped_count:
        available.append((f"🕷️ Scraped Images ({scraped_count} images)", str(SCRAPED_DIR)))

    return available


def parse_scrape_urls(raw: str) -> tuple:
    """Split pasted lines into (valid, invalid) URLs.

    Bare domains are upgraded to https://; anything without a plausible host is
    reported back instead of being handed to the scraper.
    """
    valid: List[str] = []
    invalid: List[str] = []

    for line in raw.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        with_scheme = candidate if "://" in candidate else f"https://{candidate}"
        parsed = urlparse(with_scheme)
        host = parsed.hostname or ""
        if (
            parsed.scheme in ("http", "https")
            and "." in host
            and " " not in host
            and not parsed.username
            and not parsed.password
        ):
            if len(valid) >= 10:
                invalid.append(candidate)
                continue
            valid.append(with_scheme)
        else:
            invalid.append(candidate)

    return valid, invalid




# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Initialize session state
if FaceRecognitionEngine is not None and 'face_engine' not in st.session_state:
    st.session_state.face_engine = FaceRecognitionEngine()

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.vector_store_error = None
    if FaceVectorStore is not None:
        try:
            st.session_state.vector_store = FaceVectorStore()
        except Exception as exc:  # noqa: BLE001 - shown in the UI below
            st.session_state.vector_store_error = f"{type(exc).__name__}: {exc}"

if 'image_scraper' not in st.session_state:
    st.session_state.image_scraper = advanced_scraper

if 'search_results' not in st.session_state:
    st.session_state.search_results = []

if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

# Initialize face search state variables
if 'detected_faces' not in st.session_state:
    st.session_state.detected_faces = None

# Initialize persons cache to avoid repeated expensive database queries
if 'cached_all_persons' not in st.session_state:
    st.session_state.cached_all_persons = []
if 'persons_cache_dirty' not in st.session_state:
    st.session_state.persons_cache_dirty = True

def get_cached_all_persons():
    """Get all persons with caching to avoid repeated expensive database queries"""
    if st.session_state.persons_cache_dirty or not st.session_state.cached_all_persons:
        st.session_state.cached_all_persons = st.session_state.vector_store.get_all_persons()
        st.session_state.persons_cache_dirty = False
    return st.session_state.cached_all_persons

def invalidate_persons_cache():
    """Mark persons cache as dirty for refresh"""
    st.session_state.persons_cache_dirty = True
    # Name changes also change which faces are still unnamed, so the reviewed
    # group suggestions must be recomputed on the next visit.
    st.session_state.person_groups_cache = None


def invalidate_gallery_cache():
    """Compatibility hook; gallery rows are fetched fresh from the database."""
    return None


def faces_to_csv(faces) -> str:
    """Serialize the gallery rows (not the embeddings) for download."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["face_id", "image_path", "location", "person_name", "added_at"])
    for face in faces:
        metadata = face.get("metadata") or {}
        writer.writerow([
            face.get("face_id", ""),
            metadata.get("image_path", ""),
            metadata.get("location", ""),
            metadata.get("full_name", ""),
            metadata.get("added_at", ""),
        ])
    return buffer.getvalue()


def show_assignment_history(face_id: str) -> None:
    """Show the per-face review trail for identity changes when available."""
    try:
        face = st.session_state.vector_store.get_face_by_id(face_id)
        raw_history = (face or {}).get("metadata", {}).get("assignment_history", "[]")
        history = json.loads(raw_history) if isinstance(raw_history, str) else []
    except (TypeError, ValueError, AttributeError):
        history = []
    if not history:
        return
    with st.expander("Identity assignment history"):
        for event in reversed(history):
            before = event.get("from_name") or "Unassigned"
            after = event.get("to_name") or "Unassigned"
            st.caption(f"{event.get('at', 'Unknown time')} · {before} → {after} · {event.get('actor', 'local-user')}")

def main():
    """Main application function"""

    if import_errors:
        st.error("❌ Required dependencies are missing. The app cannot start.")
        for name, (install, exc) in import_errors.items():
            st.markdown(f"- **{name}** – `{install}`")
            st.caption(f"{type(exc).__name__}: {exc}")
        st.info("Restart the app after installing the dependencies.")
        st.stop()

    if st.session_state.get('vector_store') is None:
        st.error("❌ The vector database could not be opened.")
        st.code(st.session_state.get('vector_store_error') or "Unknown error")
        st.info("Check that the data directory exists and is writable.")
        st.stop()

    st.title("Face Search", icon=":material/face:")

    # Messages from a processing run that deliberately triggered a rerun so the
    # sidebar stats reflect the newly written faces.
    flash_messages = st.session_state.pop("_flash_messages", [])
    for flash_kind, flash_message in flash_messages:
        getattr(st, flash_kind)(flash_message)
    if any(kind == "success" for kind, _ in flash_messages):
        if st.button("Search indexed faces", key="search_after_import"):
            st.session_state.pending_page = "face_search"
            st.rerun()

    # Buttons deep in a page queue a navigation target; applying it before the
    # selectbox is instantiated is the supported way to move the sidebar.
    if "pending_page" in st.session_state:
        st.session_state.nav_page = _stable_page_id(
            st.session_state.pop("pending_page")
        )
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "face_search"
    else:
        st.session_state.nav_page = _stable_page_id(st.session_state.nav_page)


    # Sidebar
    with st.sidebar:
        st.header("Navigation", icon=":material/menu:")
        page = st.selectbox(
            "Section", PAGES, key="nav_page", format_func=PAGE_LABELS.__getitem__
        )

        # Quick stats
        stats = st.session_state.vector_store.get_collection_stats()
        with st.container(border=True):
            st.subheader("Library", icon=":material/database:")
            if "error" not in stats:
                st.metric("Faces", stats.get("total_faces", 0))
                st.metric("Images", stats.get("unique_images", 0))
            else:
                st.warning("Database unavailable")

    # Handle page changes and clear modal states when switching pages
    if 'active_page' in st.session_state:
        st.session_state.active_page = _stable_page_id(st.session_state.active_page)
    if 'active_page' not in st.session_state:
        st.session_state.active_page = page
    elif st.session_state.active_page != page:
        # Page changed - clear all modal states
        modal_keys_to_clear = [
            'info_face_id', 'info_image_path', 'info_face_location',
            'full_image_path', 'full_image_face_location',
            'analysis_image_path', 'analysis_face_location', 'analysis_face_id',
            'name_assign_face_id', 'name_assign_image_path', 'name_assign_face_location'
        ]

        for key in modal_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # Update active page
        st.session_state.active_page = page

    # A fresh install has no faces yet. Point the user at the two ways to add
    # data, but not on the pages that add data themselves (see DATA_ENTRY_PAGES).
    if page not in DATA_ENTRY_PAGES:
        try:
            empty_state_stats = st.session_state.vector_store.get_collection_stats()
        except Exception:  # noqa: BLE001 - the sidebar will still show its own error
            empty_state_stats = {}
        if "error" not in empty_state_stats and empty_state_stats.get("total_faces", 0) == 0:
            st.info(
                "📭 Your database is empty. Upload images in "
                "**Import images** or start an import from "
                "**Web import**. Detected faces will appear in "
                "the face library."
            )
            if st.button("Add local images", key="empty_state_import_images", type="primary"):
                st.session_state.pending_page = "image_import"
                st.rerun()

    # Check for modal displays - they will show as overlays, don't interrupt main page
    if 'info_face_id' in st.session_state and 'info_image_path' in st.session_state and 'info_face_location' in st.session_state:
        show_face_info_modal()
        # Don't return - let main page continue to render

    # Legacy modal support (for backward compatibility)
    if 'full_image_path' in st.session_state and 'full_image_face_location' in st.session_state:
        show_full_image_modal()
        # Don't return - let main page continue to render

    if 'analysis_image_path' in st.session_state and 'analysis_face_location' in st.session_state:
        show_analysis_modal()
        # Don't return - let main page continue to render

    if 'name_assign_face_id' in st.session_state:
        show_name_assignment_modal()
        # Don't return - let main page continue to render

    # Main content area - always render, modals will overlay
    if page == "face_search":
        face_search_page()
    elif page == "image_import":
        upload_processing_page()
    elif page == "batch_processing":
        batch_face_processing_page()
    elif page == "web_import":
        web_scraping_page()
    elif page == "face_gallery":
        face_gallery_page()
    elif page == "people":
        name_gallery_page()
    elif page == "person_groups":
        person_groups_page()
    elif page == "duplicates":
        duplicate_manager_page()
    elif page == "database_statistics":
        database_stats_page()
    elif page == "settings":
        settings_page()

    # A successful upload changes the sidebar stats, which were rendered before
    # the page body ran. Refresh once the page function has returned.
    if st.session_state.pop("_refresh_after_processing", False):
        st.rerun()

def face_search_page():
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

        except Exception as e:
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
    except Exception as e:
        # Fall back to similarity sorting
        return sorted(faces, key=lambda x: x['similarity'], reverse=True)

def search_with_selected_face(face_index: int):
    """Perform search using the selected face"""

    detected_data = st.session_state.detected_faces
    face_locations = detected_data['face_locations']
    query_image = detected_data['query_image']
    max_results = int(st.session_state.get(
        'face_search_max_results', detected_data['max_results']
    ))
    similarity_threshold = float(st.session_state.get(
        'face_search_similarity_threshold', detected_data['similarity_threshold']
    ))

    selected_face_location = face_locations[face_index]

    with st.spinner(f"🔍 Searching for similar faces for face {face_index + 1}..."):
        try:
            # Extract embedding for selected face
            query_embedding = st.session_state.face_engine.extract_face_embedding(query_image, selected_face_location)

            if query_embedding is None:
                st.error("❌ Could not extract face features. Try another face or image.")
                return

            # Search for similar faces
            similar_faces = st.session_state.vector_store.search_similar_faces(
                query_embedding,
                n_results=max_results,
                min_similarity=similarity_threshold
            )

            # Store results
            st.session_state.search_results = similar_faces

            # Show results summary
            if similar_faces:
                avg_similarity = sum(face['similarity'] for face in similar_faces) / len(similar_faces)
                top_similarity = similar_faces[0]['similarity'] if similar_faces else 0

                st.success(f"""
                🎯 **Found {len(similar_faces)} similar faces for face {face_index + 1}!**

                📊 **Match quality:**
                - Top similarity: {top_similarity*100:.1f}%
                - Average similarity: {avg_similarity*100:.1f}%
                """)
            else:
                st.warning(f"""
                ⚠️ **No similar faces found for face {face_index + 1}**

                **Try this:**
                - Lower the similarity threshold (currently {similarity_threshold*100:.0f}%)
                - Select a different face in the image
                - Add more images to the database
                """)

            st.rerun()

        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")

def show_full_image_with_face_box(image_path, face_location):
    """Store image data and trigger full image modal display"""
    try:
        # Store data in session state for the modal
        st.session_state.full_image_path = image_path
        st.session_state.full_image_face_location = face_location

        # The modal will be triggered by the session state check
        st.rerun()

    except Exception as e:
        st.error(f"❌ Could not prepare the full image view: {str(e)}")

def analyze_facial_attributes(image_path, face_location):
    """Analyze facial attributes using DeepFace"""
    try:
        import tempfile
        from deepface import DeepFace

        # Load the full image
        full_image = load_and_preprocess_image(image_path)
        if full_image is None:
            return None

        # Parse face location
        if isinstance(face_location, str):
            coords = face_location.split(',')
            if len(coords) == 4:
                top, right, bottom, left = map(int, coords)
            else:
                return None
        else:
            top, right, bottom, left = face_location

        # Extract face region with some padding
        face_width = right - left
        face_height = bottom - top
        padding = max(10, min(30, min(face_width, face_height) // 8))

        height, width = full_image.shape[:2]
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(height, bottom + padding)
        right = min(width, right + padding)

        face_image = full_image[top:bottom, left:right]

        # Validate face size
        if face_image.shape[0] < 50 or face_image.shape[1] < 50:
            return {"error": "Face is too small for attribute analysis"}

        # Save face to temporary file for DeepFace
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Convert RGB to BGR for cv2.imwrite
            face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_file.name, face_bgr)

            # Analyze with DeepFace
            try:
                analysis_result = DeepFace.analyze(
                    img_path=tmp_file.name,
                    actions=['age', 'gender', 'race', 'emotion'],
                    enforce_detection=False,
                    silent=True
                )

                # Clean up temp file
                Path(tmp_file.name).unlink(missing_ok=True)

                # Handle different return formats from DeepFace
                if isinstance(analysis_result, list):
                    result = analysis_result[0] if analysis_result else {}
                else:
                    result = analysis_result

                return result

            except Exception as df_error:
                # Clean up temp file
                Path(tmp_file.name).unlink(missing_ok=True)
                logger.error(f"DeepFace analysis error: {df_error}")
                return {"error": f"Analysis error: {str(df_error)[:100]}..."}

    except ImportError:
        return {"error": "DeepFace is not installed. Install it with: pip install deepface"}
    except Exception as e:
        logger.error(f"Facial attribute analysis error: {e}")
        return {"error": f"Unexpected error: {str(e)[:100]}..."}

def show_facial_attributes_popup(image_path, face_location, face_id="unknown"):
    """Display facial attributes analysis in a full-width dialog-style container"""

    with st.spinner("Analyzing facial attributes..."):
        analysis = analyze_facial_attributes(image_path, face_location)

    if analysis is None:
        st.error("❌ Could not analyze facial attributes.")
        return

    if "error" in analysis:
        st.error(f"❌ {analysis['error']}")
        return

    # Create a dialog-style container that takes full width
    st.markdown("---")  # Visual separator

    # Create a container that uses the full width
    with st.container():
        # Header with larger title
        st.markdown("# Facial attribute analysis")
        st.markdown(f"**Face ID:** `{face_id}`")
        st.markdown(f"**File:** `{Path(image_path).name}`")

        # Show face preview in a separate column layout
        col_preview, col_spacer = st.columns([1, 3])

        with col_preview:
            try:
                full_image = load_and_preprocess_image(image_path)
                if full_image is not None:
                    if isinstance(face_location, str):
                        coords = face_location.split(',')
                        if len(coords) == 4:
                            top, right, bottom, left = map(int, coords)
                            face_crop = full_image[top:bottom, left:right]
                            if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                                st.image(face_crop, caption="Analyzed face", width=200, alt="Face used for attribute analysis")
            except:
                pass

        st.markdown("---")

        # Analysis results in a clean 2-column layout
        col1, col2 = st.columns(2)

        with col1:
            # Age Analysis
            if 'age' in analysis:
                st.markdown("## Age estimate")
                age = analysis['age']
                st.metric("Estimated age", f"{age} years", help="A model estimate; accuracy is not guaranteed.")

                # Age category with visual indicator
                if age < 18:
                    st.info("**Category:** Teenager")
                elif age < 30:
                    st.info("**Category:** Young adult")
                elif age < 50:
                    st.info("**Category:** Adult")
                elif age < 70:
                    st.info("**Category:** Older adult")
                else:
                    st.info("**Category:** Senior")

            st.markdown("---")

            # Gender Analysis
            if 'gender' in analysis:
                st.markdown("## Gender estimate")
                gender_data = analysis['gender']
                if isinstance(gender_data, dict):
                    male_prob = float(gender_data.get('Man', 0))
                    female_prob = float(gender_data.get('Woman', 0))

                    # Visual representation with progress bars
                    st.write("**Man**")
                    st.progress(male_prob / 100.0)
                    st.write(f"`{male_prob:.1f}%`")

                    st.write("**Woman**")
                    st.progress(female_prob / 100.0)
                    st.write(f"`{female_prob:.1f}%`")

                    # Highlight prediction
                    dominant = "Man" if male_prob > female_prob else "Woman"
                    confidence = max(male_prob, female_prob)
                    st.success(f"**Model output:** {dominant} ({confidence:.1f}%)")
                else:
                    st.write(f"**Gender estimate:** {gender_data}")

        with col2:
            # Emotion Analysis
            if 'emotion' in analysis:
                st.markdown("## Emotion estimate")
                emotions = analysis['emotion']
                if isinstance(emotions, dict):
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                    emotion_map = {
                        'happy': ('Happy', '#28a745'),
                        'neutral': ('😐 Neutral', '#6c757d'),
                        'sad': ('Sad', '#007bff'),
                        'angry': ('Angry', '#dc3545'),
                        'surprise': ('Surprised', '#ffc107'),
                        'fear': ('Fearful', '#fd7e14'),
                        'disgust': ('Disgusted', '#6f42c1')
                    }

                    # Show all emotions with visual bars
                    for emotion, confidence in sorted_emotions:
                        emotion_label, color = emotion_map.get(emotion, (emotion.title(), '#6c757d'))
                        st.write(f"**{emotion_label}**")
                        st.progress(float(confidence / 100.0))
                        st.write(f"`{confidence:.1f}%`")

                    # Dominant emotion highlight
                    top_emotion, top_confidence = sorted_emotions[0]
                    top_label, _ = emotion_map.get(top_emotion, (top_emotion.title(), '#6c757d'))
                    st.success(f"**Top emotion:** {top_label} ({top_confidence:.1f}%)")

            st.markdown("---")

            # Race/Ethnicity Analysis
            if 'race' in analysis:
                st.markdown("## Ethnicity estimate")
                race_data = analysis['race']
                if isinstance(race_data, dict):
                    sorted_races = sorted(race_data.items(), key=lambda x: x[1], reverse=True)
                    race_map = {
                        'asian': 'Asian',
                        'white': 'White',
                        'middle eastern': 'Middle Eastern',
                        'indian': 'Indian',
                        'latino hispanic': 'Latino/Hispanic',
                        'black': 'Black'
                    }

                    # Show top predictions with progress bars
                    for race, confidence in sorted_races:
                        race_label = race_map.get(race.lower(), race.title())
                        st.write(f"**{race_label}**")
                        st.progress(float(confidence / 100.0))
                        st.write(f"`{confidence:.1f}%`")

                    # Dominant prediction
                    top_race, top_confidence = sorted_races[0]
                    top_race_label = race_map.get(top_race.lower(), top_race.title())
                    st.info(f"**Top model output:** {top_race_label} ({top_confidence:.1f}%)")

        # Footer with important disclaimer
        st.markdown("---")
        st.markdown("### Important notice")
        st.warning("""
        This analysis is generated by AI models and is intended for **demonstration and research only**.

        **Please note:**
        - Results are **estimates** and may be inaccurate.
        - Do not use these outputs for **identification, discrimination, or decision-making**.
        - Respect the privacy and rights of people shown in the images.
        - Accuracy can vary with image quality, lighting, and camera angle.
        """)

        # Close button
        st.markdown("---")
        if st.button("Close analysis", key=f"close_analysis_{face_id}", type="primary"):
            st.rerun()

@st.dialog("Assign a name", width="large")
def show_name_assignment_modal():
    """Enhanced name assignment interface with extended fields and existing person selection"""

    # Required keys for name assignment
    required_keys = ['name_assign_face_id', 'name_assign_image_path', 'name_assign_face_location']
    missing_keys = [key for key in required_keys if key not in st.session_state]

    if missing_keys:
        st.error(f"Missing face data for name assignment: {', '.join(missing_keys)}")
        st.info("Try opening the name assignment again.")
        if st.button("Close"):
            # Clear any partial session state
            for key in required_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return

    face_id = st.session_state.name_assign_face_id
    image_path = st.session_state.name_assign_image_path
    face_location = st.session_state.name_assign_face_location

    # Ensure image_path is a Path object
    if isinstance(image_path, str):
        image_path = Path(image_path)

    # Header info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Face ID:** `{face_id}`")
        st.markdown(f"**File:** `{image_path.name}`")

    # Show face preview
    with col2:
        try:
            full_image = load_and_preprocess_image(image_path)
            face_crop = None
            if full_image is not None:
                if isinstance(face_location, str):
                    coords = face_location.split(',')
                    if len(coords) == 4:
                        top, right, bottom, left = map(int, coords)
                        face_crop = full_image[top:bottom, left:right]
                elif isinstance(face_location, (tuple, list)) and len(face_location) == 4:
                    top, right, bottom, left = face_location
                    face_crop = full_image[top:bottom, left:right]

                if face_crop is not None and face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    st.image(face_crop, caption="Face to assign", width=150, alt="Selected face")
        except Exception as e:
            st.warning(f"Could not display the face: {e}")

    st.markdown("---")

    # Get existing person data if available
    existing_face = st.session_state.vector_store.get_face_by_id(face_id)
    existing_data = {}
    if existing_face and existing_face.get('metadata', {}).get('person_id'):
        metadata = existing_face['metadata']
        existing_data = {
            'first_name': metadata.get('first_name', ''),
            'middle_names': metadata.get('middle_names', ''),
            'last_name': metadata.get('last_name', ''),
            'full_name': metadata.get('full_name', ''),
            'birth_date': metadata.get('birth_date', ''),
            'birth_place': metadata.get('birth_place', ''),
            'notes': metadata.get('notes', ''),
            'person_id': metadata.get('person_id', '')
        }
        st.info(f"✅ **Current name:** {existing_data['full_name']}")

    # Get all existing persons for selection
    all_persons = get_cached_all_persons()  # Use cached version

    # Choice between new person or existing person
    st.subheader("Select a person")
    st.caption(
        "The name will be assigned only to this face. Review similar faces "
        "separately in **Review groups**."
    )

    assignment_type = st.radio(
        "How would you like to assign the name?",
        ["Create a new person", "Select an existing person"],
        key="assignment_type",
        horizontal=True
    )

    if assignment_type == "Select an existing person" and all_persons:
        # Show existing persons selection
        st.subheader("Existing people")

        # Create searchable list of persons
        person_options = {"-- Select a person --": None}
        for person in all_persons:
            display_name = f"{person['full_name']}"
            if person.get('birth_date'):
                display_name += f" (*{person['birth_date']})"
            display_name += f" ({person['face_count']} faces)"
            person_options[display_name] = person

        selected_person_display = st.selectbox(
            "Select a person:",
            options=list(person_options.keys()),
            key="selected_existing_person"
        )

        if selected_person_display and person_options[selected_person_display] is not None:
            selected_person = person_options[selected_person_display]

            # Ensure selected_person is not None
            if selected_person is None:
                st.error("Could not load person data.")
                return

            # Show person details in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Full name:**")
                st.write(f"*{selected_person['full_name']}*")
                st.write(f"**First name:** {selected_person['first_name']}")
                if selected_person.get('middle_names'):
                    st.write(f"**Middle names:** {selected_person['middle_names']}")
                st.write(f"**Last name:** {selected_person['last_name']}")

            with col2:
                st.write(f"**Date of birth:** {selected_person.get('birth_date', '-')}")
                st.write(f"**Place of birth:** {selected_person.get('birth_place', '-')}")
                st.write(f"**Faces:** {selected_person['face_count']}")

            with col3:
                if selected_person.get('notes'):
                    st.write(f"**Notes:**")
                    st.write(selected_person['notes'])

            # Assign button
            if st.button("Assign this person", type="primary", width='stretch', icon=":material/person_add:"):
                person_data = {
                    'first_name': selected_person['first_name'],
                    'middle_names': selected_person.get('middle_names', ''),
                    'last_name': selected_person['last_name'],
                    'birth_date': selected_person.get('birth_date', ''),
                    'birth_place': selected_person.get('birth_place', ''),
                    'notes': selected_person.get('notes', '')
                }

                result = st.session_state.vector_store.assign_person_name(
                    face_id, person_data, selected_person['person_id']
                )

                if result:
                    st.success("Person assigned.")

                    # Invalidate cached names and gallery metadata
                    invalidate_persons_cache()
                    invalidate_gallery_cache()

                    # Clear session state and close dialog
                    _clear_name_assignment_session()
                    st.rerun()
                else:
                    st.error("❌ Could not assign the person.")

    elif assignment_type == "Select an existing person" and not all_persons:
        st.info("No existing people found. Create a new person instead.")

    else:
        # New person creation form
        st.subheader("Create a new person")

        # Extended person form in columns
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Basic information:**")
            first_name = st.text_input(
                "First name *",
                value=existing_data.get('first_name', '') or '',
                key="new_first_name"
            )
            middle_names = st.text_input(
                "Middle names",
                value=existing_data.get('middle_names', '') or '',
                help="For example: Moritz Luitpold Damian",
                key="new_middle_names"
            )
            last_name = st.text_input(
                "Last name *",
                value=existing_data.get('last_name', '') or '',
                key="new_last_name"
            )

        with col2:
            st.write("**Additional information:**")
            birth_date = st.text_input(
                "Date of birth",
                value=existing_data.get('birth_date', ''),
                help="For example: 1985-03-15 or 15.03.1985",
                key="new_birth_date"
            )
            birth_place = st.text_input(
                "Place of birth",
                value=existing_data.get('birth_place', ''),
                help="For example: Munich, Germany",
                key="new_birth_place"
            )

        notes = st.text_area(
            "Notizen:",
            value=existing_data.get('notes', ''),
            help="Additional notes about this person.",
            key="new_notes",
            height=80
        )

        # Preview full name
        if first_name or middle_names or last_name:
            name_parts = [p.strip() for p in [first_name or '', middle_names or '', last_name or ''] if p and p.strip()]
            full_name = ' '.join(name_parts)
            st.success(f"**Full name:** {full_name}")

        # Save new person button
        if st.button("Save person", type="primary", width='stretch', icon=":material/save:"):
            if (first_name and first_name.strip()) or (last_name and last_name.strip()):
                # Create person data
                person_data = {
                    'first_name': (first_name or '').strip(),
                    'middle_names': (middle_names or '').strip(),
                    'last_name': (last_name or '').strip(),
                    'birth_date': (birth_date or '').strip(),
                    'birth_place': (birth_place or '').strip(),
                    'notes': (notes or '').strip()
                }

                result = st.session_state.vector_store.assign_person_name(face_id, person_data)

                if result:
                    st.success("Person created and assigned.")

                    # Invalidate cached names and gallery metadata
                    invalidate_persons_cache()
                    invalidate_gallery_cache()

                    # Clear session state and close dialog
                    _clear_name_assignment_session()
                    st.rerun()
                else:
                    st.error("❌ Could not create the person.")
            else:
                st.warning("⚠️ Enter at least a first or last name.")

    # Action buttons at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if existing_data.get('full_name'):
            if st.button("Remove name", width='stretch'):
                if st.session_state.vector_store.remove_person_name(face_id):
                    st.success("✅ Name removed.")
                    # Invalidate cached names and gallery metadata
                    invalidate_persons_cache()
                    invalidate_gallery_cache()
                    _clear_name_assignment_session()
                    st.rerun()
                else:
                    st.error("❌ Could not remove the name.")

    with col3:
        if st.button("Cancel", width='stretch'):
            _clear_name_assignment_session()
            st.rerun()

def _clear_name_assignment_session():
    """Helper function to clear name assignment session state"""
    keys_to_clear = [
        'name_assign_face_id', 'name_assign_image_path', 'name_assign_face_location',
        'assignment_type', 'selected_existing_person', 'new_first_name',
        'new_middle_names', 'new_last_name', 'new_birth_date',
        'new_birth_place', 'new_notes'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Restore Name Gallery state if it was preserved
    if 'name_gallery_selected_person' in st.session_state:
        st.session_state.selected_person = st.session_state.name_gallery_selected_person
        del st.session_state.name_gallery_selected_person

@st.dialog("Facial attribute analysis", width="large")
def show_analysis_modal():
    """Display the analysis modal as a proper popup dialog"""

    # Make the dialog content even wider using custom CSS
    st.markdown("""
    <style>
    .stDialog > div:first-child > div:first-child > div:first-child {
        width: 95vw !important;
        max-width: 1400px !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.dialog("Face information", width="large")
def show_face_info_modal():
    """Unified modal for all face information: Full Image, Analysis, and Name Assignment"""

    # Make the dialog content even wider using custom CSS
    st.markdown("""
    <style>
    .stDialog > div:first-child > div:first-child > div:first-child {
        width: 95vw !important;
        max-width: 1600px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Required keys for face info display
    required_keys = ['info_face_id', 'info_image_path', 'info_face_location']
    missing_keys = [key for key in required_keys if key not in st.session_state]

    if missing_keys:
        st.error(f"Missing face data: {', '.join(missing_keys)}")
        if st.button("Close"):
            for key in required_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return

    face_id = st.session_state.info_face_id
    image_path = st.session_state.info_image_path
    face_location = st.session_state.info_face_location

    # Ensure image_path is a Path object
    if isinstance(image_path, str):
        image_path = Path(image_path)

    # Header info
    st.markdown(f"**🆔 Face ID:** `{face_id[:20]}...`")
    st.markdown(f"**File:** `{image_path.name}`")
    show_assignment_history(face_id)

    # Tabs for different information types
    tab1, tab2, tab3 = st.tabs(["Full image", "Face analysis", "Assign name"])

    with tab1:
        show_full_image_tab(image_path, face_location)

    with tab2:
        show_analysis_tab(image_path, face_location, face_id)

    with tab3:
        show_name_assignment_tab(face_id, image_path, face_location)

    # Close button
    st.markdown("---")
    if st.button("Close", key="close_face_info_modal", type="primary", width='stretch'):
        # Clear face info data
        for key in required_keys:
            if key in st.session_state:
                del st.session_state[key]

        # Restore Name Gallery state if it was preserved
        if 'name_gallery_selected_person' in st.session_state:
            st.session_state.selected_person = st.session_state.name_gallery_selected_person
            del st.session_state.name_gallery_selected_person

        st.rerun()

def show_full_image_tab(image_path, face_location):
    """Display full image with face bounding box in tab"""
    try:
        # Load the full image
        full_image = load_and_preprocess_image(image_path)
        if full_image is None:
            st.error("❌ Could not load the image.")
            return

        # Parse face location if provided
        image_with_box = full_image.copy()
        if face_location:
            if isinstance(face_location, str):
                coords = face_location.split(',')
                if len(coords) == 4:
                    top, right, bottom, left = map(int, coords)
                    # Draw bounding box on the image
                    cv2.rectangle(image_with_box, (left, top), (right, bottom), (0, 255, 0), 3)
                    st.success(f"Face marked at: top={top}, right={right}, bottom={bottom}, left={left}")
            else:
                top, right, bottom, left = face_location
                cv2.rectangle(image_with_box, (left, top), (right, bottom), (0, 255, 0), 3)
                st.success(f"Face marked at: top={top}, right={right}, bottom={bottom}, left={left}")

        # Display the image with maximum width
        caption = f"Full image - {image_path.name}"
        if face_location:
            caption += " (face highlighted)"

        st.image(image_with_box, caption=caption, width='stretch', alt=caption)

        # Image info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Image size", f"{full_image.shape[1]} × {full_image.shape[0]} px")

        with col2:
            file_size = image_path.stat().st_size if image_path.exists() else 0
            st.metric("File size", f"{file_size / 1024:.1f} KB")

        with col3:
            if face_location:
                st.metric("Face detected", "Yes")
            else:
                st.metric("Face detected", "No")

    except Exception as e:
        st.error(f"❌ Could not display the full image: {str(e)}")

def show_analysis_tab(image_path, face_location, face_id):
    """Display facial analysis in tab"""

    with st.spinner("Analyzing facial attributes..."):
        analysis = analyze_facial_attributes(image_path, face_location)

    if analysis is None:
        st.error("❌ Could not analyze facial attributes.")
        return

    if "error" in analysis:
        st.error(f"❌ {analysis['error']}")
        return

    # Show face preview
    try:
        full_image = load_and_preprocess_image(image_path)
        if full_image is not None and face_location:
            if isinstance(face_location, str):
                coords = face_location.split(',')
                if len(coords) == 4:
                    top, right, bottom, left = map(int, coords)
                    face_crop = full_image[top:bottom, left:right]
                    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                        thumbnail = create_thumbnail(face_crop, (200, 200))
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(thumbnail, caption="Analyzed face", alt="Face used for analysis")
    except:
        pass

    # Analysis results in a clean 2-column layout
    col1, col2 = st.columns(2)

    with col1:
        # Age Analysis
        if 'age' in analysis:
            st.markdown("### 🎂 **Age estimate**")
            age = analysis['age']
            st.metric("Estimated age", f"{age} years", help="Model estimate; accuracy is not guaranteed.")

            # Age category
            if age < 18:
                st.info("📚 Teenager")
            elif age < 30:
                st.info("🌟 Young adult")
            elif age < 50:
                st.info("💼 Adult")
            elif age < 70:
                st.info("🎓 Mature adult")
            else:
                st.info("👴 Senior")

        st.markdown("---")

        # Gender Analysis
        if 'gender' in analysis:
            st.markdown("### ⚧️ **Gender estimate**")
            gender_data = analysis['gender']
            if isinstance(gender_data, dict):
                male_conf = gender_data.get('Man', 0)
                female_conf = gender_data.get('Woman', 0)

                if male_conf > female_conf:
                    st.metric("Gender estimate", "👨 Man", f"{male_conf:.1f}% score")
                else:
                    st.metric("Gender estimate", "👩 Woman", f"{female_conf:.1f}% score")

    with col2:
        # Emotion Analysis
        if 'emotion' in analysis:
            st.markdown("### 😊 **Emotion estimate**")
            emotions = analysis['emotion']
            if isinstance(emotions, dict):
                # Find dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                emotion_name, emotion_conf = dominant_emotion

                emotion_emojis = {
                    'happy': '😊', 'sad': '😢', 'angry': '😠',
                    'surprise': '😲', 'fear': '😰', 'disgust': '🤢',
                    'neutral': '😐'
                }
                emoji = emotion_emojis.get(emotion_name.lower(), '😐')
                st.metric("Most likely emotion", f"{emoji} {emotion_name.title()}", f"{emotion_conf:.1f}% confidence")

        st.markdown("---")

        # Race/Ethnicity Analysis
        if 'race' in analysis:
            st.markdown("### 🌍 **Ethnicity estimate**")
            race_data = analysis['race']
            if isinstance(race_data, dict):
                # Find dominant ethnicity
                dominant_race = max(race_data.items(), key=lambda x: x[1])
                race_name, race_conf = dominant_race
                st.metric("Most likely estimate", race_name.title(), f"{race_conf:.1f}% confidence")

    # Important disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ Important:** This analysis is based on AI models and is intended for **demonstration and research purposes only**.
    Results are estimates and should not be used for identification, discrimination, or decision-making.
    """)

def show_name_assignment_tab(face_id, image_path, face_location):
    """Display name assignment interface in tab"""

    # Show face preview
    try:
        full_image = load_and_preprocess_image(image_path)
        if full_image is not None and face_location:
            if isinstance(face_location, str):
                coords = face_location.split(',')
                if len(coords) == 4:
                    top, right, bottom, left = map(int, coords)
                    face_crop = full_image[top:bottom, left:right]
                    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                        thumbnail = create_thumbnail(face_crop, (150, 150))
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(thumbnail, caption="Face for name assignment", alt="Selected face")
    except:
        pass

    # Get existing person data if available
    existing_face = st.session_state.vector_store.get_face_by_id(face_id)
    existing_data = {}
    if existing_face and existing_face.get('metadata', {}).get('person_id'):
        metadata = existing_face['metadata']
        existing_data = {
            'first_name': metadata.get('first_name', ''),
            'middle_names': metadata.get('middle_names', ''),
            'last_name': metadata.get('last_name', ''),
            'full_name': metadata.get('full_name', ''),
            'birth_date': metadata.get('birth_date', ''),
            'birth_place': metadata.get('birth_place', ''),
            'notes': metadata.get('notes', ''),
            'person_id': metadata.get('person_id', '')
        }
        st.info(f"✅ **Current name:** {existing_data['full_name']}")

    # Get all existing persons for selection
    all_persons = get_cached_all_persons()  # Use cached version

    # Choice between new person or existing person
    st.subheader("🎯 Select a person")
    st.caption(
        "The name is assigned only to this face. Review similar faces "
        "in **Review groups**."
    )

    assignment_type = st.radio(
        "How would you like to assign the name?",
        ["Create a new person", "Select an existing person"],
        key="tab_assignment_type",
        horizontal=True
    )

    if assignment_type == "Select an existing person" and all_persons:
        # Existing person selection logic (simplified)
        person_options = {"-- Select a person --": None}
        for person in all_persons:
            display_name = f"{person['full_name']} ({person['face_count']} faces)"
            person_options[display_name] = person

        selected_person_display = st.selectbox(
            "Select a person:",
            options=list(person_options.keys()),
            key="tab_selected_person"
        )

        if selected_person_display and person_options[selected_person_display] is not None:
            selected_person = person_options[selected_person_display]

            # Ensure selected_person is not None
            if selected_person is None:
                st.error("Could not load person data.")
                return

            # Show person details
            st.success(f"**Name:** {selected_person['full_name']}")
            st.write(f"**Faces:** {selected_person['face_count']}")

            # Assign button
            if st.button("✅ Assign this person", type="primary", width='stretch', key="tab_assign_existing"):
                person_data = {
                    'first_name': selected_person['first_name'],
                    'last_name': selected_person['last_name'],
                    'person_id': selected_person['person_id']
                }

                result = st.session_state.vector_store.assign_person_name(
                    face_id, person_data, selected_person['person_id']
                )

                if result:
                    st.success(f"Name assigned to {selected_person['full_name']}.")

                    invalidate_persons_cache()
                    invalidate_gallery_cache()
                    st.balloons()
                else:
                    st.error("❌ Could not assign the name.")

    else:
        # New person creation form (simplified)
        st.subheader("📝 Create a new person")

        col1, col2 = st.columns(2)

        with col1:
            first_name = st.text_input("First name *", value=existing_data.get('first_name', ''), key="tab_first_name")
            last_name = st.text_input("Last name *", value=existing_data.get('last_name', ''), key="tab_last_name")

        with col2:
            birth_date = st.text_input("Date of birth", value=existing_data.get('birth_date', ''), key="tab_birth_date")
            birth_place = st.text_input("Place of birth", value=existing_data.get('birth_place', ''), key="tab_birth_place")

        notes = st.text_area("Notes", value=existing_data.get('notes', ''), key="tab_notes", height=60)

        # Preview full name
        if first_name or last_name:
            name_parts = [p.strip() for p in [(first_name or ''), (last_name or '')] if p and p.strip()]
            full_name = ' '.join(name_parts)
            st.success(f"**Full name:** {full_name}")

        # Save button
        if st.button("💾 Save person", type="primary", width='stretch', key="tab_save_person"):
            if (first_name and first_name.strip()) or (last_name and last_name.strip()):
                person_data = {
                    'first_name': (first_name or '').strip(),
                    'last_name': (last_name or '').strip(),
                    'birth_date': (birth_date or '').strip(),
                    'birth_place': (birth_place or '').strip(),
                    'notes': (notes or '').strip()
                }

                full_name = ' '.join([p for p in [person_data['first_name'], person_data['last_name']] if p])

                person_id = str(uuid.uuid4())

                result = st.session_state.vector_store.assign_person_name(
                    face_id, person_data, person_id
                )

                if result:
                    st.success(f"Person '{full_name}' created and assigned.")

                    invalidate_persons_cache()
                    invalidate_gallery_cache()
                    st.balloons()
                else:
                    st.error("❌ Could not save the person.")
            else:
                    st.error("❌ Enter at least a first or last name.")

@st.dialog("🖼️ Full image view", width="large")
def show_full_image_modal():
    """Display the full image modal as a proper popup dialog (legacy - for backward compatibility)"""

    # Make the dialog content even wider using custom CSS
    st.markdown("""
    <style>
    .stDialog > div:first-child > div:first-child > div:first-child {
        width: 95vw !important;
        max-width: 1600px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Required keys for full image display
    required_keys = ['full_image_path', 'full_image_face_location']
    missing_keys = [key for key in required_keys if key not in st.session_state]

    if missing_keys:
        st.error(f"Missing image data for full image view: {', '.join(missing_keys)}")
        st.info("Try opening the full image again.")
        if st.button("Close"):
            # Clear any partial session state
            for key in required_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return

    image_path = st.session_state.full_image_path
    face_location = st.session_state.full_image_face_location

    # Ensure image_path is a Path object
    if isinstance(image_path, str):
        image_path = Path(image_path)

    # Header info
    st.markdown(f"**File:** `{image_path.name}`")
    st.markdown(f"**📍 Pfad:** `{image_path}`")

    try:
        # Load the full image
        full_image = load_and_preprocess_image(image_path)
        if full_image is None:
            st.error("❌ Could not load the image.")
            return

        # Parse face location if provided
        image_with_box = full_image.copy()
        if face_location:
            if isinstance(face_location, str):
                coords = face_location.split(',')
                if len(coords) == 4:
                    top, right, bottom, left = map(int, coords)
                    # Draw bounding box on the image
                    cv2.rectangle(image_with_box, (left, top), (right, bottom), (0, 255, 0), 3)
                    st.success(f"✅ Face marked at: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
            else:
                top, right, bottom, left = face_location
                cv2.rectangle(image_with_box, (left, top), (right, bottom), (0, 255, 0), 3)
                st.success(f"✅ Face marked at: Top={top}, Right={right}, Bottom={bottom}, Left={left}")

        st.markdown("---")

        # Display the image with maximum width
        caption = f"Full image - {image_path.name}"
        if face_location:
            caption += " (face highlighted)"

        st.image(image_with_box, caption=caption, width='stretch', alt=caption)

        # Image info
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Image size", f"{full_image.shape[1]} × {full_image.shape[0]} px")

        with col2:
            file_size = image_path.stat().st_size if image_path.exists() else 0
            st.metric("File size", f"{file_size / 1024:.1f} KB")

        with col3:
            if face_location:
                st.metric("face detected", "Yes")
            else:
                st.metric("face detected", "No")

    except Exception as e:
        st.error(f"❌ Could not display the full image: {str(e)}")

    # Close button
    st.markdown("---")
    if st.button("❌ Close full image", key="close_full_image_modal", type="primary", width='stretch'):
        # Clear full image data
        for key in required_keys:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # Display the detailed analysis
    if all(key in st.session_state for key in ['analysis_image_path', 'analysis_face_location', 'analysis_face_id']):

        # Get analysis data
        with st.spinner("🧠 Analyzing facial attributes..."):
            analysis = analyze_facial_attributes(
                st.session_state.analysis_image_path,
                st.session_state.analysis_face_location
            )

        if analysis is None:
            st.error("❌ Could not analyze facial attributes.")
            return

        if "error" in analysis:
            st.error(f"❌ {analysis['error']}")
            return

        # Header info
        st.markdown(f"**Face ID:** `{st.session_state.analysis_face_id}`")
        st.markdown(f"**File:** `{Path(st.session_state.analysis_image_path).name}`")

        # Show face preview
        try:
            full_image = load_and_preprocess_image(st.session_state.analysis_image_path)
            if full_image is not None:
                face_location = st.session_state.analysis_face_location
                if isinstance(face_location, str):
                    coords = face_location.split(',')
                    if len(coords) == 4:
                        top, right, bottom, left = map(int, coords)
                        face_crop = full_image[top:bottom, left:right]
                        if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.image(face_crop, caption="Analyzed face", width=250, alt="Face used for analysis")
        except:
            pass

        st.markdown("---")

        # Analysis results in a clean 2-column layout
        col1, col2 = st.columns(2)

        with col1:
            # Age Analysis
            if 'age' in analysis:
                st.markdown("## Age estimate")
                age = analysis['age']
                st.metric("Estimated age", f"{age} years", help="Model estimate; accuracy is not guaranteed.")

                # Age category with visual indicator
                if age < 18:
                    st.info("**Category:** Teenager")
                elif age < 30:
                    st.info("**Category:** Young adult")
                elif age < 50:
                    st.info("**Category:** Adult")
                elif age < 70:
                    st.info("**Category:** Older adult")
                else:
                    st.info("**Category:** Senior")

            st.markdown("---")

            # Gender Analysis
            if 'gender' in analysis:
                st.markdown("## ⚧️ **Gender estimate**")
                gender_data = analysis['gender']
                if isinstance(gender_data, dict):
                    male_prob = float(gender_data.get('Man', 0))
                    female_prob = float(gender_data.get('Woman', 0))

                    # Visual representation with progress bars
                    st.write("**👨 Man**")
                    st.progress(male_prob / 100.0)
                    st.write(f"`{male_prob:.1f}%`")

                    st.write("**👩 Woman**")
                    st.progress(female_prob / 100.0)
                    st.write(f"`{female_prob:.1f}%`")

                    # Highlight prediction
                    dominant = "👨 Man" if male_prob > female_prob else "👩 Woman"
                    confidence = max(male_prob, female_prob)
                    st.success(f"**Estimate:** {dominant} (confidence: {confidence:.1f}%)")
                else:
                    st.write(f"**Gender estimate:** {gender_data}")

        with col2:
            # Emotion Analysis
            if 'emotion' in analysis:
                st.markdown("## 😊 **Emotion estimate**")
                emotions = analysis['emotion']
                if isinstance(emotions, dict):
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                    emotion_map = {
                        'happy': ('😊 Happy', '#28a745'),
                        'neutral': ('😐 Neutral', '#6c757d'),
                        'sad': ('😢 Sad', '#007bff'),
                        'angry': ('😠 Angry', '#dc3545'),
                        'surprise': ('😲 Surprised', '#ffc107'),
                        'fear': ('😨 Fearful', '#fd7e14'),
                        'disgust': ('🤢 Disgusted', '#6f42c1')
                    }

                    # Show top 5 emotions with visual bars
                    for emotion, confidence in sorted_emotions[:5]:
                        emotion_label, color = emotion_map.get(emotion, (emotion.title(), '#6c757d'))
                        st.write(f"**{emotion_label}**")
                        st.progress(float(confidence / 100.0))
                        st.write(f"`{confidence:.1f}%`")

                    # Dominant emotion highlight
                    top_emotion, top_confidence = sorted_emotions[0]
                    top_emotion_label, _ = emotion_map.get(top_emotion, (top_emotion.title(), '#6c757d'))
                    st.success(f"**Most likely emotion:** {top_emotion_label} ({top_confidence:.1f}%)")

            st.markdown("---")

            # Race/Ethnicity Analysis
            if 'race' in analysis:
                st.markdown("## 🌍 **Ethnicity estimate**")
                race_data = analysis['race']
                if isinstance(race_data, dict):
                    sorted_races = sorted(race_data.items(), key=lambda x: x[1], reverse=True)
                    race_map = {
                        'asian': '🌏 Asian',
                        'white': '🌍 White',
                        'middle eastern': '🌍 Middle Eastern',
                        'indian': '🌏 Indian',
                        'latino hispanic': '🌎 Latino/Hispanic',
                        'black': '🌍 Black'
                    }

                    # Show top predictions with progress bars
                    for race, confidence in sorted_races:
                        race_label = race_map.get(race.lower(), race.title())
                        st.write(f"**{race_label}**")
                        st.progress(float(confidence / 100.0))
                        st.write(f"`{confidence:.1f}%`")

                    # Dominant prediction
                    top_race, top_confidence = sorted_races[0]
                    top_race_label = race_map.get(top_race.lower(), top_race.title())
                    st.info(f"**Most likely estimate:** {top_race_label} ({top_confidence:.1f}%)")

        # Footer with important disclaimer
        st.markdown("---")
        st.warning("""
        **⚠️ Important:** This analysis is based on AI models and is intended for **demonstration and research purposes only**.
        Results are estimates and should not be used for identification, discrimination, or decision-making.
        """)

        # Close button
        if st.button("✅ Close analysis", key="close_popup_analysis", type="primary", width='stretch'):
            # Clear analysis data
            if 'analysis_image_path' in st.session_state:
                del st.session_state.analysis_image_path
            if 'analysis_face_location' in st.session_state:
                del st.session_state.analysis_face_location
            if 'analysis_face_id' in st.session_state:
                del st.session_state.analysis_face_id
            st.rerun()

    else:
        st.error("❌ Analysis data is unavailable.")
        if st.button("Close", key="close_error_modal"):
            st.rerun()

def display_search_results(results: List[Dict[str, Any]]):
    """Display enhanced search results with filtering options"""

    st.subheader(f"🎯 Enhanced Search Results ({len(results)} faces)")

    if not results:
        st.info("No results to display")
        return

    # Real-time filtering options
    with st.expander("🔧 Filter and sort results", expanded=False):
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            min_similarity_filter = st.slider(
                "Min cosine similarity %",
                min_value=0,
                max_value=100,
                value=0,
                help="Hide results below this similarity"
            )

            min_size_filter = st.slider(
                "Min Face Size (px)",
                min_value=20,
                max_value=200,
                value=20,
                help="Hide small faces"
            )

        with col_f2:
            show_named_only = st.checkbox(
                "Named Faces Only",
                value=False,
                help="Show only faces with assigned names"
            )

        with col_f2:
            sort_order = st.selectbox(
                "Sort By",
                ["Similarity ↓", "Similarity ↑", "Size ↓", "Size ↑", "Name A-Z"],
                index=0,
                help="Change result ordering"
            )

    # Apply filters
    filtered_results = []
    for result in results:
        # Similarity filter
        if result['similarity'] * 100 < min_similarity_filter:
            continue

        # Size filter
        metadata = result.get('metadata', {})
        location_str = metadata.get('location', '0,0,0,0')
        try:
            coords = location_str.split(',')
            if len(coords) == 4:
                top, right, bottom, left = map(int, coords)
                face_size = min(right - left, bottom - top)
                if face_size < min_size_filter:
                    continue
        except:
            pass

        # Named faces filter
        if show_named_only:
            person_name = metadata.get('full_name', '')
            if not person_name:
                continue

        filtered_results.append(result)

    # Apply sorting
    if sort_order == "Similarity ↓":
        filtered_results = sorted(filtered_results, key=lambda x: x['similarity'], reverse=True)
    elif sort_order == "Similarity ↑":
        filtered_results = sorted(filtered_results, key=lambda x: x['similarity'])
    elif sort_order in ("Size ↓", "Size ↑"):
        def get_size(x):
            try:
                coords = x.get('metadata', {}).get('location', '0,0,0,0').split(',')
                if len(coords) == 4:
                    top, right, bottom, left = map(int, coords)
                    return (right - left) * (bottom - top)
                return 0
            except (TypeError, ValueError):
                return 0
        filtered_results = sorted(
            filtered_results, key=get_size, reverse=(sort_order == "Size ↓")
        )
    elif sort_order == "Name A-Z":
        filtered_results = sorted(filtered_results, key=lambda x: x.get('metadata', {}).get('full_name', 'zzz'))

    # Show filter results summary
    if len(filtered_results) != len(results):
        st.info(f"📋 Filtered: {len(filtered_results)} of {len(results)} results shown (filtered out {len(results) - len(filtered_results)})")
    if not filtered_results:
        st.info("No results match these filters. Lower the similarity or face-size filter to see more results.")
        return

    # Display options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{len(filtered_results)} results displayed**")

    with col2:
        show_all = st.toggle("Show All Images", value=False, help="Toggle pagination on/off")

    # Use filtered results for display
    results = filtered_results

    # Determine what to display
    if show_all:
        # Show all results
        page_results = results
        if len(results) > 20:
            st.info(f"Displaying all {len(results)} results")
    else:
        # Pagination
        total_pages = max(1, (len(results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)
        current_page = min(
            max(1, int(st.session_state.get("current_page", 1))), total_pages
        )
        st.session_state.current_page = current_page

        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                if st.button("⬅️ Previous") and current_page > 1:
                    st.session_state.current_page = current_page - 1
                    st.rerun()

            with col2:
                st.write(f"Page {current_page} of {total_pages}")

            with col3:
                if st.button("➡️ Next") and current_page < total_pages:
                    st.session_state.current_page = current_page + 1
                    st.rerun()

        # Calculate page slice
        start_idx = (current_page - 1) * RESULTS_PER_PAGE
        end_idx = start_idx + RESULTS_PER_PAGE
        page_results = results[start_idx:end_idx]

    # Display results in grid
    cols = st.columns(4)

    for i, face_data in enumerate(page_results):
        with cols[i % 4]:
            try:
                metadata = face_data.get('metadata', {})
                image_path = Path(metadata.get('image_path', ''))
                location = metadata.get('location')
                box = parse_face_location(location)

                if not image_path.exists():
                    st.error("Image not found")
                    continue

                # Cached thumbnail: the full-resolution photo is decoded at most once
                thumbnail = face_thumbnail(image_path, location, THUMBNAIL_SIZE)
                if thumbnail is None and box is not None:
                    image = load_and_preprocess_image(image_path)
                    if image is not None:
                        top, right, bottom, left = box
                        thumbnail = create_thumbnail(image[top:bottom, left:right], THUMBNAIL_SIZE)

                if thumbnail is not None:
                    st.image(thumbnail, width='stretch', alt="Face thumbnail")

                face_width = face_height = 0
                if box is not None:
                    top, right, bottom, left = box
                    face_width = right - left
                    face_height = bottom - top
                face_size = min(face_width, face_height)
                face_area = face_width * face_height

                similarity_percentage = face_data['similarity'] * 100

                quality_indicators = []
                if face_size >= 80:
                    quality_indicators.append("🔍 High Res")
                elif face_size >= 50:
                    quality_indicators.append("📏 Medium Res")
                else:
                    quality_indicators.append("📱 Low Res")
                if face_area > 6400:
                    quality_indicators.append("✨ Large")

                st.metric("Cosine similarity", f"{similarity_percentage:.1f}%")
                st.caption(f"Ranking score, not a match probability · face size: {face_size}px")

                if quality_indicators:
                    st.caption(" | ".join(quality_indicators))

                st.write(f"**📁 {image_path.name}**")

                person_name = metadata.get('full_name', '')
                if person_name:
                    st.success(f"👤 **{person_name}**")

                face_id = face_data.get('face_id', face_data.get('id', f"face_{i}"))
                if st.button("ℹ️ Details", key=f"search_result_details_{face_id}", help="Open full image, analyze face, or assign a name.", width='stretch'):
                    st.session_state.info_face_id = face_id
                    st.session_state.info_image_path = image_path
                    st.session_state.info_face_location = location
                    st.rerun()

            except Exception as e:
                st.error(f"Error displaying result: {e}")


def batch_face_processing_page():
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
                    process_with_fast_engine(selected_dir, update_existing)

    with tab2:
        show_processing_statistics()


def process_with_fast_engine(directory_path, update_existing=False):
    """Process images using fast_process.py engine with Streamlit integration."""
    import sys
    import os

    try:
        target_directory = Path(directory_path).expanduser().resolve()
        allowed_roots = [Path(IMAGES_DIR).resolve(), Path(SCRAPED_DIR).resolve()]
        if not target_directory.is_dir() or not any(
            target_directory == root or root in target_directory.parents
            for root in allowed_roots
        ):
            st.error("The selected folder is outside the configured image directories.")
            return

        sys.path.append(os.path.dirname(__file__))
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


## (removed duplicate/empty process_directory_batch function)


def process_directory_batch(directory_path, batch_size=25, max_workers=1,
                           skip_existing=True, detailed_progress=True, save_crops=False):
    """Process a directory of images in batches with detailed progress tracking"""

    image_files = get_image_files(directory_path)

    if not image_files:
        st.warning(f"No images found in {directory_path}")
        return

    st.info(f"🚀 Starting batch processing of {len(image_files)} images...")

    # Progress tracking
    overall_progress = st.progress(0)
    status_container = st.container()

    with status_container:
        current_status = st.empty()
        batch_info = st.empty()
        detailed_container = None

        if detailed_progress:
            st.subheader("📋 Detailed Processing Log")
            detailed_container = st.container()

        total_faces_found = 0
        total_processed = 0
        total_batches = (len(image_files) - 1) // batch_size + 1

        try:
            # Process in batches
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(image_files))
                batch_files = image_files[start_idx:end_idx]

                # Update progress
                progress = batch_num / total_batches
                overall_progress.progress(progress)
                current_status.text(f"Processing batch {batch_num + 1}/{total_batches}...")
                batch_info.info(f"📦 Batch {batch_num + 1}: Processing {len(batch_files)} images")

                # Process batch
                results = st.session_state.face_engine.process_images_batch(
                    batch_files,
                    max_workers=max_workers
                )

                # Prepare data for vector store
                face_ids = []
                embeddings = []
                metadatas = []
                batch_faces = 0
                batch_errors = 0
                batch_no_faces = 0

                # Enhanced per-image logging
                if detailed_progress and detailed_container:
                    with detailed_container:
                        st.markdown(f"**🔍 Batch {batch_num + 1} - Processing {len(batch_files)} images:**")

                for i, result in enumerate(results):
                    image_name = Path(result.get('image_path', 'unknown')).name

                    if "error" not in result and result.get("face_count", 0) > 0:
                        total_processed += 1
                        batch_faces += result["face_count"]
                        face_count = result["face_count"]

                        # Log success with face details
                        if detailed_progress and detailed_container:
                            with detailed_container:
                                if face_count == 1:
                                    st.success(f"✅ {image_name} → {face_count} face detected")
                                else:
                                    st.success(f"✅ {image_name} → {face_count} faces detected")

                        for face_data in result["faces"]:
                            face_ids.append(face_data["face_id"])
                            embeddings.append(face_data["embedding"])

                            # Convert location tuple to string
                            location = face_data["location"]
                            location_str = f"{location[0]},{location[1]},{location[2]},{location[3]}"

                            metadatas.append({
                                "image_path": result["image_path"],
                                "location": location_str,
                                "face_id": face_data["face_id"]
                            })
                    elif "error" in result:
                        batch_errors += 1
                        error_msg = result['error']

                        # Enhanced error logging
                        if detailed_progress and detailed_container:
                            with detailed_container:
                                st.error(f"❌ {image_name} → FEHLER: {error_msg}")

                        # Also log to console for debugging
                        logger.error(f"Image processing error - {image_name}: {error_msg}")

                    else:
                        # Image processed successfully but no faces found
                        total_processed += 1
                        batch_no_faces += 1

                        # Log no faces found
                        if detailed_progress and detailed_container:
                            with detailed_container:
                                st.warning(f"⚠️ {image_name} → No faces detected")

                # Add to vector store
                if face_ids:
                    added_count = st.session_state.vector_store.add_face_embeddings_batch(
                        face_ids, embeddings, metadatas
                    )
                    total_faces_found += added_count

                    if detailed_progress and detailed_container:
                        with detailed_container:
                            st.info(f"💾 Batch {batch_num + 1}: {added_count} faces added to the database")
                            st.markdown("---")  # Separator between batches
                else:
                    if detailed_progress and detailed_container:
                        with detailed_container:
                            st.warning(f"⚪ Batch {batch_num + 1}: No faces found to add to the database")
                            st.markdown("---")  # Separator between batches

                # Enhanced batch summary with detailed statistics
                batch_success_count = len(batch_files) - batch_errors
                if batch_faces > 0:
                    batch_info.success(f"✅ Batch {batch_num + 1}: {batch_faces} faces found ({batch_success_count}/{len(batch_files)} images processed)")
                elif batch_errors > 0:
                    batch_info.error(f"❌ Batch {batch_num + 1}: {batch_errors} errors, {batch_no_faces} images without faces")
                else:
                    batch_info.info(f"⚪ Batch {batch_num + 1}: {batch_no_faces} images without detected faces")

            # Final update
            overall_progress.progress(1.0)
            current_status.text("✅ Processing complete!")

            # Final summary
            st.success(f"🎉 **Processing Complete!**")

            # Enhanced final statistics with error tracking
            total_errors = sum(1 for batch_num in range(total_batches)
                             for result in st.session_state.get('last_batch_results', [])
                             if "error" in result)

            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            with summary_col1:
                st.metric("Images Processed", total_processed)
            with summary_col2:
                st.metric("Total Faces Found", total_faces_found)
            with summary_col3:
                success_rate = (total_processed / len(image_files)) * 100 if len(image_files) > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with summary_col4:
                avg_faces = total_faces_found / total_processed if total_processed > 0 else 0
                st.metric("Avg Faces/Image", f"{avg_faces:.1f}")

            # Enhanced detailed summary in log
            if detailed_progress and detailed_container:
                with detailed_container:
                    st.markdown("---")
                    st.markdown("### 📊 **PROCESSING SUMMARY**")
                    st.markdown(f"""
                    - **Directory:** `{directory_path}`
                    - **Total Images:** {len(image_files)}
                    - **Successfully Processed:** {total_processed}
                    - **Total Faces Found:** {total_faces_found}
                    - **Average Faces per Image:** {avg_faces:.2f}
                    - **Processing Success Rate:** {success_rate:.1f}%
                    """)

        except Exception as e:
            st.error(f"❌ Error during batch processing: {e}")
            current_status.text("❌ Processing failed!")

def quick_process_all_directories(directories, batch_size=25, max_workers=2):
    """Quick process all directories with unified progress tracking and detailed per-image logging"""

    st.info("🚀 **QUICK PROCESS MODE ACTIVATED**")

    # Calculate totals
    total_images = sum(count for _, _, count in directories)
    total_batches = sum((count - 1) // batch_size + 1 for _, _, count in directories)

    st.write(f"**Total Images:** {total_images}")
    st.write(f"**Total Batches:** {total_batches}")
    st.write(f"**Workers:** {max_workers}")

    # Global progress tracking
    overall_progress = st.progress(0)
    current_status = st.empty()
    results_container = st.container()

    # Enhanced logging container
    st.subheader("📋 Detailed Processing Log")
    detailed_log_container = st.container()

    global_batch_count = 0
    total_faces_found = 0
    total_images_processed = 0
    total_errors = 0
    total_no_faces = 0

    try:
        for dir_name, dir_path, image_count in directories:
            current_status.text(f"🔄 Processing {dir_name} directory ({image_count} images)...")

            image_files = get_image_files(dir_path)

            # Process directory in batches
            dir_batches = (len(image_files) - 1) // batch_size + 1

            for batch_num in range(dir_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(image_files))
                batch_files = image_files[start_idx:end_idx]

                # Update global progress
                global_progress = global_batch_count / total_batches
                overall_progress.progress(global_progress)

                current_status.text(f"📦 {dir_name} - Batch {batch_num + 1}/{dir_batches}")

                # Process batch
                results = st.session_state.face_engine.process_images_batch(
                    batch_files,
                    max_workers=max_workers
                )

                # Process results with detailed per-image logging
                face_ids = []
                embeddings = []
                metadatas = []
                batch_faces = 0
                batch_errors = 0
                batch_no_faces = 0

                # Log batch header
                with detailed_log_container:
                    st.markdown(f"**🔍 {dir_name} - Batch {batch_num + 1}/{dir_batches} - Processing {len(batch_files)} images:**")

                for result in results:
                    image_name = Path(result.get('image_path', 'unknown')).name

                    if "error" not in result and result.get("face_count", 0) > 0:
                        total_images_processed += 1
                        face_count = result["face_count"]
                        batch_faces += face_count

                        # Enhanced success logging with face count
                        with detailed_log_container:
                            if face_count == 1:
                                st.success(f"✅ {image_name} → {face_count} face detected")
                            else:
                                st.success(f"✅ {image_name} → {face_count} faces detected")

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

                        # Enhanced error logging
                        with detailed_log_container:
                            st.error(f"❌ {image_name} → FEHLER: {error_msg}")

                        # Log to console for debugging
                        logger.error(f"Quick process error - {image_name}: {error_msg}")

                    else:
                        # No faces found but processing successful
                        total_images_processed += 1
                        batch_no_faces += 1
                        total_no_faces += 1

                        with detailed_log_container:
                            st.warning(f"⚠️ {image_name} → No faces detected")

                # Add to database
                if face_ids:
                    added_count = st.session_state.vector_store.add_face_embeddings_batch(
                        face_ids, embeddings, metadatas
                    )
                    total_faces_found += added_count

                    # Log database addition
                    with detailed_log_container:
                        st.info(f"💾 {dir_name} Batch {batch_num + 1}: {added_count} faces added to the database")
                        st.markdown("---")  # Separator between batches
                else:
                    with detailed_log_container:
                        st.warning(f"⚪ {dir_name} Batch {batch_num + 1}: No faces found to add to the database")
                        st.markdown("---")  # Separator between batches

                global_batch_count += 1

                # Enhanced batch summary in results container
                with results_container:
                    batch_success_count = len(batch_files) - batch_errors
                    if batch_faces > 0:
                        st.success(f"{dir_name}, batch {batch_num + 1}: {batch_faces} faces found ({batch_success_count}/{len(batch_files)} images).")
                    elif batch_errors > 0:
                        st.error(f"{dir_name}, batch {batch_num + 1}: {batch_errors} errors, {batch_no_faces} images without faces.")
                    else:
                        st.info(f"⚪ {dir_name} Batch {batch_num + 1}: {batch_no_faces} images without faces")

        # Final completion
        overall_progress.progress(1.0)
        current_status.text("🎉 All directories processed successfully!")

        # Final summary
        st.balloons()

        st.success("🎉 **QUICK PROCESSING COMPLETE!**")

        # Enhanced final statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Images Processed", total_images_processed)
        with col2:
            st.metric("Faces Found", total_faces_found)
        with col3:
            st.metric("Errors", total_errors)
        with col4:
            st.metric("No Faces Found", total_no_faces)

        # Additional summary statistics
        success_rate = 0
        avg_faces = 0
        if total_images_processed > 0:
            success_rate = ((total_images_processed - total_errors) / total_images_processed) * 100
            avg_faces = total_faces_found / total_images_processed

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col2:
                st.metric("Avg Faces/Image", f"{avg_faces:.1f}")

        # Final detailed summary log
        with detailed_log_container:
            st.markdown("---")
            st.markdown("### 📊 **FINAL SUMMARY**")
            st.markdown(f"""
            - **Total Images Processed:** {total_images_processed}
            - **Total Faces Found:** {total_faces_found}
            - **Total Errors:** {total_errors}
            - **Images without Faces:** {total_no_faces}
            - **Success Rate:** {success_rate:.1f}% ({(total_images_processed - total_errors)} successful)
            - **Average Faces per Image:** {avg_faces:.2f}
            """)

        st.info(f"🗃️ **Database now contains {total_faces_found} new face embeddings ready for searching!**")

    except Exception as e:
        st.error(f"❌ Error during quick processing: {e}")
        current_status.text("❌ Quick processing failed!")

def upload_processing_page():
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
                process_uploaded_images(uploaded_files)

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
                        process_existing_images(image_files)

def process_uploaded_images(uploaded_files):
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
                process_existing_images(saved_paths)
            else:
                st.warning("No valid images were selected. Choose readable JPG, PNG, BMP, or WEBP files.")

        except Exception as e:
            st.error(f"Error processing uploaded images: {e}")

def process_existing_images(image_paths: List[Path]):
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

def web_scraping_page():
    """Web scraping page"""

    st.header("🌐 Web Image Scraping")
    st.warning(
        "Only scrape images from domains and for purposes you are authorized to use. "
        "The crawler is restricted to the hostname entered, follows robots.txt, and blocks private-network targets."
    )

    tab1, tab2 = st.tabs(["Scrape New Sites", "Scraped Images"])

    with tab1:
        st.subheader("Scrape Images from Websites")

        # URL input
        urls_input = st.text_area(
            "Enter website URLs (one per line):",
            height=100,
            placeholder="https://example.com/gallery\nhttps://another-site.com/photos",
            help="Enter URLs of websites to scrape images from"
        )

        col1, col2 = st.columns(2)

        with col1:
            max_images = st.slider(
                "Maximum images per hostname",
                min_value=1,
                max_value=500,
                value=100,
                help="Web import has a finite limit to avoid unexpectedly large downloads.",
            )

        with col2:
            auto_process = st.checkbox(
                "Auto-process scraped images",
                value=False,
                help="Automatically process scraped images for face detection"
            )

        confirm_authorization = st.checkbox(
            "I confirm that I am authorized to collect images from the entered website(s).",
            value=False,
            key="scrape_authorization_confirmed",
        )

        # Show duplicate detection status.
        st.info("🔍 **Duplicate detection is on:** Existing images are skipped automatically")

        if st.button("🕷️ Start Scraping", type="primary"):
            urls, invalid_urls = parse_scrape_urls(urls_input)
            if invalid_urls:
                listed = ", ".join(f"`{value}`" for value in invalid_urls)
                st.error(
                    f"Invalid URL(s): {listed}. Enter complete URLs such as "
                    "https://example.com."
                )
            elif not urls:
                st.error("Please enter at least one URL")
            elif not confirm_authorization:
                st.error("Confirm that you are authorized to collect images before starting web import.")
            elif st.session_state.image_scraper is None:
                st.error(
                    "Web import is unavailable because old_image_scraper could not "
                    "be loaded. Check the missing dependency details above."
                )
            else:
                scrape_websites(urls, max_images, auto_process)

    with tab2:
        display_scraped_images()

def scrape_websites(urls: List[str], max_images: int, auto_process: bool):
    """Scrape images from websites"""

    with st.spinner("Scraping images from websites..."):
        try:
            # Run async scraping; asyncio.run() shuts the loop down cleanly
            # instead of closing it while Playwright tasks are still pending.
            results = run_coroutine(
                st.session_state.image_scraper.scrape_multiple_websites(urls, max_images)
            )

            # Display results
            success_count = sum(1 for r in results if r.get("success", False))
            total_downloaded = sum(r.get("images_downloaded", 0) for r in results)

            if success_count > 0:
                st.success(f"Successfully scraped {success_count}/{len(urls)} websites. Downloaded {total_downloaded} images.")

                # Show detailed results
                for result in results:
                    if result.get("success"):
                        st.info(f"✅ {result['url']}: Downloaded {result['images_downloaded']} images")
                    else:
                        st.error(f"❌ {result['url']}: {result.get('error', 'Unknown error')}")

                # Auto-process if requested
                if auto_process and total_downloaded > 0:
                    st.info("Auto-processing scraped images...")
                    scraped_images = st.session_state.image_scraper.get_scraped_images()
                    if scraped_images:
                        process_existing_images(scraped_images)

            else:
                st.error("Failed to scrape any websites. Please check the URLs and try again.")

        except Exception as e:
            st.error(f"Error during scraping: {e}")

def display_scraped_images():
    """Display scraped images"""

    st.subheader("Scraped Images")

    scraped_images = st.session_state.image_scraper.get_scraped_images()

    if scraped_images:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Found {len(scraped_images)} scraped images**")

        with col2:
            show_all_scraped = st.toggle("Show All Scraped", value=True, help="Show all scraped images or limit to first 20")

        # Determine how many images to show
        images_to_show = scraped_images if show_all_scraped else scraped_images[:20]

        if not show_all_scraped and len(scraped_images) > 20:
            st.info(f"Showing first 20 of {len(scraped_images)} images. Toggle 'Show All Scraped' to see all.")

        # Display in grid
        cols_per_row = 5
        for i in range(0, len(images_to_show), cols_per_row):
            cols = st.columns(cols_per_row)

            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(images_to_show):
                    with col:
                        try:
                            image_path = images_to_show[idx]
                            thumbnail = face_thumbnail(image_path, None, THUMBNAIL_SIZE)
                            if thumbnail is not None:
                                st.image(thumbnail, width='stretch', alt=f"Scraped image: {image_path.name}")
                                st.caption(f"{image_path.name}")
                        except Exception:
                            st.error("Failed to load")

        if len(scraped_images) > 20:
            st.info(f"Showing first 20 of {len(scraped_images)} images")

        # Process button
        if st.button("🔄 Process All Scraped Images"):
            process_existing_images(scraped_images)
    else:
        st.info("No scraped images found. Use the scraping tab to download images from websites.")

def _render_duplicate_set(directory: Path, hash_value: str, file_paths: List[Path], index: int) -> None:
    """Show one duplicate set and let the user keep exactly one file."""
    st.write(f"**Duplicate Set {index + 1}** ({len(file_paths)} files):")

    labels = {str(path): Path(path).name for path in file_paths}
    keep = st.radio(
        "Keep file:",
        options=list(labels.keys()),
        format_func=lambda value: labels[value],
        key=f"keep_{hash_value}",
    )

    columns = st.columns(min(len(file_paths), 4))
    for position, file_path in enumerate(file_paths):
        with columns[position % 4]:
            try:
                thumbnail = face_thumbnail(Path(file_path), None, (120, 120))
                if thumbnail is not None:
                    st.image(thumbnail, width='stretch', alt=f"Duplicate image: {Path(file_path).name}")
                st.caption(Path(file_path).name)
                size_kb = Path(file_path).stat().st_size / 1024
                st.caption(f"{size_kb:.1f} KB")
            except Exception as exc:  # noqa: BLE001 - one broken file must not stop the page
                st.error(f"Error: {exc}")

    pending_key = "pending_duplicate_delete"
    if st.button("Review deletion", key=f"delete_{hash_value}"):
        st.session_state[pending_key] = hash_value
        st.rerun()

    if st.session_state.get(pending_key) == hash_value:
        st.warning(
            f"This will permanently delete {len(file_paths) - 1} files. "
            "The selected file will be kept. Review the filenames above before continuing."
        )
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            confirm = st.button("Confirm deletion", key=f"confirm_delete_{hash_value}", type="primary")
        with cancel_col:
            cancel = st.button("Cancel", key=f"cancel_delete_{hash_value}")
        if cancel:
            st.session_state.pop(pending_key, None)
            st.rerun()
        if confirm:
            st.session_state.pop(pending_key, None)
            # Recheck the candidate set immediately before deleting: a rescan
            # or another session may have changed the files since preview.
            existing_paths = [Path(path) for path in file_paths if Path(path).is_file()]
            if Path(keep) not in existing_paths or len(existing_paths) < 2:
                st.error("The duplicate set changed since review. Scan the folder again.")
                st.rerun()
            file_paths = existing_paths
            result = delete_duplicate_files(file_paths, keep)

            store = st.session_state.get("vector_store")
            removed_faces = 0
            for deleted_path in result["deleted"]:
                if store is not None:
                    removed_faces += store.delete_faces_by_image_path(str(deleted_path))
            if store is not None and result["deleted"]:
                # Reconcile vectors against the files that still exist. This
                # also cleans up records when a direct per-path delete failed.
                removed_faces += store.prune_missing_faces(root_directory=directory)
            duplicate_detector.cleanup_missing_files()
            invalidate_persons_cache()
            invalidate_gallery_cache()

            # Re-scan so the remaining sets and the counters reflect the disk state.
            st.session_state.duplicate_results[str(directory)] = (
                duplicate_detector.find_duplicates_in_directory(directory)
            )

            messages = []
            if result["deleted"]:
                names = ", ".join(path.name for path in result["deleted"])
                messages.append(
                    f"✅ {len(result['deleted'])} file(s) deleted ({names}), "
                    f"{removed_faces} database entries removed."
                )
            if result["missing"]:
                names = ", ".join(path.name for path in result["missing"])
                messages.append(f"Already missing: {names}")
            if result["errors"]:
                messages.append("⚠️ " + "; ".join(result["errors"]))
            if not messages:
                messages.append("Nothing was deleted.")

            st.session_state.duplicate_flash = messages
            st.rerun()


def duplicate_manager_page():
    """Duplicate detection and management page"""

    st.header("Duplicate manager", icon=":material/content_copy:")

    st.session_state.setdefault("duplicate_results", {})
    for message in st.session_state.pop("duplicate_flash", []) or []:
        st.success(message)

    st.markdown("**Duplicate detection:** Finds identical images even when their names differ")

    tab1, tab2, tab3 = st.tabs(["🔍 Find Duplicates", "📊 Database Stats", "🧹 Cleanup"])

    with tab1:
        st.subheader("Find duplicates in folders")

        directories = [IMAGES_DIR, SCRAPED_DIR]

        for directory in directories:
            if directory.exists():
                st.write(f"**{directory.name}**: {len(get_image_files(directory, validate=False))} images")

                mode_key = f"duplicate_mode_{directory.name}"
                st.session_state.setdefault(mode_key, True)
                use_content_hash = st.checkbox(
                    "Compare file contents (SHA-256)",
                    value=st.session_state[mode_key],
                    key=f"content_{directory.name}",
                    help=(
                        "On: compare file contents exactly (also finds renamed copies). "
                        "Off: use a faster comparison based on file name and size."
                    ),
                )
                if st.session_state[mode_key] != use_content_hash:
                    st.session_state[mode_key] = use_content_hash
                    st.session_state.duplicate_results.pop(str(directory), None)

                if st.button(f"🔍 Find Duplicates in {directory.name}", key=f"find_dups_{directory.name}"):
                    with st.spinner(f"Scanning {directory.name} for duplicates..."):
                        st.session_state.duplicate_results[str(directory)] = (
                            duplicate_detector.find_duplicates_in_directory(
                                directory, use_content_hash=use_content_hash
                            )
                        )

                found_sets = st.session_state.duplicate_results.get(str(directory)) or {}
                if found_sets:
                    st.error(f"🔍 Found {len(found_sets)} sets of duplicate images:")
                    for index, (hash_value, file_paths) in enumerate(sorted(found_sets.items())):
                        _render_duplicate_set(directory, hash_value, file_paths, index)
                elif str(directory) in st.session_state.duplicate_results:
                    st.success("No duplicates found! All images are unique.")

    with tab2:
        st.subheader("Hash database statistics")

        stats = duplicate_detector.get_stats()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Tracked Files", stats['total_files'])

        with col2:
            st.metric("Hash Algorithm", stats['hash_algorithm'].upper())

        with col3:
            db_size_kb = stats['database_size'] / 1024
            st.metric("Database Size", f"{db_size_kb:.1f} KB")

        if st.button("🔄 Cleanup Missing Files"):
            removed = duplicate_detector.cleanup_missing_files()
            if removed > 0:
                st.success(f"✅ Removed {removed} entries for missing files")
            else:
                st.info("No missing files found in database")

    with tab3:
        st.subheader("Database Cleanup")

        st.warning("⚠️ **Careful:** These operations cannot be undone!")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear Hash Database", type="secondary"):
                if st.session_state.get("confirm_hash_clear", False):
                    duplicate_detector.hash_database.clear()
                    duplicate_detector.save_hash_database()
                    st.success("Hash database cleared!")
                    st.session_state.confirm_hash_clear = False
                    st.rerun()
                else:
                    st.session_state.confirm_hash_clear = True
                    st.warning("Click again to confirm")

        with col2:
            st.info("The hash database keeps track of all processed images to prevent duplicates. Clear only if you want to start fresh.")

def find_duplicates_in_directory(directory: Path):
    """Find and display duplicates in a directory"""

    with st.spinner(f"Scanning {directory.name} for duplicates..."):
        duplicates = duplicate_detector.find_duplicates_in_directory(directory, use_content_hash=True)

    if duplicates:
        st.error(f"🔍 Found {len(duplicates)} sets of duplicate images:")

        for i, (hash_value, file_paths) in enumerate(duplicates.items()):
            st.write(f"**Duplicate Set {i+1}** ({len(file_paths)} files):")

            cols = st.columns(min(len(file_paths), 4))

            for j, file_path in enumerate(file_paths):
                with cols[j % 4]:
                    try:
                        if file_path.exists():
                            thumbnail = face_thumbnail(file_path, None, (120, 120))
                            if thumbnail is not None:
                                st.image(thumbnail, width='stretch', alt=f"Duplicate image: {file_path.name}")
                                st.caption(f"{file_path.name}")

                                # File size info
                                size_mb = file_path.stat().st_size / (1024 * 1024)
                                st.caption(f"Size: {size_mb:.1f} MB")
                        else:
                            st.error("File not found")
                    except Exception as e:
                        st.error(f"Error: {e}")

            st.markdown("---")
    else:
        st.success("✅ No duplicates found! All images are unique.")

def database_stats_page():
    """Database statistics page"""

    st.header("📊 Database Statistics")
    show_embedding_pipeline_status()

    # Get stats
    stats = st.session_state.vector_store.get_collection_stats()

    if "error" not in stats:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Faces", stats.get("total_faces", 0))

        with col2:
            st.metric("Unique Images", stats.get("unique_images", 0))

        with col3:
            avg_dim = stats.get("avg_embedding_dimension", 0)
            st.metric("Avg Embedding Dim", f"{avg_dim:.0f}")

        with col4:
            if stats.get("total_faces", 0) > 0 and stats.get("unique_images", 0) > 0:
                faces_per_image = stats["total_faces"] / stats["unique_images"]
                st.metric("Faces per Image", f"{faces_per_image:.1f}")

        st.markdown("---")

        # Database info
        st.subheader("Database Information")
        st.write(f"**Collection Name:** {stats['collection_name']}")
        st.write(f"**Database Path:** {stats['db_path']}")

        # Actions
        st.subheader("Database Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Refresh Stats"):
                st.session_state.vector_store.get_collection_stats(force_refresh=True)
                st.rerun()

        with col2:
            if st.button("🧹 Remove missing images", help="Delete database entries whose image file no longer exists."):
                if st.session_state.get("confirm_prune", False):
                    with st.spinner("Removing orphaned database entries..."):
                        removed = st.session_state.vector_store.prune_missing_faces()
                    invalidate_persons_cache()
                    invalidate_gallery_cache()
                    st.success(f"✅ {removed} orphaned face entries removed.")
                    st.session_state.confirm_prune = False
                    st.rerun()
                else:
                    st.session_state.confirm_prune = True
                    st.warning("⚠️ Click again to delete entries without an image file.")

        with col3:
            if st.button("⚠️ Clear Database", type="secondary"):
                if st.session_state.get("confirm_clear", False):
                    st.session_state.vector_store.clear_collection()
                    invalidate_persons_cache()
                    invalidate_gallery_cache()
                    st.success("Database cleared!")
                    st.session_state.confirm_clear = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("Click again to confirm database clearing")

    else:
        st.error(f"Error getting database stats: {stats['error']}")

def settings_page():
    """Settings and configuration page"""

    st.header("Settings", icon=":material/settings:")
    show_embedding_pipeline_status()

    store = st.session_state.get("vector_store")
    stats = store.get_collection_stats() if store is not None else {"error": "Vector database is not available."}
    empty_library = not stats.get("error") and int(stats.get("total_faces", 0)) == 0
    with st.expander("Setup checklist and diagnostics", expanded=empty_library):
        st.caption("Check the app runtime and local index before importing a collection.")
        if import_errors:
            for dependency, (install, exc) in import_errors.items():
                st.error(f"{dependency} is unavailable. Install with `{install}`. Details: {exc}")
        else:
            st.success("Required Python dependencies are available.")

        data_directory = settings_path().parent
        try:
            data_directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".face-write-check-", dir=data_directory):
                pass
            st.success(f"Local storage is writable: {data_directory}")
        except OSError as exc:
            st.error(f"Local storage is not writable: {data_directory} ({exc})")

        if stats.get("error"):
            st.error(f"ChromaDB is unavailable: {stats['error']}")
        else:
            st.success(f"ChromaDB is ready. Indexed faces: {stats.get('total_faces', 0):,}.")

        pipeline_status = store.get_embedding_pipeline_status() if store is not None else {}
        if pipeline_status.get("reindex_required") or pipeline_status.get("reindex_in_progress"):
            st.warning("The saved index needs an update before all faces can be searched. Use Batch processing → Update All Images.")
        else:
            st.success(f"Embedding pipeline: {active_embedding_model()} is compatible with the current index.")

        st.caption("Model availability is checked on demand because first use can download model weights.")
        if st.button("Check embedding model availability", key="check_embedding_model"):
            engine = st.session_state.get("face_engine")
            with st.spinner("Loading the selected model and checking a sample embedding…"):
                try:
                    sample = np.zeros((112, 112, 3), dtype=np.uint8)
                    embedding = engine._extract_single_model_embedding(sample, active_embedding_model()) if engine else None
                    if embedding is not None and len(embedding) == 512:
                        st.session_state.embedding_model_check = (True, "The selected model loaded and returned a 512-dimensional embedding.")
                    else:
                        st.session_state.embedding_model_check = (False, "The model did not return a valid 512-dimensional embedding. Check network access, model weights, and the application logs.")
                except Exception as exc:  # noqa: BLE001 - render an actionable diagnostic
                    st.session_state.embedding_model_check = (False, f"Model check failed: {type(exc).__name__}: {exc}")
        model_check = st.session_state.get("embedding_model_check")
        if model_check:
            (st.success if model_check[0] else st.error)(model_check[1])

    flash = st.session_state.pop("settings_flash", None)
    if flash:
        st.success(flash)

    current_settings = load_settings()

    st.subheader("Editable settings")
    st.caption(f"Saved to: {settings_path()}")

    with st.form("settings_form"):
        embedding_model = st.selectbox(
            "Face search embedding model",
            options=["Buffalo_L", "ArcFace"],
            index=0 if current_settings["embedding_model"] == "buffalo_l" else 1,
            help=(
                "Buffalo_L is more conservative on frontal portraits. ArcFace found "
                "more profile matches in the CFP sample but also more false accepts, "
                "so it uses a stricter grouping threshold. Changing models requires "
                "reindexing your image folders."
            ),
        )
        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(current_settings["similarity_threshold"]),
            step=0.05,
            help="Minimum similarity for search results (0 = everything, 1 = identical)",
        )
        max_results = st.number_input(
            "Maximum results",
            min_value=1,
            max_value=200,
            value=int(current_settings["max_results"]),
            step=1,
            help="Maximum results per search",
        )
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=256,
            value=int(current_settings["batch_size"]),
            step=1,
            help="Images per batch during processing",
        )
        detection_model = st.selectbox(
            "Fallback face detector",
            options=["hog", "cnn"],
            index=1 if current_settings["detection_model"] == "cnn" else 0,
            help="HOG is faster; CNN may detect more faces. This applies only when face_recognition is installed.",
        )
        submitted = st.form_submit_button("💾 Save settings", type="primary")

    if submitted:
        saved, errors = save_settings({
            "embedding_model": embedding_model.lower(),
            "similarity_threshold": similarity_threshold,
            "max_results": max_results,
            "batch_size": batch_size,
            "detection_model": detection_model,
        })
        if saved:
            engine = st.session_state.get("face_engine")
            if engine is not None:
                engine.primary_model = embedding_model
                engine.set_detection_model(detection_model)
            store = st.session_state.get("vector_store")
            if store is not None:
                store.refresh_embedding_pipeline_version()
            st.session_state.search_results = []
            st.session_state.pop("detected_faces", None)
            st.session_state.person_groups_cache = None
            st.session_state.pop("person_groups_threshold", None)
            st.session_state.settings_flash = f"✅ Settings saved: {settings_path()}"
            st.rerun()
        else:
            details = "; ".join(f"{name}: {message}" for name, message in errors.items())
            st.error(f"❌ Invalid values. Nothing was saved – {details}")

    st.divider()

    st.subheader("Face detection settings")

    active_model = current_settings["detection_model"]
    try:
        import face_recognition as _face_recognition  # noqa: F401
        face_recognition_available = True
    except Exception:  # noqa: BLE001 - optional package
        face_recognition_available = False

    st.write(f"**Saved dlib selection:** {active_model.upper()}")
    if face_recognition_available:
        st.caption(
            f"**Active detection:** DeepFace (RetinaFace → MTCNN → OpenCV), "
            f"then face_recognition {active_model.upper()} and OpenCV Haar. "
            "The dlib selection is used only when DeepFace does not find a face."
        )
    else:
        st.info(
            "**Active detection:** DeepFace (RetinaFace → MTCNN → OpenCV), "
            "then OpenCV Haar. face_recognition/dlib is not installed, so the "
            f"saved {active_model.upper()} selection takes effect only after "
            "installing the optional package."
        )

    # Model comparison table
    st.subheader("Optional dlib detector comparison")
    comparison_data = {
        "Model": ["HOG", "CNN"],
        "Speed": ["Very fast", "Slower"],
        "Detection quality": ["Good", "Very good"],
        "Resource use": ["Low", "High"],
        "Recommended for": ["Real-time processing, large collections", "Highest quality"]
    }

    st.table(comparison_data)

    st.subheader("Current settings")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Embedding Model:** DeepFace {active_embedding_model()}")
        if face_recognition_available:
            st.write(f"**dlib-Fallback:** {active_model.upper()}")
        else:
            st.write("**Active detector:** DeepFace with RetinaFace/MTCNN fallback")
        st.write("**Vector Database:** ChromaDB")

    with col2:
        st.write(f"**Similarity Threshold:** {current_settings['similarity_threshold']:.2f}")
        st.write(f"**Max Results:** {current_settings['max_results']}")
        st.write(f"**Batch Size:** {current_settings['batch_size']}")

    st.markdown("---")

    st.subheader("Storage Information")

    # Directory info
    directories = [
        ("Images Directory", IMAGES_DIR),
        ("Scraped Directory", SCRAPED_DIR),
    ]

    for name, path in directories:
        if path.exists():
            file_count = len(get_image_files(path, validate=False))
            st.write(f"**{name}:** {path} ({file_count} images)")
        else:
            st.write(f"**{name}:** {path} (not found)")

    st.markdown("---")

    st.subheader("About")
    st.write("""
    This Face Recognition Search application allows you to:
    - Upload images and search for similar faces
    - Scrape images from websites (with permission)
    - Build a searchable database of face embeddings
    - Find similar faces using advanced AI models

    **Technologies Used:**
    - Streamlit for the web interface
    - DeepFace and face_recognition for AI models
    - ChromaDB for vector similarity search
    - Playwright for web scraping
    """)

def face_gallery_page():
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
                                        if delete_face_from_gallery(face_id):
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
                                            if delete_face_from_gallery(face_info['face_id']):
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

def delete_face_from_gallery(face_id: str) -> bool:
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

def display_face_from_metadata(metadata, show_metadata=True, compact=False):
    """Enhanced face display from metadata information (cached thumbnails)."""
    try:
        image_path = metadata.get('image_path', '')
        face_id = metadata.get('face_id', 'unknown')
        location_str = metadata.get('location', '0,0,0,0')

        if not image_path or not Path(image_path).exists():
            st.error("❌ Image not found")
            st.caption(f"📁 {Path(image_path).name if image_path else 'Unknowner Pfad'}")
            return

        box = parse_face_location(location_str)
        thumbnail_size = (100, 100) if compact else THUMBNAIL_SIZE
        # Cached thumbnail: the original photo is decoded at most once
        thumbnail = face_thumbnail(image_path, location_str, thumbnail_size)

        if thumbnail is None:
            st.error("❌ Could not load image")
            st.caption(f"📁 {Path(image_path).name}")
            return

        st.image(thumbnail, width='stretch', alt=f"Face from {Path(image_path).name}")

        # Single unified info button
        if st.button("ℹ️ Details", key=f"gallery_info_{face_id}", help="Open full image, analyze face, or assign a name.", width='stretch'):
            st.session_state.info_face_id = face_id
            st.session_state.info_image_path = image_path
            st.session_state.info_face_location = location_str
            st.rerun()

        if not show_metadata:
            return

        person_name = metadata.get('full_name', '')
        if person_name:
            st.success(f"👤 **{person_name}**")

        if compact:
            st.caption(f"🆔 {face_id[:8]}...")
            if person_name:
                st.caption(f"👤 {person_name}")
            return

        st.caption(f"🆔 Face ID: {face_id}")
        st.caption(f"📁 {Path(image_path).name}")
        st.caption(f"📍 Position: {location_str}")

        if box is not None:
            top, right, bottom, left = box
            st.caption(f"📐 Size: {right - left}×{bottom - top}px")

        try:
            file_size = Path(image_path).stat().st_size / 1024
            st.caption(f"💾 {file_size:.1f} KB")
        except OSError:
            pass

    except Exception as e:
        st.error(f"❌ Display error: {str(e)[:50]}...")
        st.caption(f"🆔 {metadata.get('face_id', 'unknown')}")

def improved_web_scraping_page():
    """Enhanced web scraping with better progress tracking"""

    st.header("🌐 Enhanced Web Image Scraping")
    st.warning("⚠️ Please ensure you have permission to scrape images from websites.")

    # URL input
    col1, col2 = st.columns([3, 1])

    with col1:
        urls_text = st.text_area(
            "Website URLs (one per line):",
            value="https://www.koenig-karlmann-gymnasium.de",
            height=100,
            help="Enter website URLs to scrape images from"
        )

    with col2:
        st.write("**Scraping Options**")
        max_images = st.number_input(
            "Maximum images per hostname:",
            min_value=1,
            max_value=500,
            value=100,
            help="Web import has a finite limit to avoid unexpectedly large downloads."
        )

        auto_process = st.checkbox(
            "Auto-process faces",
            value=True,
            help="Automatically extract faces after scraping"
        )

        deep_crawl = st.checkbox(
            "Deep crawling",
            value=True,
            help="Crawl all subpages and directories"
        )

    # Advanced options
    with st.expander("🔧 Advanced Options"):
        crawl_depth = st.slider("Maximum crawl depth:", 1, 5, 3)
        timeout = st.slider("Page timeout (seconds):", 10, 60, 30)
        concurrent_downloads = st.slider("Concurrent downloads:", 1, 20, 10)

        quality_priority = st.selectbox(
            "Image quality priority:",
            ["Highest", "Balanced", "Speed"],
            index=0,
            help="Highest: prioritize largest images, Speed: accept smaller images"
        )

    # Scraping button
    if st.button("🚀 Start Enhanced Scraping", type="primary"):
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]

        if urls:
            st.info(f"Starting enhanced scraping of {len(urls)} website(s)...")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()

            try:
                total_downloaded = 0
                all_results = []

                for i, url in enumerate(urls):
                    progress = (i) / len(urls)
                    progress_bar.progress(progress)
                    status_text.text(f"Scraping {url}...")

                    # Enhanced scraping with options
                    result = asyncio.run(
                        st.session_state.image_scraper.scrape_website_enhanced(
                            url,
                            max_images=max_images,
                            crawl_depth=crawl_depth if deep_crawl else 1,
                            quality_priority=quality_priority,
                            timeout=timeout
                        )
                    )

                    all_results.append(result)

                    if result.get("success"):
                        downloaded = result.get("images_downloaded", 0)
                        total_downloaded += downloaded

                        with results_container.container():
                            st.success(f"✅ {url}: {downloaded} images downloaded")

                progress_bar.progress(1.0)
                status_text.text("Scraping complete!")

                # Summary
                st.success(f"🎉 Scraping complete! Downloaded {total_downloaded} total images")

                # Auto-process if requested
                if auto_process and total_downloaded > 0:
                    st.info("🔄 Auto-processing scraped images...")
                    process_scraped_images_enhanced()

            except Exception as e:
                st.error(f"Error during scraping: {e}")
        else:
            st.error("Please enter at least one URL")

    # Display scraped images
    display_scraped_images_enhanced()

def process_scraped_images_enhanced():
    """Enhanced processing of scraped images with better progress tracking"""

    scraped_images = st.session_state.image_scraper.get_scraped_images()

    if not scraped_images:
        st.warning("No scraped images found to process")
        return

    st.info(f"Processing {len(scraped_images)} scraped images...")

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process in batches
    batch_size = 20
    total_faces = 0
    processed_images = 0

    try:
        for i in range(0, len(scraped_images), batch_size):
            batch = scraped_images[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(scraped_images) - 1) // batch_size + 1

            progress = i / len(scraped_images)
            progress_bar.progress(progress)
            status_text.text(f"Processing batch {batch_num}/{total_batches}...")

            # Process batch
            results = st.session_state.face_engine.process_images_batch(batch, max_workers=1)

            # Prepare data for vector store
            face_ids = []
            embeddings = []
            metadatas = []

            for result in results:
                if "error" not in result and result["face_count"] > 0:
                    processed_images += 1
                    for face_data in result["faces"]:
                        face_ids.append(face_data["face_id"])
                        embeddings.append(face_data["embedding"])

                        # Convert location tuple to string
                        location = face_data["location"]
                        location_str = f"{location[0]},{location[1]},{location[2]},{location[3]}"

                        metadatas.append({
                            "image_path": result["image_path"],
                            "location": location_str,
                            "face_id": face_data["face_id"]
                        })
                        total_faces += 1

            # Add to vector store
            if face_ids:
                added_count = st.session_state.vector_store.add_face_embeddings_batch(
                    face_ids, embeddings, metadatas
                )
                status_text.text(f"Batch {batch_num}: Added {added_count} faces to database")

        progress_bar.progress(1.0)
        status_text.text("Processing complete!")

        # Final summary
        st.success(f"🎉 Processing complete!")
        st.info(f"📊 Summary:\n"
               f"- Images processed: {processed_images}/{len(scraped_images)}\n"
               f"- Total faces found: {total_faces}\n"
               f"- Faces added to database: {total_faces}")

    except Exception as e:
        st.error(f"Error processing images: {e}")

def display_scraped_images_enhanced():
    """Enhanced display of scraped images with filtering and search"""

    st.subheader("📸 Scraped Images")

    scraped_images = st.session_state.image_scraper.get_scraped_images()

    if not scraped_images:
        st.info("No scraped images found. Use the scraping tool above to collect images.")
        return

    # Filter and display options
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        show_count = st.selectbox("Images to show:", [20, 50, 100, "All"], index=1)

    with col2:
        sort_by = st.selectbox("Sort by:", ["Name", "Size", "Date"], index=0)

    with col3:
        filter_text = st.text_input("Filter by filename:", "")

    with col4:
        show_full_size = st.toggle("Show Full Size", value=False, help="Display images in original quality (slower loading)")

    # Apply filters
    filtered_images = scraped_images

    if filter_text:
        filtered_images = [img for img in scraped_images if filter_text.lower() in img.name.lower()]

    # Apply show count limit
    if show_count != "All":
        filtered_images = filtered_images[:int(show_count)]

    st.write(f"**Displaying {len(filtered_images)} of {len(scraped_images)} images**")

    # Display images in grid
    cols_per_row = 5
    for i in range(0, len(filtered_images), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(filtered_images):
                with col:
                    try:
                        image_path = filtered_images[idx]
                        image = load_and_preprocess_image(image_path)

                        if image is not None:
                            # Show full size or thumbnail based on user preference
                            if show_full_size:
                                # Display in original quality for better face recognition inspection
                                st.image(image, caption=image_path.name, width='stretch', alt=f"Image: {image_path.name}")
                            else:
                                # Create larger thumbnail for better quality
                                thumbnail = create_thumbnail(image, THUMBNAIL_SIZE)
                                st.image(thumbnail, caption=image_path.name, width='stretch', alt=f"Thumbnail: {image_path.name}")

                            # Enhanced image info
                            st.caption(f"📁 {image_path.name}")
                            if image_path.stat().st_size:
                                size_kb = image_path.stat().st_size / 1024
                                st.caption(f"💾 {size_kb:.1f} KB")

                            # Show original dimensions
                            original_height, original_width = image.shape[:2]
                            st.caption(f"📐 {original_width}x{original_height} px")
                        else:
                            st.error(f"Could not load: {image_path.name}")
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")

    # Processing buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🔄 Process All for Face Detection", help="Process all scraped images to extract faces"):
            process_scraped_images_enhanced()

    with col2:
        if st.button("🗑️ Clear All Scraped Images", help="Delete all scraped images from storage"):
            if st.session_state.get('confirm_clear_scraped', False):
                # Actually clear the images
                for img_path in scraped_images:
                    try:
                        img_path.unlink()
                    except:
                        pass
                st.success("Cleared all scraped images!")
                st.session_state.confirm_clear_scraped = False
                st.rerun()
            else:
                st.session_state.confirm_clear_scraped = True
                st.warning("Click again to confirm deletion of all scraped images!")

    with col3:
        st.write(f"**Total: {len(scraped_images)} images**")

def fast_process_all_images():
    """Ultra fast processing mode using optimized settings"""

    st.info("⚡ **ULTRA FAST MODE ACTIVATED**")
    st.warning("Using optimized settings for maximum speed.")

    # Get all image directories
    directories = []
    if IMAGES_DIR.exists():
        image_count = len(get_image_files(IMAGES_DIR, validate=False))
        if image_count > 0:
            directories.append(("Images", IMAGES_DIR, image_count))

    if SCRAPED_DIR.exists():
        scraped_count = len(get_image_files(SCRAPED_DIR, validate=False))
        if scraped_count > 0:
            directories.append(("Scraped", SCRAPED_DIR, scraped_count))

    if not directories:
        st.error("No images found to process.")
        return

    total_images = sum(count for _, _, count in directories)

    # Global progress tracking
    overall_progress = st.progress(0)
    current_status = st.empty()

    # Enhanced logging container
    st.subheader("⚡ Fast Processing Log")
    detailed_log_container = st.container()

    global_batch_count = 0
    total_faces_found = 0
    total_images_processed = 0
    total_errors = 0

    # Fast processing settings
    fast_batch_size = 10  # Smaller batches for faster feedback
    max_workers = 1       # Single worker to avoid issues

    start_time = time.time()

    try:
        for dir_name, dir_path, image_count in directories:
            current_status.text(f"⚡ Fast processing {dir_name} directory ({image_count} images)...")

            image_files = get_image_files(dir_path)
            dir_batches = (len(image_files) - 1) // fast_batch_size + 1

            for batch_num in range(dir_batches):
                start_idx = batch_num * fast_batch_size
                end_idx = min(start_idx + fast_batch_size, len(image_files))
                batch_files = image_files[start_idx:end_idx]

                # Update global progress
                total_batches = sum((count - 1) // fast_batch_size + 1 for _, _, count in directories)
                global_progress = global_batch_count / total_batches
                overall_progress.progress(global_progress)

                current_status.text(f"⚡ {dir_name} - Fast Batch {batch_num + 1}/{dir_batches}")

                # Process batch with shorter timeout
                batch_start_time = time.time()
                results = st.session_state.face_engine.process_images_batch(
                    batch_files,
                    max_workers=max_workers
                )
                batch_time = time.time() - batch_start_time

                # Process results
                face_ids = []
                embeddings = []
                metadatas = []
                batch_faces = 0
                batch_errors = 0

                # Enhanced per-image logging
                with detailed_log_container:
                    st.markdown(f"**⚡ {dir_name} - Fast Batch {batch_num + 1} ({batch_time:.1f}s):**")

                for result in results:
                    image_name = Path(result.get('image_path', 'unknown')).name

                    if "error" not in result and result.get("face_count", 0) > 0:
                        total_images_processed += 1
                        face_count = result["face_count"]
                        batch_faces += face_count

                        with detailed_log_container:
                            st.success(f"⚡ {image_name} → {face_count} face(s) detected")

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

                        with detailed_log_container:
                            st.error(f"❌ {image_name} → FEHLER: {error_msg}")

                    else:
                        total_images_processed += 1
                        with detailed_log_container:
                            st.warning(f"⚠️ {image_name} → No faces detected")

                # Add to database
                if face_ids:
                    added_count = st.session_state.vector_store.add_face_embeddings_batch(
                        face_ids, embeddings, metadatas
                    )
                    total_faces_found += added_count

                    with detailed_log_container:
                        st.info(f"💾 Fast Batch: {added_count} faces added to the database")
                        st.markdown("---")

                global_batch_count += 1

        # Final completion
        overall_progress.progress(1.0)
        total_time = time.time() - start_time
        current_status.text("⚡ Fast processing complete!")

        # Enhanced final summary
        st.balloons()
        st.success("⚡ **FAST PROCESSING COMPLETE!**")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Images Processed", total_images_processed)
        with col2:
            st.metric("Faces Found", total_faces_found)
        with col3:
            st.metric("Errors", total_errors)
        with col4:
            avg_time = total_time / total_images_processed if total_images_processed > 0 else 0
            st.metric("Avg Time/Image", f"{avg_time:.1f}s")

        # Speed comparison
        if total_images_processed > 0:
            images_per_minute = (total_images_processed / total_time) * 60
            st.success(f"🚀 **Processing speed: {images_per_minute:.1f} images/minute**")

    except Exception as e:
        st.error(f"❌ Error during fast processing: {e}")
        current_status.text("❌ Fast processing failed!")

def _assign_face_group(group: Dict[str, Any], person_data: Dict[str, str], person_id: str) -> int:
    """Assign one reviewed group to a person and return the number of faces."""
    assigned = 0
    for face in group.get("faces", []):
        if st.session_state.vector_store.assign_person_name(
            face["face_id"], person_data, person_id
        ):
            assigned += 1
    return assigned


def person_groups_page():
    """Review identity groups of unnamed faces and assign them to a person."""
    st.header("Review face groups", icon=":material/groups:")
    st.markdown(
        "Unlabeled faces are grouped based on their embeddings. "
        "Suggestions are assigned to a person only after you confirm them."
    )

    notice = st.session_state.pop("person_groups_notice", None)
    if notice:
        st.success(notice)

    threshold = st.slider(
        "Similarity threshold",
        min_value=0.30,
        max_value=0.90,
        value=active_grouping_threshold(),
        step=0.01,
        help=(
            "Cosine similarity. Starting points are 0.35 for Buffalo_L and 0.55 for ArcFace. "
            "Higher values create smaller, stricter groups. Review all suggestions before assigning names."
        ),
        key="person_groups_threshold",
    )

    if st.button("Recalculate groups", key="refresh_person_groups", icon=":material/refresh:"):
        st.session_state.person_groups_cache = None

    try:
        face_count = st.session_state.vector_store.collection.count()
    except Exception:  # noqa: BLE001 - the page still shows a readable error
        face_count = 0
    cache_key = f"{active_embedding_model()}:{threshold:.2f}:{face_count}"
    if (
        st.session_state.get("person_groups_cache") is None
        or st.session_state.get("person_groups_cache_key") != cache_key
    ):
        with st.spinner("Grouping faces..."):
            st.session_state.person_groups_cache = (
                st.session_state.vector_store.get_face_groups(threshold=threshold)
            )
            st.session_state.person_groups_cache_key = cache_key

    data = st.session_state.person_groups_cache
    if data.get("error"):
        st.error(f"Could not group faces: {data['error']}")
        return

    groups = data.get("groups", [])
    unnamed_faces = data.get("unnamed_faces", 0)
    grouped_faces = data.get("grouped_faces", 0)

    if unnamed_faces == 0:
        st.info("All faces are already assigned to a person.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Unassigned faces", unnamed_faces)
    col2.metric("Groups", len(groups))
    col3.metric("Faces in groups", grouped_faces)

    if not groups:
        st.info(
            "No group with at least two faces was found. Lower the threshold or assign names from the gallery."
        )
        return

    st.caption(
        f"{data.get('singletons', 0)} unassigned faces do not match a group and remain separate."
    )
    st.markdown("---")

    persons = get_cached_all_persons()
    person_choices = {"Create a new person": None}
    for person in persons:
        label = f"{person['full_name']} ({person['face_count']} faces)"
        person_choices[label] = person

    for index, group in enumerate(groups):
        group_key = group["face_ids"][0]
        with st.expander(
            f"Group {index + 1}: {group['size']} faces · "
            f"mean similarity {group['mean_similarity']:.0%}",
            expanded=index < 3,
        ):
            preview_columns = st.columns(min(6, group["size"]))
            for face_index, face in enumerate(group["faces"][:6]):
                metadata = face.get("metadata", {})
                with preview_columns[face_index]:
                    thumbnail = face_thumbnail(
                        metadata.get("image_path", ""),
                        metadata.get("location", ""),
                        (120, 120),
                    )
                    if thumbnail is not None:
                        st.image(
                            thumbnail,
                            caption=Path(metadata.get("image_path", "")).name,
                            width="stretch",
                            alt=f"Face from {Path(metadata.get('image_path', '')).name}",
                        )
                    else:
                        st.warning("Image unavailable")

            suggestion = group.get("suggested_person")
            if suggestion:
                st.info(
                    f"Closest existing person: **{suggestion['full_name']}** "
                    f"({suggestion['similarity']:.0%}). Not assigned yet."
                )

            default_choice = 0
            if suggestion:
                for option_index, (_, person) in enumerate(person_choices.items()):
                    if person and person["person_id"] == suggestion["person_id"]:
                        default_choice = option_index
                        break
            choice = st.selectbox(
                "Assign group to",
                list(person_choices.keys()),
                index=default_choice,
                key=f"group_target_{group_key}",
            )
            first_name = ""
            last_name = ""
            if person_choices[choice] is None:
                name_col1, name_col2 = st.columns(2)
                first_name = name_col1.text_input(
                    "First name", key=f"group_first_{group_key}"
                )
                last_name = name_col2.text_input(
                    "Last name", key=f"group_last_{group_key}"
                )

            if st.button(
                "Assign group",
                key=f"assign_group_{group_key}",
                type="primary",
            ):
                selected_person = person_choices[choice]
                if selected_person is None:
                    if not first_name.strip() and not last_name.strip():
                        st.warning("Enter at least a first or last name.")
                        continue
                    person_data = {
                        "first_name": first_name.strip(),
                        "middle_names": "",
                        "last_name": last_name.strip(),
                        "birth_date": "",
                        "birth_place": "",
                        "notes": "Assigned from review groups",
                    }
                    person_id = str(uuid.uuid4())
                else:
                    person_data = {
                        "first_name": selected_person.get("first_name", ""),
                        "middle_names": selected_person.get("middle_names", ""),
                        "last_name": selected_person.get("last_name", ""),
                        "birth_date": selected_person.get("birth_date", ""),
                        "birth_place": selected_person.get("birth_place", ""),
                        "notes": selected_person.get("notes", ""),
                    }
                    person_id = selected_person["person_id"]

                assigned = _assign_face_group(group, person_data, person_id)
                if assigned:
                    invalidate_persons_cache()
                    invalidate_gallery_cache()
                    st.session_state.person_groups_cache = None
                    st.session_state.person_groups_notice = (
                        f"Assigned {assigned} faces to "
                        f"{person_data['first_name']} {person_data['last_name']}."
                    )
                    st.rerun()
                else:
                    st.error("❌ Could not assign any faces.")


def name_gallery_page():
    """Name Gallery page to view and manage person names"""

    st.header("People", icon=":material/person:")
    st.markdown("Manage all named people in the database")

    try:
        # Get all persons from database with caching
        persons = get_cached_all_persons()

        if not persons:
            st.info("No names assigned yet. Open Face search or Face library to assign a name.")
            return

        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👤 People", len(persons))
        with col2:
            total_assigned_faces = sum(p['face_count'] for p in persons)
            st.metric("Named faces", total_assigned_faces)
        with col3:
            avg_faces = total_assigned_faces / len(persons) if persons else 0
            st.metric("Average faces per person", f"{avg_faces:.1f}")

        st.markdown("---")

        # Search functionality
        col1, col2 = st.columns([2, 1])
        with col1:
            name_search = st.text_input("Search by name", placeholder="Enter a first or last name")
        with col2:
            sort_option = st.selectbox(
                "Sort by", ["Name (A–Z)", "Name (Z–A)", "Face count (ascending)", "Face count (descending)"]
            )

        # Filter persons based on search
        filtered_persons = persons
        if name_search.strip():
            search_term = name_search.lower()
            filtered_persons = [p for p in persons
                              if search_term in p['full_name'].lower()]

        # Sort persons
        if sort_option == "Name (A–Z)":
            filtered_persons.sort(key=lambda x: x['full_name'])
        elif sort_option == "Name (Z–A)":
            filtered_persons.sort(key=lambda x: x['full_name'], reverse=True)
        elif sort_option == "Face count (ascending)":
            filtered_persons.sort(key=lambda x: x['face_count'])
        elif sort_option == "Face count (descending)":
            filtered_persons.sort(key=lambda x: x['face_count'], reverse=True)

        if not filtered_persons:
            st.warning(f"No people found for '{name_search}'")
            return

        st.markdown(f"**{len(filtered_persons)} people found:**")

        # Display persons
        for person in filtered_persons:
            with st.expander(f"👤 **{person['full_name']}** ({person['face_count']} faces)", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.write(f"**First name:** {person['first_name']}")
                    st.write(f"**Last name:** {person['last_name']}")
                    if person.get('birth_date'):
                        st.write(f"**Date of birth:** {person['birth_date']}")
                    st.write(f"**Person ID:** `{person['person_id']}`")

                with col2:
                    st.write(f"**Faces:** {person['face_count']}")
                    if st.button("Show all faces", key=f"show_faces_{person['person_id']}", icon=":material/visibility:"):
                        st.session_state.selected_person = person['person_id']
                        st.rerun()

                with col3:
                    # Simplified one-click deletion
                    if st.button("Remove name", key=f"remove_person_{person['person_id']}",
                                help="Remove this name from all faces assigned to this person.",
                                type="secondary"):
                        # Direct deletion without confirmation - as requested
                        with st.spinner("Removing the name from all faces..."):
                            removed_count = 0
                            for face_id in person['face_ids']:
                                if st.session_state.vector_store.remove_person_name(face_id):
                                    removed_count += 1

                            if removed_count > 0:
                                st.success(f"Removed the name from {removed_count} faces.")
                                # Invalidate cached names and gallery metadata
                                invalidate_persons_cache()
                                invalidate_gallery_cache()
                                st.rerun()
                            else:
                                st.error("❌ Could not remove the name.")

        # Display faces for selected person
        if st.session_state.get('selected_person'):
            person_id = st.session_state.selected_person
            person_faces = st.session_state.vector_store.get_faces_by_person_id(person_id)

            if person_faces:
                st.markdown("---")
                selected_person_name = next((p['full_name'] for p in persons if p['person_id'] == person_id), "Unknown")
                st.subheader(f"Faces assigned to **{selected_person_name}**")

                # Display faces in grid with enhanced functionality
                cols = st.columns(3)  # Use 3 columns for better layout with more buttons
                for idx, face in enumerate(person_faces):
                    with cols[idx % 3]:
                        metadata = face['metadata']
                        image_path = metadata.get('image_path', '')
                        face_location = metadata.get('location', '')
                        face_id = face['face_id']

                        try:
                            # Load and display face thumbnail
                            thumbnail = face_thumbnail(image_path, face_location, (150, 150))
                            if thumbnail is not None:
                                st.image(thumbnail, caption=f"Face {face_id[:8]}...", width="stretch", alt="Face assigned to this person")
                            else:
                                st.warning("Could not load face")

                        except Exception as e:
                            st.error(f"Error while loading: {e}")

                        # Face info
                        st.caption(f"**ID:** {face_id[:12]}...")
                        st.caption(f"**Image:** {Path(image_path).name}")

                        # Buttons in columns for better layout
                        btn_col1, btn_col2 = st.columns(2)

                        with btn_col1:
                            # Single unified info button
                            if st.button("Details", key=f"name_gallery_info_{face_id}", help="Open the full image, analyze the face, or change its assigned name."):
                                # Preserve selected_person state for Name Gallery
                                st.session_state.name_gallery_selected_person = st.session_state.selected_person
                                # Set face info data and trigger modal
                                st.session_state.info_face_id = face_id
                                st.session_state.info_image_path = image_path
                                st.session_state.info_face_location = face_location
                                st.rerun()

                        with btn_col2:
                            # Remove photo button
                            if st.button("Remove", key=f"remove_photo_{face_id}", help="Remove this photo from this person.", type="secondary"):
                                # Remove person name from this specific face
                                if st.session_state.vector_store.remove_person_name(face_id):
                                    st.success(f"Removed the photo from {selected_person_name}.")
                                    invalidate_persons_cache()
                                    invalidate_gallery_cache()
                                    # Refresh the view
                                    st.rerun()
                                else:
                                    st.error("❌ Could not remove the photo.")

                # Close view button
                st.markdown("---")
                if st.button("Close view", type="primary"):
                    del st.session_state.selected_person
                    st.rerun()

    except Exception as e:
        st.error(f"Could not load the people gallery: {e}")

if __name__ == "__main__":
    main()
