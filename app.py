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
import hashlib
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Dict, Any
import logging
import tempfile
from datetime import datetime
from access_control import (
    access_decision,
    allowed_email_addresses,
    shared_auth_required,
)

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

from portable_export import create_face_export
from image_provenance import read_image_provenance
from utils import (
    IMAGE_EXTENSIONS
)
from async_utils import run_coroutine
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT,
    DATA_DIR, IMAGES_DIR, SCRAPED_DIR
)

from app_pages.face_library import (
    render_face_library_page,
)
from app_pages.processing import (
    process_existing_images as _process_existing_images,
    render_batch_processing_page,
    render_image_import_page,
)
from app_pages.web_import import render_web_import_page
from app_pages.duplicates import render_duplicate_manager_page
from app_pages.people import render_people_page, render_person_groups_page
from app_pages.face_search import load_query_image, render_face_search_page
from app_pages.face_details import (
    render_face_info_modal,
    render_name_assignment_modal,
    show_analysis_modal,
    show_full_image_modal,
)
from app_pages.database_statistics import render_database_stats_page
from app_pages.settings import render_settings_page

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


def create_pipeline_change_backup(store: Any) -> Dict[str, Any]:
    """Persist a searchable, embedding-inclusive backup of a complete index."""
    if store is None:
        return {"success": False, "error": "The vector database is unavailable."}
    status = store.get_embedding_pipeline_status()
    if not status.get("compatible") or not status.get("complete"):
        return {
            "success": False,
            "error": "Only a complete index compatible with the active pipeline can be backed up for migration.",
        }
    face_count = int(store.collection.count())
    if face_count <= 0:
        return {"success": False, "error": "There are no indexed faces to back up."}
    try:
        archive_data = create_face_export(
            store.collection,
            store._embedding_pipeline_version,
            include_embeddings=True,
            include_images=False,
        )
        backup_directory = DATA_DIR / "pipeline_backups"
        backup_directory.mkdir(parents=True, exist_ok=True)
        version_hash = hashlib.sha256(
            store._embedding_pipeline_version.encode("utf-8")
        ).hexdigest()[:12]
        backup_path = backup_directory / (
            f"face-index-{version_hash}-{datetime.now().strftime('%Y%m%dT%H%M%S%f')}.zip"
        )
        with tempfile.NamedTemporaryFile(
            prefix=".face-index-backup-", suffix=".tmp", dir=backup_directory,
            delete=False,
        ) as temporary:
            temporary.write(archive_data)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, backup_path)
        return {
            "success": True,
            "backup_path": str(backup_path),
            "pipeline_version": store._embedding_pipeline_version,
            "face_count": face_count,
        }
    except Exception as exc:  # noqa: BLE001 - report backup failures before index mutation
        logger.exception("Could not create embedding pipeline backup")
        return {"success": False, "error": f"Could not create the index backup: {exc}"}
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


def display_image_provenance(image_path: str | Path) -> None:
    """Show recorded web source and the user's authorization confirmation."""
    provenance = read_image_provenance(image_path)
    source_url = str(provenance.get("source_url", "") or "").strip()
    if provenance.get("source_type") != "web_import" or not source_url:
        return
    key = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:16]
    st.link_button("Open original source", source_url, key=f"source_link_{key}")
    collected_at = str(provenance.get("collected_at", "") or "Unknown time")
    confirmation = (
        "User confirmed authorization" if provenance.get("authorization_confirmed")
        else "Authorization was not confirmed"
    )
    st.caption(f"Collected {collected_at} · {confirmation}")


def _authenticated_actor() -> str:
    """Return the verified OIDC email, or an honest local-only actor label."""
    try:
        email = str(getattr(st.user, "email", "") or "").strip()
        return email or "local-user"
    except Exception:  # noqa: BLE001 - local mode has no configured identity
        return "local-user"


def _enforce_shared_access_control() -> None:
    """Gate all app pages and data actions when shared auth is enabled."""
    required = shared_auth_required()
    if not required:
        return
    allowlist = allowed_email_addresses()
    try:
        user = st.user
        logged_in = bool(user.is_logged_in)
        email = str(getattr(user, "email", "") or "").strip()
        email_verified = getattr(user, "email_verified", False) is True
    except Exception:  # noqa: BLE001 - missing OIDC config fails closed
        logged_in, email, email_verified = False, "", False
    decision = access_decision(
        required=required,
        logged_in=logged_in,
        email=email,
        email_verified=email_verified,
        allowlist=allowlist,
    )
    if decision == "setup_required":
        st.error(
            "Shared access is locked: configure FACE_ALLOWED_EMAILS with at least "
            "one exact email address before enabling FACE_AUTH_REQUIRED."
        )
        st.stop()
    if decision == "login_required":
        st.title("Sign in required")
        st.caption("This face library is restricted to approved accounts.")
        st.button("Sign in", on_click=st.login, type="primary")
        st.stop()
    if decision == "not_authorized":
        st.error(
            "This account is not authorized to access the face library. "
            "Use an approved account with a verified email address."
        )
        st.button("Sign out", on_click=st.logout)
        st.stop()

def main():
    """Main application function"""

    _enforce_shared_access_control()

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
        if shared_auth_required():
            st.caption(f"Signed in as {_authenticated_actor()}")
            st.button("Sign out", on_click=st.logout, key="shared_auth_logout")
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
    render_face_search_page(show_embedding_pipeline_status)




def show_name_assignment_modal():
    render_name_assignment_modal(
        get_cached_all_persons=get_cached_all_persons,
        invalidate_persons_cache=invalidate_persons_cache,
        invalidate_gallery_cache=invalidate_gallery_cache,
        authenticated_actor=_authenticated_actor,
    )


def show_face_info_modal():
    render_face_info_modal(
        display_image_provenance=display_image_provenance,
        show_assignment_history=show_assignment_history,
        get_cached_all_persons=get_cached_all_persons,
        invalidate_persons_cache=invalidate_persons_cache,
        invalidate_gallery_cache=invalidate_gallery_cache,
        authenticated_actor=_authenticated_actor,
    )




def batch_face_processing_page():
    render_batch_processing_page(
        discover_image_directories=discover_image_directories,
        show_embedding_pipeline_status=show_embedding_pipeline_status,
        invalidate_gallery_cache=invalidate_gallery_cache,
        invalidate_persons_cache=invalidate_persons_cache,
    )


def upload_processing_page():
    render_image_import_page(
        load_query_image=load_query_image,
        invalidate_persons_cache=invalidate_persons_cache,
        invalidate_gallery_cache=invalidate_gallery_cache,
    )


def process_existing_images(image_paths):
    return _process_existing_images(
        image_paths,
        invalidate_persons_cache=invalidate_persons_cache,
        invalidate_gallery_cache=invalidate_gallery_cache,
    )


def web_scraping_page():
    render_web_import_page(
        parse_scrape_urls=parse_scrape_urls,
        run_coroutine=run_coroutine,
        display_image_provenance=display_image_provenance,
        process_existing_images=process_existing_images,
    )


def duplicate_manager_page():
    render_duplicate_manager_page(
        invalidate_persons_cache=invalidate_persons_cache,
        invalidate_gallery_cache=invalidate_gallery_cache,
    )



def database_stats_page():
    render_database_stats_page(
        show_embedding_pipeline_status=show_embedding_pipeline_status,
        invalidate_persons_cache=invalidate_persons_cache,
        invalidate_gallery_cache=invalidate_gallery_cache,
    )


def settings_page():
    render_settings_page(
        import_errors=import_errors,
        show_embedding_pipeline_status=show_embedding_pipeline_status,
        create_pipeline_change_backup=create_pipeline_change_backup,
        invalidate_persons_cache=invalidate_persons_cache,
        invalidate_gallery_cache=invalidate_gallery_cache,
    )


def face_gallery_page():
    render_face_library_page(
        faces_to_csv=faces_to_csv,
        invalidate_persons_cache=invalidate_persons_cache,
        invalidate_gallery_cache=invalidate_gallery_cache,
    )


def person_groups_page():
    render_person_groups_page(
        get_cached_all_persons=get_cached_all_persons,
        invalidate_persons_cache=invalidate_persons_cache,
        invalidate_gallery_cache=invalidate_gallery_cache,
        authenticated_actor=_authenticated_actor,
    )


def name_gallery_page():
    render_people_page(
        get_cached_all_persons=get_cached_all_persons,
        invalidate_persons_cache=invalidate_persons_cache,
        invalidate_gallery_cache=invalidate_gallery_cache,
        authenticated_actor=_authenticated_actor,
    )


if __name__ == "__main__":
    main()
