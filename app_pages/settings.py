"""Settings and privacy page UI."""
import tempfile
import uuid
from pathlib import Path

import numpy as np
import streamlit as st

from config import DATA_DIR, IMAGES_DIR, SCRAPED_DIR
from duplicate_detector import duplicate_detector
from privacy_data import delete_local_biometric_data
from settings_store import (
    active_embedding_model,
    active_embedding_pipeline_version,
    load_settings,
    save_settings,
    settings_path,
)
from utils import get_image_files


def render_settings_page(import_errors, show_embedding_pipeline_status, create_pipeline_change_backup, invalidate_persons_cache, invalidate_gallery_cache):
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

    with st.expander("Embedding model change backup", expanded=False):
        st.caption(
            "Changing the encoder or effective detector requires a full reindex. "
            "A searchable backup lets you restore the previous index after switching back. "
            "It contains biometric embeddings and metadata, but not original images."
        )
        backup_state = st.session_state.get("pipeline_change_backup") or {}
        pipeline_status = store.get_embedding_pipeline_status() if store is not None else {}
        indexed_faces = int(pipeline_status.get("total_faces", 0))
        if indexed_faces and pipeline_status.get("compatible") and pipeline_status.get("complete"):
            if st.button("Prepare current index backup", key="prepare_pipeline_backup"):
                outcome = create_pipeline_change_backup(store)
                if outcome.get("success"):
                    st.session_state.pipeline_change_backup = outcome
                    st.session_state["confirm_pipeline_backup"] = False
                    backup_state = outcome
                    st.success("Backup is ready. Download it and keep a copy outside the app data folder.")
                else:
                    st.error(outcome.get("error", "Could not create the index backup."))
        elif indexed_faces:
            st.info("Finish or restore the current index update before creating a migration backup.")

        backup_path = Path(str(backup_state.get("backup_path") or ""))
        backup_is_current = (
            backup_path.is_file()
            and backup_state.get("pipeline_version") == getattr(store, "_embedding_pipeline_version", None)
            and int(backup_state.get("face_count", -1)) == indexed_faces
        )
        if backup_is_current:
            st.download_button(
                "Download searchable index backup",
                data=backup_path.read_bytes(),
                file_name=backup_path.name,
                mime="application/zip",
                key="download_pipeline_backup",
            )
            st.checkbox(
                "I saved this backup outside the app data folder.",
                key="confirm_pipeline_backup",
                help="The local copy is on the same machine as the database. Keep a separate copy before changing the face pipeline.",
            )
            st.caption(f"Local backup: {backup_path}")
        elif backup_state:
            st.warning("The saved backup no longer matches the current index. Create a fresh backup before changing pipelines.")

    st.subheader("Simple settings")
    st.caption("Start with the recommended face model and search threshold. Defaults are ready for most personal libraries.")
    st.caption(f"Saved to: {settings_path()}")

    with st.form("settings_form"):
        embedding_model = st.selectbox(
            "Face search embedding model",
            options=["Buffalo_L", "ArcFace"],
            index=0 if current_settings["embedding_model"] == "buffalo_l" else 1,
            key="settings_embedding_model",
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
        with st.expander("Advanced settings", expanded=False):
            st.caption("Adjust processing limits and the optional fallback detector.")
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
        store = st.session_state.get("vector_store")
        requested_pipeline = active_embedding_pipeline_version(
            model_name=embedding_model.lower(),
            detection_model=detection_model,
        )
        pipeline_change = (
            store is not None
            and int(store.collection.count()) > 0
            and requested_pipeline != store._embedding_pipeline_version
        )
        if pipeline_change:
            backup_state = st.session_state.get("pipeline_change_backup") or {}
            backup_path = Path(str(backup_state.get("backup_path") or ""))
            backup_ready = (
                backup_path.is_file()
                and backup_state.get("pipeline_version") == store._embedding_pipeline_version
                and int(backup_state.get("face_count", -1)) == int(store.collection.count())
            )
            if not backup_ready:
                outcome = create_pipeline_change_backup(store)
                if not outcome.get("success"):
                    st.error(outcome.get("error", "Create a complete index backup before changing the model."))
                    return
                st.session_state.pipeline_change_backup = outcome
                st.session_state["confirm_pipeline_backup"] = False
                st.session_state.settings_flash = (
                    "A searchable backup was created. Download it, save a separate copy, "
                    "and confirm before submitting the pipeline change again."
                )
                st.rerun()
            if not st.session_state.get("confirm_pipeline_backup", False):
                st.error("Download the searchable backup, save it outside the app data folder, and confirm before changing the face pipeline.")
                return

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

    st.divider()
    st.subheader("Privacy and local data")
    with st.expander("Delete biometric data and backups"):
        st.warning(
            "This permanently clears indexed face records, person labels, cached thumbnails, "
            "and local recovery archives. It cannot be undone."
        )
        delete_sources = st.checkbox(
            "Also delete original images in the app folders and files referenced by the index.",
            key="delete_biometric_source_images",
            help="Referenced images may be outside the app data folder. They are removed only when this option is selected.",
        )
        deletion_confirmation_nonce = st.session_state.setdefault(
            "delete_biometric_nonce", uuid.uuid4().hex[:12]
        )
        confirmation = st.text_input(
            "Type DELETE to confirm",
            key=f"delete_biometric_confirmation_{deletion_confirmation_nonce}",
        )
        if st.button(
            "Permanently delete local biometric data",
            key="delete_local_biometric_data",
            type="secondary",
        ):
            if confirmation != "DELETE":
                st.error("Type DELETE to confirm the removal.")
            else:
                outcome = delete_local_biometric_data(
                    st.session_state.get("vector_store"),
                    delete_source_images=delete_sources,
                    managed_image_directories=(IMAGES_DIR, SCRAPED_DIR),
                    derived_directories=(DATA_DIR / "embeddings",),
                    backup_directories=(
                        DATA_DIR / "duplicate_backups",
                        DATA_DIR / "pipeline_backups",
                    ),
                    metadata_stores=(duplicate_detector,),
                    auxiliary_files=(
                        Path(__file__).resolve().parent / "duplicates.db",
                    ),
                )
                invalidate_persons_cache()
                invalidate_gallery_cache()
                st.session_state.search_results = []
                st.session_state.pop("detected_faces", None)
                st.session_state.pop("person_groups_last_assignment", None)
                if outcome.get("success"):
                    st.session_state.delete_biometric_nonce = uuid.uuid4().hex[:12]
                    st.success(
                        f"Deleted {outcome['faces_removed']} face records, "
                        f"{outcome['backup_files_removed']} recovery archives, and "
                        f"{outcome['images_removed']} image files. Duplicate-file metadata was cleared."
                    )
                    st.rerun()
                else:
                    st.error(outcome.get("error", "Local data deletion failed."))
