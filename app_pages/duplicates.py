"""Duplicate detection, review, backup, and cleanup page."""
from pathlib import Path
import hashlib
import logging

import streamlit as st

from config import DATA_DIR, IMAGES_DIR, SCRAPED_DIR
from duplicate_backup import create_duplicate_backup, restore_duplicate_backup
from duplicate_detector import delete_duplicate_files, duplicate_detector
from ui_cache import face_thumbnail
from utils import get_image_files

logger = logging.getLogger(__name__)


def _render_duplicate_set(directory, hash_value, file_paths, index, invalidate_persons_cache, invalidate_gallery_cache):
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
        backup_state_key = f"duplicate_backup_{hash_value}"
        backup_state = st.session_state.get(backup_state_key)
        deleted_candidates = [Path(path) for path in file_paths if Path(path) != Path(keep)]
        selection_hash = hashlib.sha256(
            "\0".join([str(Path(keep)), *sorted(str(path) for path in deleted_candidates)]).encode("utf-8")
        ).hexdigest()
        if backup_state and backup_state.get("selection_hash") != selection_hash:
            old_backup = backup_state.get("backup_path")
            if old_backup:
                Path(old_backup).unlink(missing_ok=True)
            st.session_state.pop(backup_state_key, None)
            backup_state = None
        if backup_state is None:
            store = st.session_state.get("vector_store")
            if store is None:
                backup_state = {"success": False, "message": "The face database is unavailable; cleanup is disabled."}
            else:
                with st.spinner("Preparing a local recovery archive for indexed faces…"):
                    backup_state = create_duplicate_backup(
                        store,
                        deleted_candidates,
                        DATA_DIR / "duplicate_backups",
                    )
            st.session_state[backup_state_key] = {
                **backup_state,
                "backup_path": str(backup_state["backup_path"]) if backup_state.get("backup_path") else None,
                "selection_hash": selection_hash,
            }
            backup_state = st.session_state[backup_state_key]

        if backup_state.get("success"):
            if backup_state.get("backup_path"):
                backup_path = Path(backup_state["backup_path"])
                st.success(
                    f"Recovery archive saved locally for {backup_state.get('face_count', 0)} indexed faces. "
                    "You can download it here or restore it later from this page."
                )
                if backup_path.is_file():
                    st.download_button(
                        "Download recovery archive",
                        data=backup_path.read_bytes(),
                        file_name=backup_path.name,
                        mime="application/zip",
                        key=f"download_duplicate_backup_{hash_value}",
                    )
            else:
                st.info(backup_state.get("message", "No indexed faces need a recovery archive."))
        else:
            st.error(backup_state.get("message", "Could not prepare a recovery archive."))

        st.warning(
            f"This removes {len(file_paths) - 1} exact duplicate file(s) and their indexed face records. "
            "The selected file will be kept. A local recovery archive is created first when indexed faces exist."
        )
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            confirm = st.button(
                "Confirm cleanup",
                key=f"confirm_delete_{hash_value}",
                type="primary",
                disabled=not backup_state.get("success", False),
            )
        with cancel_col:
            cancel = st.button("Cancel", key=f"cancel_delete_{hash_value}")
        if cancel:
            st.session_state.pop(pending_key, None)
            st.session_state.pop(backup_state_key, None)
            st.rerun()
        if confirm:
            st.session_state.pop(pending_key, None)
            # Recheck the candidate set immediately before deleting: a rescan
            # or another session may have changed the files since preview.
            existing_paths = [Path(path) for path in file_paths if Path(path).is_file()]
            current_duplicates = duplicate_detector.find_duplicates_in_directory(directory, use_content_hash=True)
            current_paths = current_duplicates.get(hash_value, [])
            if (
                Path(keep) not in existing_paths
                or len(existing_paths) < 2
                or set(existing_paths) != set(current_paths)
            ):
                st.session_state.pop(pending_key, None)
                st.session_state.pop(backup_state_key, None)
                st.session_state.duplicate_results.pop(str(directory), None)
                st.session_state.duplicate_flash = [
                    "The duplicate set changed since review. Scan the folder again."
                ]
                st.rerun()
            file_paths = existing_paths
            result = delete_duplicate_files(file_paths, keep)
            if (
                not result["deleted"]
                and result.get("rolled_back")
                and backup_state.get("backup_path")
            ):
                # The file transaction returned to its original state, so the
                # just-created temporary recovery copy is no longer needed.
                Path(backup_state["backup_path"]).unlink(missing_ok=True)

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
                if backup_state.get("backup_path"):
                    messages.append(
                        f"Recovery archive retained at {Path(backup_state['backup_path']).name}. "
                        "Restore it from the Duplicate manager page if needed."
                    )
            if result["missing"]:
                names = ", ".join(path.name for path in result["missing"])
                messages.append(f"Already missing: {names}")
            if result["errors"]:
                messages.append("⚠️ " + "; ".join(result["errors"]))
            if not messages:
                messages.append("Nothing was deleted.")

            st.session_state.duplicate_flash = messages
            st.session_state.pop(backup_state_key, None)
            st.rerun()


def render_duplicate_manager_page(invalidate_persons_cache, invalidate_gallery_cache):
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
                st.caption("Only byte-identical files are grouped as duplicates.")

                if st.button(f"🔍 Find Duplicates in {directory.name}", key=f"find_dups_{directory.name}"):
                    with st.spinner(f"Scanning {directory.name} for duplicates..."):
                        st.session_state.duplicate_results[str(directory)] = (
                            duplicate_detector.find_duplicates_in_directory(
                                directory, use_content_hash=True
                            )
                        )

                found_sets = st.session_state.duplicate_results.get(str(directory)) or {}
                if found_sets:
                    st.error(f"🔍 Found {len(found_sets)} sets of duplicate images:")
                    for index, (hash_value, file_paths) in enumerate(sorted(found_sets.items())):
                        _render_duplicate_set(
                            directory,
                            hash_value,
                            file_paths,
                            index,
                            invalidate_persons_cache,
                            invalidate_gallery_cache,
                        )
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

    backup_directory = DATA_DIR / "duplicate_backups"
    recovery_archives = sorted(backup_directory.glob("duplicate-recovery-*.zip"), reverse=True) if backup_directory.exists() else []
    with st.expander(f"Recovery archives ({len(recovery_archives)})"):
        st.caption("Cleanup backups contain source images and biometric embeddings. Keep them only as long as needed.")
        if not recovery_archives:
            st.info("No duplicate cleanup backups are stored locally.")
        else:
            for archive_path in recovery_archives:
                col_info, col_download, col_restore = st.columns([3, 1, 1])
                with col_info:
                    st.write(f"{archive_path.name} · {archive_path.stat().st_size / (1024 * 1024):.1f} MiB")
                with col_download:
                    st.download_button(
                        "Download",
                        data=archive_path.read_bytes(),
                        file_name=archive_path.name,
                        mime="application/zip",
                        key=f"download_recovery_{archive_path.name}",
                    )
                with col_restore:
                    if st.button("Restore", key=f"restore_recovery_{archive_path.name}"):
                        store = st.session_state.get("vector_store")
                        if store is None:
                            st.error("The face database is unavailable. The archive was kept.")
                        else:
                            with st.spinner("Restoring faces and original images…"):
                                restored = restore_duplicate_backup(archive_path, store, IMAGES_DIR)
                            if restored.get("success"):
                                invalidate_persons_cache()
                                invalidate_gallery_cache()
                                st.success(
                                    f"Restored {restored.get('restored_faces', 0)} faces and "
                                    f"{restored.get('restored_images', 0)} images."
                                )
                                st.rerun()
                            else:
                                st.error(f"Restore failed; the recovery archive was retained: {restored.get('error')}")
