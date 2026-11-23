"""Database statistics page UI."""
import streamlit as st


def render_database_stats_page(show_embedding_pipeline_status, invalidate_persons_cache, invalidate_gallery_cache):
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
