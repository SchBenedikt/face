"""Reusable Streamlit components shared across pages."""
from pathlib import Path

import streamlit as st

from config import THUMBNAIL_SIZE
from ui_cache import face_thumbnail
from utils import parse_face_location


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
