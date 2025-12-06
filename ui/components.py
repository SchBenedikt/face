"""
Reusable UI Components
"""
import streamlit as st
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import cv2


def display_face_thumbnail(
    image_path: str,
    face_location: tuple,
    face_id: str,
    show_metadata: bool = True,
    thumbnail_size: tuple = (150, 150)
):
    """
    Display a face thumbnail with metadata
    
    Args:
        image_path: Path to the image file
        face_location: Face bounding box (top, right, bottom, left)
        face_id: Unique face identifier
        show_metadata: Whether to show metadata below thumbnail
        thumbnail_size: Size of thumbnail to create
    """
    try:
        from utils import load_and_preprocess_image, create_thumbnail
        
        # Load image
        full_image = load_and_preprocess_image(Path(image_path))
        if full_image is None:
            st.error("Failed to load image")
            return
        
        # Extract face region
        if isinstance(face_location, str):
            coords = face_location.split(',')
            if len(coords) == 4:
                top, right, bottom, left = map(int, coords)
            else:
                st.error("Invalid face coordinates")
                return
        else:
            top, right, bottom, left = face_location
        
        # Crop face with padding
        face_width = right - left
        face_height = bottom - top
        padding = max(10, min(20, min(face_width, face_height) // 10))
        
        height, width = full_image.shape[:2]
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(height, bottom + padding)
        right = min(width, right + padding)
        
        face_image = full_image[top:bottom, left:right]
        
        if face_image.shape[0] > 0 and face_image.shape[1] > 0:
            thumbnail = create_thumbnail(face_image, thumbnail_size)
            st.image(thumbnail, width='stretch')
            
            if show_metadata:
                st.caption(f"🆔 {face_id[:12]}...")
                st.caption(f"📁 {Path(image_path).name}")
        else:
            st.error("Invalid face region")
            
    except Exception as e:
        st.error(f"Error displaying face: {e}")


def display_search_results_grid(
    results: List[Dict[str, Any]],
    cols_per_row: int = 5,
    show_similarity: bool = True
):
    """
    Display search results in a grid layout
    
    Args:
        results: List of search result dictionaries
        cols_per_row: Number of columns per row
        show_similarity: Whether to show similarity scores
    """
    if not results:
        st.info("No results to display")
        return
    
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(results):
                result = results[idx]
                with col:
                    _display_result_card(result, show_similarity)


def _display_result_card(result: Dict[str, Any], show_similarity: bool):
    """Display a single result card"""
    try:
        metadata = result.get('metadata', {})
        image_path = metadata.get('image_path', '')
        face_location = metadata.get('location', '')
        face_id = result.get('face_id', metadata.get('face_id', 'unknown'))
        similarity = result.get('similarity', 0.0)
        
        # Display face thumbnail
        display_face_thumbnail(
            image_path,
            face_location,
            face_id,
            show_metadata=False,
            thumbnail_size=(120, 120)
        )
        
        # Display similarity score
        if show_similarity:
            st.progress(similarity)
            st.caption(f"Ähnlichkeit: {similarity*100:.1f}%")
        
        # Display name if available
        person_name = metadata.get('full_name', '')
        if person_name:
            st.success(f"👤 {person_name}")
        
        # Info button
        if st.button("ℹ️", key=f"info_{face_id}", help="Details anzeigen"):
            st.session_state.info_face_id = face_id
            st.session_state.info_image_path = image_path
            st.session_state.info_face_location = face_location
            st.rerun()
            
    except Exception as e:
        st.error(f"Error displaying result: {e}")


def show_progress_bar(
    current: int,
    total: int,
    status_text: str = "Processing..."
):
    """
    Display a progress bar with status
    
    Args:
        current: Current progress value
        total: Total value
        status_text: Status message to display
    """
    progress = current / max(total, 1)
    st.progress(progress)
    st.text(f"{status_text} ({current}/{total})")


def show_stats_metrics(stats: Dict[str, Any]):
    """
    Display statistics metrics in columns
    
    Args:
        stats: Dictionary of statistics to display
    """
    if not stats or "error" in stats:
        st.warning("No statistics available")
        return
    
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


def show_face_quality_indicator(quality_score: float):
    """
    Display face quality indicator
    
    Args:
        quality_score: Quality score between 0 and 1
    """
    if quality_score >= 0.8:
        st.success(f"✅ Hohe Qualität ({quality_score*100:.0f}%)")
    elif quality_score >= 0.6:
        st.info(f"✓ Gute Qualität ({quality_score*100:.0f}%)")
    elif quality_score >= 0.4:
        st.warning(f"⚠️ Mittlere Qualität ({quality_score*100:.0f}%)")
    else:
        st.error(f"❌ Niedrige Qualität ({quality_score*100:.0f}%)")


def show_model_info(model_info: Dict[str, Any]):
    """
    Display model information
    
    Args:
        model_info: Dictionary containing model information
    """
    st.markdown(f"### 🤖 {model_info.get('name', 'Unknown Model')}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Dimension:** {model_info.get('dimension', 'N/A')}")
    
    with col2:
        st.write(f"**Genauigkeit:** {model_info.get('accuracy', 'N/A')}")
    
    with col3:
        st.write(f"**Geschwindigkeit:** {model_info.get('speed', 'N/A')}")
    
    st.write(f"**Beschreibung:** {model_info.get('description', 'N/A')}")
