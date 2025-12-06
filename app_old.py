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
import streamlit as st
import asyncio
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import time
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# Import our modules
from face_recognition_engine import FaceRecognitionEngine
from vector_store import FaceVectorStore
from old_image_scraper import AdvancedImageScraper, advanced_scraper
from duplicate_detector import duplicate_detector
from utils import load_and_preprocess_image, create_thumbnail, get_image_files
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, 
    RESULTS_PER_PAGE, THUMBNAIL_SIZE,
    IMAGES_DIR, SCRAPED_DIR,
    FACE_RECOGNITION_MODEL
)

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'face_engine' not in st.session_state:
    st.session_state.face_engine = FaceRecognitionEngine()

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = FaceVectorStore()

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

def main():
    """Main application function"""
    
    st.title("🔍 Face Recognition Search")
    
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Navigation")
        
        page = st.selectbox(
            "Choose a page:",
            [
                "🔍 Face Search",
                "📥 Image Upload & Processing", 
                "🧠 Batch Face Processing",
                "🌐 Web Scraping",
                "👥 Face Gallery",
                "🏷️ Name Gallery",
                "🔧 Duplicate Manager",
                "📊 Database Statistics",
                "⚙️ Settings"
            ]
        )
        
        st.markdown("---")
        
        # Quick stats
        stats = st.session_state.vector_store.get_collection_stats()
        st.subheader("📈 Quick Stats")
        
        if "error" not in stats:
            st.metric("Total Faces", stats.get("total_faces", 0))
            st.metric("Unique Images", stats.get("unique_images", 0))
        else:
            st.warning("Database not initialized")
    
    # Handle page changes and clear modal states when switching pages
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
    if page == "🔍 Face Search":
        face_search_page()
    elif page == "📥 Image Upload & Processing":
        upload_processing_page()
    elif page == "🧠 Batch Face Processing":
        batch_face_processing_page()
    elif page == "🌐 Web Scraping":
        web_scraping_page()
    elif page == "👥 Face Gallery":
        face_gallery_page()
    elif page == "🏷️ Name Gallery":
        name_gallery_page()
    elif page == "🔧 Duplicate Manager":
        duplicate_manager_page()
    elif page == "📊 Database Statistics":
        database_stats_page()
    elif page == "⚙️ Settings":
        settings_page()

def face_search_page():
    """Face search functionality with multi-face selection"""
    
    st.header("🔍 Search for Similar Faces")
    
    # Upload query image
    uploaded_file = st.file_uploader(
        "Upload a photo to search for similar faces:",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Upload an image containing a face to search for similar faces in the database"
    )
    
    # Preserve uploaded file in session state for UI consistency
    if uploaded_file is not None:
        st.session_state.current_uploaded_file = uploaded_file
    elif 'current_uploaded_file' in st.session_state:
        uploaded_file = st.session_state.current_uploaded_file
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file is not None:
            # Display uploaded image
            query_image = Image.open(uploaded_file)
            st.image(query_image, caption="Query Image", width='stretch')
            
            # Enhanced Search Parameters
            st.subheader("🎯 Enhanced Search Parameters")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                max_results = st.slider(
                    "Maximum Results", 
                    min_value=5, 
                    max_value=200, 
                    value=50,
                    help="Maximum number of similar faces to return"
                )
                
                similarity_threshold = st.slider(
                    "Similarity Threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.4,
                    step=0.05,
                    help="Mindestähnlichkeit für Suchergebnisse (0.4 = 40% Ähnlichkeit). Höhere Werte = genauere Ergebnisse aber weniger Treffer."
                )
                
                # Image Quality Filter
                quality_filter = st.selectbox(
                    "🖼️ Image Quality Filter",
                    ["All Quality", "High Quality Only", "Medium+ Quality", "Remove Poor Quality"],
                    index=2,
                    help="Filter results by image quality to improve accuracy"
                )
            
            with col_b:
                # Face Size Filter
                min_face_size = st.slider(
                    "Minimum Face Size (px)",
                    min_value=20,
                    max_value=200,
                    value=50,
                    help="Filter out very small faces which are often false positives"
                )
                
                # Search Mode
                search_mode = st.selectbox(
                    "🎯 Search Mode",
                    ["Balanced", "High Precision", "High Recall"],
                    index=0,
                    help="Balanced: Normal results | Precision: Fewer but better matches | Recall: More results, some false positives"
                )
                
                # Result Sorting
                sort_by = st.selectbox(
                    "📊 Sort Results By",
                    ["Similarity Score", "Image Quality", "Face Size", "Date Added", "File Name"],
                    index=0,
                    help="Order search results by different criteria"
                )
            
            # Advanced Filters
            with st.expander("🔧 Advanced Filters", expanded=False):
                col_adv1, col_adv2 = st.columns(2)
                
                with col_adv1:
                    confidence_threshold = st.slider(
                        "Confidence Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.8,
                        step=0.05,
                        help="Minimum confidence in face detection (higher = more reliable faces)"
                    )
                    
                    exclude_blurry = st.checkbox(
                        "Exclude Blurry Images",
                        value=True,
                        help="Filter out blurry or low-quality images"
                    )
                
                with col_adv2:
                    aspect_ratio_filter = st.checkbox(
                        "Filter Extreme Aspect Ratios",
                        value=True,
                        help="Remove faces with unusual width/height ratios (often false positives)"
                    )
                    
                    brightness_filter = st.checkbox(
                        "Filter Dark/Overexposed Images",
                        value=False,
                        help="Remove images with poor lighting conditions"
                    )
            
            # Facial Attribute Analysis Info
            with st.expander("🧬 Facial Attribute Analysis", expanded=False):
                st.markdown("""
                **🎯 New Feature: Detailed Facial Analysis**
                
                In addition to similarity search, you can now perform detailed analysis on any found face:
                
                **📊 Analyzed Attributes:**
                - **👤 Age**: Estimated age (±4.65 years accuracy)  
                - **⚧️ Gender**: Male/Female (97.44% accuracy)
                - **😊 Emotions**: Happy, Neutral, Sad, Angry, Surprised, Fearful, Disgusted
                - **🌍 Ethnicity**: Asian, Caucasian, Middle Eastern, Indian, Latino, African
                
                **🔬 Technical Details:**
                - Based on DeepFace library with neural networks
                - High-precision models for each attribute type
                - Confidence values for every prediction
                
                **🚀 Usage:**
                Click the **🧬 Analysis** button for any search result to get detailed information.
                """)
                
                # Check if DeepFace is available
                try:
                    import deepface
                    st.success("✅ DeepFace is installed and ready for facial analysis!")
                except ImportError:
                    st.warning("⚠️ DeepFace not installed. Install with: `pip install deepface`")
                    if st.button("📥 Show DeepFace Installation"):
                        st.code("pip install deepface", language="bash")
                        st.info("Restart the application after installation.")
            
            # Step 1: Analyze uploaded image for faces
            if st.button("🔍 Analyze Image for Faces", type="primary"):
                # Store search parameters in session state for enhanced search
                st.session_state.quality_filter = quality_filter
                st.session_state.min_face_size = min_face_size
                st.session_state.search_mode = search_mode
                st.session_state.sort_by = sort_by
                st.session_state.confidence_threshold = confidence_threshold
                st.session_state.exclude_blurry = exclude_blurry
                st.session_state.aspect_ratio_filter = aspect_ratio_filter
                st.session_state.brightness_filter = brightness_filter
                
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
            st.subheader("🎯 Search Results")
        with col_clear:
            if st.button("🗑️ Clear", help="Clear search results", key="clear_search_results"):
                st.session_state.search_results = []
                if 'detected_faces' in st.session_state:
                    del st.session_state.detected_faces
                if 'current_uploaded_file' in st.session_state:
                    del st.session_state.current_uploaded_file
                st.rerun()
        
        display_search_results(st.session_state.search_results)

def analyze_uploaded_image_for_faces(uploaded_file, max_results: int, similarity_threshold: float):
    """Analyze uploaded image and detect all faces"""
    
    temp_path = Path("/tmp/query_image.jpg")
    
    with st.spinner("🔍 Analyzing image and detecting faces..."):
        try:
            # Save uploaded file temporarily
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Load and preprocess image
            query_image = load_and_preprocess_image(temp_path)
            
            if query_image is None:
                st.error("❌ Error loading image. Supported formats: JPG, PNG, BMP, WEBP")
                return
            
            # Detect faces
            face_locations = st.session_state.face_engine.detect_faces(query_image)
            
            if not face_locations:
                st.error("""
                ❌ **Keine Gesichter erkannt!** 
                
                **Tipps für bessere Ergebnisse:**
                - Verwenden Sie ein Bild mit deutlich sichtbarem Gesicht
                - Stellen Sie sicher, dass das Gesicht gut beleuchtet ist
                - Vermeiden Sie zu kleine Bilder (Mindestgröße: 30x30 Pixel pro Gesicht)
                - Probieren Sie ein anderes Bild mit frontaler Gesichtsansicht
                """)
                return
            
            # Store detected faces in session state
            st.session_state.detected_faces = {
                'image_data': uploaded_file.getvalue(),
                'face_locations': face_locations,
                'query_image': query_image,
                'max_results': max_results,
                'similarity_threshold': similarity_threshold,
                'temp_path': str(temp_path)
            }
            
            st.success(f"✅ {len(face_locations)} face(s) detected! Select the desired face for search on the right.")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error analyzing image: {str(e)}")
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

def show_face_selection_interface():
    """Show interface for selecting which face to search for"""
    
    st.subheader("👥 Face Selection")
    
    detected_data = st.session_state.detected_faces
    face_locations = detected_data['face_locations']
    query_image = detected_data['query_image']
    
    st.write(f"**{len(face_locations)} face(s) detected. Select the face for search:**")
    
    # Display faces in a grid
    cols = st.columns(min(len(face_locations), 3))
    
    for idx, face_loc in enumerate(face_locations):
        with cols[idx % 3]:
            # Extract face region
            if isinstance(face_loc, dict):
                top, right, bottom, left = face_loc['top'], face_loc['right'], face_loc['bottom'], face_loc['left']
            else:
                top, right, bottom, left = face_loc
            
            # Extract and display face
            face_image = query_image[top:bottom, left:right]
            face_thumbnail = create_thumbnail(face_image, (120, 120))
            
            st.image(face_thumbnail, caption=f"Gesicht {idx+1}")
            
            # Search button for this face with enhanced parameters
            if st.button(f"🔍 Nach Gesicht {idx+1} suchen", key=f"search_face_{idx}", width='stretch'):
                search_with_selected_face_enhanced(idx)
    
    # Clear faces button - only clear face detection, preserve search results
    if st.button("🗑️ Reset Face Detection", key="clear_faces"):
        if 'detected_faces' in st.session_state:
            del st.session_state.detected_faces
        st.rerun()

def search_with_selected_face_enhanced(face_index: int):
    """Perform enhanced search using the selected face with advanced filtering"""
    
    detected_data = st.session_state.detected_faces
    face_locations = detected_data['face_locations']
    query_image = detected_data['query_image']
    max_results = detected_data['max_results']
    similarity_threshold = detected_data['similarity_threshold']
    
    # Get advanced search parameters from session state
    search_params = {
        'quality_filter': st.session_state.get('quality_filter', 'Medium+ Quality'),
        'min_face_size': st.session_state.get('min_face_size', 50),
        'search_mode': st.session_state.get('search_mode', 'Balanced'),
        'sort_by': st.session_state.get('sort_by', 'Similarity Score'),
        'confidence_threshold': st.session_state.get('confidence_threshold', 0.8),
        'exclude_blurry': st.session_state.get('exclude_blurry', True),
        'aspect_ratio_filter': st.session_state.get('aspect_ratio_filter', True),
        'brightness_filter': st.session_state.get('brightness_filter', False)
    }
    
    selected_face_location = face_locations[face_index]
    
    with st.spinner(f"🔍 Enhanced search for Face {face_index + 1} with quality filters..."):
        try:
            # Extract embedding for selected face
            query_embedding = st.session_state.face_engine.extract_face_embedding(query_image, selected_face_location)
            
            if query_embedding is None:
                st.error("❌ Fehler beim Extrahieren der Gesichtsmerkmale. Versuchen Sie ein anderes Gesicht.")
                return
            
            # Adjust search parameters based on mode
            adjusted_threshold = similarity_threshold
            adjusted_max_results = max_results
            
            if search_params['search_mode'] == "High Precision":
                adjusted_threshold = min(similarity_threshold + 0.1, 0.9)
                adjusted_max_results = min(max_results, 25)
            elif search_params['search_mode'] == "High Recall":
                adjusted_threshold = max(similarity_threshold - 0.1, 0.2)
                adjusted_max_results = max_results * 2
            
            # Search for similar faces (local database)
            similar_faces = st.session_state.vector_store.search_similar_faces(
                query_embedding, 
                n_results=adjusted_max_results,
                min_similarity=adjusted_threshold
            )
            
            # Use local results only
            all_results = similar_faces
            
            # Apply advanced filters to combined results
            if all_results:
                filtered_faces = apply_advanced_face_filters(all_results, search_params)
                
                # Sort results based on selected criteria
                sorted_faces = sort_search_results(filtered_faces, search_params['sort_by'])
                
                # Store results
                st.session_state.search_results = sorted_faces
                
                # Show results summary
                if sorted_faces:
                    original_count = len(all_results)
                    filtered_count = len(sorted_faces)
                    avg_similarity = sum(face['similarity'] for face in sorted_faces) / len(sorted_faces)
                    top_similarity = sorted_faces[0]['similarity'] if sorted_faces else 0
                    
                    st.success(f"""
                    🎯 **Search Results for Face {face_index + 1}**
                    
                    📊 **Search Summary:**
                    - Found: {filtered_count} faces (from {original_count} total)
                    - Top Similarity: {top_similarity*100:.1f}%
                    - Average Similarity: {avg_similarity*100:.1f}%
                    - Search Mode: {search_params['search_mode']}
                    - Quality Filter: {search_params['quality_filter']}
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **No high-quality matches found for Face {face_index + 1}**
                    
                    **Try adjusting:**
                    - Lower similarity threshold (currently: {similarity_threshold*100:.0f}%)
                    - Change quality filter to "All Quality"
                    - Switch to "High Recall" search mode
                    - Reduce minimum face size filter
                    """)
            else:
                st.warning(f"No similar faces found for Face {face_index + 1} in any database")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error in enhanced search: {str(e)}")

def apply_advanced_face_filters(faces, search_params):
    """Apply advanced quality and size filters to search results"""
    filtered_faces = []
    
    for face in faces:
        try:
            metadata = face.get('metadata', {})
            face_size = 50  # Default face size
            
            # Face size filter
            location_str = metadata.get('location', '0,0,0,0')
            coords = location_str.split(',')
            if len(coords) == 4:
                top, right, bottom, left = map(int, coords)
                face_width = right - left
                face_height = bottom - top
                face_size = min(face_width, face_height)
                
                if face_size < search_params['min_face_size']:
                    continue
                
                # Aspect ratio filter
                if search_params['aspect_ratio_filter']:
                    aspect_ratio = face_width / max(face_height, 1)
                    if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # Skip extremely wide/tall faces
                        continue
            
            # Confidence filter (if available in metadata)
            confidence = metadata.get('confidence', 1.0)
            if confidence < search_params['confidence_threshold']:
                continue
            
            # Quality-based filtering
            if search_params['quality_filter'] == "High Quality Only":
                # Only include faces with high similarity and good size
                if face['similarity'] < 0.7 or face_size < 80:
                    continue
            elif search_params['quality_filter'] == "Remove Poor Quality":
                # Remove obviously poor quality results
                if face['similarity'] < 0.3 or face_size < 30:
                    continue
            
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
    max_results = detected_data['max_results']
    similarity_threshold = detected_data['similarity_threshold']
    
    selected_face_location = face_locations[face_index]
    
    with st.spinner(f"🔍 Suche nach ähnlichen Gesichtern für Gesicht {face_index + 1}..."):
        try:
            # Extract embedding for selected face
            query_embedding = st.session_state.face_engine.extract_face_embedding(query_image, selected_face_location)
            
            if query_embedding is None:
                st.error("❌ Fehler beim Extrahieren der Gesichtsmerkmale. Versuchen Sie ein anderes Gesicht.")
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
                🎯 **{len(similar_faces)} ähnliche Gesichter gefunden für Gesicht {face_index + 1}!**
                
                📊 **Ergebnisqualität:**
                - Top-Ähnlichkeit: {top_similarity*100:.1f}%
                - Durchschnitts-Ähnlichkeit: {avg_similarity*100:.1f}%
                """)
            else:
                st.warning(f"""
                ⚠️ **Keine ähnlichen Gesichter für Gesicht {face_index + 1} gefunden**
                
                **Versuchen Sie:**
                - Ähnlichkeitsschwelle senken (aktuell: {similarity_threshold*100:.0f}%)
                - Ein anderes Gesicht aus dem Bild wählen
                - Mehr Bilder zur Datenbank hinzufügen
                """)
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Fehler bei der Suche: {str(e)}")

def show_full_image_with_face_box(image_path, face_location):
    """Store image data and trigger full image modal display"""
    try:
        # Store data in session state for the modal
        st.session_state.full_image_path = image_path
        st.session_state.full_image_face_location = face_location
        
        # The modal will be triggered by the session state check
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Fehler beim Vorbereiten der Vollbildansicht: {str(e)}")

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
            return {"error": "Gesicht zu klein für Attribut-Analyse"}
        
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
                return {"error": f"Analyse-Fehler: {str(df_error)[:100]}..."}
                
    except ImportError:
        return {"error": "DeepFace nicht installiert. Bitte installieren Sie: pip install deepface"}
    except Exception as e:
        logger.error(f"Facial attribute analysis error: {e}")
        return {"error": f"Allgemeiner Fehler: {str(e)[:100]}..."}

def show_facial_attributes_popup(image_path, face_location, face_id="unknown"):
    """Display facial attributes analysis in a full-width dialog-style container"""
    
    with st.spinner("🧠 Analysiere Gesichtsattribute..."):
        analysis = analyze_facial_attributes(image_path, face_location)
    
    if analysis is None:
        st.error("❌ Keine Analyse möglich")
        return
    
    if "error" in analysis:
        st.error(f"❌ {analysis['error']}")
        return
    
    # Create a dialog-style container that takes full width
    st.markdown("---")  # Visual separator
    
    # Create a container that uses the full width
    with st.container():
        # Header with larger title
        st.markdown(f"# 🧬 **Detaillierte Gesichtsanalyse**")
        st.markdown(f"**Face ID:** `{face_id}`")
        st.markdown(f"**Datei:** `{Path(image_path).name}`")
        
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
                                st.image(face_crop, caption="Analysiertes Gesicht", width=200)
            except:
                pass
        
        st.markdown("---")
        
        # Analysis results in a clean 2-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Age Analysis
            if 'age' in analysis:
                st.markdown("## 🎂 **Altersschätzung**")
                age = analysis['age']
                st.metric("Geschätztes Alter", f"{age} Jahre", help="Genauigkeit: ±4,65 Jahre")
                
                # Age category with visual indicator
                if age < 18:
                    st.info("👶 **Kategorie:** Jugendlich")
                elif age < 30:
                    st.info("👤 **Kategorie:** Junge(r) Erwachsene(r)")
                elif age < 50:
                    st.info("👨 **Kategorie:** Erwachsene(r)")
                elif age < 70:
                    st.info("👴 **Kategorie:** Ältere(r) Erwachsene(r)")
                else:
                    st.info("👵 **Kategorie:** Senior(in)")
            
            st.markdown("---")
            
            # Gender Analysis
            if 'gender' in analysis:
                st.markdown("## ⚧️ **Geschlechtsbestimmung**")
                gender_data = analysis['gender']
                if isinstance(gender_data, dict):
                    male_prob = float(gender_data.get('Man', 0))
                    female_prob = float(gender_data.get('Woman', 0))
                    
                    # Visual representation with progress bars
                    st.write("**👨 Männlich**")
                    st.progress(male_prob / 100.0)
                    st.write(f"`{male_prob:.1f}%`")
                    
                    st.write("**👩 Weiblich**") 
                    st.progress(female_prob / 100.0)
                    st.write(f"`{female_prob:.1f}%`")
                    
                    # Highlight prediction
                    dominant = "👨 Männlich" if male_prob > female_prob else "👩 Weiblich"
                    confidence = max(male_prob, female_prob)
                    st.success(f"**Vorhersage:** {dominant} (Konfidenz: {confidence:.1f}%)")
                else:
                    st.write(f"**Geschlecht:** {gender_data}")
        
        with col2:
            # Emotion Analysis
            if 'emotion' in analysis:
                st.markdown("## 😊 **Emotionsanalyse**")
                emotions = analysis['emotion']
                if isinstance(emotions, dict):
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                    emotion_map = {
                        'happy': ('😊 Glücklich', '#28a745'),
                        'neutral': ('😐 Neutral', '#6c757d'),
                        'sad': ('😢 Traurig', '#007bff'),
                        'angry': ('😠 Wütend', '#dc3545'),
                        'surprise': ('😲 Überrascht', '#ffc107'),
                        'fear': ('😨 Ängstlich', '#fd7e14'),
                        'disgust': ('🤢 Angeekelt', '#6f42c1')
                    }
                    
                    # Show all emotions with visual bars
                    for emotion, confidence in sorted_emotions:
                        german_emotion, color = emotion_map.get(emotion, (emotion.title(), '#6c757d'))
                        st.write(f"**{german_emotion}**")
                        st.progress(float(confidence / 100.0))
                        st.write(f"`{confidence:.1f}%`")
                    
                    # Dominant emotion highlight
                    top_emotion, top_confidence = sorted_emotions[0]
                    german_top, _ = emotion_map.get(top_emotion, (top_emotion.title(), '#6c757d'))
                    st.success(f"**Dominante Emotion:** {german_top} ({top_confidence:.1f}%)")
            
            st.markdown("---")
            
            # Race/Ethnicity Analysis
            if 'race' in analysis:
                st.markdown("## 🌍 **Ethnische Herkunftsschätzung**")
                race_data = analysis['race']
                if isinstance(race_data, dict):
                    sorted_races = sorted(race_data.items(), key=lambda x: x[1], reverse=True)
                    race_map = {
                        'asian': '🌏 Asiatisch',
                        'white': '🌍 Kaukasisch', 
                        'middle eastern': '🌍 Nahöstlich',
                        'indian': '🌏 Indisch',
                        'latino hispanic': '🌎 Lateinamerikanisch',
                        'black': '🌍 Afrikanisch'
                    }
                    
                    # Show top predictions with progress bars
                    for race, confidence in sorted_races:
                        german_race = race_map.get(race.lower(), race.title())
                        st.write(f"**{german_race}**")
                        st.progress(float(confidence / 100.0))
                        st.write(f"`{confidence:.1f}%`")
                    
                    # Dominant prediction
                    top_race, top_confidence = sorted_races[0]
                    german_top_race = race_map.get(top_race.lower(), top_race.title())
                    st.info(f"**Hauptvorhersage:** {german_top_race} ({top_confidence:.1f}%)")
        
        # Footer with important disclaimer
        st.markdown("---")
        st.markdown("### ⚠️ **Wichtiger Hinweis**")
        st.warning("""
        Diese Analyse basiert auf KI-Modellen und dient ausschließlich zu **Demonstrations- und Forschungszwecken**. 
        
        **Bitte beachten Sie:**
        - Die Ergebnisse sind **Schätzungen** und können ungenau sein
        - Diese Technologie sollte **nicht für Identifikation, Diskriminierung oder Entscheidungsfindung** verwendet werden
        - Respektieren Sie die Privatsphäre und Rechte der abgebildeten Personen
        - Die Genauigkeit kann je nach Bildqualität, Beleuchtung und Kamerawinkel variieren
        """)
        
        # Close button
        st.markdown("---")
        if st.button("❌ Analyse schließen", key=f"close_analysis_{face_id}", type="primary"):
            st.rerun()

@st.dialog("👤 Erweiterte Namenszuweisung", width="large")
def show_name_assignment_modal():
    """Enhanced name assignment interface with extended fields and existing person selection"""
    
    # Required keys for name assignment
    required_keys = ['name_assign_face_id', 'name_assign_image_path', 'name_assign_face_location']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    
    if missing_keys:
        st.error(f"Fehlende Gesichtsdaten für Namenszuweisung: {', '.join(missing_keys)}")
        st.info("Bitte versuchen Sie erneut, den Namen zuzuweisen.")
        if st.button("❌ Schließen"):
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
        st.markdown(f"**Datei:** `{image_path.name}`")
    
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
                    st.image(face_crop, caption="Zuzuweisendes Gesicht", width=150)
        except Exception as e:
            st.warning(f"Konnte Gesicht nicht anzeigen: {e}")
    
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
        st.info(f"✅ **Aktueller Name:** {existing_data['full_name']}")
    
    # Get all existing persons for selection
    all_persons = get_cached_all_persons()  # Use cached version
    
    # Choice between new person or existing person
    st.subheader("🎯 Person auswählen")
    
    assignment_type = st.radio(
        "Wie möchten Sie den Namen zuweisen?",
        ["🆕 Neue Person erstellen", "👥 Bestehende Person auswählen"],
        key="assignment_type",
        horizontal=True
    )
    
    if assignment_type == "👥 Bestehende Person auswählen" and all_persons:
        # Show existing persons selection
        st.subheader("📋 Bestehende Personen")
        
        # Create searchable list of persons
        person_options = {"-- Bitte Person auswählen --": None}
        for person in all_persons:
            display_name = f"{person['full_name']}"
            if person.get('birth_date'):
                display_name += f" (*{person['birth_date']})"
            display_name += f" ({person['face_count']} Gesichter)"
            person_options[display_name] = person
        
        selected_person_display = st.selectbox(
            "Person auswählen:",
            options=list(person_options.keys()),
            key="selected_existing_person"
        )
        
        if selected_person_display and person_options[selected_person_display] is not None:
            selected_person = person_options[selected_person_display]
            
            # Ensure selected_person is not None
            if selected_person is None:
                st.error("Fehler beim Laden der Personendaten")
                return
            
            # Show person details in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Vollständiger Name:**")
                st.write(f"*{selected_person['full_name']}*")
                st.write(f"**Vorname:** {selected_person['first_name']}")
                if selected_person.get('middle_names'):
                    st.write(f"**Zweite Namen:** {selected_person['middle_names']}")
                st.write(f"**Nachname:** {selected_person['last_name']}")
            
            with col2:
                st.write(f"**Geburtsdatum:** {selected_person.get('birth_date', '-')}")
                st.write(f"**Geburtsort:** {selected_person.get('birth_place', '-')}")
                st.write(f"**Anzahl Gesichter:** {selected_person['face_count']}")
            
            with col3:
                if selected_person.get('notes'):
                    st.write(f"**Notizen:**")
                    st.write(selected_person['notes'])
            
            # Assign button
            if st.button("✅ Diese Person zuweisen", type="primary", width='stretch'):
                person_data = {
                    'first_name': selected_person['first_name'],
                    'middle_names': selected_person.get('middle_names', ''),
                    'last_name': selected_person['last_name'],
                    'birth_date': selected_person.get('birth_date', ''),
                    'birth_place': selected_person.get('birth_place', ''),
                    'notes': selected_person.get('notes', '')
                }
                
                result = st.session_state.vector_store.assign_person_name_with_auto_matching(
                    face_id, person_data, selected_person['person_id'], similarity_threshold=0.8
                )
                
                if result['success']:
                    st.success(f"✅ Person erfolgreich zugewiesen!")
                    
                    if result['total_assigned'] > 1:
                        st.success(f"🔄 {result['total_assigned']-1} ähnliche Gesichter (>80%) automatisch zugewiesen!")
                    
                    # Invalidate persons cache for refresh
                    invalidate_persons_cache()
                    
                    # Clear session state and close dialog
                    _clear_name_assignment_session()
                    st.rerun()
                else:
                    st.error("❌ Fehler beim Zuweisen der Person")
    
    elif assignment_type == "👥 Bestehende Person auswählen" and not all_persons:
        st.info("🔍 Keine bestehenden Personen gefunden. Erstellen Sie eine neue Person.")
    
    else:
        # New person creation form
        st.subheader("📝 Neue Person erstellen")
        
        # Extended person form in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Grunddaten:**")
            first_name = st.text_input(
                "Vorname: *", 
                value=existing_data.get('first_name', '') or '', 
                key="new_first_name"
            )
            middle_names = st.text_input(
                "Zweite Namen:", 
                value=existing_data.get('middle_names', '') or '',
                help="Z.B.: Moritz Luitpold Damian",
                key="new_middle_names"
            )
            last_name = st.text_input(
                "Nachname: *", 
                value=existing_data.get('last_name', '') or '', 
                key="new_last_name"
            )
        
        with col2:
            st.write("**Zusätzliche Informationen:**")
            birth_date = st.text_input(
                "Geburtsdatum:",
                value=existing_data.get('birth_date', ''),
                help="Z.B.: 1985-03-15 oder 15.03.1985",
                key="new_birth_date"
            )
            birth_place = st.text_input(
                "Geburtsort:",
                value=existing_data.get('birth_place', ''),
                help="Z.B.: München, Deutschland",
                key="new_birth_place"
            )
        
        notes = st.text_area(
            "Notizen:",
            value=existing_data.get('notes', ''),
            help="Zusätzliche Informationen zur Person",
            key="new_notes",
            height=80
        )
        
        # Preview full name
        if first_name or middle_names or last_name:
            name_parts = [p.strip() for p in [first_name or '', middle_names or '', last_name or ''] if p and p.strip()]
            full_name = ' '.join(name_parts)
            st.success(f"**Vollständiger Name:** {full_name}")
        
        # Save new person button
        if st.button("💾 Person speichern", type="primary", width='stretch'):
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
                
                # Assign person with auto-matching
                result = st.session_state.vector_store.assign_person_name_with_auto_matching(
                    face_id, person_data, similarity_threshold=0.8
                )
                
                if result['success']:
                    st.success(f"✅ Person erfolgreich erstellt und zugewiesen!")
                    
                    if result['total_assigned'] > 1:
                        st.success(f"🔄 {result['total_assigned']-1} ähnliche Gesichter (>80%) automatisch zugewiesen!")
                    
                    # Invalidate persons cache for refresh
                    invalidate_persons_cache()
                    
                    # Clear session state and close dialog
                    _clear_name_assignment_session()
                    st.rerun()
                else:
                    st.error("❌ Fehler beim Erstellen der Person")
            else:
                st.warning("⚠️ Bitte mindestens Vor- oder Nachname eingeben")
    
    # Action buttons at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if existing_data.get('full_name'):
            if st.button("🗑️ Namen entfernen", width='stretch'):
                if st.session_state.vector_store.remove_person_name(face_id):
                    st.success("✅ Name erfolgreich entfernt!")
                    # Invalidate persons cache for refresh
                    invalidate_persons_cache()
                    _clear_name_assignment_session()
                    st.rerun()
                else:
                    st.error("❌ Fehler beim Entfernen des Namens")
    
    with col3:
        if st.button("❌ Abbrechen", width='stretch'):
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

@st.dialog("🧬 Detaillierte Gesichtsanalyse", width="large")
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

@st.dialog("ℹ️ Gesicht Informationen", width="large")
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
        st.error(f"Fehlende Gesichtsdaten: {', '.join(missing_keys)}")
        if st.button("❌ Schließen"):
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
    st.markdown(f"**📁 Datei:** `{image_path.name}`")
    
    # Tabs for different information types
    tab1, tab2, tab3 = st.tabs(["🖼️ Vollbild", "🧬 Gesichtsanalyse", "🏷️ Namen zuweisen"])
    
    with tab1:
        show_full_image_tab(image_path, face_location)
    
    with tab2:
        show_analysis_tab(image_path, face_location, face_id)
    
    with tab3:
        show_name_assignment_tab(face_id, image_path, face_location)
    
    # Close button
    st.markdown("---")
    if st.button("✅ Schließen", key="close_face_info_modal", type="primary", width='stretch'):
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
            st.error("❌ Fehler beim Laden des Bildes")
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
                    st.success(f"✅ Gesicht markiert bei Koordinaten: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
            else:
                top, right, bottom, left = face_location
                cv2.rectangle(image_with_box, (left, top), (right, bottom), (0, 255, 0), 3)
                st.success(f"✅ Gesicht markiert bei Koordinaten: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
        
        # Display the image with maximum width
        caption = f"Vollbild - {image_path.name}"
        if face_location:
            caption += " (mit markiertem Gesicht)"
            
        st.image(image_with_box, caption=caption, width='stretch')
        
        # Image info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Bildgröße", f"{full_image.shape[1]} × {full_image.shape[0]} px")
        
        with col2:
            file_size = image_path.stat().st_size if image_path.exists() else 0
            st.metric("Dateigröße", f"{file_size / 1024:.1f} KB")
        
        with col3:
            if face_location:
                st.metric("Gesicht erkannt", "✅ Ja")
            else:
                st.metric("Gesicht erkannt", "❌ Nein")
    
    except Exception as e:
        st.error(f"❌ Fehler beim Anzeigen des Vollbildes: {str(e)}")

def show_analysis_tab(image_path, face_location, face_id):
    """Display facial analysis in tab"""
    
    with st.spinner("🧠 Analysiere Gesichtsattribute..."):
        analysis = analyze_facial_attributes(image_path, face_location)
    
    if analysis is None:
        st.error("❌ Keine Analyse möglich")
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
                            st.image(thumbnail, caption="Analysiertes Gesicht")
    except:
        pass
    
    # Analysis results in a clean 2-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Analysis
        if 'age' in analysis:
            st.markdown("### 🎂 **Altersschätzung**")
            age = analysis['age']
            st.metric("Geschätztes Alter", f"{age} Jahre", help="Genauigkeit: ±4,65 Jahre")
            
            # Age category
            if age < 18:
                st.info("📚 Jugendlich")
            elif age < 30:
                st.info("🌟 Jung erwachsen")
            elif age < 50:
                st.info("💼 Erwachsen")
            elif age < 70:
                st.info("🎓 Reife Jahre")
            else:
                st.info("👴 Senior")
        
        st.markdown("---")
        
        # Gender Analysis
        if 'gender' in analysis:
            st.markdown("### ⚧️ **Geschlechtsbestimmung**")
            gender_data = analysis['gender']
            if isinstance(gender_data, dict):
                male_conf = gender_data.get('Man', 0)
                female_conf = gender_data.get('Woman', 0)
                
                if male_conf > female_conf:
                    st.metric("Geschlecht", "👨 Männlich", f"{male_conf:.1f}% Sicherheit")
                else:
                    st.metric("Geschlecht", "👩 Weiblich", f"{female_conf:.1f}% Sicherheit")
    
    with col2:
        # Emotion Analysis
        if 'emotion' in analysis:
            st.markdown("### 😊 **Emotionsanalyse**")
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
                st.metric("Hauptemotion", f"{emoji} {emotion_name.title()}", f"{emotion_conf:.1f}% Sicherheit")
        
        st.markdown("---")
        
        # Race/Ethnicity Analysis
        if 'race' in analysis:
            st.markdown("### 🌍 **Ethnische Herkunftsschätzung**")
            race_data = analysis['race']
            if isinstance(race_data, dict):
                # Find dominant ethnicity
                dominant_race = max(race_data.items(), key=lambda x: x[1])
                race_name, race_conf = dominant_race
                st.metric("Wahrscheinlichste Herkunft", race_name.title(), f"{race_conf:.1f}% Sicherheit")
    
    # Important disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ Wichtiger Hinweis:** Diese Analyse basiert auf KI-Modellen und dient ausschließlich zu **Demonstrations- und Forschungszwecken**. 
    Die Ergebnisse sind Schätzungen und sollten nicht für Identifikation, Diskriminierung oder Entscheidungsfindung verwendet werden.
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
                            st.image(thumbnail, caption="Gesicht für Namenszuweisung")
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
        st.info(f"✅ **Aktueller Name:** {existing_data['full_name']}")
    
    # Get all existing persons for selection
    all_persons = get_cached_all_persons()  # Use cached version
    
    # Choice between new person or existing person
    st.subheader("🎯 Person auswählen")
    
    assignment_type = st.radio(
        "Wie möchten Sie den Namen zuweisen?",
        ["🆕 Neue Person erstellen", "👥 Bestehende Person auswählen"],
        key="tab_assignment_type",
        horizontal=True
    )
    
    if assignment_type == "👥 Bestehende Person auswählen" and all_persons:
        # Existing person selection logic (simplified)
        person_options = {"-- Bitte Person auswählen --": None}
        for person in all_persons:
            display_name = f"{person['full_name']} ({person['face_count']} Gesichter)"
            person_options[display_name] = person
        
        selected_person_display = st.selectbox(
            "Person auswählen:",
            options=list(person_options.keys()),
            key="tab_selected_person"
        )
        
        if selected_person_display and person_options[selected_person_display] is not None:
            selected_person = person_options[selected_person_display]
            
            # Ensure selected_person is not None
            if selected_person is None:
                st.error("Fehler beim Laden der Personendaten")
                return
            
            # Show person details
            st.success(f"**Name:** {selected_person['full_name']}")
            st.write(f"**Anzahl Gesichter:** {selected_person['face_count']}")
            
            # Assign button
            if st.button("✅ Diese Person zuweisen", type="primary", width='stretch', key="tab_assign_existing"):
                person_data = {
                    'first_name': selected_person['first_name'],
                    'last_name': selected_person['last_name'],
                    'person_id': selected_person['person_id']
                }
                
                result = st.session_state.vector_store.assign_person_name_with_auto_matching(
                    face_id, person_data, selected_person['person_id'], similarity_threshold=0.8
                )
                
                if result['success']:
                    st.success(f"✅ Name '{selected_person['full_name']}' erfolgreich zugewiesen!")
                    
                    if result['total_assigned'] > 1:
                        st.success(f"🔄 {result['total_assigned']-1} ähnliche Gesichter automatisch zugewiesen!")
                    
                    st.balloons()
                else:
                    st.error("❌ Fehler beim Zuweisen des Namens")
    
    else:
        # New person creation form (simplified)
        st.subheader("📝 Neue Person erstellen")
        
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("Vorname: *", value=existing_data.get('first_name', ''), key="tab_first_name")
            last_name = st.text_input("Nachname: *", value=existing_data.get('last_name', ''), key="tab_last_name")
        
        with col2:
            birth_date = st.text_input("Geburtsdatum:", value=existing_data.get('birth_date', ''), key="tab_birth_date")
            birth_place = st.text_input("Geburtsort:", value=existing_data.get('birth_place', ''), key="tab_birth_place")
        
        notes = st.text_area("Notizen:", value=existing_data.get('notes', ''), key="tab_notes", height=60)
        
        # Preview full name
        if first_name or last_name:
            name_parts = [p.strip() for p in [(first_name or ''), (last_name or '')] if p and p.strip()]
            full_name = ' '.join(name_parts)
            st.success(f"**Vollständiger Name:** {full_name}")
        
        # Save button
        if st.button("💾 Person speichern", type="primary", width='stretch', key="tab_save_person"):
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
                
                result = st.session_state.vector_store.assign_person_name_with_auto_matching(
                    face_id, person_data, person_id, similarity_threshold=0.8
                )
                
                if result['success']:
                    st.success(f"✅ Person '{full_name}' erfolgreich erstellt und zugewiesen!")
                    
                    if result['total_assigned'] > 1:
                        st.success(f"🔄 {result['total_assigned']-1} ähnliche Gesichter automatisch zugewiesen!")
                    
                    st.balloons()
                else:
                    st.error("❌ Fehler beim Speichern der Person")
            else:
                st.error("❌ Bitte mindestens Vor- oder Nachname eingeben")

@st.dialog("🖼️ Vollbild-Ansicht", width="large")
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
        st.error(f"Fehlende Bilddaten für Vollansicht: {', '.join(missing_keys)}")
        st.info("Bitte versuchen Sie erneut, das Vollbild anzuzeigen.")
        if st.button("❌ Schließen"):
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
    st.markdown(f"**📁 Datei:** `{image_path.name}`")
    st.markdown(f"**📍 Pfad:** `{image_path}`")
    
    try:
        # Load the full image
        full_image = load_and_preprocess_image(image_path)
        if full_image is None:
            st.error("❌ Fehler beim Laden des Bildes")
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
                    st.success(f"✅ Gesicht markiert bei Koordinaten: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
            else:
                top, right, bottom, left = face_location
                cv2.rectangle(image_with_box, (left, top), (right, bottom), (0, 255, 0), 3)
                st.success(f"✅ Gesicht markiert bei Koordinaten: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
        
        st.markdown("---")
        
        # Display the image with maximum width
        caption = f"Vollbild - {image_path.name}"
        if face_location:
            caption += " (mit markiertem Gesicht)"
            
        st.image(image_with_box, caption=caption, width='stretch')
        
        # Image info
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Bildgröße", f"{full_image.shape[1]} × {full_image.shape[0]} px")
        
        with col2:
            file_size = image_path.stat().st_size if image_path.exists() else 0
            st.metric("Dateigröße", f"{file_size / 1024:.1f} KB")
        
        with col3:
            if face_location:
                st.metric("Gesicht erkannt", "✅ Ja")
            else:
                st.metric("Gesicht erkannt", "❌ Nein")
    
    except Exception as e:
        st.error(f"❌ Fehler beim Anzeigen des Vollbildes: {str(e)}")
    
    # Close button
    st.markdown("---")
    if st.button("❌ Vollbild schließen", key="close_full_image_modal", type="primary", width='stretch'):
        # Clear full image data
        for key in required_keys:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # Display the detailed analysis
    if all(key in st.session_state for key in ['analysis_image_path', 'analysis_face_location', 'analysis_face_id']):
        
        # Get analysis data
        with st.spinner("🧠 Analysiere Gesichtsattribute..."):
            analysis = analyze_facial_attributes(
                st.session_state.analysis_image_path,
                st.session_state.analysis_face_location
            )
        
        if analysis is None:
            st.error("❌ Keine Analyse möglich")
            return
        
        if "error" in analysis:
            st.error(f"❌ {analysis['error']}")
            return
        
        # Header info
        st.markdown(f"**Face ID:** `{st.session_state.analysis_face_id}`")
        st.markdown(f"**Datei:** `{Path(st.session_state.analysis_image_path).name}`")
        
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
                                st.image(face_crop, caption="Analysiertes Gesicht", width=250)
        except:
            pass
        
        st.markdown("---")
        
        # Analysis results in a clean 2-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Age Analysis
            if 'age' in analysis:
                st.markdown("## 🎂 **Alter**")
                age = analysis['age']
                st.metric("Geschätztes Alter", f"{age} Jahre", help="Genauigkeit: ±4,65 Jahre")
                
                # Age category with visual indicator
                if age < 18:
                    st.info("👶 **Kategorie:** Jugendlich")
                elif age < 30:
                    st.info("👤 **Kategorie:** Junge(r) Erwachsene(r)")
                elif age < 50:
                    st.info("👨 **Kategorie:** Erwachsene(r)")
                elif age < 70:
                    st.info("👴 **Kategorie:** Ältere(r) Erwachsene(r)")
                else:
                    st.info("👵 **Kategorie:** Senior(in)")
            
            st.markdown("---")
            
            # Gender Analysis
            if 'gender' in analysis:
                st.markdown("## ⚧️ **Geschlecht**")
                gender_data = analysis['gender']
                if isinstance(gender_data, dict):
                    male_prob = float(gender_data.get('Man', 0))
                    female_prob = float(gender_data.get('Woman', 0))
                    
                    # Visual representation with progress bars
                    st.write("**👨 Männlich**")
                    st.progress(male_prob / 100.0)
                    st.write(f"`{male_prob:.1f}%`")
                    
                    st.write("**👩 Weiblich**") 
                    st.progress(female_prob / 100.0)
                    st.write(f"`{female_prob:.1f}%`")
                    
                    # Highlight prediction
                    dominant = "👨 Männlich" if male_prob > female_prob else "👩 Weiblich"
                    confidence = max(male_prob, female_prob)
                    st.success(f"**Vorhersage:** {dominant} (Konfidenz: {confidence:.1f}%)")
                else:
                    st.write(f"**Geschlecht:** {gender_data}")
        
        with col2:
            # Emotion Analysis
            if 'emotion' in analysis:
                st.markdown("## 😊 **Emotionen**")
                emotions = analysis['emotion']
                if isinstance(emotions, dict):
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                    emotion_map = {
                        'happy': ('😊 Glücklich', '#28a745'),
                        'neutral': ('😐 Neutral', '#6c757d'),
                        'sad': ('😢 Traurig', '#007bff'),
                        'angry': ('😠 Wütend', '#dc3545'),
                        'surprise': ('😲 Überrascht', '#ffc107'),
                        'fear': ('😨 Ängstlich', '#fd7e14'),
                        'disgust': ('🤢 Angeekelt', '#6f42c1')
                    }
                    
                    # Show top 5 emotions with visual bars
                    for emotion, confidence in sorted_emotions[:5]:
                        german_emotion, color = emotion_map.get(emotion, (emotion.title(), '#6c757d'))
                        st.write(f"**{german_emotion}**")
                        st.progress(float(confidence / 100.0))
                        st.write(f"`{confidence:.1f}%`")
                    
                    # Dominant emotion highlight
                    top_emotion, top_confidence = sorted_emotions[0]
                    german_top, _ = emotion_map.get(top_emotion, (top_emotion.title(), '#6c757d'))
                    st.success(f"**Dominante Emotion:** {german_top} ({top_confidence:.1f}%)")
            
            st.markdown("---")
            
            # Race/Ethnicity Analysis
            if 'race' in analysis:
                st.markdown("## 🌍 **Ethnische Herkunft**")
                race_data = analysis['race']
                if isinstance(race_data, dict):
                    sorted_races = sorted(race_data.items(), key=lambda x: x[1], reverse=True)
                    race_map = {
                        'asian': '🌏 Asiatisch',
                        'white': '🌍 Kaukasisch', 
                        'middle eastern': '🌍 Nahöstlich',
                        'indian': '🌏 Indisch',
                        'latino hispanic': '🌎 Lateinamerikanisch',
                        'black': '🌍 Afrikanisch'
                    }
                    
                    # Show top predictions with progress bars
                    for race, confidence in sorted_races:
                        german_race = race_map.get(race.lower(), race.title())
                        st.write(f"**{german_race}**")
                        st.progress(float(confidence / 100.0))
                        st.write(f"`{confidence:.1f}%`")
                    
                    # Dominant prediction
                    top_race, top_confidence = sorted_races[0]
                    german_top_race = race_map.get(top_race.lower(), top_race.title())
                    st.info(f"**Hauptvorhersage:** {german_top_race} ({top_confidence:.1f}%)")
        
        # Footer with important disclaimer
        st.markdown("---")
        st.warning("""
        **⚠️ Wichtiger Hinweis:** Diese Analyse basiert auf KI-Modellen und dient ausschließlich zu **Demonstrations- und Forschungszwecken**. 
        Die Ergebnisse sind Schätzungen und sollten nicht für Identifikation, Diskriminierung oder Entscheidungsfindung verwendet werden.
        """)
        
        # Close button
        if st.button("✅ Analyse schließen", key="close_popup_analysis", type="primary", width='stretch'):
            # Clear analysis data
            if 'analysis_image_path' in st.session_state:
                del st.session_state.analysis_image_path
            if 'analysis_face_location' in st.session_state:
                del st.session_state.analysis_face_location  
            if 'analysis_face_id' in st.session_state:
                del st.session_state.analysis_face_id
            st.rerun()
    
    else:
        st.error("❌ Analysedaten nicht verfügbar")
        if st.button("❌ Schließen", key="close_error_modal"):
            st.rerun()

def display_search_results(results: List[Dict[str, Any]]):
    """Display enhanced search results with filtering options"""
    
    st.subheader(f"🎯 Enhanced Search Results ({len(results)} faces)")
    
    if not results:
        st.info("No results to display")
        return
    
    # Real-time filtering options
    with st.expander("🔧 Filter Results", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            min_similarity_filter = st.slider(
                "Min Similarity %",
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
            quality_level = st.selectbox(
                "Quality Level",
                ["All", "Medium+", "High Only"],
                index=0,
                help="Filter by estimated quality"
            )
            
            show_named_only = st.checkbox(
                "Named Faces Only",
                value=False,
                help="Show only faces with assigned names"
            )
        
        with col_f3:
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
        
        # Quality filter
        if quality_level == "High Only" and result['similarity'] < 0.7:
            continue
        elif quality_level == "Medium+" and result['similarity'] < 0.4:
            continue
        
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
    elif sort_order == "Size ↓":
        def get_size(x):
            try:
                coords = x.get('metadata', {}).get('location', '0,0,0,0').split(',')
                if len(coords) == 4:
                    top, right, bottom, left = map(int, coords)
                    return (right - left) * (bottom - top)
                return 0
            except:
                return 0
        filtered_results = sorted(filtered_results, key=get_size, reverse=True)
    elif sort_order == "Name A-Z":
        filtered_results = sorted(filtered_results, key=lambda x: x.get('metadata', {}).get('full_name', 'zzz'))
    
    # Show filter results summary
    if len(filtered_results) != len(results):
        st.info(f"📋 Filtered: {len(filtered_results)} of {len(results)} results shown (filtered out {len(results) - len(filtered_results)})")
    
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
        total_pages = (len(results) - 1) // RESULTS_PER_PAGE + 1
        
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("⬅️ Previous") and st.session_state.current_page > 1:
                    st.session_state.current_page -= 1
                    st.rerun()
            
            with col2:
                st.write(f"Page {st.session_state.current_page} of {total_pages}")
            
            with col3:
                if st.button("➡️ Next") and st.session_state.current_page < total_pages:
                    st.session_state.current_page += 1
                    st.rerun()
        
        # Calculate page slice
        start_idx = (st.session_state.current_page - 1) * RESULTS_PER_PAGE
        end_idx = start_idx + RESULTS_PER_PAGE
        page_results = results[start_idx:end_idx]
    
    # Display results in grid
    cols = st.columns(4)
    
    for i, face_data in enumerate(page_results):
        with cols[i % 4]:
            try:
                # Check if this is an external result
                is_external = face_data.get('source') == 'external'
                source_db = face_data.get('source_db', '')
                
                if is_external:
                    # Handle external image (URL)
                    image_url = face_data['image_path']  # This is actually a URL for external results
                    ext_metadata = face_data.get('metadata', {})
                    thumbnail_url = ext_metadata.get('thumbnail_url')
                    
                    # Display external image placeholder or thumbnail
                    if thumbnail_url and thumbnail_url.startswith('http'):
                        try:
                            st.image(thumbnail_url, width='stretch')
                        except:
                            # Fallback to placeholder
                            st.markdown("🖼️ **External Image**")
                            st.info("Preview not available")
                    else:
                        # Show placeholder for external result
                        st.markdown("🌐 **External Face Match**")
                        st.info("Face analysis result")
                    
                    # External source indicator
                    api_provider = ext_metadata.get('api_provider', 'External API')
                    st.markdown(f"🔗 **{api_provider}**")
                    
                    # External metadata display
                    if 'name' in ext_metadata:
                        st.markdown(f"👤 **{ext_metadata['name']}**")
                    
                    if 'category' in ext_metadata:
                        st.markdown(f"📂 {ext_metadata['category']}")
                    
                    if 'database_type' in ext_metadata:
                        db_type = ext_metadata['database_type'].replace('_', ' ').title()
                        st.markdown(f"🗃️ {db_type} Database")
                    
                    # Show detected attributes
                    if 'detected_age' in ext_metadata:
                        st.markdown(f"� Age: {ext_metadata['detected_age']}")
                    
                    if 'detected_gender' in ext_metadata:
                        st.markdown(f"⚥ Gender: {ext_metadata['detected_gender']}")
                    
                    # Show emotion if available
                    if 'emotion_analysis' in ext_metadata:
                        emotion = ext_metadata['emotion_analysis']
                        if emotion:
                            top_emotion = max(emotion.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                            st.markdown(f"� Emotion: {top_emotion[0]} ({top_emotion[1]:.0f}%)")
                    
                    # Similarity and confidence
                    similarity_percentage = face_data['similarity'] * 100
                    confidence = ext_metadata.get('confidence', face_data['similarity']) * 100
                    
                    # Color-coded similarity display for external results
                    if similarity_percentage >= 70:
                        st.success(f"**🎯 {similarity_percentage:.1f}%**")
                    elif similarity_percentage >= 50:
                        st.info(f"**✅ {similarity_percentage:.1f}%**")
                    elif similarity_percentage >= 30:
                        st.warning(f"**⚠️ {similarity_percentage:.1f}%**")
                    else:
                        st.error(f"**❓ {similarity_percentage:.1f}%**")
                    
                    st.caption(f"🔍 Confidence: {confidence:.1f}%")
                    
                    # Show similarity factors if available
                    if 'similarity_factors' in ext_metadata:
                        factors = ext_metadata['similarity_factors']
                        if factors:
                            with st.expander("📊 Match Details", expanded=False):
                                for factor in factors:
                                    st.write(f"• {factor}")
                    
                    # External link if available
                    if ext_metadata.get('image_url') and ext_metadata['image_url'].startswith('http'):
                        st.markdown(f"[🔗 View Source]({ext_metadata['image_url']})")
                
                else:
                    # Handle local image (existing code)
                    image_path = Path(face_data['metadata']['image_path'])
                    if image_path.exists():
                        image = load_and_preprocess_image(image_path)
                        
                        if image is not None:
                            # Extract face region
                            location = face_data['metadata'].get('location')
                            if location:
                                # Parse location string "top,right,bottom,left"
                                if isinstance(location, str):
                                    coords = location.split(',')
                                    if len(coords) == 4:
                                        top, right, bottom, left = map(int, coords)
                                        face_image = image[top:bottom, left:right]
                                        
                                        # Create thumbnail
                                        thumbnail = create_thumbnail(face_image, THUMBNAIL_SIZE)
                                        
                                        # Display with enhanced similarity information
                                        st.image(thumbnail, width='stretch')
                                        
                                        # Local source indicator
                                        st.markdown("💾 **Local Database**")
                                        
                                        # Enhanced similarity display with quality indicators
                                        similarity_percentage = face_data['similarity'] * 100
                                        confidence_score = face_data.get('confidence_score', similarity_percentage / 100) * 100
                                        
                                        # Calculate face quality metrics
                                        face_width = right - left
                                        face_height = bottom - top
                                        face_size = min(face_width, face_height)
                                        face_area = face_width * face_height
                                    
                                    # Quality indicators
                                    quality_indicators = []
                                    if face_size >= 80:
                                        quality_indicators.append("🔍 High Res")
                                    elif face_size >= 50:
                                        quality_indicators.append("📏 Medium Res")
                                    else:
                                        quality_indicators.append("📱 Low Res")
                                    
                                    if face_area > 6400:  # 80x80
                                        quality_indicators.append("✨ Large")
                                    
                                    # Color-coded similarity display with quality info
                                    if similarity_percentage >= 70:
                                        st.success(f"**🎯 {similarity_percentage:.1f}%**")
                                        st.caption(f"🔒 Confidence: {confidence_score:.1f}% | Size: {face_size}px")
                                    elif similarity_percentage >= 50:
                                        st.info(f"**✅ {similarity_percentage:.1f}%**")
                                        st.caption(f"📊 Confidence: {confidence_score:.1f}% | Size: {face_size}px")
                                    elif similarity_percentage >= 30:
                                        st.warning(f"**⚠️ {similarity_percentage:.1f}%**")
                                        st.caption(f"📈 Confidence: {confidence_score:.1f}% | Size: {face_size}px")
                                    else:
                                        st.error(f"**❓ {similarity_percentage:.1f}%**")
                                        st.caption(f"🔍 Confidence: {confidence_score:.1f}% | Size: {face_size}px")
                                    
                                    # Display quality indicators
                                    if quality_indicators:
                                        st.caption(" | ".join(quality_indicators))
                                    
                                    st.write(f"**📁 {image_path.name}**")
                                    
                                    # Display person name if assigned
                                    face_metadata = face_data.get('metadata', {})
                                    person_name = face_metadata.get('full_name', '')
                                    if person_name:
                                        st.success(f"👤 **{person_name}**")
                                    
                                    # Single unified info button
                                    if st.button("ℹ️ Infos", key=f"info_{i}", help="Vollbild, Gesichtsanalyse und Namen zuweisen", width='stretch'):
                                        face_id = face_data.get('face_id', face_data.get('id', f"face_{i}"))  # Try both face_id and id
                                        # Set face info data and trigger modal
                                        st.session_state.info_face_id = face_id
                                        st.session_state.info_image_path = image_path
                                        st.session_state.info_face_location = location
                                        st.rerun()
                                else:
                                    # Fallback: show full image
                                    st.image(image, width='stretch')
                                    st.write(f"**Similarity:** {face_data['similarity']:.3f}")
                            else:
                                # Handle tuple format
                                try:
                                    top, right, bottom, left = location
                                    face_image = image[top:bottom, left:right]
                                    
                                    # Create thumbnail
                                    thumbnail = create_thumbnail(face_image, THUMBNAIL_SIZE)
                                    
                                    # Display face thumbnail
                                    st.image(thumbnail, width='stretch')
                                    
                                    # Enhanced similarity display
                                    similarity_percentage = face_data['similarity'] * 100
                                    confidence_score = face_data.get('confidence_score', similarity_percentage / 100) * 100
                                    
                                    if similarity_percentage >= 70:
                                        st.success(f"**🎯 {similarity_percentage:.1f}%**")
                                    elif similarity_percentage >= 50:
                                        st.info(f"**✅ {similarity_percentage:.1f}%**")
                                    elif similarity_percentage >= 30:
                                        st.warning(f"**⚠️ {similarity_percentage:.1f}%**")
                                    else:
                                        st.error(f"**❓ {similarity_percentage:.1f}%**")
                                    
                                    st.write(f"**📁 {image_path.name}**")
                                    st.caption(f"🔒 Vertrauen: {confidence_score:.1f}%")
                                    
                                    # Display person name if assigned
                                    face_metadata = face_data.get('metadata', {})
                                    person_name = face_metadata.get('full_name', '')
                                    if person_name:
                                        st.success(f"👤 **{person_name}**")
                                    
                                    # Single unified info button
                                    if st.button("ℹ️ Infos", key=f"info_tuple_{i}", help="Vollbild, Gesichtsanalyse und Namen zuweisen", width='stretch'):
                                        face_id = face_data.get('face_id', face_data.get('id', f"face_{i}"))  # Try both face_id and id
                                        # Set face info data and trigger modal
                                        st.session_state.info_face_id = face_id
                                        st.session_state.info_image_path = image_path
                                        st.session_state.info_face_location = (top, right, bottom, left)
                                        st.rerun()
                                        
                                except:
                                    # Fallback: show full image
                                    st.image(image, width='stretch')
                                    st.write(f"**Similarity:** {face_data['similarity']:.3f}")
                                    
                                    # Add full image view button  
                                    if st.button("🖼️ Ganzes Bild", key=f"full_img_fallback_{i}", help="Ganzes Bild anzeigen"):
                                        show_full_image_with_face_box(image_path, None)
                        else:
                            st.image(image, width='stretch')
                            
                            similarity_percentage = face_data['similarity'] * 100
                            confidence_score = face_data.get('confidence_score', similarity_percentage / 100) * 100
                            
                            if similarity_percentage >= 70:
                                st.success(f"**🎯 {similarity_percentage:.1f}%**")
                            elif similarity_percentage >= 50:
                                st.info(f"**✅ {similarity_percentage:.1f}%**")
                            elif similarity_percentage >= 30:
                                st.warning(f"**⚠️ {similarity_percentage:.1f}%**")
                            else:
                                st.error(f"**❓ {similarity_percentage:.1f}%**")
                            
                            st.write(f"**📁 {image_path.name}**")
                            st.caption(f"🔒 Vertrauen: {confidence_score:.1f}%")
                            
                            # Add full image view button
                            if st.button("🖼️ Ganzes Bild", key=f"full_img_no_loc_{i}", help="Ganzes Bild anzeigen"):
                                show_full_image_with_face_box(image_path, None)
                    else:
                        st.error("Image not found")
                        
            except Exception as e:
                st.error(f"Error displaying result: {e}")

def batch_face_processing_page():
    """Enhanced batch processing page using fast_process.py for optimal performance."""
    
    st.header("🚀 Fast Batch Face Processing")
    st.markdown("**High-performance face detection and processing using optimized algorithms**")
    
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
        available_dirs = []
        
        # Check for downloaded images directories
        images_base = Path("data/images")
        if images_base.exists():
            downloaded_dirs = [d for d in images_base.iterdir() 
                             if d.is_dir() and d.name.startswith("downloaded_images")]
            
            for d in sorted(downloaded_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
                image_count = len(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg")))
                if image_count > 0:
                    time_str = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    available_dirs.append((f"📥 {d.name} ({image_count} images, {time_str})", str(d)))
        
        # Check scraped directory
        scraped_dir = Path("data/scraped")
        if scraped_dir.exists():
            scraped_count = len(list(scraped_dir.glob("*.jpg")) + list(scraped_dir.glob("*.png")) + list(scraped_dir.glob("*.jpeg")))
            if scraped_count > 0:
                available_dirs.append((f"🕷️ Scraped Images ({scraped_count} images)", str(scraped_dir)))
        
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
    
    # Temporarily update the processing directory in fast_process
    original_dir = None
    try:
        # Import the processing function
        sys.path.append(os.path.dirname(__file__))
        from fast_process import process_images_streamlit, IMAGES_DIR
        
        # Temporarily change the images directory
        import fast_process
        original_dir = fast_process.IMAGES_DIR
        fast_process.IMAGES_DIR = Path(directory_path)
        
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
            # Ensure percent is a regular Python int/float, not numpy type
            progress_bar.progress(float(percent))
            progress_text.text(f"Processing: {percent}%")
        
        def update_status(message):
            status_text.info(message)
        
        # Start processing
        start_time = time.time()
        
        with st.spinner("🧠 Initializing fast face processing engine..."):
            result = process_images_streamlit(
                update_existing=update_existing,
                progress_callback=update_progress,
                status_callback=update_status
            )
        
        # Show results
        with results_container:
            processing_time = time.time() - start_time
            
            if result["success"]:
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
                
                # Suggest next steps
                st.subheader("🎯 Next Steps")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔍 View Face Gallery", width='stretch'):
                        st.session_state.page = "Face Gallery"
                        st.rerun()
                
                with col2:
                    if st.button("📊 Analyze Results", width='stretch'):
                        st.session_state.page = "Face Search"
                        st.rerun()
            
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
    
    finally:
        # Restore original directory
        if original_dir and 'fast_process' in locals():
            fast_process.IMAGES_DIR = original_dir


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
                                    st.success(f"✅ {image_name} → {face_count} Gesicht erkannt")
                                else:
                                    st.success(f"✅ {image_name} → {face_count} Gesichter erkannt")
                        
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
                                st.warning(f"⚠️ {image_name} → Keine Gesichter erkannt")
                
                # Add to vector store
                if face_ids:
                    added_count = st.session_state.vector_store.add_face_embeddings_batch(
                        face_ids, embeddings, metadatas
                    )
                    total_faces_found += added_count
                    
                    if detailed_progress and detailed_container:
                        with detailed_container:
                            st.info(f"💾 Batch {batch_num + 1}: {added_count} Gesichter zur Datenbank hinzugefügt")
                            st.markdown("---")  # Separator between batches
                else:
                    if detailed_progress and detailed_container:
                        with detailed_container:
                            st.warning(f"⚪ Batch {batch_num + 1}: Keine Gesichter für Datenbank gefunden")
                            st.markdown("---")  # Separator between batches
                
                # Enhanced batch summary with detailed statistics
                batch_success_count = len(batch_files) - batch_errors
                if batch_faces > 0:
                    batch_info.success(f"✅ Batch {batch_num + 1}: {batch_faces} Gesichter gefunden ({batch_success_count}/{len(batch_files)} Bilder verarbeitet)")
                elif batch_errors > 0:
                    batch_info.error(f"❌ Batch {batch_num + 1}: {batch_errors} Fehler, {batch_no_faces} Bilder ohne Gesichter")
                else:
                    batch_info.info(f"⚪ Batch {batch_num + 1}: {batch_no_faces} Bilder ohne Gesichter erkannt")
            
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
                                st.success(f"✅ {image_name} → {face_count} Gesicht erkannt")
                            else:
                                st.success(f"✅ {image_name} → {face_count} Gesichter erkannt")
                        
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
                            st.warning(f"⚠️ {image_name} → Keine Gesichter erkannt")
                
                # Add to database
                if face_ids:
                    added_count = st.session_state.vector_store.add_face_embeddings_batch(
                        face_ids, embeddings, metadatas
                    )
                    total_faces_found += added_count
                    
                    # Log database addition
                    with detailed_log_container:
                        st.info(f"💾 {dir_name} Batch {batch_num + 1}: {added_count} Gesichter zur Datenbank hinzugefügt")
                        st.markdown("---")  # Separator between batches
                else:
                    with detailed_log_container:
                        st.warning(f"⚪ {dir_name} Batch {batch_num + 1}: Keine Gesichter für Datenbank gefunden")
                        st.markdown("---")  # Separator between batches
                
                global_batch_count += 1
                
                # Enhanced batch summary in results container
                with results_container:
                    batch_success_count = len(batch_files) - batch_errors
                    if batch_faces > 0:
                        st.success(f"✅ {dir_name} Batch {batch_num + 1}: {batch_faces} Gesichter gefunden ({batch_success_count}/{len(batch_files)} Bilder)")
                    elif batch_errors > 0:
                        st.error(f"❌ {dir_name} Batch {batch_num + 1}: {batch_errors} Fehler, {batch_no_faces} ohne Gesichter")
                    else:
                        st.info(f"⚪ {dir_name} Batch {batch_num + 1}: {batch_no_faces} Bilder ohne Gesichter")
        
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
            # Save uploaded files
            temp_paths = []
            for i, uploaded_file in enumerate(uploaded_files):
                temp_path = Path(f"/tmp/uploaded_{i}_{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                temp_paths.append(temp_path)
            
            # Copy to images directory
            saved_paths = []
            for temp_path, uploaded_file in zip(temp_paths, uploaded_files):
                final_path = IMAGES_DIR / uploaded_file.name
                
                # Handle duplicate names
                counter = 1
                while final_path.exists():
                    stem = Path(uploaded_file.name).stem
                    suffix = Path(uploaded_file.name).suffix
                    final_path = IMAGES_DIR / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                # Copy file
                final_path.write_bytes(temp_path.read_bytes())
                saved_paths.append(final_path)
                temp_path.unlink()  # Clean up temp file
            
            # Process images
            process_existing_images(saved_paths)
            
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
                failed_images += 1
        
        progress_bar.progress(75)
        status_text.text("Adding to vector database...")
        
        # Add to vector store
        added_count = 0
        if face_ids:
            added_count = st.session_state.vector_store.add_face_embeddings_batch(
                face_ids, embeddings, metadatas
            )
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        
        # Provide detailed summary
        if added_count > 0:
            st.success(f"🎉 Successfully processed {len(image_paths)} images and added {added_count} face embeddings to the database!")
        elif total_faces > 0:
            st.warning(f"⚠️ Found {total_faces} faces but failed to add them to database. Please check the database connection.")
        elif successful_images > 0:
            st.info(f"✅ Successfully processed {successful_images} images, but no faces were detected in any of them.")
        else:
            st.warning("❌ No images could be processed successfully. Please check the image files.")
        
        # Show detailed processing summary
        st.info(f"**Processing Summary:**\n"
               f"- Total images: {len(image_paths)}\n"
               f"- Successfully processed: {successful_images}\n"
               f"- Failed to process: {failed_images}\n"
               f"- Total faces detected: {total_faces}\n"
               f"- Face embeddings added to database: {added_count}")
            
    except Exception as e:
        st.error(f"Error processing images: {e}")
        import traceback
        st.code(traceback.format_exc())

def web_scraping_page():
    """Web scraping page"""
    
    st.header("🌐 Web Image Scraping")
    st.warning("⚠️ Please ensure you have permission to scrape images from websites and comply with their terms of service.")
    
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
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            unlimited = st.checkbox(
                "📥 Unlimited Download", 
                value=True,
                help="Download alle verfügbaren Bilder (empfohlen)"
            )
            
        with col2:
            if not unlimited:
                max_images = st.slider(
                    "Max images per site", 
                    min_value=10, 
                    max_value=1000, 
                    value=100
                )
            else:
                max_images = 0  # 0 = unlimited
                st.info("🚀 Alle Bilder werden heruntergeladen")
        
        with col3:
            auto_process = st.checkbox(
                "Auto-process scraped images", 
                value=True,
                help="Automatically process scraped images for face detection"
            )
        
        # Duplikat-Info anzeigen
        st.info("🔍 **Duplikat-Erkennung aktiviert:** Bereits vorhandene Bilder werden automatisch übersprungen")
        
        if st.button("🕷️ Start Scraping", type="primary"):
            if urls_input.strip():
                urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
                scrape_websites(urls, max_images, auto_process)
            else:
                st.error("Please enter at least one URL")
    
    with tab2:
        display_scraped_images()

def scrape_websites(urls: List[str], max_images: int, auto_process: bool):
    """Scrape images from websites"""
    
    with st.spinner("Scraping images from websites..."):
        try:
            # Run async scraping
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            results = loop.run_until_complete(
                st.session_state.image_scraper.scrape_multiple_websites(urls, max_images)
            )
            
            loop.close()
            
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
                            image = load_and_preprocess_image(image_path)
                            
                            if image is not None:
                                thumbnail = create_thumbnail(image, THUMBNAIL_SIZE)
                                st.image(thumbnail, width='stretch')
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

def duplicate_manager_page():
    """Duplicate detection and management page"""
    
    st.header("🗂️ Duplicate Manager")
    
    st.markdown("**Intelligente Duplikat-Erkennung:** Erkennt identische Bilder auch wenn sie umbenannt wurden")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Find Duplicates", "📊 Database Stats", "🧹 Cleanup"])
    
    with tab1:
        st.subheader("Duplikate in Verzeichnissen finden")
        
        directories = [IMAGES_DIR, SCRAPED_DIR]
        
        for directory in directories:
            if directory.exists():
                st.write(f"**{directory.name}**: {len(get_image_files(directory))} images")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔍 Find Duplicates in {directory.name}", key=f"find_dups_{directory.name}"):
                        find_duplicates_in_directory(directory)
                
                with col2:
                    content_hash = st.checkbox(f"Content-based detection", 
                                             key=f"content_{directory.name}",
                                             value=True,
                                             help="Erkennt identische Bilder auch wenn sie umbenannt wurden")
    
    with tab2:
        st.subheader("Hash-Datenbank Statistiken")
        
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
                            image = load_and_preprocess_image(file_path)
                            if image is not None:
                                thumbnail = create_thumbnail(image, (120, 120))
                                st.image(thumbnail, width='stretch')
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Stats"):
                st.rerun()
        
        with col2:
            if st.button("⚠️ Clear Database", type="secondary"):
                if st.session_state.get("confirm_clear", False):
                    st.session_state.vector_store.clear_collection()
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
    
    st.header("⚙️ Settings")
    
    st.subheader("Face Detection Model Settings")
    
    # Model selection
    detection_model = st.selectbox(
        "Face Detection Model:",
        options=["hog", "cnn"],
        index=1 if FACE_RECOGNITION_MODEL == "cnn" else 0,
        help="HOG: Schneller, weniger genau | CNN: Langsamer, aber genauer"
    )
    
    if detection_model != FACE_RECOGNITION_MODEL:
        st.info(f"Model wird geändert von {FACE_RECOGNITION_MODEL} zu {detection_model}")
        st.warning("⚠️ Diese Änderung erfordert einen Neustart der App")
    
    # Model comparison table
    st.subheader("Model Vergleich")
    comparison_data = {
        "Model": ["HOG", "CNN"],
        "Geschwindigkeit": ["⚡ Sehr schnell", "🐌 Langsamer"],
        "Genauigkeit": ["👍 Gut", "⭐ Sehr gut"],
        "Ressourcen": ["💚 Niedrig", "⚠️ Hoch"],
        "Empfohlen für": ["Echtzeit, viele Bilder", "Höchste Qualität"]
    }
    
    st.table(comparison_data)
    
    st.subheader("Aktuelle Einstellungen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Embedding Model:** DeepFace Facenet512")
        st.write(f"**Detection Model:** {FACE_RECOGNITION_MODEL.upper()} {'✅ (Genauer)' if FACE_RECOGNITION_MODEL == 'cnn' else '⚡ (Schneller)'}")
        st.write("**Vector Database:** ChromaDB")
    
    with col2:
        st.write("**Similarity Threshold:** 0.7")
        st.write("**Max Results:** 50")
        st.write("**Batch Size:** 32")
    
    st.markdown("---")
    
    st.subheader("Storage Information")
    
    # Directory info
    directories = [
        ("Images Directory", IMAGES_DIR),
        ("Scraped Directory", SCRAPED_DIR),
    ]
    
    for name, path in directories:
        if path.exists():
            file_count = len(get_image_files(path))
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
    
    st.header("👥 Face Gallery")
    st.markdown("Durchsuchen Sie alle Gesichter in der Datenbank")
    
    # Get all faces from database
    try:
        stats = st.session_state.vector_store.get_collection_stats()
        total_faces = stats.get("total_faces", 0)
        unique_images = stats.get("unique_images", 0)
        
        if total_faces == 0:
            st.info("Noch keine Gesichter in der Datenbank. Laden Sie Bilder hoch oder scrapen Sie Websites, um Gesichter hinzuzufügen.")
            return
            
        # Enhanced stats display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Total Faces", total_faces)
        with col2:
            st.metric("📷 Unique Images", unique_images)
        with col3:
            avg_faces = total_faces / unique_images if unique_images > 0 else 0
            st.metric("📊 Avg Faces/Image", f"{avg_faces:.1f}")
        
        st.markdown("---")
        
        # Gallery options
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            faces_per_page = st.selectbox("Gesichter pro Seite:", [20, 50, 100, 200, 500], index=2)
        
        with col2:
            show_metadata = st.checkbox("Metadaten anzeigen", value=True)
        
        with col3:
            group_by_image = st.checkbox("Nach Bild gruppieren", value=False)
            
        with col4:
            sort_by = st.selectbox("Sortieren nach:", ["Neueste", "Bild-Name", "Face-ID"])
        
        # Search and filter options
        st.markdown("---")
        
        # Show deletion statistics if available
        if 'deleted_faces_session' in st.session_state and st.session_state.deleted_faces_session > 0:
            st.info(f"📊 In dieser Sitzung wurden {st.session_state.deleted_faces_session} falsch erkannte Gesichter entfernt.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_term = st.text_input("🔍 Suche nach Dateiname oder Face-ID:", 
                                      placeholder="z.B. 'image_001' oder 'abc123'")
        
        with col2:
            st.markdown("**Filter-Optionen:**")
            min_face_size = st.slider("Min. Gesichtsgröße (px):", 30, 200, 50)
        
        # Apply filters to results if search term is provided
        filtered_total = total_faces
        search_active = bool(search_term.strip())
        
        if search_active:
            with st.spinner("Durchsuche Gesichter..."):
                # This would need to be implemented in vector_store
                st.info(f"🔍 Suche nach: '{search_term}' - Erweiterte Suchfunktion wird implementiert...")
        
        # Pagination controls
        st.markdown("---")
        
        if 'gallery_page' not in st.session_state:
            st.session_state.gallery_page = 0
        
        current_page = st.session_state.gallery_page
        total_pages = max(1, (filtered_total - 1) // faces_per_page + 1)
        
        # Page info and navigation in one row
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])
        
        with nav_col1:
            if st.button("⬅️ Vorherige", disabled=(current_page == 0)):
                st.session_state.gallery_page = max(0, current_page - 1)
                st.rerun()
        
        with nav_col2:
            page_input = st.number_input(
                "Gehe zu Seite:", 
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
            st.write(f"**Seite {current_page + 1} von {total_pages}**")
            remaining_faces = filtered_total - (current_page * faces_per_page)
            faces_on_page = min(faces_per_page, remaining_faces)
            st.write(f"Zeige {faces_on_page} von {filtered_total} Gesichtern")
        
        with nav_col4:
            if st.button("Nächste ➡️", disabled=(current_page >= total_pages - 1)):
                st.session_state.gallery_page = min(total_pages - 1, current_page + 1)
                st.rerun()
        
        with nav_col5:
            if st.button("🔄 Aktualisieren"):
                st.cache_data.clear()
                st.rerun()
        
        st.markdown("---")
        
        # Calculate offset for pagination
        offset = current_page * faces_per_page
        remaining_faces = filtered_total - offset
        limit = min(faces_per_page, remaining_faces)
        
        if limit <= 0:
            st.warning("Keine Gesichter auf dieser Seite. Gehen Sie zu einer früheren Seite.")
            return
        
        with st.spinner(f"Lade {limit} Gesichter..."):
            # Get face data from database with improved robust pagination
            try:
                results = st.session_state.vector_store.get_faces_paginated(offset=offset, limit=limit)
                
                if results['metadatas'] and len(results['metadatas']) > 0:
                    st.subheader(f"📋 Zeige {len(results['metadatas'])} Gesichter (Seite {current_page + 1})")
                    
                    # Debug information for troubleshooting
                    if st.checkbox("🔧 Debug-Info anzeigen", value=False):
                        st.info(f"""
                        **Debug-Informationen:**
                        - Aktuelle Seite: {current_page + 1}
                        - Offset: {offset}
                        - Limit: {limit} 
                        - Gefundene Gesichter: {len(results['metadatas'])}
                        - Total Gesichter in DB: {filtered_total}
                        - Hat Embeddings: {len(results.get('embeddings', [])) > 0}
                        - Hat IDs: {len(results.get('ids', [[]])[0]) > 0 if results.get('ids') else False}
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
                        if st.button("📤 Alle Gesichter exportieren"):
                            export_all_faces()
                    
                    with col2:
                        # Batch delete current page faces
                        if st.button("🗑️ Alle auf Seite löschen", help="Alle Gesichter auf der aktuellen Seite löschen"):
                            if st.session_state.get('confirm_batch_delete', False):
                                # Perform batch deletion
                                deleted_count = 0
                                face_ids = [metadata.get('face_id', '') for metadata in results['metadatas'] if metadata.get('face_id')]
                                
                                with st.spinner(f"Lösche {len(face_ids)} Gesichter..."):
                                    for face_id in face_ids:
                                        if delete_face_from_gallery(face_id):
                                            deleted_count += 1
                                
                                st.success(f"✅ {deleted_count} von {len(face_ids)} Gesichtern gelöscht!")
                                st.session_state.confirm_batch_delete = False
                                
                                # Go back to page 1 to avoid pagination errors
                                st.session_state.gallery_page = 0
                                st.rerun()
                            else:
                                st.session_state.confirm_batch_delete = True
                                st.warning("⚠️ Nochmals klicken zum Bestätigen der Löschung ALLER Gesichter auf dieser Seite!")
                                st.rerun()
                    
                    with col3:
                        # Delete by similarity threshold
                        if st.button("🧹 Niedrige Qualität löschen", help="Gesichter mit niedriger erkannter Qualität löschen"):
                            # Analyze current page faces for quality issues
                            low_quality_faces = identify_low_quality_faces(results['metadatas'])
                            
                            if low_quality_faces:
                                st.warning(f"⚠️ {len(low_quality_faces)} potentiell falsch erkannte Gesichter gefunden:")
                                
                                # Show preview of faces to be deleted
                                with st.expander("� Gesichter mit niedriger Qualität anzeigen", expanded=False):
                                    for i, face_info in enumerate(low_quality_faces[:5]):  # Show first 5
                                        st.write(f"• **{face_info['face_id'][:12]}...**: {face_info['reason']}")
                                    
                                    if len(low_quality_faces) > 5:
                                        st.write(f"... und {len(low_quality_faces) - 5} weitere")
                                
                                if st.button("🗑️ Niedrigqualitative Gesichter löschen", key="confirm_quality_delete"):
                                    deleted_count = 0
                                    
                                    with st.spinner(f"Lösche {len(low_quality_faces)} niedrigqualitative Gesichter..."):
                                        for face_info in low_quality_faces:
                                            if delete_face_from_gallery(face_info['face_id']):
                                                deleted_count += 1
                                    
                                    st.success(f"✅ {deleted_count} niedrigqualitative Gesichter erfolgreich gelöscht!")
                                    st.rerun()
                            else:
                                st.info("✨ Keine niedrigqualitativen Gesichter auf dieser Seite gefunden!")
                            
                    with col4:
                        if st.button("🔄 Gallery aktualisieren"):
                            st.cache_data.clear()
                            # Reset to first page to avoid pagination errors
                            st.session_state.gallery_page = 0
                            st.rerun()
                else:
                    st.warning("Keine Gesichtsdaten auf dieser Seite gefunden")
                    
                    # Helpful suggestions for user
                    st.info("""
                    **Mögliche Lösungen:**
                    - Gehen Sie zu einer früheren Seite zurück
                    - Reduzieren Sie die Anzahl Gesichter pro Seite
                    - Klicken Sie auf 'Gallery aktualisieren'
                    - Starten Sie bei Seite 1 neu
                    """)
                    
                    if st.button("🏠 Zur ersten Seite"):
                        st.session_state.gallery_page = 0
                        st.rerun()
                    
            except Exception as db_e:
                st.error(f"Datenbankfehler beim Laden der Gesichter: {db_e}")
                st.info("Versuchen Sie, die Gallery zu aktualisieren oder eine andere Seitenzahl.")
                
    except Exception as e:
        st.error(f"Fehler beim Laden der Face Gallery: {e}")
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
        
        with st.expander(f"📸 {image_name} ({len(face_list)} Gesichter)", expanded=True):
            # Show original image thumbnail
            if Path(image_path).exists():
                try:
                    original_image = load_and_preprocess_image(Path(image_path))
                    if original_image is not None:
                        original_thumbnail = create_thumbnail(original_image, (200, 200))
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(original_thumbnail, caption="Original Image", width='stretch')
                        with col2:
                            st.write(f"**📁 Dateiname:** {image_name}")
                            st.write(f"**👥 Anzahl Gesichter:** {len(face_list)}")
                            if Path(image_path).exists():
                                file_size = Path(image_path).stat().st_size / 1024
                                st.write(f"**💾 Dateigröße:** {file_size:.1f} KB")
                except Exception as e:
                    st.warning(f"Konnte Original-Bild nicht laden: {e}")
            
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

def export_all_faces():
    """Export all faces to a zip file"""
    st.info("Export-Funktion wird implementiert...")
    # TODO: Implement face export functionality

def delete_face_from_gallery(face_id: str) -> bool:
    """Delete a face from the gallery and vector store"""
    try:
        # Delete from vector store
        if st.session_state.vector_store.delete_face(face_id):
            # Update deletion statistics
            if 'deleted_faces_session' not in st.session_state:
                st.session_state.deleted_faces_session = 0
            st.session_state.deleted_faces_session += 1
            
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
    """Enhanced face display from metadata information"""
    try:
        image_path = metadata.get('image_path', '')
        face_id = metadata.get('face_id', 'unknown')
        location_str = metadata.get('location', '0,0,0,0')
        
        if image_path and Path(image_path).exists():
            # Load the original image
            original_image = load_and_preprocess_image(Path(image_path))
            
            if original_image is not None:
                # Parse face location
                try:
                    top, right, bottom, left = map(int, location_str.split(','))
                    
                    # Extract face region with adaptive padding
                    face_width = right - left
                    face_height = bottom - top
                    padding = max(10, min(20, min(face_width, face_height) // 10))
                    
                    height, width = original_image.shape[:2]
                    
                    top = max(0, top - padding)
                    left = max(0, left - padding)
                    bottom = min(height, bottom + padding)
                    right = min(width, right + padding)
                    
                    # Extract face
                    face_image = original_image[top:bottom, left:right]
                    
                    # Validate face extraction
                    if face_image.shape[0] > 0 and face_image.shape[1] > 0:
                        # Create thumbnail with different sizes based on display mode
                        thumbnail_size = (100, 100) if compact else THUMBNAIL_SIZE
                        thumbnail = create_thumbnail(face_image, thumbnail_size)
                        
                        # Display face with enhanced styling
                        st.image(thumbnail, width='stretch')
                        
                        # Single unified info button
                        if st.button("ℹ️ Infos", key=f"gallery_info_{face_id}", help="Vollbild, Gesichtsanalyse und Namen zuweisen", width='stretch'):
                            # Set face info data and trigger modal
                            st.session_state.info_face_id = face_id
                            st.session_state.info_image_path = image_path
                            st.session_state.info_face_location = location_str
                            st.rerun()
                        
                        if show_metadata:
                            # Show person name if assigned
                            person_name = metadata.get('full_name', '')
                            if person_name:
                                st.success(f"👤 **{person_name}**")
                            
                            if compact:
                                st.caption(f"🆔 {face_id[:8]}...")
                                if person_name:
                                    st.caption(f"👤 {person_name}")
                            else:
                                st.caption(f"🆔 Face ID: {face_id}")
                                st.caption(f"📁 {Path(image_path).name}")
                                st.caption(f"📍 Position: {location_str}")
                                
                                # Additional metadata
                                face_size = f"{face_width}×{face_height}px"
                                st.caption(f"📐 Size: {face_size}")
                                
                                # Add hover info with image stats
                                if Path(image_path).exists():
                                    file_size = Path(image_path).stat().st_size / 1024
                                    st.caption(f"💾 {file_size:.1f} KB")
                    else:
                        st.error("❌ Ungültiger Gesichtsbereich")
                        
                except ValueError as ve:
                    st.error(f"❌ Ungültige Koordinaten: {location_str}")
                except Exception as e:
                    st.error(f"❌ Fehler beim Extrahieren: {str(e)[:50]}...")
            else:
                st.error("❌ Bild konnte nicht geladen werden")
                st.caption(f"📁 {Path(image_path).name}")
        else:
            st.error("❌ Bild nicht gefunden")
            st.caption(f"📁 {Path(image_path).name if image_path else 'Unbekannter Pfad'}")
            
    except Exception as e:
        st.error(f"❌ Anzeigefehler: {str(e)[:50]}...")
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
            "Max images per site (0 = unlimited):",
            min_value=0,
            value=0,
            help="0 means unlimited images"
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
                                st.image(image, caption=image_path.name, width='stretch')
                            else:
                                # Create larger thumbnail for better quality
                                thumbnail = create_thumbnail(image, THUMBNAIL_SIZE)
                                st.image(thumbnail, caption=image_path.name, width='stretch')
                            
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
    st.warning("Verwende optimierte Einstellungen für maximale Geschwindigkeit!")
    
    # Get all image directories
    directories = []
    if IMAGES_DIR.exists():
        image_count = len(get_image_files(IMAGES_DIR))
        if image_count > 0:
            directories.append(("Images", IMAGES_DIR, image_count))
    
    if SCRAPED_DIR.exists():
        scraped_count = len(get_image_files(SCRAPED_DIR))
        if scraped_count > 0:
            directories.append(("Scraped", SCRAPED_DIR, scraped_count))
    
    if not directories:
        st.error("Keine Bilder zum Verarbeiten gefunden!")
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
                            st.success(f"⚡ {image_name} → {face_count} Gesicht(er) erkannt")
                        
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
                            st.warning(f"⚠️ {image_name} → Keine Gesichter erkannt")
                
                # Add to database
                if face_ids:
                    added_count = st.session_state.vector_store.add_face_embeddings_batch(
                        face_ids, embeddings, metadatas
                    )
                    total_faces_found += added_count
                    
                    with detailed_log_container:
                        st.info(f"💾 Fast Batch: {added_count} Gesichter zur Datenbank hinzugefügt")
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
            st.success(f"🚀 **Verarbeitungsgeschwindigkeit: {images_per_minute:.1f} Bilder/Minute**")
    
    except Exception as e:
        st.error(f"❌ Error during fast processing: {e}")
        current_status.text("❌ Fast processing failed!")

def name_gallery_page():
    """Name Gallery page to view and manage person names"""
    
    st.header("🏷️ Name Gallery")
    st.markdown("Verwalten Sie alle Personen-Namen in der Datenbank")
    
    try:
        # Get all persons from database with caching
        persons = get_cached_all_persons()
        
        if not persons:
            st.info("Noch keine Namen vergeben. Gehen Sie zu Face Search oder Face Gallery, um Namen zuzuweisen.")
            return
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👤 Anzahl Personen", len(persons))
        with col2:
            total_assigned_faces = sum(p['face_count'] for p in persons)
            st.metric("🏷️ Benannte Gesichter", total_assigned_faces)
        with col3:
            avg_faces = total_assigned_faces / len(persons) if persons else 0
            st.metric("📊 Ø Gesichter/Person", f"{avg_faces:.1f}")
        
        st.markdown("---")
        
        # Search functionality
        col1, col2 = st.columns([2, 1])
        with col1:
            name_search = st.text_input("🔍 Suche nach Namen:", 
                                      placeholder="Vor- oder Nachname eingeben...")
        with col2:
            sort_option = st.selectbox("Sortieren nach:", 
                                     ["Name (A-Z)", "Name (Z-A)", "Gesichter (↑)", "Gesichter (↓)"])
        
        # Filter persons based on search
        filtered_persons = persons
        if name_search.strip():
            search_term = name_search.lower()
            filtered_persons = [p for p in persons 
                              if search_term in p['full_name'].lower()]
        
        # Sort persons
        if sort_option == "Name (A-Z)":
            filtered_persons.sort(key=lambda x: x['full_name'])
        elif sort_option == "Name (Z-A)":
            filtered_persons.sort(key=lambda x: x['full_name'], reverse=True)
        elif sort_option == "Gesichter (↑)":
            filtered_persons.sort(key=lambda x: x['face_count'])
        elif sort_option == "Gesichter (↓)":
            filtered_persons.sort(key=lambda x: x['face_count'], reverse=True)
        
        if not filtered_persons:
            st.warning(f"Keine Personen gefunden für '{name_search}'")
            return
        
        st.markdown(f"**{len(filtered_persons)} Personen gefunden:**")
        
        # Display persons
        for person in filtered_persons:
            with st.expander(f"👤 **{person['full_name']}** ({person['face_count']} Gesichter)", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"**Vorname:** {person['first_name']}")
                    st.write(f"**Nachname:** {person['last_name']}")
                    if person.get('birth_date'):
                        st.write(f"**Geburtsdatum:** {person['birth_date']}")
                    st.write(f"**Person ID:** `{person['person_id']}`")
                
                with col2:
                    st.write(f"**Anzahl Gesichter:** {person['face_count']}")
                    if st.button(f"🔍 Alle Gesichter anzeigen", key=f"show_faces_{person['person_id']}"):
                        st.session_state.selected_person = person['person_id']
                        st.rerun()
                
                with col3:
                    # Simplified one-click deletion
                    if st.button("🗑️ Namen löschen", key=f"remove_person_{person['person_id']}",
                                help="Entfernt den Namen von ALLEN Gesichtern dieser Person (einmaliger Klick)", 
                                type="secondary"):
                        # Direct deletion without confirmation - as requested
                        with st.spinner("Lösche Namen von allen Gesichtern..."):
                            removed_count = 0
                            for face_id in person['face_ids']:
                                if st.session_state.vector_store.remove_person_name(face_id):
                                    removed_count += 1
                            
                            if removed_count > 0:
                                st.success(f"✅ Name von {removed_count} Gesichtern entfernt!")
                                # Invalidate persons cache for refresh
                                invalidate_persons_cache()
                                st.rerun()
                            else:
                                st.error("❌ Fehler beim Entfernen der Namen")
        
        # Display faces for selected person
        if st.session_state.get('selected_person'):
            person_id = st.session_state.selected_person
            person_faces = st.session_state.vector_store.get_faces_by_person_id(person_id)
            
            if person_faces:
                st.markdown("---")
                selected_person_name = next((p['full_name'] for p in persons if p['person_id'] == person_id), "Unbekannt")
                st.subheader(f"🔍 Alle Gesichter von **{selected_person_name}**")
                
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
                            full_image = load_and_preprocess_image(image_path)
                            if full_image is not None and face_location:
                                coords = face_location.split(',')
                                if len(coords) == 4:
                                    top, right, bottom, left = map(int, coords)
                                    face_crop = full_image[top:bottom, left:right]
                                    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                                        # Create thumbnail for consistent display
                                        thumbnail = create_thumbnail(face_crop, (150, 150))
                                        st.image(thumbnail, caption=f"Face {face_id[:8]}...", width='stretch')
                            else:
                                st.warning("Gesicht konnte nicht geladen werden")
                                
                        except Exception as e:
                            st.error(f"Fehler beim Laden: {e}")
                        
                        # Face info
                        st.caption(f"**ID:** {face_id[:12]}...")
                        st.caption(f"**Bild:** {Path(image_path).name}")
                        
                        # Buttons in columns for better layout
                        btn_col1, btn_col2 = st.columns(2)
                        
                        with btn_col1:
                            # Single unified info button
                            if st.button("ℹ️ Infos", key=f"name_gallery_info_{face_id}", help="Vollbild, Gesichtsanalyse und Namen zuweisen"):
                                # Preserve selected_person state for Name Gallery
                                st.session_state.name_gallery_selected_person = st.session_state.selected_person
                                # Set face info data and trigger modal
                                st.session_state.info_face_id = face_id
                                st.session_state.info_image_path = image_path
                                st.session_state.info_face_location = face_location
                                st.rerun()
                        
                        with btn_col2:
                            # Remove photo button
                            if st.button("🗑️", key=f"remove_photo_{face_id}", help="Foto von dieser Person entfernen", type="secondary"):
                                # Remove person name from this specific face
                                if st.session_state.vector_store.remove_person_name(face_id):
                                    st.success(f"✅ Foto erfolgreich von {selected_person_name} entfernt!")
                                    # Refresh the view
                                    st.rerun()
                                else:
                                    st.error("❌ Fehler beim Entfernen des Fotos")
                
                # Close view button
                st.markdown("---")
                if st.button("❌ Ansicht schließen", type="primary"):
                    del st.session_state.selected_person
                    st.rerun()
    
    except Exception as e:
        st.error(f"Fehler beim Laden der Name Gallery: {e}")

if __name__ == "__main__":
    main()