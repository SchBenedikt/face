"""
Face Recognition Search Application - Streamlit Frontend
"""
# Set up environment before importing other modules to prevent threading issues
from setup_env import setup_environment
setup_environment()

import warnings
# Suppress the pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

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

logger = logging.getLogger(__name__)

# Import our modules
from face_recognition_engine import FaceRecognitionEngine
from vector_store import FaceVectorStore
from image_scraper import AdvancedImageScraper, advanced_scraper
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

def main():
    """Main application function"""
    
    st.title("🔍 Face Recognition Search")
    
    # Show improvements banner
    with st.container():
        st.success("🚀 **Verbesserte Gesichtserkennung aktiv!** Diese Version verwendet Ensemble-Modelle und erweiterte Ähnlichkeitsalgorithmen für höchste Genauigkeit.")
        with st.expander("ℹ️ Neue Features & Verbesserungen", expanded=False):
            st.markdown("""
            **🎯 Ensemble-Gesichtserkennung:**
            - Mehrere DeepFace-Modelle: Facenet512, ArcFace, VGG-Face, Facenet
            - Gewichtete Kombination für maximale Genauigkeit
            - Automatische Fallback-Mechanismen
            
            **📊 Erweiterte Ähnlichkeitsberechnung:**
            - 7+ Ähnlichkeitsmetriken (Cosine, Euclidean, Correlation, Angular, etc.)
            - Ensemble-Scoring mit optimierten Gewichtungen
            - Vertrauens-Score basierend auf Metrik-Konsistenz
            
            **🔧 Qualitätsverbesserungen:**
            - Erweiterte Gesichtsvorverarbeitung mit Ausrichtung
            - Qualitätsvalidierung und Optimierung
            - Mehrere Erkennungsbackends (OpenCV, DeepFace, face_recognition)
            
            **⚡ Performance-Optimierungen:**
            - Konfigurierbare Verarbeitungsmodi
            - Adaptive Gesichtsfilterung
            - Verbesserte Fehlerbehandlung
            """)
    
    st.markdown("---")
    
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
    
    # Main content area
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
    elif page == "🔧 Duplicate Manager":
        duplicate_manager_page()
    elif page == "📊 Database Statistics":
        database_stats_page()
    elif page == "⚙️ Settings":
        settings_page()

def face_search_page():
    """Face search functionality"""
    
    st.header("🔍 Search for Similar Faces")
    
    # Upload query image
    uploaded_file = st.file_uploader(
        "Upload a photo to search for similar faces:",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Upload an image containing a face to search for similar faces in the database"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file is not None:
            # Display uploaded image
            query_image = Image.open(uploaded_file)
            st.image(query_image, caption="Query Image", use_container_width=True)
            
            # Search parameters
            st.subheader("Search Parameters")
            
            max_results = st.slider(
                "Maximum Results", 
                min_value=5, 
                max_value=50, 
                value=20,
                help="Maximum number of similar faces to return"
            )
            
            similarity_threshold = st.slider(
                "Similarity Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.4,  # Erhöhter Default für bessere Ergebnisse
                step=0.05,
                help="Mindestähnlichkeit für Suchergebnisse (0.4 = 40% Ähnlichkeit). Höhere Werte = genauere Ergebnisse aber weniger Treffer."
            )
            
            if st.button("🔍 Search Similar Faces", type="primary"):
                search_similar_faces(uploaded_file, max_results, similarity_threshold)
    
    with col2:
        # Display search results
        if st.session_state.search_results:
            display_search_results(st.session_state.search_results)

def search_similar_faces(uploaded_file, max_results: int, similarity_threshold: float):
    """Perform enhanced face similarity search with ensemble processing"""
    
    # Initialize temp_path outside try block
    temp_path = Path("/tmp/query_image.jpg")
    
    # Create progress container
    progress_container = st.container()
    
    with st.spinner("🔍 Erweiterte Gesichtserkennung läuft..."):
        try:
            # Save uploaded file temporarily
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process query image with detailed progress
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()
            
            # Step 1: Load and preprocess image
            status_text.text("📷 Lade und verarbeite Bild...")
            progress_bar.progress(15)
            query_image = load_and_preprocess_image(temp_path)
            
            if query_image is None:
                st.error("❌ Fehler beim Laden des Bildes. Unterstützte Formate: JPG, PNG, BMP, WEBP")
                return
            
            # Step 2: Detect faces with multiple backends
            status_text.text("👤 Erkenne Gesichter mit mehreren Algorithmen...")
            progress_bar.progress(35)
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
            
            # Show detected faces info
            if len(face_locations) > 1:
                st.info(f"👥 {len(face_locations)} Gesichter erkannt. Verwende das erste Gesicht für die Suche.")
            else:
                st.success("✅ 1 Gesicht erkannt und verarbeitet.")
            
            # Step 3: Extract embedding with ensemble models
            face_location = face_locations[0]
            status_text.text("🧠 Extrahiere Gesichtsmerkmale mit Ensemble-Modellen...")
            progress_bar.progress(60)
            
            query_embedding = st.session_state.face_engine.extract_face_embedding(query_image, face_location)
            
            if query_embedding is None:
                st.error("""
                ❌ **Fehler bei der Gesichtsanalyse!**
                
                Dies kann folgende Ursachen haben:
                - Das Gesicht ist zu unscharf oder schlecht beleuchtet
                - Das Bild hat eine zu niedrige Auflösung
                - Das erkannte Gesicht ist zu klein
                
                Versuchen Sie es mit einem anderen Bild.
                """)
                return
            
            # Step 4: Search with enhanced similarity calculation  
            status_text.text("🔍 Suche ähnliche Gesichter mit erweiterten Algorithmen...")
            progress_bar.progress(85)
            
            similar_faces = st.session_state.vector_store.search_similar_faces(
                query_embedding, 
                n_results=max_results,
                min_similarity=similarity_threshold
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Suche abgeschlossen!")
            
            # Store results in session state
            st.session_state.search_results = similar_faces
            
            # Show enhanced result summary
            if similar_faces:
                avg_similarity = sum(face['similarity'] for face in similar_faces) / len(similar_faces)
                top_similarity = similar_faces[0]['similarity'] if similar_faces else 0
                
                st.success(f"""
                🎯 **{len(similar_faces)} ähnliche Gesichter gefunden!**
                
                📊 **Ergebnisqualität:**
                - Top-Ähnlichkeit: {top_similarity*100:.1f}%
                - Durchschnitts-Ähnlichkeit: {avg_similarity*100:.1f}%
                - Ensemble-Modelle: Aktiv
                - Vertrauens-Scoring: Aktiv
                """)
            else:
                st.warning(f"""
                ⚠️ **Keine ähnlichen Gesichter gefunden**
                
                **Versuchen Sie:**
                - Ähnlichkeitsschwelle senken (aktuell: {similarity_threshold*100:.0f}%)
                - Mehr Bilder zur Datenbank hinzufügen
                - Ein anderes Queryimage verwenden
                
                Aktuelle Datenbank: {st.session_state.vector_store.get_collection_stats().get('total_faces', 0)} Gesichter
                """)
            
        except Exception as e:
            st.error(f"""
            💥 **Unerwarteter Fehler bei der Gesichtssuche:**
            
            `{str(e)}`
            
            Bitte versuchen Sie es erneut oder verwenden Sie ein anderes Bild.
            """)
            logger.error(f"Search error: {e}")
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
            
            # Clear progress indicators
            progress_container.empty()
            
            # Search for similar faces
            progress_bar.progress(80)
            similar_faces = st.session_state.vector_store.search_similar_faces(
                query_embedding, 
                n_results=max_results,
                min_similarity=similarity_threshold
            )
            
            progress_bar.progress(100)
            
            # Store results in session state
            st.session_state.search_results = similar_faces
            
            if similar_faces:
                st.success(f"Found {len(similar_faces)} similar faces!")
            else:
                st.warning("No similar faces found. Try lowering the similarity threshold or adding more images to the database.")
            
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

def display_search_results(results: List[Dict[str, Any]]):
    """Display search results"""
    
    st.subheader(f"Search Results ({len(results)} faces)")
    
    if not results:
        st.info("No results to display")
        return
    
    # Display options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{len(results)} results found**")
    
    with col2:
        show_all = st.toggle("Show All Images", value=False, help="Toggle pagination on/off")
    
    # Determine what to display
    if show_all:
        # Show all results
        page_results = results
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
                # Load image
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
                                    st.image(thumbnail, use_container_width=True)
                                    
                                    # Enhanced similarity display with confidence levels
                                    similarity_percentage = face_data['similarity'] * 100
                                    confidence_score = face_data.get('confidence_score', similarity_percentage / 100) * 100
                                    
                                    # Color-coded similarity display based on face recognition standards
                                    if similarity_percentage >= 70:
                                        st.success(f"**🎯 {similarity_percentage:.1f}%** (Sehr hohe Ähnlichkeit)")
                                        st.caption(f"🔒 Vertrauen: {confidence_score:.1f}%")
                                    elif similarity_percentage >= 50:
                                        st.info(f"**✅ {similarity_percentage:.1f}%** (Hohe Ähnlichkeit)")
                                        st.caption(f"📊 Vertrauen: {confidence_score:.1f}%")
                                    elif similarity_percentage >= 30:
                                        st.warning(f"**⚠️ {similarity_percentage:.1f}%** (Mittlere Ähnlichkeit)")
                                        st.caption(f"📈 Vertrauen: {confidence_score:.1f}%")
                                    else:
                                        st.error(f"**❓ {similarity_percentage:.1f}%** (Niedrige Ähnlichkeit)")
                                        st.caption(f"🔍 Vertrauen: {confidence_score:.1f}%")
                                    
                                    st.write(f"**📁 {image_path.name}**")
                                    
                                    # Show detailed metrics in expander for advanced users
                                    with st.expander("🔬 Detaillierte Ensemble-Metriken", expanded=False):
                                        similarity_metrics = face_data.get('similarity_metrics', {})
                                        
                                        # Primary metrics
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("🎯 Haupt-Ähnlichkeit", f"{face_data.get('similarity', 0):.3f}")
                                            st.metric("📐 Cosine Similarity", f"{similarity_metrics.get('cosine_similarity', 0):.3f}")
                                            st.metric("📏 Euclidean Similarity", f"{similarity_metrics.get('euclidean_similarity', 0):.3f}")
                                        
                                        with col2:
                                            st.metric("🔢 Ensemble Score", f"{similarity_metrics.get('ensemble_score', 0):.3f}")
                                            st.metric("📊 Correlation", f"{similarity_metrics.get('correlation_similarity', 0):.3f}")
                                            st.metric("📐 Angular Similarity", f"{similarity_metrics.get('angular_similarity', 0):.3f}")
                                        
                                        # Additional metrics if available
                                        if 'manhattan_similarity' in similarity_metrics:
                                            st.metric("🏃 Manhattan Similarity", f"{similarity_metrics.get('manhattan_similarity', 0):.3f}")
                                        
                                        # Model performance indicator
                                        ensemble_score = similarity_metrics.get('ensemble_score', 0)
                                        if ensemble_score > 0.7:
                                            st.success("🤖 Ensemble-Modell: Sehr hohe Übereinstimmung")
                                        elif ensemble_score > 0.5:
                                            st.info("🤖 Ensemble-Modell: Gute Übereinstimmung") 
                                        elif ensemble_score > 0.3:
                                            st.warning("🤖 Ensemble-Modell: Mäßige Übereinstimmung")
                                        else:
                                            st.error("🤖 Ensemble-Modell: Niedrige Übereinstimmung")
                                else:
                                    # Fallback: show full image
                                    st.image(image, use_container_width=True)
                                    st.write(f"**Similarity:** {face_data['similarity']:.3f}")
                            else:
                                # Handle tuple format
                                try:
                                    top, right, bottom, left = location
                                    face_image = image[top:bottom, left:right]
                                    
                                    # Create thumbnail
                                    thumbnail = create_thumbnail(face_image, THUMBNAIL_SIZE)
                                    
                                    # Display
                                    st.image(thumbnail, use_container_width=True)
                                    # Display with enhanced similarity information
                                    st.image(thumbnail, use_container_width=True)
                                    
                                    # Enhanced similarity display
                                    similarity_percentage = face_data['similarity'] * 100
                                    confidence_score = face_data.get('confidence_score', similarity_percentage / 100) * 100
                                    
                                    if similarity_percentage >= 70:
                                        st.success(f"**🎯 {similarity_percentage:.1f}%** (Sehr hohe Ähnlichkeit)")
                                    elif similarity_percentage >= 50:
                                        st.info(f"**✅ {similarity_percentage:.1f}%** (Hohe Ähnlichkeit)")
                                    elif similarity_percentage >= 30:
                                        st.warning(f"**⚠️ {similarity_percentage:.1f}%** (Mittlere Ähnlichkeit)")
                                    else:
                                        st.error(f"**❓ {similarity_percentage:.1f}%** (Niedrige Ähnlichkeit)")
                                    
                                    st.write(f"**📁 {image_path.name}**")
                                    st.caption(f"🔒 Vertrauen: {confidence_score:.1f}%")
                                except:
                                    # Fallback: show full image with enhanced similarity display
                                    st.image(image, use_container_width=True)
                                    
                                    similarity_percentage = face_data['similarity'] * 100
                                    confidence_score = face_data.get('confidence_score', similarity_percentage / 100) * 100
                                    
                                    if similarity_percentage >= 70:
                                        st.success(f"**🎯 {similarity_percentage:.1f}%** (Sehr hohe Ähnlichkeit)")
                                    elif similarity_percentage >= 50:
                                        st.info(f"**✅ {similarity_percentage:.1f}%** (Hohe Ähnlichkeit)")
                                    elif similarity_percentage >= 30:
                                        st.warning(f"**⚠️ {similarity_percentage:.1f}%** (Mittlere Ähnlichkeit)")
                                    else:
                                        st.error(f"**❓ {similarity_percentage:.1f}%** (Niedrige Ähnlichkeit)")
                                    
                                    st.write(f"**📁 {image_path.name}**")
                                    st.caption(f"🔒 Vertrauen: {confidence_score:.1f}%")
                        else:
                            st.image(image, use_container_width=True)
                            
                            similarity_percentage = face_data['similarity'] * 100
                            confidence_score = face_data.get('confidence_score', similarity_percentage / 100) * 100
                            
                            if similarity_percentage >= 70:
                                st.success(f"**🎯 {similarity_percentage:.1f}%** (Sehr hohe Ähnlichkeit)")
                            elif similarity_percentage >= 50:
                                st.info(f"**✅ {similarity_percentage:.1f}%** (Hohe Ähnlichkeit)")
                            elif similarity_percentage >= 30:
                                st.warning(f"**⚠️ {similarity_percentage:.1f}%** (Mittlere Ähnlichkeit)")
                            else:
                                st.error(f"**❓ {similarity_percentage:.1f}%** (Niedrige Ähnlichkeit)")
                            
                            st.write(f"**📁 {image_path.name}**")
                            st.caption(f"🔒 Vertrauen: {confidence_score:.1f}%")
                else:
                    st.error("Image not found")
                    
            except Exception as e:
                st.error(f"Error displaying result: {e}")

def batch_face_processing_page():
    """Dedicated batch face processing page with comprehensive options"""
    
    st.header("🧠 Batch Face Processing")
    st.markdown("**Comprehensive face detection and processing tools for large image collections**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Process Directories", "⚡ Quick Process", "🔄 Re-process Images", "📊 Processing Stats"])
    
    with tab1:
        st.subheader("Process Image Directories")
        
        # Show available directories
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
            st.info("No image directories found with images. Upload images or scrape websites first.")
            return
        
        # Directory selection and processing
        for name, path, count in directories:
            with st.expander(f"📁 {name} Directory - {count} images"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Path:** `{path}`")
                    st.write(f"**Images:** {count}")
                
                with col2:
                    batch_size = st.slider(
                        f"Batch size for {name}:", 
                        min_value=5, 
                        max_value=50, 
                        value=25, 
                        key=f"batch_{name}"
                    )
                
                with col3:
                    max_workers = st.slider(
                        f"Workers for {name}:", 
                        min_value=1, 
                        max_value=4, 
                        value=1, 
                        key=f"workers_{name}",
                        help="More workers = faster but uses more CPU"
                    )
                
                # Advanced options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    skip_existing = st.checkbox(
                        f"Skip existing faces", 
                        value=True, 
                        key=f"skip_{name}",
                        help="Don't reprocess images that already have faces in database"
                    )
                
                with col2:
                    detailed_progress = st.checkbox(
                        f"Detailed progress", 
                        value=True, 
                        key=f"progress_{name}"
                    )
                
                with col3:
                    save_face_images = st.checkbox(
                        f"Save face crops", 
                        value=False, 
                        key=f"save_{name}",
                        help="Save cropped face images to disk"
                    )
                
                if st.button(f"🚀 Process {name} Directory", type="primary", key=f"process_{name}"):
                    process_directory_batch(
                        path, 
                        batch_size=batch_size,
                        max_workers=max_workers,
                        skip_existing=skip_existing,
                        detailed_progress=detailed_progress,
                        save_crops=save_face_images
                    )
    
    with tab2:
        st.subheader("⚡ Quick Processing")
        st.markdown("**One-click processing of all available images**")
        
        # Summary of available images
        total_images = 0
        summary_data = []
        
        for name, path, count in directories:
            total_images += count
            summary_data.append({
                "Directory": name,
                "Path": str(path),
                "Images": count
            })
        
        if summary_data:
            st.table(summary_data)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"**Total Images Available:** {total_images}")
            
            with col2:
                processing_speed = st.select_slider(
                    "Processing Speed:",
                    options=["Careful", "Balanced", "Fast"],
                    value="Balanced",
                    help="Careful: 1 worker, Fast: 4 workers"
                )
            
            # Quick processing options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                quick_batch_size = st.selectbox("Batch Size:", [10, 25, 50], index=1)
            
            with col2:
                workers_map = {"Careful": 1, "Balanced": 2, "Fast": 4}
                workers = workers_map[processing_speed]
                st.write(f"**Workers:** {workers}")
            
            with col3:
                est_time = (total_images / quick_batch_size) * 2  # rough estimate
                st.write(f"**Est. Time:** ~{est_time:.0f} min")
            
            # Big process button
            if st.button("🚀 PROCESS ALL IMAGES", type="primary", use_container_width=True):
                quick_process_all_directories(
                    directories,
                    batch_size=quick_batch_size,
                    max_workers=workers
                )
    
    with tab3:
        st.subheader("🔄 Re-process Images")
        st.markdown("**Re-analyze images that might have been missed or need better face detection**")
        
        # Database analysis
        stats = st.session_state.vector_store.get_collection_stats()
        total_faces = stats.get("total_faces", 0)
        
        if total_faces > 0:
            st.info(f"Current database contains {total_faces} faces")
            
            # Re-processing options
            col1, col2 = st.columns([2, 1])
            
            with col1:
                reprocess_mode = st.selectbox(
                    "Re-processing Mode:",
                    [
                        "Failed images only",
                        "Images with no faces found", 
                        "All images (full re-scan)",
                        "Images processed before today"
                    ]
                )
            
            with col2:
                clear_existing = st.checkbox(
                    "Clear existing faces first",
                    value=False,
                    help="WARNING: This will remove all current face data!"
                )
            
            if clear_existing:
                st.warning("⚠️ This will delete ALL existing face data from the database!")
            
            if st.button("🔄 Start Re-processing", type="secondary"):
                if clear_existing:
                    if st.button("⚠️ CONFIRM: Clear database and re-process", type="primary"):
                        st.session_state.vector_store.clear_collection()
                        st.success("Database cleared! Now re-processing...")
                        quick_process_all_directories(directories, batch_size=25, max_workers=1)
                else:
                    st.info(f"Re-processing in mode: {reprocess_mode}")
                    # Implementation would go here
        else:
            st.warning("No faces in database yet. Use the other tabs to process images first.")
    
    with tab4:
        st.subheader("📊 Processing Statistics & Insights")
        
        # Current stats
        stats = st.session_state.vector_store.get_collection_stats()
        
        if stats and "error" not in stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Faces", stats.get("total_faces", 0))
            
            with col2:
                st.metric("Unique Images", stats.get("unique_images", 0))
            
            with col3:
                if stats.get("total_faces", 0) > 0 and stats.get("unique_images", 0) > 0:
                    avg_faces = stats["total_faces"] / stats["unique_images"]
                    st.metric("Avg Faces/Image", f"{avg_faces:.1f}")
                else:
                    st.metric("Avg Faces/Image", "0")
            
            with col4:
                # Processing efficiency (mock calculation)
                total_available = sum(count for _, _, count in directories) if directories else 0
                if total_available > 0 and stats.get("unique_images", 0) > 0:
                    efficiency = (stats["unique_images"] / total_available) * 100
                    st.metric("Processing Rate", f"{efficiency:.0f}%")
                else:
                    st.metric("Processing Rate", "0%")
            
            # Processing recommendations
            st.subheader("💡 Processing Recommendations")
            
            if stats.get("total_faces", 0) == 0:
                st.info("🚀 **Start Here:** Use 'Quick Process' to process all your images at once")
            elif stats.get("total_faces", 0) < 10:
                st.info("📈 **Build Database:** Process more images to improve face search quality")
            else:
                st.success("✅ **Good Coverage:** Your database has a good number of faces for searching")
            
            # Performance tips
            st.subheader("⚡ Performance Tips")
            st.markdown("""
            - **Batch Size:** Larger batches = more memory usage but better performance
            - **Workers:** More workers = faster processing but higher CPU usage  
            - **Skip Existing:** Enable to avoid reprocessing already analyzed images
            - **Image Quality:** Higher quality images = better face detection accuracy
            """)
            
        else:
            st.error("Could not load database statistics")

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
                
                for result in results:
                    if "error" not in result and result.get("face_count", 0) > 0:
                        total_processed += 1
                        batch_faces += result["face_count"]
                        
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
                        if detailed_progress and detailed_container:
                            with detailed_container:
                                st.error(f"❌ Error processing {Path(result.get('image_path', 'unknown')).name}: {result['error']}")
                    else:
                        # Image processed successfully but no faces found
                        total_processed += 1
                
                # Add to vector store
                if face_ids:
                    added_count = st.session_state.vector_store.add_face_embeddings_batch(
                        face_ids, embeddings, metadatas
                    )
                    total_faces_found += added_count
                    
                    if detailed_progress and detailed_container:
                        with detailed_container:
                            st.success(f"✅ Batch {batch_num + 1}: Found {batch_faces} faces, added {added_count} to database")
                else:
                    if detailed_progress and detailed_container:
                        with detailed_container:
                            st.warning(f"⚠️ Batch {batch_num + 1}: No faces detected")
                
                # Update batch info with comprehensive status
                if batch_faces > 0:
                    batch_info.success(f"✅ Batch {batch_num + 1}: {batch_faces} faces found")
                elif batch_errors > 0:
                    batch_info.error(f"❌ Batch {batch_num + 1}: {batch_errors} errors, no faces found")
                else:
                    batch_info.info(f"⚪ Batch {batch_num + 1}: No faces detected")
            
            # Final update
            overall_progress.progress(1.0)
            current_status.text("✅ Processing complete!")
            
            # Final summary
            st.success(f"🎉 **Processing Complete!**")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric("Images Processed", total_processed)
            with summary_col2:
                st.metric("Total Faces Found", total_faces_found)
            with summary_col3:
                if total_processed > 0:
                    avg_faces = total_faces_found / total_processed
                    st.metric("Avg Faces/Image", f"{avg_faces:.1f}")
            
        except Exception as e:
            st.error(f"❌ Error during batch processing: {e}")
            current_status.text("❌ Processing failed!")

def quick_process_all_directories(directories, batch_size=25, max_workers=2):
    """Quick process all directories with unified progress tracking"""
    
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
    
    global_batch_count = 0
    total_faces_found = 0
    total_images_processed = 0
    
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
                
                # Process results
                face_ids = []
                embeddings = []
                metadatas = []
                batch_faces = 0
                
                for result in results:
                    if "error" not in result and result["face_count"] > 0:
                        total_images_processed += 1
                        batch_faces += result["face_count"]
                        
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
                
                # Add to database
                if face_ids:
                    added_count = st.session_state.vector_store.add_face_embeddings_batch(
                        face_ids, embeddings, metadatas
                    )
                    total_faces_found += added_count
                
                global_batch_count += 1
                
                # Show progress in results container
                with results_container:
                    progress_text = f"✅ {dir_name} Batch {batch_num + 1}: {batch_faces} faces"
                    if batch_faces > 0:
                        st.success(progress_text)
                    else:
                        st.info(f"⚪ {dir_name} Batch {batch_num + 1}: No faces detected")
        
        # Final completion
        overall_progress.progress(1.0)
        current_status.text("🎉 All directories processed successfully!")
        
        # Final summary
        st.balloons()
        
        st.success("🎉 **QUICK PROCESSING COMPLETE!**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Images Processed", total_images_processed)
        with col2:
            st.metric("Faces Found", total_faces_found)  
        with col3:
            if total_images_processed > 0:
                avg_faces = total_faces_found / total_images_processed
                st.metric("Avg Faces/Image", f"{avg_faces:.1f}")
        
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
                                st.image(thumbnail, use_container_width=True)
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
                                st.image(thumbnail, use_container_width=True)
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
    """Face gallery to display all faces in the database"""
    
    st.header("👥 Face Gallery")
    st.markdown("Browse all faces stored in the database")
    
    # Get all faces from database
    try:
        stats = st.session_state.vector_store.get_collection_stats()
        total_faces = stats.get("total_faces", 0)
        
        if total_faces == 0:
            st.info("No faces in database yet. Upload images or scrape websites to add faces.")
            return
            
        st.success(f"Found {total_faces} faces in database")
        
        # Gallery options
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            faces_per_page = st.selectbox("Faces per page:", [20, 50, 100, 200], index=1)
        
        with col2:
            show_metadata = st.checkbox("Show metadata", value=True)
        
        with col3:
            group_by_image = st.checkbox("Group by source image", value=False)
        
        # Get faces data
        offset = 0
        limit = min(faces_per_page, total_faces)
        
        with st.spinner("Loading faces..."):
            # Get face data from database
            results = st.session_state.vector_store.collection.get(
                limit=limit,
                offset=offset,
                include=['metadatas', 'embeddings']
            )
            
            if results['metadatas']:
                st.subheader(f"Displaying {len(results['metadatas'])} faces")
                
                # Group faces if requested
                if group_by_image:
                    faces_by_image = {}
                    for i, metadata in enumerate(results['metadatas']):
                        image_path = metadata.get('image_path', 'unknown')
                        if image_path not in faces_by_image:
                            faces_by_image[image_path] = []
                        faces_by_image[image_path].append((i, metadata))
                    
                    # Display grouped faces
                    for image_path, face_list in faces_by_image.items():
                        image_name = Path(image_path).name
                        st.subheader(f"📸 {image_name} ({len(face_list)} faces)")
                        
                        cols = st.columns(min(4, len(face_list)))
                        for idx, (face_idx, metadata) in enumerate(face_list):
                            with cols[idx % 4]:
                                display_face_from_metadata(metadata, show_metadata)
                        
                        st.markdown("---")
                else:
                    # Display all faces in grid
                    cols_per_row = 4
                    for i in range(0, len(results['metadatas']), cols_per_row):
                        cols = st.columns(cols_per_row)
                        
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(results['metadatas']):
                                with col:
                                    metadata = results['metadatas'][idx]
                                    display_face_from_metadata(metadata, show_metadata)
                
                # Pagination controls (for future enhancement)
                if total_faces > faces_per_page:
                    st.info(f"Showing first {faces_per_page} faces. Pagination coming soon!")
            else:
                st.warning("No face data found in database")
                
    except Exception as e:
        st.error(f"Error loading faces: {e}")

def display_face_from_metadata(metadata, show_metadata=True):
    """Display a face from metadata information"""
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
                    
                    # Extract face region with some padding
                    padding = 20
                    height, width = original_image.shape[:2]
                    
                    top = max(0, top - padding)
                    left = max(0, left - padding)
                    bottom = min(height, bottom + padding)
                    right = min(width, right + padding)
                    
                    # Extract face
                    face_image = original_image[top:bottom, left:right]
                    
                    # Create thumbnail for display
                    thumbnail = create_thumbnail(face_image, THUMBNAIL_SIZE)
                    
                    # Display face
                    st.image(thumbnail, caption=f"Face: {face_id}", use_container_width=True)
                    
                    if show_metadata:
                        st.caption(f"📁 {Path(image_path).name}")
                        st.caption(f"📍 {location_str}")
                        
                except Exception as e:
                    st.error(f"Error extracting face: {e}")
            else:
                st.error("Could not load image")
        else:
            st.error(f"Image not found: {Path(image_path).name}")
            
    except Exception as e:
        st.error(f"Error displaying face: {e}")

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
                                st.image(image, caption=image_path.name, use_container_width=True)
                            else:
                                # Create larger thumbnail for better quality
                                thumbnail = create_thumbnail(image, THUMBNAIL_SIZE)
                                st.image(thumbnail, caption=image_path.name, use_container_width=True)
                            
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

if __name__ == "__main__":
    main()