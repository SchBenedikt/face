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
from datetime import datetime
import threading
import queue
import tempfile

# Live camera functionality
LIVE_CAMERA_AVAILABLE = True
try:
    import mediapipe as mp
    import tensorflow as tf
except ImportError:
    LIVE_CAMERA_AVAILABLE = False

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
                "� Live Camera AI",
                "�👥 Face Gallery",
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
    
    # Main content area
    if page == "🔍 Face Search":
        face_search_page()
    elif page == "📥 Image Upload & Processing":
        upload_processing_page()
    elif page == "🧠 Batch Face Processing":
        batch_face_processing_page()
    elif page == "🌐 Web Scraping":
        web_scraping_page()
    elif page == "� Live Camera AI":
        live_camera_page()
    elif page == "�👥 Face Gallery":
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
                max_value=200, 
                value=50,
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
            
            # Facial Attribute Analysis Info
            with st.expander("🧬 Facial Attribute Analysis", expanded=False):
                st.markdown("""
                **🎯 Neue Funktion: Detaillierte Gesichtsanalyse**
                
                Zusätzlich zur Ähnlichkeitssuche können Sie jetzt für jedes gefundene Gesicht eine detaillierte Analyse durchführen:
                
                **📊 Analysierte Attribute:**
                - **👤 Alter**: Geschätztes Alter (±4,65 Jahre Genauigkeit)  
                - **⚧️ Geschlecht**: Männlich/Weiblich (97,44% Genauigkeit)
                - **😊 Emotionen**: Glücklich, Neutral, Traurig, Wütend, Überrascht, Ängstlich, Angeekelt
                - **🌍 Ethnische Herkunft**: Asiatisch, Kaukasisch, Nahöstlich, Indisch, Lateinamerikanisch, Afrikanisch
                
                **🔬 Technische Details:**
                - Basiert auf DeepFace-Bibliothek mit neuronalen Netzen
                - Hochpräzise Modelle für jeden Attributtyp
                - Konfidenzwerte für jede Vorhersage
                
                **🚀 Verwendung:**
                Klicken Sie auf den **🧬 Analyse**-Button bei jedem Suchergebnis für detaillierte Informationen.
                """)
                
                # Check if DeepFace is available
                try:
                    import deepface
                    st.success("✅ DeepFace ist installiert und bereit für Gesichtsanalyse!")
                except ImportError:
                    st.warning("⚠️ DeepFace nicht installiert. Installieren Sie mit: `pip install deepface`")
                    if st.button("📥 DeepFace Installation anzeigen"):
                        st.code("pip install deepface", language="bash")
                        st.info("Nach der Installation starten Sie die Anwendung neu.")
            
            if st.button("🔍 Search Similar Faces", type="primary"):
                search_similar_faces(uploaded_file, max_results, similarity_threshold)
    
    with col2:
        # Display search results
        if st.session_state.search_results:
            display_search_results(st.session_state.search_results)
    
    # The modals are now called directly from button events, no need for session state checks

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
            
            # Show detected faces info with selection option
            if len(face_locations) > 1:
                st.warning(f"👥 {len(face_locations)} Gesichter erkannt!")
                
                # Show all detected faces for user to choose from
                st.write("**Wählen Sie das Gesicht für die Suche aus:**")
                
                cols = st.columns(min(len(face_locations), 4))
                face_choice = None
                
                for idx, face_loc in enumerate(face_locations):
                    with cols[idx % 4]:
                        # Extract face region
                        if isinstance(face_loc, dict):
                            top, right, bottom, left = face_loc['top'], face_loc['right'], face_loc['bottom'], face_loc['left']
                        else:
                            top, right, bottom, left = face_loc
                        
                        face_image = query_image[top:bottom, left:right]
                        face_thumbnail = create_thumbnail(face_image, (100, 100))
                        
                        st.image(face_thumbnail, caption=f"Gesicht {idx+1}")
                        if st.button(f"Wählen", key=f"face_{idx}"):
                            face_choice = idx
                
                if face_choice is not None:
                    face_location = face_locations[face_choice]
                    st.success(f"✅ Gesicht {face_choice+1} ausgewählt für die Suche.")
                else:
                    st.info("👆 Bitte wählen Sie ein Gesicht aus, um fortzufahren.")
                    return
            else:
                st.success("✅ 1 Gesicht erkannt und verarbeitet.")
                face_location = face_locations[0]
            # Step 3: Extract embedding with ensemble models
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

def show_full_image_with_face_box(image_path, face_location):
    """Display full image with face bounding box"""
    try:
        # Load the full image
        full_image = load_and_preprocess_image(image_path)
        if full_image is None:
            st.error("❌ Fehler beim Laden des Bildes")
            return
        
        # Parse face location
        if isinstance(face_location, str):
            coords = face_location.split(',')
            if len(coords) == 4:
                top, right, bottom, left = map(int, coords)
            else:
                st.error("❌ Ungültige Gesichtskoordinaten")
                return
        else:
            top, right, bottom, left = face_location
        
        # Draw bounding box on the image
        image_with_box = full_image.copy()
        # The image from load_and_preprocess_image is already in RGB format
        # OpenCV rectangle expects (B,G,R) for color, so for RGB we need (R,G,B)
        cv2.rectangle(image_with_box, (left, top), (right, bottom), (0, 255, 0), 3)
        
        # No color conversion needed - image is already in RGB format
        
        # Create a modal-like display
        with st.expander(f"🖼️ Vollbild: {image_path.name}", expanded=True):
            st.image(image_with_box, caption=f"Vollbild mit markiertem Gesicht - {image_path.name}", use_container_width=True)
            st.success(f"✅ Gesicht markiert bei Koordinaten: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
    
    except Exception as e:
        st.error(f"❌ Fehler beim Anzeigen des Vollbildes: {str(e)}")

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
    all_persons = st.session_state.vector_store.get_all_persons()
    
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
            if st.button("✅ Diese Person zuweisen", type="primary", use_container_width=True):
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
                    st.success(f"✅ Person erfolgreich zugewiesen!")
                    
                    # Auto-assign to similar faces if enabled
                    auto_assign_count = st.session_state.vector_store.auto_assign_similar_faces(
                        face_id, similarity_threshold=0.8
                    )
                    
                    if auto_assign_count > 0:
                        st.success(f"🔄 {auto_assign_count} ähnliche Gesichter (>80%) automatisch zugewiesen!")
                    
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
                value=existing_data.get('first_name', ''), 
                key="new_first_name"
            )
            middle_names = st.text_input(
                "Zweite Namen:", 
                value=existing_data.get('middle_names', ''),
                help="Z.B.: Moritz Luitpold Damian",
                key="new_middle_names"
            )
            last_name = st.text_input(
                "Nachname: *", 
                value=existing_data.get('last_name', ''), 
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
            name_parts = [p.strip() for p in [first_name, middle_names, last_name] if p.strip()]
            full_name = ' '.join(name_parts)
            st.success(f"**Vollständiger Name:** {full_name}")
        
        # Save new person button
        if st.button("💾 Person speichern", type="primary", use_container_width=True):
            if first_name.strip() or last_name.strip():
                # Create person data
                person_data = {
                    'first_name': first_name.strip(),
                    'middle_names': middle_names.strip(),
                    'last_name': last_name.strip(),
                    'birth_date': birth_date.strip(),
                    'birth_place': birth_place.strip(),
                    'notes': notes.strip()
                }
                
                # Assign person
                person_id = st.session_state.vector_store.assign_person_name(
                    face_id, person_data
                )
                
                if person_id:
                    st.success(f"✅ Person erfolgreich erstellt und zugewiesen!")
                    
                    # Auto-assign to similar faces if enabled
                    auto_assign_count = st.session_state.vector_store.auto_assign_similar_faces(
                        face_id, similarity_threshold=0.8
                    )
                    
                    if auto_assign_count > 0:
                        st.success(f"🔄 {auto_assign_count} ähnliche Gesichter (>80%) automatisch zugewiesen!")
                    
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
            if st.button("🗑️ Namen entfernen", use_container_width=True):
                if st.session_state.vector_store.remove_person_name(face_id):
                    st.success("✅ Name erfolgreich entfernt!")
                    _clear_name_assignment_session()
                    st.rerun()
                else:
                    st.error("❌ Fehler beim Entfernen des Namens")
    
    with col3:
        if st.button("❌ Abbrechen", use_container_width=True):
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
        if st.button("✅ Analyse schließen", key="close_popup_analysis", type="primary", use_container_width=True):
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
                                        st.success(f"**🎯 {similarity_percentage:.1f}%**")
                                        st.caption(f"🔒 Vertrauen: {confidence_score:.1f}%")
                                    elif similarity_percentage >= 50:
                                        st.info(f"**✅ {similarity_percentage:.1f}%**")
                                        st.caption(f"📊 Vertrauen: {confidence_score:.1f}%")
                                    elif similarity_percentage >= 30:
                                        st.warning(f"**⚠️ {similarity_percentage:.1f}%**")
                                        st.caption(f"📈 Vertrauen: {confidence_score:.1f}%")
                                    else:
                                        st.error(f"**❓ {similarity_percentage:.1f}%**")
                                        st.caption(f"🔍 Vertrauen: {confidence_score:.1f}%")
                                    
                                    st.write(f"**📁 {image_path.name}**")
                                    
                                    # Display person name if assigned
                                    face_metadata = face_data.get('metadata', {})
                                    person_name = face_metadata.get('full_name', '')
                                    if person_name:
                                        st.success(f"👤 **{person_name}**")
                                    
                                    # Action buttons in columns
                                    # Action buttons (stacked vertically to avoid nested columns)
                                    if st.button("🖼️ Ganzes Bild", key=f"full_img_{i}", help="Ganzes Bild mit markiertem Gesicht anzeigen"):
                                        show_full_image_with_face_box(image_path, location)
                                    if st.button("🧬 Analyse", key=f"analyze_face_{i}", help="Detaillierte Gesichtsanalyse: Alter, Geschlecht, Emotionen, Ethnie"):
                                        face_id = face_data.get('id', f"face_{i}")
                                        # Set analysis data and directly call modal
                                        st.session_state.analysis_image_path = image_path
                                        st.session_state.analysis_face_location = location
                                        st.session_state.analysis_face_id = face_id
                                        show_analysis_modal()
                                    if st.button("🏷️ Namen zuweisen", key=f"assign_name_{i}", help="Person einen Namen zuweisen"):
                                        face_id = face_data.get('face_id', face_data.get('id', f"face_{i}"))  # Try both face_id and id
                                        # Set name assignment data and directly call modal
                                        st.session_state.name_assign_face_id = face_id
                                        st.session_state.name_assign_image_path = image_path
                                        st.session_state.name_assign_face_location = location
                                        show_name_assignment_modal()
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
                                    
                                    # Display face thumbnail
                                    st.image(thumbnail, use_container_width=True)
                                    
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
                                    
                                    # Action buttons in columns
                                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                                    
                                    with btn_col1:
                                        # Add full image view button
                                        if st.button("🖼️ Ganzes Bild", key=f"full_img_tuple_{i}", help="Ganzes Bild mit markiertem Gesicht anzeigen"):
                                            show_full_image_with_face_box(image_path, (top, right, bottom, left))
                                    
                                    with btn_col2:
                                        # Add facial attribute analysis button
                                        if st.button("🧬 Analyse", key=f"analyze_tuple_{i}", help="Detaillierte Gesichtsanalyse: Alter, Geschlecht, Emotionen, Ethnie"):
                                            face_id = face_data.get('id', f"face_{i}")
                                            # Set analysis data and directly call modal
                                            st.session_state.analysis_image_path = image_path
                                            st.session_state.analysis_face_location = (top, right, bottom, left)
                                            st.session_state.analysis_face_id = face_id
                                            show_analysis_modal()
                                    
                                    with btn_col3:
                                        # Add name assignment button
                                        if st.button("🏷️ Namen", key=f"assign_name_tuple_{i}", help="Person einen Namen zuweisen"):
                                            face_id = face_data.get('face_id', face_data.get('id', f"face_{i}"))  # Try both face_id and id
                                            # Set name assignment data and directly call modal
                                            st.session_state.name_assign_face_id = face_id
                                            st.session_state.name_assign_image_path = image_path
                                            st.session_state.name_assign_face_location = (top, right, bottom, left)
                                            show_name_assignment_modal()
                                        
                                except:
                                    # Fallback: show full image
                                    st.image(image, use_container_width=True)
                                    st.write(f"**Similarity:** {face_data['similarity']:.3f}")
                                    
                                    # Add full image view button  
                                    if st.button("🖼️ Ganzes Bild", key=f"full_img_fallback_{i}", help="Ganzes Bild anzeigen"):
                                        with st.expander(f"�️ Vollbild: {image_path.name}", expanded=True):
                                            st.image(image, caption=f"Vollbild - {image_path.name}", use_container_width=True)
                        else:
                            st.image(image, use_container_width=True)
                            
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
                                with st.expander(f"🖼️ Vollbild: {image_path.name}", expanded=True):
                                    st.image(image, caption=f"Vollbild - {image_path.name}", use_container_width=True)
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
                    for metadata in existing_data['metadatas']:
                        if 'image_path' in metadata:
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
                    use_container_width=True,
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
                    if st.button("🔍 View Face Gallery", use_container_width=True):
                        st.session_state.page = "Face Gallery"
                        st.rerun()
                
                with col2:
                    if st.button("📊 Analyze Results", use_container_width=True):
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
        if original_dir:
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
        for metadata in metadatas:
            if 'image_path' in metadata:
                img_path = metadata['image_path']
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
                            st.image(original_thumbnail, caption="Original Image", use_container_width=True)
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
                        st.image(thumbnail, use_container_width=True)
                        
                        # Action buttons in columns
                        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                        
                        with btn_col1:
                            # Add full image view button
                            if st.button("🖼️ Ganzes Bild", key=f"gallery_full_{face_id}", help="Ganzes Bild mit markiertem Gesicht anzeigen"):
                                show_full_image_with_face_box(Path(image_path), location_str)
                        
                        with btn_col2:
                            # Add facial attribute analysis button
                            if st.button("🧬 Analyse", key=f"gallery_analyze_{face_id}", help="Detaillierte Gesichtsanalyse: Alter, Geschlecht, Emotionen, Ethnie"):
                                # Set analysis data and directly call modal
                                st.session_state.analysis_image_path = Path(image_path)
                                st.session_state.analysis_face_location = location_str
                                st.session_state.analysis_face_id = face_id
                                show_analysis_modal()
                        
                        with btn_col3:
                            # Add name assignment button
                            if st.button("🏷️ Namen", key=f"gallery_assign_name_{face_id}", help="Person einen Namen zuweisen"):
                                # Set name assignment data and directly call modal
                                st.session_state.name_assign_face_id = face_id
                                st.session_state.name_assign_image_path = image_path  # Keep as string to avoid path issues
                                st.session_state.name_assign_face_location = location_str
                                show_name_assignment_modal()
                        
                        with btn_col4:
                            # Add delete button with enhanced confirmation
                            if not compact:  # Only show delete in full view, not compact
                                if st.button("🗑️ Löschen", key=f"gallery_delete_{face_id}", help="Dieses falsch erkannte Gesicht löschen", type="secondary"):
                                    # Use a more robust confirmation system
                                    confirm_key = f'confirm_delete_{face_id}'
                                    
                                    if st.session_state.get(confirm_key, False):
                                        # Perform deletion
                                        with st.spinner("Lösche Gesicht..."):
                                            if delete_face_from_gallery(face_id):
                                                st.success(f"✅ Gesicht erfolgreich gelöscht!")
                                                # Clean up confirmation state
                                                if confirm_key in st.session_state:
                                                    del st.session_state[confirm_key]
                                                # Clear cache and refresh
                                                st.cache_data.clear()
                                                st.rerun()
                                            else:
                                                st.error(f"❌ Fehler beim Löschen des Gesichts")
                                                if confirm_key in st.session_state:
                                                    del st.session_state[confirm_key]
                                    else:
                                        # Request confirmation
                                        st.session_state[confirm_key] = True
                                        st.warning(f"⚠️ **{face_id[:8]}...** wirklich löschen? Nochmals klicken zum Bestätigen.")
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
        # Get all persons from database
        persons = st.session_state.vector_store.get_all_persons()
        
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
                    st.write(f"**Person ID:** `{person['person_id']}`")
                
                with col2:
                    st.write(f"**Anzahl Gesichter:** {person['face_count']}")
                    if st.button(f"🔍 Alle Gesichter anzeigen", key=f"show_faces_{person['person_id']}"):
                        st.session_state.selected_person = person['person_id']
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Namen löschen", key=f"remove_person_{person['person_id']}",
                                help="Entfernt den Namen von ALLEN Gesichtern dieser Person"):
                        if st.session_state.get(f'confirm_remove_{person["person_id"]}', False):
                            # Remove names from all faces of this person
                            removed_count = 0
                            for face_id in person['face_ids']:
                                if st.session_state.vector_store.remove_person_name(face_id):
                                    removed_count += 1
                            
                            st.success(f"✅ Name von {removed_count} Gesichtern entfernt!")
                            st.session_state[f'confirm_remove_{person["person_id"]}'] = False
                            st.rerun()
                        else:
                            st.session_state[f'confirm_remove_{person["person_id"]}'] = True
                            st.warning("⚠️ Nochmals klicken zum Bestätigen!")
                            st.rerun()
        
        # Display faces for selected person
        if st.session_state.get('selected_person'):
            person_id = st.session_state.selected_person
            person_faces = st.session_state.vector_store.get_faces_by_person_id(person_id)
            
            if person_faces:
                st.markdown("---")
                selected_person_name = next((p['full_name'] for p in persons if p['person_id'] == person_id), "Unbekannt")
                st.subheader(f"🔍 Alle Gesichter von **{selected_person_name}**")
                
                # Display faces in grid
                cols = st.columns(4)
                for idx, face in enumerate(person_faces):
                    with cols[idx % 4]:
                        metadata = face['metadata']
                        image_path = metadata.get('image_path', '')
                        face_location = metadata.get('location', '')  # Use 'location' instead of 'face_location'
                        
                        try:
                            # Load and display face thumbnail
                            full_image = load_and_preprocess_image(image_path)
                            if full_image is not None and face_location:
                                coords = face_location.split(',')
                                if len(coords) == 4:
                                    top, right, bottom, left = map(int, coords)
                                    face_crop = full_image[top:bottom, left:right]
                                    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                                        st.image(face_crop, caption=f"Face {face['face_id'][:8]}...", width=150)
                        except Exception as e:
                            st.error(f"Fehler beim Laden: {e}")
                        
                        # Face info
                        st.text(f"ID: {face['face_id'][:12]}...")
                        st.text(f"Bild: {Path(image_path).name}")
                        
                        # Remove individual face name button
                        if st.button("🗑️ Name entfernen", key=f"remove_face_{face['face_id']}"):
                            if st.session_state.vector_store.remove_person_name(face['face_id']):
                                st.success("✅ Name entfernt!")
                                st.rerun()
                            else:
                                st.error("❌ Fehler beim Entfernen")
                
                if st.button("❌ Ansicht schließen"):
                    del st.session_state.selected_person
                    st.rerun()
    
    except Exception as e:
        st.error(f"Fehler beim Laden der Name Gallery: {e}")

def live_camera_page():
    """
    Vereinfachte Live-Kamera mit stabilem Video-Feed
    """
    st.header("📹 Live-Kamera")
    st.markdown("**Echtzeit-Kamera mit Gesichtserkennung**")
    
    # Check for available modules
    if not LIVE_CAMERA_AVAILABLE:
        st.warning("⚠️ Live-Kamera benötigt MediaPipe und TensorFlow")
        st.code("pip install mediapipe tensorflow", language="bash")
        st.info("Nach Installation die Anwendung neu starten")
        return
    
    # Initialize session state
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'camera_source' not in st.session_state:
        st.session_state.camera_source = 0
    if 'show_fps' not in st.session_state:
        st.session_state.show_fps = True
    if 'detect_faces' not in st.session_state:
        st.session_state.detect_faces = True
    if 'identify_persons' not in st.session_state:
        st.session_state.identify_persons = False
        
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Einstellungen")
        
        # Only show camera 0 by default to avoid confusion
        camera_source = st.selectbox("Kamera:", [0], index=0)
        st.session_state.camera_source = camera_source
        
        show_fps = st.checkbox("FPS anzeigen", value=st.session_state.show_fps)
        st.session_state.show_fps = show_fps
        
        detect_faces = st.checkbox("Gesichter erkennen", value=st.session_state.detect_faces)
        st.session_state.detect_faces = detect_faces
        
        if detect_faces:
            identify_persons = st.checkbox("Personen identifizieren", value=st.session_state.identify_persons)
            st.session_state.identify_persons = identify_persons
    
    with col2:
        st.subheader("🎮 Steuerung")
        
        # Start/Stop buttons
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("▶️ Kamera starten", disabled=st.session_state.camera_active, use_container_width=True):
                st.session_state.camera_active = True
                st.rerun()
        
        with col_stop:
            if st.button("⏹️ Kamera stoppen", disabled=not st.session_state.camera_active, use_container_width=True):
                st.session_state.camera_active = False
                st.rerun()
    
    st.divider()
    
    # Camera display
    if st.session_state.camera_active:
        st.subheader("📺 Live-Feed")
        
        # Create placeholder for video
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Initialize camera
        try:
            cap = cv2.VideoCapture(st.session_state.camera_source)
            
            if not cap.isOpened():
                st.error(f"❌ Konnte Kamera {st.session_state.camera_source} nicht öffnen")
                st.session_state.camera_active = False
                return
            
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # MediaPipe setup if available
            mp_face_detection = None
            mp_drawing = None
            face_detection = None
            
            if LIVE_CAMERA_AVAILABLE and st.session_state.detect_faces:
                try:
                    import mediapipe as mp
                    mp_face_detection = mp.solutions.face_detection
                    mp_drawing = mp.solutions.drawing_utils
                    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
                except:
                    pass
            
            # FPS tracking
            fps_counter = 0
            start_time = time.time()
            last_fps_time = start_time
            current_fps = 0
            
            # Main camera loop with automatic refresh
            frame_count = 0
            while st.session_state.camera_active:
                ret, frame = cap.read()
                
                if not ret:
                    status_placeholder.error("❌ Fehler beim Lesen der Kamera")
                    break
                
                frame_count += 1
                fps_counter += 1
                
                # Calculate FPS every second
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    current_fps = fps_counter / (current_time - last_fps_time)
                    fps_counter = 0
                    last_fps_time = current_time
                
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = rgb_frame.copy()
                
                faces_detected = 0
                persons_identified = 0
                
                # Face detection
                if face_detection is not None:
                    try:
                        results = face_detection.process(rgb_frame)
                        
                        if results.detections:
                            faces_detected = len(results.detections)
                            
                            for detection in results.detections:
                                # Draw bounding box
                                bbox = detection.location_data.relative_bounding_box
                                h, w, _ = processed_frame.shape
                                
                                x = int(bbox.xmin * w)
                                y = int(bbox.ymin * h)
                                width = int(bbox.width * w)
                                height = int(bbox.height * h)
                                
                                # Draw rectangle
                                cv2.rectangle(processed_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                                
                                # Add confidence score
                                confidence = detection.score[0]
                                cv2.putText(processed_frame, f"{confidence:.2f}", 
                                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                                # Person identification (simplified)
                                if st.session_state.identify_persons:
                                    try:
                                        # Extract face region
                                        face_crop = rgb_frame[y:y+height, x:x+width]
                                        if face_crop.size > 0:
                                            # Detect with face recognition engine
                                            face_locations = st.session_state.face_engine.detect_faces(face_crop)
                                            if face_locations:
                                                embedding = st.session_state.face_engine.extract_face_embedding(
                                                    face_crop, face_locations[0]
                                                )
                                                if embedding is not None:
                                                    # Search for similar faces
                                                    similar_faces = st.session_state.vector_store.search_similar_faces(
                                                        embedding, n_results=1, min_similarity=0.75
                                                    )
                                                    if similar_faces:
                                                        person_name = similar_faces[0]['metadata'].get('full_name', 'Unbekannt')
                                                        if person_name and person_name != 'Unbekannt':
                                                            cv2.putText(processed_frame, person_name, 
                                                                      (x, y + height + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                                                      0.6, (255, 0, 0), 2)
                                                            persons_identified += 1
                                    except:
                                        pass
                    except:
                        pass
                
                # Add FPS overlay
                if st.session_state.show_fps:
                    cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add status info
                cv2.putText(processed_frame, f"Gesichter: {faces_detected}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if st.session_state.identify_persons:
                    cv2.putText(processed_frame, f"Identifiziert: {persons_identified}", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                video_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
                
                # Update status
                status_text = f"🎥 Live-Feed aktiv | Frame {frame_count}"
                if st.session_state.show_fps:
                    status_text += f" | FPS: {current_fps:.1f}"
                if faces_detected > 0:
                    status_text += f" | {faces_detected} Gesicht(er)"
                if persons_identified > 0:
                    status_text += f" | {persons_identified} identifiziert"
                
                status_placeholder.info(status_text)
                
                # Small delay to prevent overwhelming the browser
                time.sleep(0.03)  # ~30 FPS limit
                
                # Check if we should stop (recheck session state)
                if not getattr(st.session_state, 'camera_active', False):
                    break
            
            # Cleanup
            cap.release()
            video_placeholder.empty()
            status_placeholder.success("✅ Kamera gestoppt")
            
        except Exception as e:
            st.error(f"❌ Kamera-Fehler: {e}")
            st.session_state.camera_active = False
            
    else:
        st.info("📹 Kamera gestoppt. Klicken Sie 'Kamera starten' um zu beginnen.")
        
        # Show preview info
        st.markdown("""
        ### 🚀 Verfügbare Features:
        - **Echtzeit-Video**: Direkter Kamera-Feed
        - **Gesichtserkennung**: Automatische Gesichtserkennung mit MediaPipe
        - **Personen-Identifikation**: Abgleich mit der Gesichtsdatenbank
        - **FPS-Anzeige**: Performance-Monitoring
        - **Stabile Wiedergabe**: Ohne Flackern oder Unterbrechungen
        """")

# ===== ENDE DER DATEI =====

if __name__ == "__main__":
    main()
        
        if camera_source == "IP-Kamera":
            ip_camera_url = st.text_input(
                "IP-Kamera URL:",
                placeholder="http://192.168.1.100:8080/video",
                help="Geben Sie die URL Ihrer IP-Kamera ein"
            )
            camera_source = ip_camera_url if ip_camera_url else 0
    
    with col2:
        enable_face_recognition = st.checkbox(
            "👤 Personenerkennung", 
            value=True,
            help="Aktiviert die Erkennung bekannter Personen aus der Datenbank"
        )
        
        enable_emotion_detection = st.checkbox(
            "😊 Emotionserkennung",
            value=True,
            help="Aktiviert die Erkennung von Emotionen"
        )
        
        enable_pose_detection = st.checkbox(
            "🧍 Pose-Erkennung",
            value=True,
            help="Aktiviert die Körperhaltungserkennung"
        )
    
    with col3:
        enable_hand_detection = st.checkbox(
            "✋ Hand-Erkennung",
            value=True,
            help="Aktiviert die Handerkennung"
        )
        
        enable_face_mesh = st.checkbox(
            "🕸️ Gesichtsmesh",
            value=False,
            help="Zeigt detailliertes Gesichtsmesh (kann Performance beeinträchtigen)"
        )
        
        similarity_threshold = st.slider(
            "Ähnlichkeits-Schwellwert",
            min_value=0.6,
            max_value=0.95,
            value=0.75,
            step=0.05,
            help="Minimale Ähnlichkeit für Personenerkennung"
        )
    
    st.divider()
    
    # Simplified fallback camera test
    st.subheader("🎥 Einfacher Kamera-Test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📷 Kamera testen", type="primary"):
            try:
                # Simple camera test
                cap = cv2.VideoCapture(camera_source)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        st.success("✅ Kamera erfolgreich getestet!")
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="Test-Frame von der Kamera", width=400)
                        
                        # Show camera info
                        st.info(f"📊 **Kamera-Info:**\n- Auflösung: {frame.shape[1]}x{frame.shape[0]}\n- Kanäle: {frame.shape[2]}")
                    else:
                        st.error("❌ Kein Frame von der Kamera erhalten")
                    cap.release()
                else:
                    st.error("❌ Kamera konnte nicht geöffnet werden")
            except Exception as e:
                st.error(f"❌ Kamera-Fehler: {e}")
    
    with col2:
        st.info("""
        **Warum sehen Sie mehrere Kameras?**
        
        Das System erkennt alle verfügbaren Video-Eingänge:
        - Kamera 0: Hauptkamera (meist die eingebaute)
        - Kamera 1,2: Virtuelle oder alternative Eingänge
        
        **Tipp:** Testen Sie Kamera 0 zuerst - das ist meist die richtige.
        """)
    
    st.markdown("---")
    
    # Information about full features
    st.subheader("🚀 Vollständige Live-AI-Features verfügbar!")
    st.success("✅ Alle Module sind installiert - Live-Kamera sollte funktionieren")
    
    # Show available features
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Verfügbare Features:**")
        st.success("✅ 👤 Echtzeit-Personenerkennung")
        st.success("✅ 😊 Emotionserkennung (extern + landmark-basiert)")  
        st.success("✅ 🧍 Pose-Erkennung (Körperhaltung)")
        st.success("✅ ✋ Hand-Tracking")
    
    with col2:
        st.write("**Erweiterte Features:**")
        st.success("✅ 🕸️ Detailliertes Gesichtsmesh")
        st.success("✅ 📊 Live-Statistiken")
        st.success("✅ 📸 Screenshot-Funktion")
        st.success("✅ 🎯 Konfigurierbare Erkennungsschwellwerte")
    
    # Show system capabilities
    st.subheader("💻 System-Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Verfügbare Module:**")
        st.success("✅ OpenCV (cv2) verfügbar") 
        
        if MEDIAPIPE_AVAILABLE:
            st.success("✅ MediaPipe verfügbar")
        else:
            st.error("❌ MediaPipe nicht verfügbar")
            
        if TENSORFLOW_AVAILABLE:
            st.success("✅ TensorFlow verfügbar")
        else:
            st.error("❌ TensorFlow nicht verfügbar")
    
    with col2:
        st.write("**Face Recognition System:**")
        if st.session_state.face_engine:
            st.success("✅ Face Recognition Engine geladen")
        else:
            st.warning("⚠️ Face Recognition Engine nicht verfügbar")
        
        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_collection_stats()
            if "error" not in stats:
                st.write(f"• Gesichter: {stats.get('total_faces', 0)}")
                st.write(f"• Personen: {stats.get('total_persons', 0)}")
            else:
                st.error("❌ Datenbank-Fehler")
        else:
            st.warning("⚠️ Vector Store nicht verfügbar")
    
    # Link to original face.py
    st.markdown("---")
    st.subheader("� Alternative: Standalone Live-Kamera")
    st.info("""
    Sie können auch die eigenständige Live-Kamera verwenden:
    
    ```bash
    cd live/
    python face.py
    ```
    
    Diese Version läuft als separate OpenCV-Anwendung mit allen Features und ist oft stabiler.
    """)
    
    if st.button("📖 Live-Kamera Dokumentation anzeigen"):
        st.subheader("📋 Live-Kamera Features (face.py)")
        st.markdown("""
        **Vollständige Feature-Liste der Standalone-Version:**
        
        🎯 **Gesichtserkennung:**
        - MediaPipe Face Detection (hochpräzise)
        - Personenerkennung aus Datenbank
        - Ähnlichkeits-Matching mit konfigurierbaren Schwellwerten
        - UTF-8 sichere Namensdisplay (Umlaute unterstützt)
        
        😊 **Emotionserkennung:**
        - Externes TensorFlow-Modell (falls verfügbar)
        - Landmark-basierte Emotionserkennung als Fallback
        - 7 Emotionen: angry, disgust, fear, happy, sad, surprise, neutral
        - Echtzeit-Konfidenz-Bewertung
        
        🧍 **Pose-Erkennung:**
        - MediaPipe Pose für Körperhaltungserkennung
        - 33 Körperpunkte-Tracking
        - Echtzeit-Visualisierung
        
        ✋ **Hand-Tracking:**
        - Bis zu 4 Hände gleichzeitig
        - 21 Handpunkte pro Hand
        - Handposition-Anzeige
        
        🕸️ **Gesichtsmesh:**
        - 468 Gesichtspunkte
        - Gesichtsausrichtung (Links/Rechts/Mitte, Oben/Unten)
        - Mund-Öffnungs-Erkennung
        - Detaillierte Landmark-Analyse
        
        ⚙️ **System-Features:**
        - Multi-Kamera-Unterstützung (lokal + IP-Kameras)
        - Threaded Processing für optimale Performance
        - Frame-Counter und Live-Statistiken
        - Stabilisierte Personenerkennung
        - Memory-optimierte Queues
        
        🎮 **Steuerung:**
        - 'q' drücken zum Beenden
        - Fenster schließen für einzelne Kamera
        - Keyboard Interrupt (Ctrl+C) unterstützt
        """)
        
        st.code("""
# Beispiel für IP-Kamera Integration in face.py:

camera_sources = [
    0,  # Lokale Webcam
    "http://192.168.1.100:8080/video",  # IP-Kamera
    "rtsp://user:pass@192.168.1.101/stream1"  # RTSP Stream
]
        """, language="python")

def detect_available_cameras():
    """Detect actually available cameras"""
    available_cameras = []
    for i in range(5):  # Check first 5 camera indices
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    available_cameras.append(i)
            cap.release()
        except:
            pass
    return available_cameras if available_cameras else [0]  # Fallback to 0

def start_live_camera_streamlit(camera_source, enable_face_recognition, enable_emotion_detection, 
                               enable_pose_detection, enable_hand_detection, enable_face_mesh, 
                               similarity_threshold):
    """Start the live camera with specified configuration"""
    
    # Load emotion model if needed
    if enable_emotion_detection and st.session_state.emotion_model is None and TENSORFLOW_AVAILABLE:
        try:
            if TENSORFLOW_AVAILABLE:
                import tensorflow as tf
                from tensorflow.keras.models import load_model
                st.session_state.emotion_model = load_model("models/fer_trained.h5")
                st.success("✅ Externes Emotionsmodell geladen")
        except Exception as e:
            st.info(f"ℹ️ Externes Emotionsmodell nicht verfügbar: {e}")
            st.info("Verwende Landmark-basierte Emotionserkennung")
    
    # Initialize camera components
    try:
        # Create queue and thread
        st.session_state.live_camera_queue = queue.Queue(maxsize=2)
        st.session_state.live_stop_flag = threading.Event()
        
        # Start camera thread
        st.session_state.live_camera_thread = threading.Thread(
            target=live_camera_worker_streamlit,
            args=(
                camera_source,
                st.session_state.live_camera_queue,
                st.session_state.live_stop_flag,
                st.session_state.face_engine if enable_face_recognition else None,
                st.session_state.vector_store if enable_face_recognition else None,
                st.session_state.emotion_model if enable_emotion_detection else None,
                enable_pose_detection,
                enable_hand_detection,
                enable_face_mesh,
                similarity_threshold
            ),
            daemon=True
        )
        
        st.session_state.live_camera_thread.start()
        st.session_state.live_camera_active = True
        
        st.success("🎬 Kamera gestartet!")
        time.sleep(0.5)  # Give camera time to initialize
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Fehler beim Starten der Kamera: {e}")

def stop_live_camera_streamlit():
    """Stop the live camera"""
    try:
        if st.session_state.live_stop_flag:
            st.session_state.live_stop_flag.set()
        
        if st.session_state.live_camera_thread and st.session_state.live_camera_thread.is_alive():
            st.session_state.live_camera_thread.join(timeout=2.0)
        
        st.session_state.live_camera_active = False
        st.session_state.live_camera_thread = None
        st.session_state.live_camera_queue = None
        st.session_state.live_stop_flag = None
        
        st.success("⏹️ Kamera gestoppt!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Fehler beim Stoppen der Kamera: {e}")

def take_screenshot_streamlit():
    """Take a screenshot of the current frame"""
    try:
        if st.session_state.live_camera_queue and not st.session_state.live_camera_queue.empty():
            frame_data = st.session_state.live_camera_queue.get_nowait()
            if frame_data:
                frame, _ = frame_data
                
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = Path(f"screenshot_{timestamp}.jpg")
                cv2.imwrite(str(screenshot_path), frame)
                
                st.success(f"📸 Screenshot gespeichert: {screenshot_path}")
                
                # Show thumbnail
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Screenshot: {screenshot_path}", width=300)
                
        else:
            st.warning("⚠️ Kein Frame verfügbar für Screenshot")
            
    except Exception as e:
        st.error(f"❌ Fehler beim Erstellen des Screenshots: {e}")

def display_live_camera_feed():
    """Display the live camera feed with auto-refresh"""
    st.subheader("� Live-Feed")
    
    # Live stats display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Frames verarbeitet", st.session_state.live_stats['frames_processed'])
    with col2:
        st.metric("Gesichter erkannt", st.session_state.live_stats['faces_detected'])
    with col3:
        st.metric("Personen identifiziert", st.session_state.live_stats['persons_identified'])
    with col4:
        st.metric("Emotionen erkannt", st.session_state.live_stats['emotions_detected'])
    
    # Video feed with auto-refresh
    video_placeholder = st.empty()
    analysis_placeholder = st.empty()
    
    # Auto-refresh mechanism
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    
    # Get and display latest frame
    frame_data = get_latest_frame()
    if frame_data:
        frame, analysis_data = frame_data
        
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display frame
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Display analysis data
        display_analysis_data(analysis_data, analysis_placeholder)
        
        # Increment refresh counter to force updates
        st.session_state.refresh_counter += 1
    else:
        video_placeholder.info("⏳ Warten auf Kamera-Feed...")
    
    # Auto-refresh every 100ms
    time.sleep(0.1)
    st.rerun()

def get_latest_frame():
    """Get the latest frame from the camera queue"""
    try:
        if st.session_state.live_camera_queue and not st.session_state.live_camera_queue.empty():
            # Get all frames in queue, keep only the latest
            latest_frame = None
            while not st.session_state.live_camera_queue.empty():
                try:
                    latest_frame = st.session_state.live_camera_queue.get_nowait()
                except queue.Empty:
                    break
            return latest_frame
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen des Frames: {e}")
    return None

def display_analysis_data(analysis_data, placeholder):
    """Display analysis data from camera processing"""
    if analysis_data:
        with placeholder.container():
            st.subheader("🔍 Aktuelle Analyse")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if analysis_data.get('faces'):
                    st.write("**Erkannte Gesichter:**")
                    for i, face in enumerate(analysis_data['faces']):
                        if face.get('person_name'):
                            st.write(f"• {face['person_name']} ({face.get('similarity', 0)*100:.1f}%)")
                        else:
                            st.write(f"• Unbekannte Person {i+1}")
                        
                        if face.get('emotion'):
                            st.write(f"  Emotion: {face['emotion']} ({face.get('emotion_confidence', 0):.2f})")
            
            with col2:
                if analysis_data.get('hands'):
                    st.write(f"**Hände erkannt:** {len(analysis_data['hands'])}")
                
                if analysis_data.get('pose_detected'):
                    st.write("**Körperhaltung:** ✅ Erkannt")
                
                if analysis_data.get('face_orientations'):
                    st.write("**Gesichtsausrichtungen:**")
                    for orientation in analysis_data['face_orientations']:
                        st.write(f"• {orientation}")

def display_system_status():
    """Display system status when camera is not active"""
    st.info("📷 Klicken Sie auf 'Kamera starten' um den Live-Feed zu beginnen")
    
    # System status
    st.subheader("💻 System-Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Verfügbare Module:**")
        st.success("✅ OpenCV (cv2) verfügbar") 
        
        if MEDIAPIPE_AVAILABLE:
            st.success("✅ MediaPipe verfügbar")
        else:
            st.error("❌ MediaPipe nicht verfügbar")
            
        if TENSORFLOW_AVAILABLE:
            st.success("✅ TensorFlow verfügbar")
        else:
            st.error("❌ TensorFlow nicht verfügbar")
    
    with col2:
        st.write("**Face Recognition System:**")
        if st.session_state.face_engine:
            st.success("✅ Face Recognition Engine geladen")
        else:
            st.warning("⚠️ Face Recognition Engine nicht verfügbar")
        
        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_collection_stats()
            if "error" not in stats:
                st.write(f"• Gesichter: {stats.get('total_faces', 0)}")
                st.write(f"• Personen: {stats.get('total_persons', 0)}")
            else:
                st.error("❌ Datenbank-Fehler")
        else:
            st.warning("⚠️ Vector Store nicht verfügbar")
    
    # Link to original face.py
    st.markdown("---")
    st.subheader("🔗 Alternative: Standalone Live-Kamera")
    st.info("""
    Sie können auch die eigenständige Live-Kamera verwenden:
    
    ```bash
    cd live/
    python face.py
    ```
    
    Diese Version läuft als separate OpenCV-Anwendung mit allen Features.
    """)

# ================================
# LIVE CAMERA WORKER FUNCTIONS
# ================================

# Constants for live camera functionality
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SIMILARITY_THRESHOLD = 0.75  # Default similarity threshold

def predict_emotion_from_landmarks_streamlit(face_landmarks, w, h):
    """
    Emotion recognition based on facial landmarks - synchronized with face.py
    """
    try:
        if not face_landmarks or not face_landmarks.landmark:
            return "neutral", 0.5
            
        lm = face_landmarks.landmark
        
        # Improved mouth analysis
        mouth_top = lm[13].y * h
        mouth_bottom = lm[14].y * h
        mouth_left = lm[308].x * w  
        mouth_right = lm[78].x * w
        mouth_center_left = lm[291].y * h
        mouth_center_right = lm[61].y * h
        
        mouth_height = abs(mouth_bottom - mouth_top)
        mouth_width = abs(mouth_right - mouth_left) 
        mouth_curve = (mouth_center_left + mouth_center_right) / 2 - mouth_top
        
        # Improved eye analysis  
        left_eye_top = lm[159].y * h
        left_eye_bottom = lm[145].y * h
        right_eye_top = lm[386].y * h  
        right_eye_bottom = lm[374].y * h
        
        left_eye_height = abs(left_eye_bottom - left_eye_top)
        right_eye_height = abs(right_eye_bottom - right_eye_top)
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        # Improved eyebrow analysis
        left_brow = lm[70].y * h
        right_brow = lm[300].y * h
        left_eye_center = (left_eye_top + left_eye_bottom) / 2
        right_eye_center = (right_eye_top + right_eye_bottom) / 2
        
        left_brow_distance = left_brow - left_eye_center
        right_brow_distance = right_brow - right_eye_center
        avg_brow_distance = (left_brow_distance + right_brow_distance) / 2
        
        # Emotion recognition with realistic thresholds
        emotions_confidence = {}
        
        # Happy: Mouth curved upward AND wider
        if mouth_curve > 1.5 and mouth_height > 3:
            happiness_score = min(0.9, 0.5 + (mouth_curve / 10) + (mouth_height / 30))
            emotions_confidence["happy"] = happiness_score
        else:
            emotions_confidence["happy"] = 0.1
            
        # Sad: Mouth curved downward OR very small eyes
        if mouth_curve < -1.0 or avg_eye_height < 5:
            sadness_score = min(0.85, 0.4 + abs(mouth_curve) / 8)
            emotions_confidence["sad"] = sadness_score
        else:
            emotions_confidence["sad"] = 0.1
            
        # Surprise: Large eyes AND open mouth
        if avg_eye_height > 8 and mouth_height > 10:
            surprise_score = min(0.9, 0.5 + (avg_eye_height / 15) + (mouth_height / 25))
            emotions_confidence["surprise"] = surprise_score
        else:
            emotions_confidence["surprise"] = 0.1
            
        # Angry: Contracted eyebrows AND tense mouth
        if avg_brow_distance > -15 and mouth_height < 4:
            angry_score = min(0.85, 0.4 + abs(avg_brow_distance + 15) / 10)
            emotions_confidence["angry"] = angry_score
        else:
            emotions_confidence["angry"] = 0.15
            
        # Fear: Large eyes AND high eyebrows
        if avg_eye_height > 7 and avg_brow_distance < -18:
            fear_score = min(0.8, 0.4 + (avg_eye_height / 20))
            emotions_confidence["fear"] = fear_score
        else:
            emotions_confidence["fear"] = 0.1
            
        # Disgust: Slightly distorted mouth
        if mouth_curve < -0.5 and mouth_curve > -2.0 and mouth_height < 3:
            emotions_confidence["disgust"] = 0.6
        else:
            emotions_confidence["disgust"] = 0.1
            
        # Neutral: Balanced values - STANDARD for most cases
        neutral_factors = []
        if abs(mouth_curve) < 1.5:  # Neutral mouth
            neutral_factors.append(0.3)
        if 5 < avg_eye_height < 9:  # Normal eye size
            neutral_factors.append(0.3)
        if -20 < avg_brow_distance < -10:  # Normal eyebrows
            neutral_factors.append(0.3)
        if 3 < mouth_height < 8:  # Normal mouth
            neutral_factors.append(0.2)
            
        neutral_score = sum(neutral_factors) + 0.4  # Base neutral score
        emotions_confidence["neutral"] = min(0.95, neutral_score)
        
        # Find best emotion
        best_emotion = max(emotions_confidence.keys(), key=lambda k: emotions_confidence[k])
        best_confidence = emotions_confidence[best_emotion]
        
        return best_emotion, best_confidence
        
    except Exception as e:
        print(f"[ERROR] Landmark emotion detection: {e}")
        return "neutral", 0.6

def predict_emotion_external_streamlit(face_gray, model):
    """
    External emotion model prediction - synchronized with face.py
    """
    if model is not None:
        try:
            # Try external model
            img = cv2.resize(face_gray, (48, 48))
            img = cv2.equalizeHist(img)
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=(0, -1))
            
            preds = model.predict(img, verbose=0)[0]
            top_indices = np.argsort(preds)[-2:][::-1]
            top_emotion_idx = top_indices[0]
            top_confidence = float(preds[top_emotion_idx])
            
            if top_confidence < 0.6:
                second_best_confidence = float(preds[top_indices[1]])
                if top_confidence - second_best_confidence < 0.1:
                    return "unsicher", top_confidence
                    
            return EMOTIONS[top_emotion_idx], top_confidence
            
        except Exception as e:
            print(f"[ERROR] External emotion model error: {e}")
    
    # Fallback
    return "neutral", 0.5

def identify_face_streamlit(face_region, face_engine, vector_store, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Face identification - synchronized with face.py
    """
    try:
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        
        # Detect face in cropped region
        face_locations = face_engine.detect_faces(face_rgb)
        
        if not face_locations:
            return None
            
        # Use first face detected
        face_location = face_locations[0]
        
        # Extract face embedding
        embedding = face_engine.extract_face_embedding(face_rgb, face_location)
        
        if embedding is None:
            return None
            
        # Search for similar faces
        similar_faces = vector_store.search_similar_faces(
            embedding, 
            n_results=5,
            min_similarity=max(0.6, similarity_threshold - 0.15)
        )
        
        if similar_faces:
            best_match = similar_faces[0]
            best_similarity = best_match['similarity']
            
            if best_similarity < 0.65:
                return None
                
            person_data = best_match['metadata']
            
            # UTF-8 safe processing
            name = person_data.get('full_name', 'Unbekannt')
            first_name = person_data.get('first_name', '')
            last_name = person_data.get('last_name', '')
            
            # Ensure names are properly encoded
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='replace')
            if isinstance(first_name, bytes):
                first_name = first_name.decode('utf-8', errors='replace')
            if isinstance(last_name, bytes):
                last_name = last_name.decode('utf-8', errors='replace')
            
            person_info = {
                'name': name,
                'first_name': first_name,
                'last_name': last_name,
                'similarity': best_similarity,
                'confidence': best_match.get('confidence_score', 0),
                'person_id': person_data.get('person_id'),
                'face_id': best_match['face_id']
            }
            return person_info
            
        return None
        
    except Exception as e:
        print(f"[ERROR] Face identification error: {e}")
        return None

def get_face_orientation_streamlit(mesh_landmarks, w, h):
    """
    Get face orientation from landmarks - synchronized with face.py
    """
    try:
        lm = mesh_landmarks.landmark
        nose = np.array([lm[1].x*w, lm[1].y*h])
        l_eye = np.array([lm[263].x*w, lm[263].y*h])
        r_eye = np.array([lm[33].x*w, lm[33].y*h])
        chin = np.array([lm[152].x*w, lm[152].y*h])
        forehead = np.array([lm[10].x*w, lm[10].y*h])
        
        center_x = (l_eye[0] + r_eye[0]) / 2
        hor = "Mitte" if abs(nose[0] - center_x) < 0.02*w else ("Rechts" if nose[0] < center_x else "Links")
        
        center_y = (forehead[1] + chin[1]) / 2
        vert = "Mitte" if abs(nose[1] - center_y) < 0.025*h else ("Oben" if nose[1] < center_y else "Unten")
        
        return f"Kopf: {hor}, {vert}"
    except:
        return "Unbekannt"

def is_mouth_open_streamlit(mesh_landmarks, w, h):
    """
    Check if mouth is open - synchronized with face.py
    """
    try:
        lm = mesh_landmarks.landmark
        y_up = lm[13].y * h
        y_low = lm[14].y * h
        lip_dist = abs(y_low - y_up)
        return lip_dist > 0.035*h
    except:
        return False

def live_camera_worker_streamlit(camera_source, result_queue, stop_flag, face_engine=None, 
                                vector_store=None, emotion_model=None, enable_pose=True, 
                                enable_hands=True, enable_face_mesh=False, similarity_threshold=0.75):
    """
    Live camera worker thread for Streamlit - synchronized with face.py camera_worker
    """
    
    # Import MediaPipe inside function to avoid import errors
    if not MEDIAPIPE_AVAILABLE:
        print("[ERROR] MediaPipe not available for live camera")
        return
    
    import mediapipe as mp
    
    # Initialize MediaPipe components
    mp_hands_instance = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=4,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) if enable_hands else None
    
    mp_fd = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.7
    )
    
    mp_fm = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) if enable_face_mesh else None
    
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) if enable_pose else None
    
    mp_draw = mp.solutions.drawing_utils

    # Open camera
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {camera_source}")
        return

    print(f"[INFO] Live camera started with source: {camera_source}")
    frame_count = 0
    
    # Simplified stabilization for person recognition
    person_history = {}  # person_id -> [timestamps]
    recognition_stability_time = 0.5
    recognition_count_threshold = 1

    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] No frame from camera")
            time.sleep(0.1)
            continue

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        analysis_lines = []
        current_time = time.time()
        
        # Analysis data for Streamlit
        analysis_data = {
            'faces': [],
            'hands': [],
            'pose_detected': False,
            'face_orientations': []
        }
        
        # Face detection + emotions + person identification
        detected_faces_info = []
        
        try:
            det = mp_fd.process(rgb)
            if det.detections:
                for i, d in enumerate(det.detections):
                    bb = d.location_data.relative_bounding_box
                    x1 = int(bb.xmin * w)
                    y1 = int(bb.ymin * h)
                    x2 = int((bb.xmin + bb.width) * w)
                    y2 = int((bb.ymin + bb.height) * h)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    roi = frame[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        # Standard emotion recognition as fallback
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        emotion_label, emotion_prob = predict_emotion_external_streamlit(gray, emotion_model)
                        
                        # Store face info for later
                        face_info = {
                            'bbox': (x1, y1, x2, y2),
                            'roi': roi,
                            'emotion_fallback': (emotion_label, emotion_prob),
                            'person_info': None
                        }
                        
                        # Face identification with less strict stabilization
                        person_info = None
                        if face_engine and vector_store and roi.shape[0] > 50 and roi.shape[1] > 50:
                            try:
                                person_info = identify_face_streamlit(roi, face_engine, vector_store, similarity_threshold)
                                
                                if person_info:
                                    person_id = person_info['person_id']
                                    if person_id:
                                        # Very simple stabilization
                                        if person_id not in person_history:
                                            person_history[person_id] = []
                                        
                                        person_history[person_id].append(current_time)
                                        
                                        # Clean old entries
                                        person_history[person_id] = [
                                            t for t in person_history[person_id] 
                                            if current_time - t <= recognition_stability_time
                                        ]
                                        
                                        # Check if person is stable
                                        if len(person_history[person_id]) >= recognition_count_threshold:
                                            # Person is confirmed
                                            pass
                                        else:
                                            # Not stable enough, use as unconfirmed
                                            person_info['name'] += " (?)"
                                            
                            except Exception as e:
                                print(f"[ERROR] Person identification: {e}")
                        
                        face_info['person_info'] = person_info
                        detected_faces_info.append(face_info)
                        
        except Exception as e:
            print(f"[ERROR] Face detection: {e}")
            
        # Draw recognized faces IMMEDIATELY
        for i, face_info in enumerate(detected_faces_info):
            x1, y1, x2, y2 = face_info['bbox']
            person_info = face_info['person_info']
            emotion_label, emotion_prob = face_info['emotion_fallback']
            
            # Draw face rectangle
            color = (0, 255, 0) if person_info else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Display person name and emotion (UTF-8 safe)
            if person_info:
                name = person_info['name']
                similarity = person_info['similarity']
                
                main_label = f"{name} ({similarity*100:.0f}%)"
                
                # UTF-8 safe text output
                try:
                    cv2.putText(frame, main_label, (x1, y1-40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                except:
                    # Fallback without umlauts
                    safe_name = name.encode('ascii', errors='ignore').decode('ascii')
                    main_label = f"{safe_name} ({similarity*100:.0f}%)"
                    cv2.putText(frame, main_label, (x1, y1-40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Emotion label
                emotion_text = f"{emotion_label} {emotion_prob:.2f}"
                cv2.putText(frame, emotion_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                analysis_lines.append(f"Person: {name} ({similarity*100:.0f}%)")
                analysis_lines.append(f"Emotion: {emotion_label} ({emotion_prob:.2f})")
                
                # Add to analysis data
                analysis_data['faces'].append({
                    'person_name': name,
                    'similarity': similarity,
                    'emotion': emotion_label,
                    'emotion_confidence': emotion_prob
                })
                
                # Display additional info
                if person_info.get('person_id'):
                    cv2.putText(frame, f"ID: {person_info['person_id'][:8]}", (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                               
            else:
                # Unknown person
                label = f"Unbekannt - {emotion_label} {emotion_prob:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                analysis_lines.append(f"Unbekannte Person: {emotion_label} ({emotion_prob:.2f})")
                
                analysis_data['faces'].append({
                    'person_name': None,
                    'emotion': emotion_label,
                    'emotion_confidence': emotion_prob
                })
            
        # FaceMesh for additional analysis
        if enable_face_mesh and mp_fm:
            try:
                mesh = mp_fm.process(rgb)
                if mesh.multi_face_landmarks:
                    for j, fl in enumerate(mesh.multi_face_landmarks):
                        # Draw FaceMesh as additional information
                        mp_draw.draw_landmarks(
                            frame, fl, 
                            mp.solutions.face_mesh.FACEMESH_TESSELATION,
                            mp_draw.DrawingSpec((0, 255, 0), 1, 1), 
                            mp_draw.DrawingSpec((0, 180, 0), 1)
                        )
                        
                        # Analyze emotions but don't overwrite main recognition
                        landmark_emotion, landmark_confidence = predict_emotion_from_landmarks_streamlit(fl, w, h)
                        
                        # Show additional FaceMesh information
                        orient = get_face_orientation_streamlit(fl, w, h)
                        mouth_open = is_mouth_open_streamlit(fl, w, h)
                        mouth_lab = "Mund offen" if mouth_open else "Mund zu"
                        nose = fl.landmark[1]
                        cx, cy = int(nose.x*w), int(nose.y*h)
                        
                        # Only show additional info
                        additional_info = f"{orient}, {mouth_lab}"
                        cv2.putText(frame, additional_info, (cx+15, cy+25+j*20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (0, 200, 255) if mouth_open else (100, 180, 255), 1)
                        
                        analysis_lines.append(f"Zusatz: {orient}, {mouth_lab}")
                        analysis_data['face_orientations'].append(f"{orient}, {mouth_lab}")
                        
            except Exception as e:
                print(f"[ERROR] FaceMesh: {e}")
                
        # Pose detection
        if enable_pose and mp_pose:
            try:
                pose_out = mp_pose.process(rgb)
                if pose_out.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame, pose_out.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec((255, 0, 130), 2, 4),
                        mp_draw.DrawingSpec((80, 144, 255), 2)
                    )
                    analysis_lines.append("Pose erkannt")
                    analysis_data['pose_detected'] = True
            except Exception as e:
                print(f"[ERROR] Pose detection: {e}")
        
        # Hand detection
        if enable_hands and mp_hands_instance:
            try:
                hand_results = mp_hands_instance.process(rgb)
                if hand_results.multi_hand_landmarks:
                    hand_count = 0
                    for i, hand_lms in enumerate(hand_results.multi_hand_landmarks):
                        mp_draw.draw_landmarks(
                            frame, hand_lms, 
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec((0, 0, 255), 2, 4),
                            mp_draw.DrawingSpec((0, 255, 255), 2)
                        )
                        
                        # Simple hand position display
                        wrist = hand_lms.landmark[0]
                        px, py = int(wrist.x*w), int(wrist.y*h)
                        cv2.putText(frame, f"Hand {i+1}", (px+10, py-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
                        hand_count += 1
                        
                    if hand_count > 0:
                        analysis_lines.append(f"Hände erkannt: {hand_count}")
                        analysis_data['hands'] = [f"Hand {i+1}" for i in range(hand_count)]
                        
            except Exception as e:
                print(f"[ERROR] Hand detection: {e}")
        
        # Display analysis text
        for idx, line in enumerate(analysis_lines):
            cv2.putText(frame, line, (15, h-10-20*idx), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (w-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Update session stats
        if 'live_stats' in st.session_state:
            st.session_state.live_stats['frames_processed'] = frame_count
            st.session_state.live_stats['faces_detected'] += len(detected_faces_info)
            st.session_state.live_stats['persons_identified'] += len([f for f in detected_faces_info if f['person_info']])
            st.session_state.live_stats['emotions_detected'] += len(detected_faces_info)
        
        # Write frame to queue
        if not result_queue.empty():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
        
        try:
            result_queue.put((frame, analysis_data), block=False)
        except queue.Full:
            pass
    
    # Cleanup
    cap.release()
    if mp_hands_instance:
        mp_hands_instance.close()
    mp_fd.close()
    if mp_fm:
        mp_fm.close()
    if mp_pose:
        mp_pose.close()
    
    print("[INFO] Live camera worker stopped")
    
    # Initialize session state for camera
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'camera_thread' not in st.session_state:
        st.session_state.camera_thread = None
    if 'camera_queue' not in st.session_state:
        st.session_state.camera_queue = None
    if 'stop_camera_flag' not in st.session_state:
        st.session_state.stop_camera_flag = None
    if 'emotion_model' not in st.session_state:
        st.session_state.emotion_model = None
    if 'live_stats' not in st.session_state:
        st.session_state.live_stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'persons_identified': 0,
            'emotions_detected': 0
        }
    
    # Configuration section
    st.subheader("⚙️ Kamera-Konfiguration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        camera_source = st.selectbox(
            "Kamera-Quelle:",
            options=[0, 1, 2, "IP-Kamera"],
            format_func=lambda x: f"Kamera {x}" if isinstance(x, int) else "IP-Kamera",
            help="Wählen Sie die Kamera-Quelle"
        )
        
        if camera_source == "IP-Kamera":
            ip_camera_url = st.text_input(
                "IP-Kamera URL:",
                placeholder="http://192.168.1.100:8080/video",
                help="Geben Sie die URL Ihrer IP-Kamera ein"
            )
            camera_source = ip_camera_url if ip_camera_url else 0
    
    with col2:
        enable_face_recognition = st.checkbox(
            "👤 Personenerkennung", 
            value=True,
            help="Aktiviert die Erkennung bekannter Personen aus der Datenbank"
        )
        
        enable_emotion_detection = st.checkbox(
            "😊 Emotionserkennung",
            value=True,
            help="Aktiviert die Erkennung von Emotionen"
        )
        
        enable_pose_detection = st.checkbox(
            "🧍 Pose-Erkennung",
            value=True,
            help="Aktiviert die Körperhaltungserkennung"
        )
    
    with col3:
        enable_hand_detection = st.checkbox(
            "✋ Hand-Erkennung",
            value=True,
            help="Aktiviert die Handerkennung"
        )
        
        enable_face_mesh = st.checkbox(
            "🕸️ Gesichtsmesh",
            value=False,
            help="Zeigt detailliertes Gesichtsmesh (kann Performance beeinträchtigen)"
        )
        
        similarity_threshold = st.slider(
            "Ähnlichkeits-Schwellwert",
            min_value=0.6,
            max_value=0.95,
            value=0.75,
            step=0.05,
            help="Minimale Ähnlichkeit für Personenerkennung"
        )
    
    st.divider()
    
    # Camera controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎬 Kamera starten", disabled=st.session_state.camera_active, type="primary"):
            start_live_camera(
                camera_source, 
                enable_face_recognition,
                enable_emotion_detection, 
                enable_pose_detection,
                enable_hand_detection,
                enable_face_mesh,
                similarity_threshold
            )
    
    with col2:
        if st.button("⏹️ Kamera stoppen", disabled=not st.session_state.camera_active, type="secondary"):
            stop_live_camera()
    
    with col3:
        if st.button("📸 Screenshot", disabled=not st.session_state.camera_active):
            take_screenshot()
    
    with col4:
        if st.button("🔄 Statistiken zurücksetzen"):
            st.session_state.live_stats = {
                'frames_processed': 0,
                'faces_detected': 0,
                'persons_identified': 0,
                'emotions_detected': 0
            }
            st.rerun()
    
    # Live stats display
    if st.session_state.camera_active:
        st.subheader("📊 Live-Statistiken")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Frames verarbeitet", st.session_state.live_stats['frames_processed'])
        with col2:
            st.metric("Gesichter erkannt", st.session_state.live_stats['faces_detected'])
        with col3:
            st.metric("Personen identifiziert", st.session_state.live_stats['persons_identified'])
        with col4:
            st.metric("Emotionen erkannt", st.session_state.live_stats['emotions_detected'])
    
    # Camera feed display
    st.subheader("📹 Live-Feed")
    
    if st.session_state.camera_active and st.session_state.camera_queue:
        # Create placeholder for video feed
        video_placeholder = st.empty()
        
        # Display live feed
        try:
            if not st.session_state.camera_queue.empty():
                frame_data = st.session_state.camera_queue.get_nowait()
                if frame_data:
                    frame, analysis_data = frame_data
                    
                    # Convert BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Display analysis data
                    if analysis_data:
                        st.subheader("🔍 Aktuelle Analyse")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if analysis_data.get('faces'):
                                st.write("**Erkannte Gesichter:**")
                                for i, face in enumerate(analysis_data['faces']):
                                    if face.get('person_name'):
                                        st.write(f"• {face['person_name']} ({face.get('similarity', 0)*100:.1f}%)")
                                    else:
                                        st.write(f"• Unbekannte Person {i+1}")
                                    
                                    if face.get('emotion'):
                                        st.write(f"  Emotion: {face['emotion']} ({face.get('emotion_confidence', 0):.2f})")
                        
                        with col2:
                            if analysis_data.get('hands'):
                                st.write(f"**Hände erkannt:** {len(analysis_data['hands'])}")
                            
                            if analysis_data.get('pose_detected'):
                                st.write("**Körperhaltung:** ✅ Erkannt")
                            
                            if analysis_data.get('face_orientations'):
                                st.write("**Gesichtsausrichtungen:**")
                                for orientation in analysis_data['face_orientations']:
                                    st.write(f"• {orientation}")
        
        except queue.Empty:
            video_placeholder.info("⏳ Warten auf Kamera-Feed...")
        except Exception as e:
            st.error(f"❌ Fehler beim Anzeigen des Feeds: {e}")
    
    elif not st.session_state.camera_active:
        st.info("📷 Klicken Sie auf 'Kamera starten' um den Live-Feed zu beginnen")
    
    # System status
    st.subheader("💻 System-Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Verfügbare Features:**")
        if st.session_state.face_engine and st.session_state.vector_store:
            st.success("✅ Personenerkennung verfügbar")
        else:
            st.warning("⚠️ Personenerkennung nicht verfügbar")
        
        # Check for emotion model
        if st.session_state.emotion_model is not None:
            st.success("✅ Externes Emotionsmodell geladen")
        else:
            st.info("ℹ️ Landmark-basierte Emotionserkennung")
    
    with col2:
        st.write("**Datenbank-Status:**")
        stats = st.session_state.vector_store.get_collection_stats()
        if "error" not in stats:
            st.write(f"• Gesichter in DB: {stats.get('total_faces', 0)}")
            st.write(f"• Personen in DB: {stats.get('total_persons', 0)}")
        else:
            st.error("❌ Datenbank-Fehler")

def start_live_camera(camera_source, enable_face_recognition, enable_emotion_detection, 
                     enable_pose_detection, enable_hand_detection, enable_face_mesh, 
                     similarity_threshold):
    """Start the live camera with specified configuration"""
    
    # Check for required modules
    if not MEDIAPIPE_AVAILABLE:
        st.error("❌ MediaPipe nicht verfügbar - Live-Kamera kann nicht gestartet werden")
        return
    
    # Load emotion model if needed
    if enable_emotion_detection and st.session_state.emotion_model is None and TENSORFLOW_AVAILABLE:
        try:
            st.session_state.emotion_model = load_model("models/fer_trained.h5")
            st.success("✅ Externes Emotionsmodell geladen")
        except Exception as e:
            st.info(f"ℹ️ Externes Emotionsmodell nicht verfügbar: {e}")
            st.info("Verwende Landmark-basierte Emotionserkennung")
    
    # Initialize camera components
    try:
        import queue
        import threading
        
        # Create queue and thread
        st.session_state.camera_queue = queue.Queue(maxsize=2)
        st.session_state.stop_camera_flag = threading.Event()
        
        # Start camera thread
        st.session_state.camera_thread = threading.Thread(
            target=live_camera_worker,
            args=(
                camera_source,
                st.session_state.camera_queue,
                st.session_state.stop_camera_flag,
                st.session_state.face_engine if enable_face_recognition else None,
                st.session_state.vector_store if enable_face_recognition else None,
                st.session_state.emotion_model if enable_emotion_detection else None,
                enable_pose_detection,
                enable_hand_detection,
                enable_face_mesh,
                similarity_threshold
            ),
            daemon=True
        )
        
        st.session_state.camera_thread.start()
        st.session_state.camera_active = True
        
        st.success("🎬 Kamera gestartet!")
        time.sleep(0.5)  # Give camera time to initialize
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Fehler beim Starten der Kamera: {e}")

def stop_live_camera():
    """Stop the live camera"""
    try:
        if st.session_state.stop_camera_flag:
            st.session_state.stop_camera_flag.set()
        
        if st.session_state.camera_thread and st.session_state.camera_thread.is_alive():
            st.session_state.camera_thread.join(timeout=2.0)
        
        st.session_state.camera_active = False
        st.session_state.camera_thread = None
        st.session_state.camera_queue = None
        st.session_state.stop_camera_flag = None
        
        st.success("⏹️ Kamera gestoppt!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Fehler beim Stoppen der Kamera: {e}")

def take_screenshot():
    """Take a screenshot of the current frame"""
    try:
        if st.session_state.camera_queue and not st.session_state.camera_queue.empty():
            frame_data = st.session_state.camera_queue.get_nowait()
            if frame_data:
                frame, _ = frame_data
                
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = Path(f"screenshot_{timestamp}.jpg")
                cv2.imwrite(str(screenshot_path), frame)
                
                st.success(f"📸 Screenshot gespeichert: {screenshot_path}")
                
                # Show thumbnail
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Screenshot: {screenshot_path}", width=300)
                
        else:
            st.warning("⚠️ Kein Frame verfügbar für Screenshot")
            
    except Exception as e:
        st.error(f"❌ Fehler beim Erstellen des Screenshots: {e}")

def live_camera_worker(camera_source, result_queue, stop_flag, face_engine, vector_store, 
                      emotion_model, enable_pose, enable_hands, enable_face_mesh, similarity_threshold):
    """
    Camera worker thread adapted from face.py for Streamlit
    """
    if not MEDIAPIPE_AVAILABLE:
        print("[ERROR] MediaPipe not available")
        return
        
    try:
        
        # Initialize MediaPipe components
        mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) if enable_hands else None
        
        mp_fd = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        
        mp_fm = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        ) if enable_face_mesh else None
        
        mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) if enable_pose else None
        
        mp_draw = mp.solutions.drawing_utils
        
        # Open camera
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera: {camera_source}")
            return
        
        frame_count = 0
        person_history = {}
        recognition_stability_time = 0.5
        
        print(f"[INFO] Live camera started with source: {camera_source}")
        
        while not stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] No frame from camera")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            current_time = time.time()
            
            # Analysis data to return
            analysis_data = {
                'faces': [],
                'hands': [],
                'pose_detected': False,
                'face_orientations': []
            }
            
            # Face detection and recognition
            detected_faces_info = []
            
            try:
                det = mp_fd.process(rgb)
                if det.detections:
                    for i, d in enumerate(det.detections):
                        bb = d.location_data.relative_bounding_box
                        x1 = int(bb.xmin * w)
                        y1 = int(bb.ymin * h)
                        x2 = int((bb.xmin + bb.width) * w)
                        y2 = int((bb.ymin + bb.height) * h)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        roi = frame[y1:y2, x1:x2]
                        
                        if roi.size > 0:
                            # Emotion detection
                            emotion_label, emotion_prob = "neutral", 0.5
                            if emotion_model is not None:
                                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                emotion_label, emotion_prob = predict_emotion_streamlit(gray, emotion_model)
                            
                            face_info = {
                                'bbox': (x1, y1, x2, y2),
                                'roi': roi,
                                'emotion': emotion_label,
                                'emotion_confidence': emotion_prob,
                                'person_info': None
                            }
                            
                            # Face identification
                            if face_engine and vector_store and roi.shape[0] > 50 and roi.shape[1] > 50:
                                person_info = identify_face_streamlit(roi, face_engine, vector_store, similarity_threshold)
                                face_info['person_info'] = person_info
                            
                            detected_faces_info.append(face_info)
                            
                            # Update stats
                            st.session_state.live_stats['faces_detected'] += 1
                            if face_info['person_info']:
                                st.session_state.live_stats['persons_identified'] += 1
                            if emotion_label != "neutral":
                                st.session_state.live_stats['emotions_detected'] += 1
            
            except Exception as e:
                print(f"[ERROR] Face detection: {e}")
            
            # Draw face information
            for face_info in detected_faces_info:
                x1, y1, x2, y2 = face_info['bbox']
                person_info = face_info['person_info']
                emotion_label = face_info['emotion']
                emotion_prob = face_info['emotion_confidence']
                
                # Draw rectangle
                color = (0, 255, 0) if person_info else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw text
                if person_info:
                    name = person_info['name']
                    similarity = person_info['similarity']
                    main_label = f"{name} ({similarity*100:.0f}%)"
                    
                    # Safe text rendering
                    try:
                        cv2.putText(frame, main_label, (x1, y1-40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    except:
                        safe_name = name.encode('ascii', errors='ignore').decode('ascii')
                        main_label = f"{safe_name} ({similarity*100:.0f}%)"
                        cv2.putText(frame, main_label, (x1, y1-40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    analysis_data['faces'].append({
                        'person_name': name,
                        'similarity': similarity,
                        'emotion': emotion_label,
                        'emotion_confidence': emotion_prob
                    })
                else:
                    label = f"Unbekannt - {emotion_label} {emotion_prob:.2f}"
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    analysis_data['faces'].append({
                        'person_name': None,
                        'emotion': emotion_label,
                        'emotion_confidence': emotion_prob
                    })
                
                # Emotion text
                emotion_text = f"{emotion_label} {emotion_prob:.2f}"
                cv2.putText(frame, emotion_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Face mesh processing
            if enable_face_mesh and mp_fm:
                try:
                    mesh = mp_fm.process(rgb)
                    if mesh.multi_face_landmarks:
                        for fl in mesh.multi_face_landmarks:
                            mp_draw.draw_landmarks(
                                frame, fl,
                                mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                mp_draw.DrawingSpec((0, 255, 0), 1, 1),
                                mp_draw.DrawingSpec((0, 180, 0), 1)
                            )
                            
                            # Face orientation
                            orientation = get_face_orientation_streamlit(fl, w, h)
                            analysis_data['face_orientations'].append(orientation)
                except Exception as e:
                    print(f"[ERROR] Face mesh: {e}")
            
            # Pose detection
            if enable_pose and mp_pose:
                try:
                    pose_out = mp_pose.process(rgb)
                    if pose_out.pose_landmarks:
                        mp_draw.draw_landmarks(
                            frame, pose_out.pose_landmarks,
                            mp.solutions.pose.POSE_CONNECTIONS,
                            mp_draw.DrawingSpec((255, 0, 130), 2, 4),
                            mp_draw.DrawingSpec((80, 144, 255), 2)
                        )
                        analysis_data['pose_detected'] = True
                except Exception as e:
                    print(f"[ERROR] Pose detection: {e}")
            
            # Hand detection
            if enable_hands and mp_hands:
                try:
                    hand_results = mp_hands.process(rgb)
                    if hand_results.multi_hand_landmarks:
                        for i, hand_lms in enumerate(hand_results.multi_hand_landmarks):
                            mp_draw.draw_landmarks(
                                frame, hand_lms,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                mp_draw.DrawingSpec((0, 0, 255), 2, 4),
                                mp_draw.DrawingSpec((0, 255, 255), 2)
                            )
                            
                            # Hand position
                            wrist = hand_lms.landmark[0]
                            px, py = int(wrist.x * w), int(wrist.y * h)
                            cv2.putText(frame, f"Hand {i+1}", (px+10, py-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
                            
                            analysis_data['hands'].append(f"Hand {i+1}")
                            
                except Exception as e:
                    print(f"[ERROR] Hand detection: {e}")
            
            # Frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (w-150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Update stats
            st.session_state.live_stats['frames_processed'] = frame_count
            
            # Put frame in queue
            try:
                if not result_queue.full():
                    result_queue.put((frame, analysis_data), block=False)
            except:
                try:
                    result_queue.get_nowait()  # Remove old frame
                    result_queue.put((frame, analysis_data), block=False)
                except:
                    pass
            
            time.sleep(0.03)  # ~30 FPS
        
        # Cleanup
        cap.release()
        if mp_hands:
            mp_hands.close()
        mp_fd.close()
        if mp_fm:
            mp_fm.close()
        if mp_pose:
            mp_pose.close()
        
        print("[INFO] Live camera stopped")
        
    except Exception as e:
        print(f"[ERROR] Camera worker: {e}")

def predict_emotion_streamlit(face_gray, model):
    """Streamlit-compatible emotion prediction"""
    try:
        if model is not None:
            img = cv2.resize(face_gray, (48, 48))
            img = cv2.equalizeHist(img)
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=(0, -1))
            
            preds = model.predict(img, verbose=0)[0]
            top_emotion_idx = np.argmax(preds)
            top_confidence = float(preds[top_emotion_idx])
            
            emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
            return emotions[top_emotion_idx], top_confidence
        
        return "neutral", 0.5
    except Exception as e:
        print(f"[ERROR] Emotion prediction: {e}")
        return "neutral", 0.5

def identify_face_streamlit(face_region, face_engine, vector_store, similarity_threshold):
    """Streamlit-compatible face identification"""
    try:
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        
        # Detect face
        face_locations = face_engine.detect_faces(face_rgb)
        if not face_locations:
            return None
        
        # Extract embedding
        face_location = face_locations[0]
        embedding = face_engine.extract_face_embedding(face_rgb, face_location)
        if embedding is None:
            return None
        
        # Search similar faces
        similar_faces = vector_store.search_similar_faces(
            embedding,
            n_results=5,
            min_similarity=max(0.6, similarity_threshold - 0.15)
        )
        
        if similar_faces:
            best_match = similar_faces[0]
            best_similarity = best_match['similarity']
            
            if best_similarity < 0.65:
                return None
            
            person_data = best_match['metadata']
            name = person_data.get('full_name', 'Unbekannt')
            
            # Safe UTF-8 handling
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='replace')
            
            return {
                'name': name,
                'similarity': best_similarity,
                'person_id': person_data.get('person_id'),
                'face_id': best_match['face_id']
            }
        
        return None
    
    except Exception as e:
        print(f"[ERROR] Face identification: {e}")
        return None

def get_face_orientation_streamlit(mesh_landmarks, w, h):
    """Get face orientation from landmarks"""
    try:
        lm = mesh_landmarks.landmark
        nose = np.array([lm[1].x*w, lm[1].y*h])
        l_eye = np.array([lm[263].x*w, lm[263].y*h])
        r_eye = np.array([lm[33].x*w, lm[33].y*h])
        chin = np.array([lm[152].x*w, lm[152].y*h])
        forehead = np.array([lm[10].x*w, lm[10].y*h])
        
        center_x = (l_eye[0] + r_eye[0]) / 2
        hor = "Mitte" if abs(nose[0] - center_x) < 0.02*w else ("Rechts" if nose[0] < center_x else "Links")
        
        center_y = (forehead[1] + chin[1]) / 2
        vert = "Mitte" if abs(nose[1] - center_y) < 0.025*h else ("Oben" if nose[1] < center_y else "Unten")
        
        return f"Kopf: {hor}, {vert}"
    except:
        return "Unbekannt"

if __name__ == "__main__":
    main()