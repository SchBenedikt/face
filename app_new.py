"""
Face Recognition Search Application - Simplified & Modular
Improved with Best AI Models: RetinaFace/MTCNN for Detection, ArcFace for Recognition
"""
# Set up environment before importing other modules
from setup_env import setup_environment
setup_environment()

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

import streamlit as st
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configuration
from config import PAGE_TITLE, PAGE_ICON, LAYOUT

# Import AI models
from models import FaceDetector, FaceRecognizer

# Import vector store
from vector_store import FaceVectorStore

# Import pages
from pages import face_search_page, settings_page

# Import remaining pages from original app (will be refactored later)
from app import (
    upload_processing_page,
    batch_face_processing_page,
    web_scraping_page,
    face_gallery_page,
    name_gallery_page,
    duplicate_manager_page,
    database_stats_page
)

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)


def initialize_models():
    """Initialize AI models with best available backends"""
    
    # Initialize Face Detector
    if 'face_detector' not in st.session_state:
        with st.spinner("🔄 Initialisiere Face Detection Modell..."):
            try:
                detection_backend = st.session_state.get('detection_backend', 'auto')
                st.session_state.face_detector = FaceDetector(backend=detection_backend)
                logger.info("Face Detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Face Detector: {e}")
                st.error("❌ Face Detector konnte nicht initialisiert werden")
    
    # Initialize Face Recognizer
    if 'face_recognizer' not in st.session_state:
        with st.spinner("🔄 Initialisiere Face Recognition Modell..."):
            try:
                recognition_model = st.session_state.get('recognition_model', 'auto')
                st.session_state.face_recognizer = FaceRecognizer(model_name=recognition_model)
                logger.info("Face Recognizer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Face Recognizer: {e}")
                st.error("❌ Face Recognizer konnte nicht initialisiert werden")
    
    # Initialize Vector Store
    if 'vector_store' not in st.session_state:
        with st.spinner("🔄 Initialisiere Vector Database..."):
            try:
                st.session_state.vector_store = FaceVectorStore()
                logger.info("Vector Store initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Vector Store: {e}")
                st.error("❌ Vector Store konnte nicht initialisiert werden")


def main():
    """Main application function"""
    
    # Initialize AI models
    initialize_models()
    
    st.title("🔍 Face Recognition Search")
    st.markdown("*Powered by State-of-the-Art AI Models*")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("⚙️ Navigation")
        
        page = st.selectbox(
            "Seite wählen:",
            [
                "🔍 Face Search (Neu)",
                "📥 Image Upload & Processing",
                "🧠 Batch Face Processing",
                "🌐 Web Scraping",
                "👥 Face Gallery",
                "🏷️ Name Gallery",
                "🔧 Duplicate Manager",
                "📊 Database Statistics",
                "⚙️ Settings (Neu)"
            ]
        )
        
        st.markdown("---")
        
        # Quick stats
        if 'vector_store' in st.session_state:
            stats = st.session_state.vector_store.get_collection_stats()
            st.subheader("📈 Quick Stats")
            
            if "error" not in stats:
                st.metric("Total Faces", stats.get("total_faces", 0))
                st.metric("Unique Images", stats.get("unique_images", 0))
            else:
                st.warning("Database not initialized")
        
        st.markdown("---")
        
        # Model info
        st.subheader("🤖 Active Models")
        
        if 'face_detector' in st.session_state:
            backend = st.session_state.face_detector.get_best_backend()
            st.info(f"**Detection:** {backend.upper()}")
        
        if 'face_recognizer' in st.session_state:
            model_info = st.session_state.face_recognizer.get_model_info()
            st.info(f"**Recognition:** {model_info['name']}")
    
    # Handle page navigation
    if page == "🔍 Face Search (Neu)":
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
    elif page == "⚙️ Settings (Neu)":
        settings_page()


if __name__ == "__main__":
    main()
