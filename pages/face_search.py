"""
Simplified Face Search Page with Best AI Models
"""
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def face_search_page():
    """Simplified face search with state-of-the-art AI models"""
    
    st.header("🔍 Face Search - Powered by Best AI")
    
    # Show active models
    _display_active_models()
    
    st.markdown("---")
    
    # Upload image
    uploaded_file = st.file_uploader(
        "📤 Bild hochladen:",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Laden Sie ein Bild mit Gesicht hoch, um nach ähnlichen Gesichtern zu suchen"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display uploaded image
            query_image = Image.open(uploaded_file)
            st.image(query_image, caption="Hochgeladenes Bild", use_column_width=True)
            
            # Search parameters
            st.subheader("🎯 Suchparameter")
            
            max_results = st.slider(
                "Maximale Ergebnisse",
                min_value=5,
                max_value=100,
                value=20,
                help="Anzahl der Suchergebnisse"
            )
            
            similarity_threshold = st.slider(
                "Ähnlichkeitsschwelle",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Mindestähnlichkeit (höher = genauer)"
            )
            
            # Search button
            if st.button("🔍 Gesichter suchen", type="primary", use_container_width=True):
                _perform_search(uploaded_file, max_results, similarity_threshold)
        
        with col2:
            # Show detected faces if available
            if 'detected_faces' in st.session_state and st.session_state.detected_faces:
                _show_detected_faces()
    
    # Display search results
    if 'search_results' in st.session_state and st.session_state.search_results:
        st.markdown("---")
        st.subheader("🎯 Suchergebnisse")
        
        # Clear button
        if st.button("🗑️ Ergebnisse löschen"):
            st.session_state.search_results = []
            if 'detected_faces' in st.session_state:
                del st.session_state.detected_faces
            st.rerun()
        
        _display_results(st.session_state.search_results)


def _display_active_models():
    """Display information about active AI models"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Face Detection")
        if 'face_detector' in st.session_state:
            backend = st.session_state.face_detector.get_best_backend()
            st.success(f"**Aktives Modell:** {backend.upper()}")
            
            # Show backend capabilities
            if backend == 'retinaface':
                st.info("✨ Höchste Genauigkeit - State-of-the-art")
            elif backend == 'mtcnn':
                st.info("✨ Sehr gute Genauigkeit")
            elif backend == 'deepface':
                st.info("✓ Gute Genauigkeit")
            else:
                st.info("⚡ Schnell - Moderate Genauigkeit")
        else:
            st.warning("Modell wird geladen...")
    
    with col2:
        st.markdown("### 🧠 Face Recognition")
        if 'face_recognizer' in st.session_state:
            model_info = st.session_state.face_recognizer.get_model_info()
            st.success(f"**Aktives Modell:** {model_info['name']}")
            st.info(f"✨ {model_info['accuracy']} - {model_info['dimension']}D Embeddings")
        else:
            st.warning("Modell wird geladen...")


def _perform_search(uploaded_file, max_results: int, similarity_threshold: float):
    """Perform face search with uploaded image"""
    
    with st.spinner("🔍 Analysiere Bild und suche nach Gesichtern..."):
        try:
            from utils import load_and_preprocess_image
            
            # Save uploaded file temporarily
            temp_path = Path("/tmp/query_image.jpg")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Load image
            query_image = load_and_preprocess_image(temp_path)
            if query_image is None:
                st.error("❌ Fehler beim Laden des Bildes")
                return
            
            # Detect faces
            if 'face_detector' not in st.session_state:
                st.error("❌ Face Detector nicht initialisiert")
                return
            
            face_locations = st.session_state.face_detector.detect_faces(
                query_image,
                min_confidence=0.9
            )
            
            if not face_locations:
                st.error("❌ Keine Gesichter im Bild erkannt")
                st.info("**Tipps:**\n- Verwenden Sie ein Bild mit klarem Gesicht\n- Stellen Sie sicher, dass das Gesicht gut beleuchtet ist\n- Vermeiden Sie zu kleine Bilder")
                return
            
            # Store detected faces
            st.session_state.detected_faces = {
                'face_locations': face_locations,
                'query_image': query_image,
                'max_results': max_results,
                'similarity_threshold': similarity_threshold
            }
            
            st.success(f"✅ {len(face_locations)} Gesicht(er) erkannt!")
            
            # If only one face, search immediately
            if len(face_locations) == 1:
                _search_with_face(0)
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Fehler bei der Suche: {str(e)}")
            logger.error(f"Search error: {e}", exc_info=True)
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()


def _show_detected_faces():
    """Show detected faces for selection"""
    
    st.subheader("👥 Erkannte Gesichter")
    
    detected_data = st.session_state.detected_faces
    face_locations = detected_data['face_locations']
    query_image = detected_data['query_image']
    
    st.write(f"**{len(face_locations)} Gesicht(er) erkannt. Wählen Sie eines zum Suchen:**")
    
    # Display faces in columns
    cols = st.columns(min(len(face_locations), 3))
    
    for idx, face_loc in enumerate(face_locations):
        with cols[idx % 3]:
            try:
                from utils import create_thumbnail
                
                top, right, bottom, left = face_loc
                
                # Extract and display face
                face_image = query_image[top:bottom, left:right]
                if face_image.shape[0] > 0 and face_image.shape[1] > 0:
                    face_thumbnail = create_thumbnail(face_image, (120, 120))
                    st.image(face_thumbnail, caption=f"Gesicht {idx+1}")
                    
                    # Search button for this face
                    if st.button(
                        f"🔍 Suchen",
                        key=f"search_face_{idx}",
                        use_container_width=True
                    ):
                        _search_with_face(idx)
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Fehler: {e}")


def _search_with_face(face_index: int):
    """Search for similar faces using selected face"""
    
    detected_data = st.session_state.detected_faces
    face_locations = detected_data['face_locations']
    query_image = detected_data['query_image']
    max_results = detected_data['max_results']
    similarity_threshold = detected_data['similarity_threshold']
    
    selected_face = face_locations[face_index]
    
    with st.spinner(f"🔍 Suche nach ähnlichen Gesichtern..."):
        try:
            # Extract embedding
            if 'face_recognizer' not in st.session_state:
                st.error("❌ Face Recognizer nicht initialisiert")
                return
            
            query_embedding = st.session_state.face_recognizer.extract_embedding(
                query_image,
                selected_face,
                align=True
            )
            
            if query_embedding is None:
                st.error("❌ Fehler beim Extrahieren der Gesichtsmerkmale")
                return
            
            # Search in vector store
            if 'vector_store' not in st.session_state:
                st.error("❌ Vector Store nicht initialisiert")
                return
            
            similar_faces = st.session_state.vector_store.search_similar_faces(
                query_embedding,
                n_results=max_results,
                min_similarity=similarity_threshold
            )
            
            # Store results
            st.session_state.search_results = similar_faces
            
            if similar_faces:
                avg_similarity = sum(f['similarity'] for f in similar_faces) / len(similar_faces)
                st.success(
                    f"✅ {len(similar_faces)} ähnliche Gesichter gefunden!\n\n"
                    f"📊 Durchschnittliche Ähnlichkeit: {avg_similarity*100:.1f}%"
                )
            else:
                st.warning("⚠️ Keine ähnlichen Gesichter gefunden. Versuchen Sie:\n"
                          "- Niedrigere Ähnlichkeitsschwelle\n"
                          "- Mehr Bilder zur Datenbank hinzufügen")
            
        except Exception as e:
            st.error(f"❌ Suchfehler: {str(e)}")
            logger.error(f"Search error: {e}", exc_info=True)


def _display_results(results: List[Dict[str, Any]]):
    """Display search results in grid"""
    
    from ui.components import display_search_results_grid
    
    # Display results
    display_search_results_grid(
        results,
        cols_per_row=5,
        show_similarity=True
    )
