"""
Settings Page
"""
import streamlit as st


def settings_page():
    """Display settings and configuration"""
    
    st.header("⚙️ Einstellungen")
    
    # Model settings section
    st.subheader("🤖 AI-Modelle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Face Detection")
        
        detection_backend = st.selectbox(
            "Detection Backend:",
            options=["auto", "retinaface", "mtcnn", "deepface", "opencv"],
            index=0,
            help="Auto wählt automatisch das beste verfügbare Modell"
        )
        
        st.info(
            "**Empfehlung:** Verwenden Sie 'auto' für die beste Genauigkeit.\n\n"
            "**Verfügbare Backends:**\n"
            "- RetinaFace: Höchste Genauigkeit (State-of-the-art)\n"
            "- MTCNN: Sehr gute Genauigkeit\n"
            "- DeepFace: Gute Genauigkeit\n"
            "- OpenCV: Schnell, moderate Genauigkeit"
        )
    
    with col2:
        st.markdown("### 🧠 Face Recognition")
        
        recognition_model = st.selectbox(
            "Recognition Model:",
            options=["auto", "arcface", "facenet512", "vggface", "facenet"],
            index=0,
            help="Auto wählt automatisch das beste verfügbare Modell"
        )
        
        st.info(
            "**Empfehlung:** Verwenden Sie 'auto' für die beste Genauigkeit.\n\n"
            "**Verfügbare Modelle:**\n"
            "- ArcFace: State-of-the-art (512D)\n"
            "- FaceNet512: Sehr gute Genauigkeit (512D)\n"
            "- VGG-Face: Gute Genauigkeit (4096D)\n"
            "- FaceNet: Schnell, kompakt (128D)"
        )
    
    st.markdown("---")
    
    # Performance settings
    st.subheader("⚡ Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_confidence = st.slider(
            "Min. Confidence",
            min_value=0.5,
            max_value=0.99,
            value=0.9,
            step=0.05,
            help="Minimale Konfidenz für Face Detection"
        )
    
    with col2:
        min_face_size = st.slider(
            "Min. Face Size (px)",
            min_value=20,
            max_value=100,
            value=30,
            help="Minimale Gesichtsgröße in Pixeln"
        )
    
    with col3:
        batch_size = st.slider(
            "Batch Size",
            min_value=10,
            max_value=100,
            value=32,
            help="Anzahl der Bilder pro Batch-Verarbeitung"
        )
    
    st.markdown("---")
    
    # Apply settings
    if st.button("💾 Einstellungen übernehmen", type="primary"):
        # Update session state
        if detection_backend != "auto":
            st.session_state.detection_backend = detection_backend
        
        if recognition_model != "auto":
            st.session_state.recognition_model = recognition_model
        
        st.session_state.min_confidence = min_confidence
        st.session_state.min_face_size = min_face_size
        st.session_state.batch_size = batch_size
        
        st.success("✅ Einstellungen gespeichert! Bitte App neu laden für vollständige Übernahme.")
    
    st.markdown("---")
    
    # Current configuration
    st.subheader("📊 Aktuelle Konfiguration")
    
    # Show active models
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Face Detection:**")
        if 'face_detector' in st.session_state:
            backend = st.session_state.face_detector.get_best_backend()
            available = st.session_state.face_detector.available_backends
            st.write(f"- Aktiv: `{backend}`")
            st.write(f"- Verfügbar: `{', '.join(available)}`")
        else:
            st.write("Nicht initialisiert")
    
    with col2:
        st.markdown("**Face Recognition:**")
        if 'face_recognizer' in st.session_state:
            model_info = st.session_state.face_recognizer.get_model_info()
            st.write(f"- Aktiv: `{model_info['name']}`")
            st.write(f"- Dimension: `{model_info['dimension']}D`")
            st.write(f"- Genauigkeit: `{model_info['accuracy']}`")
        else:
            st.write("Nicht initialisiert")
    
    st.markdown("---")
    
    # System information
    st.subheader("ℹ️ System Information")
    
    st.markdown("""
    ### Über diese Anwendung
    
    **Face Recognition Search** - Powered by Best AI Models
    
    Diese Anwendung verwendet state-of-the-art Deep Learning Modelle für:
    - **Face Detection**: RetinaFace, MTCNN, oder DeepFace
    - **Face Recognition**: ArcFace, FaceNet512, oder VGG-Face
    
    **Technologien:**
    - Streamlit (Web Interface)
    - DeepFace (AI Models)
    - ChromaDB (Vector Database)
    - OpenCV (Image Processing)
    
    **Features:**
    - Hochpräzise Gesichtserkennung
    - Schnelle Ähnlichkeitssuche
    - Modulare, erweiterbare Architektur
    - Datenschutz durch lokale Verarbeitung
    """)
