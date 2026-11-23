"""Optional facial analysis and analysis-modal UI."""
from pathlib import Path
import logging

import cv2
import streamlit as st

from utils import create_thumbnail, load_and_preprocess_image

logger = logging.getLogger(__name__)


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
            return {"error": "Face is too small for attribute analysis"}

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
                return {"error": f"Analysis error: {str(df_error)[:100]}..."}

    except ImportError:
        return {"error": "DeepFace is not installed. Install it with: pip install deepface"}
    except Exception as e:
        logger.error(f"Facial attribute analysis error: {e}")
        return {"error": f"Unexpected error: {str(e)[:100]}..."}


@st.dialog("Facial attribute analysis", width="large")
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


def show_analysis_tab(image_path, face_location, face_id):
    """Display facial analysis in tab"""

    with st.spinner("Analyzing facial attributes..."):
        analysis = analyze_facial_attributes(image_path, face_location)

    if analysis is None:
        st.error("❌ Could not analyze facial attributes.")
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
                            st.image(thumbnail, caption="Analyzed face", alt="Face used for analysis")
    except:
        pass

    # Analysis results in a clean 2-column layout
    col1, col2 = st.columns(2)

    with col1:
        # Age Analysis
        if 'age' in analysis:
            st.markdown("### 🎂 **Age estimate**")
            age = analysis['age']
            st.metric("Estimated age", f"{age} years", help="Model estimate; accuracy is not guaranteed.")

            # Age category
            if age < 18:
                st.info("📚 Teenager")
            elif age < 30:
                st.info("🌟 Young adult")
            elif age < 50:
                st.info("💼 Adult")
            elif age < 70:
                st.info("🎓 Mature adult")
            else:
                st.info("👴 Senior")

        st.markdown("---")

        # Gender Analysis
        if 'gender' in analysis:
            st.markdown("### ⚧️ **Gender estimate**")
            gender_data = analysis['gender']
            if isinstance(gender_data, dict):
                male_conf = gender_data.get('Man', 0)
                female_conf = gender_data.get('Woman', 0)

                if male_conf > female_conf:
                    st.metric("Gender estimate", "👨 Man", f"{male_conf:.1f}% score")
                else:
                    st.metric("Gender estimate", "👩 Woman", f"{female_conf:.1f}% score")

    with col2:
        # Emotion Analysis
        if 'emotion' in analysis:
            st.markdown("### 😊 **Emotion estimate**")
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
                st.metric("Most likely emotion", f"{emoji} {emotion_name.title()}", f"{emotion_conf:.1f}% confidence")

        st.markdown("---")

        # Race/Ethnicity Analysis
        if 'race' in analysis:
            st.markdown("### 🌍 **Ethnicity estimate**")
            race_data = analysis['race']
            if isinstance(race_data, dict):
                # Find dominant ethnicity
                dominant_race = max(race_data.items(), key=lambda x: x[1])
                race_name, race_conf = dominant_race
                st.metric("Most likely estimate", race_name.title(), f"{race_conf:.1f}% confidence")

    # Important disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ Important:** This analysis is based on AI models and is intended for **demonstration and research purposes only**.
    Results are estimates and should not be used for identification, discrimination, or decision-making.
    """)


@st.dialog("🖼️ Full image view", width="large")
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
        st.error(f"Missing image data for full image view: {', '.join(missing_keys)}")
        st.info("Try opening the full image again.")
        if st.button("Close"):
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
    st.markdown(f"**File:** `{image_path.name}`")
    st.markdown(f"**📍 Pfad:** `{image_path}`")

    try:
        # Load the full image
        full_image = load_and_preprocess_image(image_path)
        if full_image is None:
            st.error("❌ Could not load the image.")
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
                    st.success(f"✅ Face marked at: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
            else:
                top, right, bottom, left = face_location
                cv2.rectangle(image_with_box, (left, top), (right, bottom), (0, 255, 0), 3)
                st.success(f"✅ Face marked at: Top={top}, Right={right}, Bottom={bottom}, Left={left}")

        st.markdown("---")

        # Display the image with maximum width
        caption = f"Full image - {image_path.name}"
        if face_location:
            caption += " (face highlighted)"

        st.image(image_with_box, caption=caption, width='stretch', alt=caption)

        # Image info
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Image size", f"{full_image.shape[1]} × {full_image.shape[0]} px")

        with col2:
            file_size = image_path.stat().st_size if image_path.exists() else 0
            st.metric("File size", f"{file_size / 1024:.1f} KB")

        with col3:
            if face_location:
                st.metric("face detected", "Yes")
            else:
                st.metric("face detected", "No")

    except Exception as e:
        st.error(f"❌ Could not display the full image: {str(e)}")

    # Close button
    st.markdown("---")
    if st.button("❌ Close full image", key="close_full_image_modal", type="primary", width='stretch'):
        # Clear full image data
        for key in required_keys:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # Display the detailed analysis
    if all(key in st.session_state for key in ['analysis_image_path', 'analysis_face_location', 'analysis_face_id']):

        # Get analysis data
        with st.spinner("🧠 Analyzing facial attributes..."):
            analysis = analyze_facial_attributes(
                st.session_state.analysis_image_path,
                st.session_state.analysis_face_location
            )

        if analysis is None:
            st.error("❌ Could not analyze facial attributes.")
            return

        if "error" in analysis:
            st.error(f"❌ {analysis['error']}")
            return

        # Header info
        st.markdown(f"**Face ID:** `{st.session_state.analysis_face_id}`")
        st.markdown(f"**File:** `{Path(st.session_state.analysis_image_path).name}`")

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
                                st.image(face_crop, caption="Analyzed face", width=250, alt="Face used for analysis")
        except:
            pass

        st.markdown("---")

        # Analysis results in a clean 2-column layout
        col1, col2 = st.columns(2)

        with col1:
            # Age Analysis
            if 'age' in analysis:
                st.markdown("## Age estimate")
                age = analysis['age']
                st.metric("Estimated age", f"{age} years", help="Model estimate; accuracy is not guaranteed.")

                # Age category with visual indicator
                if age < 18:
                    st.info("**Category:** Teenager")
                elif age < 30:
                    st.info("**Category:** Young adult")
                elif age < 50:
                    st.info("**Category:** Adult")
                elif age < 70:
                    st.info("**Category:** Older adult")
                else:
                    st.info("**Category:** Senior")

            st.markdown("---")

            # Gender Analysis
            if 'gender' in analysis:
                st.markdown("## ⚧️ **Gender estimate**")
                gender_data = analysis['gender']
                if isinstance(gender_data, dict):
                    male_prob = float(gender_data.get('Man', 0))
                    female_prob = float(gender_data.get('Woman', 0))

                    # Visual representation with progress bars
                    st.write("**👨 Man**")
                    st.progress(male_prob / 100.0)
                    st.write(f"`{male_prob:.1f}%`")

                    st.write("**👩 Woman**")
                    st.progress(female_prob / 100.0)
                    st.write(f"`{female_prob:.1f}%`")

                    # Highlight prediction
                    dominant = "👨 Man" if male_prob > female_prob else "👩 Woman"
                    confidence = max(male_prob, female_prob)
                    st.success(f"**Estimate:** {dominant} (confidence: {confidence:.1f}%)")
                else:
                    st.write(f"**Gender estimate:** {gender_data}")

        with col2:
            # Emotion Analysis
            if 'emotion' in analysis:
                st.markdown("## 😊 **Emotion estimate**")
                emotions = analysis['emotion']
                if isinstance(emotions, dict):
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                    emotion_map = {
                        'happy': ('😊 Happy', '#28a745'),
                        'neutral': ('😐 Neutral', '#6c757d'),
                        'sad': ('😢 Sad', '#007bff'),
                        'angry': ('😠 Angry', '#dc3545'),
                        'surprise': ('😲 Surprised', '#ffc107'),
                        'fear': ('😨 Fearful', '#fd7e14'),
                        'disgust': ('🤢 Disgusted', '#6f42c1')
                    }

                    # Show top 5 emotions with visual bars
                    for emotion, confidence in sorted_emotions[:5]:
                        emotion_label, color = emotion_map.get(emotion, (emotion.title(), '#6c757d'))
                        st.write(f"**{emotion_label}**")
                        st.progress(float(confidence / 100.0))
                        st.write(f"`{confidence:.1f}%`")

                    # Dominant emotion highlight
                    top_emotion, top_confidence = sorted_emotions[0]
                    top_emotion_label, _ = emotion_map.get(top_emotion, (top_emotion.title(), '#6c757d'))
                    st.success(f"**Most likely emotion:** {top_emotion_label} ({top_confidence:.1f}%)")

            st.markdown("---")

            # Race/Ethnicity Analysis
            if 'race' in analysis:
                st.markdown("## 🌍 **Ethnicity estimate**")
                race_data = analysis['race']
                if isinstance(race_data, dict):
                    sorted_races = sorted(race_data.items(), key=lambda x: x[1], reverse=True)
                    race_map = {
                        'asian': '🌏 Asian',
                        'white': '🌍 White',
                        'middle eastern': '🌍 Middle Eastern',
                        'indian': '🌏 Indian',
                        'latino hispanic': '🌎 Latino/Hispanic',
                        'black': '🌍 Black'
                    }

                    # Show top predictions with progress bars
                    for race, confidence in sorted_races:
                        race_label = race_map.get(race.lower(), race.title())
                        st.write(f"**{race_label}**")
                        st.progress(float(confidence / 100.0))
                        st.write(f"`{confidence:.1f}%`")

                    # Dominant prediction
                    top_race, top_confidence = sorted_races[0]
                    top_race_label = race_map.get(top_race.lower(), top_race.title())
                    st.info(f"**Most likely estimate:** {top_race_label} ({top_confidence:.1f}%)")

        # Footer with important disclaimer
        st.markdown("---")
        st.warning("""
        **⚠️ Important:** This analysis is based on AI models and is intended for **demonstration and research purposes only**.
        Results are estimates and should not be used for identification, discrimination, or decision-making.
        """)

        # Close button
        if st.button("✅ Close analysis", key="close_popup_analysis", type="primary", width='stretch'):
            # Clear analysis data
            if 'analysis_image_path' in st.session_state:
                del st.session_state.analysis_image_path
            if 'analysis_face_location' in st.session_state:
                del st.session_state.analysis_face_location
            if 'analysis_face_id' in st.session_state:
                del st.session_state.analysis_face_id
            st.rerun()

    else:
        st.error("❌ Analysis data is unavailable.")
        if st.button("Close", key="close_error_modal"):
            st.rerun()
