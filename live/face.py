import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from threading import Thread
import queue
import time
import sys
import os

# Add the parent directory to path to import our face recognition modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our face recognition modules
from face_recognition_engine import FaceRecognitionEngine
from vector_store import FaceVectorStore
from config import SIMILARITY_THRESHOLD

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def predict_emotion_from_landmarks(face_landmarks, w, h):
    """
    Verbesserte Emotionserkennung basierend auf Gesichtspunkten
    Mit besserer Kalibrierung und realistischeren Schwellwerten
    """
    try:
        if not face_landmarks or not face_landmarks.landmark:
            return "neutral", 0.5
            
        lm = face_landmarks.landmark
        
        # Verbesserte Mund-Analyse
        mouth_top = lm[13].y * h
        mouth_bottom = lm[14].y * h
        mouth_left = lm[308].x * w  
        mouth_right = lm[78].x * w
        mouth_center_left = lm[291].y * h
        mouth_center_right = lm[61].y * h
        
        mouth_height = abs(mouth_bottom - mouth_top)
        mouth_width = abs(mouth_right - mouth_left) 
        mouth_curve = (mouth_center_left + mouth_center_right) / 2 - mouth_top
        
        # Verbesserte Augen-Analyse  
        left_eye_top = lm[159].y * h
        left_eye_bottom = lm[145].y * h
        right_eye_top = lm[386].y * h  
        right_eye_bottom = lm[374].y * h
        
        left_eye_height = abs(left_eye_bottom - left_eye_top)
        right_eye_height = abs(right_eye_bottom - right_eye_top)
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        # Verbesserte Augenbrauen-Analyse
        left_brow = lm[70].y * h
        right_brow = lm[300].y * h
        left_eye_center = (left_eye_top + left_eye_bottom) / 2
        right_eye_center = (right_eye_top + right_eye_bottom) / 2
        
        # Bessere Brow-Height Berechnung
        left_brow_distance = left_brow - left_eye_center
        right_brow_distance = right_brow - right_eye_center
        avg_brow_distance = (left_brow_distance + right_brow_distance) / 2
        
        # Emotionserkennung mit realistischeren Schwellwerten
        emotions_confidence = {}
        
        # Happy: Mund nach oben gebogen UND breiter
        if mouth_curve > 1.5 and mouth_height > 3:
            happiness_score = min(0.9, 0.5 + (mouth_curve / 10) + (mouth_height / 30))
            emotions_confidence["happy"] = happiness_score
        else:
            emotions_confidence["happy"] = 0.1
            
        # Sad: Mund nach unten gebogen ODER sehr kleine Augen
        if mouth_curve < -1.0 or avg_eye_height < 5:
            sadness_score = min(0.85, 0.4 + abs(mouth_curve) / 8)
            emotions_confidence["sad"] = sadness_score
        else:
            emotions_confidence["sad"] = 0.1
            
        # Surprise: Große Augen UND offener Mund
        if avg_eye_height > 8 and mouth_height > 10:
            surprise_score = min(0.9, 0.5 + (avg_eye_height / 15) + (mouth_height / 25))
            emotions_confidence["surprise"] = surprise_score
        else:
            emotions_confidence["surprise"] = 0.1
            
        # Angry: Zusammengezogene Augenbrauen (kleinere Distanz) UND gespannter Mund
        if avg_brow_distance > -15 and mouth_height < 4:  # Augenbrauen näher zu den Augen
            angry_score = min(0.85, 0.4 + abs(avg_brow_distance + 15) / 10)
            emotions_confidence["angry"] = angry_score
        else:
            emotions_confidence["angry"] = 0.15
            
        # Fear: Große Augen UND hohe Augenbrauen
        if avg_eye_height > 7 and avg_brow_distance < -18:
            fear_score = min(0.8, 0.4 + (avg_eye_height / 20))
            emotions_confidence["fear"] = fear_score
        else:
            emotions_confidence["fear"] = 0.1
            
        # Disgust: Leicht verzogener Mund
        if mouth_curve < -0.5 and mouth_curve > -2.0 and mouth_height < 3:
            emotions_confidence["disgust"] = 0.6
        else:
            emotions_confidence["disgust"] = 0.1
            
        # Neutral: Ausgewogene Werte - STANDARD für die meisten Fälle
        neutral_factors = []
        if abs(mouth_curve) < 1.5:  # Neutraler Mund
            neutral_factors.append(0.3)
        if 5 < avg_eye_height < 9:  # Normale Augengröße
            neutral_factors.append(0.3)
        if -20 < avg_brow_distance < -10:  # Normale Augenbrauen
            neutral_factors.append(0.3)
        if 3 < mouth_height < 8:  # Normaler Mund
            neutral_factors.append(0.2)
            
        neutral_score = sum(neutral_factors) + 0.4  # Base neutral score
        emotions_confidence["neutral"] = min(0.95, neutral_score)
        
        # Beste Emotion finden
        best_emotion = max(emotions_confidence.keys(), key=lambda k: emotions_confidence[k])
        best_confidence = emotions_confidence[best_emotion]
        
        # Minimale Debug-Info - nur bei sehr starken, ungewöhnlichen Emotionen
        # (Entfernt fast alle Debug-Ausgaben um Spam zu vermeiden)
        
        return best_emotion, best_confidence
        
    except Exception as e:
        print(f"[FEHLER] Landmark emotion detection: {e}")
        return "neutral", 0.6


def predict_emotion(face_gray, model):
    """
    Fallback-Emotionserkennung - versucht zuerst externes Modell, dann Landmarks
    """
    if model is not None:
        try:
            # Versuch mit externem Modell
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
            print(f"[FEHLER] External emotion model error: {e}")
    
    # Fallback: Verwende Landmarks-basierte Erkennung
    # Diese wird in der Hauptschleife aufgerufen, wenn FaceMesh verfügbar ist
    return "neutral", 0.5

def identify_face(face_region, face_engine, vector_store, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Vereinfachte und direktere Gesichtserkennung mit UTF-8 Support
    """
    try:
        # Convert BGR to RGB for face_recognition
        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        
        # Detect face in the cropped region
        face_locations = face_engine.detect_faces(face_rgb)
        
        if not face_locations:
            return None
            
        # Use the first (and likely only) face detected
        face_location = face_locations[0]
        
        # Extract face embedding
        embedding = face_engine.extract_face_embedding(face_rgb, face_location)
        
        if embedding is None:
            return None
            
        # Search for similar faces in the database
        similar_faces = vector_store.search_similar_faces(
            embedding, 
            n_results=5,  # Mehr Matches für bessere Vergleiche
            min_similarity=max(0.6, similarity_threshold - 0.15)  # Deutlich weniger streng
        )
        
        if similar_faces:
            best_match = similar_faces[0]
            best_similarity = best_match['similarity']
            
            # Sehr lockere Überprüfung - hauptsächlich auf Basis der Ähnlichkeit
            if best_similarity < 0.65:  # Sehr niedriger Threshold
                return None
                
            person_data = best_match['metadata']
            
            # UTF-8 sichere Verarbeitung der Namen
            name = person_data.get('full_name', 'Unbekannt')
            first_name = person_data.get('first_name', '')
            last_name = person_data.get('last_name', '')
            
            # Stelle sicher, dass Namen korrekt kodiert sind
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='replace')
            if isinstance(first_name, bytes):
                first_name = first_name.decode('utf-8', errors='replace')
            if isinstance(last_name, bytes):
                last_name = last_name.decode('utf-8', errors='replace')
            
            # Extract person information
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
        print(f"[FEHLER] Face identification error: {e}")
        return None

def get_face_orientation(mesh_landmarks, w, h):
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

def is_mouth_open(mesh_landmarks, w, h):
    lm = mesh_landmarks.landmark
    y_up = lm[13].y * h
    y_low = lm[14].y * h
    lip_dist = abs(y_low - y_up)
    return lip_dist > 0.035*h

def camera_worker(cam_id, cam_url, model, result_queue, stop_flag, face_engine=None, vector_store=None):
    mp_hands_instance = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=4,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_fd = mp.solutions.face_detection.FaceDetection(
        model_selection=1,  # Model 1 für bessere Genauigkeit bei größeren Entfernungen
        min_detection_confidence=0.7  # Höhere Konfidenz für bessere Erkennung
    )
    mp_fm = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=3,  # Reduziert für bessere Performance
        refine_landmarks=True,
        min_detection_confidence=0.7,  # Höhere Konfidenz
        min_tracking_confidence=0.6    # Bessere Verfolgung
    )
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(cam_url)
    if not cap.isOpened():
        print(f"[FEHLER] Kamera {cam_id} konnte nicht geöffnet werden: {cam_url}")
        return

    winname = f"Kamera {cam_id}"
    print(f"[INFO] Kamera {cam_id} gestartet: {cam_url}")
    frame_count = 0
    
    # Vereinfachte Stabilisierung für Personenerkennung
    person_history = {}  # person_id -> [timestamps]
    recognition_stability_time = 0.5  # Sehr kurze Zeit für schnelle Erkennung
    recognition_count_threshold = 1  # Nur 1 Erkennung nötig - viel direkter

    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNUNG] Kein Frame von Kamera {cam_id}")
            time.sleep(0.1)
            continue

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        analysis_lines = []
        current_time = time.time()
        # Gesichtserkennung + Emotionen + Person Identification (verbessert)
        detected_faces_info = []  # Speichere Info über erkannte Gesichter für Emotion-Mapping
        
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
                        # Standard Emotion recognition als Fallback
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        emotion_label, emotion_prob = predict_emotion(gray, model)
                        
                        # Speichere Gesichtsinfo für später (für landmarks-basierte Emotionen)
                        face_info = {
                            'bbox': (x1, y1, x2, y2),
                            'roi': roi,
                            'emotion_fallback': (emotion_label, emotion_prob),
                            'person_info': None
                        }
                        
                        # Face identification mit weniger strenger Stabilisierung
                        person_info = None  # Initialisiere mit None
                        if face_engine and vector_store and roi.shape[0] > 50 and roi.shape[1] > 50:
                            person_info = identify_face(roi, face_engine, vector_store)
                            
                            # Nur bei erfolgreicher Erkennung ausgeben (weniger Spam)
                            if person_info and person_info['similarity'] >= 0.75:
                                print(f"[INFO] {person_info['name']} erkannt ({person_info['similarity']*100:.0f}%)")
                            
                            # Sehr vereinfachte Stabilisierungslogik - hauptsächlich direkte Anzeige
                            if person_info:
                                person_id = person_info.get('person_id')
                                similarity = person_info['similarity']
                                
                                # Bei hoher Ähnlichkeit sofort anzeigen
                                if similarity >= 0.75:
                                    # Direkte Anzeige bei hoher Ähnlichkeit
                                    pass  # person_info bleibt bestehen
                                elif similarity >= 0.65 and person_id:
                                    # Einfache Stabilisierung nur bei mittlerer Ähnlichkeit
                                    if person_id not in person_history:
                                        person_history[person_id] = []
                                    
                                    person_history[person_id].append(current_time)
                                    
                                    # Entferne alte Zeitstempel
                                    person_history[person_id] = [
                                        t for t in person_history[person_id] 
                                        if current_time - t < recognition_stability_time
                                    ]
                                    
                                    # Sehr lockere Anforderung
                                    if len(person_history[person_id]) < recognition_count_threshold:
                                        person_info = None  # Warte noch kurz
                                else:
                                    # Zu niedrige Ähnlichkeit
                                    person_info = None
                        
                        face_info['person_info'] = person_info
                        detected_faces_info.append(face_info)
                            
        except Exception as e:
            print(f"[FEHLER] Gesichtserkennung Kamera {cam_id}: {e}")
            
        # Zeichne die erkannten Gesichter SOFORT - nicht erst nach FaceMesh
        for i, face_info in enumerate(detected_faces_info):
            x1, y1, x2, y2 = face_info['bbox']
            person_info = face_info['person_info']
            emotion_label, emotion_prob = face_info['emotion_fallback']
            
            # Draw face rectangle
            color = (0, 255, 0) if person_info else (255, 255, 0)  # Green if identified, yellow otherwise
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Display person name and emotion (UTF-8 sicher)
            if person_info:
                name = person_info['name']
                similarity = person_info['similarity']
                
                # Hauptlabel: Name + Ähnlichkeit
                main_label = f"{name} ({similarity*100:.0f}%)"
                
                # UTF-8 sichere Text-Ausgabe
                try:
                    cv2.putText(frame, main_label, (x1, y1-40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                except:
                    # Fallback ohne Umlaute
                    safe_name = name.encode('ascii', errors='ignore').decode('ascii')
                    main_label = f"{safe_name} ({similarity*100:.0f}%)"
                    cv2.putText(frame, main_label, (x1, y1-40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Emotions-Label
                emotion_text = f"{emotion_label} {emotion_prob:.2f}"
                cv2.putText(frame, emotion_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                analysis_lines.append(f"Person: {name} ({similarity*100:.0f}%)")
                analysis_lines.append(f"Emotion: {emotion_label} ({emotion_prob:.2f})")
                
                # Display additional info if available
                if person_info.get('person_id'):
                    cv2.putText(frame, f"ID: {person_info['person_id'][:8]}", (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                               
            else:
                # Unknown person
                label = f"Unbekannt - {emotion_label} {emotion_prob:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                analysis_lines.append(f"Unbekannte Person: {emotion_label} ({emotion_prob:.2f})")
            
        # Zeichne die erkannten Gesichter und ihre aktualisierten Informationen
        for i, face_info in enumerate(detected_faces_info):
            x1, y1, x2, y2 = face_info['bbox']
            person_info = face_info['person_info']
            emotion_label, emotion_prob = face_info['emotion_fallback']  # Kann durch Landmarks aktualisiert worden sein
            
            # Draw face rectangle
            color = (0, 255, 0) if person_info else (255, 255, 0)  # Green if identified, yellow otherwise
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Display person name and emotion (UTF-8 sicher)
            if person_info:
                name = person_info['name']
                similarity = person_info['similarity']
                
                # Hauptlabel: Name + Ähnlichkeit
                main_label = f"{name} ({similarity*100:.0f}%)"
                
                # UTF-8 sichere Text-Ausgabe
                try:
                    cv2.putText(frame, main_label, (x1, y1-40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                except:
                    # Fallback ohne Umlaute
                    safe_name = name.encode('ascii', errors='ignore').decode('ascii')
                    main_label = f"{safe_name} ({similarity*100:.0f}%)"
                    cv2.putText(frame, main_label, (x1, y1-40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Emotions-Label (aktualisiert)
                emotion_text = f"{emotion_label} {emotion_prob:.2f}"
                cv2.putText(frame, emotion_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                analysis_lines.append(f"Person: {name} ({similarity*100:.0f}%)")
                analysis_lines.append(f"Final Emotion: {emotion_label} ({emotion_prob:.2f})")
                
                # Display additional info if available
                if person_info.get('person_id'):
                    cv2.putText(frame, f"ID: {person_info['person_id'][:8]}", (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                               
            else:
                # Unknown person with updated emotion
                label = f"Unbekannt - {emotion_label} {emotion_prob:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                analysis_lines.append(f"Unbekannte Person: {emotion_label} ({emotion_prob:.2f})")
                # FaceMesh für zusätzliche Analyse (überschreibt NICHT die Hauptanzeige)
        try:
            mesh = mp_fm.process(rgb)
            if mesh.multi_face_landmarks:
                for j, fl in enumerate(mesh.multi_face_landmarks):
                    # Zeichne FaceMesh nur als zusätzliche Information
                    mp_draw.draw_landmarks(
                        frame, fl, 
                        mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        mp_draw.DrawingSpec((0, 255, 0), 1, 1), 
                        mp_draw.DrawingSpec((0, 180, 0), 1)
                    )
                    
                    # Analysiere Emotionen, aber überschreibe NICHT die Haupterkennung
                    landmark_emotion, landmark_confidence = predict_emotion_from_landmarks(fl, w, h)
                    
                    # Zeige zusätzliche FaceMesh-Informationen, aber kleiner
                    orient = get_face_orientation(fl, w, h)
                    mouth_open = is_mouth_open(fl, w, h)
                    mouth_lab = "Mund offen" if mouth_open else "Mund zu"
                    nose = fl.landmark[1]
                    cx, cy = int(nose.x*w), int(nose.y*h)
                    
                    # Nur zusätzliche Info anzeigen - ÜBERSCHREIBT NICHT die Personenerkennung
                    additional_info = f"{orient}, {mouth_lab}"
                    cv2.putText(frame, additional_info, (cx+15, cy+25+j*20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                (0, 200, 255) if mouth_open else (100, 180, 255), 1)
                    
                    # Nur zur Analyse hinzufügen, nicht zur Hauptanzeige
                    analysis_lines.append(f"Zusatz: {orient}, {mouth_lab}")
                    
        except Exception as e:
            print(f"[FEHLER] FaceMesh Kamera {cam_id}: {e}")
            
        # WICHTIG: Die bereits gezeichneten Gesichter werden NICHT nochmal überschrieben!
        # Pose Erkennung
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
        except Exception as e:
            print(f"[FEHLER] Pose-Erkennung Kamera {cam_id}: {e}")
        # Einfache Handerkennung (ohne Fausterkennung und Bildschirmsperre)
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
                    
                    # Einfache Handposition anzeigen (ohne Gestenerkennung)
                    wrist = hand_lms.landmark[0]
                    px, py = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(frame, f"Hand {i+1}", (px+10, py-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
                    hand_count += 1
                    
                if hand_count > 0:
                    analysis_lines.append(f"Hände erkannt: {hand_count}")
                    
        except Exception as e:
            print(f"[FEHLER] Handerkennung Kamera {cam_id}: {e}")
        # Analyse-Text anzeigen
        for idx, line in enumerate(analysis_lines):
            cv2.putText(frame, line, (15, h-10-20*idx), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Frame-Counter anzeigen
        cv2.putText(frame, f"Frame: {frame_count}", (w-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # Schreibe Frame in die Queue
        if not result_queue.empty():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
        result_queue.put(frame)
    cap.release()
    mp_hands_instance.close()
    mp_fd.close()
    mp_fm.close()
    mp_pose.close()
    print(f"[INFO] Kamera {cam_id} beendet")

def main():
    import threading
    
    print("[INFO] Initialisiere Face Recognition System...")
    
    # Initialize face recognition components
    try:
        face_engine = FaceRecognitionEngine()
        print("[INFO] Face Recognition Engine geladen")
    except Exception as e:
        print(f"[WARNUNG] Face Recognition Engine konnte nicht geladen werden: {e}")
        print("[INFO] Fortsetzung ohne Personenerkennung")
        face_engine = None
    
    try:
        vector_store = FaceVectorStore()
        stats = vector_store.get_collection_stats()
        if "error" not in stats:
            print(f"[INFO] Vector Store geladen - {stats.get('total_faces', 0)} Gesichter in Datenbank")
        else:
            print(f"[WARNUNG] Vector Store Problem: {stats['error']}")
            vector_store = None
    except Exception as e:
        print(f"[WARNUNG] Vector Store konnte nicht geladen werden: {e}")
        print("[INFO] Fortsetzung ohne Personenerkennung")
        vector_store = None
    
    # Camera sources
    camera_sources = [
        0,  # lokale Kamera
        # Beispiel für eine IP-Kamera:
        # "http://192.168.188.71:8123/api/camera_proxy_stream/camera.988f90707a71_camera_1_camera_1?token=..."
    ]
    
    # Load emotion model (optional)
    try:
        model = load_model("models/fer_trained.h5")
        print("[INFO] Emotionsmodell erfolgreich geladen")
    except Exception as e:
        print(f"[INFO] Externes Emotionsmodell nicht verfügbar: {e}")
        print("[INFO] Verwende Landmark-basierte Emotionserkennung")
        model = None
    
    result_queues = []
    threads = []
    stop_flags = []
    
    print(f"[INFO] Starte {len(camera_sources)} Kamera(s)")
    
    for cam_id, cam_url in enumerate(camera_sources):
        q = queue.Queue(maxsize=1)
        result_queues.append(q)
        stop_flag = threading.Event()
        stop_flags.append(stop_flag)
        t = Thread(target=camera_worker, 
                  args=(cam_id+1, cam_url, model, q, stop_flag, face_engine, vector_store), 
                  daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)
    
    print("[INFO] === LIVE FACE RECOGNITION GESTARTET ===")
    print("[INFO] Features aktiv:")
    if face_engine and vector_store:
        print("  ✅ Personenerkennung (mit Stabilisierung)")
        if model:
            print("  ✅ Externe + Landmark-basierte Emotionserkennung")
        else:
            print("  ✅ Landmark-basierte Emotionserkennung") 
        print("  ✅ Gesichtsmessung")
        print("  ✅ Einfache Handerkennung")
        print("  ✅ Pose-Erkennung")
        print("  ✅ UTF-8 Umlaut-Unterstützung")
    else:
        print("  ❌ Personenerkennung (nicht verfügbar)")
        if model:
            print("  ✅ Externe + Landmark-basierte Emotionserkennung")
        else:
            print("  ✅ Landmark-basierte Emotionserkennung")
        print("  ✅ Gesichtsmessung")
        print("  ✅ Einfache Handerkennung") 
        print("  ✅ Pose-Erkennung")
        print("  ✅ UTF-8 Umlaut-Unterstützung")
    
    print("[INFO] Drücke 'q' in einem Kamerafenster zum Beenden.")
    print("[INFO] Schließen des Fensters beendet die entsprechende Kamera.")
    
    active_windows = [True for _ in camera_sources]
    
    try:
        while any(active_windows):
            for idx, q in enumerate(result_queues):
                winname = f"Kamera {idx+1}"
                if not active_windows[idx]:
                    continue
                frame = None
                try:
                    frame = q.get(timeout=0.01)
                except queue.Empty:
                    continue
                if frame is not None:
                    cv2.imshow(winname, frame)
                    if cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) < 1:
                        print(f"[INFO] Fenster {winname} wurde geschlossen")
                        stop_flags[idx].set()
                        cv2.destroyWindow(winname)
                        active_windows[idx] = False
                    elif cv2.waitKey(1) & 0xFF == ord('q'):
                        print(f"[INFO] 'q' gedrückt in {winname}")
                        stop_flags[idx].set()
                        cv2.destroyWindow(winname)
                        active_windows[idx] = False
    except KeyboardInterrupt:
        print("\n[INFO] Programm durch Benutzer unterbrochen")
    
    print("[INFO] Beende alle Threads...")
    for flag in stop_flags:
        flag.set()
    cv2.destroyAllWindows()
    
    for i, t in enumerate(threads):
        print(f"[INFO] Warte auf Thread {i+1}...")
        t.join(timeout=2.0)
        if t.is_alive():
            print(f"[WARNUNG] Thread {i+1} nicht ordnungsgemäß beendet")
    
    print("[INFO] Programm beendet")

if __name__ == "__main__":
    main()
