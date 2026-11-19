"""
Minimal stand-in for the ``face_recognition`` package (which needs dlib).

Only used by the UI tests when the real package is not installed. It exposes
exactly the surface that ``face_recognition_engine.py`` calls.
"""
from __future__ import annotations

import numpy as np


def _boxes(image):
    height, width = image.shape[:2]
    side = max(48, min(width, height) // 3)
    margin = max(8, side // 6)
    boxes = []
    for row in range(2):
        for column in range(2):
            top = margin + row * (side + margin)
            left = margin + column * (side + margin)
            bottom = min(height, top + side)
            right = min(width, left + side)
            if bottom - top >= 32 and right - left >= 32:
                boxes.append((top, right, bottom, left))
    return boxes


def face_locations(image, number_of_times_to_upsample=0, model="hog"):
    """Return deterministic face boxes that fit inside the image."""
    return _boxes(image)


def face_encodings(face_image, known_face_locations=None, num_jitters=1, model="large"):
    """Return one deterministic 128-d encoding per detected face."""
    locations = known_face_locations or _boxes(face_image)
    encodings = []
    for index in range(len(locations)):
        vector = np.linspace(0.05, 0.25, 128, dtype=np.float32) + index * 0.01
        encodings.append(vector)
    return encodings


def face_distance(known_face_encodings, face_encoding_to_check):
    return np.linalg.norm(np.asarray(known_face_encodings) - face_encoding_to_check, axis=1)


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
