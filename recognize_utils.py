# recognize_utils.py
import os, pickle
import numpy as np
import face_recognition
from config import ENCODINGS_DIR, ENCODING_EXT

def load_encodings_for(person_name: str):
    p = os.path.join(ENCODINGS_DIR, f"{person_name}{ENCODING_EXT}")
    if not os.path.exists(p): return None
    with open(p, "rb") as f: return pickle.load(f)

def recognize_from_image(image_bgr: np.ndarray, person_name: str, tolerance: float = 0.5) -> bool:
    known = load_encodings_for(person_name)
    if not known: return False
    img_rgb = image_bgr[:, :, ::-1]
    boxes = face_recognition.face_locations(img_rgb)
    if not boxes: return False
    unknowns = face_recognition.face_encodings(img_rgb, boxes)
    for u in unknowns:
        matches = face_recognition.compare_faces(known, u, tolerance=tolerance)
        if any(matches): return True
    return False
