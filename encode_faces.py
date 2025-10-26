# encode_faces.py
import os, pickle
import face_recognition
from config import DATASET_DIR, ENCODINGS_DIR, ENCODING_EXT

VALID_EXTS = (".jpg", ".jpeg", ".png")

def encode_faces_for(person_name: str):
    folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(folder):
        print(f"[ERROR] 폴더 없음: {folder}"); return

    encs = []
    print(f"[INFO] 인코딩 시작: {person_name}")
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(VALID_EXTS): continue
        path = os.path.join(folder, fn)
        try:
            img = face_recognition.load_image_file(path)
        except Exception as e:
            print(f"[WARN] 열기 실패 {fn}: {e}"); continue
        boxes = face_recognition.face_locations(img)
        if len(boxes) != 1:
            print(f"[WARN] 얼굴 수 {len(boxes)}: {fn}"); continue
        enc = face_recognition.face_encodings(img, boxes)[0]
        encs.append(enc); print(f"[ENCODED] {fn}")

    if not encs:
        print(f"[ERROR] 유효한 얼굴 없음: {person_name}"); return

    out = os.path.join(ENCODINGS_DIR, f"{person_name}{ENCODING_EXT}")
    with open(out, "wb") as f: pickle.dump(encs, f)
    print(f"[SUCCESS] {person_name}: {len(encs)}개 → {out}")
