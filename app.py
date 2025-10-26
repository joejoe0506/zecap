# app.py
import os
import io
import json
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import face_recognition

# ===== 너희 기존 모듈들과의 연결 =====
# 아래 함수들은 너희가 이미 가지고 있는 형태에 맞춰 사용.
# (없으면 최소 동작을 위한 더미 구현을 아래에 넣어두었어.)
try:
    from db import get_user, save_user, load_user_name_by_id
except ImportError:
    def get_user(user_id, password):
        # DEMO only
        db_path = "users.json"
        if not os.path.exists(db_path):
            return None
        with open(db_path, "r", encoding="utf-8") as f:
            u = json.load(f)
        for it in u:
            if it["user_id"] == user_id and it["password"] == password:
                return it
        return None

    def save_user(user_id, password, name, student_id, is_admin=False):
        db_path = "users.json"
        arr = []
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                arr = json.load(f)
        # 중복 체크 간단 처리
        arr = [x for x in arr if not (x["user_id"] == user_id or x["student_id"] == student_id)]
        arr.append({
            "user_id": user_id, "password": password, "name": name,
            "student_id": student_id, "is_admin": bool(is_admin)
        })
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(arr, f, ensure_ascii=False, indent=2)

    def load_user_name_by_id(student_id):
        db_path = "users.json"
        if not os.path.exists(db_path):
            return None
        with open(db_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        for it in arr:
            if it["student_id"] == student_id:
                return it["name"]
        return None

# ===== 경로/상수 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "faces")      # 학생별 원본 이미지 저장
ENC_DIR   = os.path.join(BASE_DIR, "encodings")  # 학생별 face encodings 저장(pickle)
LOG_DIR   = os.path.join(BASE_DIR, "logs")

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(ENC_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ===== 유틸: 인코딩 저장/로드 =====
import pickle

def encode_student_folder(student_id: str):
    """
    faces/{sid} 폴더의 모든 이미지 -> face_recognition encoding 추출 후 encodings/{sid}.pkl 저장
    """
    sid_dir = os.path.join(FACES_DIR, student_id)
    if not os.path.isdir(sid_dir):
        raise RuntimeError("해당 학생의 얼굴 폴더가 없습니다.")

    encodings = []
    for name in sorted(os.listdir(sid_dir)):
        path = os.path.join(sid_dir, name)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(img_rgb, model="hog")  # cpu 환경 우선
        if not boxes:
            continue
        encs = face_recognition.face_encodings(img_rgb, boxes)
        encodings.extend(encs)

    if not encodings:
        raise RuntimeError("인코딩될 얼굴이 없습니다. 더 선명한 이미지를 등록하세요.")

    with open(os.path.join(ENC_DIR, f"{student_id}.pkl"), "wb") as f:
        pickle.dump(encodings, f)

def load_student_encodings(student_id: str):
    path = os.path.join(ENC_DIR, f"{student_id}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)  # List[np.ndarray]

def recognize_from_image(image_bgr: np.ndarray, student_id: str, tolerance: float = 0.5) -> bool:
    """단일 업로드 이미지에서 student_id의 인코딩과 매칭되는지 확인"""
    encs = load_student_encodings(student_id)
    if not encs:
        return False
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(img_rgb, model="hog")
    if not boxes:
        return False
    unknowns = face_recognition.face_encodings(img_rgb, boxes)
    # 하나라도 매칭되면 True
    for u in unknowns:
        matches = face_recognition.compare_faces(encs, u, tolerance=tolerance)
        if any(matches):
            return True
    return False

def write_attendance_log(student_id: str, name: str, ok: bool):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "OK" if ok else "FAIL"
    line = f"[{ts}] {student_id} {name} {status}"
    fname = datetime.now().strftime("%Y%m%d") + ".log"
    with open(os.path.join(LOG_DIR, fname), "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ===== FastAPI =====
app = FastAPI()

# 정적 파일 (SPA)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# 편의를 위한 CORS(동일 도메인에서만 쓸 거면 생략 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 정확 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 라우트 ----------

@app.get("/")
def root():
    # /static/index.html 제공
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.post("/api/login")
async def api_login(payload: dict):
    user_id = payload.get("user_id","")
    password = payload.get("password","")
    user = get_user(user_id, password)
    if not user:
        raise HTTPException(401, "아이디 또는 비밀번호가 올바르지 않습니다.")
    # 실서비스에선 JWT/세션 발급. 여기선 데모로 dummy token.
    return {
        "ok": True,
        "token": "dummy",
        "name": user["name"],
        "student_id": user["student_id"],
        "is_admin": str(user["is_admin"]).lower() == "true" or bool(user["is_admin"])
    }

@app.post("/api/register")
async def api_register(
    meta: UploadFile = File(...),
    images: list[UploadFile] = File(default=[]),
):
    """
    meta: JSON (user_id, password, student_id, name, is_admin)
    images: 얼굴 이미지 여러 장(JPEG)
    """
    try:
        payload = json.loads((await meta.read()).decode("utf-8"))
    except Exception:
        raise HTTPException(400, "meta(JSON) 파싱 실패")

    user_id = payload.get("user_id","").strip()
    password = payload.get("password","")
    student_id = payload.get("student_id","").strip()
    name = payload.get("name","").strip()
    is_admin = bool(payload.get("is_admin", False))

    if not all([user_id, password, student_id, name]):
        raise HTTPException(400, "모든 항목을 입력해주세요.")

    # 사용자 저장(중복 처리 등은 db.py에서)
    save_user(user_id, password, name, student_id, is_admin)

    # 얼굴 이미지 저장
    sid_dir = os.path.join(FACES_DIR, student_id)
    os.makedirs(sid_dir, exist_ok=True)
    saved = 0
    for idx, file in enumerate(images):
        content = await file.read()
        if not content:
            continue
        img_array = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            continue
        # 간단 전처리(옵션): 얼굴이 실제로 보이는지 체크하고 싶다면 face_locations 검사 후만 저장
        out = os.path.join(sid_dir, f"{idx:03d}.jpg")
        cv2.imwrite(out, img)
        saved += 1

    if saved < 3:
        raise HTTPException(400, "얼굴 이미지가 충분하지 않습니다. (최소 3장 권장)")

    # 인코딩 생성(학생별 pkl)
    encode_student_folder(student_id)
    return {"ok": True, "saved": saved}

@app.post("/api/recognize")
async def api_recognize(
    image: UploadFile = File(...),
    student_id: str = Form(default=""),
):
    content = await image.read()
    if not content:
        raise HTTPException(400, "이미지 업로드가 비어있습니다.")
    img_array = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "이미지 디코딩 실패")

    if not student_id:
        raise HTTPException(400, "student_id 가 필요합니다.")

    matched = recognize_from_image(frame, student_id)
    name = load_user_name_by_id(student_id) or "Unknown"
    write_attendance_log(student_id, name, matched)
    return {
        "ok": True,
        "matched": bool(matched),
        "name": name,
        "student_id": student_id
    }

@app.get("/api/logs")
def api_logs():
    logs = []
    if os.path.exists(LOG_DIR):
        for fname in sorted(os.listdir(LOG_DIR)):
            path = os.path.join(LOG_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                logs.append(f.read())
    return {"ok": True, "logs": logs}
