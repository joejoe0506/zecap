# app.py
import os
import io
import json
import threading
import time
import pickle
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import face_recognition

# ===== 너희 기존 모듈들과의 연결 (있으면 import, 없으면 데모 대체) =====
try:
    from db import get_user, save_user, load_user_name_by_id
except Exception:
    def get_user(user_id, password):
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
        # 중복 체크 간단 처리: 같은 user_id 또는 student_id면 덮어쓰기
        arr = [x for x in arr if not (x.get("user_id") == user_id or x.get("student_id") == student_id)]
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
            if it.get("student_id") == student_id:
                return it.get("name")
        return None

# ===== 경로/상수 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "faces")      # 학생별 원본 이미지 저장
ENC_DIR   = os.path.join(BASE_DIR, "encodings")  # 학생별 face encodings 저장(pickle)
LOG_DIR   = os.path.join(BASE_DIR, "logs")

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(ENC_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ===== 유틸: 인코딩 저장/로드 및 인식 =====
def encode_student_folder(student_id: str):
    """
    faces/{student_id} 폴더의 모든 이미지 -> face_recognition encoding 추출 후 encodings/{student_id}.pkl 저장
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

# ===== FastAPI 초기화 =====
app = FastAPI()

# 정적 파일 (SPA)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# 편의를 위한 CORS(동일 도메인에서만 쓸 거면 제한 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 정확 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 라우트 : 기존 업로드 기반 기능들 ----------

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

# ===== 라즈베리파이(서버) 카메라 관련 기능 추가 시작 =====

# ===== Pi 카메라 전역 상태/설정 =====
PI_CAM_INDEX = 0                 # 기본 카메라 인덱스 (필요시 1로 바꿔)
PI_CAM_WIDTH = 640
PI_CAM_HEIGHT = 480
# OpenCV flag (플랫폼에 따라 CAP_V4L2 또는 0 사용)
try:
    PI_CAM_OPEN_FLAGS = cv2.CAP_V4L2
except Exception:
    PI_CAM_OPEN_FLAGS = 0

_cam = None
_cam_lock = threading.Lock()

# ===== 라즈베리파이 카메라 유틸 =====
def _open_pi_cam():
    global _cam
    if _cam is not None and _cam.isOpened():
        return _cam
    cam = cv2.VideoCapture(PI_CAM_INDEX, PI_CAM_OPEN_FLAGS)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, PI_CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, PI_CAM_HEIGHT)
    # 카메라 웜업: 한 프레임 읽어보고 실패하면 예외
    ok, _ = cam.read()
    if not ok:
        try:
            cam.release()
        except Exception:
            pass
        raise RuntimeError("라즈베리파이 카메라를 열 수 없습니다. 연결/권한/raspi-config 확인 필요")
    _cam = cam
    return _cam

def _close_pi_cam():
    global _cam
    if _cam is not None:
        try:
            _cam.release()
        except Exception:
            pass
        _cam = None

def _grab_frame(timeout_sec: float = 2.0):
    """
    카메라에서 1프레임 캡처. timeout 내 실패 시 예외.
    """
    start = time.time()
    while time.time() - start < timeout_sec:
        ok, frame = _cam.read() if _cam else (False, None)
        if ok and frame is not None:
            return frame
        time.sleep(0.02)
    raise RuntimeError("카메라 프레임 캡처 실패")

def _capture_n_frames(n: int = 10, interval_ms: int = 120):
    """
    n장의 프레임을 캡처해서 리스트로 반환. 간격은 interval_ms.
    """
    frames = []
    for _ in range(max(1, n)):
        frame = _grab_frame()
        frames.append(frame)
        time.sleep(max(0, interval_ms) / 1000.0)
    return frames

# ===== 카메라 직접 사용 API들 =====

@app.post("/api/register_pi")
async def api_register_pi(
    user_id: str = Form(...),
    password: str = Form(...),
    student_id: str = Form(...),   # 여기서는 '이름' 또는 '학번' 등 식별자
    name: str = Form(...),
    is_admin: bool = Form(False),
    num_frames: int = Form(10),
    interval_ms: int = Form(120),
):
    """
    서버(라즈베리파이) 카메라를 이용해 직접 촬영 후 등록.
    기존 /api/register(브라우저 업로드 기반)와 병행 사용 가능.
    """
    # 1) 사용자 저장
    save_user(user_id, password, name, student_id, is_admin)

    # 2) 카메라로 촬영해서 faces/{student_id}/ 에 저장
    sid_dir = os.path.join(FACES_DIR, student_id)
    os.makedirs(sid_dir, exist_ok=True)

    saved = 0
    with _cam_lock:
        cam = _open_pi_cam()
        try:
            frames = _capture_n_frames(n=num_frames, interval_ms=interval_ms)
            for idx, frame in enumerate(frames):
                out = os.path.join(sid_dir, f"{idx:03d}.jpg")
                cv2.imwrite(out, frame)
                saved += 1
        finally:
            # 안정성을 위해 카메라 닫기 (원하면 닫지 않고 유지 가능)
            _close_pi_cam()

    if saved < 3:
        raise HTTPException(400, "얼굴 이미지가 충분하지 않습니다. (최소 3장 권장)")

    # 3) 인코딩 생성
    encode_student_folder(student_id)
    return {"ok": True, "saved": saved}

@app.post("/api/recognize_pi")
async def api_recognize_pi(student_id: str = Form(...)):
    """
    서버(라즈베리파이) 카메라로 한 프레임 촬영하여 student_id 인식 시도
    """
    with _cam_lock:
        cam = _open_pi_cam()
        try:
            frame = _grab_frame()
        finally:
            _close_pi_cam()

    matched = recognize_from_image(frame, student_id)
    name = load_user_name_by_id(student_id) or "Unknown"
    write_attendance_log(student_id, name, matched)
    return {"ok": True, "matched": bool(matched), "name": name, "student_id": student_id}

@app.get("/api/cam_check")
def api_cam_check():
    with _cam_lock:
        try:
            cam = _open_pi_cam()
            frame = _grab_frame()
            h, w = frame.shape[:2]
            return {"ok": True, "resolution": [w, h]}
        except Exception as e:
            return {"ok": False, "message": str(e)}
        finally:
            _close_pi_cam()

# ===== End of file =====
