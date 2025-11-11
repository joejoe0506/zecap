# app.py (원본 + 관리자 로그 페이지/엔드포인트 추가)
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
import ssl

# ===== 기존 db 모듈 연결 (없으면 간단 데모) =====
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
FACES_DIR = os.path.join(BASE_DIR, "faces")
ENC_DIR   = os.path.join(BASE_DIR, "encodings")
LOG_DIR   = os.path.join(BASE_DIR, "logs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(ENC_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)  # static 폴더 존재 보장

# ===== 유틸: 인코딩 저장/로드 및 인식 =====
def encode_student_folder(student_id: str):
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
        boxes = face_recognition.face_locations(img_rgb, model="hog")
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
        return pickle.load(f)

def recognize_from_image(image_bgr: np.ndarray, student_id: str, tolerance: float = 0.5) -> bool:
    encs = load_student_encodings(student_id)
    if not encs:
        return False
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(img_rgb, model="hog")
    if not boxes:
        return False
    unknowns = face_recognition.face_encodings(img_rgb, boxes)
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

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 전역 예외 처리 =====
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    print(f"Unhandled exception for {request.url}: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"ok": False, "message": "Internal Server Error"}
    )

# ===== 라우트 (JSON 보장) =====
@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/api/login")
async def api_login(payload: dict):
    try:
        user_id = payload.get("user_id","")
        password = payload.get("password","")
        user = get_user(user_id, password)
        if not user:
            return JSONResponse(status_code=401, content={"ok": False, "message": "아이디 또는 비밀번호가 올바르지 않습니다."})
        return {
            "ok": True,
            "token": "dummy",
            "name": user["name"],
            "student_id": user["student_id"],
            "is_admin": str(user["is_admin"]).lower() == "true" or bool(user["is_admin"])
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "message": str(e)})

@app.post("/api/register")
async def api_register(
    meta: UploadFile = File(...),
    images: list[UploadFile] = File(default=[]),
):
    try:
        payload = json.loads((await meta.read()).decode("utf-8"))
        user_id = payload.get("user_id","").strip()
        password = payload.get("password","")
        student_id = payload.get("student_id","").strip()
        name = payload.get("name","").strip()
        is_admin = bool(payload.get("is_admin", False))

        if not all([user_id, password, student_id, name]):
            return JSONResponse(status_code=400, content={"ok": False, "message": "모든 항목을 입력해주세요."})

        save_user(user_id, password, name, student_id, is_admin)

        # 학생 폴더 생성 및 기존 사진 확인
        sid_dir = os.path.join(FACES_DIR, student_id)
        os.makedirs(sid_dir, exist_ok=True)
        existing_files = len([f for f in os.listdir(sid_dir) if os.path.isfile(os.path.join(sid_dir,f))])
        saved = 0

        for idx, file in enumerate(images):
            content = await file.read()
            if not content:
                continue
            img_array = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                continue
            out = os.path.join(sid_dir, f"{existing_files + idx:03d}.jpg")
            cv2.imwrite(out, img)
            saved += 1

        if saved + existing_files < 3:
            return JSONResponse(status_code=400, content={"ok": False, "message": "얼굴 이미지가 충분하지 않습니다. (최소 3장 권장)"})

        # 얼굴 인코딩 생성
        encode_student_folder(student_id)
        return {"ok": True, "saved": saved, "total": saved + existing_files}

    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "message": str(e)})

@app.post("/api/recognize")
async def api_recognize(
    image: UploadFile = File(...),
    student_id: str = Form(default=""),
):
    try:
        content = await image.read()
        if not content:
            return JSONResponse(status_code=400, content={"ok": False, "message": "이미지 업로드가 비어있습니다."})
        img_array = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(status_code=400, content={"ok": False, "message": "이미지 디코딩 실패"})

        if not student_id:
            return JSONResponse(status_code=400, content={"ok": False, "message": "student_id 가 필요합니다."})

        matched = recognize_from_image(frame, student_id)
        name = load_user_name_by_id(student_id) or "Unknown"
        write_attendance_log(student_id, name, matched)
        return {
            "ok": True,
            "matched": bool(matched),
            "name": name,
            "student_id": student_id
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "message": str(e)})

@app.get("/api/logs")
def api_logs():
    try:
        logs = []
        if os.path.exists(LOG_DIR):
            for fname in sorted(os.listdir(LOG_DIR)):
                path = os.path.join(LOG_DIR, fname)
                with open(path, "r", encoding="utf-8") as f:
                    logs.append(f.read())
        return {"ok": True, "logs": logs}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "message": str(e)})

# ===== Pi 카메라 관련 설정 =====
PI_CAM_INDEX = 1
PI_CAM_WIDTH = 640
PI_CAM_HEIGHT = 480
try:
    PI_CAM_OPEN_FLAGS = cv2.CAP_V4L2
except Exception:
    PI_CAM_OPEN_FLAGS = 0

_cam = None
_cam_lock = threading.Lock()

def _open_pi_cam():
    global _cam
    if _cam is not None and _cam.isOpened():
        return _cam
    cam = cv2.VideoCapture(PI_CAM_INDEX, PI_CAM_OPEN_FLAGS)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, PI_CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, PI_CAM_HEIGHT)
    ok, _ = cam.read()
    if not ok:
        try:
            cam.release()
        except Exception:
            pass
        raise RuntimeError("라즈베리파이 카메라를 열 수 없습니다.")
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
    start = time.time()
    while time.time() - start < timeout_sec:
        ok, frame = _cam.read() if _cam else (False, None)
        if ok and frame is not None:
            print("프레임 획득 성공")
            return frame
        print("프레임 획득 실패, 재시도 중...")
        time.sleep(0.02)
    raise RuntimeError("카메라 프레임 캡처 실패")

def _capture_n_frames(n: int = 20, interval_ms: int = 100):
    frames = []
    for _ in range(max(1, n)):
        frame = _grab_frame()
        frames.append(frame)
        time.sleep(max(0, interval_ms) / 1000.0)
    return frames

@app.post("/api/register_pi")
async def api_register_pi(
    user_id: str = Form(...),
    password: str = Form(...),
    student_id: str = Form(...),
    name: str = Form(...),
    is_admin: bool = Form(False),
    num_frames: int = Form(30),
    interval_ms: int = Form(100),
):
    try:
        save_user(user_id, password, name, student_id, is_admin)
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
                _close_pi_cam()
        if saved < 3:
            return JSONResponse(status_code=400, content={"ok": False, "message": "얼굴 이미지가 충분하지 않습니다."})
        encode_student_folder(student_id)
        return {"ok": True, "saved": saved}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "message": str(e)})

# -- 기존에 추가된 improved register_pi kept below (avoid duplicate route conflict) --
@app.post("/api/register_pi")
async def api_register_pi(
    user_id: str = Form(...),
    password: str = Form(...),
    student_id: str = Form(...),
    name: str = Form(...),
    is_admin: bool = Form(False),
    num_frames: int = Form(10),
    interval_ms: int = Form(120),
):
    try:
        # 1. 유저 정보 DB에 저장
        save_user(user_id, password, name, student_id, is_admin)

        # 2. 학생 폴더 생성 (기존 사진 유지)
        sid_dir = os.path.join(FACES_DIR, student_id)
        os.makedirs(sid_dir, exist_ok=True)
        existing_files = len(os.listdir(sid_dir))
        saved = 0

        # 3. Pi 카메라로 사진 촬영
        with _cam_lock:
            cam = _open_pi_cam()
            try:
                frames = _capture_n_frames(n=num_frames, interval_ms=interval_ms)
                for idx, frame in enumerate(frames):
                    out = os.path.join(sid_dir, f"{existing_files + idx:03d}.jpg")
                    cv2.imwrite(out, frame)
                    saved += 1
            finally:
                _close_pi_cam()

        # 4. 사진이 최소 3장 이상인지 확인
        if saved + existing_files < 3:
            return JSONResponse(status_code=400, content={"ok": False, "message": "얼굴 이미지가 충분하지 않습니다."})

        # 5. 얼굴 인코딩 생성
        encode_student_folder(student_id)

        return {"ok": True, "saved": saved, "total": saved + existing_files}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "message": str(e)})

@app.get("/api/cam_check")
def api_cam_check():
    try:
        with _cam_lock:
            cam = _open_pi_cam()
            frame = _grab_frame()
            h, w = frame.shape[:2]
            return {"ok": True, "resolution": [w, h]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "message": str(e)})
    finally:
        _close_pi_cam()

# ===== 관리용 HTML 제공 및 로그 JSON API 추가 (여기서부터 새로 추가된 부분) =====

@app.get("/admin_logs")
def admin_page():
    """
    관리자용 출석 로그 페이지 제공.
    static/admin_logs.html 파일을 프로젝트의 static/ 아래에 넣어주세요.
    """
    admin_html = os.path.join(STATIC_DIR, "admin_logs.html")
    if not os.path.exists(admin_html):
        return JSONResponse(status_code=404, content={"ok": False, "message": "admin_logs.html 이 static 폴더에 없습니다."})
    return FileResponse(admin_html)

@app.get("/api/admin/logs")
def api_admin_logs():
    try:
        entries = []
        if os.path.exists(LOG_DIR):
            for fname in sorted(os.listdir(LOG_DIR), reverse=True):
                path = os.path.join(LOG_DIR, fname)
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            if line.startswith("["):
                                ts_part, rest = line.split("]", 1)
                                ts = ts_part.strip("[]").strip()
                                rest = rest.strip()
                                parts = rest.split()
                                student = parts[0] if len(parts) > 0 else ""
                                name = parts[1] if len(parts) > 1 else ""
                                status = parts[2] if len(parts) > 2 else ""
                            else:
                                ts = ""
                                student = ""
                                name = ""
                                status = ""
                            entries.append({
                                "file": fname,
                                "timestamp": ts,
                                "student_id": student,
                                "name": name,
                                "status": status,
                                "raw": line
                            })
                        except Exception:
                            entries.append({
                                "file": fname,
                                "timestamp": "",
                                "student_id": "",
                                "name": "",
                                "status": "",
                                "raw": line
                            })
        return {"ok": True, "entries": entries}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "message": str(e)})

# ===== SSL 인증서 자동 저장 폴더 설정 =====
CERT_DIR = os.path.join(BASE_DIR, "certs")
os.makedirs(CERT_DIR, exist_ok=True)

CERT_FILE = os.path.join(CERT_DIR, "localhost+1.pem")
KEY_FILE = os.path.join(CERT_DIR, "localhost+1-key.pem")

if __name__ == "__main__":
    import uvicorn
    print(f"✅ 인증서 파일 경로: {CERT_FILE}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=KEY_FILE,
        ssl_certfile=CERT_FILE
    )
