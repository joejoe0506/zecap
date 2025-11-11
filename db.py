# db.py
import os, json
from passlib.hash 
import bcrypt

DB_PATH = "users.json"

def _load():
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save(arr):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

def _is_hashed(pw: str) -> bool:
    # bcrypt 해시 문자열은 보통 "$2b$" 또는 "$2y$" 등으로 시작합니다.
    return isinstance(pw, str) and pw.startswith("$2")

def get_user(user_id, password):
    """
    아이디+평문 비밀번호를 받아서, 저장된 해시와 비교. 성공하면 사용자 dict 반환.
    """
    for u in _load():
        if u.get("user_id") == user_id:
            stored = u.get("password", "")
            # 만약 데이터베이스에 평문이 남아있다면(구버전), 바로 검증 가능하되
            # 평문이면 우선 비교 후 해시로 교체(마이그레이션)
            if _is_hashed(stored):
                try:
                    if bcrypt.verify(password, stored):
                        return u
                except Exception:
                    return None
            else:
                # stored가 평문(구버전)인 경우: 비교 후 해시로 업데이트
                if password == stored:
                    # 즉시 해시로 교체해서 저장
                    _rehash_password_for_user(user_id, password)
                    # 반환할 때는 업데이트된 사용자 dict을 가져오기 위해 재로딩
                    for uu in _load():
                        if uu.get("user_id") == user_id:
                            return uu
                else:
                    return None
    return None

def _rehash_password_for_user(user_id, plain_password):
    users = _load()
    changed = False
    for u in users:
        if u.get("user_id") == user_id:
            u["password"] = bcrypt.hash(plain_password)
            changed = True
            break
    if changed:
        _save(users)

def save_user(user_id, password, name, student_id, is_admin=False):
    """
    비밀번호는 항상 해시로 저장합니다.
    동일 user_id 또는 student_id가 있으면 덮어씁니다.
    """
    users = _load()
    users = [u for u in users if not (u.get("user_id") == user_id or u.get("student_id") == student_id)]
    pw_hash = password
    if not _is_hashed(password):
        pw_hash = bcrypt.hash(password)
    users.append({
        "user_id": user_id,
        "password": pw_hash,
        "name": name,
        "student_id": student_id,
        "is_admin": bool(is_admin)
    })
    _save(users)

def load_user_name_by_id(student_id):
    for u in _load():
        if u.get("student_id") == student_id:
            return u.get("name")
    return None
