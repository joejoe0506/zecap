# db.py
import os, json

DB_PATH = "users.json"

def _load():
    if not os.path.exists(DB_PATH): return []
    with open(DB_PATH, "r", encoding="utf-8") as f: return json.load(f)

def _save(arr): 
    with open(DB_PATH, "w", encoding="utf-8") as f: json.dump(arr, f, ensure_ascii=False, indent=2)

def get_user(user_id, password):
    for u in _load():
        if u["user_id"] == user_id and u["password"] == password: return u
    return None

def save_user(user_id, password, name, student_name, is_admin=False):
    arr = _load()
    arr = [u for u in arr if not (u["user_id"] == user_id or u["student_id"] == student_name)]
    arr.append({"user_id":user_id, "password":password, "name":name, "student_id":student_name, "is_admin":bool(is_admin)})
    _save(arr)

def load_user_name_by_id(student_name):
    for u in _load():
        if u["student_id"] == student_name: return u["name"]
    return None
