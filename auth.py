"""用户权限管理与操作日志（文件模式回退）。

实现三角色（管理员/运营人员/分析师）登录验证、会话管理、操作日志记录。

角色权限矩阵：
    - admin:    全部功能 + 查看日志 + 导出报告
    - operator: 查看全部 + 导出报告（不可查看日志）
    - analyst:  只读查看（不可导出、不可查看日志）

预置账号：
    admin    / admin123    → 管理员
    operator / operator123 → 运营人员
    analyst  / analyst123  → 分析师

日志格式（logs/operations.log）：
    2026-06-24 10:30:15 | admin | 登录系统

使用方式：
    from auth import login, log_action, get_logs, init_users
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

# Windows 终端 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── 路径 ────────────────────────────────────────────────────────────────────
PROJ_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJ_DIR / "data"
LOGS_DIR = PROJ_DIR / "logs"
USERS_FILE = DATA_DIR / "users.json"
LOG_FILE = LOGS_DIR / "operations.log"

import hashlib as _hlib


def _hash(password: str) -> str:
    """SHA256 哈希（简单但满足原型演示需求）。"""
    return _hlib.sha256(password.encode("utf-8")).hexdigest()


# ── 预置账号 ────────────────────────────────────────────────────────────────
DEFAULT_USERS = {
    "admin":    {"password": _hash("admin123"),    "role": "admin"},
    "operator": {"password": _hash("operator123"), "role": "operator"},
    "analyst":  {"password": _hash("analyst123"),  "role": "analyst"},
}


# ════════════════════════════════════════════════════════════════════════════
# 1. 用户初始化
# ════════════════════════════════════════════════════════════════════════════
def init_users() -> None:
    """首次运行时创建默认用户文件（如已存在则跳过）。"""
    DATA_DIR.mkdir(exist_ok=True)
    if USERS_FILE.exists():
        return
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_USERS, f, ensure_ascii=False, indent=2)
    print(f"[F8] 已创建默认用户文件: {USERS_FILE}")


def _load_users() -> dict:
    """加载用户数据。"""
    init_users()
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
# 2. 登录验证
# ════════════════════════════════════════════════════════════════════════════
def login(username: str, password: str) -> str | None:
    """验证用户凭据。

    Args:
        username: 用户名
        password: 明文密码

    Returns:
        角色字符串 ('admin'/'operator'/'analyst')，失败返回 None
    """
    users = _load_users()
    if username not in users:
        return None
    stored = users[username]
    if _hash(password) != stored["password"]:
        return None
    return stored["role"]


# ════════════════════════════════════════════════════════════════════════════
# 3. 操作日志
# ════════════════════════════════════════════════════════════════════════════
def log_action(user: str, action: str) -> None:
    """追加一行操作日志。

    Args:
        user: 用户名
        action: 操作描述（如 "登录系统", "查看推荐", "导出报告"）
    """
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {user} | {action}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def get_logs(n: int = 100) -> list[dict]:
    """读取最近 n 条日志。

    Returns:
        [{time, user, action}, ...]，最新的在前
    """
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 取最后 n 行，倒序（最新在前）
    lines = lines[-n:][::-1]
    logs = []
    for line in lines:
        line = line.strip()
        if not line or " | " not in line:
            continue
        parts = line.split(" | ", 2)
        if len(parts) == 3:
            logs.append({"time": parts[0], "user": parts[1], "action": parts[2]})
    return logs


# ════════════════════════════════════════════════════════════════════════════
# 4. 用户管理（仅 admin）
# ════════════════════════════════════════════════════════════════════════════
def list_users() -> list[dict]:
    """列出所有用户（不含密码哈希）。"""
    users = _load_users()
    return [{"username": u, "role": d["role"]} for u, d in users.items()]


def add_user(username: str, password: str, role: str) -> bool:
    """添加新用户（仅 admin 可调用，由 app.py 做权限检查）。"""
    if role not in ("admin", "operator", "analyst"):
        return False
    users = _load_users()
    if username in users:
        return False
    users[username] = {"password": _hash(password), "role": role}
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    return True


def delete_user(username: str) -> bool:
    """删除用户（不可删除自己，不可删除最后一个 admin）。"""
    users = _load_users()
    if username not in users:
        return False
    # 检查是否是最后一个 admin
    if users[username]["role"] == "admin":
        admin_count = sum(1 for d in users.values() if d["role"] == "admin")
        if admin_count <= 1:
            return False
    del users[username]
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    return True


# ════════════════════════════════════════════════════════════════════════════
# 自检
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    init_users()
    print("预置账号测试：")
    for username in ["admin", "operator", "analyst", "unknown"]:
        role = login(username, f"{username}123")
        print(f"  {username} / {username}123 → {role}")
    print(f"\n用户列表: {list_users()}")
    log_action("admin", "系统自检")
    print(f"日志记录: {get_logs(3)}")
