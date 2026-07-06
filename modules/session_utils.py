"""会话工具 — 权限检查、操作日志、项目路径。

供 app.py 和 pages/*.py 共同导入，避免循环依赖。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PROJ_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_DIR / "data"
CACHE_DIR = PROJ_DIR / "cache"

# ════════════════════════════════════════════════════════════════════════════
# 权限系统初始化
# ════════════════════════════════════════════════════════════════════════════
USE_DB_AUTH = False
permissions = None  # 数据库权限模块（仅 USE_DB_AUTH=True 时可用）

try:
    from modules import permissions as _perms
    from modules import db as _db_module

    USE_DB_AUTH = _db_module.ping()
    if USE_DB_AUTH:
        permissions = _perms
        print("[权限] 使用数据库权限系统")
    else:
        print("[权限] 数据库不可用，回退到文件权限系统")
        import auth as _auth

        permissions = None
except Exception as e:
    print(f"[权限] 数据库权限系统加载失败，回退到文件系统: {e}")
    import auth as _auth

    USE_DB_AUTH = False
    permissions = None


# ════════════════════════════════════════════════════════════════════════════
# 权限检查
# ════════════════════════════════════════════════════════════════════════════
def has_permission(perm: str) -> bool:
    """检查当前用户是否拥有指定权限"""
    if USE_DB_AUTH and permissions is not None:
        user = st.session_state.get("current_user")
        return user is not None and user.has_permission(perm)
    else:
        role = st.session_state.get("role", "")
        role_perms = {
            "admin": [
                "view_transactions", "view_inventory", "view_statistics",
                "view_recommendations", "generate_recommendations",
                "view_predictions", "run_predictions", "edit_inventory",
                "view_alerts", "export_reports", "view_charts",
                "manage_users", "manage_permissions", "view_logs",
            ],
            "operator": [
                "view_transactions", "view_inventory", "view_statistics",
                "view_recommendations", "generate_recommendations",
                "view_predictions", "view_alerts", "export_reports", "view_charts",
            ],
            "analyst": [
                "view_transactions", "view_inventory", "view_statistics",
                "view_recommendations", "view_predictions", "view_charts",
            ],
        }
        return perm in role_perms.get(role, [])


# ════════════════════════════════════════════════════════════════════════════
# 操作日志
# ════════════════════════════════════════════════════════════════════════════
def log_user_action(action: str) -> None:
    """记录当前用户操作"""
    username = st.session_state.get("username", "unknown")
    if USE_DB_AUTH and permissions is not None:
        permissions.log_action(username, action)
    else:
        _auth.log_action(username, action)


def get_logs(n: int = 50):
    """获取最近 n 条操作日志（旧文件系统）"""
    if USE_DB_AUTH and permissions is not None:
        return permissions.get_recent_logs(n)
    else:
        from auth import get_logs as _get_logs

        return _get_logs(n)
