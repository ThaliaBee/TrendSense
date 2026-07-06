"""初始化权限系统（用户表 + 权限表 + 默认数据）。

做三件事：
  1. 建用户、权限、用户-权限关联表；
  2. 初始化权限列表；
  3. 创建默认管理员账号。

运行（在项目根目录）：
    python scripts/init_permissions.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from sqlalchemy import text
import hashlib

PROJ_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_DIR))

from modules import db

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ── 1. 建表 ──────────────────────────────────────────────────────────────────
DDL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_PERMISSIONS = """
CREATE TABLE IF NOT EXISTS permissions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    permission_name VARCHAR(50) UNIQUE NOT NULL,
    description VARCHAR(255),
    category VARCHAR(50)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_USER_PERMISSIONS = """
CREATE TABLE IF NOT EXISTS user_permissions (
    user_id INT,
    permission_id INT,
    granted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, permission_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def create_tables() -> None:
    eng = db.get_engine()
    with eng.connect() as conn:
        conn.execute(text(DDL_USERS))
        conn.execute(text(DDL_PERMISSIONS))
        conn.execute(text(DDL_USER_PERMISSIONS))
        conn.commit()
    print("[1/3] 权限表就绪：users, permissions, user_permissions")


# ── 2. 初始化权限列表 ────────────────────────────────────────────────────────
PERMISSION_LIST = [
    # 数据查看权限
    ("view_transactions", "查看交易记录", "数据"),
    ("view_inventory", "查看库存", "数据"),
    ("view_statistics", "查看统计数据", "数据"),
    
    # 推荐权限
    ("view_recommendations", "查看推荐结果", "推荐"),
    ("generate_recommendations", "生成推荐", "推荐"),
    
    # 预测权限
    ("view_predictions", "查看流行性预测", "预测"),
    ("run_predictions", "运行预测模型", "预测"),
    
    # 库存管理权限
    ("edit_inventory", "编辑库存", "库存"),
    ("view_alerts", "查看库存预警", "库存"),
    
    # 报告权限
    ("export_reports", "导出报告", "报告"),
    ("view_charts", "查看图表", "报告"),
    
    # 用户管理权限（仅管理员）
    ("manage_users", "管理用户", "管理"),
    ("manage_permissions", "管理权限", "管理"),
    ("view_logs", "查看操作日志", "管理"),
]


def init_permissions() -> None:
    eng = db.get_engine()
    
    # 清空权限表（可重复运行）
    with eng.connect() as conn:
        conn.execute(text("DELETE FROM user_permissions"))
        conn.execute(text("DELETE FROM permissions"))
        conn.commit()
    
    # 插入权限
    for perm_name, desc, category in PERMISSION_LIST:
        with eng.connect() as conn:
            conn.execute(text(
                "INSERT INTO permissions (permission_name, description, category) "
                "VALUES (:name, :desc, :cat)"
            ), {"name": perm_name, "desc": desc, "cat": category})
            conn.commit()
    
    print(f"[2/3] 权限初始化完成：{len(PERMISSION_LIST)} 个权限")


# ── 3. 创建默认账号 ──────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    """简单哈希（生产环境应使用 bcrypt）"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_default_users() -> None:
    eng = db.get_engine()
    
    # 管理员（拥有所有权限）
    admin_password = hash_password("admin123")
    with eng.connect() as conn:
        # 删除已存在的用户
        conn.execute(text("DELETE FROM users WHERE username = 'admin'"))
        conn.commit()
        
        # 创建管理员
        conn.execute(text(
            "INSERT INTO users (username, password) VALUES (:user, :pwd)"
        ), {"user": "admin", "pwd": admin_password})
        conn.commit()
        
        # 获取管理员ID
        result = conn.execute(text("SELECT id FROM users WHERE username = 'admin'"))
        admin_id = result.fetchone()[0]
        
        # 授予所有权限
        result = conn.execute(text("SELECT id FROM permissions"))
        perm_ids = [row[0] for row in result.fetchall()]
        
        for perm_id in perm_ids:
            conn.execute(text(
                "INSERT INTO user_permissions (user_id, permission_id) "
                "VALUES (:uid, :pid)"
            ), {"uid": admin_id, "pid": perm_id})
        conn.commit()
    
    print("[3/3] 默认账号创建成功")
    print("      管理员: admin / admin123（拥有所有权限）")


def verify() -> None:
    import pandas as pd
    eng = db.get_engine()
    
    n_users = pd.read_sql(text("SELECT COUNT(*) AS n FROM users"), eng)["n"][0]
    n_perms = pd.read_sql(text("SELECT COUNT(*) AS n FROM permissions"), eng)["n"][0]
    
    print(f"\n校验: users={n_users} 个, permissions={n_perms} 个")
    print("\n权限列表:")
    perms = pd.read_sql(text(
        "SELECT category, permission_name, description FROM permissions "
        "ORDER BY category, permission_name"
    ), eng)
    print(perms.to_string(index=False))


def main() -> None:
    print("=" * 60)
    print("初始化权限系统")
    print("=" * 60)
    
    if not db.ping():
        print("无法连接数据库，请先运行 init_db.py")
        return
    
    create_tables()
    init_permissions()
    create_default_users()
    verify()
    
    print("\n[完成] 权限系统初始化结束。")
    print("\n现在可以使用以下账号登录：")
    print("  用户名: admin")
    print("  密码: admin123")


if __name__ == "__main__":
    main()
