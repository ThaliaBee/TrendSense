"""权限管理模块（F8 用户权限管理与日志）。

提供用户登录、权限检查、单项权限管理功能。

核心功能：
1. 用户认证（登录/登出）
2. 权限检查（检查用户是否拥有某权限）
3. 单项权限管理（管理员添加/移除用户权限）
4. 操作日志记录
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from sqlalchemy import text

from . import db

# ── 路径 ────────────────────────────────────────────────────────────────────
PROJ_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJ_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "operations.log"


# ════════════════════════════════════════════════════════════════════════════
# 1. 密码工具
# ════════════════════════════════════════════════════════════════════════════
def hash_password(password: str) -> str:
    """密码哈希（生产环境应使用 bcrypt）"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """验证密码"""
    return hash_password(password) == hashed


# ════════════════════════════════════════════════════════════════════════════
# 2. 用户认证
# ════════════════════════════════════════════════════════════════════════════
class User:
    """用户对象"""
    
    def __init__(self, user_id: int, username: str, permissions: list[str]):
        self.id = user_id
        self.username = username
        self.permissions = set(permissions)
    
    def has_permission(self, permission: str) -> bool:
        """检查是否拥有某权限"""
        return permission in self.permissions
    
    def is_admin(self) -> bool:
        """是否是管理员（拥有管理用户权限）"""
        return self.has_permission("manage_users")


def login(username: str, password: str) -> Optional[User]:
    """用户登录验证
    
    Args:
        username: 用户名
        password: 密码（明文）
    
    Returns:
        User对象（登录成功）或 None（登录失败）
    """
    try:
        eng = db.get_engine()
        
        # 1. 查询用户
        result = pd.read_sql(text(
            "SELECT id, username, password FROM users WHERE username = :user"
        ), eng, params={"user": username})
        
        if result.empty:
            log_action("SYSTEM", f"登录失败: 用户不存在 ({username})")
            return None
        
        user_row = result.iloc[0]
        
        # 2. 验证密码
        if not verify_password(password, user_row["password"]):
            log_action("SYSTEM", f"登录失败: 密码错误 ({username})")
            return None
        
        # 3. 加载用户权限
        perms = pd.read_sql(text("""
            SELECT p.permission_name
            FROM user_permissions up
            JOIN permissions p ON up.permission_id = p.id
            WHERE up.user_id = :uid
        """), eng, params={"uid": user_row["id"]})
        
        permission_list = perms["permission_name"].tolist()
        
        # 4. 创建User对象
        user = User(
            user_id=int(user_row["id"]),
            username=user_row["username"],
            permissions=permission_list
        )
        
        log_action(username, f"登录成功 (权限: {len(permission_list)}个)")
        return user
        
    except Exception as e:
        print(f"[auth] 登录异常: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# 3. 权限查询
# ════════════════════════════════════════════════════════════════════════════
def get_all_permissions() -> pd.DataFrame:
    """获取所有权限列表
    
    Returns:
        DataFrame: id, permission_name, description, category
    """
    eng = db.get_engine()
    return pd.read_sql(text(
        "SELECT id, permission_name, description, category "
        "FROM permissions ORDER BY category, permission_name"
    ), eng)


def get_user_permissions(user_id: int) -> list[str]:
    """获取用户的权限列表
    
    Args:
        user_id: 用户ID
    
    Returns:
        权限名称列表
    """
    eng = db.get_engine()
    result = pd.read_sql(text("""
        SELECT p.permission_name
        FROM user_permissions up
        JOIN permissions p ON up.permission_id = p.id
        WHERE up.user_id = :uid
    """), eng, params={"uid": user_id})
    
    return result["permission_name"].tolist()


def get_all_users() -> pd.DataFrame:
    """获取所有用户列表（不含密码）
    
    Returns:
        DataFrame: id, username, created_at
    """
    eng = db.get_engine()
    return pd.read_sql(text(
        "SELECT id, username, created_at FROM users ORDER BY id"
    ), eng)


# ════════════════════════════════════════════════════════════════════════════
# 4. 单项权限管理（管理员功能）
# ════════════════════════════════════════════════════════════════════════════
def grant_permission(admin_user: User, target_username: str, permission_name: str) -> tuple[bool, str]:
    """授予用户权限（管理员操作）
    
    Args:
        admin_user: 操作的管理员
        target_username: 目标用户名
        permission_name: 权限名称
    
    Returns:
        (成功?, 消息)
    """
    # 1. 检查管理员权限
    if not admin_user.has_permission("manage_permissions"):
        return False, "无权限：需要 manage_permissions"
    
    try:
        eng = db.get_engine()
        
        # 2. 查询目标用户ID
        user_result = pd.read_sql(text(
            "SELECT id FROM users WHERE username = :user"
        ), eng, params={"user": target_username})
        
        if user_result.empty:
            return False, f"用户不存在: {target_username}"
        
        target_user_id = int(user_result.iloc[0]["id"])
        
        # 3. 查询权限ID
        perm_result = pd.read_sql(text(
            "SELECT id FROM permissions WHERE permission_name = :perm"
        ), eng, params={"perm": permission_name})
        
        if perm_result.empty:
            return False, f"权限不存在: {permission_name}"
        
        perm_id = int(perm_result.iloc[0]["id"])
        
        # 4. 检查是否已拥有
        existing = pd.read_sql(text(
            "SELECT 1 FROM user_permissions WHERE user_id = :uid AND permission_id = :pid"
        ), eng, params={"uid": target_user_id, "pid": perm_id})
        
        if not existing.empty:
            return False, f"用户已拥有权限: {permission_name}"
        
        # 5. 授予权限
        with eng.connect() as conn:
            conn.execute(text(
                "INSERT INTO user_permissions (user_id, permission_id) VALUES (:uid, :pid)"
            ), {"uid": target_user_id, "pid": perm_id})
            conn.commit()
        
        log_action(admin_user.username, 
                  f"授予权限: {target_username} → {permission_name}")
        return True, f"成功授予 {target_username} 权限: {permission_name}"
        
    except Exception as e:
        return False, f"操作失败: {e}"


def revoke_permission(admin_user: User, target_username: str, permission_name: str) -> tuple[bool, str]:
    """移除用户权限（管理员操作）
    
    Args:
        admin_user: 操作的管理员
        target_username: 目标用户名
        permission_name: 权限名称
    
    Returns:
        (成功?, 消息)
    """
    # 1. 检查管理员权限
    if not admin_user.has_permission("manage_permissions"):
        return False, "无权限：需要 manage_permissions"
    
    try:
        eng = db.get_engine()
        
        # 2. 查询目标用户ID
        user_result = pd.read_sql(text(
            "SELECT id FROM users WHERE username = :user"
        ), eng, params={"user": target_username})
        
        if user_result.empty:
            return False, f"用户不存在: {target_username}"
        
        target_user_id = int(user_result.iloc[0]["id"])
        
        # 3. 查询权限ID
        perm_result = pd.read_sql(text(
            "SELECT id FROM permissions WHERE permission_name = :perm"
        ), eng, params={"perm": permission_name})
        
        if perm_result.empty:
            return False, f"权限不存在: {permission_name}"
        
        perm_id = int(perm_result.iloc[0]["id"])
        
        # 4. 移除权限
        with eng.connect() as conn:
            result = conn.execute(text(
                "DELETE FROM user_permissions WHERE user_id = :uid AND permission_id = :pid"
            ), {"uid": target_user_id, "pid": perm_id})
            conn.commit()
            
            if result.rowcount == 0:
                return False, f"用户未拥有权限: {permission_name}"
        
        log_action(admin_user.username, 
                  f"移除权限: {target_username} → {permission_name}")
        return True, f"成功移除 {target_username} 权限: {permission_name}"
        
    except Exception as e:
        return False, f"操作失败: {e}"


def create_user(admin_user: User, username: str, password: str, 
                initial_permissions: list[str] = None) -> tuple[bool, str]:
    """创建新用户（管理员操作）
    
    Args:
        admin_user: 操作的管理员
        username: 新用户名
        password: 密码（明文）
        initial_permissions: 初始权限列表
    
    Returns:
        (成功?, 消息)
    """
    # 1. 检查管理员权限
    if not admin_user.has_permission("manage_users"):
        return False, "无权限：需要 manage_users"
    
    try:
        eng = db.get_engine()
        
        # 2. 检查用户名是否已存在
        existing = pd.read_sql(text(
            "SELECT 1 FROM users WHERE username = :user"
        ), eng, params={"user": username})
        
        if not existing.empty:
            return False, f"用户名已存在: {username}"
        
        # 3. 创建用户
        hashed_pwd = hash_password(password)
        with eng.connect() as conn:
            conn.execute(text(
                "INSERT INTO users (username, password) VALUES (:user, :pwd)"
            ), {"user": username, "pwd": hashed_pwd})
            conn.commit()
        
        # 4. 授予初始权限
        if initial_permissions:
            for perm in initial_permissions:
                grant_permission(admin_user, username, perm)
        
        log_action(admin_user.username, 
                  f"创建用户: {username} (初始权限: {len(initial_permissions or [])}个)")
        return True, f"成功创建用户: {username}"
        
    except Exception as e:
        return False, f"创建失败: {e}"


def delete_user(admin_user: User, username: str) -> tuple[bool, str]:
    """删除用户（管理员操作）
    
    Args:
        admin_user: 操作的管理员
        username: 要删除的用户名
    
    Returns:
        (成功?, 消息)
    """
    # 1. 检查管理员权限
    if not admin_user.has_permission("manage_users"):
        return False, "无权限：需要 manage_users"
    
    # 2. 不能删除自己
    if username == admin_user.username:
        return False, "不能删除自己"
    
    try:
        eng = db.get_engine()
        
        # 3. 检查用户是否存在
        user_result = pd.read_sql(text(
            "SELECT id FROM users WHERE username = :user"
        ), eng, params={"user": username})
        
        if user_result.empty:
            return False, f"用户不存在: {username}"
        
        user_id = int(user_result.iloc[0]["id"])
        
        # 4. 检查是否是最后一个管理员
        # 查询该用户是否有 manage_users 权限
        has_admin_perm = pd.read_sql(text("""
            SELECT 1 FROM user_permissions up
            JOIN permissions p ON up.permission_id = p.id
            WHERE up.user_id = :uid AND p.permission_name = 'manage_users'
        """), eng, params={"uid": user_id})
        
        if not has_admin_perm.empty:
            # 是管理员，检查是否是最后一个
            admin_count = pd.read_sql(text("""
                SELECT COUNT(DISTINCT up.user_id) as cnt
                FROM user_permissions up
                JOIN permissions p ON up.permission_id = p.id
                WHERE p.permission_name = 'manage_users'
            """), eng)
            
            if admin_count.iloc[0]["cnt"] <= 1:
                return False, "不能删除最后一个管理员"
        
        # 5. 删除用户（CASCADE会自动删除关联的权限）
        with eng.connect() as conn:
            conn.execute(text(
                "DELETE FROM users WHERE id = :uid"
            ), {"uid": user_id})
            conn.commit()
        
        log_action(admin_user.username, f"删除用户: {username}")
        return True, f"成功删除用户: {username}"
        
    except Exception as e:
        return False, f"删除失败: {e}"


# ════════════════════════════════════════════════════════════════════════════
# 5. 操作日志
# ════════════════════════════════════════════════════════════════════════════
def log_action(username: str, action: str) -> None:
    """记录操作日志
    
    Args:
        username: 操作用户
        action: 操作描述
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {username}: {action}\n"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)


def get_recent_logs(n: int = 100) -> list[str]:
    """获取最近的操作日志
    
    Args:
        n: 返回最近n条
    
    Returns:
        日志行列表
    """
    if not LOG_FILE.exists():
        return []
    
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    return lines[-n:]


# ════════════════════════════════════════════════════════════════════════════
# 6. 权限装饰器（用于API）
# ════════════════════════════════════════════════════════════════════════════
def require_permission(permission: str):
    """权限检查装饰器
    
    用法:
        @require_permission("view_transactions")
        def get_transactions(current_user: User):
            ...
    """
    def decorator(func):
        def wrapper(*args, current_user: User = None, **kwargs):
            if current_user is None:
                raise PermissionError("未登录")
            
            if not current_user.has_permission(permission):
                raise PermissionError(f"无权限: {permission}")
            
            return func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("权限管理模块测试")
    print("=" * 60)
    
    # 1. 登录测试
    print("\n[测试1] 管理员登录")
    admin = login("admin", "admin123")
    if admin:
        print(f"✓ 登录成功: {admin.username}")
        print(f"  权限数量: {len(admin.permissions)}")
        print(f"  是否管理员: {admin.is_admin()}")
    else:
        print("✗ 登录失败")
    
    # 2. 权限列表
    print("\n[测试2] 所有权限")
    perms = get_all_permissions()
    print(perms.to_string(index=False))
    
    # 3. 创建用户
    print("\n[测试3] 创建普通用户")
    success, msg = create_user(admin, "analyst", "analyst123", 
                               ["view_transactions", "view_statistics"])
    print(f"{'✓' if success else '✗'} {msg}")
    
    # 4. 单项授权
    print("\n[测试4] 授予权限")
    success, msg = grant_permission(admin, "analyst", "view_predictions")
    print(f"{'✓' if success else '✗'} {msg}")
    
    # 5. 移除权限
    print("\n[测试5] 移除权限")
    success, msg = revoke_permission(admin, "analyst", "view_predictions")
    print(f"{'✓' if success else '✗'} {msg}")
    
    print("\n[完成] 测试结束")
