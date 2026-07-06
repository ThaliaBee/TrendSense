"""数据库连接模块。

统一管理 MySQL 连接，供各模块读写「消费记录 transactions」「库存 inventory」等表。

设计：
  - 用 SQLAlchemy engine（pandas 的 read_sql / to_sql 直接吃 engine）；
  - 连接参数集中在此处，通过 .env 环境变量覆盖；
  - get_engine() 单例复用，避免重复建连接池。

依赖：pip install sqlalchemy pymysql
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ── 加载 .env ──────────────────────────────────────────────
# 在所有连接参数读取之前，手动解析项目根目录的 .env 并注入 os.environ
# （避免引入 python-dotenv 额外依赖，与 ai_chat/api.py 的 _load_env() 做法一致）
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

# ── 连接配置 ──────────────────────────────────────────────
# 密码优先从环境变量 DB_PASSWORD 读取（各人可在 .env 中设置自己的密码），
# 未设置时回退到默认值。
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123456")
DB_NAME = "recommender"
CHARSET = "utf8mb4"

_ENGINE: Engine | None = None


def _url(database: str | None = DB_NAME) -> str:
    """构造 SQLAlchemy 连接串；database 传 None 时连到服务器但不选库（用于建库）。"""
    db = f"/{database}" if database else "/"
    return (f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}{db}"
            f"?charset={CHARSET}&ssl_disabled=true"
            f"&connect_timeout=3")


def get_engine() -> Engine:
    """返回连接到 recommender 库的 engine（单例）。"""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(_url(DB_NAME), pool_pre_ping=True, future=True)
    return _ENGINE


def get_server_engine() -> Engine:
    """返回未选库的 engine（仅用于 CREATE DATABASE）。"""
    return create_engine(_url(None), future=True)


def read_table(table: str, engine: Engine | None = None):
    """读取整张表为 DataFrame（薄封装，方便各模块调用）。"""
    import pandas as pd
    return pd.read_sql(f"SELECT * FROM `{table}`", engine or get_engine())


def query(sql: str, engine: Engine | None = None):
    """执行任意 SELECT，返回 DataFrame。"""
    import pandas as pd
    return pd.read_sql(text(sql), engine or get_engine())


_DATA_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"


def load_transactions():
    """读「消费记录」：优先数据库 transactions 表，连不上则回退 parquet 文件。"""
    import pandas as pd
    try:
        df = read_table("transactions")
        return df.drop(columns=["id"], errors="ignore")   # 去掉自增主键列
    except Exception as e:
        print(f"[db] 读 transactions 失败，回退文件: {e}")
        return pd.read_parquet(_DATA_DIR / "clean_transactions.parquet")


def transaction_stats() -> dict:
    """消费记录的聚合统计（行数 / 去重用户数 / 总销量）。

    用 SQL 聚合（COUNT/SUM）只返回一行，避免把 39 万行全表拉回 Python——
    这是前端仪表盘真正需要的，比 load_transactions() 全量读快几个数量级。
    """
    import pandas as pd
    try:
        row = pd.read_sql(text(
            "SELECT COUNT(*) AS n_rows, COUNT(DISTINCT CustomerID) AS n_users, "
            "COALESCE(SUM(Quantity), 0) AS total_qty FROM transactions"), get_engine()).iloc[0]
        return {"n_rows": int(row.n_rows), "n_users": int(row.n_users), "total_qty": int(row.total_qty)}
    except Exception as e:
        print(f"[db] 统计查询失败，回退文件: {e}")
        df = pd.read_parquet(_DATA_DIR / "clean_transactions.parquet")
        return {"n_rows": len(df), "n_users": int(df["CustomerID"].nunique()),
                "total_qty": int(df["Quantity"].sum())}


def weekly_revenue():
    """周销售额聚合：SUM(Quantity × Price) GROUP BY week_idx + 每周起始日期。

    优先用 SQL 聚合（一行 SUM 算完 39 万行，≈0.1s），
    连不上则回退文件用 pandas groupby。
    """
    import pandas as pd
    try:
        df = pd.read_sql(text(
            "SELECT week_idx, MIN(InvoiceDate) AS week_start, "
            "SUM(Quantity * Price) AS revenue "
            "FROM transactions GROUP BY week_idx ORDER BY week_idx"
        ), get_engine())
        df["week_start"] = pd.to_datetime(df["week_start"])
        return df
    except Exception as e:
        print(f"[db] 周销售额查询失败，回退文件: {e}")
        tx = pd.read_parquet(_DATA_DIR / "clean_transactions.parquet")
        tx["revenue"] = tx["Quantity"] * tx["Price"]
        rev = tx.groupby("week_idx").agg(
            revenue=("revenue", "sum"),
            week_start=("InvoiceDate", "min"),
        ).reset_index()
        rev["week_start"] = pd.to_datetime(rev["week_start"])
        return rev


def weekly_active_counts():
    """最近两周活跃商品/用户数：返回 (本周商品, 本周用户, 上周商品, 上周用户)。

    用于 KPI 卡片计算环比变化。优先用 SQL，连不上则回退文件。
    """
    import pandas as pd
    try:
        df = pd.read_sql(text("""
            SELECT week_idx,
                   COUNT(DISTINCT StockCode) AS n_items,
                   COUNT(DISTINCT CustomerID) AS n_users
            FROM transactions
            WHERE week_idx >= (SELECT MAX(week_idx) FROM transactions) - 1
            GROUP BY week_idx ORDER BY week_idx
        """), get_engine())
        if len(df) >= 2:
            return (int(df["n_items"].iloc[1]), int(df["n_users"].iloc[1]),
                    int(df["n_items"].iloc[0]), int(df["n_users"].iloc[0]))
        if len(df) == 1:
            return (int(df["n_items"].iloc[0]), int(df["n_users"].iloc[0]), 0, 0)
        return 0, 0, 0, 0
    except Exception as e:
        print(f"[db] 周活跃统计查询失败，回退文件: {e}")
        tx = pd.read_parquet(_DATA_DIR / "clean_transactions.parquet")
        weeks = sorted(tx["week_idx"].unique())[-2:]
        if len(weeks) >= 2:
            curr = tx[tx["week_idx"] == weeks[1]]
            prev = tx[tx["week_idx"] == weeks[0]]
            return (int(curr["StockCode"].nunique()), int(curr["CustomerID"].nunique()),
                    int(prev["StockCode"].nunique()), int(prev["CustomerID"].nunique()))
        if len(weeks) == 1:
            w = tx[tx["week_idx"] == weeks[0]]
            return (int(w["StockCode"].nunique()), int(w["CustomerID"].nunique()), 0, 0)
        return 0, 0, 0, 0


def load_inventory():
    """读「库存」：优先数据库 inventory 表，连不上则回退 csv 文件。"""
    import pandas as pd
    try:
        return read_table("inventory")
    except Exception as e:
        print(f"[db] 读 inventory 失败，回退文件: {e}")
        return pd.read_csv(_DATA_DIR / "inventory.csv")


def save_inventory(df) -> None:
    """写「库存」到数据库 inventory 表（清空重写，保留表结构）。"""
    eng = get_engine()
    try:
        with eng.begin() as conn:
            conn.execute(text("TRUNCATE TABLE inventory"))
        df.to_sql("inventory", eng, if_exists="append", index=False)
    except Exception:
        df.to_sql("inventory", eng, if_exists="replace", index=False)


def ping() -> bool:
    """检测 recommender 库是否可连（库需已建）。"""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[db] 连接失败: {e}")
        return False


def ping_server() -> bool:
    """检测 MySQL 服务是否可连（不要求 recommender 库已存在，用于建库前）。"""
    try:
        eng = get_server_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception as e:
        print(f"[db] 服务器连接失败: {e}")
        return False


if __name__ == "__main__":
    print("数据库连通性自检 ...")
    print("OK" if ping() else "FAILED（确认 MySQL 已启动、recommender 库已建）")
