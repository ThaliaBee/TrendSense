"""一次性初始化数据库。

做三件事：
  1. 建库 recommender（utf8mb4）；
  2. 建表 transactions（消费记录）、inventory（库存）；
  3. 把 data/clean_transactions.parquet、data/inventory.csv 灌进对应表。

角色权限账号由 permissions.py 管理，本脚本不涉及。

运行（在项目根目录，用项目所用 Python 环境）：
    python scripts/init_db.py

前置：已装好 MySQL 并启动，已 pip install sqlalchemy pymysql。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

# 让脚本能 import 到 modules/db.py
PROJ_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_DIR))
DATA_DIR = PROJ_DIR / "data"

from modules import db  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ── 1. 建库 ──────────────────────────────────────────────────────────────────
def create_database() -> None:
    eng = db.get_server_engine()
    with eng.connect() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{db.DB_NAME}` "
            f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        conn.commit()
    eng.dispose()
    print(f"[1/3] 数据库 `{db.DB_NAME}` 就绪")


# ── 2. 建表 ──────────────────────────────────────────────────────────────────
DDL_TRANSACTIONS = """
CREATE TABLE IF NOT EXISTS transactions (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    Invoice     VARCHAR(20),
    StockCode   VARCHAR(20),
    Description VARCHAR(255),
    Quantity    INT,
    InvoiceDate DATETIME,
    Price       DECIMAL(10,2),
    CustomerID  INT,
    week_idx    INT,
    INDEX idx_stockcode (StockCode),
    INDEX idx_customer  (CustomerID),
    INDEX idx_week      (week_idx)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_INVENTORY = """
CREATE TABLE IF NOT EXISTS inventory (
    StockCode VARCHAR(20) PRIMARY KEY,
    stock     INT NOT NULL,
    threshold INT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def create_tables() -> None:
    eng = db.get_engine()
    with eng.connect() as conn:
        conn.execute(text(DDL_TRANSACTIONS))
        conn.execute(text(DDL_INVENTORY))
        conn.commit()
    print("[2/3] 数据表 transactions、inventory 就绪")


# ── 3. 灌数据 ────────────────────────────────────────────────────────────────
def load_data() -> None:
    eng = db.get_engine()

    # 消费记录：先清空再灌（可重复运行）
    tx = pd.read_parquet(DATA_DIR / "clean_transactions.parquet")
    # Description 截断到 255，防止超长入库报错
    tx["Description"] = tx["Description"].astype(str).str.slice(0, 255)
    with eng.connect() as conn:
        conn.execute(text("TRUNCATE TABLE transactions"))
        conn.commit()
    tx.to_sql("transactions", eng, if_exists="append", index=False,
              chunksize=1000, method="multi")
    print(f"      消费记录 transactions: 灌入 {len(tx):,} 行")

    # 库存
    inv = pd.read_csv(DATA_DIR / "inventory.csv")
    inv["StockCode"] = inv["StockCode"].astype(str)
    with eng.connect() as conn:
        conn.execute(text("TRUNCATE TABLE inventory"))
        conn.commit()
    inv.to_sql("inventory", eng, if_exists="append", index=False,
               chunksize=1000, method="multi")
    print(f"      库存 inventory: 灌入 {len(inv):,} 行")
    print("[3/3] 数据灌入完成")


def verify() -> None:
    eng = db.get_engine()
    n_tx = pd.read_sql(text("SELECT COUNT(*) AS n FROM transactions"), eng)["n"][0]
    n_inv = pd.read_sql(text("SELECT COUNT(*) AS n FROM inventory"), eng)["n"][0]
    print(f"\n校验: transactions={n_tx:,} 行, inventory={n_inv:,} 行")
    print("示例 transactions:")
    print(pd.read_sql(text("SELECT Invoice,StockCode,Quantity,Price,CustomerID FROM transactions LIMIT 3"), eng).to_string(index=False))


def main() -> None:
    print("=" * 60)
    print("初始化 MySQL 数据库（消费记录 + 库存）")
    print("=" * 60)
    if not db.ping_server():
        print("无法连接 MySQL，请确认服务已启动、root/123456 正确。")
        return
    create_database()
    create_tables()
    load_data()
    verify()
    print("\n[完成] 数据库初始化结束。")


if __name__ == "__main__":
    main()
