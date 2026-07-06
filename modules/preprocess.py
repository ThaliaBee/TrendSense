"""数据采集与预处理 pipeline。

读取唯一原始数据源 data/online_retail_II.csv，完成清洗 / 去噪 / 聚合，
按《03_接口设计文档》产出下游各模块的输入文件：

    data/clean_transactions.parquet   清洗后交易明细
    data/item_week_sales.parquet      商品 × 周 销量（供 F3/F4 LSTM）
    data/user_item.parquet            用户 × 商品 隐式评分（供 F2 协同过滤）
    data/item_meta.parquet            商品元数据（供 F5 推荐 / F6 库存）
    data/active_items.json            活跃商品清单（销售周数 ≥ 30）

运行：
    python modules/preprocess.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows 终端默认 GBK，强制 UTF-8 输出（出错也不崩），保证中文正常打印
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 路径（相对项目根目录，脚本在 modules/ 下）────────────────────────────────
PROJ_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_DIR / "data"
RAW_CSV = DATA_DIR / "online_retail_II.csv"

# 活跃商品阈值：有销售记录的周数 ≥ 此值（共 53 周）
ACTIVE_MIN_WEEKS = 30


# ════════════════════════════════════════════════════════════════════════════
# 1. 读取
# ════════════════════════════════════════════════════════════════════════════
def load_raw(raw_csv: Path = RAW_CSV) -> pd.DataFrame:
    """读取原始 CSV。数据含 £ 符号，必须用 latin-1 编码。"""
    df = pd.read_csv(raw_csv, encoding="latin-1")
    # 列名规整：'Customer ID' -> 'CustomerID'
    df = df.rename(columns={"Customer ID": "CustomerID"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. 清洗 / 去噪
# ════════════════════════════════════════════════════════════════════════════
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """清洗：去退货、缺失、异常、非商品编码、重复，并加 week_idx。"""
    n0 = len(df)
    steps = []

    # 2.1 去重
    df = df.drop_duplicates()
    steps.append(("去除重复记录", n0 - len(df)))

    # 2.2 缺失 CustomerID（匿名访客无法用于协同过滤）
    n = len(df)
    df = df[df["CustomerID"].notna()]
    steps.append(("去除缺失 CustomerID", n - len(df)))

    # 2.3 去退货 / 非正数量（Invoice 以 'C' 开头为退货，Quantity 为负）
    n = len(df)
    df = df[df["Quantity"] > 0]
    steps.append(("去除退货/非正数量", n - len(df)))

    # 2.4 异常价格
    n = len(df)
    df = df[df["Price"] > 0]
    steps.append(("去除价格<=0", n - len(df)))

    # 2.5 去掉无效日期
    n = len(df)
    df = df[df["InvoiceDate"].notna()]
    steps.append(("去除无效日期", n - len(df)))

    # 2.6 去掉非商品编码（POST/DOT/M/BANK CHARGES 等运费/手续费，首字符非数字）
    n = len(df)
    df = df[df["StockCode"].astype(str).str[0].str.isdigit()]
    steps.append(("去除非商品编码", n - len(df)))

    # 类型规整
    df = df.copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Invoice"] = df["Invoice"].astype(str)

    # 2.7 week_idx：以最早日期所在周（周一）为 0，按自然周递增
    min_date = df["InvoiceDate"].min()
    anchor = min_date.normalize() - pd.Timedelta(days=int(min_date.weekday()))
    df["week_idx"] = ((df["InvoiceDate"].dt.normalize() - anchor).dt.days // 7).astype(int)

    # 打印清洗报告（满足 F1 验收：清洗前后对比）
    print("─" * 56)
    print(f"清洗前记录数: {n0:>8}")
    for name, removed in steps:
        print(f"  - {name:<18}: 去除 {removed:>7} 条")
    print(f"清洗后记录数: {len(df):>8}")
    print(f"清洗后商品数: {df['StockCode'].nunique():>8}")
    print(f"清洗后用户数: {df['CustomerID'].nunique():>8}")
    print(f"周数(week_idx): 0 ~ {df['week_idx'].max()}（共 {df['week_idx'].nunique()} 周）")
    print("─" * 56)

    cols = ["Invoice", "StockCode", "Description", "Quantity",
            "InvoiceDate", "Price", "CustomerID", "week_idx"]
    return df[cols]


# ════════════════════════════════════════════════════════════════════════════
# 3. 聚合产出
# ════════════════════════════════════════════════════════════════════════════
def build_item_week_sales(clean_df: pd.DataFrame) -> pd.DataFrame:
    """商品 × 周 销量（长表，补 0 对齐时间序列，供 LSTM）。"""
    agg = (clean_df.groupby(["StockCode", "week_idx"])["Quantity"]
           .sum().rename("sales").reset_index())
    # 补全 (所有商品 × 所有周) 网格，缺失周补 0（"那周没卖"是有意义的信号）
    items = clean_df["StockCode"].unique()
    weeks = range(0, int(clean_df["week_idx"].max()) + 1)
    full_idx = pd.MultiIndex.from_product([items, weeks], names=["StockCode", "week_idx"])
    out = (agg.set_index(["StockCode", "week_idx"])
           .reindex(full_idx, fill_value=0).reset_index())
    out["sales"] = out["sales"].astype(int)
    return out


def build_user_item(clean_df: pd.DataFrame) -> pd.DataFrame:
    """用户 × 商品 隐式评分：n_purchase=购买次数(不同订单数), total_qty=累计数量。"""
    g = clean_df.groupby(["CustomerID", "StockCode"])
    out = g.agg(
        n_purchase=("Invoice", "nunique"),
        total_qty=("Quantity", "sum"),
    ).reset_index()
    out["total_qty"] = out["total_qty"].astype(int)
    return out


def build_item_meta(clean_df: pd.DataFrame) -> pd.DataFrame:
    """商品元数据：名称、平均单价、累计销量。"""
    out = clean_df.groupby("StockCode").agg(
        Description=("Description", "first"),
        avg_price=("Price", "mean"),
        total_sales=("Quantity", "sum"),
    ).reset_index()
    out["avg_price"] = out["avg_price"].round(2)
    out["total_sales"] = out["total_sales"].astype(int)
    return out.sort_values("total_sales", ascending=False).reset_index(drop=True)


def build_active_items(clean_df: pd.DataFrame, min_weeks: int = ACTIVE_MIN_WEEKS) -> list[str]:
    """活跃商品：有销售记录的周数 ≥ min_weeks。"""
    weeks_per_item = clean_df.groupby("StockCode")["week_idx"].nunique()
    return sorted(weeks_per_item[weeks_per_item >= min_weeks].index.tolist())


# ════════════════════════════════════════════════════════════════════════════
# 4. 归一化演示（F1.3）
#    说明：流行性预测(LSTM)的归一化在 F3 模块内按商品、仅用训练集统计量完成
#    （防数据泄露），故此处不对保存的销量做全局归一化，仅演示归一化结果。
# ════════════════════════════════════════════════════════════════════════════
def demo_normalization(item_meta: pd.DataFrame) -> None:
    s = item_meta["total_sales"].astype(float)
    minmax = (s - s.min()) / (s.max() - s.min() + 1e-9)
    zscore = (s - s.mean()) / (s.std() + 1e-9)
    print("归一化演示（以商品累计销量为例，前 3 个商品）：")
    print(f"  原始值   : {s.head(3).round(1).tolist()}")
    print(f"  Min-Max  : {minmax.head(3).round(3).tolist()}")
    print(f"  Z-Score  : {zscore.head(3).round(3).tolist()}")
    print("─" * 56)


# ════════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════════
def run_preprocess(raw_csv: Path = RAW_CSV) -> None:
    print(f"读取原始数据: {raw_csv}")
    raw = load_raw(raw_csv)
    clean_df = clean(raw)

    item_week = build_item_week_sales(clean_df)
    user_item = build_user_item(clean_df)
    item_meta = build_item_meta(clean_df)
    active = build_active_items(clean_df)

    # 写出（数据表用 parquet，活跃清单用 JSON）
    clean_df.to_parquet(DATA_DIR / "clean_transactions.parquet", index=False)
    item_week.to_parquet(DATA_DIR / "item_week_sales.parquet", index=False)
    user_item.to_parquet(DATA_DIR / "user_item.parquet", index=False)
    item_meta.to_parquet(DATA_DIR / "item_meta.parquet", index=False)
    with open(DATA_DIR / "active_items.json", "w", encoding="utf-8") as f:
        json.dump(active, f, ensure_ascii=False)

    demo_normalization(item_meta)
    print("产出文件：")
    print(f"  clean_transactions.parquet : {len(clean_df):>8} 行")
    print(f"  item_week_sales.parquet    : {len(item_week):>8} 行")
    print(f"  user_item.parquet          : {len(user_item):>8} 行")
    print(f"  item_meta.parquet          : {len(item_meta):>8} 行")
    print(f"  active_items.json          : {len(active):>8} 个活跃商品（≥{ACTIVE_MIN_WEEKS}周）")
    print("[完成] F1 数据预处理完成。")


if __name__ == "__main__":
    run_preprocess()
