"""共享数据加载器 —— 所有页面都可以 import 使用。
所有加载函数都带 @st.cache_resource，数据常驻内存，跨页面零开销共享。
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from modules import db

PROJ_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_DIR / "data"
CACHE_DIR = PROJ_DIR / "cache"


@st.cache_resource
def load_backtest():
    """LSTM 回测预测 vs 实际（测试集周 47-53）。"""
    p = CACHE_DIR / "backtest.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_resource
def load_item_sim():
    """Item-CF 余弦相似度 Top-50 对。"""
    p = CACHE_DIR / "item_sim_top50.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_resource
def load_item_meta():
    """商品元数据，含中文名 Description_CN 和品类 category（优先从 MySQL 读取）。"""
    im = pd.read_parquet(DATA_DIR / "item_meta.parquet")
    # 尝试从 MySQL 加载中文翻译 + 品类
    try:
        cn = db.read_table("item_descriptions")
        if len(cn) > 0:
            merge_cols = ["StockCode"]
            if "Description_CN" in cn.columns:
                merge_cols.append("Description_CN")
            if "category" in cn.columns:
                merge_cols.append("category")
            im = im.merge(cn[merge_cols], on="StockCode", how="left")
    except Exception:
        pass
    # 无中文时 fallback 原英文
    if "Description_CN" not in im.columns:
        im["Description_CN"] = im["Description"]
    im["Description_CN"] = im["Description_CN"].fillna(im["Description"])
    # 无品类时 fallback
    if "category" not in im.columns:
        im["category"] = "未分类"
    im["category"] = im["category"].fillna("未分类")
    return im


@st.cache_resource
def load_item_week():
    return pd.read_parquet(DATA_DIR / "item_week_sales.parquet")


@st.cache_resource
def load_forecasts():
    p = CACHE_DIR / "forecasts.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_resource
def load_cf_metrics():
    p = CACHE_DIR / "cf_metrics.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


@st.cache_resource
def load_lstm_metrics():
    p = CACHE_DIR / "lstm_metrics.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


@st.cache_resource
def load_cf_topn():
    p = CACHE_DIR / "cf_topn.parquet"
    return pd.read_parquet(p) if p.exists() else None


def load_inventory_status():
    """读取库存状态文件（无缓存，每次重读）"""
    p = CACHE_DIR / "inventory_status.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        print(f"[data_loaders] load_inventory_status: {len(df)} 行, "
              f"status={dict(df['status'].value_counts())}")
        return df
    print("[data_loaders] load_inventory_status: 文件不存在，返回 None")
    return None


@st.cache_resource
def load_transactions():
    return db.load_transactions()


@st.cache_resource
def load_tx_stats():
    return db.transaction_stats()


@st.cache_resource
def load_weekly_revenue():
    return db.weekly_revenue()


@st.cache_resource
def load_weekly_active_counts():
    return db.weekly_active_counts()


@st.cache_resource
def load_weekly_avp():
    """返回每周客单价（revenue / orders）。"""
    rev = load_weekly_revenue()
    orders = load_weekly_orders()
    avp = rev.merge(orders, on="week_idx", how="left")
    avp["avp"] = (avp["revenue"] / avp["n_orders"]).round(2)
    return avp[["week_idx", "avp"]]


@st.cache_resource
def load_weekly_orders():
    """返回每周订单数 DataFrame（week_idx, n_orders）。"""
    tx = pd.read_parquet(DATA_DIR / "clean_transactions.parquet")
    orders = tx.groupby("week_idx")["Invoice"].nunique().reset_index()
    orders.columns = ["week_idx", "n_orders"]
    return orders.sort_values("week_idx")


@st.cache_resource
def load_weekly_new_users():
    """返回每周新用户数 + 回头客数 DataFrame。"""
    tx = pd.read_parquet(DATA_DIR / "clean_transactions.parquet")
    first_wk = tx.groupby("CustomerID")["week_idx"].min().reset_index()
    first_wk.columns = ["CustomerID", "first_week"]
    tx2 = tx.merge(first_wk, on="CustomerID")
    tx2["is_new"] = tx2["week_idx"] == tx2["first_week"]
    wk = tx.groupby("week_idx")["CustomerID"].nunique().reset_index()
    wk.columns = ["week_idx", "total_users"]
    new_u = tx2[tx2["is_new"]].groupby("week_idx")["CustomerID"].nunique().reset_index()
    new_u.columns = ["week_idx", "new_users"]
    result = wk.merge(new_u, on="week_idx", how="left").fillna(0)
    result["new_users"] = result["new_users"].astype(int)
    result["returning_users"] = result["total_users"] - result["new_users"]
    return result.sort_values("week_idx")


@st.cache_resource
def load_country_revenue():
    """返回各国累计营收 DataFrame（Country, revenue）。"""
    tx = pd.read_parquet(DATA_DIR / "clean_transactions.parquet")
    tx["revenue"] = tx["Quantity"] * tx["Price"]
    # Country 在清洗时被删，回原始 CSV 取
    raw = pd.read_csv(DATA_DIR / "online_retail_II.csv", encoding="latin1")
    inv_country = raw[["Invoice", "Country"]].drop_duplicates()
    tx = tx.merge(inv_country, on="Invoice", how="left")
    tx["Country"] = tx["Country"].fillna("Unknown")
    country = tx.groupby("Country")["revenue"].sum().sort_values(ascending=False).reset_index()
    return country


@st.cache_resource
def load_category_weekly_revenue():
    """返回每周每品类的营收（category, week_idx, revenue）。"""
    im = load_item_meta()
    iw = pd.read_parquet(DATA_DIR / "item_week_sales.parquet")
    cat = iw.merge(im[["StockCode", "category", "avg_price"]], on="StockCode", how="left")
    cat["category"] = cat["category"].fillna("其他")
    cat["revenue"] = cat["sales"] * cat["avg_price"]
    return cat.groupby(["week_idx", "category"], as_index=False)["revenue"].sum()


@st.cache_resource
def load_category_weekly_orders():
    """返回每周每品类的销量（category, week_idx, sales）。"""
    im = load_item_meta()
    tx = pd.read_parquet(DATA_DIR / "clean_transactions.parquet")
    cat_tx = tx.merge(im[["StockCode", "category"]], on="StockCode", how="left")
    cat_tx["category"] = cat_tx["category"].fillna("其他")
    sales = cat_tx.groupby(["week_idx", "category"])["Quantity"].sum().reset_index()
    sales.columns = ["week_idx", "category", "sales"]
    return sales


@st.cache_resource
def load_active_items():
    return json.load(open(DATA_DIR / "active_items.json", encoding="utf-8"))


@st.cache_resource
def load_user_item():
    return pd.read_parquet(DATA_DIR / "user_item.parquet")


@st.cache_resource
def load_weekly_active_series():
    """返回完整周的 (活跃商品数序列, 活跃用户数序列)，供 sparkline 使用。
    活跃 = 该周有过交易记录。
    """
    tx = pd.read_parquet(DATA_DIR / "clean_transactions.parquet")
    items = tx.groupby("week_idx")["StockCode"].nunique().sort_index()
    users = tx.groupby("week_idx")["CustomerID"].nunique().sort_index()
    return items, users


# ════════════════════════════════════════════════════════════════════════════
# 库存数据获取（无缓存，每次实时读取）
# ════════════════════════════════════════════════════════════════════════════

def get_inventory():
    """统一库存数据获取：优先缓存文件，失败则基于销量模拟"""
    df = load_inventory_status()
    if df is not None and len(df) > 0:
        df["status"] = df["status"].replace({"告警": "警告"})
        a = (df["status"] == "警告").sum()
        l = (df["status"] == "偏低").sum()
        o = (df["status"] == "充足").sum()
        # 写日志文件方便排查
        (PROJ_DIR / "cache" / "_debug_inventory.txt").write_text(
            f"FROM FILE: total={len(df)} ok={o} low={l} alert={a}\n"
            f"sample_statuses={df['status'].value_counts().to_dict()}\n"
            f"sample_rows=\n{df[df['status']=='警告'].head(3).to_string()}\n",
            encoding="utf-8")
        return df
    # fallback: 旧版随机模拟
    import numpy as np
    im = load_item_meta()
    rng = np.random.default_rng(42)
    wa = im["total_sales"] / 53.0
    s = (wa * rng.uniform(4, 10, len(im))).round(0).astype(int)
    th = (wa * 3).round(0).astype(int)
    dd = (th / 1.5).round(0).astype(int)
    out = pd.DataFrame({"StockCode": im["StockCode"], "stock": s, "predicted_demand": dd, "threshold": th})
    out["status"] = "充足"
    out.loc[out["stock"] < out["threshold"], "status"] = "偏低"
    out.loc[out["stock"] < out["predicted_demand"], "status"] = "警告"
    a2 = (out["status"] == "警告").sum()
    l2 = (out["status"] == "偏低").sum()
    o2 = (out["status"] == "充足").sum()
    (PROJ_DIR / "cache" / "_debug_inventory.txt").write_text(
        f"FALLBACK: total={len(out)} ok={o2} low={l2} alert={a2}\n",
        encoding="utf-8")
    return out


def get_inventory_with_meta():
    """库存 + 商品描述，供 overview / recommend 等页面使用"""
    im = load_item_meta()
    inv = get_inventory()
    return inv.merge(im[["StockCode", "Description_CN"]], on="StockCode", how="left")


def get_inventory_full():
    """库存 + 商品描述 + 均价，供 inventory 页面使用"""
    im = load_item_meta()
    inv = get_inventory()
    return inv.merge(im[["StockCode", "Description_CN", "avg_price"]], on="StockCode", how="left")
