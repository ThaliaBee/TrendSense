"""导出 Item-CF 相似度 Top-50 对 → cache/item_sim_top50.parquet"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

TOP_N = 50


def main():
    print("加载 user_item...")
    ui = pd.read_parquet(DATA_DIR / "user_item.parquet")
    print(f"  {len(ui)} 条, {ui['CustomerID'].nunique()} 用户, {ui['StockCode'].nunique()} 商品")

    # 构建用户-商品矩阵
    matrix_df = ui.pivot(index="CustomerID", columns="StockCode", values="n_purchase").fillna(0)
    matrix = matrix_df.values.astype(np.float32)
    items = matrix_df.columns.tolist()
    print(f"  矩阵: {matrix.shape}, 稀疏度: {(matrix == 0).sum() / matrix.size:.2%}")

    # 计算物品-物品余弦相似度
    print("计算 item-item 余弦相似度...")
    sim = cosine_similarity(matrix.T)
    np.fill_diagonal(sim, -1)  # 排除自身
    print(f"  相似度矩阵: {sim.shape}")

    # 每个物品取 Top-N
    print(f"取 Top-{TOP_N}...")
    rows = []
    for i, code in enumerate(items):
        top_idx = np.argpartition(sim[i], -TOP_N)[-TOP_N:]   # 快排分区，O(n) 取 Top-N
        top_idx = top_idx[np.argsort(sim[i][top_idx])[::-1]]  # Top-N 内降序
        for rank, j in enumerate(top_idx):
            rows.append({
                "StockCode": code,
                "sim_StockCode": items[j],
                "similarity": round(float(sim[i][j]), 6),
                "rank": rank + 1,
            })

    df = pd.DataFrame(rows)
    out = CACHE_DIR / "item_sim_top50.parquet"
    df.to_parquet(out)
    print(f"导出 {len(df)} 条 → {out}")
    print("完成!")


if __name__ == "__main__":
    main()
