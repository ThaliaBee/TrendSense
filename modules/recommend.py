"""个性化推荐引擎（三因子融合）。

融合协同过滤（F2）与流行性预测（F3/F4），生成个性化推荐卡片。

输入：
    cache/cf_topn.parquet           F2协同过滤Top-N推荐
    cache/forecasts.parquet         F3/F4流行性预测
    data/item_meta.parquet          商品元数据
    cache/inventory_status.parquet  库存状态（如无则模拟）

输出：
    推荐卡片列表（供F7可视化使用）

运行：
    python modules/recommend.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Windows 终端 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 路径 ────────────────────────────────────────────────────────────────────
PROJ_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_DIR / "data"
CACHE_DIR = PROJ_DIR / "cache"

# ── 融合参数 ────────────────────────────────────────────────────────────────
ALPHA = 0.4    # CF 分数权重
BETA = 0.3     # 流行度权重
GAMMA = 0.3    # 个性化多样性权重（惩罚大众爆款 + 用户扰动）


# ════════════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ════════════════════════════════════════════════════════════════════════════
def load_data():
    """加载所有需要的数据文件。"""
    print("加载数据文件...")
    
    # F2 协同过滤推荐
    cf_topn = pd.read_parquet(CACHE_DIR / "cf_topn.parquet")
    print(f"  ✓ CF推荐: {len(cf_topn)} 条")
    
    # F3/F4 流行性预测（取未来第1周的预测）
    forecasts = pd.read_parquet(CACHE_DIR / "forecasts.parquet")
    # 只保留未来第1周的预测
    forecasts = forecasts[forecasts["week_idx"] == forecasts["week_idx"].min()]
    print(f"  ✓ 流行性预测: {len(forecasts)} 个商品")
    
    # 商品元数据
    item_meta = pd.read_parquet(DATA_DIR / "item_meta.parquet")
    print(f"  ✓ 商品元数据: {len(item_meta)} 个商品")
    
    # 库存状态（如果F6还没完成，则模拟）
    inventory_path = CACHE_DIR / "inventory_status.parquet"
    if inventory_path.exists():
        inventory = pd.read_parquet(inventory_path)
        print(f"  ✓ 库存状态: {len(inventory)} 个商品")
    else:
        print("  ⚠ 库存状态文件不存在，使用模拟数据")
        inventory = _simulate_inventory(item_meta)
    
    return cf_topn, forecasts, item_meta, inventory


def _simulate_inventory(item_meta: pd.DataFrame) -> pd.DataFrame:
    """模拟库存状态（临时方案，等F6完成后使用真实数据）。"""
    np.random.seed(42)
    
    inventory = item_meta[["StockCode"]].copy()
    
    # 根据商品销量模拟库存
    # 高销量商品：库存较多
    # 低销量商品：库存较少
    sales_normalized = (item_meta["total_sales"] - item_meta["total_sales"].min()) / \
                      (item_meta["total_sales"].max() - item_meta["total_sales"].min() + 1e-9)
    
    # 模拟当前库存（销量越高，库存越多，但加入随机性）
    inventory["stock"] = (sales_normalized * 500 + np.random.randint(50, 200, len(item_meta))).astype(int)
    
    # 模拟预测需求（基于历史销量）
    inventory["predicted_demand"] = (item_meta["total_sales"] / 53 * 4).round(0).astype(int)  # 未来4周需求
    
    # 安全阈值（需求的1.5倍）
    inventory["threshold"] = (inventory["predicted_demand"] * 1.5).astype(int)
    
    # 库存状态
    inventory["status"] = "充足"
    inventory.loc[inventory["stock"] < inventory["threshold"], "status"] = "偏低"
    inventory.loc[inventory["stock"] < inventory["predicted_demand"], "status"] = "警告"
    
    return inventory


# ════════════════════════════════════════════════════════════════════════════
# 2. 流行度计算
# ════════════════════════════════════════════════════════════════════════════
def calculate_popularity(forecasts: pd.DataFrame) -> pd.DataFrame:
    """从流行性预测中提取流行度分数（保留原始预测值，归一化在 per-user 阶段做）。"""
    popularity = forecasts[["StockCode", "pred"]].copy()
    popularity = popularity.rename(columns={"pred": "popularity_raw"})
    return popularity


# ════════════════════════════════════════════════════════════════════════════
# 3. 推荐融合
# ════════════════════════════════════════════════════════════════════════════
def recommend(user_id: int, n: int = 10,
              cf_topn: pd.DataFrame = None,
              forecasts: pd.DataFrame = None,
              item_meta: pd.DataFrame = None,
              inventory: pd.DataFrame = None,
              user_item: pd.DataFrame = None,
              strategy: Literal["balanced", "cf_focused", "trending"] = "balanced") -> list[dict]:
    """为用户生成个性化推荐卡片。

    Args:
        user_id: 用户ID
        n: 推荐数量
        cf_topn: CF推荐数据（需含 CustomerID, StockCode, cf_score）
        forecasts: 流行性预测数据
        item_meta: 商品元数据
        inventory: 库存状态
        user_item: 用户-商品购买记录（用于过滤已购商品）
        strategy: 推荐策略
            - "balanced": 平衡CF、流行度与多样性（α=0.5, β=0.3, γ=0.2）
            - "cf_focused": 侧重协同过滤（α=0.7, β=0.2, γ=0.1）
            - "trending": 侧重流行度（α=0.2, β=0.6, γ=0.2）

    Returns:
        推荐卡片列表
    """
    # 如果没有提供数据，则加载
    if cf_topn is None or forecasts is None or item_meta is None or inventory is None:
        cf_topn, forecasts, item_meta, inventory = load_data()

    # 确保 forecasts 只含第一预测周（外部可能传入多周数据）
    if "week_idx" in forecasts.columns:
        min_week = forecasts["week_idx"].min()
        forecasts = forecasts[forecasts["week_idx"] == min_week]

    # 获取该用户的CF推荐候选
    user_cf = cf_topn[cf_topn["CustomerID"] == user_id].copy()

    if len(user_cf) == 0:
        return _popularity_fallback(forecasts, item_meta, inventory, n, user_id)

    # ── 1. 过滤已购买商品 ──
    purchased = set()
    if user_item is not None and len(user_item) > 0:
        purchased = set(user_item[user_item["CustomerID"] == user_id]["StockCode"].tolist())
        user_cf = user_cf[~user_cf["StockCode"].isin(purchased)]

    if len(user_cf) == 0 and len(purchased) == 0:
        return _popularity_fallback(forecasts, item_meta, inventory, n, user_id)

    # ── 1.5 策略决定候选偏好：CF 侧重保留更多CF，热门侧重保留更多趋势 ──
    rng = np.random.default_rng(abs(hash(user_id)) % (2**31))
    if strategy == "cf_focused":
        elig_pct = 70   # 保留大部分 CF 商品
        backfill_n = 10  # 少量热门补全
    elif strategy == "trending":
        elig_pct = 20   # 大量丢弃 CF，靠热门补全
        backfill_n = 40
    else:  # balanced
        elig_pct = 40
        backfill_n = 20

    def _eligible(item_code):
        return (hash(f"{item_code}_{user_id}") % 100) < elig_pct
    mask = user_cf["StockCode"].apply(_eligible)
    if mask.sum() >= n:
        user_cf = user_cf[mask]
    else:
        user_cf = user_cf.iloc[rng.permutation(len(user_cf))[:n]]
    user_cf = user_cf.iloc[rng.permutation(len(user_cf))]
    user_cf = user_cf.head(n * 2)

    # ── 2. 候选不足时用热门商品补全 ──
    if len(user_cf) < n * 2:
        existing = set(user_cf["StockCode"].tolist())
        pop_pool = forecasts.nlargest(min(n * backfill_n, len(forecasts)), "pred")[["StockCode"]].copy()
        pop_pool = pop_pool[~pop_pool["StockCode"].isin(purchased | existing)]
        if len(pop_pool) > 0:
            n_sample = min(n * 2 - len(user_cf), len(pop_pool))
            pop_sample = pop_pool.sample(n=n_sample, random_state=int(rng.integers(0, 2**31 - 1)))
            pop_sample["CustomerID"] = user_id
            pop_sample["cf_score"] = 0.0
            pop_sample["rank"] = 99
            user_cf = pd.concat([user_cf, pop_sample], ignore_index=True)

    # ── 3. 合并流行度 ──
    popularity = calculate_popularity(forecasts)
    user_cf = user_cf.merge(popularity, on="StockCode", how="left")
    user_cf["popularity_raw"] = user_cf["popularity_raw"].fillna(0)

    # ── 4. Per-user 分位数排名（避免全局归一化导致的「千人一面」）──
    # CF 分数在用户候选集内的分位数
    user_cf["cf_rank"] = user_cf["cf_score"].rank(pct=True)
    # 流行度在用户候选集内的分位数
    user_cf["pop_rank"] = user_cf["popularity_raw"].rank(pct=True)

    # ── 5. 多样性惩罚：在全局CF中出现越频繁的商品得分越低 ──
    item_freq = cf_topn["StockCode"].value_counts(normalize=True)  # 全球出现频率
    user_cf["diversity"] = 1.0 - user_cf["StockCode"].map(item_freq).fillna(0)
    # 归一化 diversity 到 0-1
    d_min, d_max = user_cf["diversity"].min(), user_cf["diversity"].max()
    if d_max > d_min:
        user_cf["diversity"] = (user_cf["diversity"] - d_min) / (d_max - d_min)
    else:
        user_cf["diversity"] = 0.5

    # ── 6. 根据策略设置权重 ──
    if strategy == "balanced":
        alpha, beta, gamma = 0.4, 0.3, 0.3
    elif strategy == "cf_focused":
        alpha, beta, gamma = 0.6, 0.2, 0.2
    elif strategy == "trending":
        alpha, beta, gamma = 0.2, 0.5, 0.3
    else:
        alpha, beta, gamma = ALPHA, BETA, GAMMA

    # ── 7. 融合得分 ──
    user_cf["cf_component"] = alpha * user_cf["cf_rank"]
    user_cf["pop_component"] = beta * user_cf["pop_rank"]
    user_cf["div_component"] = gamma * user_cf["diversity"]
    user_cf["final_score"] = (user_cf["cf_component"] +
                              user_cf["pop_component"] +
                              user_cf["div_component"])

    # ── 7.5 每用户确定性扰动（确保不同用户的排序各不相同）──
    user_cf["perturb"] = user_cf["StockCode"].apply(
        lambda x: (hash(f"{user_id}_{x}") % 1000) / 10000  # 0 ~ 0.10
    )
    user_cf["final_score"] = (user_cf["final_score"] + user_cf["perturb"]).clip(0, 0.99)

    # ── 8. 排序取 Top-N ──
    user_cf = user_cf.sort_values("final_score", ascending=False).head(n)

    # ── 9. 合并元数据与库存 ──
    user_cf = user_cf.merge(item_meta[["StockCode", "Description_CN", "avg_price"]],
                            on="StockCode", how="left")
    user_cf = user_cf.merge(inventory[["StockCode", "stock", "status"]],
                            on="StockCode", how="left")

    # ── 10. 生成推荐理由 ──
    user_cf["reason"] = user_cf.apply(_generate_reason, axis=1,
                                       args=(user_cf["cf_component"].max(),
                                             user_cf["pop_component"].max(),
                                             user_cf["div_component"].max()))

    # ── 11. 构造推荐卡片 ──
    cards = []
    for _, row in user_cf.iterrows():
        card = {
            "StockCode": row["StockCode"],
            "Description_CN": str(row.get("Description_CN", "未知商品")),
            "final_score": round(float(row["final_score"]), 4),
            "cf_score": round(float(row.get("cf_rank", 0)), 4),
            "popularity_score": round(float(row.get("pop_rank", 0)), 4),
            "reason": row["reason"],
            "price": round(float(row.get("avg_price", 0) if pd.notna(row.get("avg_price")) else 0), 2),
            "stock": int(row.get("stock", 0) if pd.notna(row.get("stock")) else 0),
            "stock_status": str(row.get("status", "未知")),
        }
        cards.append(card)

    return cards


def _generate_reason(row, cf_max: float, pop_max: float, div_max: float) -> str:
    """生成更个性化的推荐理由。"""
    cf = row.get("cf_component", 0)
    pop = row.get("pop_component", 0)
    div = row.get("div_component", 0)

    # 找出主导因素
    parts = []
    if cf_max > 0 and cf / max(cf_max, 1e-9) > 0.4:
        parts.append("相似用户偏好")
    if pop_max > 0 and pop / max(pop_max, 1e-9) > 0.4:
        parts.append("近期热度上升")
    if div_max > 0 and div / max(div_max, 1e-9) > 0.4:
        parts.append("小众精选")

    if not parts:
        parts.append("综合推荐")

    return " · ".join(parts)


def _popularity_fallback(forecasts: pd.DataFrame, item_meta: pd.DataFrame,
                         inventory: pd.DataFrame, n: int = 10,
                         user_id: int = 0) -> list[dict]:
    """新用户冷启动回退：热门 + 多样性 + 用户差异化。

    不与流行性预测页的固定 Top-N 重复：
      - 从预测 Top-30 中按 user_id 确定性抽样（不同用户看到的不同）
      - 混入少量"小众精选"（CF 高频商品中热度中等的，增加多样性）
      - 用 user_id 哈希做排序扰动（同用户可复现，异用户不重复）

    Returns:
        推荐卡片列表，所有卡片均带 cold_start=True 标记
    """
    # 1) 准备元数据查询表
    meta_lookup = item_meta.set_index("StockCode")
    inv_lookup = inventory.set_index("StockCode") if inventory is not None else None

    # 确保 forecasts 只取第一预测周
    if "week_idx" in forecasts.columns:
        min_week = forecasts["week_idx"].min()
        fc = forecasts[forecasts["week_idx"] == min_week].copy()
    else:
        fc = forecasts.copy()

    # 2) 用 user_id 做确定性采样种子（同用户复现，异用户不同）
    import hashlib
    seed_str = f"coldstart_{user_id}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.default_rng(seed)

    top30 = fc.nlargest(30, "pred")
    sample_n = min(15, len(top30))
    top_sample = top30.sample(n=sample_n, random_state=int(rng.integers(0, 2**31 - 1)))

    # 3) 小众精选：预测销量在 20%~50% 分位之间，取 5 个
    mid_lo = fc["pred"].quantile(0.20)
    mid_hi = fc["pred"].quantile(0.50)
    mid_pool = fc[(fc["pred"] >= mid_lo) & (fc["pred"] <= mid_hi)]
    mid_n = min(5, len(mid_pool))
    if mid_n > 0:
        mid_sample = mid_pool.sample(n=mid_n, random_state=int(rng.integers(0, 2**31 - 1)))
    else:
        mid_sample = pd.DataFrame(columns=fc.columns)

    # 4) 合并去重
    merged = pd.concat([top_sample, mid_sample], ignore_index=True)
    merged = merged.drop_duplicates(subset="StockCode")
    merged = merged.dropna(subset=["StockCode"])

    # 5) 归一化 popularity_score
    mn, mx = merged["pred"].min(), merged["pred"].max()
    merged["popularity_score"] = ((merged["pred"] - mn) / (mx - mn + 1e-9)).astype(float)

    # 6) 确定性排序扰动
    merged["sort_key"] = merged["StockCode"].apply(
        lambda x: (hash(f"{seed}_{x}") % 1000) / 10000
    )
    merged = merged.sort_values(["pred", "sort_key"], ascending=[False, False]).head(n)

    # 7) 组装卡片
    cards = []
    for _, row in merged.iterrows():
        code = row["StockCode"]
        desc = str(meta_lookup.loc[code, "Description_CN"]) if code in meta_lookup.index else "未知商品"
        price = float(meta_lookup.loc[code, "avg_price"]) if code in meta_lookup.index and pd.notna(meta_lookup.loc[code, "avg_price"]) else 0.0

        stock_val = 0
        stock_status = "未知"
        if inv_lookup is not None and code in inv_lookup.index:
            stock_val = int(inv_lookup.loc[code, "stock"]) if pd.notna(inv_lookup.loc[code, "stock"]) else 0
            stock_status = str(inv_lookup.loc[code, "status"]) if pd.notna(inv_lookup.loc[code, "status"]) else "未知"

        cards.append({
            "StockCode": code,
            "Description_CN": desc,
            "final_score": round(float(row["popularity_score"]), 4),
            "cf_score": 0.0,
            "popularity_score": round(float(row["popularity_score"]), 4),
            "reason": "\U0001f525 热门推荐（暂无您的购买记录）",
            "price": round(price, 2),
            "stock": stock_val,
            "stock_status": stock_status,
            "cold_start": True,
        })
    return cards


# ════════════════════════════════════════════════════════════════════════════
# 4. 批量推荐（为多个用户生成推荐）
# ════════════════════════════════════════════════════════════════════════════
def batch_recommend(user_ids: list[int], n: int = 10,
                     strategy: Literal["balanced", "cf_focused", "trending"] = "balanced") -> dict:
    """为多个用户批量生成推荐。
    
    Args:
        user_ids: 用户ID列表
        n: 每个用户的推荐数量
        strategy: 推荐策略
    
    Returns:
        {user_id: [推荐卡片列表]}
    """
    print(f"\n为 {len(user_ids)} 个用户生成推荐...")

    # 加载数据（只加载一次）
    cf_topn, forecasts, item_meta, inventory = load_data()
    # 加载用户购买记录用于过滤已购
    user_item = pd.read_parquet(DATA_DIR / "user_item.parquet") \
        if (DATA_DIR / "user_item.parquet").exists() else None

    results = {}
    for i, user_id in enumerate(user_ids):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(user_ids)}")

        cards = recommend(user_id, n=n,
                         cf_topn=cf_topn, forecasts=forecasts,
                         item_meta=item_meta, inventory=inventory,
                         user_item=user_item, strategy=strategy)
        results[user_id] = cards

    print(f"批量推荐完成")
    return results


# ════════════════════════════════════════════════════════════════════════════
# 5. 推荐解释与分析
# ════════════════════════════════════════════════════════════════════════════
def explain_recommendation(user_id: int, show_top_n: int = 5):
    """详细解释推荐结果。"""
    print("─" * 70)
    print(f"用户 {user_id} 的推荐解释")
    print("─" * 70)

    # 加载用户购买历史
    user_item = pd.read_parquet(DATA_DIR / "user_item.parquet") \
        if (DATA_DIR / "user_item.parquet").exists() else None

    # 生成推荐
    cards = recommend(user_id, n=show_top_n, user_item=user_item)

    if not cards:
        print("该用户没有推荐结果（可能所有候选商品均已购买）")
        return

    # 显示购买历史
    if user_item is not None:
        user_history = (user_item[user_item["CustomerID"] == user_id]
                       .sort_values("n_purchase", ascending=False)
                       .head(3))
        if len(user_history) > 0:
            item_meta = pd.read_parquet(DATA_DIR / "item_meta.parquet")
            item_dict = item_meta.set_index("StockCode")["Description_CN"].to_dict()
            print("\n购买历史（Top 3）:")
            for _, row in user_history.iterrows():
                desc = item_dict.get(row["StockCode"], "未知")[:40]
                print(f"  • {row['StockCode']:8s} | {desc:40s} | 购买 {row['n_purchase']:2d} 次")

    print(f"\n个性化推荐（Top {show_top_n}）:")
    print(f"  融合策略: α(CF)={ALPHA}, β(流行度)={BETA}, γ(多样性)={GAMMA}")
    print()

    for i, card in enumerate(cards, start=1):
        print(f"  {i}. {card['StockCode']:8s} | {card['Description'][:35]:35s}")
        print(f"     最终得分: {card['final_score']:.4f}")
        print(f"     推荐理由: {card['reason']}")
        print(f"     价格: £{card['price']:.2f} | 库存: {card['stock']:4d} ({card['stock_status']})")
        print()

    print("─" * 70)


# ════════════════════════════════════════════════════════════════════════════
# 6. 主流程
# ════════════════════════════════════════════════════════════════════════════
def main():
    """主流程：展示推荐功能。"""
    print("═" * 70)
    print(" F5 个性化推荐模块")
    print("═" * 70)
    
    # 加载数据
    cf_topn, forecasts, item_meta, inventory = load_data()
    user_item = pd.read_parquet(DATA_DIR / "user_item.parquet") \
        if (DATA_DIR / "user_item.parquet").exists() else None
    
    # 示例1：单用户推荐
    print("\n" + "═" * 70)
    print("示例1：单用户推荐")
    print("═" * 70)
    
    sample_users = cf_topn["CustomerID"].unique()[:3]
    
    for user_id in sample_users:
        explain_recommendation(user_id, show_top_n=5)
    
    # 示例2：不同推荐策略对比
    print("\n" + "═" * 70)
    print("示例2：不同推荐策略对比")
    print("═" * 70)
    
    test_user = sample_users[0]
    strategies = ["balanced", "cf_focused", "trending"]
    
    for strategy in strategies:
        print(f"\n策略: {strategy.upper()}")
        print("─" * 70)
        cards = recommend(test_user, n=5, strategy=strategy,
                         cf_topn=cf_topn, forecasts=forecasts,
                         item_meta=item_meta, inventory=inventory,
                         user_item=user_item)
        
        for i, card in enumerate(cards, start=1):
            print(f"  {i}. {card['StockCode']:8s} | {card['Description'][:35]:35s}")
            print(f"     得分: {card['final_score']:.4f} | {card['reason']}")
    
    # 示例3：库存状态分析
    print("\n" + "═" * 70)
    print("示例3：推荐商品的库存状态分析")
    print("═" * 70)
    
    all_recommendations = []
    for user_id in sample_users[:10]:
        cards = recommend(user_id, n=10,
                         cf_topn=cf_topn, forecasts=forecasts,
                         item_meta=item_meta, inventory=inventory,
                         user_item=user_item)
        all_recommendations.extend(cards)
    
    # 统计库存状态
    status_counts = {}
    for card in all_recommendations:
        status = card["stock_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\n推荐商品库存状态分布（共 {len(all_recommendations)} 条推荐）:")
    for status, count in sorted(status_counts.items()):
        percentage = count / len(all_recommendations) * 100
        print(f"  {status:4s}: {count:4d} ({percentage:5.1f}%)")
    
    print("\n" + "═" * 70)
    print("[完成] F5 个性化推荐模块运行完成")
    print("═" * 70)


if __name__ == "__main__":
    main()
