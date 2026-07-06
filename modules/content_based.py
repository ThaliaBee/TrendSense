"""F9 内容感知推荐（新商品冷启动）。

用商品 Description 文本的 TF-IDF 向量化 + 余弦相似度，为没有历史销售数据的
"冷启动商品"找到描述最相似的其他商品。

输入：
    data/item_meta.parquet     商品元数据（含 Description）

产出：
    cache/content_sim.parquet  预计算的 Top-K 相似商品对（首次 fit 后缓存）

使用：
    cbr = ContentBasedRecommender()
    cbr.fit(item_meta)          # 只需跑一次，结果缓存到 cache/
    similar = cbr.find_similar("22222")   # 返回 [(code, score), ...]
    cards = cbr.recommend_cards("22222", item_meta, inventory, n=5)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── 路径 ────────────────────────────────────────────────────────────────────
PROJ_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_DIR / "data"
CACHE_DIR = PROJ_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── 参数 ────────────────────────────────────────────────────────────────────
MAX_FEATURES = 2000          # TF-IDF 最大特征词数
TOP_K_SIMILAR = 50           # 缓存时每个商品保留的相似商品数


class ContentBasedRecommender:
    """基于商品描述文本的内容感知推荐器。

    核心思路：用 TF-IDF 向量化所有商品的 Description 字段，
    计算余弦相似度矩阵，对于任意商品可按相似度排序找到描述最接近的 N 个商品。
    不依赖销售历史，天然适用新商品冷启动。
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=MAX_FEATURES,
            lowercase=True,
            strip_accents="unicode",
        )
        self.item_ids: list[str] = []
        self.tfidf_matrix = None
        self._sim_cache: pd.DataFrame | None = None  # 预计算的 Top-K 相似对

    # ── 1. 训练 / 加载 ─────────────────────────────────────────────────────
    def fit(self, item_meta: pd.DataFrame, force: bool = False) -> "ContentBasedRecommender":
        """构建 TF-IDF 矩阵并预计算 Top-K 相似对。

        首次 fit 后结果自动缓存到 cache/content_sim.parquet，
        后续调用直接从缓存加载（除非 force=True 强制重算）。

        Args:
            item_meta: 须含 StockCode 和 Description 两列
            force: 是否强制重新计算（忽略缓存）
        """
        cache_path = CACHE_DIR / "content_sim.parquet"

        if not force and cache_path.exists():
            self._sim_cache = pd.read_parquet(cache_path)
            self.item_ids = sorted(self._sim_cache["StockCode"].unique().tolist())
            print(f"[F9] 从缓存加载内容相似度: {len(self._sim_cache)} 条相似对 "
                  f"({len(self.item_ids)} 个商品)")
            return self

        # 清理描述文本
        descs = item_meta["Description"].fillna("").astype(str).tolist()
        print(f"[F9] 构建 TF-IDF 矩阵: {len(descs)} 个商品, max_features={MAX_FEATURES}")

        self.tfidf_matrix = self.vectorizer.fit_transform(descs)
        self.item_ids = item_meta["StockCode"].astype(str).tolist()

        # 预计算 Top-K 相似对（避免每次查询都算完整矩阵）
        print(f"[F9] 计算余弦相似度 Top-{TOP_K_SIMILAR}...")
        rows = []
        sim_mat = cosine_similarity(self.tfidf_matrix)

        item_id_map = {i: code for i, code in enumerate(self.item_ids)}
        for i in range(len(self.item_ids)):
            # 按相似度降序排列，跳过自身（idx 0 = 自己，相似度 1.0）
            nearest = np.argsort(sim_mat[i])[::-1]
            count = 0
            for j in nearest:
                if j == i:
                    continue
                rows.append({
                    "StockCode": self.item_ids[i],
                    "similar_code": item_id_map[j],
                    "similarity": round(float(sim_mat[i, j]), 6),
                    "rank": count + 1,
                })
                count += 1
                if count >= TOP_K_SIMILAR:
                    break

        self._sim_cache = pd.DataFrame(rows)
        self._sim_cache.to_parquet(cache_path, index=False)
        print(f"[F9] 已缓存 {len(self._sim_cache)} 条相似对 → {cache_path}")
        return self

    # ── 2. 查询相似商品 ─────────────────────────────────────────────────────
    def find_similar(self, stock_code: str, n: int = 10) -> list[tuple[str, float]]:
        """返回与给定商品描述最相似的 N 个商品。

        Args:
            stock_code: 目标商品 StockCode
            n: 返回数量

        Returns:
            [(StockCode, similarity_score), ...]，相似度已按降序排列
            如果商品不在矩阵中则返回空列表
        """
        if self._sim_cache is None:
            raise RuntimeError("请先调用 fit(item_meta) 或确保缓存存在")

        code = str(stock_code)
        mask = self._sim_cache["StockCode"] == code
        if not mask.any():
            return []

        result = (self._sim_cache[mask]
                  .sort_values("similarity", ascending=False)
                  .head(n))
        return [(str(r["similar_code"]), float(r["similarity"])) for _, r in result.iterrows()]

    # ── 3. 组装推荐卡片 ────────────────────────────────────────────────────
    def recommend_cards(self, stock_code: str, item_meta: pd.DataFrame,
                        inventory: pd.DataFrame | None = None,
                        n: int = 5) -> list[dict]:
        """为给定商品生成相似商品推荐卡片。

        Args:
            stock_code: 目标商品 StockCode
            item_meta: 商品元数据（至少含 StockCode, Description, avg_price）
            inventory: 库存状态（可选）
            n: 推荐数量

        Returns:
            推荐卡片列表，格式与 recommend.recommend() 一致
        """
        similar = self.find_similar(stock_code, n=n)
        if not similar:
            return []

        meta_lookup = item_meta.set_index("StockCode")
        inv_lookup = inventory.set_index("StockCode") if inventory is not None else None

        # 获取源商品名
        source_desc = "未知商品"
        code = str(stock_code)
        if code in meta_lookup.index:
            source_desc = str(meta_lookup.loc[code, "Description_CN"])[:40]

        cards = []
        for sim_code, sim_score in similar:
            desc = (str(meta_lookup.loc[sim_code, "Description_CN"])
                    if sim_code in meta_lookup.index else "未知商品")
            price = 0.0
            if sim_code in meta_lookup.index:
                p = meta_lookup.loc[sim_code, "avg_price"]
                price = float(p) if pd.notna(p) else 0.0

            stock_val = 0
            stock_status = "未知"
            if inv_lookup is not None and sim_code in inv_lookup.index:
                stock_val = int(inv_lookup.loc[sim_code, "stock"]) if pd.notna(inv_lookup.loc[sim_code, "stock"]) else 0
                stock_status = str(inv_lookup.loc[sim_code, "status"]) if pd.notna(inv_lookup.loc[sim_code, "status"]) else "未知"

            cards.append({
                "StockCode": sim_code,
                "Description_CN": desc,
                "final_score": round(sim_score, 4),
                "cf_score": 0.0,
                "popularity_score": round(sim_score, 4),
                "reason": f"\U0001f4dd 与「{source_desc}」描述相似",
                "price": round(price, 2),
                "stock": stock_val,
                "stock_status": stock_status,
                "cold_start": False,
                "content_based": True,
            })
        return cards


# ════════════════════════════════════════════════════════════════════════════
# 便捷函数：懒加载全局实例
# ════════════════════════════════════════════════════════════════════════════
_CBR = None


def get_cbr(item_meta: pd.DataFrame | None = None) -> ContentBasedRecommender:
    """获取全局 ContentBasedRecommender 实例（懒加载 + 缓存）。

    首次调用时自动 fit（或从 cache/content_sim.parquet 加载），
    后续调用返回同一实例。
    """
    global _CBR
    if _CBR is None:
        _CBR = ContentBasedRecommender()
        if item_meta is not None:
            _CBR.fit(item_meta)
        elif (CACHE_DIR / "content_sim.parquet").exists():
            _CBR.fit(pd.DataFrame())  # fit 会从缓存加载
    return _CBR


# ════════════════════════════════════════════════════════════════════════════
def main():
    """自检：构建内容相似度矩阵并展示示例。"""
    print("═" * 56)
    print(" F9 内容感知推荐")
    print("═" * 56)

    item_meta = pd.read_parquet(DATA_DIR / "item_meta.parquet")
    cbr = ContentBasedRecommender()
    cbr.fit(item_meta)

    # 随机选 3 个商品展示相似品
    sample_codes = item_meta["StockCode"].sample(3, random_state=42).tolist()
    for code in sample_codes:
        desc = item_meta[item_meta["StockCode"] == code]["Description"].values[0][:50]
        print(f"\n─" * 56)
        print(f"📦 {code} — {desc}")
        similar = cbr.find_similar(code, n=5)
        for rank, (sim_code, score) in enumerate(similar, 1):
            sim_desc = item_meta[item_meta["StockCode"] == sim_code]["Description"].values
            sim_desc = sim_desc[0][:50] if len(sim_desc) > 0 else "?"
            print(f"  {rank}. {sim_code} — {sim_desc}  (相似度: {score:.4f})")

    # 生成推荐卡片样例
    print(f"\n─" * 56)
    print("示例推荐卡片:")
    inventory = None
    inv_path = CACHE_DIR / "inventory_status.parquet"
    if inv_path.exists():
        inventory = pd.read_parquet(inv_path)
    cards = cbr.recommend_cards(sample_codes[0], item_meta, inventory, n=3)
    for i, card in enumerate(cards, 1):
        print(f"  {i}. {card['Description']:<40s} 相似度={card['final_score']:.4f}  "
              f"价格=£{card['price']:.2f}  库存={card['stock']} ({card['stock_status']})")

    print(f"\n[完成] F9 内容感知推荐模块运行完成")
    print("═" * 56)


if __name__ == "__main__":
    main()
