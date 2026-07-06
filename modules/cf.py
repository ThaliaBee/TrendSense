"""协同过滤（User-CF + Item-CF）。

实现基于用户(User-CF)和基于物品(Item-CF)的协同过滤算法，
用于商品推荐和偏好预测。采用隐式反馈（购买次数作为隐式评分）。

输入：
    data/user_item.parquet          用户×商品隐式评分
    data/clean_transactions.parquet  清洗后交易明细（用于时间留出评估）

输出：
    cache/cf_topn.parquet           每用户预计算Top-N推荐
    cache/cf_metrics.json           评估指标

运行：
    python modules/cf.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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
CACHE_DIR.mkdir(exist_ok=True)

# ── 参数 ────────────────────────────────────────────────────────────────────
DEFAULT_K = 10              # Top-K 推荐数量
MIN_SIMILARITY = 0.0        # 最小相似度阈值（过滤噪声）
TOP_N_NEIGHBORS = 50        # 计算推荐时考虑的最近邻数量


# ════════════════════════════════════════════════════════════════════════════
# 1. CFModel 基类
# ════════════════════════════════════════════════════════════════════════════
class CFModel:
    """协同过滤模型基类。"""
    
    def __init__(self, method: Literal["user", "item"] = "item"):
        self.method = method
        self.user_item_matrix = None  # 用户-商品矩阵（稀疏）
        self.similarity_matrix = None  # 相似度矩阵
        self.users = None              # 用户ID列表
        self.items = None              # 商品ID列表
        
    def fit(self, user_item_df: pd.DataFrame):
        """训练模型：构建评分矩阵并计算相似度。
        
        Args:
            user_item_df: 包含 CustomerID, StockCode, n_purchase 的数据框
        """
        # 构建用户-商品矩阵
        matrix_df = user_item_df.pivot(
            index="CustomerID",
            columns="StockCode", 
            values="n_purchase"
        ).fillna(0)
        
        self.users = matrix_df.index.tolist()
        self.items = matrix_df.columns.tolist()
        self.user_item_matrix = matrix_df.values
        
        print(f"构建评分矩阵: {len(self.users)} 用户 × {len(self.items)} 商品")
        print(f"稀疏度: {(self.user_item_matrix == 0).sum() / self.user_item_matrix.size:.2%}")
        
        # 计算相似度矩阵
        if self.method == "user":
            # User-CF: 用户-用户相似度
            print("计算用户-用户相似度（余弦相似度）...")
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
            np.fill_diagonal(self.similarity_matrix, 0)  # 自己和自己不相似
            print(f"相似度矩阵: {self.similarity_matrix.shape}")
            
        elif self.method == "item":
            # Item-CF: 物品-物品相似度
            print("计算物品-物品相似度（余弦相似度）...")
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            np.fill_diagonal(self.similarity_matrix, 0)
            print(f"相似度矩阵: {self.similarity_matrix.shape}")
        
        return self
    
    def predict(self, user_id: int, k: int = DEFAULT_K,
                allow_repurchase: bool = True) -> list[tuple[str, float]]:
        """为用户预测商品偏好并返回Top-K推荐。
        
        Args:
            user_id: 用户ID
            k: 返回的推荐数量
            allow_repurchase: 是否允许推荐用户已购买过的商品
            
        Returns:
            [(StockCode, cf_score), ...] 推荐列表
        """
        if user_id not in self.users:
            return []
        
        user_idx = self.users.index(user_id)
        user_ratings = self.user_item_matrix[user_idx]
        
        if self.method == "user":
            # User-CF: 基于相似用户的喜好预测
            scores = self._predict_user_cf(user_idx, user_ratings)
        else:
            # Item-CF: 基于相似物品的偏好预测
            scores = self._predict_item_cf(user_idx, user_ratings)
        
        # 过滤已购买的商品（如果不允许复购）
        if not allow_repurchase:
            purchased_mask = user_ratings > 0
            scores[purchased_mask] = -np.inf
        
        # 返回Top-K
        top_k_indices = np.argsort(scores)[::-1][:k]
        recommendations = [
            (self.items[idx], float(scores[idx]))
            for idx in top_k_indices
            if scores[idx] > MIN_SIMILARITY
        ]
        
        return recommendations
    
    def _predict_user_cf(self, user_idx: int, user_ratings: np.ndarray) -> np.ndarray:
        """User-CF预测：基于相似用户的加权平均。"""
        similarities = self.similarity_matrix[user_idx]
        
        # 选择Top-N相似用户
        top_n_idx = np.argsort(similarities)[::-1][:TOP_N_NEIGHBORS]
        top_n_sim = similarities[top_n_idx]
        top_n_ratings = self.user_item_matrix[top_n_idx]
        
        # 加权求和
        numerator = np.dot(top_n_sim, top_n_ratings)
        denominator = np.abs(top_n_sim).sum() + 1e-9
        
        scores = numerator / denominator
        return scores
    
    def _predict_item_cf(self, user_idx: int, user_ratings: np.ndarray) -> np.ndarray:
        """Item-CF预测：基于用户已购买商品的相似商品。"""
        purchased_items = np.where(user_ratings > 0)[0]
        
        if len(purchased_items) == 0:
            return np.zeros(len(self.items))
        
        scores = np.zeros(len(self.items))
        
        # 对每个已购买的商品，找相似商品并累加得分
        for item_idx in purchased_items:
            item_similarities = self.similarity_matrix[item_idx]
            # 加权：购买次数 × 相似度
            weight = user_ratings[item_idx]
            scores += weight * item_similarities
        
        # 归一化
        scores = scores / (len(purchased_items) + 1e-9)
        
        return scores


# ════════════════════════════════════════════════════════════════════════════
# 2. 训练函数
# ════════════════════════════════════════════════════════════════════════════
def train_cf(user_item_path: str | Path = DATA_DIR / "user_item.parquet",
             method: Literal["user", "item"] = "item") -> CFModel:
    """训练协同过滤模型。
    
    Args:
        user_item_path: 用户-商品隐式评分文件路径
        method: 'user' 或 'item'
        
    Returns:
        CFModel 训练好的模型
    """
    print(f"[F2 协同过滤] 训练 {method.upper()}-CF 模型")
    print("─" * 56)
    
    user_item_df = pd.read_parquet(user_item_path)
    print(f"加载数据: {len(user_item_df)} 条用户-商品记录")
    
    model = CFModel(method=method)
    model.fit(user_item_df)
    
    print("─" * 56)
    print(f"{method.upper()}-CF 模型训练完成")
    
    return model


# ════════════════════════════════════════════════════════════════════════════
# 3. 推荐函数
# ════════════════════════════════════════════════════════════════════════════
def recommend_cf(model: CFModel, user_id: int, k: int = DEFAULT_K,
                 allow_repurchase: bool = True) -> list[tuple[str, float]]:
    """为用户生成Top-K推荐。
    
    Args:
        model: 训练好的CF模型
        user_id: 用户ID
        k: 推荐数量
        allow_repurchase: 是否允许推荐已购买商品
        
    Returns:
        [(StockCode, cf_score), ...] 推荐列表
    """
    return model.predict(user_id, k=k, allow_repurchase=allow_repurchase)


# ════════════════════════════════════════════════════════════════════════════
# 4. 评估函数（时间留出法）
# ════════════════════════════════════════════════════════════════════════════
def evaluate_cf(model: CFModel, test_week_start: int = 47) -> dict:
    """时间留出评估：用最后几周的购买作为测试集。
    
    Args:
        model: 训练好的CF模型
        test_week_start: 测试集起始周（默认47，即周47-53为测试）
        
    Returns:
        评估指标字典
    """
    print(f"\n[评估] 时间留出法，测试集：周 {test_week_start}~53")
    print("─" * 56)
    
    # 加载完整交易数据（包含时间信息）—— 改从数据库读（连不上自动回退文件）
    if str(PROJ_DIR) not in sys.path:
        sys.path.insert(0, str(PROJ_DIR))
    from modules import db
    clean_df = db.load_transactions()
    
    # 分离训练集和测试集
    train_df = clean_df[clean_df["week_idx"] < test_week_start]
    test_df = clean_df[clean_df["week_idx"] >= test_week_start]
    
    # 构建测试集真实购买集合（每个用户在测试期购买的商品）
    test_purchases = (test_df.groupby("CustomerID")["StockCode"]
                      .apply(set).to_dict())
    
    print(f"训练集: 周 0~{test_week_start-1}, 用户数 {train_df['CustomerID'].nunique()}")
    print(f"测试集: 周 {test_week_start}~53, 用户数 {test_df['CustomerID'].nunique()}")
    
    # 只评估在测试集中有购买的用户
    test_users = list(test_purchases.keys())
    test_users = [u for u in test_users if u in model.users]
    
    if len(test_users) == 0:
        print("警告: 没有可评估的用户（测试集用户不在训练集中）")
        return {}
    
    print(f"可评估用户数: {len(test_users)}")
    
    # 计算推荐并评估
    k = DEFAULT_K
    hits = 0
    total_precision = 0.0
    total_recall = 0.0
    
    for user_id in test_users:
        # 获取推荐（允许复购，因为要评估是否推荐了测试期购买的商品）
        recs = recommend_cf(model, user_id, k=k, allow_repurchase=True)
        rec_items = {item for item, _ in recs}
        
        # 真实购买
        true_items = test_purchases.get(user_id, set())
        
        # 计算指标
        hit_items = rec_items & true_items
        
        if len(hit_items) > 0:
            hits += 1
        
        precision = len(hit_items) / len(rec_items) if len(rec_items) > 0 else 0.0
        recall = len(hit_items) / len(true_items) if len(true_items) > 0 else 0.0
        
        total_precision += precision
        total_recall += recall
    
    # 平均指标
    hit_rate = hits / len(test_users)
    avg_precision = total_precision / len(test_users)
    avg_recall = total_recall / len(test_users)
    
    # 基线：推荐用户历史高频购买商品
    baseline_hits = 0
    for user_id in test_users:
        user_idx = model.users.index(user_id)
        user_ratings = model.user_item_matrix[user_idx]
        
        # 历史购买最多的K个商品
        top_k_idx = np.argsort(user_ratings)[::-1][:k]
        baseline_recs = {model.items[idx] for idx in top_k_idx if user_ratings[idx] > 0}
        
        true_items = test_purchases.get(user_id, set())
        if len(baseline_recs & true_items) > 0:
            baseline_hits += 1
    
    baseline_hit_rate = baseline_hits / len(test_users)
    
    metrics = {
        "method": model.method + "-cf",
        "hit_rate@10": round(hit_rate, 4),
        "precision@10": round(avg_precision, 4),
        "recall@10": round(avg_recall, 4),
        "baseline_hit_rate@10": round(baseline_hit_rate, 4),
        "test_users": len(test_users)
    }
    
    print(f"Hit Rate@{k}      : {hit_rate:.4f}")
    print(f"Precision@{k}     : {avg_precision:.4f}")
    print(f"Recall@{k}        : {avg_recall:.4f}")
    print(f"Baseline Hit@{k}  : {baseline_hit_rate:.4f}")
    print(f"提升              : {(hit_rate - baseline_hit_rate):.4f}")
    print("─" * 56)
    
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# 5. 预计算Top-N（缓存到文件，供F5/F7使用）
# ════════════════════════════════════════════════════════════════════════════
def precompute_topn(model: CFModel, k: int = DEFAULT_K,
                    output_path: Path = CACHE_DIR / "cf_topn.parquet") -> None:
    """为所有用户预计算Top-N推荐，缓存到文件。"""
    print(f"\n[预计算] 为所有用户生成 Top-{k} 推荐...")
    
    results = []
    for i, user_id in enumerate(model.users):
        if (i + 1) % 500 == 0:
            print(f"  进度: {i+1}/{len(model.users)}")
        
        recs = recommend_cf(model, user_id, k=k, allow_repurchase=True)
        
        for rank, (stock_code, cf_score) in enumerate(recs, start=1):
            results.append({
                "CustomerID": user_id,
                "StockCode": stock_code,
                "cf_score": round(cf_score, 4),
                "rank": rank
            })
    
    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False)
    
    print(f"预计算完成: {len(df)} 条推荐记录")
    print(f"保存至: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# 6. 主流程
# ════════════════════════════════════════════════════════════════════════════
def main():
    """主流程：训练模型、评估、预计算Top-N。"""
    print("═" * 56)
    print(" F2 协同过滤模块（融合版本）")
    print("═" * 56)

    # 选择方法：改为融合策略
    method = "hybrid"  # ← 使用融合策略！

    if method == "hybrid":
        # 融合策略：训练两个模型
        print("\n>>> 训练 Item-CF 模型...")
        model_item = train_cf(method="item")

        print("\n>>> 训练 User-CF 模型...")
        model_user = train_cf(method="user")

        # 使用0.3/0.7权重融合
        print("\n>>> 使用融合策略 (0.3 Item + 0.7 User)...")

        # 创建融合模型包装器
        class HybridModel:
            def __init__(self, model_item, model_user, alpha=0.3, beta=0.7):
                self.model_item = model_item
                self.model_user = model_user
                self.alpha = alpha
                self.beta = beta
                self.users = model_item.users
                self.items = model_item.items
                self.user_item_matrix = model_item.user_item_matrix  # ← 添加这个
                self.method = f"hybrid-{alpha}/{beta}"

            def predict(self, user_id, k=10, allow_repurchase=True):
                # Item-CF推荐
                item_recs = self.model_item.predict(user_id, k=k*3, allow_repurchase=allow_repurchase)
                item_scores = {code: score for code, score in item_recs}

                # User-CF推荐
                user_recs = self.model_user.predict(user_id, k=k*3, allow_repurchase=allow_repurchase)
                user_scores_raw = {code: score for code, score in user_recs}

                # 归一化User-CF得分
                if user_scores_raw:
                    user_max = max(user_scores_raw.values())
                    user_min = min(user_scores_raw.values())
                    if user_max > user_min:
                        user_scores = {
                            code: (score - user_min) / (user_max - user_min)
                            for code, score in user_scores_raw.items()
                        }
                    else:
                        user_scores = {code: 0.5 for code in user_scores_raw}
                else:
                    user_scores = {}

                # 融合
                all_items = set(item_scores.keys()) | set(user_scores.keys())
                hybrid_scores = {}
                for item in all_items:
                    hybrid_scores[item] = (
                        self.alpha * item_scores.get(item, 0) +
                        self.beta * user_scores.get(item, 0)
                    )

                # Top-K
                sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
                return sorted_items[:k]

        model = HybridModel(model_item, model_user, alpha=0.3, beta=0.7)
    else:
        # 单一方法
        model = train_cf(method=method)
    
    # 2. 评估
    metrics = evaluate_cf(model, test_week_start=47)
    
    # 3. 预计算Top-N
    precompute_topn(model, k=DEFAULT_K)
    
    # 4. 保存评估指标
    metrics_path = CACHE_DIR / "cf_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估指标已保存至: {metrics_path}")
    print("\n[完成] F2 协同过滤模块运行完成")
    print("═" * 56)


if __name__ == "__main__":
    main()
