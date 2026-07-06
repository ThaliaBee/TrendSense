"""LSTM 深度学习流行性预测 + 置信区间。

用一个 LSTM 时间序列模型预测「商品未来每周销量」（= 流行度），并给出置信区间。

核心设计：
  - 输入：data/item_week_sales.parquet（商品 × 周 销量），仅对活跃商品建模；
  - 一个模型套用到所有活跃商品：滑动窗口 (过去 LOOKBACK 周 → 下一周)；
  - 按商品 log1p + 标准化，统计量仅用训练集算（防数据泄露）；
  - 时间三分：训练(周0-40) / 验证(41-46) / 测试(47-53)，绝不随机打乱；
  - 损失 Huber(SmoothL1)；评估 MAE/RMSE（还原成件）+ 移动平均基线 + Top-K 命中率；
  - 置信区间：验证集残差标准差 σ，pred ± 1.96σ（多步按 √step 放宽）。

产出：
  models/lstm.pt          模型权重 + 配置
  models/scaler.json      每商品归一化参数 + 全局 σ（与模型成对，缺一不可）
  cache/forecasts.parquet 各活跃商品未来若干周预测（StockCode, week_idx, pred, lower, upper）
  cache/lstm_metrics.json 评估指标

运行：
  python modules/lstm_popularity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── 路径 ─────────────────────────────────────────────────────────────────────
PROJ_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_DIR / "data"
MODELS_DIR = PROJ_DIR / "models"
CACHE_DIR = PROJ_DIR / "cache"
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── 超参 / 划分 ──────────────────────────────────────────────────────────────
LOOKBACK = 8           # 回看周数
TRAIN_END = 40         # 训练集目标周上界（含）：目标周 8..40
VAL_END = 46           # 验证集：目标周 41..46
TEST_END = 53          # 测试集：目标周 47..53（数据最后一周）
HIDDEN = 128
NUM_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 200
LR = 2e-3
BATCH = 256
PATIENCE = 25          # 早停
HORIZON_FUTURE = 4     # 预计算缓存：往数据末尾之后预测多少周
TOPK = 20              # Top-K 命中率的 K
CONF_Z = 1.28          # 置信区间分位数：1.28≈80%、1.645≈90%、1.96≈95%
USE_TIME_FEATURES = True   # 是否加入季节性时间特征（可做消融对比）
WEEK_PERIOD = 52       # 季节周期（一年约 52 周）
SEED = 42

N_FEATURES = 3 if USE_TIME_FEATURES else 1
DEVICE = torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ════════════════════════════════════════════════════════════════════════════
# 数据：商品 × 周 销量矩阵
# ════════════════════════════════════════════════════════════════════════════
def load_series() -> tuple[list[str], np.ndarray]:
    """返回 (活跃商品ID列表, 销量矩阵[n_items, n_weeks])。"""
    df = pd.read_parquet(DATA_DIR / "item_week_sales.parquet")
    active = json.load(open(DATA_DIR / "active_items.json", encoding="utf-8"))
    active = [c for c in active if c in set(df["StockCode"])]
    n_weeks = int(df["week_idx"].max()) + 1
    # 透视成 [item, week] 矩阵（item_week_sales 已补 0 对齐）
    pivot = (df[df["StockCode"].isin(active)]
             .pivot(index="StockCode", columns="week_idx", values="sales")
             .reindex(columns=range(n_weeks), fill_value=0)
             .fillna(0))
    item_ids = pivot.index.tolist()
    mat = pivot.to_numpy(dtype=np.float32)        # [n_items, n_weeks]
    return item_ids, mat


def build_scaler(mat: np.ndarray, train_end: int) -> dict:
    """每商品在 log1p 空间、仅用训练周(0..train_end)算 mean/std。"""
    log_train = np.log1p(mat[:, : train_end + 1])
    mean = log_train.mean(axis=1)
    std = log_train.std(axis=1)
    std = np.maximum(std, 1e-3)                   # 防止常数序列除以 0
    return {"mean": mean, "std": std}


def normalize(mat: np.ndarray, scaler: dict) -> np.ndarray:
    return (np.log1p(mat) - scaler["mean"][:, None]) / scaler["std"][:, None]


def denorm_value(z: float, mean: float, std: float) -> float:
    """单值反归一化：z → 件数，clip 到 0。"""
    return float(max(np.expm1(z * std + mean), 0.0))


def time_feats(week_idx: int) -> list[float]:
    ang = 2 * np.pi * (week_idx % WEEK_PERIOD) / WEEK_PERIOD
    return [float(np.sin(ang)), float(np.cos(ang))]


# ════════════════════════════════════════════════════════════════════════════
# 构造滑动窗口样本（按目标周划分 train/val/test，防泄露）
# ════════════════════════════════════════════════════════════════════════════
def build_samples(norm_mat: np.ndarray) -> dict:
    n_items, n_weeks = norm_mat.shape
    buckets = {"train": [], "val": [], "test": []}  # 每项: (item_idx, X[L,F], y)
    for t in range(LOOKBACK, n_weeks):
        if t <= TRAIN_END:
            split = "train"
        elif t <= VAL_END:
            split = "val"
        else:
            split = "test"
        feats_t = [time_feats(t - LOOKBACK + k) for k in range(LOOKBACK)]  # 各步时间特征
        for i in range(n_items):
            window = norm_mat[i, t - LOOKBACK: t]                 # [L]
            if USE_TIME_FEATURES:
                X = np.column_stack([window, np.array(feats_t)])  # [L, 3]
            else:
                X = window[:, None]                               # [L, 1]
            buckets[split].append((i, X.astype(np.float32), np.float32(norm_mat[i, t])))
    return buckets


def to_tensors(samples: list) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    X = torch.tensor(np.stack([s[1] for s in samples]))           # [N, L, F]
    y = torch.tensor(np.array([s[2] for s in samples]))[:, None]  # [N, 1]
    item_idx = np.array([s[0] for s in samples])
    return X, y, item_idx


# ════════════════════════════════════════════════════════════════════════════
# 模型
# ════════════════════════════════════════════════════════════════════════════
class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = HIDDEN, layers: int = NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=DROPOUT if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: [B, L, F]
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])        # 取最后时间步 → [B, 1]


# ════════════════════════════════════════════════════════════════════════════
# 训练
# ════════════════════════════════════════════════════════════════════════════
def train_lstm(item_week_path: str | None = None, active_items=None, lookback: int = LOOKBACK):
    """训练并保存 models/lstm.pt + models/scaler.json。返回 (model, scaler_dict)。"""
    set_seed()
    item_ids, mat = load_series()
    scaler = build_scaler(mat, TRAIN_END)
    norm_mat = normalize(mat, scaler)
    buckets = build_samples(norm_mat)

    Xtr, ytr, _ = to_tensors(buckets["train"])
    Xva, yva, _ = to_tensors(buckets["val"])
    print(f"样本数  训练={len(Xtr)}  验证={len(Xva)}  测试={len(buckets['test'])}"
          f"  (活跃商品 {len(item_ids)} 个, 特征数 {N_FEATURES})")

    model = LSTMForecaster().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()                # Huber

    best_val, best_state, wait = float("inf"), None, 0
    n = len(Xtr)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)               # 打乱样本顺序≠打乱时间，安全
        total = 0.0
        for b in range(0, n, BATCH):
            idx = perm[b: b + BATCH]
            xb, yb = Xtr[idx].to(DEVICE), ytr[idx].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva.to(DEVICE)), yva.to(DEVICE)).item()
        if val_loss < best_val - 1e-5:
            best_val, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  train_loss={total / n:.4f}  val_loss={val_loss:.4f}")
        if wait >= PATIENCE:
            print(f"  早停于 epoch {epoch}（验证损失 {PATIENCE} 轮未改善）")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # 置信区间：每个商品在归一化(z)空间各自的残差标准差 σ_i。
    # 在 z(log) 空间 ±1.96σ 再 expm1 还原 → 区间不对称且恒为正，且随商品自适应，
    # 不会再被海量长尾/间歇商品的相对误差污染（旧的全局倍率法的根因）。
    model.eval()
    _, _, tr_item = to_tensors(buckets["train"])
    _, _, va_item = to_tensors(buckets["val"])
    with torch.no_grad():
        pred_tr_z = model(Xtr.to(DEVICE)).cpu().numpy().ravel()
        pred_va_z = model(Xva.to(DEVICE)).cpu().numpy().ravel()
    fit_item = np.concatenate([tr_item, va_item])                 # 每个残差属于哪个商品
    resid_z = np.concatenate([ytr.numpy().ravel() - pred_tr_z,    # z 空间残差(近似零均值)
                              yva.numpy().ravel() - pred_va_z])
    sigma_z = float(np.sqrt(np.mean(resid_z ** 2))) if len(resid_z) else 1.0  # 全局残差 std(收缩基准/兜底)

    # 按商品聚合残差 → σ_i，并向全局 sigma_z 收缩(样本少的商品估计更稳)
    n_items = len(item_ids)
    N0 = 8.0                                                       # 收缩强度(等效先验样本数)
    cnt = np.bincount(fit_item, minlength=n_items).astype(float)
    ss = np.bincount(fit_item, weights=resid_z ** 2, minlength=n_items)
    var_i = ss / np.maximum(cnt, 1.0)
    var_shrunk = (cnt * var_i + N0 * sigma_z ** 2) / (cnt + N0)
    sigma_i = np.sqrt(var_shrunk)                                 # [n_items] 每商品 z 空间残差 std

    # 保存（模型 + 归一化参数 成对）
    torch.save({"state_dict": model.state_dict(),
                "config": {"n_features": N_FEATURES, "hidden": HIDDEN,
                           "num_layers": NUM_LAYERS, "lookback": LOOKBACK,
                           "use_time_features": USE_TIME_FEATURES}}, MODELS_DIR / "lstm.pt")
    scaler_json = {
        "method": "log1p+standardize",
        "lookback": LOOKBACK,
        "sigma_z": sigma_z,                      # 全局残差 std：σ_i 缺失时兜底
        "week_period": WEEK_PERIOD,
        "items": {item_ids[i]: {"mean": float(scaler["mean"][i]),
                                "std": float(scaler["std"][i]),
                                "sigma": float(sigma_i[i])}   # 该商品 z 空间残差 std
                  for i in range(len(item_ids))},
    }
    json.dump(scaler_json, open(MODELS_DIR / "scaler.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"已保存 models/lstm.pt 与 models/scaler.json（σ_z={sigma_z:.4f}）")
    return model, scaler_json


# ════════════════════════════════════════════════════════════════════════════
# 推理用：懒加载模型 + scaler + 历史
# ════════════════════════════════════════════════════════════════════════════
_ARTIFACTS = {}


def _load_artifacts():
    if _ARTIFACTS:
        return _ARTIFACTS
    ckpt = torch.load(MODELS_DIR / "lstm.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = LSTMForecaster(cfg["n_features"], cfg["hidden"], cfg["num_layers"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = json.load(open(MODELS_DIR / "scaler.json", encoding="utf-8"))
    df = pd.read_parquet(DATA_DIR / "item_week_sales.parquet")
    n_weeks = int(df["week_idx"].max()) + 1
    hist = (df.pivot(index="StockCode", columns="week_idx", values="sales")
            .reindex(columns=range(n_weeks), fill_value=0).fillna(0))
    _ARTIFACTS.update(model=model, cfg=cfg, scaler=scaler, hist=hist, n_weeks=n_weeks)
    return _ARTIFACTS


def forecast(item_id: str, horizon: int = 1) -> dict:
    """返回 {history, pred, lower, upper}。history 为该商品历史真实销量(件)。"""
    art = _load_artifacts()
    model, scaler, hist = art["model"], art["scaler"], art["hist"]
    if item_id not in scaler["items"]:
        raise KeyError(f"{item_id} 不是活跃商品，未建模")
    mean, std = scaler["items"][item_id]["mean"], scaler["items"][item_id]["std"]
    sigma = scaler["items"][item_id].get("sigma", scaler.get("sigma_z", 1.0))  # z 空间残差 std
    L, use_tf, period = scaler["lookback"], art["cfg"]["use_time_features"], scaler["week_period"]
    n_weeks = art["n_weeks"]

    raw_hist = hist.loc[item_id].to_numpy(dtype=np.float32)        # 件
    z_seq = list((np.log1p(raw_hist) - mean) / std)               # 归一化序列（可追加预测）

    preds, lowers, uppers = [], [], []
    for h in range(horizon):
        t = n_weeks + h                                            # 预测的未来周序号
        window = np.array(z_seq[-L:], dtype=np.float32)
        if use_tf:
            ang = 2 * np.pi * ((np.arange(t - L, t)) % period) / period
            X = np.column_stack([window, np.sin(ang), np.cos(ang)]).astype(np.float32)
        else:
            X = window[:, None]
        with torch.no_grad():
            z_pred = float(model(torch.tensor(X)[None]).item())
        p = denorm_value(z_pred, mean, std)                        # 点预测（件）
        widen = float(np.sqrt(h + 1))                              # 多步按 √(h+1) 放宽（递归误差累积）
        half = CONF_Z * sigma * widen                              # z 空间半宽（CONF_Z 决定置信水平）
        lower = max(float(np.expm1((z_pred - half) * std + mean)), 0.0)  # log 空间 ±σ 后还原 → 不对称
        upper = float(np.expm1((z_pred + half) * std + mean))
        # 封顶：log 空间上界是乘性的，多步会指数放大；限制在「历史周峰值×1.5」与「2×点预测」的较大者，
        # 既挡住爆炸，又保留 log 空间漂亮的非对称下界（业务含义：不预期销量超历史峰值 1.5 倍）。
        upper = min(upper, max(float(raw_hist.max()) * 1.5, p * 2.0))
        preds.append(p)
        lowers.append(lower)                                       # expm1 单调，天然 lower ≤ p ≤ upper
        uppers.append(max(upper, p))                               # cap 后仍保证 upper ≥ pred
        z_seq.append(z_pred)                                       # 递归：预测接回输入
    return {"history": raw_hist.tolist(), "pred": preds, "lower": lowers, "upper": uppers}


# ════════════════════════════════════════════════════════════════════════════
# 评估（测试集：MAE/RMSE + 移动平均基线 + Top-K 命中率）
# ════════════════════════════════════════════════════════════════════════════
def evaluate_lstm(test_weeks=None) -> dict:
    art = _load_artifacts()
    model, scaler, hist = art["model"], art["scaler"], art["hist"]
    item_ids = list(scaler["items"].keys())
    means = np.array([scaler["items"][c]["mean"] for c in item_ids])
    stds = np.array([scaler["items"][c]["std"] for c in item_ids])
    mat = hist.loc[item_ids].to_numpy(dtype=np.float32)
    norm_mat = (np.log1p(mat) - means[:, None]) / stds[:, None]
    use_tf = art["cfg"]["use_time_features"]
    weeks = test_weeks or list(range(VAL_END + 1, TEST_END + 1))

    abs_err, sq_err, base_abs = [], [], []
    hit_list = []
    for t in weeks:
        # 预测全部活跃商品在第 t 周的销量
        feats_t = [time_feats(t - LOOKBACK + k) for k in range(LOOKBACK)]
        Xs = []
        for i in range(len(item_ids)):
            window = norm_mat[i, t - LOOKBACK: t]
            X = np.column_stack([window, np.array(feats_t)]) if use_tf else window[:, None]
            Xs.append(X.astype(np.float32))
        with torch.no_grad():
            z_pred = model(torch.tensor(np.stack(Xs))).cpu().numpy().ravel()
        pred = np.expm1(z_pred * stds + means).clip(min=0)          # 还原成件
        actual = mat[:, t]
        # 移动平均基线：过去 LOOKBACK 周均值
        base = mat[:, t - LOOKBACK: t].mean(axis=1)

        abs_err.append(np.abs(pred - actual))
        sq_err.append((pred - actual) ** 2)
        base_abs.append(np.abs(base - actual))
        # Top-K 命中率：预测最火 K 个 vs 真实最火 K 个
        pred_top = set(np.argsort(-pred)[:TOPK])
        true_top = set(np.argsort(-actual)[:TOPK])
        hit_list.append(len(pred_top & true_top) / TOPK)

    mae = float(np.mean(np.concatenate(abs_err)))
    rmse = float(np.sqrt(np.mean(np.concatenate(sq_err))))
    base_mae = float(np.mean(np.concatenate(base_abs)))
    precision_at_k = float(np.mean(hit_list))
    return {"mae": round(mae, 3), "rmse": round(rmse, 3),
            "baseline_mae": round(base_mae, 3),
            f"precision@{TOPK}": round(precision_at_k, 3),
            "test_weeks": weeks}


# ════════════════════════════════════════════════════════════════════════════
# 预计算缓存：各活跃商品未来 HORIZON_FUTURE 周预测 → cache/forecasts.parquet
# ════════════════════════════════════════════════════════════════════════════
def precompute_forecasts(horizon: int = HORIZON_FUTURE) -> pd.DataFrame:
    art = _load_artifacts()
    item_ids = list(art["scaler"]["items"].keys())
    n_weeks = art["n_weeks"]
    rows = []
    for c in item_ids:
        fc = forecast(c, horizon=horizon)
        for h in range(horizon):
            rows.append({"StockCode": c, "week_idx": n_weeks + h,
                         "pred": round(fc["pred"][h], 2),
                         "lower": round(fc["lower"][h], 2),
                         "upper": round(fc["upper"][h], 2)})
    out = pd.DataFrame(rows)
    out.to_parquet(CACHE_DIR / "forecasts.parquet", index=False)
    return out


# ════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("F3/F4 LSTM 流行性预测")
    print("=" * 60)
    train_lstm()
    print("-" * 60)
    metrics = evaluate_lstm()
    print("评估（测试集 周47-53）：")
    for k, v in metrics.items():
        print(f"  {k:18}: {v}")
    print(f"  → LSTM MAE {metrics['mae']} vs 移动平均基线 {metrics['baseline_mae']}"
          f"（{'更优 ✓' if metrics['mae'] < metrics['baseline_mae'] else '未超过基线'}）")
    json.dump(metrics, open(CACHE_DIR / "lstm_metrics.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("-" * 60)
    fc = precompute_forecasts()
    print(f"已预计算 {fc['StockCode'].nunique()} 个商品未来 {HORIZON_FUTURE} 周预测 "
          f"→ cache/forecasts.parquet（{len(fc)} 行）")
    print("[完成] LSTM 流行性预测。")


if __name__ == "__main__":
    main()
