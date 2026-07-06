"""导出 LSTM 回测预测 → cache/backtest.parquet（历史周预测 vs 实际）"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# 与 lstm_popularity.py 一致
LOOKBACK = 8
HIDDEN = 128
NUM_LAYERS = 2
TRAIN_END = 40
VAL_END = 46
TEST_END = 53
WEEK_PERIOD = 52
DEVICE = torch.device("cpu")


class LSTMPredictor(torch.nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, HIDDEN, NUM_LAYERS, batch_first=True, dropout=0.1)
        self.head = torch.nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def time_feats(week_idx: int) -> list[float]:
    ang = 2 * np.pi * (week_idx % WEEK_PERIOD) / WEEK_PERIOD
    return [float(np.sin(ang)), float(np.cos(ang))]


def load_series():
    active = json.load(open(DATA_DIR / "active_items.json", encoding="utf-8"))
    df = pd.read_parquet(DATA_DIR / "item_week_sales.parquet")
    n_weeks = int(df["week_idx"].max()) + 1
    pivot = (df[df["StockCode"].isin(active)]
             .pivot(index="StockCode", columns="week_idx", values="sales")
             .reindex(columns=range(n_weeks), fill_value=0)
             .fillna(0))
    return pivot.index.tolist(), pivot.values.astype(np.float32), pivot.to_numpy(dtype=np.float32)


def main():
    print("加载模型...")
    ckpt = torch.load(MODEL_DIR / "lstm.pt", map_location="cpu")
    scaler_data = json.load(open(MODEL_DIR / "scaler.json", encoding="utf-8"))

    model = LSTMPredictor()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print("加载数据...")
    item_ids, mat, _ = load_series()

    # 计算归一化参数
    items_scaler = scaler_data["items"]
    means = np.array([items_scaler.get(c, {}).get("mean", mat[i].mean()) for i, c in enumerate(item_ids)], dtype=np.float32)
    stds = np.array([items_scaler.get(c, {}).get("std", mat[i].std()) for i, c in enumerate(item_ids)], dtype=np.float32)
    stds[stds < 1e-6] = 1.0
    norm_mat = (np.log1p(mat.clip(min=0)) - means[:, None]) / stds[:, None]

    # 逐周逐商品预测测试集
    test_weeks = list(range(VAL_END + 1, TEST_END + 1))  # 47-53
    print(f"预测测试周: {test_weeks}")

    rows = []
    for t in test_weeks:
        feats_t = [time_feats(t - LOOKBACK + k) for k in range(LOOKBACK)]
        Xs = []
        for i in range(len(item_ids)):
            window = norm_mat[i, t - LOOKBACK: t]
            X = np.column_stack([window, np.array(feats_t)]).astype(np.float32)
            Xs.append(X)

        with torch.no_grad():
            z_pred = model(torch.tensor(np.stack(Xs))).cpu().numpy().ravel()
        pred = np.expm1(z_pred * stds + means).clip(min=0)

        for i, code in enumerate(item_ids):
            actual = float(mat[i, t])
            if actual > 0 or pred[i] > 0:
                rows.append({
                    "StockCode": code,
                    "week_idx": t,
                    "pred": round(float(pred[i]), 1),
                    "actual": round(float(actual), 1),
                })

        print(f"  周 {t}: {len(rows)} 条累积")

    df = pd.DataFrame(rows)
    out = CACHE_DIR / "backtest.parquet"
    df.to_parquet(out)

    df["abs_error"] = (df["pred"] - df["actual"]).abs()
    mae = df["abs_error"].mean()
    print(f"\n导出 {len(df)} 条 → {out}")
    print(f"覆盖 {df['StockCode'].nunique()} 商品")
    print(f"回测 MAE: {mae:.1f} 件")
    print("完成!")


if __name__ == "__main__":
    main()
