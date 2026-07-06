"""库存优化与预警（三级告警 + 补货建议）。

根据预测流行度（未来销量）估算需求，模拟库存状态，输出库存看板与预警信息。

模拟策略（v2 — 周级滚动仿真）：
  1. 以第 47 周真实销量为基准，随机模拟初始库存
  2. 第 48～53 周，每周：
       - 进货 = 近 4 周移动平均（模拟根据近期趋势补货）
       - 出库 = 当周真实销量
       - 期末库存 = 期初 + 进货 − 出库
  3. 第 53 周期末库存即为「当前库存」
  4. 安全阈值基于 LSTM 未来 4 周预测需求设定

这样偏低/告警商品的分布更贴合真实供需波动，而不是纯随机猜测。

输入：
    data/clean_transactions.parquet         消费记录（含 week_idx + 周销量）
    data/item_meta.parquet                  商品元数据
    cache/forecasts.parquet                 流行性预测（F3/F4产出，54~57周）

输出：
    data/inventory.csv                      模拟初始库存
    cache/inventory_status.parquet           库存状态/警告

运行：
    python modules/inventory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows 终端 UTF-8 输出
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

# ── 库存模拟参数 ────────────────────────────────────────────────────────────
SIM_START_WEEK = 47          # 仿真起始周
SIM_END_WEEK = 53            # 仿真终止周
INIT_STOCK_MIN = 3           # 初始库存 = week47销量 × random(3~8)
INIT_STOCK_MAX = 8
RESTOCK_LOOKBACK = 4         # 基准订货量 = 近 N 周移动平均
TREND_RATIO_CAP = 2.5        # 趋势倍率上限（防止异常值）
SAFETY_BUFFER = 1.15         # 安全缓冲倍率
SAFETY_FACTOR = 1.5          # 安全阈值倍率（LSTM 未来需求 × 此值）
SEED = 42


# ════════════════════════════════════════════════════════════════════════════
# 1. 初始化模拟库存（v2 — 周级滚动仿真）
# ════════════════════════════════════════════════════════════════════════════
def init_inventory() -> None:
    """从第 47 周开始逐周模拟进货/出库到第 53 周，生成合理的当前库存。

    流程：
      1. 读真实交易数据，计算每商品每周销量
      2. 第 47 周初：库存 = week47 销量 × random(3~8)
      3. 第 48~53 周每周期初：
          进货 = 近 4 周移动平均销量（模拟按趋势补货）
          出库 = 当周真实销量
          期末库存 = max(0, 期初 + 进货 − 出库)
      4. 第 53 周末库存 → 写入 data/inventory.csv + MySQL
    """
    rng = np.random.default_rng(SEED)

    # ── 读取真实交易，计算每周每商品销量 ──
    tx = pd.read_parquet(DATA_DIR / "clean_transactions.parquet")
    tx = tx[tx["week_idx"].between(0, SIM_END_WEEK)]  # 只需要 0~53 周
    weekly_sales = (
        tx.groupby(["StockCode", "week_idx"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"Quantity": "sales"})
    )

    # ── 读商品列表 ──
    item_meta = pd.read_parquet(DATA_DIR / "item_meta.parquet")
    all_items = item_meta[["StockCode"]].copy()

    # 交叉所有商品 × 周 47~53（确保每商品在每周都有一行）
    weeks = list(range(SIM_START_WEEK, SIM_END_WEEK + 1))
    grid = all_items.merge(pd.DataFrame({"week_idx": weeks}), how="cross")
    grid = grid.merge(weekly_sales, on=["StockCode", "week_idx"], how="left")
    grid["sales"] = grid["sales"].fillna(0).astype(int)

    # ── 生成初始库存（第 47 周）──
    # 初始库存 = week47 真实销量 × random(3~8)，最少 10 件防止零库存
    wk47 = grid[grid["week_idx"] == SIM_START_WEEK].copy()
    multipliers = rng.uniform(INIT_STOCK_MIN, INIT_STOCK_MAX, len(wk47))
    wk47["inventory_start"] = (wk47["sales"] * multipliers).round(0).astype(int)
    wk47["inventory_start"] = wk47["inventory_start"].clip(lower=10)
    wk47["inventory_end"] = wk47["inventory_start"]  # placeholder

    sim_rows = [wk47]

    # ── 逐周模拟 48 → 53 ──
    prev = wk47[["StockCode", "inventory_end"]].rename(
        columns={"inventory_end": "inventory_start"}
    )

    for w in range(SIM_START_WEEK + 1, SIM_END_WEEK + 1):
        cur = grid[grid["week_idx"] == w].copy()

        # 期初库存 = 上周期末库存
        cur = cur.merge(prev, on="StockCode", how="left")
        cur["inventory_start"] = cur["inventory_start"].fillna(10).astype(int)

        # ── 进货量 = 趋势感知策略 ────────────────
        # 步骤：
        #   ① 取近 4 周分为「近2周」和「前2周」
        #   ② 趋势倍率 = 近2周均值 / 前2周均值（裁剪到 [0.7, 2.5]）
        #   ③ 增长因子 = max(1.0, 趋势倍率)  ← 上涨时放大，平稳/下降时不缩小
        #   ④ 订货量 = 4周均值 × 增长因子 × 1.15 安全缓冲
        lb = list(range(max(0, w - RESTOCK_LOOKBACK), w))
        ws = weekly_sales[weekly_sales["week_idx"].isin(lb)]

        # 4 周均值（基准订货量）
        avg4 = ws.groupby("StockCode")["sales"].mean().reset_index()
        avg4.columns = ["StockCode", "avg_4w"]

        # 近 2 周均值
        near_ws = lb[-2:]
        near = ws[ws["week_idx"].isin(near_ws)]
        near_avg = near.groupby("StockCode")["sales"].mean().reset_index()
        near_avg.columns = ["StockCode", "avg_near"]

        # 前 2 周均值
        far_ws = lb[:2] if len(lb) >= 4 else lb[:1]
        far = ws[ws["week_idx"].isin(far_ws)]
        far_avg = far.groupby("StockCode")["sales"].mean().reset_index()
        far_avg.columns = ["StockCode", "avg_far"]

        # 合并计算趋势
        rd = avg4.merge(near_avg, on="StockCode", how="left")
        rd = rd.merge(far_avg, on="StockCode", how="left")
        rd["avg_near"] = rd["avg_near"].fillna(rd["avg_4w"])
        rd["avg_far"] = rd["avg_far"].fillna(rd["avg_4w"])

        # 趋势倍率 & 增长因子
        rd["trend_ratio"] = (
            rd["avg_near"] / rd["avg_far"].replace(0, 1)
        ).clip(0.7, TREND_RATIO_CAP)
        rd["growth_factor"] = rd["trend_ratio"].clip(lower=1.0)

        # 最终订货量
        rd["restock"] = (
            rd["avg_4w"] * rd["growth_factor"] * SAFETY_BUFFER
        ).round(0).astype(int)

        cur = cur.merge(rd[["StockCode", "restock"]], on="StockCode", how="left")
        cur["restock"] = cur["restock"].fillna(cur["sales"]).round(0).astype(int)

        # 期末库存 = 期初 + 进货 − 出库
        cur["inventory_end"] = (
            cur["inventory_start"] + cur["restock"] - cur["sales"]
        ).clip(lower=0)

        sim_rows.append(cur)

        # 本周期末 → 下周期初
        prev = cur[["StockCode", "inventory_end"]].rename(
            columns={"inventory_end": "inventory_start"}
        )

    # ── 汇总：取第 53 周期末库存 + 计算阈值 ──
    sim_all = pd.concat(sim_rows, ignore_index=True)
    final = sim_all[sim_all["week_idx"] == SIM_END_WEEK].copy()

    # 阈值 = 第 53 周近 4 周均值 × SAFETY_FACTOR（关联未来预测的安全库存）
    last4_weeks = list(range(SIM_END_WEEK - 3, SIM_END_WEEK + 1))
    l4 = weekly_sales[weekly_sales["week_idx"].isin(last4_weeks)]
    l4_avg = (
        l4.groupby("StockCode")["sales"]
        .mean()
        .reset_index()
        .rename(columns={"sales": "avg_4w"})
    )
    final = final.merge(l4_avg, on="StockCode", how="left")
    final["avg_4w"] = final["avg_4w"].fillna(0)
    # 如果该商品实际有 LSTM 预测，阈值会由 update_inventory_status 覆盖
    final["threshold"] = (final["avg_4w"] * SAFETY_FACTOR * 3).round(0).astype(int)
    # 零销量/休眠商品阈值=0，不参与偏低判定
    has_sales = final["avg_4w"] > 0
    final["threshold"] = final["threshold"].where(has_sales, 0)

    out = pd.DataFrame({
        "StockCode": final["StockCode"],
        "stock": final["inventory_end"].astype(int),
        "threshold": final["threshold"].astype(int),
    })

    # 确保所有 item_meta 中的商品都在输出中
    out = all_items.merge(out, on="StockCode", how="left")
    out["stock"] = out["stock"].fillna(0).astype(int)
    out["threshold"] = out["threshold"].fillna(0).astype(int)

    out.to_csv(DATA_DIR / "inventory.csv", index=False, encoding="utf-8")
    if str(PROJ_DIR) not in sys.path:
        sys.path.insert(0, str(PROJ_DIR))
    from modules import db
    db.save_inventory(out)

    n_zero = (out["stock"] == 0).sum()
    mean_stock = out["stock"].mean()
    print(f"[F6] 库存仿真完成 (v2 周级滚动): {len(out)} 商品 → "
          f"MySQL inventory 表 + data/inventory.csv")
    print(f"  仿真区间: 第 {SIM_START_WEEK}~{SIM_END_WEEK} 周")
    print(f"  库存范围: {out['stock'].min()} ~ {out['stock'].max()}, "
          f"均值 {mean_stock:.0f}, 零库存 {n_zero} 个")
    print(f"  阈值范围: {out['threshold'].min()} ~ {out['threshold'].max()}")


# ════════════════════════════════════════════════════════════════════════════
# 2. 更新库存状态（结合 LSTM 未来预测）
# ════════════════════════════════════════════════════════════════════════════
def update_inventory_status() -> pd.DataFrame:
    """结合 cache/forecasts.parquet（第 54~57 周 LSTM 预测）判定库存状态。

    Returns:
        DataFrame 包含 StockCode, stock, predicted_demand, threshold, status
    """
    if str(PROJ_DIR) not in sys.path:
        sys.path.insert(0, str(PROJ_DIR))
    from modules import db
    fc_path = CACHE_DIR / "forecasts.parquet"

    inv = db.load_inventory()
    if inv is None or len(inv) == 0:
        print("[F6] 库存为空，先执行 init_inventory() ...")
        init_inventory()
        inv = db.load_inventory()

    if not fc_path.exists():
        print("[F6] 预测文件不存在，使用历史均值估算需求。")
        out = inv.copy()
        out["predicted_demand"] = (out["threshold"] / (SAFETY_FACTOR * 3)).round(0).astype(int)
        out["status"] = "充足"
        out.loc[out["stock"] < out["threshold"], "status"] = "偏低"
        out.loc[out["stock"] < out["predicted_demand"], "status"] = "警告"
        out.to_parquet(CACHE_DIR / "inventory_status.parquet", index=False)
        return out

    # ── 有 LSTM 预测：用未来 4 周（54~57）预测总和作为需求 ──
    forecasts = pd.read_parquet(fc_path)

    # 每个商品未来 4 周总预测销量
    demand = (forecasts.groupby("StockCode")["pred"]
              .sum().round(0).astype(int)
              .reset_index()
              .rename(columns={"pred": "predicted_demand"}))

    out = inv.merge(demand, on="StockCode", how="left")
    out["predicted_demand"] = out["predicted_demand"].fillna(0).astype(int)

    # ── 覆盖阈值：优先用 LSTM 未来需求 × SAFETY_FACTOR ──
    has_pred = out["predicted_demand"] > 0
    out.loc[has_pred, "threshold"] = (
        out.loc[has_pred, "predicted_demand"] * SAFETY_FACTOR
    ).round(0).astype(int)
    # 无预测的商品保留 init_inventory 中基于历史均值算出的阈值

    # ── 状态判定：充足 > 偏低 > 警告（警告优先级最高）──
    out["status"] = "充足"
    out.loc[out["stock"] < out["threshold"], "status"] = "偏低"
    out.loc[out["stock"] < out["predicted_demand"], "status"] = "警告"

    # 修正：无预测需求且零库存的商品 = 休眠品，不判偏低
    dormant = (out["predicted_demand"] <= 0) & (out["stock"] <= 0)
    out.loc[dormant, "status"] = "充足"

    out.to_parquet(CACHE_DIR / "inventory_status.parquet", index=False)

    n_ok = (out["status"] == "充足").sum()
    n_low = (out["status"] == "偏低").sum()
    n_alert = (out["status"] == "警告").sum()
    n_has_pred = has_pred.sum()

    print(f"[F6] 库存状态更新完成: {len(out)} 商品 → cache/inventory_status.parquet")
    print(f"  有 LSTM 预测: {n_has_pred} | "
          f"充足: {n_ok} | 偏低: {n_low} | 警告: {n_alert}")
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. 导出报告
# ════════════════════════════════════════════════════════════════════════════
def export_report(out_path: str | Path) -> None:
    """汇总趋势+推荐+库存，导出 Markdown 报告。

    Args:
        out_path: 输出文件路径（.md）
    """
    out_path = Path(out_path)

    # 收集数据
    forecasts = pd.read_parquet(CACHE_DIR / "forecasts.parquet")
    inv_status = pd.read_parquet(CACHE_DIR / "inventory_status.parquet")
    item_meta = pd.read_parquet(DATA_DIR / "item_meta.parquet")

    # 合并商品名
    inv_status = inv_status.merge(item_meta[["StockCode", "Description"]], on="StockCode", how="left")

    # 趋势摘要：Top-10 预测销量最高的商品
    latest_week = forecasts["week_idx"].max()
    top_trending = (forecasts[forecasts["week_idx"] == latest_week]
                    .nlargest(10, "pred")
                    .merge(item_meta[["StockCode", "Description"]], on="StockCode", how="left"))

    # 警告摘要
    alerts = inv_status[inv_status["status"] == "警告"].sort_values("predicted_demand", ascending=False)
    low_stock = inv_status[inv_status["status"] == "偏低"].sort_values("predicted_demand", ascending=False)

    # 生成 Markdown
    from datetime import datetime

    lines = []
    lines.append("# 商品流行性预测系统 — 分析报告")
    lines.append(f"")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**数据范围**: Online Retail II (2010-12 ~ 2011-12)")
    lines.append(f"")
    lines.append("---")
    lines.append("")

    # 1. 流行趋势
    lines.append("## 1. 流行趋势分析")
    lines.append("")
    lines.append("### 预测热度 Top-10 商品")
    lines.append("")
    lines.append("| 排名 | StockCode | 商品名称 | 预测销量（第{}周） | 置信下界 | 置信上界 |".format(latest_week))
    lines.append("|------|-----------|----------|-------------------|----------|----------|")
    for i, (_, row) in enumerate(top_trending.iterrows(), 1):
        desc = str(row.get("Description", ""))[:40]
        lines.append(f"| {i} | {row['StockCode']} | {desc} | {row['pred']:.0f} | {row['lower']:.0f} | {row['upper']:.0f} |")
    lines.append("")

    # 2. 库存状况
    lines.append("## 2. 库存状况分析")
    lines.append("")
    n_ok = (inv_status["status"] == "充足").sum()
    n_low = (inv_status["status"] == "偏低").sum()
    n_alert = (inv_status["status"] == "警告").sum()
    lines.append(f"- 库存充足: **{n_ok}** 个商品")
    lines.append(f"- 库存偏低: **{n_low}** 个商品")
    lines.append(f"- 库存警告: **{n_alert}** 个商品")
    lines.append("")

    if len(alerts) > 0:
        lines.append("### ⚠️ 警告商品（库存不足）")
        lines.append("")
        lines.append("| StockCode | 商品名称 | 当前库存 | 预测需求 | 安全阈值 |")
        lines.append("|-----------|----------|----------|----------|----------|")
        for _, row in alerts.head(20).iterrows():
            desc = str(row.get("Description", ""))[:35]
            lines.append(f"| {row['StockCode']} | {desc} | {row['stock']} | {row['predicted_demand']} | {row['threshold']} |")
        lines.append("")

    if len(low_stock) > 0:
        lines.append("### ⚡ 偏低商品（建议关注）")
        lines.append("")
        lines.append("| StockCode | 商品名称 | 当前库存 | 预测需求 | 安全阈值 |")
        lines.append("|-----------|----------|----------|----------|----------|")
        for _, row in low_stock.head(10).iterrows():
            desc = str(row.get("Description", ""))[:35]
            lines.append(f"| {row['StockCode']} | {desc} | {row['stock']} | {row['predicted_demand']} | {row['threshold']} |")
        lines.append("")

    # 3. 建议
    lines.append("## 3. 库存优化建议")
    lines.append("")
    if n_alert > 0:
        lines.append(f"- ⚠️ **紧急**: {n_alert} 个商品库存低于预测需求，建议**立即补货**")
    if n_low > 0:
        lines.append(f"- ⚡ **关注**: {n_low} 个商品库存偏低，建议**纳入补货计划**")
    if n_ok > 0:
        lines.append(f"- ✅ **正常**: {n_ok} 个商品库存充足")
    lines.append("")
    lines.append("---")
    lines.append("*本报告由商品流行性预测系统自动生成*")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[F6] 报告已导出至: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("═" * 56)
    print(" F6 库存优化与预警模块 (v2 周级滚动仿真)")
    print("═" * 56)

    # 1. 周级滚动仿真库存（第 47→53 周）
    init_inventory()

    # 2. 更新库存状态
    status_df = update_inventory_status()

    # 3. 摘要
    alerts = status_df[status_df["status"] == "警告"]
    print(f"\n─" * 56)
    if len(alerts) > 0:
        print(f"⚠️  警告商品 ({len(alerts)} 个):")
        item_meta = pd.read_parquet(DATA_DIR / "item_meta.parquet")
        alerts = alerts.merge(item_meta[["StockCode", "Description"]], on="StockCode", how="left")
        for _, row in alerts.head(10).iterrows():
            desc = str(row.get("Description", ""))[:40]
            print(f"  {row['StockCode']:8s} | {desc:40s} | 库存:{row['stock']:5d} | 需求:{row['predicted_demand']:5d}")
    else:
        print("所有商品库存状态正常。")

    print("\n[完成] F6 库存优化与预警模块运行完成")
    print("═" * 56)


if __name__ == "__main__":
    main()
