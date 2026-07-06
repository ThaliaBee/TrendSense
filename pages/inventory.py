"""库存预警页面 — 库存状态分布 + 需求预测 + 智能补货建议"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from modules.chart_utils import trend_fig
from modules.data_loaders import (
    load_item_meta,
    load_item_week,
    load_forecasts,
    get_inventory_full,
)


# ════════════════════════════════════════════════════════════════════════════
# 库存表格 + 选中商品趋势
# ════════════════════════════════════════════════════════════════════════════
def _inv_table(df, im, iw, forecasts, key_prefix):
    disp = df[["StockCode", "Description_CN", "stock", "predicted_demand", "threshold", "status"]].copy()
    disp["缺口"] = disp["stock"] - disp["predicted_demand"]
    disp = disp.sort_values("缺口")

    evt = st.dataframe(
        disp.rename(columns={
            "StockCode": "ID", "Description_CN": "商品名", "stock": "库存",
            "predicted_demand": "需求", "threshold": "阈值", "status": "状态",
        }),
        use_container_width=True, height=350, hide_index=True,
        selection_mode="single-row", on_select="rerun", key=f"inv_{key_prefix}",
    )

    if evt is not None and len(evt.get("selection", {}).get("rows", [])) > 0:
        row_idx = evt["selection"]["rows"][0]
        code = disp.iloc[row_idx]["StockCode"]
        desc = disp.iloc[row_idx]["Description_CN"]
        hist = iw[iw["StockCode"] == code].sort_values("week_idx")
        fc = (
            forecasts[forecasts["StockCode"] == code].sort_values("week_idx")
            if forecasts is not None else pd.DataFrame()
        )

        st.markdown("---")
        st.markdown(f"#### :material/trending_up: {code} — {desc}")
        if len(hist) > 0:
            trend_fig(hist, fc, code)
        else:
            st.info("无历史数据")


# ════════════════════════════════════════════════════════════════════════════
# 页面入口
# ════════════════════════════════════════════════════════════════════════════
st.header(":material/inventory_2: 库存预警")

im = load_item_meta()
iw = load_item_week()
forecasts = load_forecasts()
inv = get_inventory_full()

n_ok = int((inv["status"] == "充足").sum())
n_low = int((inv["status"] == "偏低").sum())
n_alert = int((inv["status"] == "警告").sum())

if n_alert > 0:
    st.error(f":material/emergency: {n_alert} 个商品库存警告，需立即处理！")
elif n_low > 0:
    st.warning(f":material/bolt: {n_low} 个商品库存偏低")

c1, c2, c3, c4 = st.columns(4)
c1.metric(":material/check_circle: 充足", n_ok)
c2.metric(":material/bolt: 偏低", n_low)
c3.metric(":material/warning: 警告", n_alert)
c4.metric(":material/inventory_2: 总计", f"{len(inv):,}")

st.markdown("---")

# ── 图表区域 ──
cL, cR = st.columns([1, 2])
cmap = {"充足": "#00B894", "偏低": "#FDCB6E", "警告": "#E17055"}

with cL:
    sc = inv["status"].value_counts()
    status_order = ["充足", "偏低", "警告"]
    bar_data = [int(sc.get(s, 0)) for s in status_order]

    bar_opts = {
        "title": {"text": "库存状态分布", "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "category",
            "data": status_order,
            "name": "状态",
        },
        "yAxis": {"type": "value", "name": "商品数"},
        "grid": {"top": "16%", "bottom": "10%", "left": "10%", "right": "4%"},
        "series": [{
            "type": "bar",
            "data": [
                {"value": bar_data[0], "itemStyle": {"color": cmap["充足"]}},
                {"value": bar_data[1], "itemStyle": {"color": cmap["偏低"]}},
                {"value": bar_data[2], "itemStyle": {"color": cmap["警告"]}},
            ],
            "label": {"show": True, "position": "top"},
        }],
    }
    st_echarts(options=bar_opts, height="350px", key="inv_bar", theme="streamlit")

with cR:
    samp = inv.sample(min(500, len(inv)))

    # 按状态分组
    series_by_status = {}
    for _, r in samp.iterrows():
        s = r["status"]
        series_by_status.setdefault(s, []).append([float(r["stock"]), float(r["predicted_demand"])])

    scatter_series = []
    for status in ["充足", "偏低", "警告"]:
        if status in series_by_status:
            scatter_series.append({
                "name": status,
                "type": "scatter",
                "data": series_by_status[status],
                "itemStyle": {"color": cmap.get(status, "#999"), "opacity": 0.6},
                "symbolSize": 6,
            })

    s_max = max(samp["stock"].max(), samp["predicted_demand"].max()) if len(samp) > 0 else 1
    mx = max(int(s_max), 1)

    scatter_opts = {
        "title": {"text": "库存 vs 预测需求", "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {
            "trigger": "item",
            "formatter": "{a}<br/>库存: {c0}<br/>需求: {c1}",
        },
        "legend": {"bottom": 0, "data": ["充足", "偏低", "警告"]},
        "xAxis": {"type": "value", "name": "当前库存", "max": mx},
        "yAxis": {"type": "value", "name": "预测需求", "max": mx},
        "grid": {"bottom": "12%", "top": "16%", "left": "10%", "right": "4%"},
        "series": scatter_series + [{
            "name": "库存=需求",
            "type": "line",
            "data": [[0, 0], [mx, mx]],
            "lineStyle": {"color": "#999", "type": "dashed", "width": 1},
            "symbol": "none",
            "silent": True,
        }],
    }
    st_echarts(options=scatter_opts, height="350px", key="inv_scatter", theme="streamlit")

st.markdown("---")

tab_a, tab_l, tab_all = st.tabs([":material/warning: 警告商品", ":material/bolt: 偏低商品", ":material/description: 全部库存"])

with tab_a:
    df = inv[inv["status"] == "警告"]
    if len(df) == 0:
        st.success("无警告商品")
    else:
        _inv_table(df, im, iw, forecasts, "alert")

with tab_l:
    df = inv[inv["status"] == "偏低"]
    if len(df) == 0:
        st.success("无偏低商品")
    else:
        _inv_table(df, im, iw, forecasts, "low")

with tab_all:
    st.dataframe(
        inv[["StockCode", "Description_CN", "stock", "predicted_demand", "threshold", "status"]]
        .rename(columns={
            "StockCode": "ID", "Description_CN": "商品名", "stock": "库存",
            "predicted_demand": "需求", "threshold": "阈值", "status": "状态",
        }),
        use_container_width=True, height=420, hide_index=True,
    )

# 补货建议
if n_alert > 0 or n_low > 0:
    st.markdown("---")
    st.markdown("### :material/lightbulb: 智能补货建议")
    need = inv[inv["status"].isin(["警告", "偏低"])].copy()
    need["建议补货量"] = (
        (need["predicted_demand"] * 2 - need["stock"]).clip(lower=0).round(0).astype(int)
    )
    need["紧急度"] = need["status"].map({"警告": "🔴紧急", "偏低": "🟡关注"})
    need = need.sort_values("建议补货量", ascending=False)
    st.caption(f"需补货 **{len(need)}** 个 · 建议总量约 **{need['建议补货量'].sum():,}** 件")
    st.dataframe(
        need[["StockCode", "Description_CN", "stock", "predicted_demand", "建议补货量", "紧急度"]]
        .rename(columns={
            "StockCode": "ID", "Description_CN": "商品名", "stock": "库存", "predicted_demand": "预测下周需求",
        })
        .head(20),
        use_container_width=True, height=380, hide_index=True,
    )
