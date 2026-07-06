"""流行性预测页面 — 单商品深度分析 + 双商品对比"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from modules.content_based import get_cbr
from modules.chart_utils import trend_fig
from modules.data_loaders import (
    load_item_meta,
    load_item_week,
    load_forecasts,
    load_active_items,
    load_backtest,
    load_item_sim,
    get_inventory,
)


# ════════════════════════════════════════════════════════════════════════════
# 双商品对比
# ════════════════════════════════════════════════════════════════════════════
def _compare_items(opts, im, iw, forecasts):
    ca, cb = st.columns(2)
    with ca:
        item_a = st.selectbox("商品 A", opts["label"].tolist(), key="cmp_a").split(" — ")[0]
    with cb:
        item_b = st.selectbox(
            "商品 B", opts["label"].tolist(),
            index=min(1, len(opts) - 1), key="cmp_b",
        ).split(" — ")[0]
    if item_a == item_b:
        st.warning("请选择不同商品")
        return

    clrs = {"A": "#6C5CE7", "B": "#FD79A8"}
    rows = []
    all_weeks = set()
    series = []

    for code, lbl, clr in [(item_a, "A", clrs["A"]), (item_b, "B", clrs["B"])]:
        hist = iw[iw["StockCode"] == code].sort_values("week_idx")
        fc = forecasts[forecasts["StockCode"] == code].sort_values("week_idx")
        meta = im[im["StockCode"] == code].iloc[0]
        desc = str(meta.get("Description_CN", ""))[:28]

        series.append({
            "name": f"{lbl}:{desc}",
            "type": "line",
            "data": [[int(w), int(s)] for w, s in zip(hist["week_idx"], hist["sales"])],
            "lineStyle": {"color": clr, "width": 2},
            "symbol": "none",
        })

        if len(fc) > 0:
            series.append({
                "name": f"{lbl}预测",
                "type": "line",
                "data": [[int(w), round(float(p), 1)] for w, p in zip(fc["week_idx"], fc["pred"])],
                "lineStyle": {"color": clr, "width": 2.5, "type": "dashed"},
                "symbol": "diamond",
                "symbolSize": 7,
            })

        nxt = float(fc["pred"].values[0]) if len(fc) > 0 else 0
        avg = hist["sales"].mean() if len(hist) > 0 else 1
        rows.append({
            "商品": f"{lbl}:{str(meta.get('Description', ''))[:30]}",
            "累计销量": f"{int(hist['sales'].sum()):,}",
            "预测下周": f"{nxt:.0f}件",
            "增长率": f"{(nxt / avg - 1) * 100:+.1f}%" if avg > 0 else "N/A",
            "均价": f"£{meta['avg_price']:.2f}",
        })

        all_weeks.update(hist["week_idx"].tolist())
        if len(fc) > 0:
            all_weeks.update(fc["week_idx"].tolist())

    fc_start = int(forecasts["week_idx"].min())

    compare_opts = {
        "title": {"text": "双商品趋势对比", "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0, "type": "scroll"},
        "xAxis": {
            "type": "value",
            "name": "周序号",
            "min": min(all_weeks) - 0.5,
            "max": max(all_weeks) + 0.5,
        },
        "yAxis": {"type": "value", "name": "销量(件)"},
        "dataZoom": [
            {"type": "inside", "start": 0, "end": 100},
            {"type": "slider", "start": 0, "end": 100, "height": 20, "bottom": 30},
        ],
        "grid": {"bottom": "20%", "top": "12%", "left": "8%", "right": "4%"},
        "series": series,
    }
    st_echarts(options=compare_opts, height="460px", key="compare", theme="streamlit")

    st.markdown("#### :material/monitoring: 关键指标对比")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# 单商品深度分析
# ════════════════════════════════════════════════════════════════════════════
def _single_item(opts, im, iw, forecasts):
    import streamlit.components.v1 as components
    components.html("<script>window.scrollTo(0,0)</script>", height=0)
    inv_df = get_inventory()
    latest_week = forecasts["week_idx"].max()
    top5 = forecasts[forecasts["week_idx"] == latest_week].nlargest(5, "pred")["StockCode"].tolist()

    selected = st.session_state.get("pop_sel", top5[0])

    st.markdown("##### :material/whatshot: 热门趋势商品（点击切换）")
    qcols = st.columns(len(top5))
    for i, code in enumerate(top5):
        desc = im[im["StockCode"] == code]["Description_CN"].values
        desc = desc[0][:24] if len(desc) > 0 else code
        # 本周销量与环比
        item_hist = iw[iw["StockCode"] == code].sort_values("week_idx")
        if len(item_hist) >= 2:
            cur_sales = int(item_hist["sales"].iloc[-1])
            prev_sales = int(item_hist["sales"].iloc[-2])
            delta = (cur_sales - prev_sales) / prev_sales * 100 if prev_sales > 0 else 0
            sales_line = f"周销{cur_sales}件  {'↑' if delta>=0 else '↓'}{abs(delta):.0f}%"
        elif len(item_hist) == 1:
            sales_line = f"周销{int(item_hist['sales'].iloc[-1])}件"
        else:
            sales_line = "无数据"
        # 库存信息
        inv_row = inv_df[inv_df["StockCode"] == code]
        if len(inv_row) > 0:
            stk = int(inv_row["stock"].values[0])
            stk_status = str(inv_row["status"].values[0])
            stk_color = {"充足": "#00B894", "偏低": "#FDCB6E", "警告": "#E17055"}.get(stk_status, "#999")
        else:
            stk, stk_status, stk_color = 0, "未知", "#999"
        with qcols[i]:
            st.markdown(f"""
            <div style="border-top:3px solid #A29BFE;border-radius:14px;
            box-shadow:0 2px 12px rgba(0,0,0,0.06);padding:0.9rem 1rem;
            margin-bottom:0.4rem;background:#fff;min-height:140px;">
                <div style="font-size:0.78rem;color:#888;">商品ID: <code>{code}</code></div>
                <div style="font-weight:700;font-size:0.9rem;">{desc}</div>
                <div style="font-size:0.8rem;color:#555;">{sales_line}</div>
                <div style="font-size:0.8rem;color:#888;">
                    {stk}件 · <span style="color:{stk_color};font-weight:bold">● {stk_status}</span>
                </div>
            </div>""", unsafe_allow_html=True)
            if st.button("查看", key=f"q_{code}", use_container_width=True):
                st.session_state["pop_sel"] = code
                st.session_state["_from_button"] = True
                st.rerun()

    opts_list = opts["label"].tolist()
    default_idx = next((i for i, lb in enumerate(opts_list) if lb.startswith(selected)), 0)
    sel = st.selectbox(":material/search: 或搜索选择商品", opts_list, index=default_idx, key="pop_srch")

    if st.session_state.pop("_from_button", False):
        pass
    else:
        st.session_state["pop_sel"] = sel.split(" — ")[0]

    hist = iw[iw["StockCode"] == selected].sort_values("week_idx")
    fc = forecasts[forecasts["StockCode"] == selected].sort_values("week_idx")
    meta = im[im["StockCode"] == selected].iloc[0]
    last_w = int(hist["week_idx"].max())
    cur = int(hist[hist["week_idx"] == last_w]["sales"].values[0]) if len(hist) > 0 else 0
    prev_row = hist[hist["week_idx"] == last_w - 1]
    prev = int(prev_row["sales"].values[0]) if len(prev_row) > 0 else 0
    cur_delta = f"{(cur - prev) / prev * 100:+.1f}%" if prev > 0 else None
    nxt = float(fc["pred"].values[0]) if len(fc) > 0 else 0
    trend = ":material/trending_up: 上升" if nxt > cur else (":material/trending_down: 下降" if nxt < cur else ":material/arrow_right_alt: 持平")

    # 库存周转天数
    stock_row = inv_df[inv_df["StockCode"] == selected]
    stock = int(stock_row["stock"].values[0]) if len(stock_row) > 0 else 0
    week_avg = hist["sales"].mean() if len(hist) > 0 else 0
    turnover = round(stock / week_avg * 7) if week_avg > 0 else None  # 天数

    st.header(meta["Description_CN"])

    # 品类信息
    cat = meta.get("category")
    cat_display = cat if (cat and isinstance(cat, str) and cat not in ("其他", "未分类", "")) else "—"
    cat_df = im[im["category"] == cat] if cat and cat in im["category"].values else None
    if cat_df is not None and len(cat_df) > 0:
        rank = (cat_df["total_sales"] > meta["total_sales"]).sum() + 1
        rank_str = f"#{rank}/{len(cat_df)}"
        cat_avg_price = cat_df["avg_price"].mean()
        price_diff = meta["avg_price"] - cat_avg_price
    else:
        rank_str = "—"
        cat_avg_price = 0
        price_diff = 0

    # 回测
    bt = load_backtest()
    mae_str = "—"
    if bt is not None:
        bt_item = bt[bt["StockCode"] == selected]
        if len(bt_item) > 0:
            bt_item["abs_error"] = (bt_item["pred"] - bt_item["actual"]).abs()
            mae_str = f"{bt_item['abs_error'].mean():.1f} 件"

    # ── 指标行 1 ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("商品ID", selected)
    c2.metric("品类", cat_display)
    c3.metric("总销售额排名", rank_str)
    c4.metric("累计销量", f"{int(meta['total_sales']):,} 件")

    # ── 指标行 2 ──
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("均价", f"£{meta['avg_price']:.2f}")
    d2.metric("品类均价", f"£{cat_avg_price:.2f}" if cat_avg_price > 0 else "—",
              delta=f"{price_diff:+.2f}" if price_diff != 0 else None, delta_color="inverse")
    d3.metric("当前库存", f"{stock} 件")
    d4.metric("预计可售", f"{turnover} 天" if turnover else "—")

    # ── 指标行 3 ──
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("当前周销量", f"{cur} 件", delta=cur_delta, delta_color="inverse")
    e2.metric("预测下周销量", f"{nxt:.0f} 件", delta=f"{nxt - cur:+.0f}", delta_color="inverse")
    e3.metric("趋势", trend)
    e4.metric("回测 MAE", mae_str)

    st.markdown("---")

    cL, cR = st.columns([3, 2])
    with cL:
        trend_fig(hist, fc, selected, height="400px")
    with cR:
        if len(fc) > 0:
            st.markdown("#### :material/table: 未来预测明细")
            disp = fc.copy()
            for c_ in ["pred", "lower", "upper"]:
                disp[c_] = disp[c_].round(0).astype(int)
            disp["趋势"] = disp["pred"].diff().apply(
                lambda x: "↑ 上升" if x > 0 else ("↓ 下降" if x < 0 else "→ 平稳")
            )
            st.dataframe(
                disp[["week_idx", "pred", "lower", "upper", "趋势"]].rename(
                    columns={"week_idx": "预测周", "pred": "预测销量", "lower": "下界", "upper": "上界"}
                ),
                use_container_width=True, hide_index=True,
                column_config={
                    "预测周": st.column_config.NumberColumn(width="small"),
                    "预测销量": st.column_config.NumberColumn(width="small"),
                    "下界": st.column_config.NumberColumn(width="small"),
                    "上界": st.column_config.NumberColumn(width="small"),
                    "趋势": st.column_config.TextColumn(width="small"),
                },
            )
            st.caption("以上预测区间为 80% 置信水平，即实际销量有 80% 概率落在下界与上界之间。")

    # ── 关联购买（行为相似度：买它的人也买了）──
    item_sim = load_item_sim()
    if item_sim is not None:
        sim_items = item_sim[item_sim["StockCode"] == selected].sort_values("rank").head(5)
        if len(sim_items) > 0:
            st.markdown("---")
            st.markdown("#### :material/shopping_bag: 关联购买（买它的人也买了）")
            sim_codes = sim_items["sim_StockCode"].tolist()
            sim_scores = sim_items["similarity"].tolist()
            sim_meta = im[im["StockCode"].isin(sim_codes)][["StockCode", "Description_CN", "avg_price"]]
            s_cols = st.columns(len(sim_codes))
            for idx, code in enumerate(sim_codes):
                row = sim_meta[sim_meta["StockCode"] == code]
                name = row["Description_CN"].values[0][:18] if len(row) > 0 else code
                price = f"£{row['avg_price'].values[0]:.2f}" if len(row) > 0 else "—"
                # 库存
                ir = inv_df[inv_df["StockCode"] == code]
                istk = int(ir["stock"].values[0]) if len(ir) > 0 else 0
                istatus = str(ir["status"].values[0]) if len(ir) > 0 else "未知"
                icolor = {"充足": "#00B894", "偏低": "#FDCB6E", "警告": "#E17055"}.get(istatus, "#999")
                with s_cols[idx]:
                    st.markdown(f"""
                    <div style="border-top:3px solid #55EFC4;border-radius:14px;
                    box-shadow:0 2px 12px rgba(0,0,0,0.06);padding:0.9rem 1rem;
                    margin-bottom:0.4rem;background:#fff;min-height:150px;">
                        <div style="font-weight:700;font-size:0.9rem;">{name}</div>
                        <div style="font-size:0.78rem;color:#888;">商品ID: <code>{code}</code></div>
                        <div style="font-size:0.8rem;color:#555;">行为相似度: {sim_scores[idx]:.0%}</div>
                        <div style="font-size:0.8rem;color:#555;">{price}</div>
                        <div style="font-size:0.8rem;color:#888;">
                            {istk}件 · <span style="color:{icolor};font-weight:bold">● {istatus}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    if st.button(":material/visibility: 查看", key=f"sim_cf_{code}_{idx}", use_container_width=True):
                        st.session_state["pop_sel"] = code
                        st.session_state["_from_button"] = True
                        st.rerun()

    # ── 相似商品推荐（内容感知）──
    st.markdown("---")
    st.markdown("#### :material/recommend: 相似商品推荐（基于描述文本）")
    try:
        cbr = get_cbr(im)
        sim_cards = cbr.recommend_cards(selected, im, get_inventory(), n=5)
        if sim_cards:
            scols = st.columns(min(len(sim_cards), 5))
            for j, sc in enumerate(sim_cards):
                with scols[j]:
                    s_st = sc.get("stock_status", "未知")
                    s_color = {
                        "充足": "#00B894", "偏低": "#FDCB6E",
                        "告警": "#E17055", "警告": "#E17055",
                    }.get(s_st, "#999")
                    # 库存
                    isc = inv_df[inv_df["StockCode"] == sc['StockCode']]
                    istk_sim = int(isc["stock"].values[0]) if len(isc) > 0 else 0
                    st.markdown(f"""
                    <div style="border-top:3px solid #FFEAA7;border-radius:14px;
                    box-shadow:0 2px 12px rgba(0,0,0,0.06);padding:0.9rem 1rem;
                    margin-bottom:0.4rem;background:#fff;min-height:150px;">
                        <div style="font-weight:700;font-size:0.9rem;">{str(sc['Description_CN']).strip()[:18]}</div>
                        <div style="font-size:0.78rem;color:#888;">商品ID: <code>{sc['StockCode']}</code></div>
                        <div style="font-size:0.8rem;color:#555;">匹配度: {sc['final_score']:.0%}</div>
                        <div style="font-size:0.8rem;color:#555;">
                            £{sc['price']:.2f}
                        </div>
                        <div style="font-size:0.8rem;color:#888;">
                            {istk_sim}件 · <span style="color:{s_color};font-weight:bold">● {s_st}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    if st.button(":material/visibility: 查看", key=f"sim_{sc['StockCode']}_{j}", use_container_width=True):
                        st.session_state["pop_sel"] = sc["StockCode"]
                        st.session_state["_from_button"] = True
                        st.rerun()
        else:
            st.info("该商品暂无相似推荐（可能描述文本不完整）")
    except Exception:
        st.caption("（相似推荐不可用，请先运行 content_based.py）")


# ════════════════════════════════════════════════════════════════════════════
# 页面入口
# ════════════════════════════════════════════════════════════════════════════
st.header(":material/trending_up: 流行性预测")

im = load_item_meta()
iw = load_item_week()
forecasts = load_forecasts()
active = load_active_items()

if forecasts is None:
    st.warning("预测数据未就绪，请先运行 lstm_popularity.py")
else:
    opts = im[im["StockCode"].isin(active)].copy()
    opts["label"] = opts["StockCode"] + " — " + opts["Description_CN"].str[:55]

    mode = st.radio(
        "分析模式", [":material/search: 单商品深度分析", ":material/compare: 双商品对比"],
        horizontal=True, key="pop_mode",
    )

    if mode.startswith(":material/search:"):
        _single_item(opts, im, iw, forecasts)
    else:
        _compare_items(opts, im, iw, forecasts)
