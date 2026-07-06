"""系统总览页面 — KPI 指标卡 + 周销售额趋势 + 三级畅销排行"""

from __future__ import annotations

import streamlit as st
from streamlit_echarts import st_echarts

from modules.data_loaders import (
    load_item_meta,
    load_item_week,
    load_forecasts,
    load_weekly_revenue,
    load_weekly_active_counts,
    load_weekly_active_series,
    load_weekly_orders,
    load_weekly_avp,
    load_weekly_new_users,
    load_country_revenue,
    load_category_weekly_revenue,
    load_category_weekly_orders,
    load_user_item,
    load_transactions,
    get_inventory_with_meta,
)


# ════════════════════════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════════════════════════
im = load_item_meta()
forecasts = load_forecasts()
inv = get_inventory_with_meta()
orders = load_weekly_orders()
avp = load_weekly_avp()
new_users = load_weekly_new_users()
tx_all = load_transactions()
country = load_country_revenue()
cat_rev = load_category_weekly_revenue()
cat_ord = load_category_weekly_orders()


# ════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════════════════
def _pct(curr, prev):
    """环比百分比，prev 为 0 时返回 None"""
    return (curr - prev) / prev * 100 if prev else None


def _fmt_delta(v):
    """格式化环比为 st.metric 的 delta 字符串"""
    return f"{v:+.1f}%" if v is not None else None


def _spark_floor(vals):
    """sparkline 基线偏移：减 min*0.9，让波动可见"""
    if vals is None or len(vals) == 0:
        return None
    floor = min(vals) * 0.9
    return [v - floor for v in vals]


def _kpi_card(col, label, value, delta, spark_data=None, spark_type=None, nav_page=None):
    with col:
        kwargs = {"label": label, "value": value, "delta": delta, "border": True, "delta_color": "inverse"}
        if spark_data:
            floor_data = _spark_floor(spark_data)
            if spark_type == "bar":
                floor_data = [int(v) for v in floor_data]
            kwargs["chart_data"] = floor_data
            kwargs["chart_type"] = spark_type or "area"
        st.metric(**kwargs)
        if nav_page:
            if st.button("查看详情 →", key=f"nav_{label}_{nav_page}", type="tertiary", use_container_width=True):
                st.switch_page(nav_page)


def _kpi_row(cards: list[tuple]):
    """cards: [(label, value, delta, spark_data, spark_type, nav_page?), ...]"""
    cols = st.columns(len(cards))
    for c, card in zip(cols, cards):
        _kpi_card(c, *card)


# ════════════════════════════════════════════════════════════════════════════
# 核心指标卡
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### :material/dashboard: 核心指标")
n_alert = int((inv["status"] == "警告").sum())

rev = load_weekly_revenue()
if rev is not None and len(rev) >= 2:
    last_week_rev = rev["revenue"].iloc[-1]
    rev_delta = _pct(last_week_rev, rev["revenue"].iloc[-2])
elif rev is not None and len(rev) == 1:
    last_week_rev = rev["revenue"].iloc[-1]
    rev_delta = None
else:
    last_week_rev = 0
    rev_delta = None

w_items, w_users, pw_items, pw_users = load_weekly_active_counts()
user_delta = _pct(w_users, pw_users)
item_delta = _pct(w_items, pw_items)

# ── 本周订单数 / 客单价 ──
if orders is not None and len(orders) >= 2:
    last_orders = int(orders["n_orders"].iloc[-1])
    prev_orders = int(orders["n_orders"].iloc[-2])
    orders_delta = _pct(last_orders, prev_orders)
    last_avp = last_week_rev / last_orders if last_orders > 0 else 0
    prev_avp = (rev["revenue"].iloc[-2] / prev_orders) if prev_orders > 0 and len(rev) >= 2 else 0
    avp_delta = _pct(last_avp, prev_avp)
elif orders is not None and len(orders) == 1:
    last_orders = int(orders["n_orders"].iloc[0])
    orders_delta = None
    last_avp = last_week_rev / last_orders if last_orders > 0 else 0
    avp_delta = None
else:
    last_orders = 0
    orders_delta = None
    last_avp = 0
    avp_delta = None

# ── Sparkline 数据（近 8 周趋势）──
items_wk, users_wk = load_weekly_active_series()
spark_rev = [float(v) for v in rev["revenue"].tail(16)] if rev is not None and len(rev) > 0 else None
spark_items = [int(v) for v in items_wk.tail(16)] if len(items_wk) > 0 else None
spark_users = [int(v) for v in users_wk.tail(16)] if len(users_wk) > 0 else None
spark_orders = [int(v) for v in orders["n_orders"].tail(16)] if orders is not None and len(orders) > 0 else None
spark_avp = [float(v) for v in avp["avp"].tail(16)] if avp is not None and len(avp) > 0 else None

# ── 总用户数（所有历史用户）──
ui_all = load_user_item()
total_users_all = ui_all["CustomerID"].nunique() if ui_all is not None else 0

# ── 概览卡片（纯数字，无 sparkline）──
gc1, gc2, gc3, gc4 = st.columns(4)
gc1.metric(":material/group: 总用户数", f"{total_users_all:,}")
gc2.metric(":material/inventory_2: 总商品", f"{len(im):,}")
gc3.metric(":material/receipt: 总交易记录", f"{len(tx_all):,}")
gc4.metric(":material/warning: 库存警告", f"{n_alert}")

# ── 趋势卡片（带 sparkline）──
t1, t2, t3, t4, t5 = st.columns(5)
_kpi_card(t1, ":material/group: 本周活跃用户", f"{w_users:,}", _fmt_delta(user_delta), spark_users, "bar", "pages/recommend.py")
_kpi_card(t2, ":material/trending_up: 本周活跃商品", f"{w_items:,}", _fmt_delta(item_delta), spark_items, "bar", "pages/popularity.py")
_kpi_card(t3, ":material/payments: 本周营业额", f"£{last_week_rev:,.0f}", _fmt_delta(rev_delta), spark_rev, "area", "pages/popularity.py")
_kpi_card(t4, ":material/receipt: 本周订单数", f"{last_orders:,}", _fmt_delta(orders_delta), spark_orders, "bar")
_kpi_card(t5, ":material/shopping_cart: 本周客单价", f"£{last_avp:.2f}", _fmt_delta(avp_delta), spark_avp, "area")

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# 周销售额趋势 + 环比增长
# ════════════════════════════════════════════════════════════════════════════
if rev is not None and len(rev) > 0:
    rev_full = rev.copy()
    rev_full["week_label"] = rev_full["week_start"].dt.strftime("%m/%d")
    # 对齐订单数据
    if orders is not None and len(orders) > 0:
        rev_full = rev_full.merge(orders[["week_idx", "n_orders"]], on="week_idx", how="left")
        rev_full["n_orders"] = rev_full["n_orders"].fillna(0).astype(int)

    rev_labels = rev_full["week_label"].tolist()
    rev_vals = rev_full["revenue"].tolist()
    ord_vals = rev_full["n_orders"].tolist() if "n_orders" in rev_full.columns else None

    cL, cR = st.columns([3, 2], gap="small")

    with cL:
        st.markdown("#### :material/trending_up: 营收趋势")

        rev_opts = {
            "toolbox": {
                "feature": {
                    "saveAsImage": {"title": "保存"},
                    "dataView": {"title": "数据", "readOnly": True},
                    "restore": {"title": "还原"},
                    "magicType": {"title": "切换", "type": ["line", "bar"]},
                },
                "top": 0, "right": 0,
            },
            "tooltip": {
                "trigger": "axis",
                "valueFormatter": "(function(v){return '£'+Math.round(v).toLocaleString()})",
            },
            "legend": {"bottom": 0},
            "xAxis": {"type": "category", "data": rev_labels},
            "yAxis": {"type": "value", "show": False},
            "dataZoom": [
                {"type": "inside", "start": max(0, 100 - 800 // max(len(rev_labels), 1)), "end": 100},
                {"type": "slider", "start": max(0, 100 - 800 // max(len(rev_labels), 1)), "end": 100, "height": 20, "bottom": 30},
            ],
            "grid": {"bottom": "18%", "top": "12%", "left": "4%", "right": "4%"},
            "legend": {"bottom": 0, "data": ["营收", "订单数"]},
            "series": [{
                "name": "营收",
                "type": "line",
                "data": rev_vals,
                "smooth": True,
                "lineStyle": {"color": "#6C5CE7", "width": 2.5},
                "areaStyle": {"color": "rgba(108,92,231,0.10)"},
                "symbol": "circle",
                "symbolSize": 6,
                "itemStyle": {"color": "#6C5CE7"},
            }] + ([{
                "name": "订单数",
                "type": "line",
                "yAxisIndex": 0,
                "data": ord_vals,
                "smooth": True,
                "lineStyle": {"color": "#FD79A8", "width": 2, "type": "dashed"},
                "symbol": "diamond",
                "symbolSize": 5,
                "itemStyle": {"color": "#FD79A8"},
            }] if ord_vals else []),
        }
        st_echarts(options=rev_opts, height="400px", key="weekly_revenue", theme="streamlit")

    with cR:
        st.markdown("#### :material/monitoring: 周环比增长")

        if len(rev_full) >= 3:
            mom = rev_full.copy()
            mom["growth"] = ((mom["revenue"] - mom["revenue"].shift(1))
                             / mom["revenue"].shift(1) * 100).round(1)
            mom = mom.dropna(subset=["growth"])
            mom_labels = mom["week_label"].tolist()
            mom_vals = mom["growth"].tolist()

            bar_data = []
            for v in mom_vals:
                color = "#ee6666" if v >= 0 else "#91cc75"
                bar_data.append({"value": v, "itemStyle": {"color": color}})

            mom_opts = {
                "tooltip": {"trigger": "axis", "formatter": "{b}<br/>增长率: {c}%"},
                "xAxis": {
                    "type": "category", "data": mom_labels,
                    "axisLabel": {"rotate": 45},
                },
                "yAxis": {"type": "value", "axisLabel": {"formatter": "{value}%"}},
                "grid": {"bottom": "20%", "top": "12%", "left": "12%", "right": "4%", "containLabel": True},
                "series": [{"type": "bar", "data": bar_data, "label": {"show": False}}],
            }
            st_echarts(options=mom_opts, height="400px", key="wow_growth", theme="streamlit")
        else:
            st.info("需要至少 3 周数据才能显示环比增长")

else:
    st.info("销售额数据未就绪")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# 新老用户 + 销售国家分布
# ════════════════════════════════════════════════════════════════════════════
cL, cR = st.columns(2)

with cL:
    st.markdown("#### :material/group: 新老用户占比")
    nu = new_users
    if nu is not None and len(nu) > 0:
        last = nu.iloc[-1]
        n_new = int(last["new_users"])
        n_return = int(last["returning_users"])
        n_total = int(last["total_users"])
        new_pct = n_new / n_total * 100 if n_total > 0 else 0
        user_donut = {
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "series": [{"type": "pie", "radius": ["55%", "80%"], "center": ["50%", "50%"],
                        "data": [
                            {"name": "新用户", "value": n_new, "itemStyle": {"color": "#6C5CE7"}},
                            {"name": "回头客", "value": n_return, "itemStyle": {"color": "#00B894"}},
                        ],
                        "label": {"show": False},
                        "emphasis": {"label": {"show": True, "formatter": "{b}\n{d}%"}}}],
        }
        st_echarts(options=user_donut, height="240px", key="user_donut", theme="streamlit")
        st.caption(f"新用户 {n_new} 人（{new_pct:.1f}%） · 回头客 {n_return} 人（{100 - new_pct:.1f}%） · 总计 {n_total} 人")
    else:
        st.info("用户数据未就绪")

with cR:
    st.markdown("#### :material/public: 销售国家分布")
    if country is not None and len(country) > 0:
        top5 = country.head(5)
        others_rev = country["revenue"].iloc[5:].sum() if len(country) > 5 else 0
        pie_data = []
        colors = ["#6C5CE7", "#FD79A8", "#00B894", "#FDCB6E", "#E17055", "#B2BEC3"]
        for i, (_, r) in enumerate(top5.iterrows()):
            pie_data.append({"name": r["Country"], "value": round(float(r["revenue"]), 0),
                             "itemStyle": {"color": colors[i]}})
        if others_rev > 0:
            pie_data.append({"name": "其他", "value": round(float(others_rev), 0),
                             "itemStyle": {"color": colors[5]}})
        pie_opts = {
            "tooltip": {"trigger": "item", "formatter": "{b}: £{c:,.0f} ({d}%)"},
            "series": [{"type": "pie", "radius": ["40%", "70%"], "center": ["50%", "50%"],
                        "data": pie_data, "label": {"formatter": "{b}\n{d}%"}}],
        }
        st_echarts(options=pie_opts, height="260px", key="country_pie", theme="streamlit")
    else:
        st.info("国家数据未就绪")

st.markdown("---")

# 品类营收占比 + 品类订单占比
cL, cR = st.columns(2)
with cL:
    st.markdown("#### :material/category: 品类营收占比（近 4 周）")
    if cat_rev is not None and len(cat_rev) > 0:
        latest_weeks = sorted(cat_rev["week_idx"].unique())[-4:]
        cat4 = cat_rev[cat_rev["week_idx"].isin(latest_weeks)]
        cat_total = cat4.groupby("category")["revenue"].sum().sort_values(ascending=False).head(8)
        cat_bar_data = [{"value": round(float(v), 0), "itemStyle": {"color": colors[i % 6]}}
                        for i, (cat_name, v) in enumerate(cat_total.items())]
        cat_opts = {
            "tooltip": {"trigger": "axis", "formatter": "{b}: £{c:,.0f}"},
            "xAxis": {"type": "category", "data": cat_total.index.tolist(), "axisLabel": {"rotate": 30}},
            "yAxis": {"type": "value", "show": False},
            "grid": {"top": "8%", "bottom": "20%", "left": "2%", "right": "2%"},
            "series": [{"type": "bar", "data": cat_bar_data}],
        }
        st_echarts(options=cat_opts, height="300px", key="cat_bar", theme="streamlit")
    else:
        st.info("品类数据未就绪")

with cR:
    st.markdown("#### :material/receipt_long: 品类销量占比（近 4 周）")
    if cat_ord is not None and len(cat_ord) > 0:
        latest_weeks_ord = sorted(cat_ord["week_idx"].unique())[-4:]
        cat4_ord = cat_ord[cat_ord["week_idx"].isin(latest_weeks_ord)]
        cat_sales_total = cat4_ord.groupby("category")["sales"].sum().sort_values(ascending=False).head(8)
        sales_bar_data = [{"value": int(v), "itemStyle": {"color": colors[i % 6]}}
                          for i, (cat_name, v) in enumerate(cat_sales_total.items())]
        ord_opts = {
            "tooltip": {"trigger": "axis", "formatter": "{b}: {c} 件"},
            "xAxis": {"type": "category", "data": cat_sales_total.index.tolist(), "axisLabel": {"rotate": 30}},
            "yAxis": {"type": "value", "show": False},
            "grid": {"top": "8%", "bottom": "20%", "left": "2%", "right": "2%"},
            "series": [{"type": "bar", "data": sales_bar_data}],
        }
        st_echarts(options=ord_opts, height="300px", key="cat_sales_bar", theme="streamlit")
    else:
        st.info("销量数据未就绪")

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# 畅销排行（年度 / 本月 / 本周）
# ════════════════════════════════════════════════════════════════════════════
st.markdown("#### :material/leaderboard: 畅销排行")
iw = load_item_week()
latest_week = iw["week_idx"].max()

cL, cM, cR = st.columns(3)
with cL:
    st.markdown("**:material/calendar_today: 年度畅销**")
    top8 = im.nlargest(8, "total_sales")[["StockCode", "Description_CN", "total_sales"]]
    top8["total_sales"] = top8["total_sales"].apply(lambda x: f"{x:,}")
    st.dataframe(top8.rename(columns={"StockCode": "ID", "Description_CN": "商品名", "total_sales": "销量"}),
                 use_container_width=True, height=310, hide_index=True)

with cM:
    st.markdown("**:material/calendar_month: 本月畅销**")
    month = iw[iw["week_idx"].between(latest_week - 3, latest_week)]
    month_top = month.groupby("StockCode", as_index=False)["sales"].sum()
    month_top = month_top.nlargest(8, "sales").merge(
        im[["StockCode", "Description_CN"]], on="StockCode", how="left"
    )[["StockCode", "Description_CN", "sales"]]
    month_top["sales"] = month_top["sales"].apply(lambda x: f"{x:,}")
    st.dataframe(month_top.rename(columns={"StockCode": "ID", "Description_CN": "商品名", "sales": "销量"}),
                 use_container_width=True, height=310, hide_index=True)

with cR:
    st.markdown("**:material/whatshot: 本周畅销**")
    week_top = iw[iw["week_idx"] == latest_week]
    week_top = week_top.groupby("StockCode", as_index=False)["sales"].sum()
    week_top = week_top.nlargest(8, "sales").merge(
        im[["StockCode", "Description_CN"]], on="StockCode", how="left"
    )[["StockCode", "Description_CN", "sales"]]
    week_top["sales"] = week_top["sales"].apply(lambda x: f"{x:,}")
    st.dataframe(week_top.rename(columns={"StockCode": "ID", "Description_CN": "商品名", "sales": "销量"}),
                 use_container_width=True, height=310, hide_index=True)
