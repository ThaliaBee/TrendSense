"""个性化推荐页面 — 用户级协同过滤 + 内容感知推荐"""

from __future__ import annotations

import html as _html
import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from modules.data_loaders import (
    load_item_meta,
    load_user_item,
    load_forecasts,
    load_cf_topn,
    load_transactions,
    get_inventory_with_meta,
)
from modules.session_utils import log_user_action

# ════════════════════════════════════════════════════════════════════════════
# 推荐引擎加载
# ════════════════════════════════════════════════════════════════════════════
_PROJ_DIR = Path(__file__).resolve().parent.parent


def _load_recommend():
    path = str(_PROJ_DIR / "modules" / "recommend.py")
    try:
        spec = importlib.util.spec_from_file_location("recommend_live", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.recommend
    except Exception:
        return None


get_recs = _load_recommend()
_REC_ENGINE = "v2" if get_recs is not None else "fallback"


# ════════════════════════════════════════════════════════════════════════════
# 回退推荐
# ════════════════════════════════════════════════════════════════════════════
def _fallback_rec(uid, n, cf_topn, im, inv, user_item=None, strategy="balanced"):
    df = cf_topn[cf_topn["CustomerID"] == uid].copy()
    if len(df) == 0:
        return []
    if user_item is not None and len(user_item) > 0:
        purchased = set(user_item[user_item["CustomerID"] == uid]["StockCode"].tolist())
        df = df[~df["StockCode"].isin(purchased)]
    if len(df) == 0:
        return []
    rng = np.random.default_rng(abs(hash(uid)) % (2 ** 31))
    if strategy == "cf_focused":
        elig_pct = 70
    elif strategy == "trending":
        elig_pct = 20
    else:
        elig_pct = 40
    mask = df["StockCode"].apply(lambda x: (hash(f"{x}_{uid}") % 100) < elig_pct)
    if mask.sum() >= n:
        df = df[mask]
    else:
        df = df.iloc[rng.permutation(len(df))[:n]]
    df = df.iloc[rng.permutation(len(df))]
    df = df.head(n * 2)
    df = df.merge(im[["StockCode", "Description_CN", "avg_price"]], on="StockCode", how="left")
    df = df.merge(inv[["StockCode", "stock", "status"]], on="StockCode", how="left")
    mn, mx = df["cf_score"].min(), df["cf_score"].max()
    df["final_score"] = (df["cf_score"] - mn) / (mx - mn + 1e-9) if mx > mn else 0.5
    df["perturb"] = df["StockCode"].apply(lambda x: (hash(f"{uid}_{x}") % 1000) / 10000)
    df["final_score"] = (df["final_score"] + df["perturb"]).clip(0, 0.99)
    df = df.sort_values("final_score", ascending=False).head(n)
    return [
        {
            "StockCode": r["StockCode"],
            "Description_CN": str(r.get("Description_CN", "?")),
            "final_score": round(float(r["final_score"]), 4),
            "cf_score": round(float(r["cf_score"]), 4),
            "popularity_score": 0.5,
            "reason": "基于您的购买偏好推荐",
            "price": round(
                float(r.get("avg_price", 0) if pd.notna(r.get("avg_price")) else 0), 2
            ),
            "stock": int(r.get("stock", 0) if pd.notna(r.get("stock")) else 0),
            "stock_status": str(r.get("status", "?")),
        }
        for _, r in df.iterrows()
    ]


# ════════════════════════════════════════════════════════════════════════════
# 页面入口
# ════════════════════════════════════════════════════════════════════════════
# 跨页面导航
nav_sel = st.query_params.get("sel")
if nav_sel:
    st.session_state["pop_sel"] = nav_sel
    st.query_params.pop("sel")
    st.switch_page("pages/popularity.py")

st.header(":material/track_changes: 个性化推荐")
import streamlit.components.v1 as comps
comps.html("<script>window.scrollTo(0,0)</script>", height=0)

cf_topn = load_cf_topn()
forecasts = load_forecasts()
im = load_item_meta()
ui = load_user_item()
inv = get_inventory_with_meta()

if cf_topn is None:
    st.warning("推荐数据未就绪")
else:
    all_users = sorted(cf_topn["CustomerID"].unique().tolist())

    st.markdown("### :material/settings: 推荐参数")
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        uid = st.number_input(
            ":material/person: 用户 ID", min_value=0, max_value=999999,
            value=st.session_state.get("rec_uid", all_users[0]), step=1,
            help="输入任意用户 ID，系统自动判断新/老用户并切换推荐策略",
        )
        st.session_state["rec_uid"] = uid
    with c2:
        n = st.slider("推荐个数", 5, 20, 10, key="rec_n")
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(":material/casino: 随机", use_container_width=True):
            st.session_state["rec_uid"] = random.choice(all_users)
            st.rerun()

    # ── 用户画像 + 品类偏好 ──
    u_data = ui[ui["CustomerID"] == uid]
    if len(u_data) > 0:
        st.markdown("#### :material/person: 用户基本信息")
        total_buys = int(u_data["n_purchase"].sum())
        total_items = int(u_data["total_qty"].sum())
        # 首次购买日期
        tx = load_transactions()
        user_tx = tx[tx["CustomerID"] == uid]
        first_date = user_tx["InvoiceDate"].min() if len(user_tx) > 0 else None
        first_str = first_date.strftime("%Y-%m-%d") if first_date is not None else "—"
        distinct_items = len(u_data)
        u_with_price = u_data.merge(im[["StockCode", "avg_price", "category"]], on="StockCode", how="left")
        total_spend = (u_with_price["total_qty"] * u_with_price["avg_price"]).sum()
        avg_spend = total_spend / total_buys if total_buys > 0 else 0
        u_cats = u_with_price.groupby("category")["n_purchase"].sum().sort_values(ascending=False).head(5)
        up0, up1, up2 = st.columns(3)
        up0.metric("用户ID", f"{uid}")
        up1.metric("购买次数", f"{total_buys}")
        up2.metric("累计件数", f"{total_items}")

        up3, up4, up5, up6 = st.columns(4)
        up3.metric("总消费额", f"£{total_spend:,.0f}")
        up4.metric("每次均消", f"£{avg_spend:.2f}")
        up5.metric("涉及品类", f"{distinct_items}种")
        up6.metric("首次购买", first_str)
        cL, cR = st.columns(2)
        with cL:
            st.markdown("#### :material/pie_chart: 偏好品类")
            cat_df = u_cats.reset_index()
            cat_df.columns = ["品类", "购买次数"]
            cat_df["购买次数"] = cat_df["购买次数"].astype(int)
            st.dataframe(cat_df, use_container_width=True, hide_index=True)
        with cR:
            st.markdown("#### :material/history: 最常购买 TOP5")
            uh = ui[ui["CustomerID"] == uid].nlargest(5, "n_purchase")
            uh = uh.merge(im[["StockCode", "Description_CN"]], on="StockCode", how="left")
            uh_df = uh[["Description_CN", "StockCode", "n_purchase"]].copy()
            uh_df.columns = ["商品名", "商品ID", "购买次数"]
            uh_df["购买次数"] = uh_df["购买次数"].astype(int)
            st.dataframe(uh_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(":material/person: :material/ac_unit: 新用户（冷启动）")

    # 推荐
    fc_clean = (
        forecasts[forecasts["week_idx"] == forecasts["week_idx"].min()]
        if forecasts is not None else None
    )

    if len(u_data) == 0:
        st.info(":material/ac_unit: **冷启动模式** — 该用户无购买记录，以下为热门推荐（不同用户看到的商品不同）")

    log_user_action(f"查看用户{uid}推荐")

    strategies = [
        ("balanced", ":material/balance: 平衡推荐", "#A29BFE"),
        ("cf_focused", ":material/hub: 偏重相似偏好", "#74B9FF"),
        ("trending", ":material/whatshot: 偏重热门趋势", "#FD79A8"),
    ]

    @st.cache_data(ttl=300, show_spinner=False)
    def _cached_recs(_uid, _n, _skey):
        try:
            if get_recs is not None:
                return get_recs(user_id=_uid, n=_n, strategy=_skey,
                                cf_topn=cf_topn, forecasts=fc_clean,
                                item_meta=im, inventory=inv, user_item=ui)
        except Exception:
            pass
        return _fallback_rec(_uid, _n, cf_topn, im, inv, ui, _skey)

    all_cards = {}
    for skey, _, _ in strategies:
        all_cards[skey] = _cached_recs(uid, n, skey)

    for skey, slabel, color in strategies:
        cards = all_cards.get(skey, [])
        st.markdown(f"### {slabel} ({len(cards)}件)")
        if not cards:
            st.caption("暂无推荐")
            continue

        COLS_PER_ROW = 5
        for row_start in range(0, len(cards[:n]), COLS_PER_ROW):
            row_cards = cards[row_start:row_start + COLS_PER_ROW]
            row_cols = st.columns(COLS_PER_ROW)
            for j, card in enumerate(row_cards):
                status = card.get("stock_status", "充足")
                scolor = {"充足": "#00B894", "偏低": "#FDCB6E", "警告": "#E17055"}.get(status, "#999")
                desc = _html.escape(str(card.get("Description_CN", "?")))[:20]
                c_code = _html.escape(str(card.get("StockCode", "")))
                score = card.get("final_score", 0)
                cf = card.get("cf_score", 0)
                pop = card.get("popularity_score", 0)
                price = card.get("price", 0)
                stock = card.get("stock", 0)

                reason_parts = "".join(
                    f'<div style="font-size:0.82rem;color:#888;">• {_html.escape(p)}</div>'
                    for p in card.get("reason", "").split(" · ")
                )
                card_html = f"""
                <div style="
                    border-top:3px solid {color};
                    border-radius:14px;
                    box-shadow:0 2px 12px rgba(0,0,0,0.06);
                    padding:1rem 1.1rem;
                    margin-bottom:0.4rem;
                    background:#fff;min-height:160px;
                ">
                    <div style="font-weight:700;font-size:0.95rem;margin-bottom:0.3rem;">{str(desc).strip()}</div>
                    <div style="font-size:0.78rem;color:#888;margin-bottom:0.3rem;">商品ID: <code>{c_code}</code></div>
                    {reason_parts}
                    <div style="font-size:0.82rem;color:#555;margin-top:0.3rem;">匹配度 {score:.0%} · £{price:.2f}</div>
                    <div style="font-size:0.8rem;color:#888;">
                        CF {cf:.0%} · 热度 {pop:.0%}
                    </div>
                    <div style="font-size:0.8rem;color:#888;">
                        {stock}件 · <span style="color:{scolor};font-weight:bold">● {status}</span>
                    </div>
                </div>"""
                with row_cols[j]:
                    st.markdown(card_html, unsafe_allow_html=True)
                    if st.button("查看", key=f"rec_{skey}_{c_code}_{row_start}_{j}", use_container_width=True):
                        st.query_params["sel"] = c_code
                        st.rerun()
        st.markdown("---")

    st.caption(f"引擎: {_REC_ENGINE} | 用户: {uid}")

