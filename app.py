"""TrendSense — 商品流行性预测与个性化推荐系统
Streamlit Dashboard · 全侧边栏导航 · F7+F8
运行：streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import datetime

import streamlit as st

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

st.set_page_config(page_title="TrendSense", page_icon=":material/auto_awesome:", layout="wide")

# 登录页侧边栏防护：JS 操作 parent 文档（CSS 跨不过 iframe）
import streamlit.components.v1 as _comp
_comp.html("""
<script>
(function() {
    var sidebar = parent.document.querySelector('[data-testid=\"stSidebar\"]');
    if (sidebar) { sidebar.style.display = 'none'; sidebar.style.width = '0'; }
    new MutationObserver(function() {
        var s = parent.document.querySelector('[data-testid=\"stSidebar\"]');
        if (s && s.style.display !== 'none') { s.style.display = 'none'; s.style.width = '0'; }
    }).observe(parent.document.body, {childList: true, subtree: true, attributes: true});
})();
</script>
""", height=0)

# ── 权限系统 ──
from modules.session_utils import (
    USE_DB_AUTH, permissions,
    has_permission, log_user_action,
)

# ── 旧系统回退 ──
if not USE_DB_AUTH:
    from auth import login, log_action, init_users

from ai_chat.widget import render_chat_widget  # AI 智能客服悬浮窗

# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.login-card {
    background: #fff; border-radius: 20px; padding: 3rem 2.5rem 2rem 2.5rem;
    box-shadow: 0 0 0 2px #6C5CE7, 0 0 0 6px rgba(108,92,231,0.12), 0 20px 60px rgba(0,0,0,0.1);
    text-align: center;
}
.login-card h2 { color: #2D3436; font-weight: 700; margin-bottom: 0.2rem; }
.login-card .sub { color: #636E72; font-size: 0.9rem; margin-bottom: 1.5rem; }
div.stButton > button { border-radius: 10px !important; font-weight: 500 !important; }
div.stButton > button:hover { transform: translateY(-1px); }
section[data-testid="stSidebar"] { display: none !important; width: 0 !important; min-width: 0 !important; overflow: hidden !important; background: linear-gradient(180deg, #F8F9FD 0%, #EDF0F7 100%); }
button[kind="tertiary"] { color: #999 !important; font-size: 0.82rem !important; padding: 0 !important; }
button[kind="tertiary"]:hover { color: #6C5CE7 !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
def init_session():
    for k, v in {
        "logged_in": False, "username": "", "role": "", "login_error": "",
        "report_ready": False, "report_content": "",
        "current_user": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

if not USE_DB_AUTH:
    init_users()
init_session()


# ════════════════════════════════════════════════════════════════════════════
# 数据加载（从共享模块导入，跨页面复用缓存）
# ════════════════════════════════════════════════════════════════════════════
from modules.data_loaders import (
    load_item_meta, load_forecasts,
    load_cf_metrics, load_lstm_metrics,
    get_inventory_with_meta,
)

# ════════════════════════════════════════════════════════════════════════════
# 登录页
# ════════════════════════════════════════════════════════════════════════════
def render_login():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div class="login-card">
            <h2>🔮 TrendSense</h2>
            <div class="sub">商品流行性预测与个性化推荐系统</div>
        """, unsafe_allow_html=True)

        if st.session_state.get("login_error"):
            st.error(st.session_state["login_error"])

        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="请输入用户名")
            password = st.text_input("密码", type="password", placeholder="请输入密码")

            if st.form_submit_button(":material/login: 登录系统", use_container_width=True):
                if not username or not password:
                    st.session_state["login_error"] = "请输入用户名和密码"
                    st.rerun()

                username = username.strip()
                password = password.strip()

                if USE_DB_AUTH:
                    user = permissions.login(username, password)
                    if user:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        st.session_state["current_user"] = user
                        if user.is_admin():
                            st.session_state["role"] = "admin"
                        else:
                            st.session_state["role"] = "user"
                        st.session_state["login_error"] = ""
                        permissions.log_action(username, "登录系统")
                        st.rerun()
                    else:
                        st.session_state["login_error"] = "用户名或密码错误"
                        st.rerun()
                else:
                    role = login(username, password)
                    if role:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        st.session_state["role"] = role
                        st.session_state["current_user"] = None
                        st.session_state["login_error"] = ""
                        log_action(username, "登录系统")
                        st.rerun()
                    else:
                        st.session_state["login_error"] = "用户名或密码错误"
                        st.rerun()

        st.markdown("---")
        if USE_DB_AUTH:
            st.caption(":material/admin_panel_settings: 数据库权限系统 · 默认账号: `admin`/`admin123`")
        else:
            st.caption("演示账号: `admin`/`admin123` · `operator`/`operator123` · `analyst`/`analyst123`")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, c1, c2, c3, c4, _ = st.columns([1, 1, 1, 1, 1, 1])
    c1.caption(":material/psychology: LSTM 时序预测")
    c2.caption(":material/hub: Item-CF 协同过滤")
    c3.caption(":material/monitoring: Streamlit 可视化")
    c4.caption(":material/dataset: UCI Retail 数据集")


# ════════════════════════════════════════════════════════════════════════════
# 报告生成
# ════════════════════════════════════════════════════════════════════════════
def generate_report() -> str:
    forecasts = load_forecasts()
    im = load_item_meta()
    inv = get_inventory_with_meta()
    cf_m = load_cf_metrics(); lstm_m = load_lstm_metrics()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# TrendSense 分析报告", "",
        f"**时间**: {now}  |  **用户**: {st.session_state['username']}",
        f"**数据**: UCI Online Retail II", "",
        "---", "",
        "## 1. 系统概览", "",
        f"| 指标 | 数值 |", f"|------|------|",
        f"| CF Hit Rate@10 | {cf_m.get('hit_rate@10','N/A')} |",
        f"| CF Precision@10 | {cf_m.get('precision@10','N/A')} |",
        f"| LSTM MAE | {lstm_m.get('mae','N/A')} 件 |",
        f"| LSTM RMSE | {lstm_m.get('rmse','N/A')} 件 |", "",
    ]
    if forecasts is not None:
        lw = forecasts["week_idx"].max()
        top = (forecasts[forecasts["week_idx"] == lw].nlargest(10, "pred")
               .merge(im[["StockCode", "Description_CN"]], on="StockCode", how="left"))
        lines.extend(["## 2. 流行趋势 Top-10", "",
                       "| # | StockCode | 商品 | 预测 | 下界 | 上界 |",
                       "|---|-----------|------|------|------|------|"])
        for i, (_, r) in enumerate(top.iterrows(), 1):
            lines.append(f"| {i} | {r['StockCode']} | {str(r.get('Description_CN',''))[:28]} | {r['pred']:.0f} | {r['lower']:.0f} | {r['upper']:.0f} |")
        lines.append("")
    alerts = inv[inv["status"] == "警告"]; low = inv[inv["status"] == "偏低"]
    ok = int((inv["status"] == "充足").sum())
    lines.extend(["## 3. 库存", "", f"充足:{ok} 偏低:{len(low)} 警告:{len(alerts)}", ""])
    if len(alerts) > 0:
        lines.extend(["### 警告商品", "",
                       "| StockCode | 商品 | 库存 | 需求 |",
                       "|-----------|------|------|------|"])
        for _, r in alerts.head(20).iterrows():
            lines.append(f"| {r['StockCode']} | {str(r.get('Description_CN',''))[:28]} | {r['stock']} | {r['predicted_demand']} |")
        lines.append("")
    lines.extend(["## 4. 建议", ""])
    if len(alerts) > 0: lines.append(f"- :material/warning: {len(alerts)} 个商品需立即补货")
    if len(low) > 0:    lines.append(f"- :material/bolt: {len(low)} 个商品建议关注")
    lines.extend(["", "---", "*TrendSense 自动生成*"])
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.get("logged_in"):
        render_login()
        return

    # 登录后显示侧边栏（JS 操作 parent 文档）
    import streamlit.components.v1 as _comp2
    _comp2.html("""
    <script>
    var s = parent.document.querySelector('[data-testid=\"stSidebar\"]');
    if (s) { s.style.display = 'flex'; s.style.width = '200px'; }
    </script>
    """, height=0)
    st.markdown("""<style>
    section[data-testid="stSidebar"] { display: flex !important; width: 200px !important; }
    </style>""", unsafe_allow_html=True)

    render_chat_widget()  # AI 智能客服悬浮窗

    # ── 构建导航（仅显示用户有权访问的页面）──
    nav_pages = []
    if has_permission("view_statistics"):
        nav_pages.append(st.Page("pages/overview.py", title="总览", icon=":material/dashboard:", default=True))
    if has_permission("view_predictions"):
        nav_pages.append(st.Page("pages/popularity.py", title="流行性预测", icon=":material/trending_up:"))
    if has_permission("view_recommendations"):
        nav_pages.append(st.Page("pages/recommend.py", title="个性化推荐", icon=":material/track_changes:"))
    if has_permission("view_inventory"):
        nav_pages.append(st.Page("pages/inventory.py", title="库存预警", icon=":material/inventory_2:"))
    if has_permission("view_logs"):
        nav_pages.append(st.Page("pages/logs.py", title="操作日志", icon=":material/description:"))
    if has_permission("manage_users") or has_permission("manage_permissions"):
        nav_pages.append(st.Page("pages/user_mgmt.py", title="用户管理", icon=":material/group:"))

    pg = st.navigation(nav_pages)

    # ── 侧边栏底部：报告 / 登出 ──
    with st.sidebar:
        if has_permission("export_reports"):
            st.markdown("#### :material/download: 报告导出")
            if st.button(":material/summarize: 生成分析报告", use_container_width=True):
                st.session_state["report_content"] = generate_report()
                st.session_state["report_ready"] = True
                log_user_action("生成分析报告")
                st.success("报告已生成！")
            if st.session_state.get("report_ready"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(":material/download: 下载报告", data=st.session_state["report_content"],
                                   file_name=f"TrendSense_{ts}.md", mime="text/markdown",
                                   use_container_width=True)

        st.markdown("---")
        st.markdown("### :material/auto_awesome: TrendSense")
        st.caption(f"用户名: **{st.session_state['username']}**")
        if st.button(":material/logout: 退出登录", use_container_width=True):
            log_user_action("退出系统")
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.caption("© 2026 TrendSense v2.0")

    pg.run()


if __name__ == "__main__":
    main()
