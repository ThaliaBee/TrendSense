"""操作日志页面"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.session_utils import (
    USE_DB_AUTH,
    permissions,
    PROJ_DIR,
    log_user_action,
    get_logs,
)


# ════════════════════════════════════════════════════════════════════════════
# 页面入口
# ════════════════════════════════════════════════════════════════════════════
st.header(":material/description: 操作日志")
log_user_action("查看操作日志")

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    n = st.slider("显示条数", 20, 200, 50, key="log_n")
with c2:
    if st.button(":material/refresh: 刷新", use_container_width=True):
        st.rerun()
with c3:
    if st.button(":material/delete: 清空", use_container_width=True, type="secondary"):
        lp = PROJ_DIR / "logs" / "operations.log"
        if lp.exists():
            lp.write_text("", encoding="utf-8")
        log_user_action("清空日志")
        st.rerun()

if USE_DB_AUTH and permissions is not None:
    logs_text = permissions.get_recent_logs(n)
    st.text_area("日志内容", "\n".join(logs_text[-n:]), height=500)
else:
    logs = get_logs(n)
    if not logs:
        st.info("暂无日志")
    else:
        st.dataframe(
            pd.DataFrame(logs).rename(columns={
                "time": "时间", "user": "用户", "action": "操作",
            }),
            use_container_width=True, height=500,
        )
