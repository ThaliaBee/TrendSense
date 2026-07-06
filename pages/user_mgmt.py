"""用户管理页面 — 数据库权限系统 + 文件系统回退"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.session_utils import (
    USE_DB_AUTH,
    permissions,
    log_user_action,
)


# ════════════════════════════════════════════════════════════════════════════
# 新系统：数据库用户管理
# ════════════════════════════════════════════════════════════════════════════
def _render_user_mgmt_db():
    st.header(":material/group: 用户管理")

    current_user = st.session_state.get("current_user")
    if not current_user or not current_user.has_permission("manage_users"):
        st.error(":material/block: 权限不足：需要 manage_users 权限")
        return

    log_user_action("查看用户管理")

    all_perms = permissions.get_all_permissions()
    perm_map = dict(zip(all_perms["permission_name"], all_perms["description"]))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        ":material/description: 用户列表", ":material/person_add: 创建用户", ":material/key: 权限管理", ":material/person_remove: 删除用户", ":material/edit_note: 操作日志",
    ])

    with tab1:
        st.subheader(":material/description: 用户列表")
        users_df = permissions.get_all_users()

        for _, user_row in users_df.iterrows():
            user_perms = permissions.get_user_permissions(int(user_row["id"]))
            with st.expander(f":material/person: {user_row['username']} ({len(user_perms)}个权限)"):
                st.write(f"**创建时间**: {user_row['created_at']}")
                st.write("**权限列表**:")
                if user_perms:
                    cols = st.columns(4)
                    for i, perm in enumerate(user_perms):
                        cols[i % 4].write(f":material/check_circle: {perm_map.get(perm, perm)}")
                else:
                    st.info("无权限")

    with tab2:
        st.subheader(":material/person_add: 创建新用户")

        with st.form("create_user_form"):
            new_username = st.text_input("用户名")
            new_password = st.text_input("密码", type="password")

            st.write("**选择初始权限**:")
            perm_categories = all_perms["category"].unique()

            selected_perms = []
            for cat in sorted(perm_categories):
                cat_perms = all_perms[all_perms["category"] == cat]
                st.write(f"**{cat}类**")
                cols = st.columns(2)
                for i, (_, perm_row) in enumerate(cat_perms.iterrows()):
                    if cols[i % 2].checkbox(
                        f"{perm_row['description']}",
                        key=f"perm_new_{perm_row['permission_name']}",
                    ):
                        selected_perms.append(perm_row["permission_name"])

            if st.form_submit_button("创建用户", use_container_width=True):
                if not new_username or not new_password:
                    st.error("请填写用户名和密码")
                else:
                    success, msg = permissions.create_user(
                        current_user, new_username, new_password, selected_perms,
                    )
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    with tab3:
        st.subheader(":material/key: 权限管理")

        users_df = permissions.get_all_users()
        usernames = users_df["username"].tolist()

        selected_username = st.selectbox("选择用户", usernames)

        if selected_username:
            user_perms = permissions.get_user_permissions(
                int(users_df[users_df["username"] == selected_username].iloc[0]["id"])
            )
            st.write("**当前权限**:")
            if user_perms:
                for perm in user_perms:
                    col1, col2 = st.columns([4, 1])
                    col1.write(f":material/check_circle: {perm_map.get(perm, perm)}")
                    if col2.button("移除", key=f"remove_{perm}"):
                        success, msg = permissions.revoke_permission(
                            current_user, selected_username, perm,
                        )
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
            else:
                st.info("该用户无任何权限")

            st.markdown("---")
            st.write("**添加权限**:")
            available_perms = [
                p for p in all_perms["permission_name"].tolist() if p not in user_perms
            ]

            if available_perms:
                perm_to_add = st.selectbox(
                    "选择权限", available_perms,
                    format_func=lambda x: perm_map.get(x, x),
                )
                if st.button("添加", use_container_width=True):
                    success, msg = permissions.grant_permission(
                        current_user, selected_username, perm_to_add,
                    )
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.info("该用户已拥有所有权限")

    with tab4:
        st.subheader(":material/person_remove: 删除用户")
        st.warning(":material/warning: 删除用户将永久移除该用户及其所有权限，此操作不可撤销！")

        users_df = permissions.get_all_users()
        other_users = users_df[users_df["username"] != current_user.username]["username"].tolist()

        if other_users:
            col1, col2 = st.columns([3, 1])

            with col1:
                user_to_delete = st.selectbox("选择要删除的用户", other_users)

                if user_to_delete:
                    user_info = users_df[users_df["username"] == user_to_delete].iloc[0]
                    user_perms = permissions.get_user_permissions(int(user_info["id"]))

                    st.info(f"""
                    **用户信息**：
                    - 用户名：{user_to_delete}
                    - 创建时间：{user_info['created_at']}
                    - 权限数量：{len(user_perms)}个
                    """)

            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button(":material/delete: 确认删除", type="secondary", use_container_width=True):
                    success, msg = permissions.delete_user(current_user, user_to_delete)
                    if success:
                        st.success(msg)
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(msg)
        else:
            st.info("没有其他用户可以删除")

    with tab5:
        st.subheader(":material/edit_note: 操作日志")
        n_logs = st.slider("显示条数", 20, 200, 50)
        logs = permissions.get_recent_logs(n_logs)
        st.text_area("日志内容", "\n".join(logs[-n_logs:]), height=400)


# ════════════════════════════════════════════════════════════════════════════
# 旧系统：文件用户管理
# ════════════════════════════════════════════════════════════════════════════
def _render_user_mgmt_file():
    st.header(":material/group: 用户管理（文件系统）")
    log_user_action("查看用户管理")

    from auth import list_users, add_user, delete_user

    users = list_users()
    st.markdown("### 当前用户")
    st.dataframe(
        pd.DataFrame(users)
        .rename(columns={"username": "用户名", "role": "角色"})
        .set_index("用户名"),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### :material/person_add: 添加用户")
    c1, c2, c3 = st.columns(3)
    nu = c1.text_input("用户名", key="nu", placeholder="新用户名")
    npwd = c2.text_input("密码", type="password", key="np", placeholder="密码")
    nr = c3.selectbox(
        "角色", ["analyst", "operator", "admin"],
        format_func=lambda x: {"admin": "管理员", "operator": "运营", "analyst": "分析师"}[x],
    )
    if st.button(":material/check_circle: 添加", use_container_width=True):
        if nu and npwd:
            if add_user(nu.strip(), npwd.strip(), nr):
                st.success(f"用户 {nu} 已添加！")
                log_user_action(f"添加用户{nu}({nr})")
                st.rerun()
            else:
                st.error("用户名已存在")
        else:
            st.error("请填写完整")

    st.markdown("### :material/person_remove: 删除用户")
    others = [u["username"] for u in users if u["username"] != st.session_state["username"]]
    if others:
        c1, c2 = st.columns([3, 1])
        du = c1.selectbox("选择用户", others, key="du")
        st.markdown("<br>", unsafe_allow_html=True)
        if c2.button(":material/delete: 删除", use_container_width=True, type="secondary"):
            if delete_user(du):
                st.success(f"已删除 {du}")
                log_user_action(f"删除用户{du}")
                st.rerun()
            else:
                st.error("删除失败")


# ════════════════════════════════════════════════════════════════════════════
# 页面入口 — 根据配置选择 DB 或文件系统
# ════════════════════════════════════════════════════════════════════════════
if USE_DB_AUTH and permissions is not None:
    _render_user_mgmt_db()
else:
    _render_user_mgmt_file()
