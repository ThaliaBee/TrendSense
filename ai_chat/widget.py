"""
AI 智能客服 — 悬浮聊天窗组件

方案：st.markdown 注入 HTML/CSS（直入页面 DOM，position:fixed 真正生效）
     + st.components.v1.html 注入 JS（小 iframe 访问 parent.document 操控 UI）
效果类似搜狗输入法悬浮窗 — 固定在右下角，不随页面滚动变化位置。
"""

from __future__ import annotations
import streamlit as st

API_URL = "http://127.0.0.1:5000/api/chat"
ID_PREFIX = "_chat_trendsense_"  # 避免和 Streamlit 组件 ID 冲突

# ═══════════════════════════════════════════════════════════════════════════════
# CSS + HTML 骨架（inject 到页面 DOM 中，不是 iframe）
# ═══════════════════════════════════════════════════════════════════════════════
SKELETON = rf"""
<style>
/* ── 悬浮按钮 ──────────────────────────────────────────────────────────── */
#{ID_PREFIX}fab {{
  position: fixed;
  bottom: 28px;
  right: 28px;
  width: 54px;
  height: 54px;
  border-radius: 50%;
  border: none;
  cursor: pointer;
  z-index: 99998;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  box-shadow: 0 4px 18px rgba(102,126,234,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.2s, box-shadow 0.2s, opacity 0.25s;
}}
#{ID_PREFIX}fab:hover {{
  transform: scale(1.08);
  box-shadow: 0 6px 24px rgba(102,126,234,0.65);
}}
#{ID_PREFIX}fab svg {{
  width: 26px;
  height: 26px;
  fill: #fff;
}}
#{ID_PREFIX}fab.hidden {{
  opacity: 0;
  pointer-events: none;
  transform: scale(0.8);
}}

/* ── 聊天面板 ──────────────────────────────────────────────────────────── */
#{ID_PREFIX}panel {{
  position: fixed;
  bottom: 94px;
  right: 28px;
  width: 380px;
  height: 500px;
  max-height: calc(100vh - 130px);
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 8px 40px rgba(0,0,0,0.15);
  display: flex;
  flex-direction: column;
  z-index: 99999;
  overflow: hidden;
  opacity: 0;
  transform: translateY(12px) scale(0.95);
  pointer-events: none;
  transition: opacity 0.25s, transform 0.25s;
}}
#{ID_PREFIX}panel.open {{
  opacity: 1;
  transform: translateY(0) scale(1);
  pointer-events: all;
}}

/* 面板头部 */
#{ID_PREFIX}header {{
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
  padding: 12px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  font-size: 14px;
  font-weight: 600;
}}
#{ID_PREFIX}header .dot {{
  width: 8px; height: 8px;
  border-radius: 50%;
  background: #2ed573;
  display: inline-block;
  margin-right: 6px;
  animation: _chat_pulse 2s infinite;
}}
@keyframes _chat_pulse {{
  0%,100% {{ opacity:1; }}
  50% {{ opacity:0.35; }}
}}
#{ID_PREFIX}close {{
  background: rgba(255,255,255,0.18);
  border: none; color: #fff;
  width: 26px; height: 26px;
  border-radius: 50%;
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
  transition: background 0.2s;
}}
#{ID_PREFIX}close:hover {{ background: rgba(255,255,255,0.32); }}

/* 消息区 */
#{ID_PREFIX}msgs {{
  flex: 1;
  overflow-y: auto;
  padding: 14px 12px;
  background: #f5f6fa;
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
#{ID_PREFIX}msgs::-webkit-scrollbar {{ width: 5px; }}
#{ID_PREFIX}msgs::-webkit-scrollbar-thumb {{
  background: #d0d5dd;
  border-radius: 10px;
}}

/* 空状态 */
#{ID_PREFIX}empty {{
  text-align: center;
  color: #a0a7b8;
  margin-top: 50px;
  font-size: 13px;
  line-height: 1.8;
}}

/* 消息气泡 */
#{ID_PREFIX}msgs .bubble {{
  max-width: 82%;
  padding: 10px 14px;
  border-radius: 16px;
  font-size: 13.5px;
  line-height: 1.6;
  word-break: break-word;
  white-space: pre-wrap;
  animation: _chat_fade 0.28s ease-out;
}}
@keyframes _chat_fade {{
  from {{ opacity:0; transform:translateY(8px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}
#{ID_PREFIX}msgs .row.user {{
  align-self: flex-end;
}}
#{ID_PREFIX}msgs .row.user .bubble {{
  background: linear-gradient(135deg, #2ed573 0%, #26ae60 100%);
  color: #fff;
  border-bottom-right-radius: 4px;
}}
#{ID_PREFIX}msgs .row.bot {{
  align-self: flex-start;
}}
#{ID_PREFIX}msgs .row.bot .bubble {{
  background: #fff;
  color: #2d3436;
  border-bottom-left-radius: 4px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
#{ID_PREFIX}msgs .time {{
  font-size: 10px;
  color: #b2bec3;
  margin-top: 2px;
  padding: 0 4px;
}}
#{ID_PREFIX}msgs .row.user .time {{ text-align: right; }}

/* 加载动画 */
.{ID_PREFIX}loading {{
  align-self: flex-start;
  background: #fff;
  border-radius: 16px;
  border-bottom-left-radius: 4px;
  padding: 10px 16px;
  display: flex;
  gap: 5px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  animation: _chat_fade 0.28s ease-out;
}}
#{ID_PREFIX}loading .dot {{
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #b2bec3;
  animation: _chat_bounce 1.4s infinite ease-in-out both;
}}
#{ID_PREFIX}loading .dot:nth-child(1) {{ animation-delay: -0.32s; }}
#{ID_PREFIX}loading .dot:nth-child(2) {{ animation-delay: -0.16s; }}
#{ID_PREFIX}loading .dot:nth-child(3) {{ animation-delay: 0s; }}
@keyframes _chat_bounce {{
  0%,80%,100% {{ transform:scale(0.6); }}
  40% {{ transform:scale(1.2); }}
}}

/* 输入区 */
#{ID_PREFIX}input_area {{
  padding: 10px 14px 14px;
  background: #fff;
  border-top: 1px solid #edf0f5;
  display: flex;
  gap: 8px;
  align-items: flex-end;
  flex-shrink: 0;
}}
#{ID_PREFIX}input_area textarea {{
  flex: 1;
  border: 1.5px solid #e0e4ea;
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 13.5px;
  font-family: inherit;
  resize: none;
  outline: none;
  max-height: 80px;
  line-height: 1.4;
  transition: border-color 0.2s;
}}
#{ID_PREFIX}input_area textarea:focus {{
  border-color: #667eea;
}}
#{ID_PREFIX}send {{
  width: 40px; height: 40px;
  border-radius: 50%;
  border: none;
  cursor: pointer;
  flex-shrink: 0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  box-shadow: 0 2px 8px rgba(102,126,234,0.35);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.2s, opacity 0.2s;
}}
#{ID_PREFIX}send:hover {{
  transform: scale(1.06);
}}
#{ID_PREFIX}send:disabled {{
  opacity: 0.45;
  cursor: not-allowed;
  transform: none;
}}
#{ID_PREFIX}send svg {{
  width: 17px; height: 17px;
  fill: #fff;
  margin-left: 1px;
}}
</style>

<!-- 悬浮按钮 -->
<button id="{ID_PREFIX}fab" title="智能客服">
  <svg viewBox="0 0 24 24">
    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.17L4 17.17V4h16v12z"/>
    <circle cx="12" cy="11" r="1.5"/><circle cx="8" cy="11" r="1.5"/><circle cx="16" cy="11" r="1.5"/>
  </svg>
</button>

<!-- 聊天面板 -->
<div id="{ID_PREFIX}panel">
  <div id="{ID_PREFIX}header">
    <span><span class="dot"></span>小T · 智能客服</span>
    <button id="{ID_PREFIX}close">✕</button>
  </div>
  <div id="{ID_PREFIX}msgs">
    <div id="{ID_PREFIX}empty">
      💬<br>你好！我是 <b>小T</b><br>
      🔥 "热门商品TOP3"<br>
      🎯 "给用户 12347 推荐"<br>
      📦 "哪些商品库存告急？"
    </div>
  </div>
  <div id="{ID_PREFIX}input_area">
    <textarea id="{ID_PREFIX}input" rows="1" placeholder="输入问题…" maxlength="500"></textarea>
    <button id="{ID_PREFIX}send" title="发送">
      <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
    </button>
  </div>
</div>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# JS 加载器（藏在 0 高度 iframe 里，通过 parent.document 操控 UI）
# ═══════════════════════════════════════════════════════════════════════════════
JS_LOADER = rf"""
<!DOCTYPE html><html><body>
<script>
(function() {{
  "use strict";
  var P = "{ID_PREFIX}";
  var API = "{API_URL}";

  // 从 parent 获取 DOM 元素
  var doc = parent.document;
  var fab  = doc.getElementById(P + "fab");
  var panel = doc.getElementById(P + "panel");
  var closeBtn = doc.getElementById(P + "close");
  var msgs  = doc.getElementById(P + "msgs");
  var empty = doc.getElementById(P + "empty");
  var input = doc.getElementById(P + "input");
  var sendBtn = doc.getElementById(P + "send");

  if (!fab || !panel) {{
    console.error("[Chat] DOM not found, prefix=" + P);
    return;
  }}

  var isOpen = false, isSending = false;
  var messages = [];

  // ── 展开/收起 ──────────────────────────────────────────────────────────
  fab.addEventListener("click", function() {{
    isOpen = true;
    panel.classList.add("open");
    fab.classList.add("hidden");
    input.focus();
  }});
  closeBtn.addEventListener("click", function() {{
    isOpen = false;
    panel.classList.remove("open");
    fab.classList.remove("hidden");
  }});
  doc.addEventListener("keydown", function(e) {{
    if (e.key === "Escape" && isOpen) {{
      isOpen = false;
      panel.classList.remove("open");
      fab.classList.remove("hidden");
    }}
  }});

  // ── 辅助 ──────────────────────────────────────────────────────────────
  function pad(n) {{ return n < 10 ? "0" + n : "" + n; }}

  function scrollBottom() {{
    requestAnimationFrame(function() {{ msgs.scrollTop = msgs.scrollHeight; }});
  }}

  // ── 消息气泡 ──────────────────────────────────────────────────────────
  function addBubble(role, text) {{
    if (empty) empty.style.display = "none";
    var row = doc.createElement("div");
    row.className = "row " + role;
    var b = doc.createElement("div");
    b.className = "bubble";
    b.textContent = text;
    var t = doc.createElement("div");
    t.className = "time";
    t.textContent = pad(new Date().getHours()) + ":" + pad(new Date().getMinutes());
    row.appendChild(b);
    row.appendChild(t);
    msgs.appendChild(row);
    scrollBottom();
  }}

  // ── 加载动画 ──────────────────────────────────────────────────────────
  function showLoading() {{
    if (empty) empty.style.display = "none";
    var el = doc.createElement("div");
    el.id = P + "loading_el";
    el.className = P + "loading";
    el.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
    msgs.appendChild(el);
    scrollBottom();
  }}

  function hideLoading() {{
    var el = doc.getElementById(P + "loading_el");
    if (el) el.remove();
  }}

  function setSending(state) {{
    isSending = state;
    sendBtn.disabled = state;
    input.disabled  = state;
    if (state) showLoading(); else hideLoading();
  }}

  // ── 发送 ──────────────────────────────────────────────────────────────
  function send() {{
    var text = input.value.trim();
    if (!text || isSending) return;
    input.value = "";
    input.style.height = "auto";
    addBubble("user", text);
    messages.push({{ role: "user", content: text }});
    setSending(true);

    fetch(API, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ messages: messages }})
    }})
    .then(function(r) {{ if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }})
    .then(function(d) {{
      setSending(false);
      var reply = d.reply || "(empty)";
      addBubble("bot", reply);
      messages.push({{ role: "assistant", content: reply }});
    }})
    .catch(function(e) {{
      setSending(false);
      addBubble("bot", "抱歉，AI 服务暂不可用 😥\\n请确认已运行 python ai_chat/api.py");
      console.error("[Chat]", e);
    }});
  }}

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", function(e) {{
    if (e.key === "Enter" && !e.shiftKey) {{ e.preventDefault(); send(); }}
  }});
  input.addEventListener("input", function() {{
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 80) + "px";
  }});

  console.log("[Chat] ready. prefix=" + P);
}})();
</script>
</body></html>
"""


def render_chat_widget():
    """渲染智能客服悬浮窗。

    Step 1: st.markdown 把 HTML/CSS 直接注入页面 DOM
            → position:fixed 真正相对于视口，滚动不动
    Step 2: st.components.v1.html 载入 JS（小 iframe 通过 parent.document 操控 UI）
    """
    st.markdown(SKELETON, unsafe_allow_html=True)
    st.components.v1.html(JS_LOADER, height=0, scrolling=False)
