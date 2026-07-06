"""
AI 智能客服 — Flask API 服务
接收前端消息，调用 DeepSeek API（带 Function Calling），返回 AI 回复。

启动方式：python ai_chat/api.py  （监听 localhost:5000）
"""

from __future__ import annotations
import json
import os
import sys
import traceback
from pathlib import Path

# Windows 终端 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# 确保项目根目录在 sys.path 中
PROJ_DIR = Path(__file__).resolve().parent.parent  # ai_chat/ -> project root
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from ai_chat.tools import TOOLS, execute_tool

# ── 配置 ──────────────────────────────────────────────────────────────────────
def _load_env():
    """从 .env 文件加载环境变量（兼容直接运行和 start.bat 启动）"""
    env_file = PROJ_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

_load_env()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"       # deepseek-chat（V3）或 deepseek-reasoner（R1）

app = Flask(__name__)
CORS(app)  # 允许 Streamlit（8501）跨端口调用

# ── 系统提示词 ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是"小T"，TrendSense 商品流行性预测与个性化推荐系统的智能客服。

## 身份与能力
- TrendSense 专属客服，可实时查询系统数据（商品、推荐、库存、预测）
- 技术栈：LSTM 时序预测 + Item-CF 协同过滤 + 内容推荐
- 数据：UCI Online Retail II，覆盖 4000+ 商品、4300+ 用户

## 回复规则（严格遵守）
1. 简洁第一：每次回复 3~8 句话
2. 纯文本：禁止 Markdown（**、##、-、|、` 等），用中文标点和空格
3. 分点必须换行：每个要点之间空一行，用数字序号或项目符号开头
   正确示例：
   1. 小爆米花桶 预测周销 1050 件，均价 0.84 英镑
   2. 二战滑翔机模型 预测周销 866 件，均价 0.29 英镑
4. 数值四舍五入：1050 件，不是 1050.23 件
5. 不说废话套话
6. 最多 1 个 emoji"""


# ═══════════════════════════════════════════════════════════════════════════════
# 核心：带 Function Calling 的对话
# ═══════════════════════════════════════════════════════════════════════════════
def chat_with_deepseek(messages: list[dict]) -> str:
    """
    调用 DeepSeek API 进行对话，支持多轮工具调用。
    返回最终的 AI 回复文本。
    """
    # 确保系统提示词在第一条
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    # 最多循环 5 次（防止工具调用死循环）
    for _ in range(5):
        resp = _call_deepseek(full_messages)
        if resp is None:
            return "抱歉，AI 服务暂时不可用，请稍后再试。"

        choice = resp["choices"][0]
        msg = choice["message"]

        # 情况1：普通文本回复 → 直接返回
        if msg.get("content") and not msg.get("tool_calls"):
            return msg["content"]

        # 情况2：AI 要求调用工具
        if msg.get("tool_calls"):
            # 把 AI 的工具调用请求加入消息历史
            full_messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": msg["tool_calls"]
            })

            # 执行每个工具调用
            for tc in msg["tool_calls"]:
                tool_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                print(f"[ChatAPI] 调用工具: {tool_name}({args})")

                try:
                    result = execute_tool(tool_name, args)
                except Exception as e:
                    result = json.dumps({
                        "error": f"工具执行失败: {str(e)}"
                    }, ensure_ascii=False)
                    traceback.print_exc()

                # 把工具执行结果加入消息历史
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result
                })

            # 继续循环，让 DeepSeek 基于工具结果生成回复
            continue

        # 情况3：既无 content 也无 tool_calls（异常）
        return "抱歉，我暂时无法处理这个问题。"

    return "抱歉，处理超时，请尝试简化您的问题。"


def _call_deepseek(messages: list[dict]) -> dict | None:
    """单次调用 DeepSeek API"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    try:
        r = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        if r.status_code != 200:
            print(f"[ChatAPI] DeepSeek API 错误: {r.status_code} {r.text[:300]}")
            return None
        return r.json()
    except requests.exceptions.Timeout:
        print("[ChatAPI] DeepSeek API 超时")
        return None
    except Exception as e:
        print(f"[ChatAPI] 请求异常: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    聊天接口
    请求：{"messages": [{"role": "user", "content": "你好"}]}
    响应：{"reply": "你好！有什么可以帮你的？"}
    """
    data = request.get_json(silent=True)
    if not data or "messages" not in data:
        return jsonify({"error": "请提供 messages 字段"}), 400

    messages = data["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        return jsonify({"error": "messages 不能为空"}), 400

    # 只保留 user 和 assistant 角色的消息（防止前端注入 system 消息）
    clean = []
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            clean.append({"role": m["role"], "content": m["content"]})

    print(f"[ChatAPI] 收到消息: {clean[-1]['content'][:60]}...")
    reply = chat_with_deepseek(clean)
    print(f"[ChatAPI] 回复: {reply[:60]}...")

    return jsonify({"reply": reply})


@app.route("/api/health", methods=["GET"])
def api_health():
    """健康检查"""
    return jsonify({"status": "ok", "model": DEEPSEEK_MODEL})


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 56)
    print("🤖 TrendSense AI 智能客服 — Flask API")
    print(f"   监听: http://localhost:5000")
    print(f"   模型: {DEEPSEEK_MODEL}")
    print(f"   API Key: {'已配置' if DEEPSEEK_API_KEY != 'sk-your-api-key-here' else '⚠️ 未配置!'}")
    print("=" * 56)
    app.run(host="127.0.0.1", port=5000, debug=False)
