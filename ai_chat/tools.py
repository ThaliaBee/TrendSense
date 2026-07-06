"""
AI 智能客服 — 工具定义模块
将 TrendSense 系统的推荐、预测、库存等功能封装为 DeepSeek 可调用的 Function Tools。
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJ_DIR = Path(__file__).resolve().parent.parent  # ai_chat/ -> project root
DATA_DIR = PROJ_DIR / "data"
CACHE_DIR = PROJ_DIR / "cache"

# ── 工具 JSON Schema（发给 DeepSeek）──────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_system_overview",
            "description": "获取系统总览信息：商品总数、用户总数、交易记录数、库存状态分布、模型性能指标等",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_popular_items",
            "description": "获取当前预测最热门的商品排行（基于LSTM深度学习预测模型），返回TOP N商品及其预测销量",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "返回TOP N个热门商品，默认10，最大20"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": "为指定用户生成个性化商品推荐，融合协同过滤(Item-CF)和流行度预测，支持新用户冷启动",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "用户ID（CustomerID）"},
                    "n": {"type": "integer", "description": "推荐数量，默认5，最大10"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory_status",
            "description": "查询库存状态分布和详情。可筛选：警告（库存不足）、偏低、充足商品，或查看全部",
            "parameters": {
                "type": "object",
                "properties": {
                    "status_filter": {
                        "type": "string",
                        "enum": ["all", "警告", "偏低", "充足"],
                        "description": "筛选条件：all=全部, 警告=需立即补货, 偏低=建议关注, 充足=正常"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_detail",
            "description": "查询单个商品的详细信息：名称、历史销量、均价、当前库存、预测需求、库存状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {"type": "string", "description": "商品编码（StockCode），如 '22326'"}
                },
                "required": ["stock_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "获取系统的技术介绍和使用说明：系统架构、算法原理、功能模块、使用方法等",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": ["架构", "算法", "功能", "使用", "全部"],
                        "description": "想了解的主题：架构=系统架构, 算法=推荐算法原理, 功能=功能模块介绍, 使用=使用方法, 全部=所有信息"
                    }
                },
                "required": []
            }
        }
    },
]

# ── 数据加载辅助函数 ──────────────────────────────────────────────────────────

def _load_item_meta():
    """加载商品元数据"""
    p = DATA_DIR / "item_meta.parquet"
    return pd.read_parquet(p) if p.exists() else None

def _load_forecasts():
    """加载预测数据"""
    p = CACHE_DIR / "forecasts.parquet"
    return pd.read_parquet(p) if p.exists() else None

def _load_inventory():
    """加载库存状态（优先用含 LSTM 预测的 parquet，其次 MySQL，最后模拟）"""
    inv = None

    # 优先：inventory_status.parquet（含 predicted_demand 和 status，与 Streamlit 同源）
    p = CACHE_DIR / "inventory_status.parquet"
    if p.exists():
        inv = pd.read_parquet(p)

    # 其次：MySQL 原始库存（只有 stock/threshold，需补算）
    if inv is None:
        try:
            from modules import db
            inv = db.load_inventory()
        except Exception:
            pass

    # 兜底：模拟数据
    if inv is None:
        im = _load_item_meta()
        if im is not None:
            rng = np.random.default_rng(42)
            wa = im["total_sales"] / 53.0
            s = (wa * rng.uniform(4, 10, len(im))).round(0).astype(int)
            th = (wa * 3).round(0).astype(int)
            dd = (th / 1.5).round(0).astype(int)
            inv = pd.DataFrame({"StockCode": im["StockCode"], "stock": s,
                               "predicted_demand": dd, "threshold": th})

    # 确保有 status 和 predicted_demand 列（MySQL 原始表可能只有 stock/threshold）
    if inv is not None:
        if "status" not in inv.columns:
            if "predicted_demand" not in inv.columns:
                inv["predicted_demand"] = (inv["threshold"] / 1.5).round(0).astype(int)
            inv["status"] = "充足"
            inv.loc[inv["stock"] < inv["threshold"], "status"] = "偏低"
            inv.loc[inv["stock"] < inv["predicted_demand"], "status"] = "警告"

    return inv

def _load_cf_topn():
    """加载协同过滤推荐"""
    p = CACHE_DIR / "cf_topn.parquet"
    return pd.read_parquet(p) if p.exists() else None

def _load_user_item():
    """加载用户购买记录"""
    p = DATA_DIR / "user_item.parquet"
    return pd.read_parquet(p) if p.exists() else None

def _load_tx_stats():
    """加载交易统计"""
    try:
        from modules import db
        return db.transaction_stats()
    except Exception:
        return {"n_users": 0, "n_rows": 0, "total_qty": 0}

def _load_cf_metrics():
    """加载CF评估指标"""
    p = CACHE_DIR / "cf_metrics.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}

def _load_lstm_metrics():
    """加载LSTM评估指标"""
    p = CACHE_DIR / "lstm_metrics.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}

def _load_active_items():
    """加载活跃商品ID列表"""
    p = DATA_DIR / "active_items.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else []

# ── 工具执行函数 ──────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, arguments: dict) -> str:
    """执行指定工具，返回结果字符串（会发回给 DeepSeek 生成自然语言回复）"""

    if tool_name == "get_system_overview":
        return _run_get_system_overview()

    elif tool_name == "get_popular_items":
        top_n = arguments.get("top_n", 10)
        top_n = min(max(int(top_n), 1), 20)
        return _run_get_popular_items(top_n)

    elif tool_name == "get_recommendations":
        user_id = int(arguments["user_id"])
        n = min(max(int(arguments.get("n", 5)), 1), 10)
        return _run_get_recommendations(user_id, n)

    elif tool_name == "get_inventory_status":
        status_filter = arguments.get("status_filter", "all")
        return _run_get_inventory_status(status_filter)

    elif tool_name == "get_item_detail":
        stock_code = str(arguments["stock_code"])
        return _run_get_item_detail(stock_code)

    elif tool_name == "get_system_info":
        topic = arguments.get("topic", "全部")
        return _run_get_system_info(topic)

    else:
        return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)


# ── 各工具实现 ────────────────────────────────────────────────────────────────

def _run_get_system_overview() -> str:
    """系统总览"""
    im = _load_item_meta()
    forecasts = _load_forecasts()
    inv = _load_inventory()
    tx_stats = _load_tx_stats()
    cf_m = _load_cf_metrics()
    lstm_m = _load_lstm_metrics()
    active = _load_active_items()

    # 库存状态统计
    status_counts = {"充足": 0, "偏低": 0, "警告": 0}
    if inv is not None and "status" in inv.columns:
        vc = inv["status"].value_counts()
        for s in vc.index:
            status_counts[str(s)] = int(vc[s])

    # Top-5 热门预测
    top_items = []
    if forecasts is not None and im is not None:
        latest_week = forecasts["week_idx"].max()
        top5 = forecasts[forecasts["week_idx"] == latest_week].nlargest(5, "pred")
        top5 = top5.merge(im[["StockCode", "Description"]], on="StockCode", how="left")
        for _, r in top5.iterrows():
            top_items.append({
                "code": r["StockCode"],
                "name": str(r.get("Description", ""))[:30],
                "predicted_sales": round(float(r["pred"]), 0)
            })

    result = {
        "系统名称": "TrendSense 商品流行性预测与个性化推荐系统",
        "数据概况": {
            "商品总数": len(im) if im is not None else 0,
            "活跃商品": len(active),
            "用户总数": tx_stats.get("n_users", 0),
            "交易记录": f"{tx_stats.get('n_rows', 0):,}",
            "数据时间范围": "2010-12 ~ 2011-12 (53周)"
        },
        "库存状态": status_counts,
        "模型性能": {
            "协同过滤": f"HitRate@10={cf_m.get('hit_rate@10', 'N/A')}, Precision@10={cf_m.get('precision@10', 'N/A')}" if cf_m else "未就绪",
            "LSTM预测": f"MAE={lstm_m.get('mae', 'N/A')}件, RMSE={lstm_m.get('rmse', 'N/A')}件" if lstm_m else "未就绪"
        },
        "热门预测TOP5": top_items
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def _run_get_popular_items(top_n: int) -> str:
    """获取热门商品排行"""
    forecasts = _load_forecasts()
    im = _load_item_meta()

    if forecasts is None:
        return json.dumps({"error": "预测数据未就绪，请先运行 LSTM 模型"}, ensure_ascii=False)

    latest_week = forecasts["week_idx"].max()
    top = forecasts[forecasts["week_idx"] == latest_week].nlargest(top_n, "pred")
    if im is not None:
        top = top.merge(im[["StockCode", "Description", "avg_price"]], on="StockCode", how="left")

    items = []
    for i, (_, r) in enumerate(top.iterrows(), 1):
        items.append({
            "排名": i,
            "商品编号": r["StockCode"],
            "商品名称": str(r.get("Description", "未知"))[:40],
            "预测销量": f"{r['pred']:.0f} 件/周",
            "预测区间": f"{r['lower']:.0f} ~ {r['upper']:.0f}",
            "均价": f"£{r.get('avg_price', 0):.2f}" if pd.notna(r.get('avg_price')) else "N/A"
        })

    result = {
        "热门商品排行": items,
        "说明": f"基于 LSTM 深度学习模型预测的第 {latest_week} 周销量排行",
        "数据更新时间": "预测模型输出，基于近8周历史数据"
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def _run_get_recommendations(user_id: int, n: int) -> str:
    """为用户生成个性化推荐"""
    try:
        from modules.recommend import recommend as rec_func
    except Exception as e:
        return json.dumps({"error": f"推荐模块加载失败: {e}"}, ensure_ascii=False)

    # 加载数据
    cf_topn = _load_cf_topn()
    forecasts = _load_forecasts()
    im = _load_item_meta()
    inv = _load_inventory()
    ui = _load_user_item()

    # 过滤 forecasts 到第一预测周
    if forecasts is not None and "week_idx" in forecasts.columns:
        min_week = forecasts["week_idx"].min()
        forecasts = forecasts[forecasts["week_idx"] == min_week]

    try:
        cards = rec_func(
            user_id=user_id, n=n,
            cf_topn=cf_topn, forecasts=forecasts,
            item_meta=im, inventory=inv,
            user_item=ui, strategy="balanced"
        )
    except Exception as e:
        return json.dumps({"error": f"推荐生成失败: {e}"}, ensure_ascii=False)

    if not cards:
        return json.dumps({
            "用户ID": user_id,
            "推荐结果": [],
            "说明": "该用户暂无推荐数据，可能是新用户或所有候选商品均已购买"
        }, ensure_ascii=False, indent=2)

    rec_list = []
    for i, card in enumerate(cards, 1):
        rec_list.append({
            "排名": i,
            "商品编号": card["StockCode"],
            "商品名称": card["Description"][:35],
            "匹配度": f"{card['final_score']:.1%}",
            "推荐理由": card["reason"],
            "价格": f"£{card['price']:.2f}",
            "库存状态": card["stock_status"],
            "库存数量": f"{card['stock']} 件",
            "冷启动": "是（新用户热门推荐）" if card.get("cold_start") else "否（个性化推荐）"
        })

    result = {
        "用户ID": user_id,
        "推荐数量": len(rec_list),
        "推荐策略": "balanced（平衡协同过滤与流行度）",
        "推荐列表": rec_list
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def _run_get_inventory_status(status_filter: str) -> str:
    """查询库存状态"""
    inv = _load_inventory()
    im = _load_item_meta()

    if inv is None:
        return json.dumps({"error": "库存数据未就绪"}, ensure_ascii=False)

    # 合并商品名称
    if im is not None:
        inv = inv.merge(im[["StockCode", "Description"]], on="StockCode", how="left")

    # 统计
    sc = inv["status"].value_counts()
    summary = {
        "充足": int(sc.get("充足", 0)),
        "偏低": int(sc.get("偏低", 0)),
        "警告": int(sc.get("警告", 0))
    }

    # 筛选
    if status_filter != "all":
        inv = inv[inv["status"] == status_filter]

    # 取前20条
    detail = []
    for _, r in inv.head(20).iterrows():
        detail.append({
            "商品编号": r["StockCode"],
            "商品名称": str(r.get("Description", "未知"))[:30],
            "当前库存": int(r["stock"]),
            "预测需求": int(r.get("predicted_demand", 0)),
            "安全阈值": int(r.get("threshold", 0)),
            "状态": r["status"],
            "缺口": int(r["stock"] - r.get("predicted_demand", 0))
        })

    result = {
        "库存总览": summary,
        "筛选条件": status_filter,
        "详情（前20条）": detail,
        "紧急程度说明": {
            "警告": "库存低于预测需求，需立即补货",
            "偏低": "库存低于安全阈值，建议关注",
            "充足": "库存充足，无需操作"
        }
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def _run_get_item_detail(stock_code: str) -> str:
    """查询单个商品详情"""
    im = _load_item_meta()
    forecasts = _load_forecasts()
    inv = _load_inventory()

    if im is None:
        return json.dumps({"error": "商品数据未就绪"}, ensure_ascii=False)

    # 查询元数据
    meta_row = im[im["StockCode"] == stock_code]
    if len(meta_row) == 0:
        # 尝试模糊匹配
        meta_row = im[im["StockCode"].astype(str).str.contains(stock_code, na=False)]
        if len(meta_row) == 0:
            return json.dumps({
                "error": f"未找到商品编号 '{stock_code}'，请检查编号是否正确。可尝试查询热门商品获取正确编号。"
            }, ensure_ascii=False)

    row = meta_row.iloc[0]
    result = {
        "商品编号": str(row["StockCode"]),
        "商品名称": str(row.get("Description", "未知")),
        "累计销量": f"{int(row['total_sales']):,} 件",
        "均价": f"£{row['avg_price']:.2f}",
    }

    # 库存信息
    if inv is not None:
        inv_row = inv[inv["StockCode"] == stock_code]
        if len(inv_row) > 0:
            ir = inv_row.iloc[0]
            result["库存信息"] = {
                "当前库存": f"{int(ir['stock'])} 件",
                "预测需求": f"{int(ir.get('predicted_demand', 0))} 件",
                "安全阈值": f"{int(ir.get('threshold', 0))} 件",
                "状态": ir["status"]
            }

    # 预测信息
    if forecasts is not None:
        fc_row = forecasts[forecasts["StockCode"] == stock_code]
        if len(fc_row) > 0:
            fc = fc_row.iloc[0]
            result["预测信息"] = {
                "预测下周销量": f"{fc['pred']:.0f} 件",
                "预测区间": f"{fc['lower']:.0f} ~ {fc['upper']:.0f} 件",
            }

    return json.dumps(result, ensure_ascii=False, indent=2)


def _run_get_system_info(topic: str) -> str:
    """系统介绍"""
    info = {
        "全部": {
            "系统名称": "TrendSense — 商品流行性预测与个性化推荐系统",
            "技术架构": {
                "前端": "Streamlit 数据看板 + Plotly 交互图表",
                "后端": "Python 数据处理管线",
                "数据库": "MySQL 8.0（存储消费记录、库存、用户权限）",
                "AI模型": "LSTM 深度学习（流行性预测）+ Item-CF 协同过滤（个性化推荐）"
            },
            "核心算法": {
                "LSTM时序预测": "双隐层(128×2) LSTM 网络，滑动窗口(8周→下周)，Huber损失函数，80%置信区间。预测未来商品销量趋势。",
                "Item-CF协同过滤": "基于商品-商品相似度矩阵，计算用户已购商品的相似商品，生成个性化推荐。支持混合权重(balanced/cf_focused/trending)。",
                "内容推荐": "基于商品描述文本的 TF-IDF 相似度匹配，推荐描述相关的商品。",
                "冷启动策略": "新用户自动切换热门+多样性推荐，用用户ID哈希做确定性差异化。"
            },
            "功能模块": {
                "系统总览": "核心指标仪表盘、模型性能、库存分布、畅销排行",
                "流行性预测": "单商品深度趋势分析、双商品对比、相似商品推荐",
                "个性化推荐": "输入用户ID，融合CF+流行度+多样性生成推荐卡片",
                "库存预警": "库存状态看板、库存vs预测散点图、智能补货建议",
                "操作日志": "记录所有用户操作，支持查看和导出"
            },
            "使用说明": {
                "登录": "演示账号 admin/admin123，或联系管理员创建账号",
                "查看预测": "点击「流行性预测」→ 选择商品 → 查看趋势图和预测",
                "生成推荐": "点击「个性化推荐」→ 输入用户ID → 调整策略和数量",
                "库存管理": "点击「库存预警」→ 查看警告商品 → 获取补货建议",
                "导出报告": "侧边栏「生成分析报告」→ 下载 Markdown 格式报告"
            },
            "数据说明": "基于 UCI Online Retail II 数据集（2010-2012年英国电商交易），约39万条清洗后记录，覆盖4000+商品和4300+用户"
        }
    }

    if topic == "全部":
        return json.dumps(info["全部"], ensure_ascii=False, indent=2)
    else:
        topic_map = {
            "架构": "技术架构",
            "算法": "核心算法",
            "功能": "功能模块",
            "使用": "使用说明"
        }
        key = topic_map.get(topic, topic)
        data = {key: info["全部"].get(key, "无相关信息")}
        return json.dumps(data, ensure_ascii=False, indent=2)
