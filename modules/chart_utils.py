"""共享 ECharts 图表工具 — 供各页面复用的图表函数"""

from __future__ import annotations

from streamlit_echarts import st_echarts


def trend_fig(hist, fc, item_id, height="460px", show_markline=True):
    """绘制商品销量趋势 + 预测 (ECharts)

    Args:
        hist: 历史数据 DataFrame (需含 week_idx, sales 列)
        fc: 预测数据 DataFrame (需含 week_idx, pred, lower, upper 列)
        item_id: 商品 StockCode
        height: 图表高度
        show_markline: 是否显示预测起点标记线
    """
    weeks = hist["week_idx"].tolist()
    sales = hist["sales"].tolist()

    series = [{
        "name": "历史销量",
        "type": "line",
        "data": sales,
        "lineStyle": {"color": "#6C5CE7", "width": 2.5},
        "itemStyle": {"color": "#6C5CE7"},
        "symbol": "circle",
        "symbolSize": 5,
    }]

    if len(fc) > 0:
        fc_weeks = fc["week_idx"].tolist()
        all_weeks = weeks + fc_weeks
        hist_data = sales + [None] * len(fc_weeks)
        series[0]["data"] = hist_data

        pred_data = [None] * len(weeks) + fc["pred"].tolist()
        series.append({
            "name": "预测销量",
            "type": "line",
            "data": pred_data,
            "lineStyle": {"color": "#FD79A8", "width": 2.5, "type": "dashed"},
            "itemStyle": {"color": "#FD79A8"},
            "symbol": "diamond",
            "symbolSize": 8,
        })

        # ── 置信区间 band (stack trick) ──
        lower_full = [None] * len(weeks) + fc["lower"].tolist()
        band_height = [None] * len(weeks) + (fc["upper"] - fc["lower"]).tolist()

        series.append({
            "name": "置信下界",
            "type": "line",
            "data": lower_full,
            "stack": "conf",
            "lineStyle": {"opacity": 0},
            "symbol": "none",
            "smooth": True,
        })
        series.append({
            "name": "置信区间",
            "type": "line",
            "data": band_height,
            "stack": "conf",
            "lineStyle": {"opacity": 0},
            "areaStyle": {"color": "rgba(253,121,168,0.12)"},
            "symbol": "none",
            "smooth": True,
        })

        # ── 预测起点标记线 ──
        if show_markline:
            series[0]["markLine"] = {
                "silent": True,
                "symbol": "none",
                "lineStyle": {"color": "#999", "type": "dashed"},
                "label": {"formatter": "预测起点"},
                "data": [{"xAxis": len(weeks) - 0.5}],
            }

        opts = {
            "title": {"text": f"商品 {item_id} 销量趋势与预测", "left": "center", "textStyle": {"fontSize": 14}},
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0, "data": ["历史销量", "预测销量"]},
            "xAxis": {"type": "category", "data": all_weeks, "name": "周序号"},
            "yAxis": {"type": "value", "name": "销量(件)"},
            "dataZoom": [
                {"type": "inside", "start": 0, "end": 100},
                {"type": "slider", "start": 0, "end": 100, "height": 20, "bottom": 30},
            ],
            "grid": {"bottom": "20%", "top": "12%", "left": "8%", "right": "4%"},
            "series": series,
        }
    else:
        opts = {
            "title": {"text": f"商品 {item_id} 销量趋势", "left": "center", "textStyle": {"fontSize": 14}},
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0},
            "xAxis": {"type": "category", "data": weeks, "name": "周序号"},
            "yAxis": {"type": "value", "name": "销量(件)"},
            "dataZoom": [
                {"type": "inside", "start": 0, "end": 100},
                {"type": "slider", "start": 0, "end": 100, "height": 20, "bottom": 30},
            ],
            "grid": {"bottom": "20%", "top": "12%", "left": "8%", "right": "4%"},
            "series": series,
        }

    st_echarts(options=opts, height=height, key=f"trend_{item_id}", theme="streamlit")
