"""
散戶情緒溫度計（v6 精準版）
═══════════════════════════════════════════════════════════════
v5 的散戶過熱是即時值（今天熱不熱）
v6 升級為「歷史百分位」：

  例如：RSI = 70，但這支股票歷史上 RSI 常到 85，
        代表現在其實還沒到過熱，不該觸發警示

  正確做法：把今天的值跟該股歷史分布比較，
            換算成 0~100 的百分位溫度計

四個維度：
  1. 融資使用率百分位（最重要）
  2. RSI 歷史百分位
  3. 成交量百分位（今日量是否異常）
  4. 動能過熱百分位（近期漲幅是否過度）

輸出：
  temperature      [0~100] 綜合散戶情緒溫度
  temp_label       文字說明
  components       各維度詳情
  extreme_signal   是否達到極端過熱（>80）
═══════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_retail_sentiment(df):
    """
    計算精準版散戶情緒溫度。
    輸入：含技術指標 + 籌碼因子的完整 DataFrame
    """
    if df is None or len(df) < 60:
        return _empty_result("資料不足")

    latest = df.iloc[-1]
    components = {}

    # ── 1. 融資使用率百分位 ──────────────────────────────────
    if "margin_ratio" in df.columns:
        series = df["margin_ratio"].dropna()
        if len(series) > 20:
            current  = float(series.iloc[-1] or 0)
            pct      = float((series < current).mean()) * 100
            label    = _pct_label(pct)
            components["margin"] = {
                "value":      round(current * 100, 1),
                "percentile": round(pct, 1),
                "label":      f"融資率歷史百分位 {pct:.0f}%：{label}",
                "weight":     0.35,
                "score":      pct,
            }

    # ── 2. RSI 歷史百分位 ─────────────────────────────────────
    if "RSI14" in df.columns:
        series  = df["RSI14"].dropna()
        current = float(series.iloc[-1] or 50)
        pct     = float((series < current).mean()) * 100
        label   = _pct_label(pct)
        components["rsi"] = {
            "value":      round(current, 1),
            "percentile": round(pct, 1),
            "label":      f"RSI {current:.1f}（歷史百分位 {pct:.0f}%）：{label}",
            "weight":     0.30,
            "score":      pct,
        }

    # ── 3. 成交量百分位 ───────────────────────────────────────
    if "VolumeRatio" in df.columns:
        series  = df["VolumeRatio"].dropna()
        current = float(series.iloc[-1] or 1)
        pct     = float((series < current).mean()) * 100
        label   = _pct_label(pct)
        components["volume"] = {
            "value":      round(current, 2),
            "percentile": round(pct, 1),
            "label":      f"量比 {current:.2f}（歷史百分位 {pct:.0f}%）：{label}",
            "weight":     0.20,
            "score":      pct,
        }

    # ── 4. 動能過熱百分位 ─────────────────────────────────────
    if "Momentum" in df.columns:
        series  = df["Momentum"].dropna()
        current = float(series.iloc[-1] or 0)
        pct     = float((series < current).mean()) * 100
        label   = _pct_label(pct)
        components["momentum"] = {
            "value":      f"{current:+.2%}",
            "percentile": round(pct, 1),
            "label":      f"20日動能 {current:+.1%}（歷史百分位 {pct:.0f}%）：{label}",
            "weight":     0.15,
            "score":      pct,
        }

    if not components:
        return _empty_result("無法計算任何維度")

    # ── 加權合成溫度 ──────────────────────────────────────────
    total_w = sum(c["weight"] for c in components.values())
    temperature = sum(
        c["weight"] * c["score"] for c in components.values()
    ) / total_w
    temperature = round(float(np.clip(temperature, 0, 100)), 1)

    extreme = temperature >= 80
    temp_label = _temp_label(temperature)

    return {
        "temperature":    temperature,
        "temp_label":     temp_label,
        "temp_bar":       _temp_bar(temperature),
        "extreme_signal": extreme,
        "components":     components,
        "warning":        _temp_warning(temperature, extreme),
    }


# ── 輔助 ──────────────────────────────────────────────────────

def _pct_label(pct):
    if pct >= 90:   return "🔴 極度偏高"
    elif pct >= 75: return "🟡 偏高"
    elif pct >= 50: return "⚪ 中等"
    elif pct >= 25: return "🟢 偏低"
    else:           return "🟢 極低（散戶冷淡）"

def _temp_label(temp):
    if temp >= 85:  return "🔴🔴 極度過熱（散戶瘋狂追高）"
    elif temp >= 70:return "🔴 明顯過熱"
    elif temp >= 55:return "🟡 偏熱，開始留意"
    elif temp >= 40:return "⚪ 正常"
    elif temp >= 25:return "🟢 偏冷（散戶興趣低）"
    else:           return "🟢🟢 極冷（主力悄悄建倉的好時機）"

def _temp_bar(temp):
    filled = int(temp / 10)
    return "█" * filled + "░" * (10 - filled) + f" {temp:.0f}°"

def _temp_warning(temp, extreme):
    if extreme:
        return "🔴 散戶極度過熱，這正是外資出貨的高危時段，建議暫緩進場"
    elif temp >= 70:
        return "🟡 散戶情緒偏熱，追高風險上升，需配合籌碼面確認"
    elif temp <= 25:
        return "🟢 散戶興趣低，主力可能正在悄悄建倉，是值得關注的底部區"
    else:
        return "⚪ 散戶情緒正常，無特別異常"

def _empty_result(reason):
    return {
        "temperature":    50,
        "temp_label":     f"無法計算（{reason}）",
        "temp_bar":       "░░░░░░░░░░ N/A",
        "extreme_signal": False,
        "components":     {},
        "warning":        reason,
    }
