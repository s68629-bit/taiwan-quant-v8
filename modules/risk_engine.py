"""
風控引擎（v5 新增）
提供：停損判斷、部位建議、集中度警示

注意：本模組為「提示工具」，最終決策由使用者自行判斷
"""
import pandas as pd
import logging
from config.settings import (STOP_LOSS_RATE, MAX_POSITION_PCT,
                              PORTFOLIO_SIZE, HOLD_DAYS)

logger = logging.getLogger(__name__)


def check_stop_loss(entry_price, current_price):
    """
    檢查是否觸及停損線。
    回傳 {triggered: bool, return_rate: float, action: str}
    """
    r = (current_price - entry_price) / entry_price
    triggered = r <= STOP_LOSS_RATE
    return {
        "triggered":   triggered,
        "return_rate": round(r, 4),
        "action":      "⛔ 建議停損出場" if triggered else "✅ 持倉正常",
    }


def suggest_position_size(composite_score, portfolio_value, n_positions=None):
    """
    根據綜合評分建議單股部位大小。
    信心越高 → 部位越大（但不超過 MAX_POSITION_PCT）。
    """
    if n_positions is None:
        n_positions = PORTFOLIO_SIZE

    # 等權基礎部位
    base_weight = 1.0 / n_positions

    # 根據信心調整（±20%）
    confidence = abs(composite_score)
    adj = base_weight * (0.8 + 0.4 * confidence)   # 0.8~1.2 倍
    adj = min(adj, MAX_POSITION_PCT)                # 不超過上限

    suggested_value = portfolio_value * adj

    return {
        "weight_pct":      round(adj * 100, 1),
        "suggested_value": round(suggested_value, 0),
        "max_position":    round(portfolio_value * MAX_POSITION_PCT, 0),
    }


def check_concentration(holdings):
    """
    檢查持倉集中度。
    holdings: {symbol: weight} 例如 {"2330.TW": 0.30, "2317.TW": 0.25}
    """
    warnings = []
    for sym, w in holdings.items():
        if w > MAX_POSITION_PCT:
            warnings.append(f"⚠️ {sym} 部位 {w:.0%} 超過上限 {MAX_POSITION_PCT:.0%}")

    # 產業集中度（若前三大超過 70%）
    top3 = sum(sorted(holdings.values(), reverse=True)[:3])
    if top3 > 0.70:
        warnings.append(f"⚠️ 前三大持股合計 {top3:.0%}，集中度偏高")

    return warnings if warnings else ["✅ 持倉集中度正常"]


def generate_risk_summary(scan_results, market_regime):
    """
    產生整體風控摘要（供 Dashboard 和報告使用）。
    scan_results: [(symbol, composite_dict), ...]
    """
    summary = {
        "market_regime":   market_regime.get("regime", "未知"),
        "is_bear":         market_regime.get("is_bear", False),
        "long_signals":    0,
        "short_warnings":  0,
        "high_heat_count": 0,
        "advice":          [],
    }

    for sym, cdict in scan_results:
        if cdict.get("signal") in ("做多","強力做多"):
            summary["long_signals"] += 1
        if cdict.get("signal") == "偏空警示":
            summary["short_warnings"] += 1
        if cdict.get("retail_heat", 0) > 0.60:
            summary["high_heat_count"] += 1

    # 整體建議
    if summary["is_bear"]:
        summary["advice"].append("🔴 大盤空頭環境，建議降低總部位至 50% 以下")
    if summary["short_warnings"] > len(scan_results) * 0.5:
        summary["advice"].append("🔴 超過半數股票出現偏空警示，市場可能過熱")
    if summary["high_heat_count"] > 3:
        summary["advice"].append(f"🟡 {summary['high_heat_count']} 支股票散戶過熱，注意追高風險")
    if not summary["advice"]:
        summary["advice"].append("🟢 整體風控正常，可依訊號操作")

    return summary
