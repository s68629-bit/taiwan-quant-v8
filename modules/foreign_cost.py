"""
外資成本線計算引擎（v6 新增）
═══════════════════════════════════════════════════════════════
外資成本線的意義：
  外資在「有淨買超」的交易日，按成交量加權計算的平均成本

  與 VWAP 的差異：
    VWAP   = 所有人的成交量加權均價（不分買賣方向）
    外資成本線 = 只計算外資「有在買」的日子的加權均價

使用方式：
  股價 > 外資成本線 → 外資持倉獲利 → 可能繼續護盤拉抬
  股價 ≈ 外資成本線 → 外資成本撐盤區 → 容易出現反彈
  股價 < 外資成本線 → 外資被套 → 觀察是否持續加碼或反手

輸出：
  cost_60d   : 近 60 日外資加權成本
  cost_20d   : 近 20 日外資加權成本（短期成本）
  gap_to_cost: 股價與外資成本的差距 %
  position   : 股價相對外資成本的位置
  support_zone: 估算外資撐盤區間
═══════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_foreign_cost(price_df, inst_df):
    """
    計算外資成本線。

    參數：
        price_df - 價格 DataFrame（需含 Close, Volume）
        inst_df  - 三大法人 DataFrame（需含 foreign_net）

    回傳：dict
    """
    if price_df is None or price_df.empty:
        return _empty_result("無價格資料")

    current_price = float(price_df["Close"].iloc[-1])

    # ── 計算各窗口外資成本線 ──────────────────────────────────
    cost_20d = _calc_cost(price_df, inst_df, 20)
    cost_60d = _calc_cost(price_df, inst_df, 60)
    cost_all = _calc_cost(price_df, inst_df, len(price_df))

    # ── 主要參考：60 日外資成本 ───────────────────────────────
    main_cost = cost_60d if cost_60d else cost_all

    if not main_cost:
        return _empty_result("外資買超資料不足")

    gap = (current_price - main_cost) / main_cost

    # ── 位置判斷 ──────────────────────────────────────────────
    position, position_label = _position_label(gap)

    # ── 撐盤區間估算（成本線 ± 3%） ──────────────────────────
    support_low  = round(main_cost * 0.97, 1)
    support_high = round(main_cost * 1.03, 1)

    # ── 操作意涵 ──────────────────────────────────────────────
    implication = _get_implication(gap, position)

    return {
        "cost_20d":      round(cost_20d, 1) if cost_20d else None,
        "cost_60d":      round(cost_60d, 1) if cost_60d else None,
        "cost_all":      round(cost_all, 1) if cost_all else None,
        "current_price": round(current_price, 1),
        "gap_pct":       round(gap * 100, 2),
        "gap_label":     f"{gap:+.1%}",
        "position":      position,
        "position_label":position_label,
        "support_low":   support_low,
        "support_high":  support_high,
        "implication":   implication,
    }


def _calc_cost(price_df, inst_df, window):
    """
    計算指定窗口內的外資加權成本。
    只用外資有淨買超的交易日計算。
    """
    if price_df is None or price_df.empty:
        return None

    price_w = price_df.iloc[-window:].copy() if len(price_df) >= window \
              else price_df.copy()

    # 如果沒有外資資料，退回用 VWAP 近似
    if inst_df is None or inst_df.empty or "foreign_net" not in inst_df.columns:
        # 退回：全部交易量加權均價（VWAP 近似）
        if "Volume" in price_w.columns:
            total_value  = (price_w["Close"] * price_w["Volume"]).sum()
            total_volume = price_w["Volume"].sum()
            if total_volume > 0:
                return float(total_value / total_volume)
        return None

    # 對齊日期
    merged = price_w.join(
        inst_df[["foreign_net"]].reindex(price_w.index), how="left"
    )
    merged["foreign_net"] = merged["foreign_net"].fillna(0)

    # 只取外資淨買超的日子
    buy_days = merged[merged["foreign_net"] > 0]

    if buy_days.empty:
        # 無外資買超 → 退回全期 VWAP
        if "Volume" in merged.columns:
            total_value  = (merged["Close"] * merged["Volume"]).sum()
            total_volume = merged["Volume"].sum()
            return float(total_value / total_volume) if total_volume > 0 else None
        return None

    # 以外資買超張數為權重計算加權均價
    # 張數 = 外資買超的「相對大小」，乘以成交量估算外資實際買量
    weights = buy_days["foreign_net"].clip(lower=0)
    weighted_price = (buy_days["Close"] * weights).sum() / weights.sum()
    return float(weighted_price)


def _position_label(gap):
    """股價相對外資成本的位置判斷"""
    if gap > 0.20:
        return "far_above", "🔴 大幅超出外資成本（漲多風險高）"
    elif gap > 0.10:
        return "above",     "🟡 高於外資成本 10%+ （持倉中）"
    elif gap > 0.03:
        return "near_above","🟢 略高於外資成本（健康區間）"
    elif gap > -0.03:
        return "at_cost",   "🟢 接近外資成本線（撐盤區）"
    elif gap > -0.10:
        return "near_below","🟡 略低於外資成本（外資輕微被套）"
    else:
        return "far_below", "🔴 大幅低於外資成本（外資深套或出場）"


def _get_implication(gap, position):
    implications = {
        "far_above":  "外資早期建倉已大幅獲利，可能面臨部分出場壓力，追高需謹慎",
        "above":      "外資持倉獲利，持續護盤可能性高，但漲幅已大",
        "near_above": "股價在外資成本上方健康區間，外資有動力繼續拉抬",
        "at_cost":    "★ 接近外資成本線，這是最佳進場甜蜜點——外資在此撐盤且即將護盤拉抬",
        "near_below": "外資小幅被套，觀察是否持續加碼，若加碼則是更低成本進場機會",
        "far_below":  "外資深度被套，需觀察是否持續加碼（逢低建倉）或反手出場",
    }
    return implications.get(position, "無法判斷")


def _empty_result(reason):
    return {
        "cost_20d":       None,
        "cost_60d":       None,
        "cost_all":       None,
        "current_price":  None,
        "gap_pct":        None,
        "gap_label":      "N/A",
        "position":       "unknown",
        "position_label": f"無法計算（{reason}）",
        "support_low":    None,
        "support_high":   None,
        "implication":    reason,
    }
