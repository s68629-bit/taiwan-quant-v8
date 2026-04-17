"""
反向操作訊號引擎（v5 全新核心模組）
═══════════════════════════════════════════════════════════════
外資操作台股的核心優勢：識別散戶的集體行為偏誤，站在對面

三大反向訊號：

1. retail_heat_score（散戶過熱分數）[0~1]
   高分 = 散戶情緒過熱、槓桿過高 = 外資出貨窗口
   來源：融資增速 + RSI 過熱 + 量比異常 + 接近 52 週高點

2. foreign_distribution_signal（外資分配訊號）[-1~+1]
   +1 = 外資明確累積  -1 = 外資邊漲邊出貨（最危險）
   來源：外資淨買超方向 vs 近期股價方向

3. followthrough_trap（散戶跟風陷阱訊號）[-1~0]
   外資早買 → 價格上漲 → 散戶現在追進 → 外資開始出貨
   這是台股最經典的「主力出貨」模式
═══════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_contrarian_score(df):
    """
    計算單支股票的反向操作綜合分數。

    輸入：已整合技術 + 籌碼因子的 DataFrame（最後一行為最新資料）
    輸出：{
        'contrarian_score': float [-1~1],  # 正值=適合做多，負值=偏空警示
        'retail_heat':      float [0~1],   # 散戶過熱程度
        'foreign_dist':     float [-1~1],  # 外資分配訊號
        'trap_signal':      float [-1~0],  # 跟風陷阱嚴重度
        'warning':          str            # 文字警示說明
    }
    """
    if df.empty or len(df) < 30:
        return _empty_result("資料不足")

    latest = df.iloc[-1]
    recent = df.iloc[-20:]   # 近 20 日用於趨勢判斷

    # ── 1. 散戶過熱分數 ──────────────────────────────────────
    heat = _retail_heat(latest, recent, df)

    # ── 2. 外資分配訊號 ──────────────────────────────────────
    f_dist = _foreign_distribution(latest, recent)

    # ── 3. 跟風陷阱偵測 ──────────────────────────────────────
    trap = _followthrough_trap(df)

    # ── 合成反向分數 ─────────────────────────────────────────
    # 反向分數邏輯：
    #   散戶越熱 → 分數越低（對做多越不利）
    #   外資累積 → 分數越高
    #   陷阱訊號 → 大幅拉低分數
    contrarian = (
        -0.40 * heat          # 散戶過熱是最主要的賣出訊號
        + 0.40 * f_dist       # 外資方向最重要
        + 0.20 * trap         # 陷阱訊號加重懲罰
    )
    contrarian = float(np.clip(contrarian, -1, 1))

    # ── 文字警示 ─────────────────────────────────────────────
    warning = _generate_warning(heat, f_dist, trap)

    return {
        "contrarian_score": round(contrarian, 4),
        "retail_heat":      round(float(heat), 4),
        "foreign_dist":     round(float(f_dist), 4),
        "trap_signal":      round(float(trap), 4),
        "warning":          warning,
    }


# ── 子模組 ────────────────────────────────────────────────────

def _retail_heat(latest, recent, df):
    """
    散戶過熱指數 [0~1]
    高分代表散戶過熱，是外資出貨的最佳時機
    """
    scores = []

    # 融資增速（最重要訊號）
    margin_chg = latest.get("margin_chg_5d", 0)
    if pd.notna(margin_chg):
        # 融資 5 日增加 >10% 視為過熱
        margin_heat = float(np.clip(margin_chg / 0.10, 0, 1))
        scores.append(("margin", 0.35, margin_heat))

    # RSI 過熱（>70 為超買區，散戶追高）
    rsi = latest.get("RSI14", 50)
    if pd.notna(rsi):
        rsi_heat = float(np.clip((rsi - 70) / 20, 0, 1))   # 70→0, 90→1
        scores.append(("rsi", 0.25, rsi_heat))

    # 成交量暴量（散戶衝進來）
    vol_ratio = latest.get("VolumeRatio", 1)
    if pd.notna(vol_ratio):
        vol_heat = float(np.clip((vol_ratio - 1.5) / 1.5, 0, 1))
        scores.append(("volume", 0.20, vol_heat))

    # 接近 52 週高點（散戶 FOMO 追高）
    dist = latest.get("Dist_52W_High", -0.5)
    if pd.notna(dist):
        # dist = 0 表示剛好在高點，-0.05 表示距高點 5%
        high_heat = float(np.clip(1 + dist / 0.05, 0, 1))
        scores.append(("high", 0.20, high_heat))

    if not scores:
        return 0.5   # 無法判斷，回傳中性值

    total_w = sum(w for _, w, _ in scores)
    heat = sum(w * s for _, w, s in scores) / total_w
    return float(np.clip(heat, 0, 1))


def _foreign_distribution(latest, recent):
    """
    外資分配訊號 [-1~+1]
    +1 = 外資持續買進（籌碼集中）
    -1 = 外資邊漲邊賣出（最危險的出貨型態）
    """
    f_net_20d = latest.get("foreign_net_20d", 0)
    consec    = latest.get("foreign_consec_buy", 0)
    pv        = latest.get("PV_diverge", 0)

    if pd.isna(f_net_20d): f_net_20d = 0
    if pd.isna(consec):    consec    = 0
    if pd.isna(pv):        pv        = 0

    # 正規化外資 20 日淨買超（以 ±10000 張為滿分）
    f_norm = float(np.clip(f_net_20d / 10000, -1, 1))

    # 連續買超天數（>10 天 = 強力積累）
    consec_norm = float(np.clip(consec / 10, 0, 1))

    # 量價背離懲罰（價漲量縮 + 外資賣出 = 出貨）
    pv_adj = float(pv) * (-0.3) if (pv < 0 and f_norm < 0) else 0

    signal = 0.50 * f_norm + 0.30 * consec_norm + 0.20 * pv_adj
    return float(np.clip(signal, -1, 1))


def _followthrough_trap(df):
    """
    跟風陷阱偵測 [-1~0]
    -1 = 高度確認的陷阱（外資已買夠，散戶剛追進，外資準備出場）
     0 = 無陷阱訊號
    """
    if len(df) < 25:
        return 0.0

    # 外資 20 天前的買超強度（正值 = 當時外資在買）
    past_foreign = df["foreign_net"].iloc[-25:-5].sum() \
                   if "foreign_net" in df.columns else 0

    # 近 5 天外資轉向賣出
    recent_foreign = df["foreign_net"].iloc[-5:].sum() \
                     if "foreign_net" in df.columns else 0

    # 近 20 天價格上漲（散戶已被吸引）
    if len(df) >= 20:
        price_chg = float(
            df["Close"].iloc[-1] / df["Close"].iloc[-20] - 1
        )
    else:
        price_chg = 0

    # 融資近 10 天增加（散戶加碼進場）
    margin_recent = df["margin_chg_5d"].iloc[-1] \
                    if "margin_chg_5d" in df.columns else 0
    if pd.isna(margin_recent): margin_recent = 0

    # 陷阱條件：過去外資買 + 近期外資賣 + 股價漲 + 散戶融資增
    trap_strength = 0.0
    if past_foreign > 5000:    trap_strength -= 0.25   # 外資早已建倉
    if recent_foreign < -2000: trap_strength -= 0.35   # 外資近期出場
    if price_chg > 0.08:       trap_strength -= 0.20   # 漲幅吸引散戶
    if margin_recent > 0.05:   trap_strength -= 0.20   # 散戶融資追進

    return float(np.clip(trap_strength, -1, 0))


def _generate_warning(heat, f_dist, trap):
    warnings = []
    if heat > 0.70:
        warnings.append("🔴 散戶嚴重過熱（融資暴增/追高）")
    elif heat > 0.45:
        warnings.append("🟡 散戶中度過熱，需留意")
    if f_dist < -0.30:
        warnings.append("🔴 外資疑似出貨（邊漲邊賣）")
    if trap < -0.50:
        warnings.append("🔴 高確信跟風陷阱，散戶剛追進外資已出場")
    elif trap < -0.25:
        warnings.append("🟡 疑似陷阱結構，外資動向須觀察")
    if not warnings:
        if heat < 0.20 and f_dist > 0.20:
            warnings.append("🟢 籌碼健康，外資持續累積")
        else:
            warnings.append("⚪ 無明顯異常訊號")
    return " | ".join(warnings)


def _empty_result(reason):
    return {
        "contrarian_score": 0.0,
        "retail_heat":      0.5,
        "foreign_dist":     0.0,
        "trap_signal":      0.0,
        "warning":          f"無法計算（{reason}）",
    }
