"""
飆漲前兆偵測引擎（v6 全新核心模組）
═══════════════════════════════════════════════════════════════
外資操盤的飆漲三階段模型：

第一階段（吸籌期）：
  - 股價橫盤整理，波動率收縮
  - 外資持續小量買進，散戶不注意
  - 成交量呈現「量縮」趨勢
  → 偵測：吸籌結構確認分數

第二階段（臨界點）：
  - 吸籌接近完成，浮額愈來愈少
  - 成交量開始溫和放大（每日小幅遞增）
  - 外資買超力道加重
  → 偵測：量能臨界點分數

第三階段（啟動訊號）：
  - 突破整理區間 + 放量
  - 外資大量買進
  - 散戶追進（此時外資已獲利）
  → 偵測：啟動確認分數

輸出：
  surge_score       [0~100]  綜合飆漲分數
  stage             [1/2/3]  目前所處階段
  est_days          估計幾天後可能啟動（只在第一二階段輸出）
  volume_quality    量能品質評估
  accumulation_ok   吸籌結構是否確認
  breakout_risk     突破失敗風險
  signals           文字訊號清單
═══════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
import logging
from config.settings import (
    ACCUMULATION_DAYS, VOLUME_SQUEEZE_THRESHOLD,
    VOLUME_EXPAND_THRESHOLD, VOLUME_EXPLOSION, SURGE_DAYS_MAX
)

logger = logging.getLogger(__name__)


def detect_surge(df):
    """
    主入口：輸入含技術指標+籌碼因子的完整 DataFrame
    回傳 dict，包含飆漲前兆分析結果
    """
    if df is None or len(df) < max(ACCUMULATION_DAYS, 60):
        return _empty_result("資料不足")

    latest  = df.iloc[-1]
    recent  = df.iloc[-ACCUMULATION_DAYS:]
    prev60  = df.iloc[-60:]

    # ── 子模組計算 ──────────────────────────────────────────
    accum   = _accumulation_score(recent, df)
    vol_q   = _volume_quality(recent, df)
    breakout= _breakout_proximity(latest, df)
    chip_s  = _chip_accumulation(recent)
    retail_s= _retail_absence(recent)

    # ── 飆漲綜合分數 ─────────────────────────────────────────
    # 各項加權：吸籌 35% + 量能品質 25% + 突破臨近 20% + 籌碼 15% + 散戶缺席 5%
    raw_score = (
        accum["score"]    * 0.35 +
        vol_q["score"]    * 0.25 +
        breakout["score"] * 0.20 +
        chip_s["score"]   * 0.15 +
        retail_s["score"] * 0.05
    )
    surge_score = int(np.clip(raw_score * 100, 0, 100))

    # ── 階段判斷 ─────────────────────────────────────────────
    stage = _determine_stage(accum, vol_q, breakout)

    # ── 估計啟動天數 ─────────────────────────────────────────
    est_days = _estimate_days(surge_score, stage, vol_q)

    # ── 訊號清單 ─────────────────────────────────────────────
    signals = _build_signals(accum, vol_q, breakout, chip_s, surge_score)

    result = {
        "surge_score":      surge_score,
        "stage":            stage,
        "stage_label":      _stage_label(stage),
        "est_days":         est_days,
        "est_days_label":   _days_label(est_days, stage),
        "accumulation_ok":  accum["confirmed"],
        "volume_quality":   vol_q["label"],
        "volume_score":     vol_q["score"],
        "breakout_score":   breakout["score"],
        "chip_score":       chip_s["score"],
        "breakout_risk":    breakout["risk"],
        "signals":          signals,
        "score_breakdown": {
            "吸籌結構":  round(accum["score"], 3),
            "量能品質":  round(vol_q["score"], 3),
            "突破臨近":  round(breakout["score"], 3),
            "籌碼累積":  round(chip_s["score"], 3),
            "散戶缺席":  round(retail_s["score"], 3),
        }
    }

    logger.debug(
        "飆漲偵測 | 分數:%d | 階段:%s | 估計:%s | 量能:%s",
        surge_score, result["stage_label"],
        result["est_days_label"], vol_q["label"]
    )
    return result


# ── 子模組 ────────────────────────────────────────────────────

def _accumulation_score(recent, df):
    """
    吸籌結構評分
    核心邏輯：主力建倉期間 → 股價橫盤 + 低波動 + 外資持續買
    """
    scores = []

    # 1. 波動率收縮（標準差低於歷史中位數 = 橫盤整理）
    if "Volatility" in recent.columns:
        hist_vol  = df["Volatility"].dropna()
        curr_vol  = recent["Volatility"].mean()
        vol_pct   = float((hist_vol < curr_vol).mean())  # 歷史百分位
        vol_score = 1 - vol_pct   # 越低百分位分數越高（越安靜越好）
        scores.append(("vol_squeeze", 0.35, vol_score))

    # 2. 價格整理（近期最高點/最低點的振幅小）
    if "Close" in recent.columns and len(recent) > 5:
        price_range = (recent["Close"].max() - recent["Close"].min()) / recent["Close"].mean()
        range_score = max(0, 1 - price_range / 0.10)  # 10% 振幅以內滿分
        scores.append(("price_range", 0.30, range_score))

    # 3. 外資連續買超（最重要的吸籌訊號）
    if "foreign_consec_buy" in recent.columns:
        consec = float(recent["foreign_consec_buy"].iloc[-1] or 0)
        consec_score = min(consec / 10, 1.0)   # 連買 10 天 = 滿分
        scores.append(("consec_buy", 0.35, consec_score))

    if not scores:
        return {"score": 0.3, "confirmed": False}

    total_w = sum(w for _, w, _ in scores)
    score   = sum(w * s for _, w, s in scores) / total_w
    confirmed = score > 0.55

    return {"score": float(score), "confirmed": confirmed}


def _volume_quality(recent, df):
    """
    量能品質分析
    區分：量縮整理 / 溫和放大 / 暴量追高
    """
    if "VolumeRatio" not in recent.columns or "Volume" not in df.columns:
        return {"score": 0.3, "label": "無法判斷", "type": "unknown"}

    vol_ratios = recent["VolumeRatio"].dropna()
    if vol_ratios.empty:
        return {"score": 0.3, "label": "無法判斷", "type": "unknown"}

    avg_ratio = float(vol_ratios.mean())
    last_5    = recent["Volume"].iloc[-5:] if len(recent) >= 5 else recent["Volume"]

    # 判斷量能趨勢（最近 5 天是否每日遞增）
    is_expanding = False
    if len(last_5) >= 3:
        diffs = last_5.diff().dropna()
        positive_days = (diffs > 0).sum()
        is_expanding = positive_days >= 2   # 5 天中至少 2 天遞增

    # 量能類型判斷
    if avg_ratio > VOLUME_EXPLOSION:
        vol_type  = "暴量"
        label     = "🔴 暴量（散戶追高，風險高）"
        score     = 0.10   # 暴量不是好時機
    elif avg_ratio > VOLUME_EXPAND_THRESHOLD:
        if is_expanding:
            vol_type = "溫和放大"
            label    = "🟢 溫和放大（量能健康啟動）"
            score    = 0.90   # 最理想的入場前量能
        else:
            vol_type = "間歇放量"
            label    = "🟡 量能不穩定"
            score    = 0.50
    elif avg_ratio < VOLUME_SQUEEZE_THRESHOLD:
        vol_type = "量縮"
        label    = "🟡 量縮整理（等待突破方向）"
        score    = 0.65   # 吸籌期的量縮是好事
    else:
        vol_type = "正常"
        label    = "⚪ 量能正常"
        score    = 0.40

    return {
        "score":       float(score),
        "label":       label,
        "type":        vol_type,
        "avg_ratio":   round(avg_ratio, 2),
        "is_expanding":is_expanding,
    }


def _breakout_proximity(latest, df):
    """
    突破臨近程度
    計算股價距離整理區間上緣的距離
    """
    result = {"score": 0.3, "risk": "中", "distance": None}

    if "Close" not in df.columns or len(df) < 20:
        return result

    # 整理區間：用近 40 天的最高點
    recent40 = df["Close"].iloc[-40:]
    resistance = float(recent40.max())
    current    = float(latest.get("Close", 0) or 0)

    if current <= 0 or resistance <= 0:
        return result

    distance_pct = (resistance - current) / current

    # 距離越小 = 越接近突破點
    if distance_pct < 0.02:      # 距高點 2% 以內
        score = 0.95
        risk  = "低（即將突破）"
    elif distance_pct < 0.05:    # 距高點 5% 以內
        score = 0.75
        risk  = "低"
    elif distance_pct < 0.10:    # 距高點 10% 以內
        score = 0.50
        risk  = "中"
    else:
        score = 0.20
        risk  = "高（距突破點遠）"

    result.update({
        "score":        float(score),
        "risk":         risk,
        "distance":     round(distance_pct * 100, 1),
        "resistance":   round(resistance, 2),
    })
    return result


def _chip_accumulation(recent):
    """
    籌碼累積強度
    外資 20 日淨買超 + 投信動向
    """
    score = 0.3

    if "foreign_net_20d" in recent.columns:
        f20 = float(recent["foreign_net_20d"].iloc[-1] or 0)
        # 20000 張以上買超 = 滿分
        score += min(f20 / 20000, 1.0) * 0.60

    if "trust_net_5d" in recent.columns:
        t5 = float(recent["trust_net_5d"].iloc[-1] or 0)
        score += min(max(t5 / 2000, 0), 1.0) * 0.40

    return {"score": min(float(score), 1.0)}


def _retail_absence(recent):
    """
    散戶缺席指數
    散戶沒有大量進場 = 好事（主力還在悄悄建倉）
    """
    score = 0.7   # 預設中性偏好

    if "margin_chg_5d" in recent.columns:
        margin_chg = float(recent["margin_chg_5d"].iloc[-1] or 0)
        if margin_chg > 0.10:    # 融資大增 = 散戶追進
            score = 0.1
        elif margin_chg > 0.05:
            score = 0.4
        else:
            score = 0.8   # 融資沒增加 = 散戶沒追

    return {"score": float(score)}


# ── 輔助函式 ──────────────────────────────────────────────────

def _determine_stage(accum, vol_q, breakout):
    """判斷目前處於飆漲三階段中的哪一階段"""
    if breakout["score"] > 0.80:
        return 3   # 啟動訊號
    elif accum["confirmed"] and vol_q["type"] in ("溫和放大", "量縮"):
        return 2   # 臨界點
    else:
        return 1   # 吸籌期


def _stage_label(stage):
    return {1: "🔴 吸籌期", 2: "🟡 臨界點", 3: "🟢 啟動訊號"}.get(stage, "未知")


def _estimate_days(surge_score, stage, vol_q):
    """估計幾天後可能啟動"""
    if stage == 3:
        return 0
    elif stage == 2:
        base = max(3, (100 - surge_score) // 5)
        if vol_q["type"] == "溫和放大":
            base = max(1, base - 3)
        return min(base, SURGE_DAYS_MAX)
    else:  # stage 1
        base = max(10, (100 - surge_score) // 3)
        return min(base, SURGE_DAYS_MAX)


def _days_label(days, stage):
    if stage == 3:
        return "🟢 已啟動或即將突破"
    elif days <= 3:
        return f"🟡 約 {days} 天內可能啟動"
    elif days <= 7:
        return f"⚪ 約 {days} 天後（1週內）"
    else:
        return f"⚪ 約 {days} 天後（持續觀察）"


def _build_signals(accum, vol_q, breakout, chip_s, score):
    signals = []
    if accum["confirmed"]:
        signals.append("✅ 吸籌結構確認（橫盤低波動+外資持續買）")
    if vol_q["type"] == "溫和放大":
        signals.append("✅ 量能溫和放大（最理想入場前型態）")
    elif vol_q["type"] == "量縮":
        signals.append("🟡 量縮整理中（等待放量突破）")
    elif vol_q["type"] == "暴量":
        signals.append("🔴 暴量警告（散戶已大量追進，風險高）")
    if breakout["score"] > 0.75:
        signals.append(f"✅ 接近突破壓力區（距高點 {breakout.get('distance',0):.1f}%）")
    if chip_s["score"] > 0.70:
        signals.append("✅ 法人籌碼持續累積")
    if score >= 80:
        signals.append("🔥 飆漲分數極高，密切追蹤")
    elif score >= 60:
        signals.append("⚡ 飆漲結構成形，進入候選清單")
    if not signals:
        signals.append("⚪ 尚未出現明顯飆漲前兆")
    return signals


def _empty_result(reason):
    return {
        "surge_score":     0,
        "stage":           1,
        "stage_label":     "🔴 吸籌期",
        "est_days":        SURGE_DAYS_MAX,
        "est_days_label":  f"無法估計（{reason}）",
        "accumulation_ok": False,
        "volume_quality":  "資料不足",
        "volume_score":    0,
        "breakout_score":  0,
        "chip_score":      0,
        "breakout_risk":   "未知",
        "signals":         [f"⚪ 資料不足（{reason}）"],
        "score_breakdown": {},
    }
