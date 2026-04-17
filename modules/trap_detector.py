"""
陷阱偵測引擎（v7 全新核心模組）
═══════════════════════════════════════════════════════════════
整合六個缺口的修正：

  缺口①：外資成本線動態窗口（找本輪建倉起點）
  缺口②：借券餘額異常偵測（最早期倒貨預警）
  缺口③：假突破警示（誘多陷阱識別）
  缺口④：KD/RSI 高檔背離偵測（轉折點確認）
  缺口⑤：外資買賣轉折點（大量買後突然轉賣）
  缺口⑥：融資斷頭預警距離（踩踏風險）

輸出：TrapResult dict
  trap_score       [0~100] 陷阱嚴重度（越高越危險）
  trap_level       "安全" / "注意" / "警告" / "危險"
  fake_breakout    假突破分析
  short_sale       借券異常分析
  divergence       KD/RSI 背離分析
  foreign_reversal 外資轉折點分析
  margin_danger    融資斷頭預警
  accumulation_start 本輪建倉起點
  signals          警示訊號清單
  entry_ok         是否適合進場（綜合判斷）
═══════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
import logging
from config.settings import (
    SHORT_SALE_SURGE_THRESHOLD, FAKE_BREAKOUT_VOLUME_RATIO,
    MARGIN_CALL_BUFFER, DIVERGENCE_LOOKBACK
)

logger = logging.getLogger(__name__)


def detect_traps(df, inst_df=None, margin_df=None):
    """
    主入口：輸入完整 DataFrame，回傳陷阱偵測結果
    """
    if df is None or len(df) < 60:
        return _empty_result("資料不足")

    latest = df.iloc[-1]
    signals = []
    danger_score = 0

    # ── 缺口②：借券異常偵測（最早期預警）──────────────────────
    short_result = _detect_short_sale_surge(margin_df)
    if short_result["alert"]:
        danger_score += 35
        signals.append(f"🔴 借券餘額異常增加（{short_result['chg_5d']:+.1%}）→ 外資準備放空對沖")

    # ── 缺口③：假突破警示 ───────────────────────────────────────
    fake_result = _detect_fake_breakout(df, inst_df)
    if fake_result["alert"]:
        danger_score += 30
        signals.append(f"🔴 假突破警示：量比{fake_result['vol_ratio']:.1f}x + 外資賣出 → 誘多陷阱")

    # ── 缺口④：KD/RSI 高檔背離 ────────────────────────────────
    div_result = _detect_divergence(df)
    if div_result["top_divergence"]:
        danger_score += 20
        signals.append(f"🟡 {div_result['type']} 頂背離（股價創高但指標未創高）→ 轉折前兆")
    elif div_result["bottom_divergence"]:
        signals.append(f"🟢 {div_result['type']} 底背離（潛在反彈機會）")

    # ── 缺口⑤：外資買賣轉折點 ─────────────────────────────────
    rev_result = _detect_foreign_reversal(df)
    if rev_result["alert"]:
        danger_score += 25
        signals.append(f"🔴 外資買賣轉折：前期大買後近期轉賣 → 出貨警示")

    # ── 缺口⑥：融資斷頭預警 ────────────────────────────────────
    margin_result = _detect_margin_danger(df, margin_df)
    if margin_result["danger"]:
        danger_score += 15
        signals.append(f"🟡 融資斷頭距離 {margin_result['distance_pct']:.1f}% → 踩踏風險")

    # ── 缺口①：本輪建倉起點（輔助信息）───────────────────────
    accum_start = _find_accumulation_start(inst_df)

    # ── 综合判斷 ─────────────────────────────────────────────
    danger_score = min(danger_score, 100)
    trap_level   = _trap_level(danger_score)
    entry_ok     = danger_score < 30

    if not signals:
        signals.append("✅ 未偵測到明顯陷阱訊號，籌碼面相對安全")

    logger.debug("陷阱偵測 | 危險分:%d | 等級:%s | 進場:%s",
                 danger_score, trap_level, "✅" if entry_ok else "❌")

    return {
        "trap_score":        danger_score,
        "trap_level":        trap_level,
        "entry_ok":          entry_ok,
        "fake_breakout":     fake_result,
        "short_sale":        short_result,
        "divergence":        div_result,
        "foreign_reversal":  rev_result,
        "margin_danger":     margin_result,
        "accumulation_start":accum_start,
        "signals":           signals,
    }


# ── 子模組 ────────────────────────────────────────────────────

def _detect_short_sale_surge(margin_df):
    """缺口②：借券餘額異常增加偵測"""
    result = {"alert": False, "chg_5d": 0.0, "chg_20d": 0.0,
              "level": "正常", "description": "借券餘額無異常"}

    if margin_df is None or margin_df.empty:
        result["description"] = "無借券資料（需 FinMind）"
        return result

    if "short_balance" not in margin_df.columns:
        return result

    sb = margin_df["short_balance"].dropna()
    if len(sb) < 6:
        return result

    # 5 日借券增幅
    chg_5d = float(sb.iloc[-1] / sb.iloc[-6] - 1) if sb.iloc[-6] > 0 else 0
    # 20 日借券增幅
    chg_20d = float(sb.iloc[-1] / sb.iloc[-min(21, len(sb))] - 1) \
              if len(sb) >= 21 and sb.iloc[-21] > 0 else 0

    result["chg_5d"]  = chg_5d
    result["chg_20d"] = chg_20d

    if chg_5d > SHORT_SALE_SURGE_THRESHOLD:
        result["alert"] = True
        result["level"] = "🔴 異常暴增"
        result["description"] = f"借券 5 日增幅 {chg_5d:+.1%}，超過警戒線（外資準備放空對沖）"
    elif chg_5d > SHORT_SALE_SURGE_THRESHOLD * 0.6:
        result["level"] = "🟡 偏高"
        result["description"] = f"借券 5 日增幅 {chg_5d:+.1%}，需持續觀察"
    else:
        result["description"] = f"借券餘額穩定（5日變化 {chg_5d:+.1%}）✅"

    return result


def _detect_fake_breakout(df, inst_df):
    """缺口③：假突破偵測"""
    result = {"alert": False, "vol_ratio": 1.0,
              "at_high": False, "foreign_selling": False,
              "description": "無假突破跡象"}

    if len(df) < 20:
        return result

    latest = df.iloc[-1]

    # 是否在高點附近（距 40 日高點 2% 以內）
    high_40d = df["Close"].iloc[-40:].max() if len(df) >= 40 else df["Close"].max()
    current  = float(latest.get("Close", 0) or 0)
    at_high  = (current / high_40d) > 0.98 if high_40d > 0 else False

    # 成交量是否暴量（量比 > 2）
    vol_ratio = float(latest.get("VolumeRatio", 1) or 1)
    high_volume = vol_ratio > FAKE_BREAKOUT_VOLUME_RATIO

    # 外資是否在賣（recent foreign net < 0）
    foreign_selling = False
    if inst_df is not None and not inst_df.empty and "foreign_net" in inst_df.columns:
        recent_f = inst_df["foreign_net"].iloc[-5:].sum() if len(inst_df) >= 5 else 0
        foreign_selling = float(recent_f) < -1000

    result.update({"at_high": at_high, "vol_ratio": vol_ratio,
                   "foreign_selling": foreign_selling})

    # 假突破條件：在高點 + 暴量 + 外資賣出
    if at_high and high_volume and foreign_selling:
        result["alert"] = True
        result["description"] = (
            f"高點暴量（量比{vol_ratio:.1f}x）+ 外資賣出 → 高確信假突破，"
            "散戶追進外資出貨"
        )
    elif at_high and high_volume:
        result["description"] = f"高點暴量（量比{vol_ratio:.1f}x），注意外資動向"
    else:
        result["description"] = "無假突破跡象 ✅"

    return result


def _detect_divergence(df):
    """缺口④：KD/RSI 高檔背離偵測"""
    result = {
        "top_divergence":    False,
        "bottom_divergence": False,
        "type":              "",
        "description":       "無明顯背離",
    }

    if "RSI14" not in df.columns or len(df) < DIVERGENCE_LOOKBACK + 5:
        return result

    lookback = DIVERGENCE_LOOKBACK
    price_recent = df["Close"].iloc[-lookback:]
    rsi_recent   = df["RSI14"].iloc[-lookback:]

    price_prev   = df["Close"].iloc[-lookback*2:-lookback]
    rsi_prev     = df["RSI14"].iloc[-lookback*2:-lookback]

    if len(price_prev) < 5:
        return result

    # 頂背離：股價創新高，RSI 沒有創新高
    price_new_high = float(price_recent.max()) > float(price_prev.max())
    rsi_new_high   = float(rsi_recent.max())   > float(rsi_prev.max())

    if price_new_high and not rsi_new_high:
        result.update({
            "top_divergence": True,
            "type":           "RSI",
            "description":    (
                f"RSI 頂背離：股價創新高（{price_recent.max():.1f}）"
                f"但 RSI 未創新高（{rsi_recent.max():.1f}）→ 動能衰竭"
            )
        })
        return result

    # 底背離：股價創新低，RSI 沒有創新低
    price_new_low = float(price_recent.min()) < float(price_prev.min())
    rsi_new_low   = float(rsi_recent.min())   < float(rsi_prev.min())

    if price_new_low and not rsi_new_low:
        result.update({
            "bottom_divergence": True,
            "type":              "RSI",
            "description":       (
                f"RSI 底背離：股價創新低但 RSI 未創新低 → 潛在反彈"
            )
        })

    return result


def _detect_foreign_reversal(df):
    """缺口⑤：外資買賣轉折點偵測"""
    result = {"alert": False, "prev_buy": 0.0, "recent_sell": 0.0,
              "description": "外資方向無明顯轉折"}

    if "foreign_net" not in df.columns:
        return result

    fn = df["foreign_net"].fillna(0)
    if len(fn) < 25:
        return result

    # 前 20 天累計買超
    prev_buy    = float(fn.iloc[-25:-5].sum())
    # 近 5 天累計
    recent_net  = float(fn.iloc[-5:].sum())

    result["prev_buy"]    = prev_buy
    result["recent_sell"] = recent_net

    if prev_buy > 5000 and recent_net < -2000:
        result["alert"] = True
        result["description"] = (
            f"外資轉折：前期累計買超 {prev_buy:,.0f} 張，"
            f"近 5 日轉賣 {recent_net:,.0f} 張 → 出貨轉折點"
        )
    elif prev_buy > 3000 and recent_net < -1000:
        result["description"] = (
            f"外資動向轉弱：前期買超 {prev_buy:,.0f} 張，"
            f"近期轉為賣出，需持續觀察"
        )
    else:
        result["description"] = f"外資方向穩定 ✅（近5日：{recent_net:+,.0f}張）"

    return result


def _detect_margin_danger(df, margin_df):
    """缺口⑥：融資斷頭預警距離"""
    result = {"danger": False, "distance_pct": 99.0,
              "danger_price": None, "description": "融資斷頭風險低"}

    if "Close" not in df.columns:
        return result

    current = float(df["Close"].iloc[-1])

    # 估算融資平均成本（用近 60 日 VWAP 近似）
    if "VWAP" in df.columns:
        est_margin_cost = float(df["VWAP"].iloc[-1])
    else:
        est_margin_cost = float(df["Close"].rolling(60).mean().iloc[-1])

    if est_margin_cost <= 0:
        return result

    # 斷頭線 = 融資成本 × 85%（台股融資維持率 130%，折算約 85%）
    danger_price = est_margin_cost * MARGIN_CALL_BUFFER
    distance_pct = (current - danger_price) / current * 100

    result["danger_price"]   = round(danger_price, 1)
    result["distance_pct"]   = round(distance_pct, 1)

    if distance_pct < 5:
        result["danger"]      = True
        result["description"] = (
            f"⚠️ 融資斷頭距離僅 {distance_pct:.1f}%"
            f"（斷頭線約 {danger_price:.1f} 元）→ 踩踏風險極高"
        )
    elif distance_pct < 10:
        result["description"] = (
            f"融資斷頭距離 {distance_pct:.1f}%，偏低，需注意"
        )
    else:
        result["description"] = (
            f"融資斷頭距離 {distance_pct:.1f}%，安全 ✅"
        )

    return result


def _find_accumulation_start(inst_df):
    """缺口①：找本輪建倉起點（外資從賣轉買的日期）"""
    if inst_df is None or inst_df.empty:
        return {"date": None, "days_ago": None,
                "description": "無法判斷（缺籌碼資料）"}

    if "foreign_net" not in inst_df.columns:
        return {"date": None, "days_ago": None,
                "description": "無外資淨買超資料"}

    fn   = inst_df["foreign_net"].fillna(0)
    idx  = fn.index

    # 從最近往回找：最後一段連續買超的起點
    # 方法：找最近一次「連續賣後轉買」的轉折日
    is_buy = fn > 0
    start_idx = None

    for i in range(len(is_buy) - 1, 0, -1):
        if is_buy.iloc[i] and not is_buy.iloc[i-1]:
            start_idx = i
            break

    if start_idx is None:
        # 全程都在買或全程都在賣
        if is_buy.all():
            start_idx = 0
        else:
            return {"date": None, "days_ago": None,
                    "description": "外資持續賣出，未見建倉起點"}

    start_date = idx[start_idx]
    days_ago   = (pd.Timestamp.today() - start_date).days

    return {
        "date":        str(start_date.date()),
        "days_ago":    days_ago,
        "description": f"本輪建倉起點：{start_date.date()}（{days_ago} 天前）",
    }


def _trap_level(score):
    if score >= 70: return "🔴 危險"
    elif score >= 40: return "🟡 警告"
    elif score >= 20: return "🟡 注意"
    else:             return "🟢 安全"


def _empty_result(reason):
    return {
        "trap_score":        0,
        "trap_level":        "🟢 安全",
        "entry_ok":          True,
        "fake_breakout":     {"alert": False, "description": reason},
        "short_sale":        {"alert": False, "description": reason},
        "divergence":        {"top_divergence": False, "description": reason},
        "foreign_reversal":  {"alert": False, "description": reason},
        "margin_danger":     {"danger": False, "description": reason},
        "accumulation_start":{"date": None, "description": reason},
        "signals":           [f"⚪ 無法偵測（{reason}）"],
    }
