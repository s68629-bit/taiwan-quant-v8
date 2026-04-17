"""
四合一綜合評分引擎 v6
新增 W_SURGE（飆漲前兆）維度
"""
import numpy as np
import logging
from config.settings import (W_AI, W_CHIP, W_CONTRARIAN, W_SURGE,
                              LONG_THRESHOLD, WATCH_THRESHOLD, SHORT_THRESHOLD,
                              BEAR_MARKET_MULTIPLIER, BEAR_LONG_THRESHOLD)

logger = logging.getLogger(__name__)

def compute_composite(symbol, ai_pred, chip_latest, contrarian_result,
                      is_bear_market=False, surge_result=None):
    ai_score   = float(np.clip(ai_pred / 0.05, -1, 1))
    chip_score = _chip_score(chip_latest)
    cont_score = float(contrarian_result.get("contrarian_score", 0))

    # 飆漲分數正規化 [-1~1]（分數 50 以上才有正貢獻）
    surge_score_raw = float((surge_result or {}).get("surge_score", 0))
    surge_score = float(np.clip((surge_score_raw - 50) / 50, -0.5, 1))

    raw = (W_AI * ai_score + W_CHIP * chip_score +
           W_CONTRARIAN * cont_score + W_SURGE * surge_score)

    bear_adjusted = False
    if is_bear_market:
        raw *= BEAR_MARKET_MULTIPLIER
        bear_adjusted = True

    composite = float(np.clip(raw, -1, 1))
    long_thr  = BEAR_LONG_THRESHOLD if is_bear_market else LONG_THRESHOLD
    signal, icon = _signal(composite, long_thr)

    breakdown = {
        "AI貢獻":   round(W_AI * ai_score, 4),
        "籌碼貢獻": round(W_CHIP * chip_score, 4),
        "反向貢獻": round(W_CONTRARIAN * cont_score, 4),
        "飆漲貢獻": round(W_SURGE * surge_score, 4),
    }

    logger.debug("%s → AI:%+.2f 籌碼:%+.2f 反向:%+.2f 飆漲:%+.2f → %+.2f [%s]",
                 symbol, ai_score, chip_score, cont_score, surge_score, composite, signal)

    return {
        "composite":        round(composite, 4),
        "ai_score":         round(ai_score, 4),
        "chip_score":       round(chip_score, 4),
        "contrarian_score": round(cont_score, 4),
        "surge_score_norm": round(surge_score, 4),
        "surge_score_raw":  int(surge_score_raw),
        "signal":           signal,
        "signal_icon":      icon,
        "confidence":       round(abs(composite), 4),
        "bear_adjusted":    bear_adjusted,
        "breakdown":        breakdown,
        "retail_heat":      contrarian_result.get("retail_heat", 0),
        "warning":          contrarian_result.get("warning", ""),
    }

def _chip_score(chip_latest):
    if chip_latest is None: return 0.0
    f20 = chip_latest.get("foreign_net_20d", 0) or 0
    t5  = chip_latest.get("trust_net_5d", 0)    or 0
    return float(np.clip(0.70 * np.clip(f20/20000,-1,1) +
                          0.30 * np.clip(t5/2000, -1,1), -1, 1))

def _signal(composite, long_threshold):
    if composite > 0.40:              return "強力做多", "★★"
    elif composite > long_threshold:  return "做多",     "★"
    elif composite > WATCH_THRESHOLD: return "觀察",     "👁"
    elif composite > SHORT_THRESHOLD: return "觀望",     "⚪"
    else:                             return "偏空警示", "⚠️"
