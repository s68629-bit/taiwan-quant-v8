"""
多時間窗口預測引擎（v6 新增）
同時輸出 20 / 40 / 60 個交易日的預測漲幅
═══════════════════════════════════════════════════════════════
設計邏輯：
  三個獨立模型分別訓練，每個模型的 Target 是不同天數的累計報酬率
  短期模型（20日）：技術面權重較高
  中期模型（40日）：籌碼面權重較高
  長期模型（60日）：趨勢 + 籌碼面為主

輸出：
  h20 / h40 / h60 : 各窗口預測報酬率
  trend           : 趨勢方向（加速向上 / 平穩 / 轉弱）
  best_horizon    : 哪個窗口信心最高
  price_targets   : 對應目標價估算
═══════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
import logging
from modules.ai_model import build_model
from config.settings import PREDICT_HORIZONS, MIN_TRAIN_SAMPLES

logger = logging.getLogger(__name__)


def predict_multi_horizon(train_df, latest_row, features):
    """
    對同一支股票訓練三個不同預測窗口的模型。

    參數：
        train_df    - 訓練用歷史資料（已 dropna）
        latest_row  - 最新一天的特徵（用於預測）
        features    - 特徵欄位清單

    回傳：dict
    """
    if len(train_df) < MIN_TRAIN_SAMPLES:
        return _empty_result(f"訓練資料不足（{len(train_df)} < {MIN_TRAIN_SAMPLES}）")

    results = {}
    price_col_exists = "Close" in latest_row.columns
    current_price = float(latest_row["Close"].iloc[0]) \
                    if price_col_exists else None

    for horizon in PREDICT_HORIZONS:
        key = f"h{horizon}"
        try:
            if "Close" not in train_df.columns:
                results[key] = None
                continue
            # 為每個窗口獨立設定 Target
            df_h = train_df.copy()
            df_h["Target"] = df_h["Close"].pct_change(fill_method=None).shift(-horizon)
            df_h = df_h.dropna(subset=features + ["Target"])

            if len(df_h) < MIN_TRAIN_SAMPLES:
                results[key] = None
                continue

            model = build_model()
            model.fit(df_h[features], df_h["Target"])
            pred  = float(model.predict(latest_row[features])[0])

            results[key] = {
                "pred":        round(pred, 4),
                "pred_pct":    f"{pred:+.2%}",
                "target_price": round(current_price * (1 + pred), 1)
                                if current_price else None,
                "confidence":  _horizon_confidence(horizon, len(df_h)),
                "label":       _pred_label(pred),
            }
        except Exception as e:
            logger.warning("多窗口預測失敗 horizon=%d [%s]：%s", horizon, symbol if False else "", e)
            import traceback; logger.debug(traceback.format_exc())
            results[key] = None

    # ── 趨勢分析 ──────────────────────────────────────────────
    trend   = _analyze_trend(results)
    best_h  = _best_horizon(results)

    return {
        "horizons":    results,
        "trend":       trend,
        "best_horizon":best_h,
        "current_price":current_price,
        "summary":     _build_summary(results, trend, best_h),
    }


def _horizon_confidence(horizon, n_samples):
    """
    預測窗口越長，信心度越低（市場不確定性增加）
    樣本越多，信心度越高
    """
    time_decay = 1.0 - (horizon - 20) / 80    # 20日=1.0, 60日=0.5
    sample_factor = min(n_samples / 300, 1.0)
    return round(float(time_decay * sample_factor), 2)


def _pred_label(pred):
    if pred > 0.15:    return "🚀 強烈看漲"
    elif pred > 0.08:  return "📈 看漲"
    elif pred > 0.03:  return "↗️ 小幅看漲"
    elif pred > -0.03: return "➡️ 持平"
    elif pred > -0.08: return "↘️ 小幅看跌"
    else:              return "📉 看跌"


def _analyze_trend(results):
    """
    分析三個窗口的趨勢方向
    若短期 < 中期 < 長期 → 持續加速向上
    若短期 > 中期 > 長期 → 短線強但長線轉弱
    """
    preds = []
    for h in PREDICT_HORIZONS:
        r = results.get(f"h{h}")
        if r and r.get("pred") is not None:
            preds.append(r["pred"])

    if len(preds) < 2:
        return {"label": "⚪ 無法判斷", "direction": "unknown"}

    # 計算短→中→長的斜率
    diffs = [preds[i+1] - preds[i] for i in range(len(preds)-1)]
    avg_diff = np.mean(diffs)

    if all(d > 0 for d in diffs):
        return {"label": "🚀 加速向上（短中長期同步看漲）", "direction": "accelerating_up"}
    elif preds[0] > 0 and avg_diff > 0:
        return {"label": "📈 持續向上", "direction": "up"}
    elif preds[0] > 0 and avg_diff < -0.02:
        return {"label": "⚡ 短線強、長線轉弱（注意)", "direction": "short_strong"}
    elif all(p > 0 for p in preds):
        return {"label": "↗️ 整體偏多", "direction": "mild_up"}
    elif all(p < 0 for p in preds):
        return {"label": "📉 整體偏空", "direction": "down"}
    else:
        return {"label": "➡️ 方向不明", "direction": "neutral"}


def _best_horizon(results):
    """找信心度最高的預測窗口"""
    best, best_conf = None, -1
    for h in PREDICT_HORIZONS:
        r = results.get(f"h{h}")
        if r and r.get("confidence", 0) > best_conf and r.get("pred", 0) > 0:
            best      = h
            best_conf = r["confidence"]
    return best


def _build_summary(results, trend, best_h):
    lines = [f"趨勢方向：{trend['label']}"]
    for h in PREDICT_HORIZONS:
        r = results.get(f"h{h}")
        if r:
            tp = f"（目標價 {r['target_price']}）" if r.get("target_price") else ""
            lines.append(f"  {h:2d}日預測：{r['pred_pct']:>8}  {r['label']}{tp}")
    if best_h:
        lines.append(f"最佳參考窗口：{best_h}日（信心度最高）")
    return "\n".join(lines)


def _empty_result(reason):
    return {
        "horizons":     {f"h{h}": None for h in PREDICT_HORIZONS},
        "trend":        {"label": f"無法判斷（{reason}）", "direction": "unknown"},
        "best_horizon": None,
        "current_price":None,
        "summary":      reason,
    }
