"""
績效分析引擎（v7 自我修正第二層）
═══════════════════════════════════════════════════════════════
功能：
  1. 自動查驗 20 天前的預測（從 yfinance 抓實際價格）
  2. 計算各特徵的 IC 值（信息係數）
  3. 輸出模型健康報告

IC 值說明：
  IC = 預測值與實際報酬率的相關係數
  > 0.10 = 優秀，0.05~0.10 = 有效，< 0.02 = 建議移除
═══════════════════════════════════════════════════════════════
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import date
from config.settings import (PREDICTION_LOG_FILE, PERFORMANCE_LOG_FILE,
                              MIN_RECORDS_TO_ANALYZE, FEATURE_IC_REMOVE_THRESHOLD)
from modules.prediction_logger import mark_verified, get_pending_verifications

logger = logging.getLogger(__name__)


def run_verification():
    """
    自動查驗所有到期的預測記錄。
    從 yfinance 抓取實際價格，計算實際報酬率。
    回傳：{verified_count, skipped, errors}
    """
    pending = get_pending_verifications()
    if pending.empty:
        logger.info("無待查驗記錄")
        return {"verified_count": 0, "skipped": 0, "errors": []}

    import yfinance as yf

    verified_syms, verified_dates, actual_rets = [], [], []
    errors = []

    # 依股票分組查驗
    for symbol, group in pending.groupby("symbol"):
        try:
            # 下載近 3 個月資料（確保有足夠的未來資料）
            df_p = yf.download(symbol, period="3mo", interval="1d",
                               progress=False, auto_adjust=True)
            if df_p.empty:
                errors.append(f"{symbol}: 無法取得資料")
                continue

            if df_p.columns.nlevels > 1:
                df_p.columns = df_p.columns.droplevel(1)

            for _, row in group.iterrows():
                scan_dt = pd.Timestamp(row["scan_date"])
                # 找掃描日當天或之後的第一個交易日收盤價
                future_prices = df_p[df_p.index > scan_dt]["Close"]
                if len(future_prices) < 20:
                    continue   # 不足 20 個交易日，跳過

                entry_price = float(future_prices.iloc[0])
                exit_price  = float(future_prices.iloc[19])  # 第 20 個交易日
                actual_ret  = (exit_price - entry_price) / entry_price

                verified_syms.append(symbol)
                verified_dates.append(row["scan_date"])
                actual_rets.append(round(actual_ret, 4))

        except Exception as e:
            errors.append(f"{symbol}: {e}")

    if verified_syms:
        mark_verified(verified_syms, verified_dates, actual_rets)

    logger.info("查驗完成：%d 筆，錯誤：%d 筆",
                len(verified_syms), len(errors))
    return {
        "verified_count": len(verified_syms),
        "skipped":        len(pending) - len(verified_syms) - len(errors),
        "errors":         errors,
    }


def compute_feature_ic():
    """
    計算各特徵欄位與實際報酬率的 IC 值。
    只用已查驗的記錄（verified = True）。
    回傳：DataFrame（feature, ic, recommendation）
    """
    if not os.path.exists(PREDICTION_LOG_FILE):
        return pd.DataFrame()

    df = pd.read_csv(PREDICTION_LOG_FILE)
    done = df[df["verified"] == True].copy()

    if len(done) < MIN_RECORDS_TO_ANALYZE:
        logger.info("已查驗記錄不足 %d 筆（目前 %d 筆），無法進行特徵分析",
                    MIN_RECORDS_TO_ANALYZE, len(done))
        return pd.DataFrame()

    # 可分析的特徵欄位
    feature_cols = [
        "composite", "ai_score", "chip_score", "contrarian",
        "surge_score", "trap_score", "retail_temp", "retail_heat",
        "pred_20d",
    ]
    available = [c for c in feature_cols if c in done.columns]

    results = []
    for feat in available:
        series = done[feat].dropna()
        actual = done.loc[series.index, "actual_20d"].dropna()
        common = series.index.intersection(actual.index)

        if len(common) < 20:
            continue

        ic = float(np.corrcoef(series[common], actual[common])[0, 1])
        rec = _ic_recommendation(feat, ic)
        results.append({"feature": feat, "ic": round(ic, 4), "recommendation": rec,
                         "n_samples": len(common)})

    result_df = pd.DataFrame(results).sort_values("ic", ascending=False)
    return result_df


def compute_weight_suggestion(current_weights: dict):
    """
    根據各評分維度與實際報酬的相關性，建議新的權重。
    current_weights: {"W_AI": 0.25, "W_CHIP": 0.35, ...}
    回傳: {"suggested": {...}, "change": {...}, "basis": str}
    """
    if not os.path.exists(PREDICTION_LOG_FILE):
        return {"suggested": current_weights, "change": {}, "basis": "無資料"}

    df   = pd.read_csv(PREDICTION_LOG_FILE)
    done = df[df["verified"] == True].copy()

    if len(done) < MIN_RECORDS_TO_ANALYZE:
        return {
            "suggested": current_weights,
            "change": {},
            "basis": f"資料不足（{len(done)}/{MIN_RECORDS_TO_ANALYZE} 筆）"
        }

    # 計算各維度與實際報酬的相關性
    dim_map = {
        "W_AI":         "ai_score",
        "W_CHIP":       "chip_score",
        "W_CONTRARIAN": "contrarian",
        "W_SURGE":      "surge_score",
    }

    corrs = {}
    for w_name, col in dim_map.items():
        if col not in done.columns: continue
        s = done[col].dropna()
        a = done.loc[s.index, "actual_20d"].dropna()
        c = s.index.intersection(a.index)
        if len(c) < 20: continue
        corrs[w_name] = max(float(np.corrcoef(s[c], a[c])[0, 1]), 0.001)

    if not corrs:
        return {"suggested": current_weights, "change": {}, "basis": "相關性計算失敗"}

    # 正規化為權重（相關性越高 → 權重越大）
    total_corr = sum(corrs.values())
    suggested  = {k: round(v / total_corr, 3) for k, v in corrs.items()}

    # 計算變化
    changes = {k: round(suggested.get(k, 0) - current_weights.get(k, 0), 3)
               for k in set(list(suggested.keys()) + list(current_weights.keys()))}

    return {
        "suggested": suggested,
        "change":    changes,
        "basis":     f"基於 {len(done)} 筆已查驗預測計算（IC 加權）",
    }


def generate_health_report():
    """
    產生模型健康報告（供 Dashboard 顯示）。
    回傳: dict
    """
    if not os.path.exists(PREDICTION_LOG_FILE):
        return {"status": "無預測記錄（請先執行一次掃描）",
                "total_predictions": 0, "verified": 0, "pending": 0,
                "win_rate": None, "avg_return": None,
                "feature_ic": None, "weight_suggestion": {}}

    df   = pd.read_csv(PREDICTION_LOG_FILE)
    done = df[df["verified"] == True]
    total = len(df)

    report = {
        "total_predictions":  total,
        "verified":           len(done),
        "pending":            total - len(done),
        "win_rate":           None,
        "avg_return":         None,
        "feature_ic":         pd.DataFrame(),
        "weight_suggestion":  {},
        "status":             "正常",
    }

    if len(done) >= 10:
        report["win_rate"]    = round(float(done["hit"].mean()), 4)
        report["avg_return"]  = round(float(done["actual_20d"].mean()), 4)

    if len(done) >= MIN_RECORDS_TO_ANALYZE:
        report["feature_ic"] = compute_feature_ic()
        from config.settings import W_AI, W_CHIP, W_CONTRARIAN, W_SURGE
        current_w = {"W_AI": W_AI, "W_CHIP": W_CHIP,
                     "W_CONTRARIAN": W_CONTRARIAN, "W_SURGE": W_SURGE}
        report["weight_suggestion"] = compute_weight_suggestion(current_w)
        report["status"] = "可分析"
    else:
        remaining = MIN_RECORDS_TO_ANALYZE - len(done)
        report["status"] = f"累積中（還需 {remaining} 筆查驗記錄才可分析）"

    return report


def _ic_recommendation(feat, ic):
    if ic >= 0.10:   return f"✅ 強效（IC={ic:.3f}），建議提高權重"
    elif ic >= 0.05: return f"✅ 有效（IC={ic:.3f}），保留"
    elif ic >= 0.02: return f"🟡 邊緣（IC={ic:.3f}），持續觀察"
    else:            return f"❌ 無效（IC={ic:.3f}），建議移除"
