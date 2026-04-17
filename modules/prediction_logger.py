"""
預測記錄系統（v7 自我修正核心）
═══════════════════════════════════════════════════════════════
每次掃描完成後，把所有股票的預測結果記錄到 CSV
20 個交易日後自動查驗實際結果，計算預測準確率

記錄欄位：
  scan_date    掃描日期
  symbol       股票代號
  pred_20d     預測 20 日報酬率
  composite    綜合評分
  ai_score     AI 分
  chip_score   籌碼分
  contrarian   反向分
  surge_score  飆漲分
  trap_score   陷阱分（v7 新增）
  ...所有特徵值（供事後分析）
  verified     是否已查驗（預設 False）
  actual_20d   實際 20 日報酬率（查驗後填入）
  hit          預測方向是否正確
═══════════════════════════════════════════════════════════════
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import date, timedelta
from config.settings import PREDICTION_LOG_FILE, PERFORMANCE_LOG_FILE

logger = logging.getLogger(__name__)
os.makedirs("data", exist_ok=True)


def log_predictions(scan_results, full_results=None):
    """
    記錄本次掃描的所有預測結果。

    scan_results  : [(symbol, composite_dict), ...]
    full_results  : [(symbol, composite_dict, full_out), ...] 含所有細節
    """
    today     = str(date.today())
    new_rows  = []

    for item in (full_results or []):
        sym  = item[0]
        c    = item[1]
        out  = item[2] if len(item) > 2 else {}

        mh    = out.get("multi_horizon", {})
        hs    = mh.get("horizons", {}) if mh else {}
        surge = out.get("surge", {}) or {}
        trap  = out.get("trap", {}) or {}
        rs    = out.get("retail_sentiment", {}) or {}

        row = {
            "scan_date":    today,
            "symbol":       sym,
            # 評分
            "composite":    c.get("composite", 0),
            "ai_score":     c.get("ai_score", 0),
            "chip_score":   c.get("chip_score", 0),
            "contrarian":   c.get("contrarian_score", 0),
            "surge_score":  c.get("surge_score_raw", 0),
            "trap_score":   trap.get("trap_score", 0),
            "signal":       c.get("signal", ""),
            # 多時間窗口預測
            "pred_20d":     (hs.get("h20") or {}).get("pred", None),
            "pred_40d":     (hs.get("h40") or {}).get("pred", None),
            "pred_60d":     (hs.get("h60") or {}).get("pred", None),
            # 輔助指標
            "retail_temp":  rs.get("temperature", 50),
            "retail_heat":  c.get("retail_heat", 0),
            "entry_ok":     trap.get("entry_ok", True),
            # 查驗欄位（事後填入）
            "verified":     False,
            "actual_20d":   None,
            "hit":          None,
            "verify_date":  str(date.today() + timedelta(days=28)),  # 約 20 交易日
        }
        new_rows.append(row)

    if not new_rows:
        logger.warning("prediction_logger: 無資料可記錄")
        return

    new_df = pd.DataFrame(new_rows)

    # 追加到現有記錄
    if os.path.exists(PREDICTION_LOG_FILE):
        existing = pd.read_csv(PREDICTION_LOG_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
        # 避免同一天同一股票重複記錄
        combined = combined.drop_duplicates(subset=["scan_date","symbol"], keep="last")
    else:
        combined = new_df

    combined.to_csv(PREDICTION_LOG_FILE, index=False)
    logger.info("預測記錄已儲存：%d 支股票 → %s（累計 %d 筆）",
                len(new_rows), PREDICTION_LOG_FILE, len(combined))


def get_pending_verifications():
    """
    取得所有「應該查驗但尚未查驗」的預測記錄
    （查驗日期 <= 今天 AND verified = False）
    """
    if not os.path.exists(PREDICTION_LOG_FILE):
        return pd.DataFrame()

    df = pd.read_csv(PREDICTION_LOG_FILE)
    df["verify_date"] = pd.to_datetime(df["verify_date"])
    today = pd.Timestamp.today()

    pending = df[(df["verified"] == False) & (df["verify_date"] <= today)]
    return pending


def mark_verified(symbols, scan_dates, actual_returns):
    """
    把查驗結果寫回 CSV
    symbols       : 股票代號列表
    scan_dates    : 對應的掃描日期
    actual_returns: 實際 20 日報酬率列表
    """
    if not os.path.exists(PREDICTION_LOG_FILE):
        return

    df = pd.read_csv(PREDICTION_LOG_FILE)

    for sym, scan_date, actual in zip(symbols, scan_dates, actual_returns):
        mask = (df["symbol"] == sym) & (df["scan_date"] == scan_date)
        if not any(mask):
            continue
        df.loc[mask, "actual_20d"] = actual
        df.loc[mask, "verified"]   = True
        # 預測方向是否正確（預測正 & 實際正，或預測負 & 實際負）
        pred = df.loc[mask, "pred_20d"].iloc[0]
        if pd.notna(pred) and actual is not None:
            df.loc[mask, "hit"] = bool((pred > 0) == (actual > 0))

    df.to_csv(PREDICTION_LOG_FILE, index=False)
    logger.info("查驗記錄已更新：%d 筆", len(symbols))


def get_summary():
    """回傳預測記錄的簡要統計"""
    if not os.path.exists(PREDICTION_LOG_FILE):
        return {"total": 0, "verified": 0, "win_rate": None}

    df   = pd.read_csv(PREDICTION_LOG_FILE)
    done = df[df["verified"] == True]

    return {
        "total":         len(df),
        "verified":      len(done),
        "win_rate":      round(float(done["hit"].mean()), 4) if len(done) > 0 else None,
        "avg_actual":    round(float(done["actual_20d"].mean()), 4) if len(done) > 0 else None,
        "latest_date":   df["scan_date"].max() if len(df) > 0 else None,
    }
