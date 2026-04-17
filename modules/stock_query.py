"""
個股深度查詢引擎（v7 新增）
═══════════════════════════════════════════════════════════════
輸入任意台股代號，輸出完整的五維分析：

  1. 四合一評分 + 進出場建議
  2. 外資行為分析（本輪成本、建倉天數、動向）
  3. 坑殺風險評估（假突破、借券、融資、KD背離）
  4. 多時間窗口預測 + 目標價
  5. 歷史預測績效（這支股的過去勝率）
═══════════════════════════════════════════════════════════════
"""
import logging
import pandas as pd
import numpy as np
from modules.features_engine import build_features
from modules.ai_model import build_model
from modules.contrarian_signals import compute_contrarian_score
from modules.composite_scorer import compute_composite
from modules.surge_detector import detect_surge
from modules.multi_horizon import predict_multi_horizon
from modules.foreign_cost import compute_foreign_cost
from modules.retail_sentiment import compute_retail_sentiment
from modules.trap_detector import detect_traps
from modules.data_engine import get_price
from modules.chipdata_engine import get_institutional, get_margin
from modules.prediction_logger import PREDICTION_LOG_FILE
from core.market_filter import get_market_regime
from config.settings import STOP_LOSS_RATE, MAX_POSITION_PCT

logger = logging.getLogger(__name__)

import os


def query_stock(symbol):
    """
    對單支股票執行完整深度分析。
    回傳：完整分析結果 dict
    """
    symbol = symbol.strip().upper()
    if not symbol.endswith(".TW"):
        symbol = symbol + ".TW"

    logger.info("個股查詢：%s", symbol)
    result = {"symbol": symbol, "error": None}

    try:
        # ── 基礎資料 ─────────────────────────────────────────
        price_df               = get_price(symbol)
        inst_df, inst_ok       = get_institutional(symbol)
        margin_df, margin_ok   = get_margin(symbol)

        train_df, latest_row, features, chip_real = build_features(symbol)

        current_price = float(price_df["Close"].iloc[-1])
        result["current_price"] = round(current_price, 1)
        result["chip_data_ok"]  = inst_ok and margin_ok

        # ── AI 模型訓練 ─────────────────────────────────────
        model = build_model()
        model.fit(train_df[features], train_df["Target"])
        ai_pred = float(model.predict(latest_row[features])[0])

        # ── 各模組分析 ───────────────────────────────────────
        full_df      = pd.concat([train_df, latest_row]).drop_duplicates()
        market       = get_market_regime()
        contrarian   = compute_contrarian_score(full_df)
        surge        = detect_surge(full_df)
        retail_sent  = compute_retail_sentiment(full_df)
        foreign_cost = compute_foreign_cost(price_df, inst_df if inst_ok else None)
        multi_h      = predict_multi_horizon(train_df, latest_row, features)
        trap         = detect_traps(full_df, inst_df if inst_ok else None,
                                    margin_df if margin_ok else None)

        chip_latest  = latest_row.iloc[0] if not latest_row.empty else None
        composite    = compute_composite(symbol, ai_pred, chip_latest,
                                         contrarian, market["is_bear"], surge)

        # ── 進出場建議 ────────────────────────────────────────
        entry_exit = _compute_entry_exit(
            current_price, foreign_cost, multi_h, composite, trap
        )

        # ── 外資行為彙整 ──────────────────────────────────────
        foreign_behavior = _summarize_foreign_behavior(
            inst_df if inst_ok else None,
            foreign_cost, trap
        )

        # ── 歷史預測績效 ──────────────────────────────────────
        history = _get_symbol_history(symbol)

        result.update({
            "composite_result": composite,
            "ai_pred":          ai_pred,
            "surge":            surge,
            "multi_horizon":    multi_h,
            "foreign_cost":     foreign_cost,
            "retail_sentiment": retail_sent,
            "trap":             trap,
            "contrarian":       contrarian,
            "market":           market,
            "entry_exit":       entry_exit,
            "foreign_behavior": foreign_behavior,
            "history":          history,
            "features_used":    len(features),
            "train_samples":    len(train_df),
        })

    except Exception as e:
        logger.error("個股查詢失敗 %s：%s", symbol, e)
        result["error"] = str(e)

    return result


def _compute_entry_exit(current_price, foreign_cost, multi_h, composite, trap):
    """計算進出場建議"""
    fc = foreign_cost or {}

    # 進場甜蜜點：外資成本線 ±3%
    support_low  = fc.get("support_low")
    support_high = fc.get("support_high")
    cost_60d     = fc.get("cost_60d")

    # 目標價（取 20 日預測）
    hs = (multi_h or {}).get("horizons", {})
    h20 = (hs.get("h20") or {}).get("pred", None)
    h60 = (hs.get("h60") or {}).get("pred", None)

    target_20d = round(current_price * (1 + h20), 1) if h20 else None
    target_60d = round(current_price * (1 + h60), 1) if h60 else None

    # 停損價
    stop_loss_price = round(current_price * (1 + STOP_LOSS_RATE), 1)

    # 進場建議
    comp_score = composite.get("composite", 0)
    trap_ok    = trap.get("entry_ok", True)
    trap_score = trap.get("trap_score", 0)

    if comp_score > 0.40 and trap_ok and trap_score < 30:
        entry_advice = "🟢 強力進場訊號，陷阱風險低，可考慮進場"
        entry_zone   = f"{support_low} ~ {support_high}" if support_low else "接近當前價位"
    elif comp_score > 0.30 and trap_ok:
        entry_advice = "🟡 做多訊號，建議等待回測至外資成本線附近再進場"
        entry_zone   = f"{support_low} ~ {support_high}" if support_low else "等待回測"
    elif trap_score > 50:
        entry_advice = "🔴 陷阱風險高，暫緩進場，等待風險解除"
        entry_zone   = "暫緩"
    else:
        entry_advice = "⚪ 訊號不明確，觀望為主"
        entry_zone   = "觀望"

    return {
        "entry_advice":    entry_advice,
        "entry_zone":      entry_zone,
        "stop_loss_price": stop_loss_price,
        "stop_loss_rate":  f"{STOP_LOSS_RATE:.0%}",
        "target_20d":      target_20d,
        "target_60d":      target_60d,
        "max_position":    f"{MAX_POSITION_PCT:.0%}",
    }


def _summarize_foreign_behavior(inst_df, foreign_cost, trap):
    """彙整外資行為摘要"""
    fc    = foreign_cost or {}
    accum = trap.get("accumulation_start", {}) or {}
    rev   = trap.get("foreign_reversal", {}) or {}
    short = trap.get("short_sale", {}) or {}

    # 近期外資動向
    recent_direction = "無資料"
    total_20d_net    = None

    if inst_df is not None and not inst_df.empty and "foreign_net" in inst_df.columns:
        fn = inst_df["foreign_net"].fillna(0)
        if len(fn) >= 5:
            recent_5d  = float(fn.iloc[-5:].sum())
            recent_20d = float(fn.iloc[-20:].sum()) if len(fn) >= 20 else None
            total_20d_net = recent_20d

            if recent_5d > 2000:
                recent_direction = f"🟢 持續買進（近5日 +{recent_5d:,.0f}張）"
            elif recent_5d > 0:
                recent_direction = f"🟡 小幅買進（近5日 +{recent_5d:,.0f}張）"
            elif recent_5d > -2000:
                recent_direction = f"🟡 小幅賣出（近5日 {recent_5d:,.0f}張）"
            else:
                recent_direction = f"🔴 明顯賣出（近5日 {recent_5d:,.0f}張）"

    return {
        "accumulation_start":  accum.get("date"),
        "accumulation_days":   accum.get("days_ago"),
        "accumulation_desc":   accum.get("description", "無資料"),
        "cost_60d":            fc.get("cost_60d"),
        "cost_20d":            fc.get("cost_20d"),
        "gap_label":           fc.get("gap_label", "N/A"),
        "position_label":      fc.get("position_label", "無資料"),
        "implication":         fc.get("implication", ""),
        "support_zone":        f"{fc.get('support_low')} ~ {fc.get('support_high')}"
                               if fc.get("support_low") else "計算中",
        "recent_direction":    recent_direction,
        "total_20d_net":       total_20d_net,
        "reversal_desc":       rev.get("description", ""),
        "short_sale_desc":     short.get("description", ""),
    }


def _get_symbol_history(symbol):
    """取得這支股票的歷史預測績效"""
    if not os.path.exists(PREDICTION_LOG_FILE):
        return {"count": 0, "win_rate": None, "avg_return": None, "records": []}

    df   = pd.read_csv(PREDICTION_LOG_FILE)
    sym_df = df[df["symbol"] == symbol]
    done   = sym_df[sym_df["verified"] == True]

    records = []
    for _, row in done.tail(10).iterrows():
        records.append({
            "date":      row["scan_date"],
            "pred":      f"{row['pred_20d']:+.2%}" if pd.notna(row.get("pred_20d")) else "N/A",
            "actual":    f"{row['actual_20d']:+.2%}" if pd.notna(row.get("actual_20d")) else "N/A",
            "hit":       "✅" if row.get("hit") else "❌",
        })

    return {
        "count":       len(done),
        "total":       len(sym_df),
        "win_rate":    round(float(done["hit"].mean()), 4) if len(done) > 0 else None,
        "avg_return":  round(float(done["actual_20d"].mean()), 4) if len(done) > 0 else None,
        "records":     records,
    }
