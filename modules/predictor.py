"""預測核心 v7 - 整合陷阱偵測"""
import logging, numpy as np, pandas as pd
from modules.features_engine import build_features
from modules.ai_model import build_model, get_feature_importance
from modules.contrarian_signals import compute_contrarian_score
from modules.composite_scorer import compute_composite
from modules.surge_detector import detect_surge
from modules.multi_horizon import predict_multi_horizon
from modules.foreign_cost import compute_foreign_cost
from modules.retail_sentiment import compute_retail_sentiment
from modules.trap_detector import detect_traps
from modules.data_engine import get_price
from modules.chipdata_engine import get_institutional, get_margin
from config.settings import MIN_TRAIN_SAMPLES, TRAIN_RATIO, LONG_THRESHOLD

logger = logging.getLogger(__name__)

def _get_chip_data(symbol):
    try:
        price_df = get_price(symbol)
        inst_df, inst_ok   = get_institutional(symbol)
        margin_df, margin_ok = get_margin(symbol)
        return price_df, inst_df if inst_ok else None, margin_df if margin_ok else None
    except Exception:
        return None, None, None

def predict_full(symbol, is_bear_market=False):
    train_df, latest_row, features, chip_real = build_features(symbol)
    if len(train_df) < MIN_TRAIN_SAMPLES:
        raise ValueError(f"{symbol}: 訓練樣本不足")

    model = build_model()
    model.fit(train_df[features], train_df["Target"])
    ai_pred = float(model.predict(latest_row[features])[0])

    full_df      = pd.concat([train_df, latest_row]).drop_duplicates()
    contrarian   = compute_contrarian_score(full_df)
    surge        = detect_surge(full_df)
    retail_sent  = compute_retail_sentiment(full_df)
    price_df, inst_df, margin_df = _get_chip_data(symbol)
    foreign_cost = compute_foreign_cost(price_df, inst_df) if price_df is not None else None
    trap         = detect_traps(full_df, inst_df, margin_df)
    chip_latest  = latest_row.iloc[0] if not latest_row.empty else None
    comp_result  = compute_composite(symbol, ai_pred, chip_latest,
                                     contrarian, is_bear_market, surge)
    multi_h = predict_multi_horizon(train_df, latest_row, features)

    logger.info("%s │ 綜合:%+.2f [%s] │ 飆:%d│陷阱:%d[%s] │ 情緒:%.0f°",
                symbol, comp_result["composite"], comp_result["signal"],
                surge["surge_score"], trap["trap_score"], trap["trap_level"],
                retail_sent["temperature"])

    return {"composite_result": comp_result, "ai_pred": ai_pred,
            "surge": surge, "multi_horizon": multi_h,
            "foreign_cost": foreign_cost, "retail_sentiment": retail_sent,
            "trap": trap, "chip_real": chip_real, "features": features}

def predict_with_validation(symbol, is_bear_market=False):
    train_df, latest_row, features, chip_real = build_features(symbol)
    if len(train_df) < MIN_TRAIN_SAMPLES:
        raise ValueError(f"{symbol}: 訓練樣本不足")

    split = int(len(train_df) * TRAIN_RATIO)
    tr, te = train_df.iloc[:split], train_df.iloc[split:]
    m_val = build_model(); m_val.fit(tr[features], tr["Target"])
    preds_oos = m_val.predict(te[features])

    val_df = pd.DataFrame({"date": te.index, "predicted": preds_oos.tolist(),
        "actual": te["Target"].values.tolist(),
        "signal": (preds_oos > LONG_THRESHOLD * 0.05).astype(int).tolist()})
    metrics = _val_metrics(val_df)

    m_final = build_model(); m_final.fit(train_df[features], train_df["Target"])
    ai_pred = float(m_final.predict(latest_row[features])[0])

    full_df      = pd.concat([train_df, latest_row]).drop_duplicates()
    contrarian   = compute_contrarian_score(full_df)
    surge        = detect_surge(full_df)
    retail_sent  = compute_retail_sentiment(full_df)
    price_df, inst_df, margin_df = _get_chip_data(symbol)
    foreign_cost = compute_foreign_cost(price_df, inst_df) if price_df is not None else None
    trap         = detect_traps(full_df, inst_df, margin_df)
    chip_latest  = latest_row.iloc[0] if not latest_row.empty else None
    comp_result  = compute_composite(symbol, ai_pred, chip_latest,
                                     contrarian, is_bear_market, surge)
    multi_h = predict_multi_horizon(train_df, latest_row, features)
    imp     = get_feature_importance(m_final, features)

    return {"composite_result": comp_result, "ai_pred": ai_pred,
            "val_df": val_df, "metrics": metrics,
            "surge": surge, "multi_horizon": multi_h,
            "foreign_cost": foreign_cost, "retail_sentiment": retail_sent,
            "trap": trap, "features": features,
            "chip_real": chip_real, "model": m_final, "importance": imp}

def _val_metrics(val_df):
    m = {"n_samples": len(val_df)}
    if len(val_df) < 10: return m
    import numpy as np
    corr = np.corrcoef(val_df["predicted"], val_df["actual"])[0, 1]
    m["IC"] = round(float(corr), 4)
    longs = val_df[val_df["signal"] == 1]
    if not longs.empty:
        m["win_rate"]   = round(float((longs["actual"] > 0).mean()), 4)
        m["avg_return"] = round(float(longs["actual"].mean()), 4)
        m["n_signals"]  = int(len(longs))
    return m
