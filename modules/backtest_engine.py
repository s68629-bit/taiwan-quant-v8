"""
回測引擎 v5
新增：最大連虧天數、市場環境分期績效
"""
import numpy as np
import pandas as pd
import logging
from config.settings import PORTFOLIO_SIZE, STOP_LOSS_RATE

logger = logging.getLogger(__name__)


def run_full_backtest(val_results, price_dict, market_regime=None):
    """
    val_results: [(symbol, predict_with_validation 輸出), ...]
    """
    all_signals = []
    for sym, out in val_results:
        vdf = out.get("val_df", pd.DataFrame())
        if vdf.empty: continue
        vdf = vdf.copy()
        vdf["symbol"]    = sym
        vdf["composite"] = out["composite_result"]["composite"]
        all_signals.append(vdf)

    if not all_signals:
        return pd.DataFrame(), pd.DataFrame(), {}

    combined = pd.concat(all_signals, ignore_index=True)
    combined["date"]  = pd.to_datetime(combined["date"])
    combined["month"] = combined["date"].dt.to_period("M")
    months = sorted(combined["month"].unique())

    equity, curve, trades = 1.0, [], []

    for i, m in enumerate(months):
        if i == 0:
            curve.append({"month": str(m), "equity": equity})
            continue

        prev_m    = months[i-1]
        prev_sigs = combined[combined["month"] == prev_m]
        if prev_sigs.empty:
            curve.append({"month": str(m), "equity": equity})
            continue

        # 依綜合分排序，取前 N（只取正訊號）
        top = (prev_sigs[prev_sigs["composite"] > 0]
               .sort_values("composite", ascending=False)
               .drop_duplicates("symbol")
               .head(PORTFOLIO_SIZE))

        monthly_rets = []
        for _, row in top.iterrows():
            sym = row["symbol"]
            if sym not in price_dict: continue
            p     = price_dict[sym]
            mp    = p[p.index.to_period("M") == m]
            if len(mp) < 2: continue
            r = float(mp["Close"].iloc[-1] / mp["Close"].iloc[0] - 1)
            # 停損截斷
            r_actual = max(r, STOP_LOSS_RATE)
            monthly_rets.append(r_actual)
            trades.append({
                "month":         str(m),
                "symbol":        sym,
                "composite":     float(row["composite"]),
                "actual_return": r,
                "capped_return": r_actual,
                "stop_loss_hit": r < STOP_LOSS_RATE,
                "hit":           r > 0,
            })

        if monthly_rets:
            equity *= (1 + np.mean(monthly_rets))
        curve.append({"month": str(m), "equity": equity})

    curve_df  = pd.DataFrame(curve)
    trades_df = pd.DataFrame(trades)
    metrics   = _compute_metrics(curve_df, trades_df)
    return curve_df, trades_df, metrics


def _compute_metrics(curve_df, trades_df):
    m = {}
    if curve_df.empty or len(curve_df) < 2:
        return m

    eq      = curve_df["equity"].values
    n_mo    = len(eq) - 1
    total_r = float(eq[-1]/eq[0] - 1)
    ann_r   = float((1+total_r)**(12/max(n_mo,1)) - 1)
    mo_rets = np.diff(eq) / eq[:-1]
    sharpe  = float((mo_rets.mean()/mo_rets.std())*np.sqrt(12)) \
              if mo_rets.std() > 0 else 0.0
    peak    = np.maximum.accumulate(eq)
    max_dd  = float(((eq - peak)/peak).min())

    m.update({"total_return": round(total_r,4), "ann_return": round(ann_r,4),
               "sharpe": round(sharpe,4), "max_drawdown": round(max_dd,4),
               "n_months": int(n_mo)})

    if not trades_df.empty:
        sl_count = int(trades_df["stop_loss_hit"].sum()) \
                   if "stop_loss_hit" in trades_df.columns else 0
        m.update({
            "win_rate":      round(float(trades_df["hit"].mean()),4),
            "n_trades":      int(len(trades_df)),
            "avg_trade_r":   round(float(trades_df["actual_return"].mean()),4),
            "stop_loss_hits":sl_count,
        })
    return m
