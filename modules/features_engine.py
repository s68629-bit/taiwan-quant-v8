"""
特徵整合引擎 v5
關鍵修正：正確的資料分割，防止 Target 洩漏到訓練資料
"""
import logging
import pandas as pd
from modules.data_engine import get_price
from modules.technical_factors import add_technical, TECH_FEATURES
from modules.institutional_cost_engine import add_cost
from modules.chipdata_engine import get_institutional, get_margin
from modules.chip_factors import add_chip_factors, CHIP_FEATURES

logger = logging.getLogger(__name__)

COST_FEATURES = ["VWAP", "CostBreak"]
ALL_FEATURES  = TECH_FEATURES + CHIP_FEATURES + COST_FEATURES


def build_features(symbol, start_date="2023-01-01"):
    """
    完整特徵工程：
    1. 下載價格資料
    2. 技術指標 + VWAP
    3. 籌碼資料（失敗時填 0，並記錄資料品質）
    4. 設定 Target（未來 20 日報酬率）

    ★ 修正資料洩漏 ★
    回傳 train_df 和 latest_row 分開：
      - train_df:   Target 已知的歷史資料（用於訓練）
      - latest_row: 最新一天的特徵（用於預測，Target 未來尚未發生）
      - features:   實際可用特徵清單
      - chip_real:  籌碼資料品質字典
    """
    # ── 價格 ──────────────────────────────────────────────────
    df = get_price(symbol)
    if df.empty:
        raise ValueError(f"{symbol}: 無法取得價格資料")

    # ── 技術指標 + 法人成本 ───────────────────────────────────
    df = add_technical(df)
    df = add_cost(df)

    # ── 籌碼（容錯） ──────────────────────────────────────────
    try:
        inst_df,   inst_ok   = get_institutional(symbol, start_date)
        margin_df, margin_ok = get_margin(symbol, start_date)
    except Exception as e:
        logger.warning("%s 籌碼取得失敗（%s），填 0 繼續", symbol, e)
        inst_df   = pd.DataFrame()
        margin_df = pd.DataFrame()

    df = add_chip_factors(df, inst_df, margin_df)
    chip_real = df.attrs.get("chip_real", {})

    # ── Target：未來 20 日累計報酬率 ─────────────────────────
    df["Target"] = df["Close"].pct_change().shift(-20)

    # ── 可用特徵欄位 ──────────────────────────────────────────
    available = [f for f in ALL_FEATURES if f in df.columns]

    # ── ★ 正確分割（防洩漏）★ ─────────────────────────────────
    # train_df：有完整特徵 AND Target 已知（dropna 兩者）
    # latest_row：有完整特徵 BUT Target 尚未發生（不 dropna Target）
    train_df    = df.dropna(subset=available + ["Target"])
    latest_rows = df.dropna(subset=available)   # Target 可能是 NaN 沒關係
    if latest_rows.empty:
        raise ValueError(f"{available}: 特徵資料不足，無法取得最新一行")
    latest_row  = latest_rows.iloc[[-1]]         # 取最後一天

    logger.debug(
        "%s: 特徵 %d 個 | 訓練樣本 %d 筆 | 籌碼真實=%s",
        symbol, len(available), len(train_df),
        chip_real.get("chip_ok", False)
    )

    return train_df, latest_row, available, chip_real
