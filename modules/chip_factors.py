"""
籌碼因子計算 v5
新增：記錄每個特徵是否來自真實資料或填零
"""
import pandas as pd

CHIP_FEATURES = [
    "foreign_net_20d",    # 外資 20 日累計淨買超（張）
    "trust_net_5d",       # 投信 5 日累計（月底作帳效應）
    "chip_net_5d",        # 三大法人 5 日合計
    "foreign_consec_buy", # 外資連續買超天數
    "margin_chg_5d",      # 融資 5 日變化率（>0 = 散戶加碼融資）
    "short_squeeze_ratio",# 融券/融資比（高 = 軋空潛力）
]


def add_chip_factors(df, inst_df, margin_df):
    df   = df.copy()
    real = {}   # 記錄哪些欄位有真實資料

    # ── 三大法人 ──────────────────────────────────────────────
    if not inst_df.empty:
        for col in ["foreign_net", "trust_net", "chip_net_total"]:
            if col in inst_df.columns:
                df[col] = inst_df[col].reindex(df.index).fillna(0)
                real[col] = True

        if "foreign_net" in df.columns:
            df["foreign_net_20d"] = df["foreign_net"].rolling(20).sum()

            # 連續買超天數
            is_buy = df["foreign_net"] > 0
            change = (is_buy != is_buy.shift()).cumsum()
            df["foreign_consec_buy"] = (
                is_buy.groupby(change).cumsum() * is_buy
            ).astype(float)

        if "trust_net" in df.columns:
            df["trust_net_5d"] = df["trust_net"].rolling(5).sum()

        if "chip_net_total" in df.columns:
            df["chip_net_5d"] = df["chip_net_total"].rolling(5).sum()

        real["chip_ok"] = True
    else:
        _fill_zeros(df, ["foreign_net","trust_net","chip_net_total",
                         "foreign_net_20d","trust_net_5d",
                         "chip_net_5d","foreign_consec_buy"])

    # ── 融資融券 ──────────────────────────────────────────────
    if not margin_df.empty:
        for col in ["margin_balance","short_balance","margin_ratio"]:
            if col in margin_df.columns:
                df[col] = margin_df[col].reindex(df.index)
                real[col] = True

        if "margin_balance" in df.columns:
            mb = df["margin_balance"].replace(0, float("nan"))
            df["margin_chg_5d"] = mb.pct_change(5, fill_method=None)

        if {"margin_balance","short_balance"}.issubset(df.columns):
            mb = df["margin_balance"].replace(0, float("nan"))
            df["short_squeeze_ratio"] = df["short_balance"] / mb

        real["margin_ok"] = True
    else:
        _fill_zeros(df, ["margin_balance","short_balance","margin_ratio",
                         "margin_chg_5d","short_squeeze_ratio"])

    df.attrs["chip_real"] = real   # 附加資料品質資訊
    return df


def _fill_zeros(df, cols):
    for c in cols:
        df[c] = 0.0
