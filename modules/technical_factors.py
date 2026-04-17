"""
技術因子（v5 精簡版）
移除高度相關的冗餘指標，保留資訊獨立性最高的 10 個特徵。

移除項目及原因：
  MA5         → 與 Close 和 MA20 高度相關
  Return_5D/10D→ 與 Momentum(20日) 高度相關
  Bias_MA20/60 → Close/MA 的線性組合，XGBoost 自己能推導
  BB_pos       → MA20 + Volatility 的函數，資訊重疊
"""
import numpy as np
import pandas as pd


def add_technical(df):
    df = df.copy()

    # ── 趨勢 ──────────────────────────────────────────────────
    df["MA20"]     = df["Close"].rolling(20).mean()
    df["MA60"]     = df["Close"].rolling(60).mean()
    df["MA_ratio"] = df["MA20"] / df["MA60"].replace(0, float("nan"))   # 多頭 > 1

    # ── 動能 & 報酬率 ─────────────────────────────────────────
    df["Return"]   = df["Close"].pct_change()
    df["Momentum"] = df["Close"] / df["Close"].shift(20) - 1

    # ── 波動率 ────────────────────────────────────────────────
    df["Volatility"] = df["Return"].rolling(20).std()

    # ── 成交量 ────────────────────────────────────────────────
    df["VolumeMA"]    = df["Volume"].rolling(20).mean()
    df["VolumeRatio"] = df["Volume"] / df["VolumeMA"].replace(0, float("nan"))

    # ── RSI 14（過熱/超賣） ───────────────────────────────────
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("nan"))
    df["RSI14"] = 100 - (100 / (1 + rs))

    # ── 52 週高點距離（散戶 FOMO 指標） ─────────────────────
    # 改用 60 日高點（原 252 日會造成前 251 行全 NaN，嚴重壓縮訓練資料）
    df["Dist_52W_High"] = (
        df["Close"] / df["Close"].rolling(60).max() - 1
    )

    # ── 量價背離（外資出貨訊號） ─────────────────────────────
    # +1 = 量價同向  -1 = 價漲量縮（偷出貨）
    df["PV_diverge"] = (
        np.sign(df["Return"].fillna(0)) *
        np.sign(df["VolumeRatio"].fillna(1) - 1)
    )

    return df


# 技術面特徵清單（供 features_engine 引用）
TECH_FEATURES = [
    "MA20", "MA60", "MA_ratio",
    "Momentum", "Volatility",
    "VolumeMA", "VolumeRatio",
    "RSI14", "Dist_52W_High", "PV_diverge",
]
