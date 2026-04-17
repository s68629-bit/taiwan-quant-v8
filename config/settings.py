# ══════════════════════════════════════════════════════════════
# Taiwan Quant Fund v8 ─ 設定中心
# ══════════════════════════════════════════════════════════════
import os

# ── 基本 ──────────────────────────────────────────────────────
DATA_LOOKBACK_YEARS = 2
TOP_N               = 20
RETRAIN_DAYS        = 30
DASHBOARD_PORT      = 8501

# ── 多時間窗口預測 ─────────────────────────────────────────────
PREDICT_HORIZONS    = [20, 40, 60]

# ── 模型訓練 ──────────────────────────────────────────────────
TRAIN_RATIO         = 0.70
MIN_TRAIN_SAMPLES   = 200

# ── 四合一評分權重（合計 = 1.0） ───────────────────────────────
W_AI          = 0.25
W_CHIP        = 0.35
W_CONTRARIAN  = 0.25
W_SURGE       = 0.15

# ── 訊號門檻 ──────────────────────────────────────────────────
LONG_THRESHOLD         =  0.30
WATCH_THRESHOLD        =  0.10
SHORT_THRESHOLD        = -0.10
BEAR_MARKET_MULTIPLIER =  0.50
BEAR_LONG_THRESHOLD    =  0.50

# ── 飆漲偵測 ───────────────────────────────────────────────────
SURGE_SCORE_THRESHOLD    = 60
SURGE_DAYS_MAX           = 20
ACCUMULATION_DAYS        = 20
VOLUME_SQUEEZE_THRESHOLD = 0.70
VOLUME_EXPAND_THRESHOLD  = 1.30
VOLUME_EXPLOSION         = 2.00

# ── 風控 ──────────────────────────────────────────────────────
STOP_LOSS_RATE       = -0.07
MAX_POSITION_PCT     =  0.25
PORTFOLIO_SIZE       =  5
HOLD_DAYS            =  20

# ── 快取 ──────────────────────────────────────────────────────
CACHE_DIR    = "cache"
CACHE_HOURS  = 12

# ── FinMind（從環境變數讀取，不寫死在程式碼裡）────────────────
# 本機執行：在電腦設定環境變數 FINMIND_TOKEN=你的token
# Streamlit Cloud：在 Secrets 填入 FINMIND_TOKEN
import streamlit as st
def _get_finmind_token():
    # 優先從 Streamlit Secrets 讀取
    try:
        return st.secrets["FINMIND_TOKEN"]
    except Exception:
        pass
    # 其次從系統環境變數讀取（本機用）
    return os.environ.get("FINMIND_TOKEN", "")

FINMIND_TOKEN = _get_finmind_token()

# ── 報告 ──────────────────────────────────────────────────────
REPORT_DIR = "reports"

# ── 自我修正系統 ────────────────────────────────────────────────
PREDICTION_LOG_FILE    = "data/predictions.csv"
PERFORMANCE_LOG_FILE   = "data/performance_log.csv"
MIN_RECORDS_TO_ANALYZE = 50
FEATURE_IC_REMOVE_THRESHOLD = 0.02
WEIGHT_UPDATE_INTERVAL = 30

# ── 陷阱偵測門檻 ────────────────────────────────────────────────
SHORT_SALE_SURGE_THRESHOLD  = 0.15
FAKE_BREAKOUT_VOLUME_RATIO  = 2.00
MARGIN_CALL_BUFFER          = 0.85
DIVERGENCE_LOOKBACK         = 14
