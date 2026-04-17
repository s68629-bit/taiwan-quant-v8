"""
大盤環境過濾器（v5 新增）
使用加權指數（^TWII）判斷目前市場處於多頭或空頭
空頭期間：所有做多訊號打折、門檻提高
"""
import logging
import pandas as pd
from modules.data_engine import get_price

logger = logging.getLogger(__name__)

TAIEX_SYMBOL = "^TWII"


def get_market_regime():
    """
    回傳大盤環境判斷結果：
    {
        "is_bear":    bool,       # True = 空頭
        "taiex_ma20": float,
        "taiex_ma60": float,
        "regime":     str,        # "多頭" / "空頭" / "整理"
        "description":str,
    }
    """
    try:
        df = get_price(TAIEX_SYMBOL)
        if df.empty or len(df) < 60:
            return _unknown_regime("加權指數資料不足")

        ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
        ma60 = float(df["Close"].rolling(60).mean().iloc[-1])
        close = float(df["Close"].iloc[-1])

        gap_pct = (ma20 - ma60) / ma60   # MA20 相對 MA60 的差距

        if gap_pct > 0.02:                # MA20 明顯在 MA60 之上 = 多頭
            is_bear = False
            regime  = "多頭"
            desc    = f"加權指數 {close:,.0f}，MA20({ma20:,.0f}) > MA60({ma60:,.0f})，趨勢向上"
        elif gap_pct < -0.02:             # MA20 明顯在 MA60 之下 = 空頭
            is_bear = True
            regime  = "空頭"
            desc    = f"加權指數 {close:,.0f}，MA20({ma20:,.0f}) < MA60({ma60:,.0f})，趨勢向下 ⚠️ 空頭環境，訊號門檻提高"
        else:
            is_bear = False
            regime  = "整理"
            desc    = f"加權指數 {close:,.0f}，MA20 ≈ MA60，盤整中，維持一般門檻"

        logger.info("大盤環境：%s｜%s", regime, desc)
        return {"is_bear": is_bear, "taiex_ma20": ma20,
                "taiex_ma60": ma60, "regime": regime, "description": desc}

    except Exception as e:
        logger.warning("無法判斷大盤環境（%s），預設為非空頭", e)
        return _unknown_regime(str(e))


def _unknown_regime(reason):
    return {"is_bear": False, "taiex_ma20": 0, "taiex_ma60": 0,
            "regime": "未知", "description": f"無法判斷（{reason}）"}
