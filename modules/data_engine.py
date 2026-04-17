"""
價格資料引擎 v6.1 - 含下市偵測
"""
import os, time, logging
import pandas as pd
import yfinance as yf
from config.settings import CACHE_DIR, CACHE_HOURS

logger = logging.getLogger(__name__)
os.makedirs(CACHE_DIR, exist_ok=True)

_INVALID_SYMBOLS = set()

def _cache_path(symbol):
    return os.path.join(CACHE_DIR, f"{symbol.replace('.','_')}_price.csv")

def _is_fresh(path, hours=CACHE_HOURS):
    return os.path.exists(path) and \
           (time.time() - os.path.getmtime(path)) < hours * 3600

def get_price(symbol):
    if symbol in _INVALID_SYMBOLS:
        raise ValueError(f"{symbol}: 已知無效（下市），略過")

    cache = _cache_path(symbol)
    if _is_fresh(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if df.empty:
            _INVALID_SYMBOLS.add(symbol)
            raise ValueError(f"{symbol}: 快取為空，可能已下市")
        logger.debug("%s: 快取 (%d 筆)", symbol, len(df))
        return df

    try:
        df = yf.download(symbol, period="2y", interval="1d",
                         progress=False, auto_adjust=True)
    except Exception as e:
        _INVALID_SYMBOLS.add(symbol)
        raise ValueError(f"{symbol}: 下載失敗，可能已下市 ({e})")

    if df.empty:
        _INVALID_SYMBOLS.add(symbol)
        raise ValueError(f"{symbol}: 無資料，代號可能已下市")

    if df.columns.nlevels > 1:
        df.columns = df.columns.droplevel(1)

    df = df.dropna()
    if len(df) < 20:
        _INVALID_SYMBOLS.add(symbol)
        raise ValueError(f"{symbol}: 資料不足 ({len(df)} 筆)")

    df.to_csv(cache)
    logger.debug("%s: 下載完成 (%d 筆)", symbol, len(df))
    return df

def clear_invalid_cache():
    _INVALID_SYMBOLS.clear()
