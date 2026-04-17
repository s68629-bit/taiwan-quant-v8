"""
第一階段：技術面快篩引擎（v8 核心）
===============================================================
對全市場 ~970 支上市股票做快速技術面篩選
只使用 yfinance（無 API 額度限制）
目標：從 970 支 → 保留約 250 支候選

篩選邏輯（多空雙向）：
  做多候選：技術面顯示有上漲潛力
  排除名單：明顯偏空、流動性差、資料不足

速度優化：
  - 批次下載（最多同時處理 30 支）
  - 結果快取 12 小時（不重複下載）
  - 只計算快篩需要的指標（不跑完整模型）

通過條件（滿足任一組合）：
  組合A：多頭趨勢  MA20 > MA60 + 動能 > 0
  組合B：強勢突破  RSI 40~70  + 近5日量比 > 1.2
  組合C：吸籌型態  低波動橫盤 + 近期量縮後放量
  組合D：反彈訊號  RSI 底背離 + 價格在支撐區

淘汰條件（任一即淘汰）：
  股價 < 10 元
  近 20 日均量 < 500 張（流動性太差）
  近 5 日跌幅 > 12%（急跌中，風險高）
  資料筆數 < 60（新股或資料不足）
===============================================================
"""
import os, time, logging
import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings import CACHE_DIR, CACHE_HOURS

logger = logging.getLogger(__name__)
os.makedirs(CACHE_DIR, exist_ok=True)

# 快篩參數
MIN_PRICE        = 10.0    # 最低股價（元）
MIN_AVG_VOLUME   = 500     # 最低均量（張）= 500,000 股
MAX_DROP_5D      = -0.12   # 近 5 日最大跌幅（超過此值淘汰）
MIN_DATA_BARS    = 60      # 最少資料筆數
STAGE1_MAX       = 250     # 第一階段最多保留支數
BATCH_SIZE       = 30      # 批次下載大小
MAX_WORKERS      = 8       # 平行下載執行緒數


def run_stage1(stocks_tw, use_cache=True):
    """
    執行第一階段技術快篩。

    stocks_tw : ["2330.TW", "2317.TW", ...] Yahoo 格式
    回傳：{
        "passed"  : [通過的代號列表（.TW 格式）],
        "scores"  : {symbol: tech_score},
        "details" : {symbol: 詳細篩選結果},
        "stats"   : 統計資訊
    }
    """
    total   = len(stocks_tw)
    logger.info("第一階段技術快篩：%d 支上市股票", total)
    print(f"\n  [第一階段] 技術快篩 {total} 支股票...")

    passed_scores = {}
    failed        = []
    details       = {}

    # 批次並行下載
    batches = [stocks_tw[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    done = 0

    for batch_idx, batch in enumerate(batches):
        pct = done / total * 100
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"\r  [{bar}] {pct:.0f}%  快篩中 {done}/{total}",
              end="", flush=True)

        # 批次下載（yfinance 支援同時下載多支）
        batch_data = _download_batch(batch)

        for sym in batch:
            df = batch_data.get(sym)
            score, detail = _evaluate_stock(sym, df)

            if score is not None:
                passed_scores[sym] = score
                details[sym]       = detail
            else:
                failed.append(sym)
                details[sym] = detail

        done += len(batch)

    print(f"\r  [{'#'*20}] 100%  快篩完成！                    ")

    # 依技術分排序，取前 STAGE1_MAX 支
    sorted_scores = sorted(passed_scores.items(),
                           key=lambda x: x[1], reverse=True)
    passed = [sym for sym, _ in sorted_scores[:STAGE1_MAX]]

    stats = {
        "total":      total,
        "passed":     len(passed),
        "failed":     len(failed),
        "eliminated": total - len(passed) - len(failed),
    }

    logger.info("第一階段完成：%d 支通過 → 進入第二階段",
                len(passed))
    print(f"\n  結果：{total} 支 → 通過 {len(passed)} 支"
          f"（資料不足/下市 {len(failed)} 支）")

    return {
        "passed":  passed,
        "scores":  passed_scores,
        "details": details,
        "stats":   stats,
    }


def _download_batch(symbols):
    """批次下載多支股票資料"""
    cache_results = {}
    to_download   = []

    # 先從快取讀
    for sym in symbols:
        cached = _load_price_cache(sym)
        if cached is not None:
            cache_results[sym] = cached
        else:
            to_download.append(sym)

    # 批次下載未快取的
    if to_download:
        try:
            raw = yf.download(
                to_download,
                period="6mo",
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )

            for sym in to_download:
                try:
                    if len(to_download) == 1:
                        df = raw.copy()
                        if df.columns.nlevels > 1:
                            df.columns = df.columns.droplevel(1)
                    else:
                        code = sym.replace(".TW","")
                        if sym in raw.columns.get_level_values(0):
                            df = raw[sym].copy()
                        elif code in raw.columns.get_level_values(0):
                            df = raw[code].copy()
                        else:
                            cache_results[sym] = None
                            continue

                    df = df.dropna()
                    if not df.empty:
                        _save_price_cache(sym, df)
                        cache_results[sym] = df
                    else:
                        cache_results[sym] = None

                except Exception:
                    cache_results[sym] = None

        except Exception as e:
            logger.debug("批次下載失敗：%s", e)
            for sym in to_download:
                cache_results[sym] = None

    return cache_results


def _evaluate_stock(symbol, df):
    """
    快速評估單支股票，回傳 (score, detail)
    score = None 表示淘汰
    """
    detail = {"symbol": symbol, "pass": False, "reason": ""}

    # 資料檢查
    if df is None or len(df) < MIN_DATA_BARS:
        detail["reason"] = f"資料不足（{len(df) if df is not None else 0}筆）"
        return None, detail

    close  = df["Close"]
    volume = df["Volume"]
    latest = float(close.iloc[-1])

    # ── 淘汰條件 ──────────────────────────────────────────────

    # 1. 股價過低
    if latest < MIN_PRICE:
        detail["reason"] = f"股價過低（{latest:.1f}元）"
        return None, detail

    # 2. 流動性不足（均量 < 500 張）
    avg_vol = float(volume.rolling(20).mean().iloc[-1] or 0)
    if avg_vol < MIN_AVG_VOLUME * 1000:   # yfinance 單位是股
        detail["reason"] = f"均量不足（{avg_vol/1000:.0f}張）"
        return None, detail

    # 3. 近期急跌
    if len(close) >= 5:
        drop_5d = float(close.iloc[-1] / close.iloc[-5] - 1)
        if drop_5d < MAX_DROP_5D:
            detail["reason"] = f"近5日急跌（{drop_5d:.1%}）"
            return None, detail

    # ── 技術評分（0~100）─────────────────────────────────────
    score = 0.0

    # MA 多頭排列（最重要）
    ma20 = float(close.rolling(20).mean().iloc[-1] or 0)
    ma60 = float(close.rolling(60).mean().iloc[-1] or 0)
    if ma20 > 0 and ma60 > 0:
        if ma20 > ma60:
            score += 30   # 多頭排列
            if latest > ma20:
                score += 10  # 站上 MA20

    # RSI（過熱或超賣都不好）
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = float((100 - 100 / (1 + rs)).iloc[-1] or 50)

    if 40 <= rsi <= 65:
        score += 25   # 強勢但未過熱的甜蜜區間
    elif 30 <= rsi < 40:
        score += 15   # 超賣反彈區
    elif rsi > 75:
        score -= 10   # 過熱扣分

    # 動能（近 20 日）
    if len(close) >= 21:
        momentum = float(close.iloc[-1] / close.iloc[-20] - 1)
        if momentum > 0.05:
            score += 20
        elif momentum > 0:
            score += 10
        elif momentum < -0.10:
            score -= 10

    # 量能（近 5 日量比）
    vol_ratio = float(
        volume.iloc[-5:].mean() / volume.rolling(20).mean().iloc[-1]
        if volume.rolling(20).mean().iloc[-1] > 0 else 1
    )
    if 1.2 <= vol_ratio <= 2.5:
        score += 15   # 溫和放大
    elif vol_ratio > 2.5:
        score += 5    # 暴量，稍扣

    # 距 60 日高點距離
    high_60d = float(close.rolling(60).max().iloc[-1] or latest)
    dist_high = (latest - high_60d) / high_60d
    if dist_high > -0.05:
        score += 10   # 接近高點，有突破潛力
    elif dist_high > -0.15:
        score += 5

    score = max(0, min(100, score))

    detail.update({
        "pass":    True,
        "score":   round(score, 1),
        "price":   round(latest, 1),
        "rsi":     round(rsi, 1),
        "ma_bull": ma20 > ma60,
        "vol_ratio": round(vol_ratio, 2),
        "reason":  f"技術分 {score:.0f}",
    })

    return score, detail


def _cache_path(symbol):
    sid = symbol.replace(".TW","").replace(".","_")
    return os.path.join(CACHE_DIR, f"{sid}_s1price.csv")

def _load_price_cache(symbol):
    path = _cache_path(symbol)
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > CACHE_HOURS * 3600:
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None

def _save_price_cache(symbol, df):
    try:
        df.to_csv(_cache_path(symbol))
    except Exception:
        pass
