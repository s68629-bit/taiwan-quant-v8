"""
籌碼資料引擎 v5.1
修正：FinMind API 回傳英文 name 欄位，更新對應表
      同時支援中英文名稱，自動合計子分類
"""
import os, time, logging
import requests
import pandas as pd
from config.settings import CACHE_DIR, CACHE_HOURS, FINMIND_TOKEN

logger = logging.getLogger(__name__)
os.makedirs(CACHE_DIR, exist_ok=True)

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

# FinMind 三大法人 name 欄位（中英文皆支援）
FOREIGN_NAMES = {"外資","外資及陸資","Foreign_Investor","Foreign_Dealer_Self"}
TRUST_NAMES   = {"投信","Investment_Trust"}
DEALER_NAMES  = {"自營商","自營商(自行買賣)","自營商(避險)","Dealer_Self","Dealer_Hedging","Dealer"}


def _cpath(symbol, tag):
    sid = symbol.replace(".TW","").replace(".","_")
    return os.path.join(CACHE_DIR, f"{sid}_{tag}.csv")

def _is_fresh(path):
    return os.path.exists(path) and \
           (time.time() - os.path.getmtime(path)) < CACHE_HOURS * 3600

def _finmind(dataset, stock_id, start_date):
    params  = {"dataset": dataset, "data_id": stock_id, "start_date": start_date}
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"} if FINMIND_TOKEN else {}
    try:
        r = requests.get(FINMIND_URL, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        d = r.json()
        if d.get("status") == 200 and d.get("data"):
            return pd.DataFrame(d["data"]), True
        logger.warning("FinMind 無資料 [%s %s]：%s", dataset, stock_id, d.get("msg",""))
        return pd.DataFrame(), False
    except Exception as e:
        logger.warning("FinMind 失敗 [%s %s]：%s", dataset, stock_id, e)
        return pd.DataFrame(), False


def get_institutional(symbol, start_date="2023-01-01"):
    cache = _cpath(symbol, "inst")
    if _is_fresh(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df, len(df) > 0

    sid = symbol.replace(".TW","")
    raw, ok = _finmind("TaiwanStockInstitutionalInvestorsBuySell", sid, start_date)
    if not ok or raw.empty:
        return pd.DataFrame(), False

    raw["date"] = pd.to_datetime(raw["date"])
    raw["buy"]  = pd.to_numeric(raw["buy"],  errors="coerce").fillna(0)
    raw["sell"] = pd.to_numeric(raw["sell"], errors="coerce").fillna(0)
    raw["net"]  = raw["buy"] - raw["sell"]

    logger.debug("%s 法人 name 清單：%s", symbol, raw["name"].unique().tolist())

    def _grp(name_set):
        mask = raw["name"].isin(name_set)
        return raw[mask].groupby("date")["net"].sum() if mask.any() else None

    frames = []
    for col, ns in [("foreign_net",FOREIGN_NAMES),("trust_net",TRUST_NAMES),("dealer_net",DEALER_NAMES)]:
        s = _grp(ns)
        if s is not None:
            frames.append(s.rename(col))

    if not frames:
        logger.warning("%s 無法對應任何法人名稱，實際值=%s", symbol, raw["name"].unique().tolist())
        return pd.DataFrame(), False

    df = pd.concat(frames, axis=1).sort_index()
    df["chip_net_total"] = df.sum(axis=1)
    df.to_csv(cache)
    time.sleep(0.5)
    return df, True


def get_margin(symbol, start_date="2023-01-01"):
    cache = _cpath(symbol, "margin")
    if _is_fresh(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df, len(df) > 0

    sid = symbol.replace(".TW","")
    raw, ok = _finmind("TaiwanStockMarginPurchaseShortSale", sid, start_date)
    if not ok or raw.empty:
        return pd.DataFrame(), False

    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date")

    result = pd.DataFrame(index=raw.index)
    for src, dst in [("MarginPurchaseTodayBalance","margin_balance"),
                     ("ShortSaleTodayBalance","short_balance")]:
        if src in raw.columns:
            result[dst] = pd.to_numeric(raw[src], errors="coerce")

    if {"margin_balance","short_balance"}.issubset(result.columns):
        total = result["margin_balance"] + result["short_balance"]
        result["margin_ratio"] = result["margin_balance"] / total.replace(0, float("nan"))

    result.to_csv(cache)
    time.sleep(0.5)
    return result, True


def diagnose_chip_data(symbol):
    _, inst_ok   = get_institutional(symbol)
    _, margin_ok = get_margin(symbol)
    status = "✅ 真實資料" if (inst_ok and margin_ok) else \
             "⚠️ 部分缺失" if (inst_ok or margin_ok) else \
             "❌ 無籌碼資料（填 0）"
    logger.info("%s 籌碼：%s", symbol, status)
    return {"inst":inst_ok,"margin":margin_ok,"chip_ok":inst_ok,"margin_ok":margin_ok,"status":status}


def test_finmind_connection():
    """完整測試 FinMind 連線與 Token 狀態"""
    result = {
        "token_set":       bool(FINMIND_TOKEN),
        "token_len":       len(FINMIND_TOKEN) if FINMIND_TOKEN else 0,
        "api_ok":          False,
        "chip_ok":         False,
        "error":           None,
        "recommendation":  "",
    }

    if not FINMIND_TOKEN:
        result["error"] = "Token 未設定"
        result["recommendation"] = "請至 FinMind 官網取得 Token 並填入 config/settings.py"
        return result

    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}

    # 測試1：基本 API
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data",
                         params={"dataset":"TaiwanStockInfo"},
                         headers=headers, timeout=15)
        d = r.json()
        if d.get("status") == 200 and d.get("data"):
            result["api_ok"] = True
        else:
            result["error"] = f"API 回應：{d.get('msg','未知錯誤')} (status={d.get('status')})"
            if d.get("status") in [401, 403] or "token" in str(d.get("msg","")).lower():
                result["recommendation"] = "Token 已失效！請重新登入 FinMind 取得新 Token，填入 config/settings.py"
            else:
                result["recommendation"] = "API 暫時無法使用，請稍後再試"
            return result
    except Exception as e:
        result["error"] = f"連線失敗：{e}"
        result["recommendation"] = "請確認網路連線是否正常"
        return result

    # 測試2：籌碼資料（台積電）
    try:
        r2 = requests.get("https://api.finmindtrade.com/api/v4/data",
                          params={"dataset":"TaiwanStockInstitutionalInvestorsBuySell",
                                  "data_id":"2330","start_date":"2025-01-01"},
                          headers=headers, timeout=15)
        d2 = r2.json()
        if d2.get("status") == 200 and d2.get("data"):
            result["chip_ok"] = True
            result["recommendation"] = "✅ FinMind 正常，籌碼資料可取得"
        else:
            result["error"] = f"籌碼資料：{d2.get('msg','')}"
            result["recommendation"] = "API 額度可能已用盡，請等下一小時自動重置（600次/小時）"
    except Exception as e:
        result["error"] = f"籌碼測試失敗：{e}"

    return result
