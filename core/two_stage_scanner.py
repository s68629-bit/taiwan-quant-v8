"""
兩階段掃描協調器（v8 核心）
===============================================================
整合第一階段（技術快篩）和第二階段（籌碼深度分析）

四種掃描模式：
  "full_market"  全市場兩階段掃描（每日盤後主要模式）
  "stage1_only"  只做技術快篩（快速預覽，不消耗 FinMind）
  "watchlist"    只掃指定的觀察清單（精確分析）
  "legacy"       沿用 v7 的固定 119 支池（向後相容）
===============================================================
"""
import logging
import time
from core.market_universe import get_listed_stocks, get_stock_tw_format
from core.stage1_screener import run_stage1
from core.stage2_analyzer import run_stage2
from core.smart_scheduler import run_scan as legacy_scan
from core.market_filter import get_market_regime
from config.settings import TOP_N

logger = logging.getLogger(__name__)


def run_two_stage_scan(mode="full_market", watchlist=None):
    """
    兩階段掃描主入口。

    mode：
      "full_market"  全市場掃描（推薦，每日使用）
      "stage1_only"  只做技術快篩（快速、不消耗 FinMind）
      "watchlist"    指定觀察清單的深度分析
      "legacy"       v7 相容模式（固定 119 支）

    回傳：標準掃描結果 dict（與 v7 相容）
    """
    start_time = time.time()
    market     = get_market_regime()

    print(f"\n{'='*65}")
    print(f"  Taiwan Quant Fund v8  ─  兩階段 AI 量化選股")
    print(f"  大盤環境：{market['regime']}　|　{market['description']}")
    print(f"{'='*65}")

    if mode == "legacy":
        print("\n  [相容模式] 使用 v7 固定股票池（119 支）")
        return legacy_scan("full")

    if mode == "watchlist" and watchlist:
        print(f"\n  [觀察清單模式] 直接分析指定 {len(watchlist)} 支股票")
        # 把觀察清單包裝成 stage1 格式，跳過技術篩選
        s1_fake = {
            "passed": [s if s.endswith(".TW") else s+".TW" for s in watchlist],
            "scores": {s: 80 for s in watchlist},
            "details": {},
            "stats": {"total": len(watchlist), "passed": len(watchlist),
                      "failed": 0, "eliminated": 0},
        }
        result = run_stage2(s1_fake, market["is_bear"])
        _print_summary(result, time.time() - start_time)
        return result

    # ── 全市場兩階段掃描 ──────────────────────────────────────
    print("\n  準備全市場股票清單...")
    all_stocks = get_listed_stocks()
    stocks_tw  = get_stock_tw_format(all_stocks)
    print(f"  上市股票：{len(stocks_tw)} 支")

    if mode == "stage1_only":
        # 只做第一階段，不做籌碼分析
        s1 = run_stage1(stocks_tw)
        elapsed = time.time() - start_time
        print(f"\n  ✅ 第一階段完成（{elapsed:.0f}秒）")
        print(f"  通過技術快篩：{s1['stats']['passed']} 支")
        print(f"  （執行完整掃描請改用 full_market 模式）\n")
        # 回傳簡化格式
        return {
            "results":          [],
            "surge_candidates": [],
            "market_regime":    market,
            "risk_summary":     {"advice": ["第一階段快篩完成，請執行完整掃描"]},
            "chip_status":      {},
            "total_scanned":    s1["stats"]["passed"],
            "stage1_result":    s1,
            "scan_mode":        "stage1_only",
        }

    # ── 完整兩階段 ────────────────────────────────────────────
    # 第一階段
    s1_result = run_stage1(stocks_tw)

    # 第二階段
    result = run_stage2(s1_result, market["is_bear"])
    result["stage1_result"] = s1_result
    result["scan_mode"]     = "full_market"

    _print_summary(result, time.time() - start_time)
    return result


def _print_summary(result, elapsed):
    """掃描完成後的摘要輸出"""
    s1 = result.get("stage1_result", {})
    s1_stats = (s1.get("stats") or {})

    print(f"\n{'─'*65}")
    print(f"  掃描摘要")
    print(f"{'─'*65}")
    if s1_stats:
        print(f"  第一階段（技術快篩）：{s1_stats.get('total',0)} 支 "
              f"→ 通過 {s1_stats.get('passed',0)} 支")
    print(f"  第二階段（籌碼分析）：有效 {result.get('total_scanned',0)} 支")
    print(f"  最終選股前 {TOP_N} 名：")

    for rank, (sym, c) in enumerate(result.get("results",[])[:10], 1):
        tier = c.get("tier","") or ""
        print(f"    #{rank:02d}  {sym:<12} "
              f"綜合:{c['composite']:+.2f}  "
              f"[{c['signal_icon']}{c['signal']}]")

    sc = len(result.get("surge_candidates",[]))
    if sc:
        print(f"\n  🚀 飆漲候選：{sc} 支")

    print(f"\n  總耗時：{elapsed/60:.1f} 分鐘")
    print(f"{'='*65}\n")
