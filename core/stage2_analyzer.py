"""
第二階段：籌碼深度分析引擎（v8 核心）
═══════════════════════════════════════════════════════════════
對第一階段通過的 ~250 支股票做完整四合一分析
使用 FinMind API（有額度限制，精準分配使用）

執行流程：
  1. 批次下載籌碼資料（含速率控制，不超過 600 次/小時）
  2. 執行完整四合一評分（AI+籌碼+反向+飆漲）
  3. 陷阱偵測（假突破、借券、KD背離等）
  4. 輸出最終排名（取前 TOP_N 支）

速率控制：
  250 支 × 2 次（外資+融資）= 500 次
  600 次/小時額度 → 安全邊際 100 次
  每次 API 呼叫後等待 0.5 秒 → 約 8 分鐘完成籌碼下載
═══════════════════════════════════════════════════════════════
"""
import logging
import pandas as pd
from modules.predictor import predict_full
from modules.risk_engine import generate_risk_summary
from modules.prediction_logger import log_predictions
from core.market_filter import get_market_regime
from core.stock_pool import get_tier
from config.settings import TOP_N, SURGE_SCORE_THRESHOLD

logger = logging.getLogger(__name__)


def run_stage2(stage1_result, is_bear_market=False):
    """
    對第一階段通過的股票執行完整分析。

    stage1_result : stage1_screener.run_stage1() 的輸出
    回傳：與 smart_scheduler.run_scan() 相同格式
    """
    candidates = stage1_result["passed"]
    s1_scores  = stage1_result["scores"]
    total      = len(candidates)

    logger.info("第二階段深度分析：%d 支候選股", total)
    print(f"\n  [第二階段] 籌碼深度分析 {total} 支候選股...")
    print(f"  預計消耗 FinMind API：約 {total*2} 次（額度 600 次/小時）\n")

    results, chip_status, surge_candidates = [], {}, []
    failed, full_results_for_log           = [], []

    for idx, sym in enumerate(candidates, 1):
        pct = idx / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r  [{bar}] {pct:.0f}%  {sym:<12} "
              f"({idx}/{total})", end="", flush=True)

        try:
            out = predict_full(sym, is_bear_market=is_bear_market)
            c   = out["composite_result"]

            # 整合第一階段技術分（加權混合）
            s1_score  = s1_scores.get(sym, 50) / 100   # 正規化到 0~1
            c["s1_tech_score"] = round(s1_score, 3)

            results.append((sym, c, out))
            chip_status[sym] = out["chip_real"]
            full_results_for_log.append((sym, c, out))

            surge = out.get("surge", {})
            trap  = out.get("trap", {})
            if surge.get("surge_score", 0) >= SURGE_SCORE_THRESHOLD:
                surge_candidates.append({
                    "symbol":      sym,
                    "tier":        get_tier(sym),
                    "surge_score": surge["surge_score"],
                    "stage":       surge["stage_label"],
                    "est_days":    surge["est_days_label"],
                    "vol_quality": surge["volume_quality"],
                    "signals":     surge["signals"],
                    "multi_h":     out.get("multi_horizon", {}),
                    "foreign_cost":out.get("foreign_cost"),
                    "retail_temp": out.get("retail_sentiment",{}).get("temperature",50),
                    "trap_score":  trap.get("trap_score", 0),
                    "trap_level":  trap.get("trap_level", "🟢 安全"),
                    "trap_signals":trap.get("signals", []),
                    "entry_ok":    trap.get("entry_ok", True),
                    "s1_score":    s1_scores.get(sym, 0),
                })

        except Exception as e:
            logger.debug("跳過 %s：%s", sym, e)
            failed.append(sym)

    print(f"\r  [{'█'*20}] 100%  深度分析完成！             ")
    print(f"\n  結果：{total} 支分析 → {len(results)} 支有效"
          f"（跳過 {len(failed)} 支）\n")

    # 依綜合分排序
    results.sort(key=lambda x: x[1]["composite"], reverse=True)
    surge_candidates.sort(key=lambda x: x["surge_score"], reverse=True)

    market = get_market_regime()
    risk   = generate_risk_summary([(s,c) for s,c,_ in results], market)

    # 記錄預測
    try:
        log_predictions(
            [(s,c) for s,c,_ in results],
            full_results_for_log
        )
    except Exception as e:
        logger.debug("預測記錄失敗（不影響掃描）：%s", e)

    # 分層統計
    tier_stats = _tier_stats(results)

    return {
        "results":          [(s,c) for s,c,_ in results[:TOP_N]],
        "results_full":     results[:TOP_N],
        "all_results":      results,
        "surge_candidates": surge_candidates,
        "market_regime":    market,
        "risk_summary":     risk,
        "chip_status":      chip_status,
        "failed":           failed,
        "total_scanned":    len(results),
        "stage1_count":     len(candidates),
        "tier_stats":       tier_stats,
    }


def _tier_stats(results):
    import numpy as np
    stats = {}
    for sym, c, _ in results:
        tier = get_tier(sym) or "其他"
        if tier not in stats:
            stats[tier] = []
        stats[tier].append(c["composite"])
    return {t: {"count": len(s), "avg": round(float(np.mean(s)), 3)}
            for t, s in stats.items() if s}
