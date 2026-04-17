"""智慧掃描排程器 v7 - 整合預測記錄"""
import logging
from core.stock_pool import (TIER_1_CORE, TIER_2_SEMI_AI, TIER_3_FINANCE,
                              TIER_4_GROWTH, ALL_STOCKS, get_tier)
from modules.predictor import predict_full
from core.market_filter import get_market_regime
from modules.risk_engine import generate_risk_summary
from modules.prediction_logger import log_predictions
from config.settings import TOP_N, SURGE_SCORE_THRESHOLD

logger = logging.getLogger(__name__)

def run_scan(mode="full"):
    pool_map = {
        "full":    ALL_STOCKS,
        "core":    TIER_1_CORE,
        "semi":    TIER_2_SEMI_AI,
        "finance": TIER_3_FINANCE,
        "growth":  TIER_4_GROWTH,
        "tier12":  list(dict.fromkeys(TIER_1_CORE + TIER_2_SEMI_AI)),
    }
    pool   = pool_map.get(mode, ALL_STOCKS)
    market = get_market_regime()
    logger.info("大盤：%s｜模式：%s｜股票數：%d", market["regime"], mode, len(pool))

    results, chip_status, surge_candidates, failed = [], {}, [], []
    full_results_for_log = []

    total = len(pool)
    for idx, sym in enumerate(pool, 1):
        if idx % 20 == 0:
            logger.info("進度：%d / %d（%.0f%%）", idx, total, idx/total*100)
        try:
            out = predict_full(sym, is_bear_market=market["is_bear"])
            c   = out["composite_result"]
            results.append((sym, c, out))
            chip_status[sym] = out["chip_real"]
            full_results_for_log.append((sym, c, out))

            surge = out.get("surge", {})
            trap  = out.get("trap", {})
            if surge.get("surge_score", 0) >= SURGE_SCORE_THRESHOLD:
                surge_candidates.append({
                    "symbol":      sym, "tier": get_tier(sym),
                    "surge_score": surge["surge_score"],
                    "stage":       surge["stage_label"],
                    "est_days":    surge["est_days_label"],
                    "vol_quality": surge["volume_quality"],
                    "signals":     surge["signals"],
                    "multi_h":     out.get("multi_horizon", {}),
                    "foreign_cost":out.get("foreign_cost"),
                    "retail_temp": out.get("retail_sentiment",{}).get("temperature", 50),
                    "trap_score":  trap.get("trap_score", 0),
                    "trap_level":  trap.get("trap_level", "🟢 安全"),
                    "trap_signals":trap.get("signals", []),
                    "entry_ok":    trap.get("entry_ok", True),
                })
        except Exception as e:
            logger.warning("✗ 跳過 %s：%s", sym, e)
            failed.append(sym)

    results.sort(key=lambda x: x[1]["composite"], reverse=True)
    surge_candidates.sort(key=lambda x: x["surge_score"], reverse=True)
    risk = generate_risk_summary([(s,c) for s,c,_ in results], market)

    # ── v7 新增：記錄預測結果 ─────────────────────────────────
    try:
        log_predictions([(s,c) for s,c,_ in results], full_results_for_log)
    except Exception as e:
        logger.warning("預測記錄失敗（不影響掃描）：%s", e)

    tier_stats = _tier_stats(results)

    return {
        "results":          [(s,c) for s,c,_ in results[:TOP_N]],
        "results_full":     results[:TOP_N],
        "all_results":      results,
        "surge_candidates": surge_candidates,
        "market_regime":    market,
        "risk_summary":     risk,
        "chip_status":      chip_status,
        "scan_mode":        mode,
        "total_scanned":    len(results),
        "failed":           failed,
        "tier_stats":       tier_stats,
    }

def _tier_stats(results):
    import numpy as np
    stats = {"權值股":[], "半導體/AI":[], "金融股":[], "成長股":[], "其他":[]}
    for sym, c, _ in results:
        t = get_tier(sym)
        if t not in stats: t = "其他"
        stats[t].append(c["composite"])
    return {t: {"count": len(s), "avg": round(float(np.mean(s)),3)}
            for t, s in stats.items() if s}
