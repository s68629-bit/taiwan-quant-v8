"""
完整回測腳本 v5
流程：
  1. 大盤環境判斷
  2. 每支股票執行 Walk-forward 樣本外驗證
  3. 計算三合一評分（AI + 籌碼 + 反向訊號）
  4. 組合層級月度換倉回測（含停損截斷）
  5. 輸出 Excel 報告（5 個工作表）

執行：python run_backtest.py
耗時：首次約 5～8 分鐘（需下載籌碼資料）；快取後約 2～3 分鐘
"""
import logging, sys, os
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest.log", encoding="utf-8"),
    ],
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.stock_pool import stocks
from core.market_filter import get_market_regime
from modules.predictor import predict_with_validation
from modules.data_engine import get_price
from modules.backtest_engine import run_full_backtest
from modules.report_engine import generate_report
from config.settings import TOP_N

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 65)
    logger.info("Taiwan Quant Fund v5 ─ 完整回測啟動")
    logger.info("=" * 65)

    # ── 1. 大盤環境 ────────────────────────────────────────
    market = get_market_regime()
    logger.info("大盤環境：%s　%s", market["regime"], market["description"])

    # ── 2. 逐股 Walk-forward 驗證 + 三合一評分 ────────────
    val_results  = {}   # {sym: predict_with_validation 輸出}
    scan_results = []   # [(sym, composite_dict)]
    price_dict   = {}
    chip_status  = {}

    for sym in stocks:
        try:
            logger.info("▶ 處理 %s ...", sym)
            out = predict_with_validation(sym, market["is_bear"])

            val_results[sym]  = out
            scan_results.append((sym, out["composite_result"]))
            price_dict[sym]   = get_price(sym)
            chip_status[sym]  = out["chip_real"]

        except Exception as e:
            logger.warning("✗ 跳過 %s：%s", sym, e)

    if not scan_results:
        logger.error("所有股票均處理失敗，請確認網路及 FinMind Token")
        return

    # ── 3. 排序 ────────────────────────────────────────────
    scan_results.sort(key=lambda x: x[1]["composite"], reverse=True)

    # ── 4. 列印選股排名 ────────────────────────────────────
    logger.info("\n%s", "─"*65)
    logger.info("【三合一選股排名 Top %d】", TOP_N)
    logger.info("  %-4s %-10s %6s %6s %6s %6s  %-10s  %s",
                "排名","股票","綜合","AI","籌碼","反向","訊號","警示")
    logger.info("  " + "─"*80)
    for rank, (sym, c) in enumerate(scan_results[:TOP_N], 1):
        logger.info(
            "  #%02d %-10s %+5.2f %+5.2f %+5.2f %+5.2f  %-10s  %s",
            rank, sym,
            c["composite"], c["ai_score"], c["chip_score"], c["contrarian_score"],
            c["signal_icon"] + c["signal"],
            c["warning"][:40],
        )

    # ── 5. 組合回測 ────────────────────────────────────────
    logger.info("\n%s", "─"*65)
    logger.info("【執行組合回測（含停損）】")
    curve_df, trades_df, bt_metrics = run_full_backtest(
        [(s, val_results[s]) for s, _ in scan_results if s in val_results],
        price_dict, market,
    )

    if bt_metrics:
        logger.info("  總報酬率    : %+.1f%%", bt_metrics.get("total_return",0)*100)
        logger.info("  年化報酬率  : %+.1f%%", bt_metrics.get("ann_return",0)*100)
        logger.info("  夏普比率    : %.2f",    bt_metrics.get("sharpe",0))
        logger.info("  最大回撤    : %.1f%%",  bt_metrics.get("max_drawdown",0)*100)
        logger.info("  月勝率      : %.1f%%",  bt_metrics.get("win_rate",0)*100)
        logger.info("  交易筆數    : %d",      bt_metrics.get("n_trades",0))
        logger.info("  觸停損次數  : %d",      bt_metrics.get("stop_loss_hits",0))

    # ── 6. Excel 報告 ──────────────────────────────────────
    logger.info("\n%s", "─"*65)
    logger.info("【輸出 Excel 報告】")
    path = generate_report(
        scan_results[:TOP_N],
        curve_df, trades_df, bt_metrics,
        val_results, chip_status,
    )
    logger.info("✓ 報告已儲存：%s", path)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
