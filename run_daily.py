"""排程執行器 v8"""
import logging, sys, os, schedule, time
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("daily.log", encoding="utf-8")]
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.two_stage_scanner import run_two_stage_scan

logger = logging.getLogger(__name__)

def job_full():
    logger.info("盤後全市場兩階段掃描啟動")
    scan = run_two_stage_scan(mode="full_market")
    for rank, (sym, c) in enumerate(scan.get("results",[])[:20], 1):
        logger.info("#%02d %-12s 綜合:%+.2f [%s%s]",
                    rank, sym, c["composite"],
                    c["signal_icon"], c["signal"])

schedule.every().day.at("18:30").do(job_full)
logger.info("排程啟動：18:30 全市場兩階段掃描（Ctrl+C 停止）")
while True:
    schedule.run_pending()
    time.sleep(60)
