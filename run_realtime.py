"""
即時掃描 v8 — 兩階段模式
用法：
  python run_realtime.py              → 全市場兩階段（推薦）
  python run_realtime.py stage1       → 只做技術快篩（快速預覽）
  python run_realtime.py legacy       → v7 相容模式（119 支）
  python run_realtime.py watch 3583 2412 6781  → 觀察清單模式
"""
import logging, sys, os
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.two_stage_scanner import run_two_stage_scan
from core.stock_pool import get_tier

def main():
    args = sys.argv[1:]
    mode = "full_market"
    watchlist = None

    if args:
        if args[0] == "stage1":
            mode = "stage1_only"
        elif args[0] == "legacy":
            mode = "legacy"
        elif args[0] == "watch":
            mode = "watchlist"
            watchlist = args[1:] if len(args) > 1 else []

    scan = run_two_stage_scan(mode=mode, watchlist=watchlist)

    if mode == "stage1_only":
        s1 = scan.get("stage1_result",{})
        passed = s1.get("passed",[])
        scores = s1.get("scores",{})
        details = s1.get("details",{})
        print(f"\n  技術快篩結果（前 30 名）：")
        print(f"  {'排名':>4} {'股票':>10} {'技術分':>6}  {'RSI':>5}  {'MA多頭':>6}  {'量比':>5}")
        print("  " + "─"*50)
        for i, sym in enumerate(passed[:30], 1):
            d = details.get(sym,{})
            print(f"  #{i:02d}  {sym:>10}  {scores.get(sym,0):>5.1f}  "
                  f"{d.get('rsi',0):>5.1f}  "
                  f"{'✅' if d.get('ma_bull') else '❌':>6}  "
                  f"{d.get('vol_ratio',1):>5.2f}")
        return

    # 完整掃描結果
    results = scan.get("results",[])
    if not results:
        print("無結果")
        return

    print(f"\n  {'排':>3} {'股票':>10} {'分類':>7} {'綜合':>6} "
          f"{'AI':>5} {'籌碼':>5} {'反向':>5} {'飆漲':>4} {'陷阱':>4}  訊號")
    print("  " + "─"*80)

    for rank, (sym, c) in enumerate(results, 1):
        tier = get_tier(sym) or "其他"
        sr   = c.get("surge_score_raw",0)
        tr   = c.get("trap_score",0) if "trap_score" in c else 0
        print(f"  #{rank:02d} {sym:>10} {tier:>7} "
              f"{c['composite']:>+5.2f} "
              f"{c['ai_score']:>+4.2f} "
              f"{c['chip_score']:>+4.2f} "
              f"{c['contrarian_score']:>+4.2f} "
              f"{sr:>4} "
              f"{tr:>4}  "
              f"{c['signal_icon']}{c['signal']}")

    cands = scan.get("surge_candidates",[])
    if cands:
        print(f"\n  🚀 飆漲候選 {len(cands)} 支")
        for c in cands[:5]:
            print(f"    [{c.get('tier','-')}] {c['symbol']}  "
                  f"飆漲:{c['surge_score']}/100  {c['stage']}  {c['est_days']}")

if __name__ == "__main__":
    main()
