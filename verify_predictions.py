"""
預測查驗腳本（手動執行）
自動查驗所有到期的預測記錄，計算實際報酬率
執行：python verify_predictions.py
"""
import logging, sys, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.performance_analyzer import run_verification, generate_health_report
from modules.prediction_logger import get_summary

def main():
    print("\n" + "="*60)
    print("  預測查驗 & 模型健康檢查")
    print("="*60)

    # 查驗到期預測
    print("\n[1] 查驗到期預測...")
    result = run_verification()
    print(f"    查驗完成：{result['verified_count']} 筆")
    if result['errors']:
        print(f"    失敗：{len(result['errors'])} 筆")

    # 模型健康報告
    print("\n[2] 模型健康報告")
    report = generate_health_report()
    summary = get_summary()

    print(f"    總預測記錄：{report['total_predictions']} 筆")
    print(f"    已查驗：    {report['verified']} 筆")
    print(f"    待查驗：    {report['pending']} 筆")
    print(f"    狀態：      {report['status']}")

    if report['win_rate'] is not None:
        print(f"\n    整體勝率：  {report['win_rate']:.1%}")
        print(f"    平均報酬：  {report['avg_return']:+.2%}")

    # 特徵 IC 值
    ic_df = report.get("feature_ic")
    if ic_df is not None and not ic_df.empty:
        print("\n[3] 特徵有效性分析")
        print(f"    {'特徵':<20} {'IC值':>8}  建議")
        print("    " + "-"*55)
        for _, row in ic_df.iterrows():
            print(f"    {row['feature']:<20} {row['ic']:>8.4f}  {row['recommendation']}")

    # 權重建議
    ws = report.get("weight_suggestion", {})
    if ws.get("suggested"):
        print("\n[4] 評分權重建議")
        print(f"    依據：{ws.get('basis','')}")
        print(f"\n    {'維度':<15} {'目前':>8} {'建議':>8} {'變化':>8}")
        print("    " + "-"*42)
        for k, v in ws["suggested"].items():
            from config.settings import W_AI, W_CHIP, W_CONTRARIAN, W_SURGE
            curr = {"W_AI": W_AI, "W_CHIP": W_CHIP,
                    "W_CONTRARIAN": W_CONTRARIAN, "W_SURGE": W_SURGE}
            change = ws["change"].get(k, 0)
            arrow  = "↑" if change > 0.01 else ("↓" if change < -0.01 else "→")
            print(f"    {k:<15} {curr.get(k,0):>8.3f} {v:>8.3f} {arrow}{abs(change):.3f}")
        print("\n    ⚠️  以上為系統建議，是否採用請自行判斷")
        print("       若決定採用，請手動更新 config/settings.py 的權重值")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
