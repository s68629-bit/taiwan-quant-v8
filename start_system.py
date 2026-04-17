import subprocess, sys, os

base = os.path.dirname(os.path.abspath(__file__))

p1 = subprocess.Popen(
    [sys.executable, os.path.join(base, "run_daily.py")], cwd=base
)
p2 = subprocess.Popen(
    ["streamlit", "run",
     os.path.join(base, "dashboard", "app.py"),
     "--server.port", "8501"],
    cwd=base,
)

print("=" * 55)
print("  Taiwan Quant Fund v5 已啟動")
print("  Dashboard  : http://localhost:8501")
print("  排程       : 每日 18:00 自動掃描")
print("  記錄檔     : daily.log")
print("  停止       : Ctrl+C")
print("=" * 55)

try:
    p1.wait()
except KeyboardInterrupt:
    p1.terminate(); p2.terminate()
    print("\n系統已停止")
