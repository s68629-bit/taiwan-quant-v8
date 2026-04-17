@echo off
chcp 65001 >nul
title Taiwan Quant Fund v8

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║    Taiwan Quant Fund v8  啟動中...           ║
echo  ║    AI 全市場兩階段量化選股系統               ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: 切換到正確路徑
cd /d "D:\Test\taiwan_quant_fund_v8\taiwan_quant_v8"

:: 確認 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo  ❌ 找不到 Python，請確認已安裝並加入 PATH
    pause
    exit
)

:: 確認 streamlit 是否已安裝
python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo  ⚙️  安裝必要套件中，請稍候...
    pip install -r requirements.txt
    echo.
)

:: 建立必要資料夾
if not exist "cache"   mkdir cache
if not exist "reports" mkdir reports
if not exist "data"    mkdir data

echo  ✅ 環境確認完成
echo.
echo  🌐 開啟 Dashboard：http://localhost:8501
echo  📊 關閉系統請按 Ctrl+C 或直接關閉此視窗
echo.

:: 啟動 Streamlit
python -m streamlit run dashboard/app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
