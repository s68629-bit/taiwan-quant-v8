# 🇹🇼 Taiwan Quant Fund v8

台灣股票 AI 量化分析系統，採用兩階段選股架構。

## 功能
- 全市場兩階段 AI 選股掃描（~970 支）
- 飆漲前兆偵測
- 個股深度分析（四合一評分）
- 坑殺陷阱風險偵測
- 多時間窗口預測（20/40/60 日）
- 模型自我修正系統

## 本機執行

```bash
# 安裝套件
pip install -r requirements.txt

# 設定環境變數（本機用）
set FINMIND_TOKEN=你的Token     # Windows
export FINMIND_TOKEN=你的Token  # Mac/Linux

# 啟動
streamlit run dashboard/app.py
```

## 部署

詳見 `.streamlit/secrets.toml.example` 設定說明。

## 免責聲明

本系統僅供研究參考，不構成任何投資建議。
