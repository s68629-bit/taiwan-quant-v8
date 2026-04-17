import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.two_stage_scanner import run_two_stage_scan
from core.stock_pool import get_tier
from dashboard.login import check_password

st.set_page_config(page_title="Taiwan Quant Fund v8", page_icon="🇹🇼", layout="wide")

# ── 登入驗證（必須通過才能看到後面的內容）──────────────────────
if not check_password():
    st.stop()

# ── 登出按鈕（右上角）─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🇹🇼 Taiwan Quant Fund v8")
    st.markdown("---")
    if st.button("🚪 登出", use_container_width=True):
        st.session_state["logged_in"] = False
        st.rerun()

st.title("🇹🇼 Taiwan Quant Fund v8　全市場兩階段 AI 量化選股")
st.caption("第一階段：技術快篩 ~970 支　→　第二階段：籌碼深度分析 ~250 支　→　最終前 20 名")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 全市場掃描", "🚀 飆漲候選", "🔍 個股查詢",
    "🧠 模型健康", "⚠️ 風控", "📈 回測報告"
])

if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

# ── Tab 1：全市場掃描 ──────────────────────────────────────────
with tab1:
    col_l, col_m, col_r = st.columns([2, 1, 1])
    with col_l:
        st.subheader("兩階段 AI 選股掃描")
    with col_m:
        mode_opts = [
            ("full_market", "🌐 全市場（推薦）"),
            ("stage1_only", "⚡ 技術快篩預覽"),
            ("legacy",      "📦 相容模式 119 支"),
        ]
        mode = st.selectbox("掃描模式", mode_opts, format_func=lambda x: x[1])
    with col_r:
        st.write("")
        run_btn = st.button("🔍 執行掃描", type="primary", use_container_width=True)
        if st.button("🔌 測試 FinMind", use_container_width=True):
            from modules.chipdata_engine import test_finmind_connection
            fm = test_finmind_connection()
            if fm["chip_ok"]:
                st.success(f"✅ FinMind 正常（Token {fm['token_len']} 字元）")
            elif fm["api_ok"]:
                st.warning(f"🟡 API 連線OK但籌碼失敗：{fm.get('error','')} | {fm.get('recommendation','')}")
            else:
                st.error(f"❌ {fm.get('error','')} | 建議：{fm.get('recommendation','')}")

    # 流程說明
    with st.expander("📋 兩階段流程說明", expanded=False):
        st.markdown("""
        **第一階段（技術快篩）**　yfinance 下載，無 API 限制
        - 掃描全市場約 970 支上市股票
        - 快速計算 MA、RSI、動能、量能評分
        - 過濾股價過低、流動性差、急跌中的股票
        - 保留技術分前 250 支進入第二階段

        **第二階段（籌碼深度分析）**　使用 FinMind API
        - 對 250 支候選股下載外資/投信/融資資料
        - 執行完整四合一評分（AI+籌碼+反向+飆漲）
        - 陷阱偵測（假突破、借券異常、KD背離等）
        - 輸出最終前 20 名

        **首次執行**約需 40～60 分鐘；快取建立後約 15 分鐘。
        """)

    if run_btn:
        # 先檢查 FinMind 狀態
        from modules.chipdata_engine import test_finmind_connection
        fm = test_finmind_connection()
        if not fm["chip_ok"]:
            if not fm["api_ok"]:
                st.error(f"⚠️ FinMind 連線失敗：{fm.get('error','')}　→　{fm.get('recommendation','')}　籌碼將以 0 填入，建議先修正 Token 再掃描")
            else:
                st.warning(f"🟡 FinMind 連線正常但籌碼資料異常：{fm.get('error','')}　→　{fm.get('recommendation','')}")
        with st.spinner("兩階段掃描中，請稍候（首次較慢）..."):
            st.session_state.scan_result = run_two_stage_scan(mode=mode[0])

    scan = st.session_state.scan_result
    if scan:
        mkt  = scan.get("market_regime",{})
        risk = scan.get("risk_summary",{})

        rc = {"多頭":"success","空頭":"error","整理":"warning"}.get(
            mkt.get("regime",""), "info")
        getattr(st, rc)(f"📊 大盤：{mkt.get('regime','?')}　|　{mkt.get('description','')}")

        # 兩階段統計
        s1 = scan.get("stage1_result",{})
        s1s = s1.get("stats",{}) if s1 else {}
        if s1s:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("全市場上市股", f"{s1s.get('total',0)} 支")
            c2.metric("第一階段通過", f"{s1s.get('passed',0)} 支", "技術快篩")
            c3.metric("第二階段完成", f"{scan.get('total_scanned',0)} 支", "籌碼分析")
            c4.metric("最終候選", f"{min(len(scan.get('results',[])),20)} 支", "前20名")

        for a in risk.get("advice",[]):
            if "🔴" in a: st.error(a)
            elif "🟡" in a: st.warning(a)
            else: st.success(a)

        sc = len(scan.get("surge_candidates",[]))
        if sc: st.info(f"🚀 飆漲候選 {sc} 支，請切換至「飆漲候選」頁籤")
        st.caption(f"掃描完成：第一階段 {s1s.get('passed',0)} 支 → 第二階段 {scan.get('total_scanned',0)} 支 | 預測已記錄")
        st.markdown("---")

        rows = []
        for sym, c in scan.get("results",[]):
            rows.append({"股票":sym,"分類":get_tier(sym) or "其他",
                "綜合":f"{c['composite']:+.2f}","AI":f"{c['ai_score']:+.2f}",
                "籌碼":f"{c['chip_score']:+.2f}","反向":f"{c['contrarian_score']:+.2f}",
                "飆漲":f"{c.get('surge_score_raw',0)}/100",
                "訊號":c["signal_icon"]+" "+c["signal"],"說明":c["warning"]})
        if rows:
            df = pd.DataFrame(rows); df.index = range(1, len(df)+1)
            st.dataframe(df, use_container_width=True, height=500)

        # 如果是 stage1_only，顯示技術快篩結果
        if scan.get("scan_mode") == "stage1_only" and s1:
            st.markdown("### 技術快篩結果（前 50 名）")
            scores  = s1.get("scores",{})
            passed  = s1.get("passed",[])
            details = s1.get("details",{})
            rows2 = []
            for sym in passed[:50]:
                d = details.get(sym,{})
                rows2.append({"股票":sym,"技術分":f"{scores.get(sym,0):.1f}",
                    "RSI":f"{d.get('rsi',0):.1f}","MA多頭":"✅" if d.get("ma_bull") else "❌",
                    "量比":f"{d.get('vol_ratio',1):.2f}","現價":d.get("price","")})
            df2 = pd.DataFrame(rows2); df2.index = range(1, len(df2)+1)
            st.dataframe(df2, use_container_width=True, height=400)

# ── Tab 2：飆漲候選 ───────────────────────────────────────────
with tab2:
    st.subheader("🚀 飆漲前兆候選（兩階段雙重確認）")
    scan = st.session_state.scan_result
    if not scan:
        st.info("請先執行掃描")
    else:
        cands = scan.get("surge_candidates",[])
        if not cands:
            st.warning("目前無飆漲候選（分數 ≥ 60）")
        else:
            tf = st.multiselect("篩選分類",
                ["全部","權值股","半導體/AI","金融股","成長股","其他"],
                default=["全部"])
            filtered = cands if "全部" in tf else [
                c for c in cands if c.get("tier","") in tf]
            st.caption(f"顯示 {len(filtered)} / {len(cands)} 支候選")
            for i,c in enumerate(filtered,1):
                score = c["surge_score"]
                trap_ok = c.get("entry_ok",True)
                icon = ("🔥" if score>=80 else "🚀") + ("" if trap_ok else "⚠️")
                with st.expander(
                    f"{icon} #{i} [{c.get('tier','-')}] {c['symbol']}  "
                    f"{score}/100分  {c.get('trap_level','?')}  {c['est_days']}",
                    expanded=(i<=3)):
                    c1,c2,c3 = st.columns(3)
                    c1.metric("飆漲分數", f"{score}/100")
                    c1.metric("量能品質", c.get("vol_quality","N/A"))
                    c1.metric("散戶溫度", f"{c.get('retail_temp',0):.0f}°")
                    c1.metric("技術快篩分", f"{c.get('s1_score',0):.0f}/100")
                    fc = c.get("foreign_cost") or {}
                    if fc.get("cost_60d"):
                        c2.metric("外資成本線", f"{fc['cost_60d']:.1f}", fc.get("gap_label",""))
                        c2.caption(fc.get("position_label",""))
                        c2.caption(f"撐盤區：{fc.get('support_low')} ~ {fc.get('support_high')}")
                    mh = c.get("multi_h") or {}; hs = mh.get("horizons",{})
                    if hs:
                        c3.markdown("**多時間窗口預測**")
                        for h in [20,40,60]:
                            hd = hs.get(f"h{h}")
                            if hd: c3.metric(f"{h}日預測", hd["pred_pct"], hd.get("label",""))
                    st.markdown("**飆漲訊號：**")
                    for sig in c.get("signals",[]): st.write(f"  {sig}")
                    trap_sigs = c.get("trap_signals",[])
                    if trap_sigs:
                        st.markdown("**陷阱警示：**")
                        for s in trap_sigs: st.write(f"  {s}")
                    if fc.get("implication"): st.info(f"💡 {fc['implication']}")

# ── Tab 3：個股查詢 ───────────────────────────────────────────
with tab3:
    st.subheader("🔍 個股深度分析（不受 119 支限制）")
    col_in, col_btn = st.columns([3,1])
    with col_in:
        sym_input = st.text_input("輸入任意台股代號（例：8069 元太、4958 臻鼎-KY）",
                                   placeholder="8069")
    with col_btn:
        st.write("")
        query_btn = st.button("🔍 深度分析", type="primary", use_container_width=True)

    if query_btn and sym_input:
        with st.spinner(f"分析 {sym_input}..."):
            from modules.stock_query import query_stock
            qr = query_stock(sym_input)
        if qr.get("error"):
            st.error(f"查詢失敗：{qr['error']}")
        else:
            c    = qr.get("composite_result",{})
            ee   = qr.get("entry_exit",{})
            fb   = qr.get("foreign_behavior",{})
            trap = qr.get("trap",{})
            mh   = qr.get("multi_horizon",{})
            hist = qr.get("history",{})
            surge= qr.get("surge",{})
            rs   = qr.get("retail_sentiment",{})

            st.markdown(f"## {qr['symbol']}　現價：{qr.get('current_price','N/A')}")
            st.caption(f"大盤：{qr.get('market',{}).get('regime','?')}　|　"
                       f"籌碼：{'✅' if qr.get('chip_data_ok') else '⚠️ 部分缺失'}")

            st.markdown("### 📊 四合一評分")
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("綜合評分",f"{c.get('composite',0):+.2f}",c.get("signal",""))
            m2.metric("AI分",    f"{c.get('ai_score',0):+.2f}")
            m3.metric("籌碼分",  f"{c.get('chip_score',0):+.2f}")
            m4.metric("反向分",  f"{c.get('contrarian_score',0):+.2f}")
            m5.metric("飆漲分",  f"{surge.get('surge_score',0)}/100")
            st.markdown("---")

            ca, cb = st.columns(2)
            with ca:
                st.markdown("### 🎯 進出場建議")
                adv = ee.get("entry_advice","")
                if "🟢" in adv: st.success(adv)
                elif "🟡" in adv: st.warning(adv)
                else: st.error(adv)
                st.metric("建議進場區", ee.get("entry_zone","N/A"))
                st.metric("停損設定", f"{ee.get('stop_loss_price','N/A')} 元（{ee.get('stop_loss_rate','')}）")
                if ee.get("target_20d"): st.metric("20日目標價",f"{ee['target_20d']} 元")
                if ee.get("target_60d"): st.metric("60日目標價",f"{ee['target_60d']} 元")
                st.markdown("### 📈 多時間窗口")
                hs = mh.get("horizons",{})
                for h in [20,40,60]:
                    hd = hs.get(f"h{h}")
                    if hd: st.metric(f"{h}日預測",hd["pred_pct"],hd.get("label",""))
            with cb:
                st.markdown("### 🏦 外資行為")
                st.write(f"**建倉：** {fb.get('accumulation_desc','無資料')}")
                st.write(f"**成本線：** {fb.get('cost_60d','N/A')} 元　{fb.get('gap_label','')}")
                st.write(f"**位置：** {fb.get('position_label','')}")
                st.write(f"**近期：** {fb.get('recent_direction','')}")
                if fb.get("implication"): st.info(fb["implication"])
                st.markdown("### 🛡️ 坑殺風險")
                ts = trap.get("trap_score",0)
                st.metric("陷阱危險分",f"{ts}/100",trap.get("trap_level",""))
                for sig in trap.get("signals",[]): st.write(sig)
            st.markdown("---")
            st.markdown("### 📋 歷史預測績效")
            if hist.get("count",0)==0:
                st.info("尚無歷史記錄（執行掃描並等 20 天查驗後才會出現）")
            else:
                h1,h2,h3 = st.columns(3)
                h1.metric("預測次數",hist["count"])
                h2.metric("勝率",f"{hist['win_rate']:.1%}" if hist.get("win_rate") else "N/A")
                h3.metric("平均報酬",f"{hist['avg_return']:+.2%}" if hist.get("avg_return") else "N/A")
                if hist.get("records"):
                    st.dataframe(pd.DataFrame(hist["records"]),
                                 use_container_width=True, hide_index=True)

# ── Tab 4：模型健康 ───────────────────────────────────────────
with tab4:
    st.subheader("🧠 模型健康儀表板（自我修正）")
    from modules.performance_analyzer import generate_health_report
    col_v, _ = st.columns(2)
    with col_v:
        if st.button("🔄 執行查驗"):
            with st.spinner("查驗中..."):
                from modules.performance_analyzer import run_verification
                vr = run_verification()
            st.success(f"查驗完成：{vr['verified_count']} 筆")

    report = generate_health_report()
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("總預測記錄", report["total_predictions"])
    m2.metric("已查驗",     report["verified"])
    m3.metric("整體勝率",   f"{report['win_rate']:.1%}" if report.get("win_rate") else "累積中")
    m4.metric("平均報酬",   f"{report['avg_return']:+.2%}" if report.get("avg_return") else "累積中")
    st.info(f"狀態：{report['status']}")
    ic_df = report.get("feature_ic")
    if ic_df is not None and not ic_df.empty:
        st.markdown("#### 特徵有效性")
        st.dataframe(ic_df, use_container_width=True, hide_index=True)
        remove_list = ic_df[ic_df["ic"]<0.02]["feature"].tolist()
        if remove_list: st.warning(f"建議移除：{', '.join(remove_list)}")
    ws = report.get("weight_suggestion",{})
    if ws.get("suggested"):
        st.markdown("#### 權重建議（系統建議，人工決定是否採用）")
        st.caption(ws.get("basis",""))
        from config.settings import W_AI,W_CHIP,W_CONTRARIAN,W_SURGE
        curr = {"W_AI":W_AI,"W_CHIP":W_CHIP,"W_CONTRARIAN":W_CONTRARIAN,"W_SURGE":W_SURGE}
        rows = [{"維度":k,"目前":f"{curr.get(k,0):.3f}","建議":f"{v:.3f}",
            "變化":f"{'↑' if ws['change'].get(k,0)>0.01 else '↓' if ws['change'].get(k,0)<-0.01 else '→'}{abs(ws['change'].get(k,0)):.3f}"}
            for k,v in ws["suggested"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.warning("若決定採用，請手動更新 config/settings.py")

# ── Tab 5：風控 ──────────────────────────────────────────────
with tab5:
    st.subheader("風控設定說明")
    from config.settings import STOP_LOSS_RATE,MAX_POSITION_PCT,PORTFOLIO_SIZE
    c1,c2,c3 = st.columns(3)
    c1.metric("單筆停損線",f"{STOP_LOSS_RATE:.0%}")
    c2.metric("單股最高部位",f"{MAX_POSITION_PCT:.0%}")
    c3.metric("最大同時持股數",f"{PORTFOLIO_SIZE} 支")
    st.markdown("---")
    st.subheader("停損試算")
    entry = st.number_input("買入價格", value=100.0, step=1.0)
    cur   = st.number_input("目前價格", value=93.0,  step=1.0)
    if entry > 0:
        from modules.risk_engine import check_stop_loss
        res = check_stop_loss(entry, cur)
        if res["triggered"]: st.error(f"報酬率 {res['return_rate']:.2%}　|　{res['action']}")
        else: st.success(f"報酬率 {res['return_rate']:.2%}　|　{res['action']}")

# ── Tab 6：回測報告 ──────────────────────────────────────────
with tab6:
    st.subheader("完整回測報告")

    # ── 執行回測按鈕 ────────────────────────────────────────
    col_rb1, col_rb2 = st.columns(2)
    with col_rb1:
        if st.button("🚀 執行完整回測（含 Excel 報告）", type="primary", use_container_width=True):
            with st.spinner("回測執行中，請稍候（約 3～5 分鐘）..."):
                try:
                    import subprocess, sys
                    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    result = subprocess.run(
                        [sys.executable, os.path.join(base, "run_backtest.py")],
                        cwd=base, capture_output=True, text=True,
                        encoding='utf-8', errors='replace', timeout=600
                    )
                    if result.returncode == 0:
                        st.success("✅ 回測完成！請點下方下載報告")
                        st.rerun()
                    else:
                        st.error(f"回測失敗：{result.stderr[-500:] if result.stderr else '未知錯誤'}")
                except subprocess.TimeoutExpired:
                    st.warning("⏰ 回測超時（超過 10 分鐘），請改用終端機執行：python run_backtest.py")
                except Exception as e:
                    st.error(f"執行失敗：{e}")

    with col_rb2:
        if st.button("🔄 查驗預測結果（更新模型健康）", use_container_width=True):
            with st.spinner("查驗中..."):
                try:
                    from modules.performance_analyzer import run_verification
                    vr = run_verification()
                    st.success(f"✅ 查驗完成：{vr['verified_count']} 筆，切換至「模型健康」頁籤查看")
                except Exception as e:
                    st.error(f"查驗失敗：{e}")

    st.markdown("---")

    # ── 報告列表 ─────────────────────────────────────────────
    report_dir = "reports"
    if os.path.exists(report_dir):
        files = sorted([f for f in os.listdir(report_dir)
                        if f.endswith(".xlsx") and not f.startswith("~$")],reverse=True)
        if files:
            st.markdown(f"**最新報告：** {files[0]}")
            for fname in files[:5]:
                fpath = os.path.join(report_dir, fname)
                col_name, col_dl = st.columns([3,1])
                col_name.write(fname)
                with open(fpath,"rb") as f:
                    col_dl.download_button(
                        "⬇️ 下載", f,
                        file_name=fname,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_{fname}"
                    )
        else:
            st.info("尚無報告，請點上方「執行完整回測」按鈕產生")

    # ── 歷史買點回測 ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📅 個股歷史買點回測")
    col_sym, col_bt = st.columns([3,1])
    with col_sym:
        bt_symbol = st.text_input("輸入股票代號（例：8069 元太）",
                                   placeholder="8069", key="bt_sym")
    with col_bt:
        st.write("")
        if st.button("▶ 執行歷史回測", use_container_width=True):
            if bt_symbol:
                with st.spinner(f"分析 {bt_symbol} 過去 2 年買點，約 3～5 分鐘..."):
                    try:
                        import subprocess, sys
                        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        result = subprocess.run(
                            [sys.executable,
                             os.path.join(base, "historical_signal_backtest.py"),
                             bt_symbol],
                            cwd=base, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=600
                        )
                        if result.returncode == 0:
                            st.success(f"✅ {bt_symbol} 歷史買點回測完成！請下載報告查看")
                            st.rerun()
                        else:
                            st.error(f"回測失敗：{result.stderr[-300:] if result.stderr else '未知錯誤'}")
                    except subprocess.TimeoutExpired:
                        st.warning("⏰ 回測超時，請改用終端機：python historical_signal_backtest.py 8069")
                    except Exception as e:
                        st.error(f"執行失敗：{e}")
            else:
                st.warning("請輸入股票代號")
