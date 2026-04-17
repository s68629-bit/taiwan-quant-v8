"""
登入驗證模組
帳號密碼存放在 Streamlit Secrets，不寫死在程式碼裡
"""
import streamlit as st
import hmac

def check_password():
    """
    回傳 True 代表已登入，False 代表尚未登入。
    使用 hmac.compare_digest 防止 timing attack。
    """
    def _verify(username, password):
        try:
            correct_user = st.secrets["LOGIN_USERNAME"].strip()
            correct_pass = st.secrets["LOGIN_PASSWORD"].strip()
        except Exception:
            st.error("⚠️ 系統設定錯誤：找不到帳號密碼設定，請聯絡管理員。")
            return False
        user_ok = hmac.compare_digest(username.strip().encode(), correct_user.encode())
        pass_ok = hmac.compare_digest(password.strip().encode(), correct_pass.encode())
        return user_ok and pass_ok

    # 已登入 → 直接通過
    if st.session_state.get("logged_in"):
        return True

    # 登入頁面
    st.markdown("""
    <style>
    .login-box {
        max-width: 420px;
        margin: 80px auto 0 auto;
        padding: 40px;
        background: #f8fafc;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    }
    .login-title {
        font-size: 1.6rem;
        font-weight: 900;
        color: #1a3a5c;
        text-align: center;
        margin-bottom: 6px;
    }
    .login-sub {
        font-size: 0.9rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 28px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-title">🇹🇼 Taiwan Quant Fund v8</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">台灣股票 AI 量化分析系統</div>', unsafe_allow_html=True)
        st.markdown("---")

        username = st.text_input("帳號", placeholder="請輸入帳號")
        password = st.text_input("密碼", type="password", placeholder="請輸入密碼")

        if st.button("🔐 登入", use_container_width=True, type="primary"):
            if _verify(username, password):
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("帳號或密碼錯誤，請再試一次。")

        st.markdown("---")
        st.caption("© 2026 Taiwan Quant Fund v8 ｜ 本系統僅供參考，不構成投資建議")

    return False
