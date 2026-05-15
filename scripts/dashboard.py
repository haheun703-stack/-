"""퀀트봇 실시간 웹 대시보드 (Streamlit).

페이퍼 트레이딩 + bluechip + ETF + 위험감지 + 장중 학습 통합 대시보드.

실행:
    streamlit run scripts/dashboard.py --server.port 8501

접속:
    SSH 터널: ssh -L 8501:localhost:8501 ubuntu@13.209.153.221
    로컬: http://localhost:8501
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"

# 페이지 설정
st.set_page_config(
    page_title="퀀트봇 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 헤더 ─────────────────────────────────────────
st.title("📊 퀀트봇 자비스 대시보드")
st.caption(f"최종 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── 사이드바: 위험감지 (P0-7) ────────────────────
with st.sidebar:
    st.header("🛡️ 위험감지 (P0-7)")
    try:
        from src.utils.risk_gate import get_risk_status_safe, get_position_multiplier_safe
        rs = get_risk_status_safe()
        if rs:
            level_kr = rs.get("level_kr", "정상")
            score = rs.get("total_score", 0)
            mult = get_position_multiplier_safe()
            emoji = {"위기": "🔴", "위험": "🟠", "경고": "🟡", "주의": "🟢", "정상": "✅"}.get(level_kr, "ℹ️")
            st.metric(label=f"{emoji} 등급", value=level_kr, delta=f"{score}점")
            st.metric(label="매수금액 배수", value=f"×{mult}")
            st.caption(f"📅 {rs.get('date', '-')}")
            with st.expander("핵심 시그널"):
                for sig in rs.get("key_signals", [])[:6]:
                    st.write(f"• {sig}")
            st.info(rs.get("recommended_action", ""))
        else:
            st.warning("위험감지 데이터 없음")
    except Exception as e:
        st.error(f"위험감지 로드 실패: {e}")

# ── 메인 탭 ──────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💼 페이퍼 통합", "🎯 시그널 통계", "🧠 장중 학습", "📈 자산 곡선", "📝 거래 내역"
])


def load_paper() -> dict:
    p = DATA_DIR / "paper_portfolio.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_bluechip() -> dict:
    p = DATA_DIR / "paper_bluechip.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ════════════════════════════════════════════════
# Tab 1: 페이퍼 통합 (paper + bluechip + ETF)
# ════════════════════════════════════════════════
with tab1:
    paper = load_paper()
    bluechip = load_bluechip()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📋 Paper Trading")
        if paper:
            equity = paper.get("daily_equity", [{}])[-1].get("equity", 0)
            initial = paper.get("initial_capital", 100_000_000)
            ret = (equity / initial - 1) * 100 if initial > 0 else 0
            st.metric("자산", f"{equity:,}원", f"{ret:+.2f}%")
            st.metric("보유 종목", len(paper.get("positions", {})))
            st.metric("거래 총수", len(paper.get("closed_trades", [])))
        else:
            st.info("paper_portfolio.json 없음")

    with col2:
        st.subheader("💎 Bluechip")
        if bluechip:
            equity = bluechip.get("daily_equity", [{}])[-1].get("equity", 0) if bluechip.get("daily_equity") else 0
            initial = bluechip.get("initial_capital", 100_000_000)
            ret = (equity / initial - 1) * 100 if initial > 0 else 0
            st.metric("자산", f"{equity:,}원", f"{ret:+.2f}%")
            st.metric("보유 종목", len(bluechip.get("positions", {})))
            st.metric("거래 총수", len(bluechip.get("closed_trades", [])))
        else:
            st.info("paper_bluechip.json 없음")

    with col3:
        st.subheader("🎯 ETF 방향")
        if paper:
            etf = paper.get("etf_trading", {})
            cap = etf.get("capital", 0)
            initial = 50_000_000
            ret = (cap / initial - 1) * 100 if initial > 0 else 0
            pos = etf.get("position")
            st.metric("자본", f"{cap:,}원", f"{ret:+.2f}%")
            if pos:
                st.metric("현재 보유", pos.get("name", "-"))
                st.caption(f"진입일: {pos.get('entry_date')} | {pos.get('direction')}")
            else:
                st.info("ETF 미보유")

    # 보유 종목 테이블
    st.markdown("---")
    st.subheader("📊 현재 보유 종목 (전체)")
    all_positions = []
    for ticker, pos in paper.get("positions", {}).items():
        all_positions.append({
            "분류": "Paper",
            "종목": pos.get("name", ticker),
            "코드": ticker,
            "진입일": pos.get("entry_date"),
            "진입가": pos.get("avg_price", 0),
            "수량": pos.get("qty", 0),
            "전략": pos.get("strategy"),
            "등급": pos.get("grade"),
        })
    for ticker, pos in bluechip.get("positions", {}).items():
        all_positions.append({
            "분류": "Bluechip",
            "종목": pos.get("name", ticker),
            "코드": ticker,
            "진입일": pos.get("entry_date"),
            "진입가": pos.get("entry_price", 0),
            "수량": pos.get("qty", 0),
            "전략": pos.get("entry_type"),
            "등급": "-",
        })
    if all_positions:
        df = pd.DataFrame(all_positions)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("보유 종목 없음")


# ════════════════════════════════════════════════
# Tab 2: 시그널 통계
# ════════════════════════════════════════════════
with tab2:
    st.subheader("🎯 전략(strategy)별 누적 적중률")
    trades = []
    paper = load_paper()
    bluechip = load_bluechip()
    for t in paper.get("closed_trades", []):
        t["원천"] = "Paper"
        trades.append(t)
    for t in bluechip.get("closed_trades", []):
        t["원천"] = "Bluechip"
        t["strategy"] = t.get("entry_type", "BLUECHIP")
        trades.append(t)

    if trades:
        df = pd.DataFrame(trades)
        df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
        df = df.dropna(subset=["pnl_pct"])
        agg = df.groupby("strategy").agg(
            건수=("pnl_pct", "count"),
            평균수익=("pnl_pct", "mean"),
            승률=("pnl_pct", lambda x: (x > 0).mean() * 100),
            최대수익=("pnl_pct", "max"),
            최대손실=("pnl_pct", "min"),
        ).round(2)
        agg["PF"] = df.groupby("strategy")["pnl_pct"].apply(
            lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if (x < 0).any() else 99.9
        ).round(2)
        agg = agg.sort_values("PF", ascending=False)
        st.dataframe(agg, use_container_width=True)
        st.caption(f"총 거래: {len(df)}건 | 전략 수: {df['strategy'].nunique()}")
    else:
        st.info("아직 청산된 거래 없음")


# ════════════════════════════════════════════════
# Tab 3: 장중 학습 (Phase 12)
# ════════════════════════════════════════════════
with tab3:
    st.subheader("🧠 장중 실시간 학습 (Phase 12)")

    # 누적 통계
    cum_path = DATA_DIR / "intraday" / "intraday_patterns_cumulative.json"
    if cum_path.exists():
        try:
            cum = json.loads(cum_path.read_text(encoding="utf-8"))
            th = cum.get("surge_thresholds", {})
            if th:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("학습일수", th.get("n_days", 0))
                col2.metric("급등 표본", th.get("n_total_surges", 0))
                col3.metric("early_ret P50", f"{th.get('early_ret_p50', 0):+.2f}%")
                col4.metric("strength P50", f"{th.get('strength_avg_p50', 0):.1f}")
            else:
                st.info("학습 표본 부족 (5건 이상 시 임계치 산출)")
        except Exception as e:
            st.warning(f"누적 통계 로드 실패: {e}")
    else:
        st.info("아직 장중 학습 시작 전 (5/18 월요일부터)")

    # 오늘 시그널
    st.markdown("---")
    today = datetime.now().strftime("%Y%m%d")
    sig_path = DATA_DIR / "intraday" / f"intraday_signals_{today}.json"
    if sig_path.exists():
        try:
            sig = json.loads(sig_path.read_text(encoding="utf-8"))
            cands = sig.get("candidates", [])
            if cands:
                st.subheader(f"오늘 학습 후보 ({len(cands)}건)")
                from src.stock_name_resolver import ticker_to_name
                rows = []
                for c in cands:
                    rows.append({
                        "코드": c.get("code"),
                        "종목": ticker_to_name(c.get("code", "")),
                        "early_ret": f"{c.get('early_ret_pct', 0):+.2f}%",
                        "체결강도": f"{c.get('strength_avg', 0):.1f}",
                        "매수비율": f"{c.get('buy_ratio', 0):.2f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception:
            pass
    else:
        st.caption(f"오늘({today}) 시그널 파일 없음 — 장중 학습 진행 중이거나 분석 대기")


# ════════════════════════════════════════════════
# Tab 4: 자산 곡선
# ════════════════════════════════════════════════
with tab4:
    st.subheader("📈 일별 자산 곡선")
    paper = load_paper()
    bluechip = load_bluechip()

    paper_eq = paper.get("daily_equity", [])
    blue_eq = bluechip.get("daily_equity", [])

    if paper_eq or blue_eq:
        chart_data = pd.DataFrame()
        if paper_eq:
            pdf = pd.DataFrame(paper_eq)
            pdf["date"] = pd.to_datetime(pdf["date"])
            chart_data = chart_data.merge(
                pdf.rename(columns={"equity": "Paper"})[["date", "Paper"]],
                on="date", how="outer"
            ) if not chart_data.empty else pdf.rename(columns={"equity": "Paper"})[["date", "Paper"]]
        if blue_eq:
            bdf = pd.DataFrame(blue_eq)
            bdf["date"] = pd.to_datetime(bdf["date"])
            bdf = bdf.rename(columns={"equity": "Bluechip"})[["date", "Bluechip"]]
            chart_data = chart_data.merge(bdf, on="date", how="outer") if not chart_data.empty else bdf

        if not chart_data.empty:
            chart_data = chart_data.sort_values("date").set_index("date")
            st.line_chart(chart_data, use_container_width=True)
    else:
        st.info("daily_equity 데이터 없음")


# ════════════════════════════════════════════════
# Tab 5: 거래 내역
# ════════════════════════════════════════════════
with tab5:
    st.subheader("📝 최근 청산 거래 (최신 50건)")
    trades = []
    paper = load_paper()
    bluechip = load_bluechip()
    for t in paper.get("closed_trades", []):
        t["원천"] = "Paper"
        trades.append(t)
    for t in bluechip.get("closed_trades", []):
        t["원천"] = "Bluechip"
        trades.append(t)

    if trades:
        df = pd.DataFrame(trades)
        if "exit_date" in df.columns:
            df = df.sort_values("exit_date", ascending=False).head(50)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("거래 내역 없음")

# 자동 새로고침 안내
st.sidebar.markdown("---")
st.sidebar.caption("🔄 5분마다 새로고침 권장 (Ctrl+R)")
