"""
Step 8: report_generator.py — 백테스트 결과 HTML 리포트 생성

생성 내용:
- 핵심 성과 지표 (CAGR, MDD, Sharpe, 승률, 손익비)
- 에쿼티 커브 차트
- 월별 수익률 히트맵
- 등급별 성과 분석
- 드로다운 차트
- 거래 상세 테이블
"""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# plotly는 선택적 (없으면 matplotlib로 대체)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


class ReportGenerator:
    """백테스트 결과를 HTML 리포트로 변환"""

    def __init__(self, initial_capital: float = 50_000_000):
        self.initial_capital = initial_capital
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def generate(self, stats: dict, trades_df: pd.DataFrame,
                 equity_df: pd.DataFrame, signals_df: pd.DataFrame) -> str:
        """전체 리포트 생성"""

        # 차트 생성
        charts_html = self._generate_charts(equity_df, trades_df)

        # HTML 조립
        html = self._build_html(stats, trades_df, equity_df, signals_df, charts_html)

        # 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"backtest_report_{timestamp}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        # latest 링크도 저장
        latest_path = self.results_dir / "backtest_report_latest.html"
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"📄 리포트 저장: {report_path}")
        return str(report_path)

    def _generate_charts(self, equity_df: pd.DataFrame,
                         trades_df: pd.DataFrame) -> str:
        """차트를 HTML 문자열로 생성"""
        charts = []

        if equity_df.empty:
            return "<p>데이터 없음</p>"

        equity_df = equity_df.copy()
        equity_df["date"] = pd.to_datetime(equity_df["date"])

        if PLOTLY_AVAILABLE:
            charts.append(self._plotly_equity_curve(equity_df))
            charts.append(self._plotly_drawdown(equity_df))
            if not trades_df.empty:
                charts.append(self._plotly_monthly_returns(equity_df))
                charts.append(self._plotly_trade_distribution(trades_df))
        elif MPL_AVAILABLE:
            charts.append(self._mpl_equity_curve(equity_df))

        return "\n".join(charts)

    # ── Plotly 차트들 ──

    def _plotly_equity_curve(self, equity_df: pd.DataFrame) -> str:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df["date"], y=equity_df["portfolio_value"],
            mode="lines", name="포트폴리오",
            line=dict(color="#2196F3", width=2),
        ))
        # 벤치마크 (초기자본 기준 직선)
        fig.add_trace(go.Scatter(
            x=equity_df["date"],
            y=[self.initial_capital] * len(equity_df),
            mode="lines", name="초기자본",
            line=dict(color="#888", width=1, dash="dash"),
        ))
        fig.update_layout(
            title="📈 에쿼티 커브",
            xaxis_title="날짜", yaxis_title="포트폴리오 가치 (원)",
            template="plotly_white", height=400,
            yaxis_tickformat=",",
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _plotly_drawdown(self, equity_df: pd.DataFrame) -> str:
        pv = equity_df["portfolio_value"]
        peak = pv.cummax()
        dd = (pv - peak) / peak * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df["date"], y=dd,
            fill="tozeroy", mode="lines", name="Drawdown",
            line=dict(color="#F44336", width=1),
            fillcolor="rgba(244, 67, 54, 0.3)",
        ))
        fig.update_layout(
            title="📉 드로다운",
            xaxis_title="날짜", yaxis_title="Drawdown (%)",
            template="plotly_white", height=300,
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _plotly_monthly_returns(self, equity_df: pd.DataFrame) -> str:
        df = equity_df.copy()
        df["month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("month")["portfolio_value"].last()
        monthly_ret = monthly.pct_change() * 100

        # 히트맵용 데이터
        idx = monthly_ret.index
        years = sorted(set(str(p.year) for p in idx))
        months = list(range(1, 13))

        z = []
        for y in years:
            row = []
            for m in months:
                key = f"{y}-{m:02d}"
                matches = [v for p, v in monthly_ret.items() if str(p) == key]
                row.append(matches[0] if matches else None)
            z.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=z, x=[f"{m}월" for m in months], y=years,
            colorscale=[[0, "#F44336"], [0.5, "#FFFFFF"], [1, "#4CAF50"]],
            zmid=0, text=[[f"{v:.1f}%" if v else "" for v in row] for row in z],
            texttemplate="%{text}", textfont_size=10,
        ))
        fig.update_layout(
            title="📅 월별 수익률 히트맵",
            template="plotly_white", height=300,
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _plotly_trade_distribution(self, trades_df: pd.DataFrame) -> str:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=trades_df["pnl_pct"], nbinsx=30,
            marker_color=["#4CAF50" if x > 0 else "#F44336" for x in trades_df["pnl_pct"]],
            name="거래 수익률 분포",
        ))
        fig.update_layout(
            title="📊 거래별 수익률 분포",
            xaxis_title="수익률 (%)", yaxis_title="빈도",
            template="plotly_white", height=300,
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    # ── Matplotlib 대체 ──

    def _mpl_equity_curve(self, equity_df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity_df["date"], equity_df["portfolio_value"], label="Portfolio", color="#2196F3")
        ax.axhline(self.initial_capital, color="#888", linestyle="--", label="Initial")
        ax.set_title("Equity Curve")
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.tight_layout()

        import io, base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%">'

    # ── HTML 빌더 ──

    def _build_html(self, stats, trades_df, equity_df, signals_df, charts_html) -> str:
        """전체 HTML 리포트 조립"""

        # 등급별 성과 테이블
        grade_rows = ""
        for grade, info in stats.get("grade_breakdown", {}).items():
            grade_rows += f"""
            <tr>
                <td><strong>{grade}</strong></td>
                <td>{info['count']}</td>
                <td>{info['win_rate']:.1f}%</td>
                <td>{info['avg_pnl_pct']:+.2f}%</td>
            </tr>"""

        # 트리거별 성과 테이블 (v2.1 신규)
        trigger_rows = ""
        trigger_emojis = {"impulse": "⚡", "confirm": "🎯", "breakout": "🚀"}
        trigger_names = {"impulse": "시동 (공격형)", "confirm": "확인 (보수형)", "breakout": "돌파 (추가매수)"}
        for ttype, info in stats.get("trigger_breakdown", {}).items():
            emoji = trigger_emojis.get(ttype, "")
            name = trigger_names.get(ttype, ttype)
            trigger_rows += f"""
            <tr>
                <td>{emoji} <strong>{name}</strong></td>
                <td>{info['count']}</td>
                <td>{info['win_rate']:.1f}%</td>
                <td>{info['avg_pnl_pct']:+.2f}%</td>
                <td>{info.get('avg_hold_days', 0)}일</td>
                <td>{info.get('total_pnl', 0):+,}원</td>
            </tr>"""

        # 🔥 v2.5: 트리거별 성과 테이블
        trigger_rows = ""
        trigger_names = {"impulse": "⚡ 시동(Impulse)", "confirm": "✅ 확인(Confirm)",
                         "breakout_add": "🚀 돌파(Breakout)", "legacy": "📋 기존(Legacy)"}
        for tt, info in stats.get("trigger_breakdown", {}).items():
            pnl_color = "#4CAF50" if info["avg_pnl_pct"] > 0 else "#F44336"
            trigger_rows += f"""
            <tr>
                <td><strong>{trigger_names.get(tt, tt)}</strong></td>
                <td>{info['count']}</td>
                <td>{info['win_rate']:.1f}%</td>
                <td style="color:{pnl_color}">{info['avg_pnl_pct']:+.2f}%</td>
                <td>1:{info.get('rr_ratio', 0):.1f}</td>
                <td>{info.get('avg_hold_days', 0)}일</td>
                <td style="color:{pnl_color}">{info.get('total_pnl', 0):+,}</td>
            </tr>"""

        # 청산 사유별 테이블
        exit_rows = ""
        exit_names = {"stop_loss": "🔴 손절", "partial_target": "🟡 50% 익절",
                      "trailing_stop": "🟢 트레일링", "trend_exit": "⚠️ 추세이탈",
                      "backtest_end": "⏹️ 종료청산"}
        for reason, info in stats.get("exit_reason_breakdown", {}).items():
            color = "#4CAF50" if info["avg_pnl_pct"] > 0 else "#F44336"
            exit_rows += f"""
            <tr>
                <td>{exit_names.get(reason, reason)}</td>
                <td>{info['count']}</td>
                <td style="color:{color}">{info['avg_pnl_pct']:+.2f}%</td>
            </tr>"""

        # 최근 거래 테이블 (상위 20건)
        trade_rows = ""
        if not trades_df.empty:
            trigger_col = "trigger_type" in trades_df.columns
            for _, t in trades_df.tail(20).iterrows():
                color = "#4CAF50" if t["pnl"] > 0 else "#F44336"
                tt_label = ""
                if trigger_col:
                    tt_short = {"impulse": "⚡T1", "confirm": "✅T2", "breakout_add": "🚀T3", "legacy": "📋"}
                    tt_label = tt_short.get(t.get("trigger_type", ""), "")
                trade_rows += f"""
                <tr>
                    <td>{t['ticker']}</td>
                    <td>{t['entry_date']}</td>
                    <td>{t['exit_date']}</td>
                    <td>{int(t['entry_price']):,}</td>
                    <td>{int(t['exit_price']):,}</td>
                    <td>{t['shares']}</td>
                    <td style="color:{color}">{t['pnl_pct']:+.1f}%</td>
                    <td>{t['exit_reason']}</td>
                    <td>{t['grade']}</td>
                    <td>{tt_label}</td>
                </tr>"""

        final_value = equity_df["portfolio_value"].iloc[-1] if not equity_df.empty else self.initial_capital
        total_pnl = final_value - self.initial_capital

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>🧊 퀀텀전략 v2.0 — 백테스트 리포트</title>
    <style>
        body {{ font-family: -apple-system, 'Malgun Gothic', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #1a237e, #0d47a1); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 5px 0 0; opacity: 0.8; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
        .stat-box {{ text-align: center; padding: 16px; border-radius: 8px; background: #f8f9fa; }}
        .stat-box .value {{ font-size: 28px; font-weight: bold; color: #1a237e; }}
        .stat-box .label {{ font-size: 12px; color: #666; margin-top: 4px; }}
        .positive {{ color: #4CAF50 !important; }}
        .negative {{ color: #F44336 !important; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #e8eaf6; padding: 8px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f5f5f5; }}
        h2 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 8px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧊 "퀀텀전략" v2.0 — 백테스트 리포트</h1>
        <p>생성일시: {datetime.now().strftime("%Y-%m-%d %H:%M")} | 기간: {stats.get('period', '2019~2024')}</p>
    </div>

    <div class="card">
        <h2>📊 핵심 성과 지표</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="value {'positive' if total_pnl >= 0 else 'negative'}">{stats['total_return_pct']:+.1f}%</div>
                <div class="label">총 수익률</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats['cagr_pct']:.1f}%</div>
                <div class="label">CAGR (연평균)</div>
            </div>
            <div class="stat-box">
                <div class="value negative">{stats['max_drawdown_pct']:.1f}%</div>
                <div class="label">최대 낙폭 (MDD)</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats['sharpe_ratio']:.2f}</div>
                <div class="label">샤프 비율</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats['win_rate']:.1f}%</div>
                <div class="label">승률</div>
            </div>
            <div class="stat-box">
                <div class="value">1:{stats['avg_rr_ratio']:.1f}</div>
                <div class="label">평균 손익비</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats['total_trades']}</div>
                <div class="label">총 거래 수</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats['profit_factor']:.2f}</div>
                <div class="label">Profit Factor</div>
            </div>
        </div>

        <div class="stats-grid" style="margin-top: 12px;">
            <div class="stat-box">
                <div class="value">{self.initial_capital:,.0f}</div>
                <div class="label">초기 자본 (원)</div>
            </div>
            <div class="stat-box">
                <div class="value {'positive' if total_pnl >= 0 else 'negative'}">{final_value:,.0f}</div>
                <div class="label">최종 자본 (원)</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats['avg_hold_days']}일</div>
                <div class="label">평균 보유 기간</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats['total_commission']:,}</div>
                <div class="label">총 수수료 (원)</div>
            </div>
        </div>
    </div>

    <div class="card">
        {charts_html}
    </div>

    <div class="card">
        <h2>📊 등급별 성과 분석</h2>
        <p>BES 등급이 높을수록 성과가 좋은지 확인</p>
        <table>
            <tr><th>등급</th><th>거래 수</th><th>승률</th><th>평균 수익률</th></tr>
            {grade_rows}
        </table>
    </div>

    <div class="card">
        <h2>🔥 트리거별 성과 분석 (v2.1)</h2>
        <p>시동(Impulse) vs 확인(Confirm) 모드 비교 — 목표: 시동은 수익폭↑, 확인은 승률↑</p>
        <table>
            <tr><th>트리거</th><th>거래 수</th><th>승률</th><th>평균 수익률</th><th>평균 보유</th><th>총 손익</th></tr>
            {trigger_rows}
        </table>
    </div>

    <div class="card">
        <h2>📋 최근 거래 내역 (최근 20건)</h2>
        <table>
            <tr><th>종목</th><th>매수일</th><th>매도일</th><th>매수가</th><th>매도가</th><th>수량</th><th>수익률</th><th>청산사유</th><th>등급</th><th>트리거</th></tr>
            {trade_rows}
        </table>
    </div>

    <div class="card" style="text-align: center; color: #666; font-size: 12px;">
        <p>이 리포트는 교육 목적의 참고자료입니다. 모든 투자 판단은 본인의 책임입니다.</p>
        <p>🧊 "퀀텀전략" v2.1 듀얼 트리거 백테스트 시스템</p>
    </div>
</body>
</html>"""

        return html
