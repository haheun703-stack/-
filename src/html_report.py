"""
장시작전 분석 보고서 HTML 생성 + PNG 변환 + 텔레그램 전송

사용법:
    from src.html_report import generate_premarket_report
    html_path, png_path = generate_premarket_report(candidates, stats)
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPORT_DIR = Path("D:/클로드 HTML 보고서")


def _position_guide_html(sig: dict) -> str:
    """포지션 사이징 가이드 HTML 생성."""
    guide = sig.get("position_guide")
    if not guide or guide.get("shares", 0) <= 0:
        return ""
    label = guide.get("label", "대기")
    bg = "#238636" if label == "매수" else "#6e7681"
    return f"""
            <div class="position-guide" style="background:{bg}22; border: 1px solid {bg}; border-radius:6px; padding:8px 12px; margin-top:8px; display:flex; justify-content:space-between; align-items:center;">
                <span style="color:{bg}; font-weight:bold;">{label}</span>
                <span>{guide['alloc']/1e4:,.0f}만원 ({guide['pct']}%)</span>
                <span>{guide['shares']}주 &times; {guide['price']:,}원</span>
            </div>"""


def generate_premarket_report(
    candidates: list[dict],
    stats: dict,
    output_dir: Path = REPORT_DIR,
) -> tuple[Path, Path | None]:
    """
    장시작전 분석 보고서 생성.

    Returns:
        (html_path, png_path) — png_path는 변환 실패 시 None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    # v9 감지: 후보에 v9_rank_score 키가 있으면 v9 모드
    is_v9 = candidates and "v9_rank_score" in candidates[0]
    if is_v9:
        html = _build_html_v9(candidates, stats, date_str, time_str)
    else:
        html = _build_html(candidates, stats, date_str, time_str)

    html_path = output_dir / f"장시작전_분석_{date_str}.html"
    html_path.write_text(html, encoding="utf-8")
    logger.info("HTML 보고서 저장: %s", html_path)

    # HTML → PNG
    png_path = html_path.with_suffix(".png")
    try:
        _html_to_png(html_path, png_path)
        logger.info("PNG 변환 완료: %s", png_path)
    except Exception as e:
        logger.error("PNG 변환 실패: %s", e)
        png_path = None

    return html_path, png_path


def send_report_to_telegram(png_path: Path, caption: str = "") -> bool:
    """PNG 보고서를 텔레그램으로 전송."""
    import os

    import requests
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        logger.error("텔레그램 토큰/채팅ID 미설정")
        return False

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(png_path, "rb") as f:
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption[:1024]  # 텔레그램 캡션 제한
        resp = requests.post(url, data=data, files={"photo": f}, timeout=30)

    if resp.status_code == 200 and resp.json().get("ok"):
        logger.info("텔레그램 이미지 전송 성공")
        return True
    else:
        logger.error("텔레그램 이미지 전송 실패: %s", resp.text)
        return False


def _html_to_png(html_path: Path, png_path: Path, width: int = 800):
    """Playwright로 HTML → PNG 캡처."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": 600})
        page.goto(f"file:///{html_path.as_posix()}")
        page.wait_for_timeout(500)

        # 콘텐츠 높이에 맞게 자동 조절
        height = page.evaluate("document.body.scrollHeight")
        page.set_viewport_size({"width": width, "height": height + 40})
        page.screenshot(path=str(png_path), full_page=True)
        browser.close()


def _build_html(
    candidates: list[dict],
    stats: dict,
    date_str: str,
    time_str: str,
) -> str:
    """장시작전 분석 보고서 HTML 생성."""

    # 종목 카드 HTML
    cards_html = ""
    for i, sig in enumerate(candidates, start=1):
        sc = sig.get("scores", {})
        q_detail = sc.get("q_detail", {})
        sd_detail = sc.get("sd_detail", {})
        news_detail = sc.get("news_detail", {})
        c_detail = sc.get("consensus_detail", {})

        total = sc.get("total", 0)
        grade = sig.get("grade", "?")
        trigger = sig.get("trigger_type", "?")

        # 뉴스 상태
        news_data = sig.get("news_data")
        if news_data:
            sentiment = news_data.get("overall_sentiment", "중립")
            if sentiment in ("긍정", "positive"):
                news_badge = '<span class="badge positive">긍정</span>'
            elif sentiment in ("부정", "negative"):
                news_badge = '<span class="badge negative">부정</span>'
            else:
                news_badge = '<span class="badge neutral">중립</span>'
            takeaway = news_data.get("key_takeaway", "")[:80]
        else:
            news_badge = '<span class="badge neutral">-</span>'
            takeaway = ""

        # 추세 배지
        trend = sig.get("trend", "?")
        trend_map = {
            "strong_up": ("강세", "positive"),
            "up": ("상승", "positive"),
            "neutral": ("중립", "neutral"),
            "down": ("하락", "negative"),
        }
        trend_label, trend_class = trend_map.get(trend, ("?", "neutral"))

        # 수급 방향
        plus_di = sig.get("plus_di", 0)
        minus_di = sig.get("minus_di", 0)
        di_dir = "매수세" if plus_di > minus_di else "매도세"
        di_class = "positive" if plus_di > minus_di else "negative"

        # 가격 변동률
        entry = sig.get("entry_price", 0)
        target = sig.get("target_price", 0)
        stop = sig.get("stop_loss", 0)
        upside = ((target / entry) - 1) * 100 if entry else 0
        downside = ((stop / entry) - 1) * 100 if entry else 0

        # 점수 바 너비 (100점 만점 기준)
        q_pct = min(sc.get("quant", 0) / 30 * 100, 100)
        sd_pct = min(sc.get("supply_demand", 0) / 25 * 100, 100)
        n_pct = min(sc.get("news", 0) / 25 * 100, 100)
        c_pct = min(sc.get("consensus", 0) / 20 * 100, 100)

        rank_class = "rank-1" if i == 1 else ""

        cards_html += f"""
        <div class="stock-card {rank_class}">
            <div class="card-header">
                <div class="rank">#{i}</div>
                <div class="stock-info">
                    <div class="stock-name">{sig.get('name', sig['ticker'])}</div>
                    <div class="stock-code">{sig['ticker']} | Grade {grade}</div>
                </div>
                <div class="total-score">{total:.0f}<span class="score-unit">점</span></div>
            </div>

            <div class="price-row">
                <div class="current-price">{entry:,}원</div>
                <div class="price-targets">
                    <span class="target-up">목표 {target:,} (+{upside:.1f}%)</span>
                    <span class="target-down">손절 {stop:,} ({downside:.1f}%)</span>
                </div>
                <div class="rr-badge">RR 1:{sig.get('risk_reward', 0):.1f}</div>
            </div>

            <div class="indicators-row">
                <span class="indicator">RSI {sig.get('rsi', 0):.0f}</span>
                <span class="indicator">ADX {sig.get('adx', 0):.0f}</span>
                <span class="indicator badge {trend_class}">{trend_label}</span>
                <span class="indicator badge {di_class}">{di_dir}</span>
                <span class="indicator">뉴스 {news_badge}</span>
            </div>

            <div class="score-bars">
                <div class="score-row">
                    <span class="score-label">Q 퀀트</span>
                    <div class="bar-container">
                        <div class="bar bar-quant" style="width:{q_pct:.0f}%"></div>
                    </div>
                    <span class="score-value">{sc.get('quant', 0):.0f}/30</span>
                </div>
                <div class="score-detail">
                    Zone {q_detail.get('zone', 0):.1f} | Trigger {q_detail.get('trigger', 0):.0f} | R:R {q_detail.get('rr', 0):.0f} | Trend {q_detail.get('trend', 0):.0f}
                </div>

                <div class="score-row">
                    <span class="score-label">SD 수급</span>
                    <div class="bar-container">
                        <div class="bar bar-sd" style="width:{sd_pct:.0f}%"></div>
                    </div>
                    <span class="score-value">{sc.get('supply_demand', 0):.0f}/25</span>
                </div>
                <div class="score-detail">
                    외국인 {sd_detail.get('foreign', 0)} | 기관 {sd_detail.get('inst', 0)} | OBV {sd_detail.get('obv_vol', 0)} | ADX방향 {sd_detail.get('adx_dir', 0)}
                </div>

                <div class="score-row">
                    <span class="score-label">N 뉴스</span>
                    <div class="bar-container">
                        <div class="bar bar-news" style="width:{n_pct:.0f}%"></div>
                    </div>
                    <span class="score-value">{sc.get('news', 0):.0f}/25</span>
                </div>
                <div class="score-detail">
                    감성 {news_detail.get('sentiment', 0)} | 영향 {news_detail.get('impact', 0)} | 이슈 {news_detail.get('living', 0)} | 실적 {news_detail.get('earnings', 0)}
                </div>

                <div class="score-row">
                    <span class="score-label">C 실적</span>
                    <div class="bar-container">
                        <div class="bar bar-consensus" style="width:{c_pct:.0f}%"></div>
                    </div>
                    <span class="score-value">{sc.get('consensus', 0):.0f}/20</span>
                </div>
                <div class="score-detail">
                    {c_detail.get('verdict', '-')} | {c_detail.get('detail', '')[:40]}
                </div>
            </div>

            {"<div class='takeaway'>" + takeaway + "</div>" if takeaway else ""}
        </div>
        """

    # 전체 HTML
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>장시작전 분석 보고서 - {date_str}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
        background: #0d1117;
        color: #e6edf3;
        padding: 20px;
        max-width: 800px;
        margin: 0 auto;
    }}

    .report-header {{
        text-align: center;
        padding: 24px 0 16px;
        border-bottom: 2px solid #30363d;
        margin-bottom: 20px;
    }}
    .report-title {{
        font-size: 22px;
        font-weight: 700;
        color: #58a6ff;
        letter-spacing: 1px;
    }}
    .report-subtitle {{
        font-size: 13px;
        color: #8b949e;
        margin-top: 6px;
    }}

    .stats-bar {{
        display: flex;
        justify-content: space-around;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 20px;
        font-size: 13px;
    }}
    .stat-item {{ text-align: center; }}
    .stat-value {{ font-size: 20px; font-weight: 700; color: #58a6ff; }}
    .stat-label {{ color: #8b949e; margin-top: 2px; }}

    .scoring-info {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 20px;
        font-size: 11px;
        color: #8b949e;
        line-height: 1.6;
    }}
    .scoring-info strong {{ color: #c9d1d9; }}

    .stock-card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
        transition: all 0.2s;
    }}
    .stock-card.rank-1 {{
        border-color: #f0883e;
        box-shadow: 0 0 12px rgba(240, 136, 62, 0.15);
    }}

    .card-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }}
    .rank {{
        font-size: 16px;
        font-weight: 800;
        color: #8b949e;
        min-width: 32px;
    }}
    .rank-1 .rank {{ color: #f0883e; font-size: 20px; }}
    .stock-name {{ font-size: 17px; font-weight: 700; }}
    .stock-code {{ font-size: 12px; color: #8b949e; }}
    .total-score {{
        margin-left: auto;
        font-size: 32px;
        font-weight: 800;
        color: #58a6ff;
    }}
    .score-unit {{ font-size: 14px; color: #8b949e; }}

    .price-row {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 10px;
        font-size: 13px;
    }}
    .current-price {{ font-size: 18px; font-weight: 700; }}
    .price-targets {{ display: flex; flex-direction: column; gap: 2px; }}
    .target-up {{ color: #3fb950; }}
    .target-down {{ color: #f85149; }}
    .rr-badge {{
        margin-left: auto;
        background: #1f2937;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        color: #79c0ff;
    }}

    .indicators-row {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 12px;
    }}
    .indicator {{
        font-size: 11px;
        background: #1f2937;
        padding: 3px 8px;
        border-radius: 4px;
        color: #c9d1d9;
    }}

    .badge {{ font-weight: 600; }}
    .badge.positive {{ background: #1a3a2a; color: #3fb950; }}
    .badge.negative {{ background: #3a1a1a; color: #f85149; }}
    .badge.neutral {{ background: #2a2a1a; color: #d29922; }}

    .score-bars {{ margin-top: 8px; }}
    .score-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 2px;
    }}
    .score-label {{
        font-size: 11px;
        font-weight: 600;
        min-width: 56px;
        color: #8b949e;
    }}
    .bar-container {{
        flex: 1;
        height: 8px;
        background: #21262d;
        border-radius: 4px;
        overflow: hidden;
    }}
    .bar {{
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }}
    .bar-quant {{ background: linear-gradient(90deg, #388bfd, #58a6ff); }}
    .bar-sd {{ background: linear-gradient(90deg, #1f6feb, #79c0ff); }}
    .bar-news {{ background: linear-gradient(90deg, #238636, #3fb950); }}
    .bar-consensus {{ background: linear-gradient(90deg, #8957e5, #bc8cff); }}

    .score-value {{
        font-size: 11px;
        font-weight: 600;
        min-width: 36px;
        text-align: right;
        color: #c9d1d9;
    }}
    .score-detail {{
        font-size: 10px;
        color: #6e7681;
        margin-left: 64px;
        margin-bottom: 6px;
    }}

    .takeaway {{
        margin-top: 10px;
        padding: 8px 12px;
        background: #1a2233;
        border-left: 3px solid #58a6ff;
        border-radius: 4px;
        font-size: 12px;
        color: #c9d1d9;
        line-height: 1.5;
    }}

    .footer {{
        text-align: center;
        padding: 16px 0;
        font-size: 11px;
        color: #484f58;
        border-top: 1px solid #21262d;
        margin-top: 8px;
    }}
</style>
</head>
<body>

<div class="report-header">
    <div class="report-title">Quant v5.0 장시작전 분석</div>
    <div class="report-subtitle">{date_str} {time_str} | 4-Axis Score System</div>
</div>

<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">{stats.get('total', 0)}</div>
        <div class="stat-label">스캔 종목</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">{stats.get('passed_pipeline', 0)}</div>
        <div class="stat-label">파이프라인 통과</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">{stats.get('after_grade_filter', 0)}</div>
        <div class="stat-label">최종 후보</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">{stats.get('scan_sec', 0) + stats.get('news_sec', 0):.0f}s</div>
        <div class="stat-label">소요 시간</div>
    </div>
</div>

<div class="scoring-info">
    <strong>4축 점수 체계 (100점)</strong><br>
    Q 퀀트(30) = Zone(15) + Trigger(6) + R:R(5) + Trend(4)<br>
    SD 수급(25) = 외국인(10) + 기관(10) + OBV/거래량(5)<br>
    N 뉴스(25) = 감성(8) + 영향(8) + 이슈(7) + 실적(7)<br>
    C 합의(20) = Reward(10) + Consistency(4) + Reliability(3) + Diversity(3)
</div>

{cards_html}

<div class="footer">
    Quantum Master v5.0 | SignalEngine + Grok News + 4-Axis Scoring<br>
    자동 생성 보고서 — 투자 판단은 본인 책임
</div>

</body>
</html>"""


def _rank_to_grade(rank: int) -> str:
    """순위 → 등급 변환 (1=S, 2=A, 3=B, 4=C, 5+=D)."""
    return {1: "S", 2: "A", 3: "B", 4: "C"}.get(rank, "D")


def _grade_color(grade: str) -> str:
    """등급별 강조 색상."""
    return {"S": "#f0883e", "A": "#58a6ff", "B": "#3fb950", "C": "#8b949e", "D": "#484f58"}.get(grade, "#8b949e")


def _build_html_v9(
    candidates: list[dict],
    stats: dict,
    date_str: str,
    time_str: str,
) -> str:
    """v10.0 Kill→Rank→Tag 보고서 HTML (S/A/B/C/D 등급제)."""
    import json

    # Kill 통계
    killed = stats.get("v9_killed", 0)
    survivors = stats.get("v9_survivors", 0)

    # US Overnight Signal 로드
    us_html = ""
    try:
        signal_path = Path("data/us_market/overnight_signal.json")
        if signal_path.exists():
            with open(signal_path, "r", encoding="utf-8") as f:
                us_signal = json.load(f)
            us_grade = us_signal.get("grade", "NEUTRAL")
            us_combined = us_signal.get("combined_score_100", 0)
            idx_dir = us_signal.get("index_direction", {})
            spy_r = idx_dir.get("SPY", {}).get("ret_1d", 0)
            qqq_r = idx_dir.get("QQQ", {}).get("ret_1d", 0)
            ewy_r = idx_dir.get("EWY", {}).get("ret_1d", 0)
            us_vix = us_signal.get("vix", {})
            vix_level = us_vix.get("level", "?")
            vix_status = us_vix.get("status", "?")

            # 등급별 색상
            us_color = "#f85149" if "BEAR" in us_grade else "#3fb950" if "BULL" in us_grade else "#d29922"

            us_html = f"""
    <div class="us-overnight">
        <div class="us-header">
            <span class="us-label">US Overnight</span>
            <span class="us-grade" style="color:{us_color}">{us_grade}</span>
            <span class="us-score" style="color:{us_color}">{us_combined:+.1f}</span>
        </div>
        <div class="us-detail">
            <span class="us-item {'up' if ewy_r >= 0 else 'down'}">EWY {ewy_r:+.1f}%</span>
            <span class="us-item {'up' if spy_r >= 0 else 'down'}">SPY {spy_r:+.1f}%</span>
            <span class="us-item {'up' if qqq_r >= 0 else 'down'}">QQQ {qqq_r:+.1f}%</span>
            <span class="us-item">VIX {vix_level} [{vix_status}]</span>
        </div>
    </div>"""
    except Exception:
        pass

    # KOSPI 레짐 정보
    kospi_html = ""
    try:
        kospi_path = Path("data/kospi_index.csv")
        if kospi_path.exists():
            kdf = pd.read_csv(kospi_path, index_col="Date", parse_dates=True).sort_index()
            kdf["ma20"] = kdf["close"].rolling(20).mean()
            kdf["ma60"] = kdf["close"].rolling(60).mean()
            log_ret = np.log(kdf["close"] / kdf["close"].shift(1))
            kdf["rv20"] = log_ret.rolling(20).std() * np.sqrt(252) * 100
            kdf["rv20_pct"] = kdf["rv20"].rolling(252, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
            kr = kdf.iloc[-1]
            k_close = float(kr["close"])
            k_ma20 = float(kr["ma20"]) if not pd.isna(kr["ma20"]) else 0
            k_ma60 = float(kr["ma60"]) if not pd.isna(kr["ma60"]) else 0
            k_rv = float(kr.get("rv20_pct", 0.5)) if not pd.isna(kr.get("rv20_pct", 0.5)) else 0.5

            if k_ma20 > 0 and k_close > k_ma20:
                k_regime = "BULL" if k_rv < 0.50 else "CAUTION"
            elif k_ma60 > 0 and k_close > k_ma60:
                k_regime = "BEAR"
            elif k_ma60 > 0:
                k_regime = "CRISIS"
            else:
                k_regime = "CAUTION"

            k_slots = {"BULL": 5, "CAUTION": 3, "BEAR": 2, "CRISIS": 0}[k_regime]
            k_color = {"BULL": "#3fb950", "CAUTION": "#d29922", "BEAR": "#f85149", "CRISIS": "#ff0000"}[k_regime]

            kospi_html = f"""
    <div class="us-overnight" style="margin-top:8px">
        <div class="us-header">
            <span class="us-label">KOSPI \ub808\uc9d0</span>
            <span class="us-grade" style="color:{k_color}">{k_regime}</span>
            <span class="us-score" style="color:{k_color}">{k_slots}\uc2ac\ub86f</span>
        </div>
        <div class="us-detail">
            <span class="us-item">KOSPI {k_close:,.0f}</span>
            <span class="us-item {'up' if k_close > k_ma20 else 'down'}">MA20 {k_ma20:,.0f}</span>
            <span class="us-item {'up' if k_close > k_ma60 else 'down'}">MA60 {k_ma60:,.0f}</span>
            <span class="us-item">RV%ile {k_rv:.0%}</span>
        </div>
    </div>"""
    except Exception:
        pass

    # 종목 카드 HTML
    cards_html = ""
    for i, sig in enumerate(candidates, start=1):
        rank_score = sig.get("v9_rank_score", 0)
        zone = sig.get("zone_score", 0)
        rr = sig.get("risk_reward", 0)
        boost = sig.get("v9_catalyst_boost", 1.0)
        us_m = sig.get("v9_us_mult", 1.0)
        den_m = sig.get("v9_density_mult", 1.0)
        tags = sig.get("v9_tags", [])
        pipe_grade = sig.get("grade", "?")
        trigger = sig.get("trigger_type", "?")
        trigger_label = {"confirm": "확인매수", "impulse": "IMP", "setup": "SETUP"}.get(trigger, trigger)

        entry = sig.get("entry_price", 0)
        target = sig.get("target_price", 0)
        stop = sig.get("stop_loss", 0)
        upside = ((target / entry) - 1) * 100 if entry else 0
        downside = ((stop / entry) - 1) * 100 if entry else 0

        # Rank Score 바 (최대값 기준 스케일링)
        max_rank = candidates[0].get("v9_rank_score", 1) if candidates else 1
        rank_pct = min(rank_score / max(max_rank, 0.01) * 100, 100)

        # 등급 + 색상
        grade_label = _rank_to_grade(i)
        g_color = _grade_color(grade_label)

        fresh_m = sig.get("v9_freshness_mult", 1.0)
        counter = sig.get("trix_counter", 0)
        counter_label = f"GC+{counter}" if counter > 0 else f"DC{counter}"

        # 배수 정보
        fresh_html = f' <span class="fresh-badge">F&times;{fresh_m:.2f}</span>' if fresh_m != 1.0 else ""
        boost_html = ' <span class="catalyst-badge">&times;1.10 촉매</span>' if boost > 1.0 else ""
        us_html_mod = f' <span class="us-mod">US{us_m:.2f}</span>' if us_m != 1.0 else ""
        den_label = "저밀도&uarr;" if den_m > 1.0 else "고밀도&darr;" if den_m < 1.0 else ""
        den_html = f' <span class="den-mod">{den_label}</span>' if den_label else ""
        tags_html = "".join(f'<span class="tag-badge">{t}</span>' for t in tags)

        # 수급 정보
        f_streak = sig.get("foreign_streak", 0)
        i_streak = sig.get("inst_streak", 0)
        supply_parts = []
        if f_streak > 0:
            supply_parts.append(f'<span class="supply-buy">외국인 {f_streak}D 연속매수</span>')
        elif f_streak < 0:
            supply_parts.append(f'<span class="supply-sell">외국인 {abs(f_streak)}D 연속매도</span>')
        if i_streak > 0:
            supply_parts.append(f'<span class="supply-buy">기관 {i_streak}D 연속매수</span>')
        elif i_streak < 0:
            supply_parts.append(f'<span class="supply-sell">기관 {abs(i_streak)}D 연속매도</span>')
        supply_html = " ".join(supply_parts)

        # DI 방향
        plus_di = sig.get("plus_di", 0)
        minus_di = sig.get("minus_di", 0)
        di_dir = "매수세" if plus_di > minus_di else "매도세"
        di_class = "positive" if plus_di > minus_di else "negative"

        # 뉴스 요약
        news_html = ""
        news_data = sig.get("news_data")
        if news_data:
            sentiment = news_data.get("overall_sentiment", "중립")
            if sentiment in ("긍정", "positive"):
                s_badge = '<span class="badge positive">긍정</span>'
            elif sentiment in ("부정", "negative"):
                s_badge = '<span class="badge negative">부정</span>'
            else:
                s_badge = '<span class="badge neutral">중립</span>'
            takeaway = news_data.get("key_takeaway", "")[:80]
            news_html = f"""
            <div class="news-row">
                <span class="news-label">뉴스</span> {s_badge}
                {"<span class='takeaway'>" + takeaway + "</span>" if takeaway else ""}
            </div>"""

        rank_class = "rank-s" if i == 1 else ""

        # 시총 표시
        mcap = sig.get("market_cap", 0)
        mcap_str = f"{mcap / 1e12:.1f}조" if mcap >= 1e12 else f"{mcap / 1e8:,.0f}억" if mcap > 0 else ""
        avg_tv = sig.get("avg_trading_value_20d", 0)
        size_str = f" | 시총 {mcap_str}" if mcap_str else ""
        size_str += f" | 거래대금 {avg_tv / 1e8:.0f}억/일" if avg_tv > 0 else ""

        cards_html += f"""
        <div class="stock-card {rank_class}" style="border-left: 4px solid {g_color}">
            <div class="card-header">
                <div class="grade-badge" style="background:{g_color}">{grade_label}</div>
                <div class="stock-info">
                    <div class="stock-name">{sig.get('name', sig['ticker'])}</div>
                    <div class="stock-code">{sig['ticker']} | {trigger_label}{size_str}</div>
                </div>
                <div class="rank-score" style="color:{g_color}">{rank_score:.3f}</div>
            </div>

            <div class="rank-formula">
                R:R({rr:.1f}) &times; Zone({zone:.2f}){fresh_html}{boost_html}{us_html_mod}{den_html} = {rank_score:.3f}
            </div>
            <div class="trix-row">
                <span class="indicator">TRIX {counter_label}일</span>
                <span class="indicator">Freshness &times;{fresh_m:.2f}</span>
            </div>

            <div class="score-bars">
                <div class="score-row">
                    <span class="score-label">Rank</span>
                    <div class="bar-container">
                        <div class="bar" style="width:{rank_pct:.0f}%; background: linear-gradient(90deg, {g_color}88, {g_color})"></div>
                    </div>
                    <span class="score-value">{rank_score:.3f}</span>
                </div>
            </div>

            <div class="price-row">
                <div class="current-price">{entry:,}원</div>
                <div class="price-targets">
                    <span class="target-up">목표 {target:,} (+{upside:.1f}%)</span>
                    <span class="target-down">손절 {stop:,} ({downside:.1f}%)</span>
                </div>
                <div class="rr-badge">RR 1:{rr:.1f}</div>
            </div>

            <div class="indicators-row">
                <span class="indicator">RSI {sig.get('rsi', 0):.0f}</span>
                <span class="indicator">ADX {sig.get('adx', 0):.0f}</span>
                <span class="indicator badge {di_class}">{di_dir}</span>
                <span class="indicator">거래량 &times;{sig.get('vol_surge', 1.0):.1f}</span>
                <span class="indicator">Zone {zone:.2f}</span>
            </div>

            {_position_guide_html(sig)}
            {"<div class='supply-row'>" + supply_html + "</div>" if supply_html else ""}
            {news_html}
            <div class="tags-row">{tags_html}</div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantum Master v10.0 장시작전 분석 - {date_str}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
        background: #0d1117;
        color: #e6edf3;
        padding: 20px;
        max-width: 800px;
        margin: 0 auto;
    }}

    .report-header {{
        text-align: center;
        padding: 24px 0 16px;
        border-bottom: 2px solid #f0883e;
        margin-bottom: 20px;
    }}
    .report-title {{
        font-size: 24px;
        font-weight: 800;
        color: #f0883e;
        letter-spacing: 1px;
    }}
    .report-subtitle {{
        font-size: 13px;
        color: #8b949e;
        margin-top: 6px;
    }}

    .us-overnight {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }}
    .us-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }}
    .us-label {{
        font-size: 12px;
        font-weight: 600;
        color: #8b949e;
    }}
    .us-grade {{
        font-size: 16px;
        font-weight: 800;
    }}
    .us-score {{
        font-size: 20px;
        font-weight: 700;
        margin-left: auto;
    }}
    .us-detail {{
        display: flex;
        gap: 16px;
        font-size: 13px;
    }}
    .us-item.up {{ color: #3fb950; }}
    .us-item.down {{ color: #f85149; }}
    .us-item {{ color: #8b949e; }}

    .stats-bar {{
        display: flex;
        justify-content: space-around;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 16px;
        font-size: 13px;
    }}
    .stat-item {{ text-align: center; }}
    .stat-value {{ font-size: 20px; font-weight: 700; color: #f0883e; }}
    .stat-label {{ color: #8b949e; margin-top: 2px; }}

    .pipeline-info {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 20px;
        font-size: 11px;
        color: #6e7681;
        line-height: 1.6;
    }}
    .pipeline-info strong {{ color: #8b949e; }}

    .stock-card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }}
    .stock-card.rank-s {{
        border-color: #f0883e;
        box-shadow: 0 0 12px rgba(240, 136, 62, 0.15);
    }}

    .card-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }}
    .grade-badge {{
        font-size: 18px;
        font-weight: 900;
        color: #0d1117;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        flex-shrink: 0;
    }}
    .stock-name {{ font-size: 17px; font-weight: 700; }}
    .stock-code {{ font-size: 12px; color: #8b949e; }}
    .rank-score {{
        margin-left: auto;
        font-size: 28px;
        font-weight: 800;
    }}

    .rank-formula {{
        font-size: 12px;
        color: #8b949e;
        margin-bottom: 10px;
        padding: 6px 10px;
        background: #1a1e26;
        border-radius: 6px;
    }}
    .fresh-badge {{
        background: #da3633;
        color: #fff;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
    }}
    .trix-row {{
        padding: 4px 0;
        font-size: 11px;
        color: #8b949e;
    }}
    .trix-row .indicator {{
        margin-right: 12px;
    }}
    .catalyst-badge {{
        background: #3fb950;
        color: #0d1117;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
    }}
    .us-mod {{
        background: #1f6feb;
        color: #fff;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
    }}
    .den-mod {{
        background: #8957e5;
        color: #fff;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
    }}

    .price-row {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 10px;
        font-size: 13px;
    }}
    .current-price {{ font-size: 18px; font-weight: 700; }}
    .price-targets {{ display: flex; flex-direction: column; gap: 2px; }}
    .target-up {{ color: #3fb950; }}
    .target-down {{ color: #f85149; }}
    .rr-badge {{
        margin-left: auto;
        background: #1f2937;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        color: #79c0ff;
    }}

    .indicators-row {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }}
    .indicator {{
        font-size: 11px;
        background: #1f2937;
        padding: 3px 8px;
        border-radius: 4px;
        color: #c9d1d9;
    }}
    .badge {{ font-weight: 600; }}
    .badge.positive {{ background: #1a3a2a; color: #3fb950; }}
    .badge.negative {{ background: #3a1a1a; color: #f85149; }}
    .badge.neutral {{ background: #2a2a1a; color: #d29922; }}

    .supply-row {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 6px;
    }}
    .supply-buy {{
        font-size: 11px;
        color: #3fb950;
        background: #1a3a2a;
        padding: 3px 8px;
        border-radius: 4px;
    }}
    .supply-sell {{
        font-size: 11px;
        color: #f85149;
        background: #3a1a1a;
        padding: 3px 8px;
        border-radius: 4px;
    }}

    .news-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 11px;
    }}
    .news-label {{
        font-weight: 600;
        color: #8b949e;
    }}
    .takeaway {{
        color: #c9d1d9;
        font-style: italic;
    }}

    .tags-row {{
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }}
    .tag-badge {{
        font-size: 10px;
        background: #1a2a3a;
        color: #79c0ff;
        padding: 3px 8px;
        border-radius: 10px;
        font-weight: 600;
    }}

    .score-bars {{ margin-bottom: 10px; }}
    .score-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 2px;
    }}
    .score-label {{
        font-size: 11px;
        font-weight: 600;
        min-width: 56px;
        color: #8b949e;
    }}
    .bar-container {{
        flex: 1;
        height: 10px;
        background: #21262d;
        border-radius: 5px;
        overflow: hidden;
    }}
    .bar {{
        height: 100%;
        border-radius: 5px;
    }}
    .score-value {{
        font-size: 11px;
        font-weight: 600;
        min-width: 50px;
        text-align: right;
        color: #c9d1d9;
    }}

    .footer {{
        text-align: center;
        padding: 16px 0;
        font-size: 11px;
        color: #484f58;
        border-top: 1px solid #21262d;
        margin-top: 8px;
    }}
</style>
</head>
<body>

<div class="report-header">
    <div class="report-title">Quantum Master v10.0</div>
    <div class="report-subtitle">{date_str} {time_str} | Kill &rarr; Rank &rarr; Tag | S/A/B/C/D</div>
</div>

{us_html}
{kospi_html}

<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">{stats.get('total', 0)}</div>
        <div class="stat-label">스캔</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">{stats.get('passed_pipeline', 0)}</div>
        <div class="stat-label">Pipeline</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">{killed}</div>
        <div class="stat-label">Kill</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">{survivors}</div>
        <div class="stat-label">생존</div>
    </div>
</div>

<div class="pipeline-info">
    <strong>Kill &rarr; Rank &rarr; Tag Pipeline</strong><br>
    Kill: K3(Trigger) + K4(&lt;50억) + K5(시총&lt;5000억) + K6(&lt;5000원) + K7(스팩/리츠/우선주) + K8(TRIX DC) + K9(폭락&gt;20%) + K10(MA120&lt;)<br>
    Rank = R:R &times; Zone &times; 촉매 &times; US부스트 &times; 밀도 | 눌림목 확인 매수 전략<br>
    등급: S(1위) &gt; A(2위) &gt; B(3위) &gt; C(4위) &gt; D(5위+)
</div>

{cards_html}

<div class="footer">
    Quantum Master v10.0 | Kill &rarr; Rank &rarr; Tag | S/A/B/C/D<br>
    자동 생성 보고서 &mdash; 투자 판단은 본인 책임
</div>

</body>
</html>"""
