"""
장시작전 분석 보고서 HTML 생성 + PNG 변환 + 텔레그램 전송

사용법:
    from src.html_report import generate_premarket_report
    html_path, png_path = generate_premarket_report(candidates, stats)
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

REPORT_DIR = Path("D:/클로드 HTML 보고서")


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
                    <span class="score-label">C 합의</span>
                    <div class="bar-container">
                        <div class="bar bar-consensus" style="width:{c_pct:.0f}%"></div>
                    </div>
                    <span class="score-value">{sc.get('consensus', 0):.0f}/20</span>
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
