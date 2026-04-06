"""컨센서스 스크리닝 HTML 보고서 생성 + PNG 변환 + 텔레그램 전송

Usage:
    from src.consensus_report import generate_consensus_report
    html_path, png_path = generate_consensus_report(result_dict)
"""

import logging
import platform
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

if platform.system() == "Windows":
    REPORT_DIR = Path("D:/클로드 HTML 보고서")
else:
    REPORT_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def generate_consensus_report(
    result: dict,
    output_dir: Path = REPORT_DIR,
) -> tuple[Path, Path | None]:
    """컨센서스 스크리닝 HTML 보고서 생성.

    Returns:
        (html_path, png_path) — PNG 변환 실패 시 None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = result.get("scan_date", datetime.now().strftime("%Y-%m-%d"))
    picks = result.get("top_picks", [])

    html = _build_html(result, picks)

    html_path = output_dir / f"consensus_{date_str}.html"
    html_path.write_text(html, encoding="utf-8")

    png_path = _html_to_png(html_path)
    return html_path, png_path


def _build_html(result: dict, picks: list[dict]) -> str:
    """HTML 보고서 문자열 생성."""
    date_str = result.get("scan_date", "")
    universe = result.get("universe_size", 0)
    with_consensus = result.get("with_consensus", 0)
    passed = result.get("passed_filter", 0)

    cards_html = ""
    for p in picks:
        cards_html += _pick_card(p)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>컨센서스 스크리닝 {date_str}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; padding: 20px; }}
  .header {{ background: linear-gradient(135deg, #1a2332, #0d1117); border: 1px solid #30363d;
             border-radius: 12px; padding: 20px; margin-bottom: 16px; }}
  .header h1 {{ font-size: 22px; color: #58a6ff; margin-bottom: 8px; }}
  .header .stats {{ color: #8b949e; font-size: 14px; }}
  .header .stats span {{ color: #58a6ff; font-weight: bold; }}
  .cards {{ display: flex; flex-direction: column; gap: 10px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px;
           padding: 16px; display: flex; align-items: center; gap: 16px; }}
  .card:hover {{ border-color: #58a6ff33; }}
  .rank {{ font-size: 24px; font-weight: bold; color: #30363d; width: 36px; text-align: center; }}
  .rank.top3 {{ color: #f0883e; }}
  .info {{ flex: 1; }}
  .info .name {{ font-size: 16px; font-weight: bold; color: #e6edf3; }}
  .info .ticker {{ color: #8b949e; font-size: 12px; margin-left: 6px; }}
  .info .details {{ margin-top: 6px; display: flex; gap: 12px; flex-wrap: wrap; }}
  .info .detail {{ font-size: 13px; color: #8b949e; }}
  .info .detail b {{ color: #e6edf3; }}
  .upside {{ text-align: right; min-width: 100px; }}
  .upside .pct {{ font-size: 22px; font-weight: bold; }}
  .upside .pct.high {{ color: #3fb950; }}
  .upside .pct.mid {{ color: #d29922; }}
  .upside .pct.low {{ color: #8b949e; }}
  .upside .target {{ font-size: 12px; color: #8b949e; }}
  .score-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
                  font-size: 12px; font-weight: bold; margin-left: 8px; }}
  .score-S {{ background: #f0883e22; color: #f0883e; border: 1px solid #f0883e44; }}
  .score-A {{ background: #3fb95022; color: #3fb950; border: 1px solid #3fb95044; }}
  .score-B {{ background: #58a6ff22; color: #58a6ff; border: 1px solid #58a6ff44; }}
  .score-C {{ background: #8b949e22; color: #8b949e; border: 1px solid #8b949e44; }}
  .bar {{ height: 4px; border-radius: 2px; background: #21262d; margin-top: 6px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 2px; }}
  .footer {{ margin-top: 16px; text-align: center; color: #484f58; font-size: 12px; }}
</style>
</head>
<body>
<div class="header">
  <h1>컨센서스 스크리닝 리포트</h1>
  <div class="stats">
    {date_str} |
    유니버스 <span>{universe}</span>종목 →
    컨센서스 <span>{with_consensus}</span>개 →
    필터 <span>{passed}</span>개 →
    TOP <span>{len(picks)}</span>
  </div>
</div>
<div class="cards">
{cards_html}
</div>
<div class="footer">
  Quantum Master Consensus Scanner | 5축 100점 (상승여력35 + PER매력20 + 확신15 + 기술20 + 배당10)
</div>
</body>
</html>"""


def _pick_card(p: dict) -> str:
    """개별 종목 카드 HTML."""
    rank = p.get("rank", 0)
    name = p.get("name", "")
    ticker = p.get("ticker", "")
    close = p.get("close", 0)
    target = p.get("target_price", 0)
    upside = p.get("upside_pct", 0)
    opinion = p.get("opinion_score")
    analysts = p.get("analyst_count", 0)
    fper = p.get("forward_per")
    div_yield = p.get("dividend_yield", 0)
    score = p.get("composite_score", 0)
    grade = p.get("grade", "D")
    rsi = p.get("rsi")
    ma60 = p.get("above_ma60")
    sar = p.get("sar_trend")
    dist = p.get("dist_high_52w")

    rank_cls = "rank top3" if rank <= 3 else "rank"
    upside_cls = "high" if upside >= 25 else "mid" if upside >= 15 else "low"

    opinion_s = f"{opinion:.2f}" if opinion else "N/A"
    fper_s = f"{fper:.1f}" if fper else "N/A"
    div_s = f"{div_yield:.1f}%" if div_yield > 0 else "-"

    # 기술 태그
    tech_tags = []
    if ma60:
        tech_tags.append("MA60↑")
    if rsi:
        tech_tags.append(f"RSI{rsi:.0f}")
    if sar == 1:
        tech_tags.append("SAR↑")
    if dist is not None:
        tech_tags.append(f"고점{dist:+.0f}%")
    tech_s = " | ".join(tech_tags) if tech_tags else ""

    bar_pct = min(100, score)
    bar_color = "#f0883e" if grade == "S" else "#3fb950" if grade == "A" else "#58a6ff" if grade == "B" else "#8b949e"

    return f"""
  <div class="card">
    <div class="{rank_cls}">{rank}</div>
    <div class="info">
      <div>
        <span class="name">{name}</span>
        <span class="ticker">({ticker})</span>
        <span class="score-badge score-{grade}">{grade} {score:.0f}점</span>
      </div>
      <div class="details">
        <span class="detail">현재가 <b>{close:,}</b></span>
        <span class="detail">의견 <b>{opinion_s}</b>/5 ({analysts}사)</span>
        <span class="detail">F-PER <b>{fper_s}</b></span>
        <span class="detail">배당 <b>{div_s}</b></span>
      </div>
      <div class="details">
        <span class="detail">{tech_s}</span>
      </div>
      <div class="bar"><div class="bar-fill" style="width:{bar_pct}%;background:{bar_color};"></div></div>
    </div>
    <div class="upside">
      <div class="pct {upside_cls}">+{upside:.1f}%</div>
      <div class="target">→ {target:,}원</div>
    </div>
  </div>"""


def _html_to_png(html_path: Path) -> Path | None:
    """Playwright로 HTML → PNG 변환."""
    png_path = html_path.with_suffix(".png")
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 800, "height": 600})
            page.goto(f"file:///{html_path.as_posix()}")
            page.wait_for_timeout(500)

            body = page.query_selector("body")
            if body:
                body.screenshot(path=str(png_path))
            else:
                page.screenshot(path=str(png_path), full_page=True)

            browser.close()

        return png_path
    except Exception as e:
        logger.warning(f"PNG 변환 실패: {e}")
        return None
