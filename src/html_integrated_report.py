"""통합 데일리 리포트 HTML 생성 + PNG 변환.

daily_integrated_report.py에서 호출.
기존 html_report.py의 다크 테마 CSS 패턴을 재활용.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("D:/클로드 HTML 보고서")


def _html_to_png(html_path: Path, png_path: Path, width: int = 800):
    """Playwright로 HTML → PNG 캡처."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": 600})
        page.goto(f"file:///{html_path.as_posix()}")
        page.wait_for_timeout(500)
        height = page.evaluate("document.body.scrollHeight")
        page.set_viewport_size({"width": width, "height": height + 40})
        page.screenshot(path=str(png_path), full_page=True)
        browser.close()


def generate_integrated_report(
    data: dict,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path | None]:
    """통합 데일리 리포트 HTML + PNG 생성.

    Args:
        data: daily_integrated_report.collect_all_signals()의 반환값 + action_plan
        output_dir: 저장 디렉토리

    Returns:
        (html_path, png_path) — png 변환 실패 시 None
    """
    date_str = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    time_str = data.get("generated_at", "").split(" ")[-1] if " " in data.get("generated_at", "") else ""

    html_content = _build_html(data, date_str, time_str)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"통합_데일리_{date_str}.html"
    png_path = output_dir / f"통합_데일리_{date_str}.png"

    html_path.write_text(html_content, encoding="utf-8")

    try:
        _html_to_png(html_path, png_path)
        logger.info("PNG 변환 완료: %s", png_path)
    except Exception as e:
        logger.error("PNG 변환 실패: %s", e)
        png_path = None

    return html_path, png_path


def _grade_color(grade: str) -> str:
    """등급별 색상."""
    return {
        "S": "#f0883e", "A": "#3fb950", "B": "#58a6ff",
        "C": "#8b949e", "D": "#6e7681",
    }.get(grade, "#8b949e")


def _pnl_color(pnl: float) -> str:
    """수익률 색상."""
    if pnl > 0:
        return "#f85149" if pnl >= 3 else "#3fb950"
    elif pnl < 0:
        return "#58a6ff" if pnl <= -3 else "#58a6ff"
    return "#8b949e"


def _stance_color(stance: str) -> str:
    """시장 스탠스 색상."""
    if "적극" in stance:
        return "#f0883e"
    elif "선별" in stance:
        return "#3fb950"
    elif "관망" in stance:
        return "#8b949e"
    elif "자제" in stance or "금지" in stance:
        return "#f85149"
    elif "청산" in stance:
        return "#da3633"
    return "#8b949e"


def _us_grade_color(grade: str) -> str:
    if "BULL" in grade:
        return "#3fb950"
    elif "BEAR" in grade:
        return "#f85149"
    return "#8b949e"


def _regime_color(regime: str) -> str:
    return {
        "BULL": "#3fb950", "CAUTION": "#d29922",
        "BEAR": "#f85149", "CRISIS": "#da3633",
    }.get(regime, "#8b949e")


def _ret_class(val: float) -> str:
    return "up" if val > 0 else "down" if val < 0 else ""


def _build_html(data: dict, date_str: str, time_str: str) -> str:
    """통합 리포트 HTML 생성."""
    plan = data.get("action_plan", {})
    us = data.get("us_overnight", {})
    kospi = data.get("kospi_regime", {})
    candidates = data.get("quantum", {}).get("candidates", [])
    relay = data.get("relay", {})
    positions = data.get("positions", {})

    # ── 섹션 1: 시장 온도 ──
    us_grade = us.get("grade", "NEUTRAL")
    us_score = us.get("combined_score_100", 0)
    idx = us.get("index_direction", {})
    vix = us.get("vix", {})

    market_temp_html = _build_market_temp(us_grade, us_score, idx, vix, kospi, plan)

    # ── 섹션 2: Quantum 매수 후보 ──
    quantum_html = _build_quantum_section(candidates[:5])

    # ── 섹션 3: 릴레이 시그널 ──
    relay_html = _build_relay_section(relay)

    # ── 섹션 4: 보유 포지션 ──
    positions_html = _build_positions_section(positions)

    # ── 섹션 5: 액션 플랜 ──
    action_html = _build_action_section(plan)

    stance = plan.get("market_stance", "관망")
    stance_color = _stance_color(stance)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantum Master 통합 데일리 - {date_str}</title>
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
        font-size: 22px;
        font-weight: 800;
        color: #f0883e;
        letter-spacing: 1px;
    }}
    .report-subtitle {{
        font-size: 13px;
        color: #8b949e;
        margin-top: 6px;
    }}
    .section-title {{
        font-size: 15px;
        font-weight: 700;
        color: #f0883e;
        margin: 20px 0 10px;
        padding-left: 8px;
        border-left: 3px solid #f0883e;
    }}
    .card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }}
    .card-highlight {{
        border-color: #f0883e;
        box-shadow: 0 0 8px rgba(240, 136, 62, 0.1);
    }}
    .row {{ display: flex; align-items: center; gap: 10px; }}
    .row-between {{ display: flex; justify-content: space-between; align-items: center; }}
    .label {{ font-size: 12px; font-weight: 600; color: #8b949e; }}
    .value {{ font-size: 16px; font-weight: 800; }}
    .small {{ font-size: 12px; color: #8b949e; }}
    .up {{ color: #f85149; }}
    .down {{ color: #58a6ff; }}
    .neutral {{ color: #8b949e; }}
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
    }}
    .stance-banner {{
        text-align: center;
        padding: 10px;
        margin: 12px 0;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 2px;
    }}
    .idx-row {{
        display: flex;
        gap: 16px;
        font-size: 13px;
        margin-top: 6px;
    }}
    .stock-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0;
        border-bottom: 1px solid #21262d;
    }}
    .stock-row:last-child {{ border-bottom: none; }}
    .grade-badge {{
        font-size: 14px;
        font-weight: 900;
        color: #0d1117;
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 6px;
        flex-shrink: 0;
    }}
    .stock-name {{ font-size: 14px; font-weight: 700; }}
    .stock-code {{ font-size: 11px; color: #8b949e; }}
    .stock-price {{ margin-left: auto; text-align: right; }}
    .fire-card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }}
    .fire-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }}
    .fire-sector {{ font-size: 15px; font-weight: 700; color: #f85149; }}
    .arrow {{ color: #f0883e; font-size: 18px; font-weight: 800; }}
    .follow-sector {{ font-size: 15px; font-weight: 700; color: #3fb950; }}
    .picks {{ font-size: 12px; color: #8b949e; margin-top: 6px; padding-left: 12px; }}
    .pos-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        font-size: 13px;
        border-bottom: 1px solid #21262d;
    }}
    .pos-row:last-child {{ border-bottom: none; }}
    .pos-source {{
        font-size: 11px;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 3px;
    }}
    .pos-source.quantum {{ background: #1f6feb; color: #fff; }}
    .pos-source.relay {{ background: #8957e5; color: #fff; }}
    .action-group {{ margin-bottom: 10px; }}
    .action-label {{
        font-size: 12px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 8px;
    }}
    .action-buy {{ background: #f8514933; color: #f85149; }}
    .action-sell {{ background: #58a6ff33; color: #58a6ff; }}
    .action-watch {{ background: #d2992233; color: #d29922; }}
    .action-item {{ font-size: 13px; margin: 4px 0 4px 16px; }}
    .footer {{
        text-align: center;
        font-size: 11px;
        color: #484f58;
        margin-top: 20px;
        padding-top: 12px;
        border-top: 1px solid #21262d;
    }}
</style>
</head>
<body>

<div class="report-header">
    <div class="report-title">QUANTUM MASTER 통합 데일리</div>
    <div class="report-subtitle">{date_str} {time_str} | v10.3</div>
</div>

{market_temp_html}

<div class="stance-banner" style="background:{stance_color}22; color:{stance_color}; border:1px solid {stance_color}44;">
    {stance}
</div>

{quantum_html}
{relay_html}
{positions_html}
{action_html}

<div class="footer">
    투자 판단은 본인 책임 | Quantum Master v10.3 통합 데일리 리포트
</div>

</body>
</html>"""


def _build_market_temp(us_grade, us_score, idx, vix, kospi, plan) -> str:
    """섹션 1: 시장 온도."""
    # US 지수
    ewy = idx.get("EWY", {})
    spy = idx.get("SPY", {})
    qqq = idx.get("QQQ", {})

    ewy_ret = ewy.get("ret_1d", 0)
    spy_ret = spy.get("ret_1d", 0)
    qqq_ret = qqq.get("ret_1d", 0)

    vix_level = vix.get("level", 0)
    vix_status = vix.get("status", "")

    # KOSPI
    regime = kospi.get("regime", "CAUTION")
    slots = kospi.get("slots", 0)
    close = kospi.get("close", 0)
    ma20 = kospi.get("ma20", 0)
    ma60 = kospi.get("ma60", 0)

    us_color = _us_grade_color(us_grade)
    regime_color = _regime_color(regime)

    return f"""
<div class="section-title">시장 온도</div>

<div class="card">
    <div class="row-between">
        <span class="label">US Overnight</span>
        <span class="value" style="color:{us_color}">{us_grade}</span>
        <span class="value" style="color:{us_color}">{us_score:+.1f}</span>
    </div>
    <div class="idx-row">
        <span class="{_ret_class(ewy_ret)}">EWY {ewy_ret:+.1f}%</span>
        <span class="{_ret_class(spy_ret)}">SPY {spy_ret:+.1f}%</span>
        <span class="{_ret_class(qqq_ret)}">QQQ {qqq_ret:+.1f}%</span>
        <span class="neutral">VIX {vix_level:.0f} [{vix_status}]</span>
    </div>
</div>

<div class="card">
    <div class="row-between">
        <span class="label">KOSPI 레짐</span>
        <span class="value" style="color:{regime_color}">{regime}</span>
        <span class="value" style="color:{regime_color}">{slots}슬롯</span>
    </div>
    <div class="idx-row">
        <span>KOSPI {close:,.0f}</span>
        <span>MA20 {ma20:,.0f}</span>
        <span>MA60 {ma60:,.0f}</span>
    </div>
</div>
"""


def _build_quantum_section(candidates: list[dict]) -> str:
    """섹션 2: Quantum 매수 후보."""
    if not candidates:
        return """
<div class="section-title">Quantum 매수 후보</div>
<div class="card">
    <span class="small">v8 게이트 통과 종목 없음 &mdash; Quantum 시그널 대기중</span>
    <div class="small" style="margin-top:6px;color:#484f58">Gate: ADX&ge;18 + Pullback&le;0.8 + Overheat&lt;0.92</div>
</div>
"""

    rows = ""
    for i, c in enumerate(candidates[:5], 1):
        grade = c.get("grade", "?")
        name = c.get("name", "")
        ticker = c.get("ticker", "")
        entry = c.get("entry_price", 0)
        target = c.get("target_price", 0)
        stop = c.get("stop_loss", 0)
        rr = c.get("risk_reward", 0)
        g_color = _grade_color(grade)

        target_pct = (target - entry) / entry * 100 if entry > 0 else 0
        stop_pct = (stop - entry) / entry * 100 if entry > 0 else 0

        highlight = ' card-highlight' if grade == 'S' else ''

        rows += f"""
<div class="card{highlight}">
    <div class="stock-row" style="border-bottom:none">
        <div class="grade-badge" style="background:{g_color}">{grade}</div>
        <div>
            <div class="stock-name">{name}</div>
            <div class="stock-code">{ticker} | RR 1:{rr:.1f}</div>
        </div>
        <div class="stock-price">
            <div style="font-size:16px;font-weight:700">{entry:,.0f}원</div>
            <div class="small">
                <span class="up">+{target:,.0f}({target_pct:+.1f}%)</span> /
                <span class="down">{stop:,.0f}({stop_pct:+.1f}%)</span>
            </div>
        </div>
    </div>
</div>
"""

    return f"""
<div class="section-title">Quantum 매수 후보 ({len(candidates[:5])}종목)</div>
{rows}
"""


def _build_relay_section(relay: dict) -> str:
    """섹션 3: 릴레이 시그널."""
    fired = relay.get("fired_sectors", [])
    signals = relay.get("relay_signals", [])

    # 분류: 진입적기 vs 이미움직임
    actionable = []
    already_moved = []
    for s in signals:
        follow_ret = s.get("follow_stats", {}).get("avg_return", 0)
        conf = s["pattern"]["confidence"]
        if follow_ret < 3.0 and conf in ("HIGH", "MED"):
            actionable.append(s)
        elif follow_ret >= 3.0:
            already_moved.append(s)

    if not actionable and not fired:
        return """
<div class="section-title">릴레이 시그널</div>
<div class="card"><span class="small">발화 섹터 없음</span></div>
"""

    cards = ""

    # 진입적기 카드
    for sig in actionable:
        p = sig["pattern"]
        follow_ret = sig.get("follow_stats", {}).get("avg_return", 0)
        conf_color = "#3fb950" if p["confidence"] == "HIGH" else "#d29922"
        picks_html = ""
        for pk in sig.get("picks", [])[:3]:
            picks_html += f"<div>{pk['name']}({pk['ticker']}) 점수:{pk.get('score', 0)} {pk.get('change_pct', 0):+.1f}%</div>"

        cards += f"""
<div class="fire-card">
    <div class="fire-header">
        <span class="fire-sector">{sig['lead_sector']} {sig['lead_return']:+.1f}%</span>
        <span class="arrow">&rarr;</span>
        <span class="follow-sector">{sig['follow_sector']}</span>
        <span class="badge" style="background:{conf_color}33;color:{conf_color}">{p['confidence']}</span>
    </div>
    <div class="small">lag{p['best_lag']}일 | 승률{p['win_rate']:.0f}% | n={p['samples']} | 후행 {follow_ret:+.1f}% 대기중</div>
    <div class="picks">{picks_html}</div>
</div>
"""

    # 이미움직임 카드 (회색)
    if already_moved:
        moved_items = ""
        for sig in already_moved[:3]:
            p = sig["pattern"]
            follow_ret = sig.get("follow_stats", {}).get("avg_return", 0)
            moved_items += (
                f"<div style='padding:4px 0;border-bottom:1px solid #21262d'>"
                f"<span style='color:#6e7681'>{sig['lead_sector']}</span> "
                f"<span style='color:#484f58'>&rarr;</span> "
                f"<span style='color:#6e7681'>{sig['follow_sector']}</span> "
                f"<span style='color:#6e7681'>+{follow_ret:.1f}%</span> "
                f"<span style='font-size:11px;color:#484f58'>이미움직임</span>"
                f"</div>"
            )
        cards += f"""
<div class="card" style="border-color:#21262d;opacity:0.7">
    <div class="small" style="margin-bottom:4px;color:#484f58">이미움직임 (참고)</div>
    {moved_items}
</div>
"""

    if not actionable and fired:
        title = f"릴레이 시그널 (발화{len(fired)}개)"
    else:
        title = f"릴레이 시그널 ({len(actionable)}건 진입적기)"

    return f"""
<div class="section-title">{title}</div>
{cards}
"""


def _build_positions_section(positions: dict) -> str:
    """섹션 4: 보유 포지션."""
    q_pos = positions.get("quantum", {}).get("positions", [])
    r_pos = positions.get("relay", {}).get("positions", [])
    total = positions.get("total_count", 0)

    if total == 0:
        return """
<div class="section-title">보유 포지션</div>
<div class="card"><span class="small">보유 포지션 없음</span></div>
"""

    rows = ""
    for p in q_pos:
        name = p.get("name", p.get("ticker", ""))
        rows += f"""
<div class="pos-row">
    <span class="pos-source quantum">Q</span>
    <span style="font-weight:600">{name}</span>
    <span class="small" style="margin-left:auto">{p.get('entry_date', '')}</span>
</div>
"""

    for p in r_pos:
        pnl = p.get("pnl_pct", 0)
        pnl_c = _pnl_color(pnl)
        days = p.get("trading_days_held", 0)
        rows += f"""
<div class="pos-row">
    <span class="pos-source relay">R</span>
    <span style="font-weight:600">{p.get('name', '')}</span>
    <span class="small">({p.get('ticker', '')})</span>
    <span style="color:{pnl_c};font-weight:700;margin-left:auto">{pnl:+.1f}%</span>
    <span class="small">{days}일째</span>
    <span class="small">{p.get('fired_sector', '')}→{p.get('sector', '')}</span>
</div>
"""

    total_invested = positions.get("total_invested", 0)
    total_pnl = positions.get("total_relay_pnl", 0)
    summary = ""
    if total_invested > 0:
        summary = f'<div class="small" style="margin-top:8px;text-align:right">투입: {total_invested:,}원 | 평가: {total_pnl:+,}원</div>'

    return f"""
<div class="section-title">보유 포지션 ({total}건)</div>
<div class="card">
    {rows}
    {summary}
</div>
"""


def _build_action_section(plan: dict) -> str:
    """섹션 5: 액션 플랜."""
    buys = plan.get("buys", [])
    sells = plan.get("sells", [])
    watches = plan.get("watches", [])
    stance = plan.get("market_stance", "")

    buy_html = ""
    if buys:
        for b in buys:
            src = "Q" if b["source"] == "Quantum" else "R"
            if b["source"] == "Quantum":
                extra = f" | {b.get('entry_price', 0):,.0f}원 RR 1:{b.get('risk_reward', 0):.1f}"
            else:
                extra = f" | {b.get('fired_sector', '')}&rarr;{b.get('sector', '')} 승률{b.get('win_rate', 0):.0f}%"
            buy_html += f'<div class="action-item">[{src}] {b["name"]}{extra}</div>'
    else:
        buy_html = '<div class="action-item">없음</div>'

    sell_html = ""
    if sells:
        for s in sells:
            sell_html += f'<div class="action-item">{s["name"]} &mdash; {s.get("reason", "")}</div>'
    else:
        sell_html = '<div class="action-item">없음</div>'

    watch_html = ""
    if watches:
        # 동일 follow 섹터 머지
        follow_map: dict[str, list[str]] = {}
        other_watches = []
        for w in watches[:5]:
            name = w.get("name", "")
            if "\u2192" in name or "->" in name:
                parts = name.replace("\u2192", "->").split("->")
                if len(parts) == 2:
                    lead = parts[0].strip()
                    follow = parts[1].strip()
                    follow_map.setdefault(follow, []).append(lead)
                    continue
            other_watches.append(w)
        for follow, leads in follow_map.items():
            watch_html += (
                f'<div class="action-item">'
                f'{follow} &larr; {"+".join(leads)} 발화'
                f'</div>'
            )
        for w in other_watches:
            watch_html += f'<div class="action-item">{w.get("name", "")} ({w.get("reason", "")})</div>'
        if "관망" in stance or "자제" in stance:
            watch_html += f'<div class="small" style="margin-top:4px;color:#d29922">* {stance} &mdash; 발동시 소량만</div>'
    else:
        watch_html = '<div class="action-item">없음</div>'

    return f"""
<div class="section-title">액션 플랜</div>
<div class="card">
    <div class="action-group">
        <span class="action-label action-buy">매수</span>
        {buy_html}
    </div>
    <div class="action-group">
        <span class="action-label action-sell">매도</span>
        {sell_html}
    </div>
    <div class="action-group">
        <span class="action-label action-watch">감시</span>
        {watch_html}
    </div>
</div>
"""
