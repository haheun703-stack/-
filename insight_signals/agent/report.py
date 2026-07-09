# -*- coding: utf-8 -*-
"""일일 관찰 리포트(markdown) 생성."""
from __future__ import annotations

from insight_signals.entities import SOURCE_DART, SOURCE_FLOW, SOURCE_NEWS

_SRC_LABEL = {
    SOURCE_NEWS: "📰 뉴스 키워드",
    SOURCE_DART: "🏛 임원 자사주 매수 (DART)",
    SOURCE_FLOW: "⚖ 역발상 수급",
}


def render_daily(date: str, signals, picks, perf_summary=None) -> str:
    lines = [
        f"# 인사이트 시그널 일일 리포트 — {date}",
        "",
        "> 관찰 모드: 이 리포트는 매매에 개입하지 않습니다. "
        "시그널의 실제 수익률이 검증되면 스코어링 승격을 검토하세요.",
        "",
        "## 오늘의 관찰 픽 (소스 합산)",
        "",
    ]
    if picks:
        lines.append("| 순위 | 종목 | 코드 | 합산점수 | 소스 | 픽 시점가 |")
        lines.append("|---|---|---|---|---|---|")
        for i, p in enumerate(picks, 1):
            srcs = ", ".join(_SRC_LABEL.get(s, s) for s in p.sources)
            price = f"{p.price_at_pick:,.0f}" if p.price_at_pick else "-"
            lines.append(
                f"| {i} | {p.stock_name} | {p.stock_code} | "
                f"{p.combined_score:.3f} | {srcs} | {price} |"
            )
        lines.append("")
        for p in picks:
            lines.append(f"### {p.stock_name} ({p.stock_code})")
            for r in p.reasons:
                lines.append(f"- {r}")
            lines.append("")
    else:
        lines.append("(오늘은 기준을 넘는 픽이 없습니다 — 이것도 데이터입니다)")
        lines.append("")

    lines.append("## 소스별 원시 시그널")
    lines.append("")
    for src in (SOURCE_NEWS, SOURCE_DART, SOURCE_FLOW):
        sigs = sorted(
            (s for s in signals if s.source == src),
            key=lambda s: s.score,
            reverse=True,
        )
        lines.append(f"### {_SRC_LABEL[src]} — {len(sigs)}건")
        for s in sigs[:15]:
            lines.append(f"- [{s.score:+.2f}] **{s.stock_name}** ({s.stock_code}) — {s.reason}")
        if not sigs:
            lines.append("- 없음")
        lines.append("")

    if perf_summary:
        lines.append("## 누적 성과 (관찰 픽의 실제 수익률)")
        lines.append("")
        lines.append("| 경과 | 표본 수 | 평균 수익률 | 승률 |")
        lines.append("|---|---|---|---|")
        for h in sorted(perf_summary):
            s = perf_summary[h]
            lines.append(f"| +{h}일 | {s['n']} | {s['avg']:+.2f}% | {s['win_rate']}% |")
        lines.append("")

    return "\n".join(lines)
