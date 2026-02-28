"""ETF AI 필터 레이어 — 룰베이스 주문에 대한 AI 방어 필터

구조:
  데이터 → 룰베이스 엔진 → [매수 신호]
                              ↓
                        AI 필터 레이어
                              ↓
                      "위험 신호 있나?"
                        /          \\
                     없음           있음
                      ↓              ↓
                 주문 실행         보류 + 알림

핵심 원칙:
  - AI는 "사라"고 절대 안 함. "사지 마라"만 할 수 있음.
  - 공격은 룰이, 방어 보강만 AI가.
  - 개입 1: KILL — 룰이 매수라는데 위험 신호 감지 → 매수 보류
  - 개입 2: HOLD — 룰이 교체라는데 기존 추세 강함 → 교체 연기
  - 개입 3: WARNING — 룰에 없는 이상 징후 → 사람에게 알림
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ── 뉴스/맥락 수집 (기존 JSON 재활용) ──

def _load_json(name: str) -> dict | list:
    path = DATA_DIR / name
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def build_news_context(max_items: int = 30) -> str:
    """기존 뉴스 JSON에서 ETF 관련 맥락 추출.

    5개 소스 재활용:
      - market_intelligence.json (Perplexity)
      - market_news.json (RSS)
      - ai_brain_judgment.json (개별종목 AI — sector_outlook 재활용)
      - dart_disclosures.json (공시)
      - overnight_signal.json (US 야간)
    """
    lines = []

    # 1. Perplexity 인텔리전스 — 가장 유용
    intel = _load_json("market_intelligence.json")
    if intel:
        us_summary = intel.get("us_market_summary", "")
        if us_summary:
            lines.append(f"[미국장] {us_summary[:200]}")
        for ev in intel.get("key_events", [])[:5]:
            lines.append(f"[이벤트] {ev.get('event', '')} — {ev.get('detail', '')[:100]}")
        for sec in intel.get("sector_impacts", [])[:5]:
            lines.append(f"[섹터:{sec.get('sector', '')}] {sec.get('reason', '')[:100]}")

    # 2. RSS 뉴스 — high/medium만
    news = _load_json("market_news.json")
    for a in (news.get("articles", []) if isinstance(news, dict) else []):
        if a.get("impact") in ("high", "medium"):
            lines.append(f"[뉴스] {a.get('title', '')}")

    # 3. AI 두뇌 sector_outlook — 개별종목 AI가 이미 분석한 섹터 전망 재활용
    ai_brain = _load_json("ai_brain_judgment.json")
    if ai_brain:
        sentiment = ai_brain.get("market_sentiment", "")
        if sentiment:
            lines.append(f"[AI 센티먼트] {sentiment}")
        themes = ai_brain.get("key_themes", [])
        if themes:
            lines.append(f"[AI 테마] {', '.join(themes[:3])}")
        for sec, info in ai_brain.get("sector_outlook", {}).items():
            if isinstance(info, dict):
                d = info.get("direction", "")
                r = info.get("reason", "")
                lines.append(f"[섹터전망:{sec}] {d} — {r[:80]}")

    # 4. DART 공시 — tier1만
    dart = _load_json("dart_disclosures.json")
    for d in (dart.get("tier1", []) if isinstance(dart, dict) else []):
        lines.append(f"[DART] {d.get('corp_name', '')} — {d.get('report_nm', '')}")

    # 5. US Overnight 요약
    us = _load_json(str(Path("us_market") / "overnight_signal.json"))
    if us:
        lines.append(
            f"[US야간] 등급:{us.get('composite', 'N/A')} "
            f"VIX:{us.get('vix_close', 'N/A')} "
            f"EWY:{us.get('ewy_change_pct', 0):+.1f}%"
        )

    # 최대 개수 제한
    if len(lines) > max_items:
        lines = lines[:max_items]

    return "\n".join(lines) if lines else ""


def build_market_context(regime_data: dict, us_data: dict) -> str:
    """시장 상태 요약 텍스트 생성."""
    lines = []
    lines.append(f"KOSPI: {regime_data.get('close', 0):,.0f}")
    lines.append(f"레짐: {regime_data.get('regime', '?')}")
    lines.append(f"MA20 위: {'O' if regime_data.get('ma20_above') else 'X'}")
    lines.append(f"MA60 위: {'O' if regime_data.get('ma60_above') else 'X'}")
    lines.append(f"US Overnight: {us_data.get('grade', 'N/A')}등급 ({us_data.get('signal', '')})")
    return " | ".join(lines)


# ── AI 필터 실행 ──

def apply_ai_filter(
    order_queue: list[dict],
    regime: str,
    allocation: dict,
    regime_data: dict,
    us_data: dict,
    model: str = "claude-sonnet-4-5-20250929",
) -> dict:
    """AI 필터를 동기적으로 실행 (asyncio.run 래퍼).

    Returns:
        {
            "filtered_queue": [...],      # KILL 제거된 주문 큐
            "killed_orders": [...],       # KILL된 주문 + 사유
            "held_orders": [...],         # HOLD된 주문 + 사유
            "warnings": [...],            # 경고 메시지
            "ai_result": {...},           # AI 원본 응답
            "stats": {"total": N, "pass": N, "kill": N, "hold": N},
        }
    """
    news_context = build_news_context()
    market_context = build_market_context(regime_data, us_data)

    # AI 호출
    from src.agents.etf_brain import ETFBrainAgent
    agent = ETFBrainAgent(model=model)
    ai_result = asyncio.run(agent.filter_orders(
        order_queue=order_queue,
        regime=regime,
        allocation=allocation,
        news_context=news_context,
        market_context=market_context,
    ))

    # 판정 적용
    verdicts = {v.get("code", ""): v for v in ai_result.get("order_verdicts", [])}

    filtered_queue = []
    killed_orders = []
    held_orders = []

    for order in order_queue:
        code = order.get("code", "")
        v = verdicts.get(code, {})
        verdict = v.get("verdict", "PASS").upper()

        if verdict == "KILL" and order.get("action") == "BUY":
            # KILL: 매수 주문만 차단 (매도는 항상 통과)
            killed_orders.append({
                **order,
                "ai_verdict": "KILL",
                "ai_reason": v.get("reason", ""),
                "ai_confidence": v.get("confidence", 0),
            })
        elif verdict == "HOLD":
            held_orders.append({
                **order,
                "ai_verdict": "HOLD",
                "ai_reason": v.get("reason", ""),
                "ai_confidence": v.get("confidence", 0),
            })
        else:
            # PASS (또는 verdict 없는 주문 → 안전 PASS)
            filtered_queue.append(order)

    stats = {
        "total": len(order_queue),
        "pass": len(filtered_queue),
        "kill": len(killed_orders),
        "hold": len(held_orders),
    }

    logger.info(
        "AI 필터 결과: %d건 중 PASS=%d, KILL=%d, HOLD=%d",
        stats["total"], stats["pass"], stats["kill"], stats["hold"],
    )

    return {
        "filtered_queue": filtered_queue,
        "killed_orders": killed_orders,
        "held_orders": held_orders,
        "warnings": ai_result.get("warnings", []),
        "ai_result": ai_result,
        "stats": stats,
    }


def build_ai_telegram_section(filter_result: dict) -> str:
    """AI 필터 결과를 텔레그램 리포트 섹션으로 변환."""
    lines = []
    ai = filter_result.get("ai_result", {})
    stats = filter_result.get("stats", {})

    lines.append("\n🧠 [AI 필터]")
    lines.append(f"  {ai.get('market_assessment', '분석 없음')}")
    lines.append(
        f"  리스크: {ai.get('risk_level', '?')} | "
        f"PASS {stats.get('pass', 0)} / "
        f"KILL {stats.get('kill', 0)} / "
        f"HOLD {stats.get('hold', 0)}"
    )

    # KILL된 주문
    for k in filter_result.get("killed_orders", []):
        lines.append(f"  🚫 KILL: {k['name']} — {k.get('ai_reason', '')[:50]}")

    # HOLD된 주문
    for h in filter_result.get("held_orders", []):
        lines.append(f"  ⏸️ HOLD: {h['name']} — {h.get('ai_reason', '')[:50]}")

    # 경고
    for w in filter_result.get("warnings", []):
        lines.append(f"  ⚠️ {w[:60]}")

    # 섹터 리스크 노트
    for sec, note in ai.get("sector_risk_notes", {}).items():
        lines.append(f"  📌 {sec}: {note[:50]}")

    return "\n".join(lines)
