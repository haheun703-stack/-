"""
장마감 저녁 통합 리포트 — BAT-D 마지막 단계에서 호출.

데이터 소스 (모두 BAT-D 이전 단계에서 생성된 JSON):
  - data/picks_history.json        → 보유종목 재판정 결과 (monitor_action)
  - data/dart_disclosures.json     → DART 공시 (tier1 + universe)
  - data/tomorrow_picks.json       → 내일 추천 TOP10
  - data/value_chain_relay.json    → 밸류체인 발화 현황
  - data/market_intelligence.json  → Perplexity 인텔리전스

수동 실행:
    python scripts/send_evening_summary.py            # 미리보기
    python scripts/send_evening_summary.py --send     # 텔레그램 발송
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"


def _load(name: str) -> dict | list:
    path = DATA_DIR / name
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ───────────────────────────────────────
# 섹션 빌더
# ───────────────────────────────────────

def _section_holdings() -> list[str]:
    """보유종목 재판정 요약 (picks_history.json에서 당일 monitor 결과)."""
    data = _load("picks_history.json")
    today = datetime.now().strftime("%Y-%m-%d")
    lines = []

    monitored = []
    for rec in data if isinstance(data, list) else []:
        if rec.get("monitor_date") == today and rec.get("monitor_action"):
            monitored.append(rec)

    if not monitored:
        return []

    ACTION_EMOJI = {
        "ADD": "\U0001f535", "HOLD": "\U0001f7e2",
        "PARTIAL_SELL": "\U0001f7e1", "FULL_SELL": "\U0001f534",
    }
    ACTION_LABEL = {
        "ADD": "추매", "HOLD": "보유",
        "PARTIAL_SELL": "부분매도", "FULL_SELL": "전량매도",
    }

    lines.append("")
    lines.append("\u2501\u2501 보유종목 현황 \u2501\u2501")
    for rec in monitored:
        action = rec.get("monitor_action", "HOLD")
        emoji = ACTION_EMOJI.get(action, "\u26aa")
        label = ACTION_LABEL.get(action, action)
        name = rec.get("name", rec.get("ticker", "?"))
        target = rec.get("monitor_target", 0)
        pnl = rec.get("pnl_pct", 0)
        lines.append(
            f"{emoji} {name}: {pnl:+.1f}% \u2192 {label}"
            + (f" (목표 {target:,.0f})" if target else "")
        )

    return lines


def _section_dart() -> list[str]:
    """DART 공시 요약 (tier1 위주)."""
    data = _load("dart_disclosures.json")
    if not data:
        return []

    tier1 = data.get("tier1", [])
    universe = data.get("universe_hits", [])

    if not tier1 and not universe:
        return []

    lines = ["\n\u2501\u2501 DART 공시 \u2501\u2501"]

    for d in tier1[:3]:
        corp = d.get("corp_name", "?")
        kw = d.get("keyword", "")
        lines.append(f"\U0001f534 [즉시] {corp} \u2014 {kw}")

    for d in universe[:3]:
        corp = d.get("corp_name", "?")
        kw = d.get("keyword", "")
        lines.append(f"\U0001f7e1 [참고] {corp} \u2014 {kw}")

    return lines


def _section_picks() -> list[str]:
    """내일 추천 TOP10."""
    data = _load("tomorrow_picks.json")
    if not data:
        return []

    picks = data.get("picks", [])
    top_tickers = set(data.get("top5", []))
    swing_set = set(data.get("top5_swing", []))
    short_set = set(data.get("top5_short", []))

    if not top_tickers:
        return []

    top_picks = [p for p in picks if p.get("ticker") in top_tickers]

    # 스윙 먼저, 단타 뒤
    swing_picks = [p for p in top_picks if p["ticker"] in swing_set]
    short_picks = [p for p in top_picks if p["ticker"] in short_set]
    others = [p for p in top_picks if p["ticker"] not in swing_set and p["ticker"] not in short_set]

    target_label = data.get("target_date_label", "")
    lines = [f"\n\u2501\u2501 내일 추천 TOP{len(top_tickers)} {target_label} \u2501\u2501"]

    def _fmt(p_list, group_label, icon):
        if not p_list:
            return
        lines.append(f"{icon} {group_label}")
        for i, p in enumerate(p_list, 1):
            name = p.get("name", "?")
            srcs = "+".join(p.get("sources", []))
            score = p.get("total_score", p.get("score", 0))
            grade = p.get("grade", "")
            sar_icon = " SAR\u2191" if p.get("sar_trend") == 1 else (" SAR\u2193" if p.get("sar_trend") == -1 else "")
            lines.append(f"  {i}. {name} | {srcs} | {score:.0f}점 {grade}{sar_icon}")

    _fmt(swing_picks, "스윙(3~7일)", "\U0001f4c8")
    _fmt(short_picks, "단타(1~3일)", "\u26a1")
    if others:
        _fmt(others, "기타", "\U0001f4cc")

    return lines


def _section_value_chain() -> list[str]:
    """밸류체인 발화 요약 (섹터+대장주만, 1줄씩)."""
    data = _load("value_chain_relay.json")
    if not data:
        return []

    fired = data.get("fired_sectors", [])
    if not fired:
        return []

    lines = ["\n\u2501\u2501 밸류체인 발화 \u2501\u2501"]
    for sec in fired[:5]:
        sector = sec.get("sector", "?")
        leaders = [l.get("name", "?") for l in sec.get("leaders", [])]
        candidates = [c.get("name", "?") for c in sec.get("candidates", [])[:2]]
        leader_str = "+".join(leaders) + "\u2191"
        cand_str = ", ".join(candidates) if candidates else "대기 없음"
        lines.append(f"\U0001f517 {sector}: {leader_str} \u2192 {cand_str}")

    return lines


def _section_intel() -> list[str]:
    """Perplexity 시장 인텔리전스 요약 (무드+핫테마만)."""
    data = _load("market_intelligence.json")
    if not data:
        return []

    mood = data.get("mood", "")
    themes = data.get("hot_themes", [])
    forecast = data.get("forecast", "")

    if not mood and not themes:
        return []

    lines = ["\n\u2501\u2501 시장 무드 \u2501\u2501"]
    parts = []
    if mood:
        parts.append(f"\U0001f30d {mood}")
    if forecast:
        fc_icon = {"상승": "\u25b2", "하락": "\u25bc", "보합": "\u2500"}.get(forecast, "")
        parts.append(f"KR {fc_icon}{forecast}")
    if parts:
        lines.append(" | ".join(parts))
    if themes:
        lines.append(f"\U0001f525 " + " | ".join(themes[:4]))

    return lines


# ───────────────────────────────────────
# 메인 빌더
# ───────────────────────────────────────

def build_evening_summary() -> str:
    """저녁 통합 리포트 1건."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [
        f"\U0001f4cb 장마감 리포트 | {now}",
        "Quantum Master v10.3",
        "\u2501" * 24,
    ]

    # 각 섹션을 우선순위 순으로 추가
    L.extend(_section_holdings())
    L.extend(_section_dart())
    L.extend(_section_picks())
    L.extend(_section_value_chain())
    L.extend(_section_intel())

    # 빈 내용 체크
    if len(L) <= 3:
        L.append("\n\u26a0 오늘 발송할 내용이 없습니다.")

    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description="저녁 통합 리포트")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송")
    args = parser.parse_args()

    msg = build_evening_summary()

    if args.send:
        try:
            from src.telegram_sender import send_message
            ok = send_message(msg)
            print(f"[텔레그램] 발송 {'성공' if ok else '실패'}")
        except Exception as e:
            print(f"[텔레그램] 발송 실패: {e}")
    else:
        print("[미리보기]")
        print(msg)
        print(f"\n[길이: {len(msg)}자]")


if __name__ == "__main__":
    main()
