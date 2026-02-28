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
    lines.append("━━━━ 💼 보유종목 현황 ━━━━")
    for rec in monitored:
        action = rec.get("monitor_action", "HOLD")
        emoji = ACTION_EMOJI.get(action, "\u26aa")
        label = ACTION_LABEL.get(action, action)
        name = rec.get("name", rec.get("ticker", "?"))
        target = rec.get("monitor_target", 0)
        pnl = rec.get("pnl_pct", 0)
        pnl_bar = "▲" if pnl > 0 else ("▼" if pnl < 0 else "─")
        lines.append(
            f"  {emoji} {name} {pnl_bar}{pnl:+.1f}% → {label}"
            + (f" (🎯{target:,.0f})" if target else "")
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

    lines = ["\n━━━━ 📢 DART 공시 ━━━━"]

    for d in tier1[:3]:
        corp = d.get("corp_name", "?")
        kw = d.get("keyword", "")
        lines.append(f"  🚨 {corp} — {kw}")

    for d in universe[:3]:
        corp = d.get("corp_name", "?")
        kw = d.get("keyword", "")
        lines.append(f"  📌 {corp} — {kw}")

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
    lines = [f"\n━━━━ 🏆 내일 추천 TOP{len(top_tickers)} {target_label} ━━━━"]

    MEDAL = {1: "🥇", 2: "🥈", 3: "🥉"}
    GRADE_EMOJI = {"적극매수": "🔥", "매수": "👍", "관심": "👀"}

    def _fmt(p_list, group_label, icon):
        if not p_list:
            return
        lines.append(f"\n  {icon} {group_label}")
        for i, p in enumerate(p_list, 1):
            name = p.get("name", "?")
            srcs = "+".join(p.get("sources", []))
            score = p.get("total_score", p.get("score", 0))
            grade = p.get("grade", "")
            grade_e = GRADE_EMOJI.get(grade, "")
            sar_icon = " ↗️" if p.get("sar_trend") == 1 else (" ↘️" if p.get("sar_trend") == -1 else "")
            medal = MEDAL.get(i, f"  {i}.")
            lines.append(f"  {medal} {name} {grade_e}{grade}")
            lines.append(f"       📊 {score:.0f}점 | {srcs}{sar_icon}")

    _fmt(swing_picks, "📈 스윙(3~7일)", "🔵")
    _fmt(short_picks, "⚡ 단타(1~3일)", "🟡")
    if others:
        _fmt(others, "📌 기타", "⚪")

    return lines


def _section_ai_largecap() -> list[str]:
    """AI 대형주 참고 — Bot 미검출, AI BUY만."""
    data = _load("tomorrow_picks.json")
    if not data:
        return []
    largecap = data.get("ai_largecap", [])
    if not largecap:
        return []
    lines = ["\n━━━━ 🧠 AI 대형주 참고 ━━━━"]
    for r in largecap[:5]:
        conf = r.get("confidence", 0)
        urg = " 🔥" if r.get("urgency") == "high" else ""
        bar = "🟩" * int(conf * 5) + "⬜" * (5 - int(conf * 5))
        lines.append(f"  💎 {r['name']} {bar} {conf:.0%}{urg}")
    return lines


def _section_value_chain() -> list[str]:
    """밸류체인 발화 요약 (섹터+대장주만, 1줄씩)."""
    data = _load("value_chain_relay.json")
    if not data:
        return []

    fired = data.get("fired_sectors", [])
    if not fired:
        return []

    lines = ["\n━━━━ 🔗 밸류체인 발화 ━━━━"]
    for sec in fired[:5]:
        sector = sec.get("sector", "?")
        leaders = [l.get("name", "?") for l in sec.get("leaders", [])]
        candidates = [c.get("name", "?") for c in sec.get("candidates", [])[:2]]
        leader_str = "+".join(leaders)
        cand_str = ", ".join(candidates) if candidates else "대기 없음"
        lines.append(f"  🏭 {sector}")
        lines.append(f"    🔺 {leader_str} → 🎯 {cand_str}")

    return lines


def _section_ai_vs_bot() -> list[str]:
    """AI 두뇌 판단 vs Bot(기술적) 판단 비교."""
    ai_data = _load("ai_brain_judgment.json")
    picks_data = _load("tomorrow_picks.json")
    if not ai_data or not ai_data.get("stock_judgments"):
        return []

    lines = ["\n━━━━ 🧠 AI 두뇌 분석 ━━━━"]
    sentiment = ai_data.get("market_sentiment", "")
    s_map = {"bullish": ("📈 강세", "🟢"), "bearish": ("📉 약세", "🔴"), "neutral": ("➡️ 중립", "🟡")}
    s_label, s_dot = s_map.get(sentiment, ("❓", "⚪"))
    lines.append(f"  {s_dot} 센티먼트: {s_label}")
    themes = ai_data.get("key_themes", [])
    if themes:
        for t in themes[:3]:
            lines.append(f"  💡 {t[:40]}")

    # Bot TOP 종목 티커 세트
    bot_tickers = set()
    if picks_data:
        for p in picks_data.get("picks", []):
            if isinstance(p, dict):
                bot_tickers.add(p.get("ticker", ""))

    ai_buys = [j for j in ai_data["stock_judgments"] if j.get("action") == "BUY"]
    ai_avoids = [j for j in ai_data["stock_judgments"] if j.get("action") == "AVOID"]

    both = [j for j in ai_buys if j.get("ticker") in bot_tickers]
    ai_only = [j for j in ai_buys if j.get("ticker") not in bot_tickers]

    if both:
        lines.append("\n  ✅ AI+Bot 동시 추천")
        for j in both[:3]:
            lines.append(f"    🟢 {j.get('name', '?')} ({j.get('confidence', 0):.0%})")

    if ai_only:
        lines.append("  🧠 AI만 포착")
        for j in ai_only[:3]:
            reason = j.get("reasoning", "")[:30]
            lines.append(f"    🟡 {j.get('name', '?')} — {reason}")

    if ai_avoids:
        lines.append("  ⛔ AI 경고")
        for j in ai_avoids[:2]:
            lines.append(f"    🔴 {j.get('name', '?')} — {j.get('reasoning', '')[:30]}")

    return lines


def _section_ai_accuracy() -> list[str]:
    """[채널 4] 지난 2주간 AI BUY 적중률."""
    hist_path = DATA_DIR / "ai_brain_history.json"
    if not hist_path.exists():
        return []
    try:
        history = json.loads(hist_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    # tracking_complete인 BUY 판단만 집계
    completed = []
    for entry in history:
        for j in entry.get("stock_judgments", []):
            if j.get("action") != "BUY" or not j.get("tracking_complete"):
                continue
            completed.append(j)

    if len(completed) < 3:
        return []  # 최소 3건 이상

    wins = sum(1 for j in completed if (j.get("ret_d5") or 0) > 0)
    total = len(completed)
    avg_ret = sum(j.get("ret_d5", 0) or 0 for j in completed) / total

    lines = ["\n━━━━ 📊 AI 적중률 ━━━━"]
    hit_pct = wins / total
    bar = "🟩" * int(hit_pct * 10) + "⬜" * (10 - int(hit_pct * 10))
    lines.append(f"  D+5 기준: {bar} {wins}/{total} ({hit_pct:.0%})")
    lines.append(f"  📈 평균수익률: {avg_ret:+.1f}%")

    # 최근 3건 상세
    recent = completed[-3:]
    for j in recent:
        ret = j.get("ret_d5", 0) or 0
        icon = "✅" if ret > 0 else "❌"
        lines.append(f"    {icon} {j.get('name', '?')} {ret:+.1f}%")

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

    lines = ["\n━━━━ 🌍 시장 무드 ━━━━"]
    parts = []
    if mood:
        parts.append(f"  🌐 {mood}")
    if forecast:
        fc_map = {"상승": "📈 상승", "하락": "📉 하락", "보합": "➡️ 보합"}
        parts.append(f"  🇰🇷 KR {fc_map.get(forecast, forecast)}")
    if parts:
        lines.extend(parts)
    if themes:
        lines.append(f"  🔥 " + " | ".join(themes[:4]))

    return lines


# ───────────────────────────────────────
# 메인 빌더
# ───────────────────────────────────────

def build_evening_summary() -> str:
    """저녁 통합 리포트 1건."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [
        "╔══════════════════════╗",
        f"  🤖 Quantum Master v10.3",
        f"  📅 {now} 장마감 리포트",
        "╚══════════════════════╝",
    ]

    # 각 섹션을 우선순위 순으로 추가
    L.extend(_section_holdings())
    L.extend(_section_dart())
    L.extend(_section_picks())
    L.extend(_section_ai_largecap())
    L.extend(_section_ai_vs_bot())
    L.extend(_section_ai_accuracy())
    L.extend(_section_value_chain())
    L.extend(_section_intel())

    # 빈 내용 체크
    if len(L) <= 4:
        L.append("\n⚠️ 오늘 발송할 내용이 없습니다.")

    # 푸터
    L.append("\n┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈")
    L.append(f"⏰ {now} | Quantum Master 🤖")

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
