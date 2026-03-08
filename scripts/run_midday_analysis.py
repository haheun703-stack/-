"""
11:30 장중 AI 분석 — 매수 후보 일괄 TradeAdvisor 분석

BAT 스케줄: 11:30 실행
결과: 텔레그램으로 5종목 AI 판단 + [매수 실행] 버튼 전송

사용법:
  python scripts/run_midday_analysis.py              # 기본 (설정된 종목)
  python scripts/run_midday_analysis.py --dry         # AI 분석만 (텔레그램 미전송)
  python scripts/run_midday_analysis.py --targets 064350:5 028050:60  # 직접 지정
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("midday_analysis")


# ─── 기본 매수 후보 (수동 설정) ───
DEFAULT_TARGETS = [
    {"ticker": "034020", "name": "두산에너빌리티", "qty": 20, "desc": "원전/SMR | AI Brain BUY 85%"},
    {"ticker": "000500", "name": "가온전선", "qty": 20, "desc": "원전 relay | AI v3 conv8"},
    {"ticker": "011690", "name": "와이투솔루션", "qty": 200, "desc": "전력인프라 | AI v3 conv8"},
    {"ticker": "028050", "name": "삼성E&A", "qty": 60, "desc": "건설 Tier2 | 눌림목"},
    {"ticker": "064350", "name": "현대로템", "qty": 5, "desc": "방산 Tier1 | 추가매수"},
]


def _tick_round(price: int) -> int:
    """KRX 호가 단위로 내림 반올림."""
    if price >= 500_000:
        return (price // 1000) * 1000
    elif price >= 100_000:
        return (price // 500) * 500
    elif price >= 50_000:
        return (price // 100) * 100
    elif price >= 10_000:
        return (price // 50) * 50
    elif price >= 5_000:
        return (price // 10) * 10
    elif price >= 1_000:
        return (price // 5) * 5
    return price


async def run_analysis(targets: list[dict], dry_run: bool = False) -> list[dict]:
    """TradeAdvisor로 각 종목 분석 실행."""
    from src.agents.trade_advisor import TradeAdvisor
    from src.adapters.kis_order_adapter import KisOrderAdapter

    advisor = TradeAdvisor()
    adapter = KisOrderAdapter()

    results = []
    for t in targets:
        ticker = t["ticker"]
        name = t["name"]
        qty = t["qty"]
        desc = t.get("desc", "")

        logger.info("[분석] %s(%s) %d주 시작...", name, ticker, qty)

        # 현재가 조회
        price_info = adapter.fetch_current_price(ticker)
        current_price = price_info.get("current_price", 0)

        if current_price <= 0:
            logger.warning("[분석] %s 현재가 조회 실패 → 스킵", name)
            results.append({
                "ticker": ticker, "name": name, "qty": qty,
                "desc": desc, "current_price": 0,
                "result": {"verdict": "ERROR", "error": "현재가 조회 실패"},
            })
            continue

        # TradeAdvisor AI 분석
        try:
            result = await advisor.analyze_buy(ticker, qty, current_price)
        except Exception as e:
            logger.error("[분석] %s AI 분석 실패: %s", name, e)
            result = {"verdict": "ERROR", "error": str(e)}

        # 지정가 계산 (현재가 -0.5%)
        limit_price = _tick_round(int(current_price * 0.995))
        invest_amount = limit_price * qty

        results.append({
            "ticker": ticker, "name": name, "qty": qty,
            "desc": desc, "current_price": current_price,
            "limit_price": limit_price, "invest_amount": invest_amount,
            "change_pct": price_info.get("change_pct", 0),
            "result": result,
        })

        logger.info("[분석] %s → %s (신뢰도 %s%%)",
                     name, result.get("verdict", "?"), result.get("confidence", "?"))

    return results


def format_telegram_message(results: list[dict]) -> str:
    """분석 결과를 텔레그램 메시지로 포맷."""
    now = datetime.now().strftime("%H:%M")

    lines = [
        f"🔍 [11:30 장중 AI 분석] {now}",
        "━━━━━━━━━━━━━━━━━━━━━━━",
        "",
    ]

    total_invest = 0
    buy_count = 0

    for r in results:
        ticker = r["ticker"]
        name = r["name"]
        qty = r["qty"]
        desc = r.get("desc", "")
        current = r.get("current_price", 0)
        change = r.get("change_pct", 0)
        limit_price = r.get("limit_price", 0)
        invest = r.get("invest_amount", 0)
        ai = r.get("result", {})

        verdict = ai.get("verdict", "ERROR")
        confidence = ai.get("confidence", 0)

        # 아이콘
        if verdict == "BUY_OK":
            icon = "✅"
            buy_count += 1
        elif verdict == "WAIT":
            icon = "⏳"
        elif verdict == "SKIP":
            icon = "❌"
        else:
            icon = "⚠️"

        lines.append(f"{icon} {name}({ticker}) — {desc}")
        lines.append(f"   현재가: {current:,}원 ({change:+.1f}%)")

        # AI 결과 요약
        tech = ai.get("technical_summary", "")
        catalyst = ai.get("catalyst", "")
        risk = ai.get("risk_warning", "")
        suggestion = ai.get("suggestion", "")

        if tech:
            lines.append(f"   📊 기술: {tech}")
        if catalyst:
            lines.append(f"   📰 촉매: {catalyst}")
        if risk:
            lines.append(f"   ⚠️ 주의: {risk}")

        lines.append(f"   💡 AI: {verdict} (신뢰도 {confidence}%)")
        if suggestion:
            lines.append(f"      → {suggestion}")

        lines.append(f"   📋 주문: {qty}주 × {limit_price:,}원 = {invest:,}원")
        lines.append(f"   명령: 매수 {name} {qty}")
        lines.append("")

        total_invest += invest

    # 합계
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"💰 투입 예정: {total_invest:,}원")
    lines.append(f"✅ BUY_OK: {buy_count}개 / ⏳ WAIT+❌ SKIP: {len(results)-buy_count}개")
    lines.append("")
    lines.append("📌 매수하실 종목 텔레그램에서 명령해 주세요")
    lines.append("   예: 매수 두산에너빌리티 20")

    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    """텔레그램 전송."""
    from src.telegram_sender import send_message
    return send_message(message)


def save_analysis(results: list[dict]):
    """분석 결과 JSON 저장."""
    output = {
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "targets": results,
    }
    path = Path("data/midday_analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("[저장] %s", path)


def main():
    parser = argparse.ArgumentParser(description="11:30 장중 AI 분석")
    parser.add_argument("--dry", action="store_true",
                        help="분석만 (텔레그램 미전송)")
    parser.add_argument("--targets", nargs="*",
                        help="종목 직접 지정 (예: 064350:5 028050:60)")
    args = parser.parse_args()

    # 대상 종목 설정
    if args.targets:
        from src.stock_name_resolver import ticker_to_name
        targets = []
        for t in args.targets:
            parts = t.split(":")
            ticker = parts[0]
            qty = int(parts[1]) if len(parts) > 1 else 10
            name = ticker_to_name(ticker) or ticker
            targets.append({"ticker": ticker, "name": name, "qty": qty})
    else:
        targets = DEFAULT_TARGETS

    logger.info("=" * 50)
    logger.info("[장중 AI 분석] %d종목 분석 시작", len(targets))
    logger.info("=" * 50)

    # 비동기 분석 실행
    results = asyncio.run(run_analysis(targets, dry_run=args.dry))

    # 결과 저장
    save_analysis(results)

    # 텔레그램 전송
    msg = format_telegram_message(results)
    print(msg)

    if not args.dry:
        ok = send_telegram(msg)
        logger.info("[텔레그램] 전송 %s", "성공" if ok else "실패")
    else:
        logger.info("[DRY] 텔레그램 미전송")

    # 요약
    verdicts = [r["result"].get("verdict", "ERROR") for r in results]
    logger.info("[결과] BUY_OK=%d, WAIT=%d, SKIP=%d, ERROR=%d",
                verdicts.count("BUY_OK"), verdicts.count("WAIT"),
                verdicts.count("SKIP"), verdicts.count("ERROR"))


if __name__ == "__main__":
    main()
