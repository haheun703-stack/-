"""
장마감 포트폴리오 방향성 판단 — 보유 종목 "내일 어디로?"

BAT 스케줄: 17:00 실행 (장마감 후 데이터 안정화 시점)
결과: 텔레그램으로 전 종목 방향성 + 행동 추천

사용법:
  python scripts/run_portfolio_outlook.py           # 라이브 (텔레그램 전송)
  python scripts/run_portfolio_outlook.py --dry      # 분석만 (텔레그램 미전송)
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
logger = logging.getLogger("portfolio_outlook")


def format_telegram(results: list[dict]) -> str:
    """분석 결과를 텔레그램 메시지로 포맷."""
    now = datetime.now().strftime("%m/%d %H:%M")

    lines = [
        f"🔮 [장마감 방향성 판단] {now}",
        "━━━━━━━━━━━━━━━━━━━━━",
        "",
    ]

    # 방향별 분류
    up_stocks = [r for r in results if r.get("direction") == "↑"]
    side_stocks = [r for r in results if r.get("direction") == "→"]
    down_stocks = [r for r in results if r.get("direction") == "↓"]

    # 요약 헤더
    lines.append(
        f"📊 ↑{len(up_stocks)} / →{len(side_stocks)} / ↓{len(down_stocks)}"
    )
    lines.append("")

    # 개별 종목
    for r in results:
        ticker = r.get("ticker", "")
        name = r.get("name", ticker)
        direction = r.get("direction", "→")
        action = r.get("action", "HOLD")
        confidence = r.get("confidence", 0)
        reason = r.get("reason", "")
        pnl = r.get("pnl_pct", 0)
        catalyst = r.get("catalyst_status", "UNKNOWN")
        risk = r.get("key_risk", "")
        target = r.get("target_note", "")

        # 방향 아이콘
        if direction == "↑":
            dir_icon = "📈"
        elif direction == "↓":
            dir_icon = "📉"
        else:
            dir_icon = "📊"

        # 행동 아이콘
        action_map = {
            "HOLD": "🟢 HOLD",
            "ADD": "🔵 추가매수",
            "TRIM": "🟡 일부매도",
            "SELL": "🔴 전량매도",
        }
        action_str = action_map.get(action, f"⚪ {action}")

        # 촉매 아이콘
        cat_map = {
            "ALIVE": "🔥",
            "FADING": "💨",
            "DEAD": "💀",
            "UNKNOWN": "❓",
        }
        cat_icon = cat_map.get(catalyst, "❓")

        lines.append(f"{dir_icon} {name} {direction} ({confidence}%)")
        lines.append(f"   {action_str} | 수익률 {pnl:+.1f}%")
        lines.append(f"   촉매: {cat_icon} {catalyst}")
        lines.append(f"   📌 {reason}")
        if risk:
            lines.append(f"   ⚠️ {risk}")
        if target:
            lines.append(f"   🎯 {target}")
        lines.append("")

    # 하단 요약
    lines.append("━━━━━━━━━━━━━━━━━━━━━")

    # 행동별 정리
    sells = [r["name"] for r in results if r.get("action") == "SELL"]
    trims = [r["name"] for r in results if r.get("action") == "TRIM"]
    adds = [r["name"] for r in results if r.get("action") == "ADD"]

    if sells:
        lines.append(f"🔴 매도 추천: {', '.join(sells)}")
    if trims:
        lines.append(f"🟡 일부매도: {', '.join(trims)}")
    if adds:
        lines.append(f"🔵 추가매수: {', '.join(adds)}")
    if not sells and not trims and not adds:
        lines.append("🟢 전 종목 홀딩 유지")

    lines.append("")
    lines.append("💡 최종 판단은 항상 사용자 확인 후 실행")

    return "\n".join(lines)


def save_results(results: list[dict]):
    """결과 JSON 저장."""
    output = {
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "holdings_count": len(results),
        "summary": {
            "up": len([r for r in results if r.get("direction") == "↑"]),
            "side": len([r for r in results if r.get("direction") == "→"]),
            "down": len([r for r in results if r.get("direction") == "↓"]),
        },
        "results": results,
    }
    path = Path("data/portfolio_outlook.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    logger.info("[저장] %s", path)


def main():
    parser = argparse.ArgumentParser(description="장마감 포트폴리오 방향성 판단")
    parser.add_argument("--dry", action="store_true",
                        help="분석만 (텔레그램 미전송)")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("[Portfolio Outlook] 방향성 분석 시작")
    logger.info("=" * 50)

    from src.agents.portfolio_outlook import PortfolioOutlook

    outlook = PortfolioOutlook()
    results = asyncio.run(outlook.analyze_all())

    if not results:
        logger.warning("[Outlook] 보유 종목 없음 — 종료")
        return

    # 저장
    save_results(results)

    # 텔레그램 포맷
    msg = format_telegram(results)
    print(msg)

    # 전송
    if not args.dry:
        from src.telegram_sender import send_message
        ok = send_message(msg)
        logger.info("[텔레그램] 전송 %s", "성공" if ok else "실패")
    else:
        logger.info("[DRY] 텔레그램 미전송")

    # 요약
    actions = [r.get("action", "HOLD") for r in results]
    logger.info("[결과] HOLD=%d, ADD=%d, TRIM=%d, SELL=%d",
                actions.count("HOLD"), actions.count("ADD"),
                actions.count("TRIM"), actions.count("SELL"))


if __name__ == "__main__":
    main()
