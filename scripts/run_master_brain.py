"""Master Brain 실행 — BAT-D 20.9단계

전체 시그널 데이터를 종합하여 추론 체인 기반 통합 판단을 생성하고
텔레그램으로 전송합니다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# PYTHONPATH 안전장치
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_telegram_message(result: dict) -> str:
    """추론 체인 + 통합 추천을 텔레그램 메시지로 포맷."""
    lines = []

    # 헤더
    regime = result.get("market_regime", {})
    lines.append(f"[Master Brain] {result.get('date', 'N/A')}")
    lines.append(f"레짐: {regime.get('overall', '?')} | US: {regime.get('us_overnight', '?')}")
    stag = regime.get("stagflation", "NONE")
    if stag != "NONE":
        lines.append(f"스태그플레이션: {stag}")
    lines.append(f"{regime.get('summary', '')}")
    lines.append("")

    # 추론 체인
    chains = result.get("reasoning_chains", [])
    if chains:
        lines.append("[ 추론 체인 ]")
        for chain in chains[:5]:
            conf = chain.get("confidence", 0)
            tf = chain.get("timeframe", "?")
            lines.append(f"  {chain.get('cause', '?')}")
            lines.append(f"  -> {chain.get('sector_impact', '?')}")
            lines.append(f"  -> {chain.get('action', '?')} ({conf:.0%}, {tf})")
            lines.append("")

    # 통합 추천
    picks = result.get("unified_picks", {})
    category_labels = {
        "individual_stocks": "개별종목",
        "sector_etf": "섹터 ETF",
        "leverage_inverse": "레버리지/인버스",
        "commodity_etf": "원자재 ETF",
        "index_etf": "인덱스 ETF",
    }
    for key, label in category_labels.items():
        items = picks.get(key, [])
        if items:
            lines.append(f"[ {label} ]")
            for p in items[:4]:
                action = p.get("action", "?")
                name = p.get("name", p.get("ticker", "?"))
                reason = p.get("reason", "")[:60]
                lines.append(f"  {action} {name} — {reason}")
            lines.append("")

    # 현금 전략
    cash = result.get("cash_strategy", {})
    if cash:
        target = cash.get("target_cash_pct", 25)
        action = cash.get("action", "")
        lines.append(f"[ 현금 ] 목표 {target}% | {action}")

    # 리스크
    alerts = result.get("risk_alerts", [])
    if alerts:
        lines.append("")
        for alert in alerts[:3]:
            lines.append(f"  ! {alert}")

    return "\n".join(lines)


async def main():
    from dotenv import load_dotenv
    load_dotenv()

    from src.agents.master_brain import MasterBrainAgent

    agent = MasterBrainAgent()
    result = await agent.think()

    # 텔레그램 전송
    msg = format_telegram_message(result)
    try:
        from src.telegram_sender import send_message
        send_message(msg)
        logger.info("Master Brain 텔레그램 전송 완료")
    except Exception as e:
        logger.warning(f"텔레그램 전송 실패: {e}")
        print(msg)

    # 요약 출력
    regime = result.get("market_regime", {})
    chains_count = len(result.get("reasoning_chains", []))
    picks = result.get("unified_picks", {})
    total_picks = sum(len(v) for v in picks.values() if isinstance(v, list))
    logger.info(
        f"Master Brain 완료: 레짐={regime.get('overall', '?')}, "
        f"체인={chains_count}개, 추천={total_picks}건"
    )


if __name__ == "__main__":
    asyncio.run(main())
