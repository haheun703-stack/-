"""보유종목 동적 목표가 재판정 CLI.

사용법:
    python scripts/position_monitor.py                  # 보유종목 재판정
    python scripts/position_monitor.py --send           # + 텔레그램 발송
    python scripts/position_monitor.py --ticker 005930  # 특정 종목만
    python scripts/position_monitor.py --dry-run        # 예시 데이터 테스트
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ── PYTHONPATH 안전장치 (BAT에서 누락 방지) ──────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(Path(PROJECT_ROOT) / ".env")

from src.entities.position_models import MonitorAction, MonitorResult, PositionTarget
from src.use_cases.position_revaluation import PositionRevaluationEngine
from src.telegram_sender import send_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("position_monitor")

DATA_DIR = Path(PROJECT_ROOT) / "data"

# 텔레그램 이모지 매핑
ACTION_EMOJI = {
    MonitorAction.ADD: "\U0001f535",           # 🔵
    MonitorAction.HOLD: "\U0001f7e2",          # 🟢
    MonitorAction.PARTIAL_SELL: "\U0001f7e1",  # 🟡
    MonitorAction.FULL_SELL: "\U0001f534",      # 🔴
}

ACTION_LABEL_KR = {
    MonitorAction.ADD: "추가매수",
    MonitorAction.HOLD: "보유유지",
    MonitorAction.PARTIAL_SELL: "부분매도",
    MonitorAction.FULL_SELL: "전량매도",
}

# ──────────────────────────────────────────
# dry-run 예시 데이터
# ──────────────────────────────────────────

DRY_RUN_HOLDINGS = [
    {
        "ticker": "005930",
        "name": "삼성전자",
        "quantity": 50,
        "avg_price": 58000,
        "current_price": 56500,
        "pnl_pct": -2.59,
    },
    {
        "ticker": "000660",
        "name": "SK하이닉스",
        "quantity": 10,
        "avg_price": 195000,
        "current_price": 210000,
        "pnl_pct": 7.69,
    },
    {
        "ticker": "035720",
        "name": "카카오",
        "quantity": 30,
        "avg_price": 48000,
        "current_price": 42000,
        "pnl_pct": -12.50,
    },
]


# ──────────────────────────────────────────
# 메인 로직
# ──────────────────────────────────────────

def fetch_real_holdings(ticker_filter: str | None = None) -> list[dict]:
    """KIS API에서 실제 보유종목 조회."""
    from src.adapters.kis_order_adapter import KisOrderAdapter

    adapter = KisOrderAdapter()
    balance = adapter.fetch_balance()
    holdings = balance.get("holdings", [])

    if ticker_filter:
        holdings = [h for h in holdings if h["ticker"] == ticker_filter]

    logger.info("보유종목 %d건 조회", len(holdings))
    return holdings


def format_console_output(result: MonitorResult) -> str:
    """콘솔 출력용 포맷."""
    lines = [
        f"\n{'='*60}",
        f"  보유종목 동적 목표가 재판정 ({result.date})",
        f"  생성: {result.generated_at}  |  총 {result.total_holdings}종목",
        f"{'='*60}",
    ]

    # 액션 요약
    summary_parts = []
    for action in MonitorAction:
        cnt = result.actions_summary.get(action.value, 0)
        if cnt > 0:
            emoji = ACTION_EMOJI[action]
            label = ACTION_LABEL_KR[action]
            summary_parts.append(f"{emoji}{label}: {cnt}건")
    if summary_parts:
        lines.append("  " + "  |  ".join(summary_parts))
    lines.append("")

    # 종목별 상세
    for pt in result.positions:
        emoji = ACTION_EMOJI[pt.action]
        label = ACTION_LABEL_KR[pt.action]
        adj = pt.adjustment

        lines.append(f"  {emoji} [{pt.ticker}] {pt.name}  →  {label}")
        lines.append(
            f"     현재가: {pt.current_price:,.0f}  |  "
            f"평단가: {pt.avg_price:,.0f}  |  "
            f"수익률: {pt.pnl_pct:+.1f}%"
        )
        lines.append(
            f"     기준목표: {pt.base_target:,.0f}  →  "
            f"최종목표: {pt.final_target:,.0f}  "
            f"(비율: {pt.ratio_to_current:.2f})"
        )

        # 7축 조정 내역 (0이 아닌 것만)
        adj_parts = []
        if adj.report_adj:
            adj_parts.append(f"리포트{adj.report_adj:+.1%}")
        if adj.news_adj:
            adj_parts.append(f"뉴스{adj.news_adj:+.1%}")
        if adj.supply_adj:
            adj_parts.append(f"수급{adj.supply_adj:+.1%}")
        if adj.macd_adj:
            adj_parts.append(f"MACD{adj.macd_adj:+.1%}")
        if adj.rsi_adj:
            adj_parts.append(f"RSI{adj.rsi_adj:+.1%}")
        if adj.bb_adj:
            adj_parts.append(f"BB{adj.bb_adj:+.1%}")
        if adj.dart_adj:
            adj_parts.append(f"DART{adj.dart_adj:+.1%}")
        if adj_parts:
            lines.append(f"     조정: {' | '.join(adj_parts)}  (합계: {adj.total:+.1%})")

        # 판단 근거
        if pt.reasons:
            lines.append(f"     근거: {', '.join(pt.reasons[:4])}")
        lines.append(f"     신뢰도: {pt.confidence:.0%}")
        lines.append("")

    if result.errors:
        lines.append(f"  [오류] {len(result.errors)}건: {', '.join(result.errors[:3])}")

    lines.append(f"{'='*60}")
    return "\n".join(lines)


def format_telegram_message(result: MonitorResult) -> str:
    """텔레그램 발송용 포맷."""
    lines = [
        f"\U0001f4ca 보유종목 재판정 ({result.date})",
        f"총 {result.total_holdings}종목 | 처리 {result.processed}건",
        "",
    ]

    # 액션 요약 한 줄
    summary_parts = []
    for action in MonitorAction:
        cnt = result.actions_summary.get(action.value, 0)
        if cnt > 0:
            emoji = ACTION_EMOJI[action]
            summary_parts.append(f"{emoji}{cnt}")
    lines.append(" ".join(summary_parts))
    lines.append("")

    # 종목별 (간결)
    for pt in result.positions:
        emoji = ACTION_EMOJI[pt.action]
        label = ACTION_LABEL_KR[pt.action]
        adj = pt.adjustment

        lines.append(f"{emoji} {pt.name}({pt.ticker}) {label}")
        lines.append(
            f"  {pt.current_price:,.0f}원 ({pt.pnl_pct:+.1f}%) "
            f"→ 목표 {pt.final_target:,.0f}"
        )

        # 비제로 조정만 간결하게
        adj_parts = []
        if adj.report_adj:
            adj_parts.append(f"리포트{adj.report_adj:+.0%}")
        if adj.news_adj:
            adj_parts.append(f"뉴스{adj.news_adj:+.0%}")
        if adj.supply_adj:
            adj_parts.append(f"수급{adj.supply_adj:+.0%}")
        if adj.macd_adj:
            adj_parts.append(f"MACD{adj.macd_adj:+.0%}")
        if adj.rsi_adj:
            adj_parts.append(f"RSI{adj.rsi_adj:+.0%}")
        if adj.bb_adj:
            adj_parts.append(f"BB{adj.bb_adj:+.0%}")
        if adj.dart_adj:
            adj_parts.append(f"DART{adj.dart_adj:+.0%}")
        if adj_parts:
            lines.append(f"  [{' '.join(adj_parts)}]")

        if pt.reasons:
            lines.append(f"  {', '.join(pt.reasons[:3])}")
        lines.append("")

    return "\n".join(lines)


def update_picks_history(result: MonitorResult) -> None:
    """picks_history.json에 monitor 판정 결과 반영."""
    path = DATA_DIR / "picks_history.json"
    if not path.exists():
        logger.warning("picks_history.json 없음 — 업데이트 스킵")
        return

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("picks_history.json 읽기 실패: %s", e)
        return

    # ticker → PositionTarget 매핑
    pt_map = {pt.ticker: pt for pt in result.positions}

    updated = 0
    for rec in data.get("records", []):
        if rec.get("status") != "holding":
            continue
        ticker = rec.get("ticker", "")
        pt = pt_map.get(ticker)
        if pt is None:
            continue

        rec["monitor_action"] = pt.action.value
        rec["monitor_target"] = pt.final_target
        rec["monitor_reasons"] = " | ".join(pt.reasons[:3])
        rec["monitor_date"] = result.date
        rec["monitor_confidence"] = pt.confidence
        updated += 1

    if updated > 0:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("picks_history.json 업데이트: %d건", updated)


def main():
    parser = argparse.ArgumentParser(description="보유종목 동적 목표가 재판정")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송")
    parser.add_argument("--ticker", type=str, help="특정 종목만 (e.g. 005930)")
    parser.add_argument("--dry-run", action="store_true", help="예시 데이터로 테스트")
    args = parser.parse_args()

    logger.info("=" * 40)
    logger.info("보유종목 동적 목표가 재판정 시작")
    logger.info("모드: %s", "DRY-RUN" if args.dry_run else "REAL")

    # 1) 보유종목 조회
    if args.dry_run:
        holdings = DRY_RUN_HOLDINGS
        if args.ticker:
            holdings = [h for h in holdings if h["ticker"] == args.ticker]
        logger.info("[DRY-RUN] 예시 보유종목 %d건", len(holdings))
    else:
        holdings = fetch_real_holdings(args.ticker)

    if not holdings:
        logger.warning("보유종목 없음 — 종료")
        return

    # 2) 엔진 실행
    engine = PositionRevaluationEngine()
    result = engine.run(holdings)

    # 3) 콘솔 출력
    console_text = format_console_output(result)
    print(console_text)

    # 4) picks_history 업데이트
    if not args.dry_run:
        update_picks_history(result)

    # 5) 텔레그램 발송
    if args.send:
        msg = format_telegram_message(result)
        ok = send_message(msg)
        if ok:
            logger.info("텔레그램 발송 완료")
        else:
            logger.warning("텔레그램 발송 실패")

    logger.info("재판정 완료: %d건 처리, 오류 %d건", result.processed, len(result.errors))


if __name__ == "__main__":
    main()
