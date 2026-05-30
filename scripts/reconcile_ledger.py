"""KIS 원장 reconcile (차단선 B⑨) — KIS 실잔고 vs data/positions.json 대조.

실매수 안전망: 내부 상태(positions.json)가 아니라 증권사 원장(KIS 잔고)을 정본으로
대조해 불일치(수량/고아/미등록)를 감지하고 텔레그램으로 알린다.

  - 기본(dry-run): 대조만, positions.json 미변경 (save 0).
  - --apply: position_tracker.sync_with_broker로 보정 + save.

★ 조회+대조만 수행 — 매매(주문) 호출 0건. KILL_SWITCH/거래시간/거래일 무관 안전.
장 마감 후 cron 권장. 첫 1~2주는 dry-run으로 "무엇이 보정될 뻔했나"만 관찰.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("reconcile_ledger")


def _compute_diff(positions, broker_map):
    """sync_with_broker를 호출하지 않고 대조만 수행 (dry-run 안전).

    Returns: (qty_mismatch, orphaned, unregistered)
      - qty_mismatch: [(ticker, name, tracker_qty, broker_qty)]
      - orphaned:     [(ticker, name, tracker_qty)]  트래커만 존재 (실잔고 없음)
      - unregistered: [(ticker, name, broker_qty)]   실잔고만 존재 (트래커 미등록)
    """
    qty_mismatch, orphaned = [], []
    tracked = {p.ticker for p in positions}
    for pos in positions:
        if pos.ticker not in broker_map:
            orphaned.append((pos.ticker, pos.name, pos.shares))
        else:
            bq = broker_map[pos.ticker].get("quantity")
            if bq != pos.shares:
                qty_mismatch.append((pos.ticker, pos.name, pos.shares, bq))
    unregistered = [
        (tk, h.get("name", ""), h.get("quantity", 0))
        for tk, h in broker_map.items()
        if tk not in tracked
    ]
    return qty_mismatch, orphaned, unregistered


def _format_report(qty_mismatch, orphaned, unregistered, applied: bool) -> str:
    tag = "[적용]" if applied else "[감지]"
    lines = [f"📒 원장 reconcile {tag} (KIS 실잔고 vs 트래커)"]
    if not (qty_mismatch or orphaned or unregistered):
        lines.append("✅ 일치 — 불일치 0건")
        return "\n".join(lines)
    if qty_mismatch:
        lines.append(f"⚠️ 수량 불일치 {len(qty_mismatch)}건:")
        for tk, nm, t, b in qty_mismatch:
            lines.append(f"  {nm}({tk}): 트래커 {t} → 실잔고 {b}")
    if orphaned:
        lines.append(f"🗑️ 고아(트래커만/실잔고 없음) {len(orphaned)}건:")
        for tk, nm, t in orphaned:
            lines.append(f"  {nm}({tk}): 트래커 {t}주")
    if unregistered:
        lines.append(f"➕ 미등록(실잔고만/트래커 없음) {len(unregistered)}건:")
        for tk, nm, q in unregistered:
            lines.append(f"  {nm}({tk}): {q}주")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="KIS 원장 reconcile (B⑨)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="대조만 (기본 동작 — 명시적 표기용)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="sync_with_broker로 실제 보정 + positions.json save",
    )
    parser.add_argument(
        "--no-telegram", action="store_true", help="텔레그램 발송 생략",
    )
    args = parser.parse_args()

    from src.use_cases.position_tracker import POSITIONS_FILE, PositionTracker

    # 1. KIS 실잔고 조회 (네트워크 — 조회 전용, 주문 0)
    try:
        from src.adapters.kis_order_adapter import KisOrderAdapter
        adapter = KisOrderAdapter()
        holdings = adapter.fetch_holdings()
    except Exception as e:
        logger.error("[reconcile] KIS 잔고 조회 실패: %s", e)
        return 1
    broker_map = {h["ticker"]: h for h in holdings}
    logger.info("[reconcile] KIS 보유종목 %d개", len(broker_map))

    # 2. 트래커 로드 (positions.json)
    tracker = PositionTracker(config={})
    logger.info("[reconcile] 트래커 포지션 %d개", len(tracker.positions))

    # 3. 대조 (sync 미호출 — dry-run 안전)
    qty_mismatch, orphaned, unregistered = _compute_diff(tracker.positions, broker_map)
    has_diff = bool(qty_mismatch or orphaned or unregistered)
    report = _format_report(qty_mismatch, orphaned, unregistered, applied=args.apply)
    print(report)

    # 4. --apply: 실제 보정 + save (dry-run 기본은 미변경)
    if args.apply:
        mtime_before = POSITIONS_FILE.stat().st_mtime if POSITIONS_FILE.exists() else 0.0
        tracker.sync_with_broker(holdings)
        mtime_after = POSITIONS_FILE.stat().st_mtime if POSITIONS_FILE.exists() else 0.0
        logger.info(
            "[reconcile] --apply 보정 완료 (positions.json mtime %s→%s)",
            mtime_before, mtime_after,
        )
    else:
        logger.info("[reconcile] dry-run — positions.json 미변경 (대조만)")

    # 5. 불일치 시 텔레그램 알림
    if has_diff and not args.no_telegram:
        try:
            from src.telegram_sender import send_message
            send_message(report)
        except Exception as e:
            logger.warning("[reconcile] 텔레그램 발송 실패: %s", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
