"""
KIS 실잔고 → positions.json 1회성 동기화 스크립트

수동 매수분이 positions.json에 미반영된 경우,
kis_balance.json에서 읽어와 positions.json에 등록합니다.

- 이미 등록된 종목은 수량만 갱신
- 미등록 종목은 avg_price를 entry_price로 하여 신규 등록
- 수동 매수이므로 grade="MANUAL", trigger_type="manual"

Usage:
    python -u -X utf8 scripts/sync_positions_from_kis.py          # dry-run
    python -u -X utf8 scripts/sync_positions_from_kis.py --apply  # 실제 적용
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

POSITIONS_FILE = PROJECT_ROOT / "data" / "positions.json"
KIS_BALANCE_FILE = PROJECT_ROOT / "data" / "kis_balance.json"


def load_json(path: Path) -> list | dict:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def sync(apply: bool = False) -> None:
    # 1. KIS 잔고 로드
    balance = load_json(KIS_BALANCE_FILE)
    if not isinstance(balance, dict) or "holdings" not in balance:
        print("[ERROR] kis_balance.json 형식이 올바르지 않습니다.")
        return

    holdings = balance["holdings"]
    fetched_at = balance.get("fetched_at", "unknown")
    print(f"[KIS 잔고] {len(holdings)}종목, 조회 시각: {fetched_at}")

    # 2. 현재 positions.json 로드
    positions = load_json(POSITIONS_FILE)
    if not isinstance(positions, list):
        positions = []
    existing_tickers = {p["ticker"] for p in positions}
    print(f"[positions.json] 기존 {len(positions)}종목: {existing_tickers or '없음'}")

    # 3. 동기화
    added = []
    updated = []
    unchanged = []

    for h in holdings:
        ticker = h["ticker"]
        name = h.get("name", "")
        qty = h["quantity"]
        avg_price = h["avg_price"]
        current_price = h.get("current_price", avg_price)
        pnl_pct = h.get("pnl_pct", 0)

        if ticker in existing_tickers:
            # 기존 포지션 수량 갱신
            for p in positions:
                if p["ticker"] == ticker:
                    if p["shares"] != qty:
                        old_qty = p["shares"]
                        p["shares"] = qty
                        p["current_price"] = current_price
                        updated.append(f"  {ticker} {name}: {old_qty}주→{qty}주")
                    else:
                        p["current_price"] = current_price
                        unchanged.append(f"  {ticker} {name}: {qty}주 (변경 없음)")
                    break
        else:
            # 신규 포지션 등록
            stop_loss_pct = 0.08  # 수동 매수: 기본 8% 손절
            new_pos = {
                "ticker": ticker,
                "name": name,
                "entry_date": datetime.now().strftime("%Y-%m-%d"),
                "entry_price": float(avg_price),
                "shares": qty,
                "current_price": float(current_price),
                "stop_loss": float(avg_price * (1 - stop_loss_pct)),
                "target_price": float(avg_price * 1.15),
                "atr_value": 0.0,
                "grade": "MANUAL",
                "trigger_type": "manual",
                "stop_loss_pct": stop_loss_pct,
                "highest_price": float(max(avg_price, current_price)),
                "trailing_stop": 0.0,
                "partial_exits_done": 0,
                "initial_shares": qty,
                "news_grade": "",
                "max_hold_days": 60,  # 수동 매수: 60일 (장기 보유 허용)
            }
            positions.append(new_pos)
            added.append(
                f"  {ticker} {name}: {qty}주 @ {avg_price:,.0f}원 "
                f"(현재 {current_price:,.0f}원, {pnl_pct:+.1f}%)"
            )

    # 4. KIS에 없지만 positions.json에 있는 종목 → 이미 매도된 것
    kis_tickers = {h["ticker"] for h in holdings}
    removed = []
    positions_cleaned = []
    for p in positions:
        if p["ticker"] in kis_tickers:
            positions_cleaned.append(p)
        else:
            removed.append(f"  {p['ticker']} {p.get('name','')}: 실잔고 없음 → 제거")
    positions = positions_cleaned

    # 5. 결과 출력
    print(f"\n{'=' * 60}")
    print(f"  동기화 결과 (dry_run={not apply})")
    print(f"{'=' * 60}")

    if added:
        print(f"\n[신규 등록] {len(added)}건:")
        for a in added:
            print(a)

    if updated:
        print(f"\n[수량 갱신] {len(updated)}건:")
        for u in updated:
            print(u)

    if removed:
        print(f"\n[제거] {len(removed)}건:")
        for r in removed:
            print(r)

    if unchanged:
        print(f"\n[변경 없음] {len(unchanged)}건:")
        for u in unchanged:
            print(u)

    print(f"\n최종: {len(positions)}종목")

    # 6. 적용
    if apply:
        POSITIONS_FILE.write_text(
            json.dumps(positions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[적용 완료] {POSITIONS_FILE} 저장됨")
    else:
        print(f"\n[DRY-RUN] --apply 옵션으로 실행하면 실제 저장됩니다.")


if __name__ == "__main__":
    apply_mode = "--apply" in sys.argv
    sync(apply=apply_mode)
