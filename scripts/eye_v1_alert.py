"""EYE 필터 v1.1 자동 텔레그램 알림 (2026-05-18 신규)

배경: 5/18 14:23 수동 시연 성공 → 사장님 카톡 도달 검증
- 5/20 출격 후 매일 14:00 자비스 강력포착 9건에 EYE 필터 적용 → 카톡 자동 발송
- 사장님이 점심 후 매수 결정 시점에 EYE 결과 자동 도착

Usage:
    python scripts/eye_v1_alert.py             # 1회 즉시 실행 + 텔레그램
    python scripts/eye_v1_alert.py --no-tg     # 텔레그램 OFF (로그만)

Cron 등록 (5/19부터 매일 자동):
    0 14 * * 1-5 cd /home/ubuntu/quantum-master && ./venv/bin/python3.11 scripts/eye_v1_alert.py >> /tmp/eye_v1.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.adapters.kis_stock_data_adapter import KisStockDataAdapter  # noqa: E402
from src.use_cases.eye_filters import evaluate_filters  # noqa: E402

TOMORROW_PICKS = PROJECT_ROOT / "data" / "tomorrow_picks.json"
DEFAULT_TOP_N = 9
DEFAULT_GRADE = "강력 포착"

logger = logging.getLogger(__name__)


def fetch_top_picks(top_n: int = DEFAULT_TOP_N, grade: str = DEFAULT_GRADE) -> list[dict]:
    """tomorrow_picks 강력포착 TOP N."""
    if not TOMORROW_PICKS.exists():
        return []
    data = json.loads(TOMORROW_PICKS.read_text(encoding="utf-8"))
    return [p for p in data.get("picks", []) if p.get("grade") == grade][:top_n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--no-tg", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    adp = KisStockDataAdapter()
    broker = adp.broker

    picks = fetch_top_picks(top_n=args.top)
    if not picks:
        print("강력포착 0건 — tomorrow_picks 확인 필요")
        return 1

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M")

    lines = [f"🎯 EYE 필터 v1.1 자동 ({time} KST)"]
    lines.append(f"자비스 강력포착 {len(picks)}건 — 사장님 EYE 통찰 적용")
    lines.append("━" * 12)

    n_skip = 0
    n_pass = 0
    pass_pnl_sum = 0.0
    pass_records = []

    for p in picks:
        tk = p.get("ticker", "")
        nm = p.get("name", tk)
        res = evaluate_filters(broker, tk, date)
        skip = res["should_skip"]
        if skip:
            n_skip += 1
            reasons = ", ".join(res["skip_reasons"])[:40]
            lines.append(f"🔴 [SKIP] {nm} — {reasons}")
        else:
            n_pass += 1
            try:
                px = broker.fetch_price(tk).get("output", {})
                cur = int(px.get("stck_prpr", 0))
                opn = int(px.get("stck_oprc", 0))
                chg = ((cur - opn) / opn * 100) if opn > 0 else 0
                pass_pnl_sum += chg
                pass_records.append((nm, chg))
                lines.append(f"🟢 [PASS] {nm} ({chg:+.2f}%)")
            except Exception:
                lines.append(f"🟢 [PASS] {nm}")

    lines.append("━" * 12)
    avg_pass = pass_pnl_sum / n_pass if n_pass > 0 else 0
    lines.append(f"결과: SKIP {n_skip}건 / PASS {n_pass}건")
    lines.append(f"PASS 평균 변동: {avg_pass:+.2f}%")
    lines.append("")
    lines.append("★ 사장님 통찰: 수익 X, 손실 회피")
    lines.append("5/20 출격 시 EYE 통과 종목만 매수")

    msg = "\n".join(lines)
    print(msg)

    if not args.no_tg:
        try:
            from src.telegram_sender import send_message

            send_message(msg)
            print("\n[TG SENT]")
        except Exception as e:
            print(f"\n[TG FAIL] {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
