#!/usr/bin/env python
"""scripts/upload_two_layer.py — 2층 포트 현황 + 드로다운 알림 골격 적재기.

설계서(2층 구조 B상시 + −15%알림)를 데이터계약 2테이블에 UPSERT:
  dashboard_two_layer       (하루 1행: core82/satellite18 + 실데이터 자리)
  dashboard_drawdown_alert  (하루 1행: 평소 level=normal)

★골격(관측) 적재 — 누적수익률·MDD·현재DD 등 실데이터는 포트 실운용(unfreeze) 후 채워진다.
  매매로직 0·실주문 0·freeze 무손상. 6/11 finality: dry 기본, --write로만 기록.

Usage:
    python -u -X utf8 scripts/upload_two_layer.py            # dry-run
    python -u -X utf8 scripts/upload_two_layer.py --write    # Supabase UPSERT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.use_cases.two_layer_portfolio import (  # noqa: E402
    build_drawdown_alert_row,
    build_two_layer_row,
)

logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))


def build_rows() -> tuple[dict, dict, str]:
    now = datetime.now(KST)
    date_str = now.strftime("%Y-%m-%d")
    snapshot_iso = now.isoformat()
    two_layer = build_two_layer_row(date_str, snapshot_iso)
    alert = build_drawdown_alert_row(date_str, snapshot_iso)
    return two_layer, alert, date_str


def upsert(two_layer: dict, alert: dict) -> bool:
    from src.adapters.flowx_uploader import FlowxUploader

    up = FlowxUploader()
    if not up.is_active:
        logger.error("Supabase 미연결 — 적재 불가")
        return False
    try:
        up.client.table("dashboard_two_layer").upsert(two_layer, on_conflict="date").execute()
        up.client.table("dashboard_drawdown_alert").upsert(alert, on_conflict="date").execute()
        logger.info("dashboard_two_layer + dashboard_drawdown_alert UPSERT 완료")
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("UPSERT 실패: %s", e)
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="2층 포트 + 드로다운 알림 골격 적재")
    p.add_argument("--write", action="store_true", help="Supabase UPSERT (기본: dry-run)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    two_layer, alert, date_str = build_rows()
    print(f"\n[2층 골격] {date_str}")
    print("  two_layer:", json.dumps(two_layer, ensure_ascii=False, default=str))
    print("  alert    :", json.dumps(alert, ensure_ascii=False, default=str))

    if args.write:
        ok = upsert(two_layer, alert)
        print(f"\n[적재] {'✅ 완료' if ok else '❌ 실패'} — {date_str}")
    else:
        print("\n[dry-run] 적재 안 함. 실제 기록은 --write")


if __name__ == "__main__":
    main()
