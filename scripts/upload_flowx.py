"""
FLOWX Supabase 업로드 — 장마감 후 실행

ETF 시그널 + 중국자금 흐름을 Supabase에 업로드.
BAT-D 마지막 단계에서 호출.

Usage:
    python scripts/upload_flowx.py            # ETF + 중국자금
    python scripts/upload_flowx.py --dry-run  # 업로드 안 하고 데이터만 확인
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.flowx_uploader import (
    FlowxUploader,
    build_etf_signal_rows,
    build_china_flow_rows,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="FLOWX Supabase 업로드")
    parser.add_argument("--dry-run", action="store_true", help="업로드 없이 데이터만 출력")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 데이터 변환
    etf_rows = build_etf_signal_rows()
    china_rows = build_china_flow_rows()

    print(f"\n[FLOWX] ETF 시그널: {len(etf_rows)}건")
    for r in etf_rows:
        if r["signal"] != "HOLD":
            print(f"  {r['signal']:4s} {r['name']:16s} score={r['score']} rank={r['sector_rotation_rank']}")

    buy_sell = [r for r in etf_rows if r["signal"] in ("BUY", "SELL")]
    hold = [r for r in etf_rows if r["signal"] == "HOLD"]
    print(f"  BUY/SELL: {len(buy_sell)}건, HOLD: {len(hold)}건")

    print(f"\n[FLOWX] 중국자금: {len(china_rows)}건")
    inflow = [r for r in china_rows if r["signal"] == "INFLOW"]
    if inflow:
        print(f"  INFLOW 종목:")
        for r in inflow:
            print(f"    {r['name']:12s} z={r['z_score']:.1f} score={r['score']}")

    if args.dry_run:
        print("\n[DRY-RUN] 업로드 스킵")
        return

    # 업로드
    uploader = FlowxUploader()
    if not uploader.is_active:
        print("\n[FLOWX] Supabase 미연결 — .env에 SUPABASE_URL/KEY 설정 필요")
        return

    ok1 = uploader.upload_etf_signals(etf_rows)
    ok2 = uploader.upload_china_flow(china_rows)

    print(f"\n[FLOWX] 업로드 완료: ETF={'OK' if ok1 else 'FAIL'}, 중국자금={'OK' if ok2 else 'FAIL'}")


if __name__ == "__main__":
    main()
