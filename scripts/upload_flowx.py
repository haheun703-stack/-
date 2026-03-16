"""
FLOWX Supabase 업로드 — 장마감 후 실행

ETF 시그널 + 외국인 자금 흐름을 Supabase에 업로드.
BAT-D 마지막 단계에서 호출.

Usage:
    python scripts/upload_flowx.py            # ETF + 외국인 자금
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
    build_foreign_flow_rows,
    build_ai_pick_rows,
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
    foreign_rows = build_foreign_flow_rows()
    ai_rows = build_ai_pick_rows()

    print(f"\n[FLOWX] ETF 시그널: {len(etf_rows)}건")
    for r in etf_rows:
        if r["signal"] != "HOLD":
            print(f"  {r['signal']:4s} {r['name']:16s} score={r['score']} rank={r['sector_rotation_rank']}")

    buy_sell = [r for r in etf_rows if r["signal"] in ("BUY", "SELL")]
    hold = [r for r in etf_rows if r["signal"] == "HOLD"]
    print(f"  BUY/SELL: {len(buy_sell)}건, HOLD: {len(hold)}건")

    # AI 추천 요약
    print(f"\n[FLOWX] AI 추천: {len(ai_rows)}건")
    for r in ai_rows:
        regime = r.get("momentum_regime", "")
        print(f"  {r['grade']:2s} {r['name']:14s} 점수={r['total_score']} 진입={r['entry_price']:,} 목표={r['target_price']:,} [{regime}]")

    print(f"\n[FLOWX] 외국인 자금: {len(foreign_rows)}건")
    inflow = [r for r in foreign_rows if r["signal"] == "INFLOW"]
    outflow = [r for r in foreign_rows if r["signal"] == "OUTFLOW"]
    neutral = [r for r in foreign_rows if r["signal"] == "NEUTRAL"]
    if inflow:
        print(f"  INFLOW ({len(inflow)}건):")
        for r in inflow:
            print(f"    {r['name']:16s} z={r['z_score']:+.2f} score={r['score']}")
    if outflow:
        print(f"  OUTFLOW ({len(outflow)}건):")
        for r in outflow:
            print(f"    {r['name']:16s} z={r['z_score']:+.2f} score={r['score']}")
    print(f"  NEUTRAL: {len(neutral)}건 (업로드 제외)")

    # 듀얼출력 필터: NEUTRAL 제외, INFLOW/OUTFLOW만 업로드
    foreign_filtered = [r for r in foreign_rows if r["signal"] != "NEUTRAL"]
    print(f"  -> 업로드 대상: {len(foreign_filtered)}건 (INFLOW+OUTFLOW)")

    if args.dry_run:
        print("\n[DRY-RUN] 업로드 스킵")
        return

    # 업로드
    uploader = FlowxUploader()
    if not uploader.is_active:
        print("\n[FLOWX] Supabase 미연결 — .env에 SUPABASE_URL/KEY 설정 필요")
        return

    ok1 = uploader.upload_etf_signals(etf_rows)          # 전체 (HOLD 포함, 섹터 순위용)
    ok2 = uploader.upload_foreign_flow(foreign_filtered)  # INFLOW/OUTFLOW만
    ok3 = uploader.upload_ai_picks(ai_rows)              # AI 추천 (short_signals)

    print(f"\n[FLOWX] 업로드 완료:")
    print(f"  ETF={'OK' if ok1 else 'FAIL'} ({len(etf_rows)}건)")
    print(f"  외국인={'OK' if ok2 else 'FAIL'} ({len(foreign_filtered)}건)")
    print(f"  AI추천={'OK' if ok3 else 'FAIL'} ({len(ai_rows)}건)")


if __name__ == "__main__":
    main()
