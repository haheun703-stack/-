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
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.flowx_uploader import (
    FlowxUploader,
    build_etf_signal_rows,
    build_foreign_flow_rows,
    build_ai_pick_rows,
    build_jarvis_payload,
    build_smart_money_rows,
    build_etf_signals_rows,
    build_relay_rows,
    build_sniper_rows,
    build_sector_rotation_rows,
    build_crash_bounce_rows,
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

    # ── 시나리오 대시보드 데이터 ──
    from dashboard_data import build_zone_scenario
    scenario_data = build_zone_scenario()
    sc_count = len(scenario_data.get("active_scenarios", []))
    cm_count = len(scenario_data.get("commodities", []))
    st_count = len(scenario_data.get("scenario_stocks", []))
    cf_count = len(scenario_data.get("conflicts", []))
    print(f"\n[FLOWX] 시나리오 대시보드: {sc_count}개 시나리오, 원자재 {cm_count}개, 종목 {st_count}개, 충돌 {cf_count}건")
    for sc in scenario_data.get("active_scenarios", []):
        print(f"  {sc['name']} (P{sc['current_phase']}/{sc['total_phases']}) 점수={sc['score']} D+{sc['days_active']}")

    # ── Row 테이블 미리보기 ──
    date_str_preview = datetime.now().strftime("%Y-%m-%d")
    sm_rows = build_smart_money_rows(date_str_preview)
    etf_sig_rows = build_etf_signals_rows(date_str_preview)
    relay_rows = build_relay_rows(date_str_preview)
    sniper_rows = build_sniper_rows(date_str_preview)
    sr_rows = build_sector_rotation_rows(date_str_preview)

    print(f"\n[FLOWX] 스마트머니: {len(sm_rows)}건")
    for r in sm_rows[:5]:
        print(f"  {r['signal_type']:12s} {r['name']:14s} 외인{r['foreign_consec_days']}일 기관{r['inst_consec_days']}일 점수={r['score']}")

    print(f"\n[FLOWX] ETF 시그널 대시보드: {len(etf_sig_rows)}건")
    for r in etf_sig_rows[:5]:
        print(f"  {r['signal_type']:10s} {r['name']:12s} score={r['score']} vol={r['volume']:,}")

    print(f"\n[FLOWX] 릴레이: {len(relay_rows)}건")
    for r in relay_rows[:5]:
        print(f"  {r['lead_sector']:8s}→{r['lag_sector']:8s} gap={r['gap']:+.1f}% {r['signal_type']} score={r['score']}")

    print(f"\n[FLOWX] 스나이퍼: {len(sniper_rows)}건")
    for r in sniper_rows[:5]:
        print(f"  {r['name']:14s} RSI={r['rsi']} BB={r['bb_position']:.2f} ADX={r['adx']} {r['signal_type']} score={r['score']}")

    cb_rows = build_crash_bounce_rows(date_str_preview)
    print(f"\n[FLOWX] 급락반등: {len(cb_rows)}건")
    for r in cb_rows[:5]:
        print(f"  {r['grade']:4s} {r['name']:14s} 이격={r['gap_20ma']:+.1f}% 거래량={r['volume_ratio']:.1f}배 {r['signal_type']} score={r['score']}")

    print(f"\n[FLOWX] 섹터로테이션: {len(sr_rows)}건")
    for r in sr_rows[:5]:
        print(f"  #{r['rank']:2d} {r['sector']:8s} score={r['score']} 5d={r['ret_5d']:+.1f}% flow={r['flow']:+.0f}억")

    if args.dry_run:
        print("\n[DRY-RUN] 업로드 스킵")
        return

    # 업로드
    uploader = FlowxUploader()
    if not uploader.is_active:
        print("\n[FLOWX] Supabase 미연결 — .env에 SUPABASE_URL/KEY 설정 필요")
        return

    date_str = datetime.now().strftime("%Y-%m-%d")

    ok1 = uploader.upload_etf_signals(etf_rows)          # 전체 (HOLD 포함, 섹터 순위용)
    ok2 = uploader.upload_foreign_flow(foreign_filtered)  # INFLOW/OUTFLOW만
    ok3 = uploader.upload_ai_picks(ai_rows)              # AI 추천 (short_signals)
    ok4 = uploader.upload_quant_scenarios(scenario_data, date_str)  # 시나리오 대시보드

    # ── 퀀트 10개 독립 테이블 (JSONB 6 + Row 4) ──
    print("\n[FLOWX] 퀀트 10테이블 업로드...")
    q_all = uploader.upload_all_quant_tables(date_str)
    q_ok = sum(v for v in q_all.values())

    # ── 자비스 컨트롤타워 (quant_jarvis) ──
    print("\n[FLOWX] 자비스 컨트롤타워 빌드...")
    jarvis = build_jarvis_payload()
    n_picks = len(jarvis.get("picks", {}).get("picks", []))
    regime = jarvis.get("brain", {}).get("regime", "?")
    print(f"  picks={n_picks}건, regime={regime}")
    ok5 = uploader.upload_jarvis_data(jarvis, date_str)

    print(f"\n[FLOWX] 업로드 완료:")
    print(f"  ETF={'OK' if ok1 else 'FAIL'} ({len(etf_rows)}건)")
    print(f"  외국인={'OK' if ok2 else 'FAIL'} ({len(foreign_filtered)}건)")
    print(f"  AI추천={'OK' if ok3 else 'FAIL'} ({len(ai_rows)}건)")
    print(f"  시나리오={'OK' if ok4 else 'FAIL'} ({sc_count}개 시나리오)")
    print(f"  자비스={'OK' if ok5 else 'FAIL'} ({n_picks}종목)")
    print(f"  퀀트테이블={q_ok}/{len(q_all)} 성공")
    for k, v in q_all.items():
        print(f"    {k}={'OK' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
