"""매크로 예실 트래커 — consensus(나우캐스트) + actual(FRED) 적재 운영 CLI.

정보봇 P1-② "발표 후 actual 자동적재" + 웹봇 "발표 완료 카드" 표출 대기분.
흐름:
  1) 클리블랜드 나우캐스트 → 현재 진행 달 consensus (MoM%)
  2) FRED → 최근 N개월 actual (지수→MoM% 등 변환, consensus 단위 정합)
  3) 기존 DB 행의 consensus 보존(read-only SELECT) — actual backfill로 surprise 완성
  4) 데이터월(event_date=YYYY-MM-01) 기준 머지 → build_forecast_row → upsert

★dry_run 기본(실제 write 0). --write 명시해야 적재. 매매·주문·scheduler·SAJANG·C60 무관.
사용:
  python -u -X utf8 scripts/macro_forecast_upsert.py            # dry_run 미리보기
  python -u -X utf8 scripts/macro_forecast_upsert.py --write    # 실제 upsert(자격증명 필요)
  python -u -X utf8 scripts/macro_forecast_upsert.py --months 12 --codes CPI_HEAD,CPI_CORE,PCE
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.macro.macro_forecast_tracker import (  # noqa: E402
    build_forecast_row,
    fetch_cleveland_nowcast,
    fetch_existing_rows,
    fetch_fred_actuals,
    upsert_forecast_actual,
)

# consensus(나우캐스트)와 actual(FRED) 둘 다 가능한 핵심 인플레 3종(예실 완성 대상)
DEFAULT_CODES = ["CPI_HEAD", "CPI_CORE", "PCE"]


def merge_rows(
    consensus_doc: dict[str, Any],
    fred_actuals: dict[str, dict[str, float]],
    existing: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """기존행(consensus 보존) + 나우캐스트 consensus(현재달) + FRED actual(과거달) → 행 목록.

    같은 (code, event_date)는 한 행으로 합쳐 build_forecast_row가 surprise를 재계산.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    # 1) 기존 행 시드 — consensus 보존(actual backfill 후에도 안 날아가게)
    for key, r in existing.items():
        merged[key] = {
            "consensus": r.get("consensus"),
            "consensus_source": r.get("consensus_source"),
            "actual": r.get("actual"),
        }

    # 2) 나우캐스트 consensus(현재 진행 달)
    target_date = consensus_doc.get("target_date")
    src = consensus_doc.get("source")
    if target_date:
        for code, val in consensus_doc.get("values", {}).items():
            d = merged.setdefault((code, target_date), {})
            d["consensus"] = val
            d["consensus_source"] = src
            d.setdefault("actual", None)

    # 3) FRED actual(과거 발표 달)
    for code, by_date in fred_actuals.items():
        for ev, av in by_date.items():
            d = merged.setdefault((code, ev), {})
            d["actual"] = av
            d.setdefault("consensus", None)
            d.setdefault("consensus_source", None)

    rows = []
    for (code, ev), d in sorted(merged.items()):
        rows.append(build_forecast_row(
            code, ev,
            consensus=d.get("consensus"),
            consensus_source=d.get("consensus_source"),
            actual=d.get("actual"),
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="macro_forecast consensus+actual 적재(dry_run 기본)")
    ap.add_argument("--write", action="store_true", help="실제 upsert(미지정 시 dry_run)")
    ap.add_argument("--months", type=int, default=6, help="FRED actual 조회 개월수(기본 6)")
    ap.add_argument("--codes", type=str, default=",".join(DEFAULT_CODES),
                    help="대상 indicator_code 콤마구분(기본 CPI_HEAD,CPI_CORE,PCE)")
    args = ap.parse_args()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]

    print(f"=== macro_forecast upsert ({'WRITE' if args.write else 'DRY_RUN'}) codes={codes} ===")

    consensus_doc = fetch_cleveland_nowcast()
    print(f"consensus(나우캐스트): target={consensus_doc.get('target_month')} "
          f"({consensus_doc.get('target_date')}) values={consensus_doc.get('values')}")

    try:
        fred_actuals = fetch_fred_actuals(codes, months=args.months)
    except RuntimeError as e:
        print(f"FRED actual 수집 실패: {e} → actual 없이 consensus만 진행")
        fred_actuals = {}
    for code in codes:
        n = len(fred_actuals.get(code, {}))
        print(f"  FRED actual {code}: {n}개월")

    existing = fetch_existing_rows(codes)
    print(f"기존 DB 행: {len(existing)}개(consensus 보존)")

    rows = merge_rows(consensus_doc, fred_actuals, existing)

    print(f"--- 머지 결과 {len(rows)}행 (code | event_date | consensus | actual | surprise | stance | impact) ---")
    for r in rows:
        print(f"  {r['indicator_code']:<9} {r['event_date']}  "
              f"cons={_fmt(r['consensus'])}  act={_fmt(r['actual'])}  "
              f"surp={_fmt(r['surprise'])}  {r['stance'] or '-':<4} {r['market_impact'] or '-'}")

    res = upsert_forecast_actual(rows, dry_run=not args.write)
    print(f"--- {res} ---")
    print("안전: dry_run=실제 write 0 · 매매무관 · 주문/scheduler/SAJANG/C60 무접촉"
          if not args.write else "★WRITE 모드: 실제 DB upsert 수행됨(매매무관·주문0)")


def _fmt(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "None"


if __name__ == "__main__":
    main()
