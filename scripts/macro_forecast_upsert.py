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
    month_to_event_date,
    upsert_forecast_actual,
)

# consensus(나우캐스트)와 actual(FRED) 둘 다 가능한 핵심 인플레 3종(예실 완성 대상)
DEFAULT_CODES = ["CPI_HEAD", "CPI_CORE", "PCE"]


def merge_rows(
    consensus: dict[str, Any] | list[dict[str, Any]],
    fred_actuals: dict[str, dict[str, float]],
    existing: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """기존행(consensus 보존) + 나우캐스트 consensus(1개 이상 달) + FRED actual(과거달) → 행 목록.

    consensus: 단일 doc(dict) 또는 여러 달 doc 리스트(진행월+임박 발표월). 같은
      (code, event_date)는 한 행으로 합쳐 build_forecast_row가 surprise를 재계산.
      서로 다른 event_date(예 5월/6월)는 별 PK라 서로 덮어쓰지 않는다.
    """
    consensus_docs = consensus if isinstance(consensus, list) else [consensus]
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    # 1) 기존 행 시드 — consensus 보존(actual backfill 후에도 안 날아가게)
    for key, r in existing.items():
        merged[key] = {
            "consensus": r.get("consensus"),
            "consensus_source": r.get("consensus_source"),
            "actual": r.get("actual"),
        }

    # 2) 나우캐스트 consensus(진행월 + 임박 발표월 등, 데이터월별 별도 event_date)
    for consensus_doc in consensus_docs:
        target_date = consensus_doc.get("target_date")
        src = consensus_doc.get("source")
        edt = consensus_doc.get("event_datetime_kst")  # 임박월 발표예정일시(KST) 주입용
        if not target_date:
            continue
        for code, val in consensus_doc.get("values", {}).items():
            d = merged.setdefault((code, target_date), {})
            d["consensus"] = val
            d["consensus_source"] = src
            if edt:
                d["event_datetime_kst"] = edt
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
            event_datetime_kst=d.get("event_datetime_kst"),
        ))
    return rows


def build_backfill_rows(
    codes: list[str],
    target_date: str,
    fred_actuals: dict[str, dict[str, float]],
    existing: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """지정 데이터월(target_date)의 actual만 채운 행 목록 — consensus/edt/prior/note 보존.

    actual-only backfill 전용(merge_rows 안 거침 → 타 월 오염 원천 차단).
      - FRED에 target_date actual이 없는 code는 skip(no-op). 예상값 임의 입력 없음.
      - target_date 행만 생성(다른 월·과거월 절대 미포함).
      - 기존 행의 consensus/consensus_source/event_datetime_kst/prior/note를 그대로 싣고
        actual만 추가 → build_forecast_row가 surprise/stance/market_impact를 기존 로직으로
        자동 산출(강제 생성 아님; consensus 없으면 surprise=None 유지).
    """
    rows: list[dict[str, Any]] = []
    for code in codes:
        act = fred_actuals.get(code, {}).get(target_date)
        if act is None:
            continue  # FRED 미등재 → no-op
        ex = existing.get((code, target_date), {})
        rows.append(build_forecast_row(
            code, target_date,
            consensus=ex.get("consensus"),
            consensus_source=ex.get("consensus_source"),
            actual=act,
            prior=ex.get("prior"),
            event_datetime_kst=ex.get("event_datetime_kst"),
            note=ex.get("note"),
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="macro_forecast consensus+actual 적재(dry_run 기본)")
    ap.add_argument("--write", action="store_true", help="실제 upsert(미지정 시 dry_run)")
    ap.add_argument("--months", type=int, default=6, help="FRED actual 조회 개월수(기본 6)")
    ap.add_argument("--codes", type=str, default=",".join(DEFAULT_CODES),
                    help="대상 indicator_code 콤마구분(기본 CPI_HEAD,CPI_CORE,PCE)")
    ap.add_argument("--imminent-month", type=str, default=None,
                    help="임박 발표 데이터월(예: 2026-5) consensus를 별도 event_date 행으로 추가 수집. "
                         "미지정 시 진행월(data[-1])만. 발표 임박월은 클리블랜드 JSON에서 별 항목이라 콕 집어 뽑음.")
    ap.add_argument("--imminent-only", action="store_true",
                    help="B 순수안: 임박월 consensus만 적재(진행월·과거 actual·기존행 전부 제외). "
                         "--imminent-month 필수.")
    ap.add_argument("--imminent-datetime-kst", type=str, default=None,
                    help="임박월 row 발표예정일시(KST ISO8601, 예: 2026-06-10T21:30:00+09:00). "
                         "웹 D-day 정상화용. actual은 여전히 발표 후 backfill.")
    ap.add_argument("--backfill-month", type=str, default=None,
                    help="actual-only backfill: 지정 데이터월(예: 2026-5) actual만 FRED에서 채움. "
                         "consensus·발표일시·prior·note 보존, 타 월·과거월 미포함. "
                         "FRED 미등재 시 no-op(0행). 발표 후 실행.")
    args = ap.parse_args()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]

    print(f"=== macro_forecast upsert ({'WRITE' if args.write else 'DRY_RUN'}) codes={codes} ===")

    if args.backfill_month:
        # ── actual-only backfill: 지정 월 actual만(consensus/edt/prior/note 보존, 타 월 무관) ──
        try:
            _y, _m = args.backfill_month.split("-")[:2]
            target_date = month_to_event_date(int(_y), int(_m))
        except (ValueError, TypeError):
            ap.error(f"--backfill-month 형식 오류(예: 2026-5): {args.backfill_month!r}")
        print(f"★ backfill-only: {target_date} actual만 채움"
              "(consensus/발표일시/prior/note 보존 · 타 월·과거월 미포함)")
        try:
            fred_actuals = fetch_fred_actuals(codes, months=args.months)
        except RuntimeError as e:
            print(f"FRED 수집 실패: {e} → 적재 0행")
            fred_actuals = {}
        avail = {c: fred_actuals.get(c, {}).get(target_date) for c in codes}
        print(f"  FRED {target_date} actual = {avail}")
        existing = fetch_existing_rows(codes)
        rows = build_backfill_rows(codes, target_date, fred_actuals, existing)
        if not rows:
            print(f"  → 적재 대상 0행: FRED {target_date} 미등재 = no-op. "
                  "기존 consensus row 그대로 유지(발표 후 재실행).")
    elif args.imminent_only:
        # ── B 순수안: 임박월 consensus만 (진행월/과거 actual/기존행 전부 제외) ──
        if not args.imminent_month:
            ap.error("--imminent-only 는 --imminent-month 가 필요합니다(예: --imminent-month 2026-5)")
        imm = fetch_cleveland_nowcast(target_month=args.imminent_month)
        if args.imminent_datetime_kst:
            imm["event_datetime_kst"] = args.imminent_datetime_kst
        print("★ imminent-only(B 순수안): 진행월·과거 actual·기존행 전부 제외 — 임박월 consensus만 적재")
        print(f"consensus(임박월): target={imm.get('target_month')} ({imm.get('target_date')}) "
              f"asof={imm.get('asof')} src={imm.get('source')} url={imm.get('source_url')}")
        print(f"  └ values={imm.get('values')}  event_datetime_kst={imm.get('event_datetime_kst')}  "
              f"(data_date=event_date={imm.get('target_date')})")
        if not imm.get("values"):
            print(f"  ⚠️ 임박월 {args.imminent_month} 항목 없음/값 비어있음 — 적재 대상 0행.")
        rows = merge_rows([imm], {}, {})
    else:
        # ── 운영 흐름: 진행월 consensus + FRED actual + 기존행 머지 ──
        consensus_doc = fetch_cleveland_nowcast()
        consensus_docs: list[dict[str, Any]] = [consensus_doc]
        print(f"consensus(진행월): target={consensus_doc.get('target_month')} "
              f"({consensus_doc.get('target_date')}) asof={consensus_doc.get('asof')} "
              f"src={consensus_doc.get('source')} values={consensus_doc.get('values')}")

        # 임박 발표월 consensus(별도 event_date 행) — 진행월과 혼동 방지 위해 별도 출력
        if args.imminent_month:
            imm = fetch_cleveland_nowcast(target_month=args.imminent_month)
            if args.imminent_datetime_kst:
                imm["event_datetime_kst"] = args.imminent_datetime_kst
            consensus_docs.append(imm)
            print(f"consensus(임박월): target={imm.get('target_month')} ({imm.get('target_date')}) "
                  f"asof={imm.get('asof')} src={imm.get('source')} url={imm.get('source_url')}")
            print(f"  └ values={imm.get('values')}  event_datetime_kst={imm.get('event_datetime_kst')}  "
                  f"(data_date=event_date={imm.get('target_date')})")
            if not imm.get("values"):
                print(f"  ⚠️ 임박월 {args.imminent_month} 항목 없음/값 비어있음 — 클리블랜드 JSON 미공개 가능.")

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
        rows = merge_rows(consensus_docs, fred_actuals, existing)

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
