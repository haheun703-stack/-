#!/usr/bin/env python
"""dashboard_valuation_band 적재기 — 밸류밴드 종목별(FCF·옥석) Supabase UPSERT.

밸류밴드 관측 패널(src/use_cases/valuation_band.py)의 미국·한국 시총 top30 스냅샷을
정보봇 데이터계약 §1 스키마(dashboard_valuation_band)로 변환해 매일 장마감 후 UPSERT.

★관측 적재 전용 — 매매로직0, freeze 무손상. 6/11 finality 교훈: 기본 dry, --write로만 기록.
하이브리드(데이터계약): PER/PBR/추정/52주/ROE는 valuation_band 1차 소스 + FCF·verdict 신규.
  checkup(quant_bluechip_checkup) 재활용은 pos_52w 폴백만(per/pbr는 실측상 0이라 제외),
  checkup이 당일 신선할 때만 적용(stale 가드).

Usage:
    python -u -X utf8 scripts/upload_valuation_band.py            # dry-run (적재 안 함)
    python -u -X utf8 scripts/upload_valuation_band.py --write    # Supabase UPSERT
    python -u -X utf8 scripts/upload_valuation_band.py --top 30 --no-checkup
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.use_cases.valuation_band import (  # noqa: E402
    KR_POOL,
    TOP_N,
    US_POOL,
    _top_by_cap,
    apply_checkup_pos,
    fetch_kr,
    fetch_us,
    to_dashboard_row,
)

logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))
CHECKUP_JSON = PROJECT_ROOT / "data" / "bluechip_checkup.json"


def load_checkup_pos(date_str: str) -> dict[str, float]:
    """checkup 로컬 JSON에서 한국 종목 position_pct 재활용 — 당일 데이터만(stale 가드)."""
    if not CHECKUP_JSON.exists():
        return {}
    try:
        with open(CHECKUP_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning("checkup JSON 로드 실패: %s", e)
        return {}
    if data.get("date") != date_str:
        logger.info("checkup stale(%s != %s) — pos 폴백 생략", data.get("date"), date_str)
        return {}
    pos: dict[str, float] = {}
    for b in data.get("bluechips", []):
        code, p = b.get("code"), b.get("position_pct")
        if code and isinstance(p, (int, float)) and p > 0:
            pos[code] = float(p)
    logger.info("checkup pos 재활용: %d종목", len(pos))
    return pos


def collect_rows(top_n: int, use_checkup: bool, market: str = "ALL") -> tuple[list[dict], str]:
    """미국·한국 스냅샷 수집 → 데이터계약 행 리스트 + 적재일(KST).

    market: US/KR/ALL. UPSERT 키가 (date,market,ticker)라 시장별 분리 적재해도 충돌 없음
    (미국=전일 마감 finality OK → 먼저 적재 / 한국=장마감 후 추가 가능).
    """
    now = datetime.now(KST)
    date_str = now.strftime("%Y-%m-%d")
    snapshot_iso = now.isoformat()

    print(f"[밸류밴드 적재] 시장={market} 시총 top{top_n} 수집 중...")
    us = _top_by_cap(fetch_us(US_POOL), top_n) if market in ("US", "ALL") else []
    kr = _top_by_cap(fetch_kr(KR_POOL), top_n) if market in ("KR", "ALL") else []

    if kr and use_checkup:
        kr = apply_checkup_pos(kr, load_checkup_pos(date_str))

    snaps = us + kr
    rows = [to_dashboard_row(s, date_str, snapshot_iso) for s in snaps]
    return rows, date_str


def upsert_rows(rows: list[dict]) -> bool:
    """dashboard_valuation_band UPSERT (on_conflict=date,market,ticker)."""
    from src.adapters.flowx_uploader import FlowxUploader

    up = FlowxUploader()
    if not up.is_active:
        logger.error("Supabase 미연결 — 적재 불가")
        return False
    try:
        up.client.table("dashboard_valuation_band").upsert(
            rows, on_conflict="date,market,ticker"
        ).execute()
        logger.info("dashboard_valuation_band UPSERT: %d행", len(rows))
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("UPSERT 실패: %s", e)
        return False


def print_summary(rows: list[dict]) -> None:
    """verdict 분포 + 가치함정/저점후보/이미오름 미리보기."""
    dist = Counter(r["verdict"] for r in rows)
    print(f"\n총 {len(rows)}행 — verdict 분포: {dict(dist)}")
    for cat in ("가치함정", "저점후보", "이미오름"):
        picks = [r for r in rows if r["verdict"] == cat]
        if not picks:
            continue
        names = ", ".join(
            f"{r['name']}(FCF{r['fcf_yield']}%)" if r["fcf_yield"] is not None else str(r["name"])
            for r in picks[:6]
        )
        print(f"  [{cat}] {len(picks)}종목: {names}")


def main() -> None:
    p = argparse.ArgumentParser(description="dashboard_valuation_band 적재")
    p.add_argument("--write", action="store_true", help="Supabase UPSERT (기본: dry-run)")
    p.add_argument("--top", type=int, default=TOP_N, help=f"시장별 top N (기본 {TOP_N})")
    p.add_argument("--market", choices=["US", "KR", "ALL"], default="ALL",
                   help="적재 시장 (기본 ALL — 미국 finality 먼저면 US)")
    p.add_argument("--no-checkup", action="store_true", help="checkup pos 폴백 생략")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    rows, date_str = collect_rows(args.top, not args.no_checkup, args.market)
    print_summary(rows)

    if args.write:
        ok = upsert_rows(rows)
        print(f"\n[적재] {'✅ 완료' if ok else '❌ 실패'} — {date_str} {len(rows)}행")
    else:
        print(f"\n[dry-run] 적재 안 함. 실제 기록은 --write (총 {len(rows)}행 준비됨)")


if __name__ == "__main__":
    main()
