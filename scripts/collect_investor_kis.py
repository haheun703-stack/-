#!/usr/bin/env python
"""scripts/collect_investor_kis.py — KIS 종목별 투자자 수급 수집 (KRX 대체).

KIS REST API(FHKST01010900)로 종목별 외국인/기관합계/개인 3주체의 일별 수급
(매수/매도/순 × 거래량/금액)을 investor_daily.db에 적재한다. KRX 계정 잠금/만료를
우회하는 정공 경로(2026-06-15 사장님 결정: "KRX 말고 KIS로").

★ KIS 종목별 한계: 외국인/기관계/개인 3주체만 제공. 기타법인·세분(금융투자/연기금)은
  KIS 종목별 미제공(키움 opt10059 별도 계좌 필요). 시장 전체 11주체 세분은 FHPTJ04040000.

★ 매매 무관·DB write 전용 (실주문 0·매매로직 무접촉·freeze 무손상).
  investor_daily.db는 단타봇 DB의 원본(단타봇이 symlink로 공유).

스키마(investor_daily): date, ticker, name, investor,
  sell_vol, buy_vol, net_vol, sell_val, buy_val, net_val (금액=원 단위)

Usage:
  python -u -X utf8 scripts/collect_investor_kis.py                 # 최근 3거래일
  python -u -X utf8 scripts/collect_investor_kis.py --days 5        # 최근 5거래일
  python -u -X utf8 scripts/collect_investor_kis.py --dates 20260615 20260612
  python -u -X utf8 scripts/collect_investor_kis.py --limit 10      # 앞 10종목(테스트)
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # BAT/cron 안전장치 (CLAUDE.md)

import requests

from src.adapters.kis_investor_adapter import (
    KIS_APP_KEY,
    KIS_APP_SECRET,
    KIS_BASE_URL,
    _issue_token,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("collect_investor_kis")

DB_PATH = PROJECT_ROOT / "data" / "investor_flow" / "investor_daily.db"
# 한글 라벨 → KIS raw 필드 접두사
INVESTORS = [("외국인", "frgn"), ("기관합계", "orgn"), ("개인", "prsn")]
TR_ID = "FHKST01010900"


def _i(v) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def fetch_raw(ticker: str, _retry: int = 0) -> list[dict]:
    """종목별 투자자 매매동향 30일치 raw(FHKST01010900). 실패 시 []."""
    hdr = {
        "content-type": "application/json",
        "authorization": f"Bearer {_issue_token()}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": TR_ID,
    }
    try:
        r = requests.get(
            f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-investor",
            headers=hdr,
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker},
            timeout=15,
        )
        d = r.json()
    except Exception:
        return []
    if d.get("rt_cd") != "0":
        if "초당" in d.get("msg1", "") and _retry < 2:
            time.sleep(1.0 + _retry * 0.5)
            return fetch_raw(ticker, _retry + 1)
        return []
    return d.get("output", []) or []


def rows_for_ticker(ticker: str, name: str, out: list[dict], target: set[str]) -> list[tuple]:
    """raw output → investor_daily 행(3주체 × 대상 거래일). tr_pbmn(백만원)→원."""
    recs = []
    for r in out:
        ds = r.get("stck_bsop_date")
        if ds not in target:
            continue
        for ko, p in INVESTORS:
            recs.append((
                ds, ticker, name, ko,
                _i(r.get(f"{p}_seln_vol")), _i(r.get(f"{p}_shnu_vol")), _i(r.get(f"{p}_ntby_qty")),
                _i(r.get(f"{p}_seln_tr_pbmn")) * 1_000_000,
                _i(r.get(f"{p}_shnu_tr_pbmn")) * 1_000_000,
                _i(r.get(f"{p}_ntby_tr_pbmn")) * 1_000_000,
            ))
    return recs


def upsert(conn: sqlite3.Connection, ticker: str, target: set[str], recs: list[tuple]) -> None:
    """대상 거래일×3주체 기존 행 삭제 후 재적재 (재실행 안전)."""
    qs = ",".join("?" * len(target))
    conn.execute(
        f"delete from investor_daily where ticker=? and date in ({qs}) "
        f"and investor in ('외국인','기관합계','개인')",
        (ticker, *sorted(target)),
    )
    conn.executemany("insert into investor_daily values (?,?,?,?,?,?,?,?,?,?)", recs)
    conn.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description="KIS 종목별 투자자 수급 → investor_daily.db (KRX 대체)")
    ap.add_argument("--days", type=int, default=3, help="최근 N거래일 (기본 3, --dates 없을 때)")
    ap.add_argument("--dates", nargs="*", default=None, help="대상 거래일 명시 (YYYYMMDD ...)")
    ap.add_argument("--limit", type=int, default=0, help="앞 N종목만 (테스트용, 0=전체)")
    ap.add_argument("--db", type=str, default=str(DB_PATH), help="investor_daily.db 경로")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error("investor_daily.db 없음: %s", db_path)
        return 1

    conn = sqlite3.connect(str(db_path), timeout=60)
    tickers = [(t, n) for t, n in
               conn.execute("select distinct ticker, name from investor_daily").fetchall()]
    if args.limit:
        tickers = tickers[:args.limit]

    target: set[str] | None = set(args.dates) if args.dates else None
    if target:
        logger.info("대상 거래일(명시): %s", sorted(target))
    logger.info("대상 %d종목 — KIS 3주체 수급 수집", len(tickers))

    ok = empty = err = ins = 0
    t0 = time.time()
    for i, (tk, nm) in enumerate(tickers):
        try:
            out = fetch_raw(tk)
            # 첫 유효 응답에서 최근 N거래일 자동 확정 (시장 공통)
            if target is None and out:
                dates = sorted({r.get("stck_bsop_date") for r in out if r.get("stck_bsop_date")},
                               reverse=True)
                target = set(dates[:args.days])
                logger.info("대상 거래일(자동): %s", sorted(target))
            recs = rows_for_ticker(tk, nm, out, target or set())
            if recs:
                upsert(conn, tk, target, recs)
                ins += len(recs); ok += 1
            else:
                empty += 1
            time.sleep(0.22)  # KIS rate limit 여유
        except Exception as e:
            err += 1
            if err <= 5:
                logger.warning("  ERR %s: %s", tk, str(e)[:60])
        if (i + 1) % 200 == 0:
            logger.info("  %d/%d ok=%d empty=%d err=%d ins=%d %.0fs",
                        i + 1, len(tickers), ok, empty, err, ins, time.time() - t0)

    conn.close()
    logger.info("완료: ok=%d empty=%d err=%d insert=%d행 %.0fs",
                ok, empty, err, ins, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
