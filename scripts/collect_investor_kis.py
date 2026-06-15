#!/usr/bin/env python
"""scripts/collect_investor_kis.py — KIS 종목별 투자자 수급 수집 (KRX 대체).

KIS REST API(FHKST01010900)로 종목별 외국인/기관합계/개인 3주체의 일별 수급
(매수/매도/순 × 거래량/금액)을 investor_daily.db에 적재한다. KRX 계정 잠금/만료를
우회하는 정공 경로(2026-06-15 사장님 결정: "KRX 말고 KIS로").

★ KIS 종목별 한계: 외국인/기관계/개인 3주체만 제공. 기타법인·세분(금융투자/연기금)은
  KIS 종목별 미제공(키움 opt10059 별도 계좌 필요). 시장 전체 11주체 세분은 FHPTJ04040000.

★ 금액 단위: KIS `_tr_pbmn`은 **백만원 해상도** 정수 → ×1_000_000으로 원 단위 저장하나
  백만원 미만은 0(절사). 어댑터(kis_investor_adapter.py)와 동일 정합. 정밀 비교 주의.

★ 매매 무관·DB write 전용 (실주문 0·매매로직 무접촉·freeze 무손상).
  investor_daily.db는 단타봇 DB의 원본(단타봇이 symlink로 공유) → WAL 모드로 동시 읽기 보호.

스키마(investor_daily): date, ticker, name, investor,
  sell_vol, buy_vol, net_vol, sell_val, buy_val, net_val (금액=원, 백만원 해상도)

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
INVESTOR_LABELS = [ko for ko, _ in INVESTORS]  # delete 조건과 insert 라벨 단일 출처 (M-1)
# 시장 공통 거래일 확정용 대표 대형주 (거래정지 대비 fallback 다단) — C-1
REF_TICKERS = ["005930", "000660", "005380", "035420", "051910"]
TR_ID = "FHKST01010900"


class TokenError(RuntimeError):
    """토큰 발급 실패 — 루프 중단 트리거 (H-2)."""


def _i(v) -> int:
    """KIS 금액/수량 문자열 → int. 콤마/실수 문자열도 견고 처리 (M-4)."""
    if v is None or v == "":
        return 0
    try:
        return int(v)
    except (ValueError, TypeError):
        try:
            return int(float(str(v).replace(",", "")))
        except (ValueError, TypeError):
            return 0


def fetch_raw(ticker: str, _retry: int = 0) -> list[dict]:
    """종목별 투자자 매매동향 30일치 raw(FHKST01010900). 실패 시 [].

    토큰 발급 실패(_issue_token raise)는 TokenError로 승격해 호출부가 abort하게 한다 (H-2).
    """
    try:
        bearer = _issue_token()
    except Exception as e:  # noqa: BLE001 — 토큰 실패는 치명적이라 승격
        raise TokenError(str(e)) from e

    hdr = {
        "content-type": "application/json",
        "authorization": f"Bearer {bearer}",
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
    except Exception:  # noqa: BLE001 — 네트워크/파싱 실패는 종목 스킵
        return []
    if d.get("rt_cd") != "0":
        if "초당" in d.get("msg1", "") and _retry < 2:
            time.sleep(1.0 + _retry * 0.5)
            return fetch_raw(ticker, _retry + 1)
        return []
    return d.get("output", []) or []


def resolve_target_dates(days: int) -> set[str]:
    """시장 공통 거래일 집합을 대표 대형주로 확정 (C-1 해결).

    첫 종목 하나에 의존하지 않고, 대표 대형주(거래정지 대비 다단 fallback)의
    실제 응답 거래일에서 최근 N개를 취한다. 모두 실패하면 빈 집합.
    """
    for ref in REF_TICKERS:
        out = fetch_raw(ref)
        dates = sorted({r.get("stck_bsop_date") for r in out if r.get("stck_bsop_date")},
                       reverse=True)
        if dates:
            return set(dates[:days])
    return set()


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
    """대상 거래일×3주체 기존 행 삭제 후 재적재 (재실행 안전).

    delete의 investor 목록은 INVESTOR_LABELS에서 파생 — insert 라벨과 단일 출처 (M-1).
    """
    date_qs = ",".join("?" * len(target))
    inv_qs = ",".join("?" * len(INVESTOR_LABELS))
    conn.execute(
        f"delete from investor_daily where ticker=? and date in ({date_qs}) "
        f"and investor in ({inv_qs})",
        (ticker, *sorted(target), *INVESTOR_LABELS),
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

    # 거래일 확정 (대표 대형주 기반 — 종목 루프 전에 1회) — C-1
    if args.dates:
        target = set(args.dates)
        logger.info("대상 거래일(명시): %s", sorted(target))
    else:
        target = resolve_target_dates(args.days)
        if not target:
            logger.error("거래일 확정 실패 (대표 종목 전부 빈 응답 — KIS 장애/휴장 의심). 중단.")
            return 1
        logger.info("대상 거래일(자동·대표종목): %s", sorted(target))

    conn = sqlite3.connect(str(db_path), timeout=60)
    # 단타봇 symlink 공유 DB — WAL로 동시 읽기 보호 (H-3)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    ok = empty = err = ins = 0
    matched_dates: set[str] = set()  # 실제 적재된 거래일 (M-2 검증용)
    aborted = False
    try:
        tickers = [(t, n) for t, n in
                   conn.execute("select distinct ticker, name from investor_daily").fetchall()]
        if args.limit:
            tickers = tickers[:args.limit]
        logger.info("대상 %d종목 — KIS 3주체 수급 수집", len(tickers))

        t0 = time.time()
        for i, (tk, nm) in enumerate(tickers):
            try:
                out = fetch_raw(tk)
                recs = rows_for_ticker(tk, nm, out, target)
                if recs:
                    upsert(conn, tk, target, recs)
                    ins += len(recs); ok += 1
                    matched_dates.update(r[0] for r in recs)
                else:
                    empty += 1
                time.sleep(0.22)  # KIS rate limit 여유
            except TokenError as e:
                logger.error("토큰 발급 실패 — 루프 중단(부분 적재 방지): %s", e)
                aborted = True
                break
            except Exception as e:  # noqa: BLE001
                err += 1
                if err <= 5:
                    logger.warning("  ERR %s: %s", tk, str(e)[:60])
            if (i + 1) % 200 == 0:
                logger.info("  %d/%d ok=%d empty=%d err=%d ins=%d %.0fs",
                            i + 1, len(tickers), ok, empty, err, ins, time.time() - t0)
    finally:
        conn.close()  # 예외 경로에도 보장 (M-3)

    # 명시/자동 거래일 중 어떤 raw에도 안 나타난 날짜 경고 (M-2: 오타·휴장 무방비 해소)
    missing = sorted(target - matched_dates)
    if missing:
        logger.warning("⚠️ 대상 거래일 중 미적재(휴장/미래/오타 의심): %s", missing)

    status = "중단(토큰실패)" if aborted else "완료"
    logger.info("%s: ok=%d empty=%d err=%d insert=%d행", status, ok, empty, err, ins)
    return 1 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
