#!/usr/bin/env python
"""scripts/collect_investor_kis.py — KIS 종목별 투자자 '세분' 수급 수집 (KRX 완전 대체).

KIS REST API(FHPTJ04160001 · investor-trade-by-stock-daily)로 종목별 11주체의
일별 수급(매수/매도/순 × 거래량/금액)을 investor_daily.db에 적재한다.
2026-06-27 사장님 결정: "KRX 포기, KIS+Naver로만". KRX 계정잠금(CD007)을 영구 우회.

★ 11주체 (KIS 필드 → 한글 라벨):
   frgn→외국인, orgn→기관합계, prsn→개인,
   scrt→금융투자, insu→보험, ivtr→투신, pe_fund→사모, bank→은행,
   mrbn→기타금융, fund→연기금(KRX '연기금등'), etc_corp→기타법인
   (검증: scrt+insu+ivtr+pe_fund+bank+mrbn+fund == orgn 합산 무결 — 2026-06-27 실측)

★ 과거 오해 정정: 구버전 TR(FHKST01010900)은 외국인/기관계/개인 3주체만 줘서
   "KIS는 금투/연기금 미제공"으로 잘못 알았으나, FHPTJ04160001은 종목별로
   금융투자·연기금 세분을 모두 제공한다(한 호출에 output2로 30거래일치).

★ 금액 단위: KIS `_tr_pbmn`은 **백만원 해상도** 정수 → ×1_000_000으로 원 단위 저장하나
   백만원 미만은 0(절사). 억원 환산(/1e8) 소비처(signal_engine)에는 영향 미미.

★ 매매 무관·DB write 전용 (실주문 0·매매로직 무접촉·freeze 무손상).
   investor_daily.db는 단타봇 DB의 원본(단타봇이 symlink로 공유) → WAL 모드로 동시 읽기 보호.
   기존 3주체(외국인/기관합계/개인) 값은 동일(Naver/KRX 교차검증) → 단타봇 무손상.

스키마(investor_daily): date, ticker, name, investor,
  sell_vol, buy_vol, net_vol, sell_val, buy_val, net_val (금액=원, 백만원 해상도)

Usage:
  python -u -X utf8 scripts/collect_investor_kis.py                 # 최근 3거래일
  python -u -X utf8 scripts/collect_investor_kis.py --days 5        # 최근 5거래일
  python -u -X utf8 scripts/collect_investor_kis.py --days 30       # 한 호출분(최대 30) 전부
  python -u -X utf8 scripts/collect_investor_kis.py --dates 20260626 20260625  # 오늘 기준 최근 30거래일 내만 유효
  python -u -X utf8 scripts/collect_investor_kis.py --limit 10      # 앞 10종목(테스트)
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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

# 한글 라벨 → (KIS raw 필드 접두사, 순매수 수량 필드 suffix)
# 대부분 {p}_ntby_qty 이나 사모/기타법인은 {p}_ntby_vol — KIS 명세 차이 (실측 2026-06-27)
INVESTORS = [
    ("외국인", "frgn", "ntby_qty"),
    ("기관합계", "orgn", "ntby_qty"),
    ("개인", "prsn", "ntby_qty"),
    ("금융투자", "scrt", "ntby_qty"),
    ("보험", "insu", "ntby_qty"),
    ("투신", "ivtr", "ntby_qty"),
    ("사모", "pe_fund", "ntby_vol"),
    ("은행", "bank", "ntby_qty"),
    ("기타금융", "mrbn", "ntby_qty"),
    ("연기금", "fund", "ntby_qty"),
    ("기타법인", "etc_corp", "ntby_vol"),
]
INVESTOR_LABELS = [ko for ko, _, _ in INVESTORS]  # delete 조건과 insert 라벨 단일 출처 (M-1)
# 시장 공통 거래일 확정용 대표 대형주 (거래정지 대비 fallback 다단) — C-1
REF_TICKERS = ["005930", "000660", "005380", "035420", "051910"]
TR_ID = "FHPTJ04160001"
ENDPOINT = "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
KST = ZoneInfo("Asia/Seoul")


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


def fetch_raw(ticker: str, base_date: str, _retry: int = 0) -> list[dict]:
    """종목별 투자자 세분 매매동향 최대 30거래일치 raw(FHPTJ04160001 output2). 실패 시 [].

    base_date(YYYYMMDD) 기준 그 이하 최근 30거래일을 반환한다.
    토큰 발급 실패(_issue_token raise)는 TokenError로 승격해 호출부가 abort하게 한다 (H-2).
    """
    try:
        bearer = _issue_token()
    except Exception as e:  # noqa: BLE001 — 토큰 실패는 치명적이라 승격
        raise TokenError(str(e)) from e

    hdr = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {bearer}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": TR_ID,
        "custtype": "P",
    }
    try:
        r = requests.get(
            f"{KIS_BASE_URL}{ENDPOINT}",
            headers=hdr,
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": ticker,
                "FID_INPUT_DATE_1": base_date,
                "FID_ORG_ADJ_PRC": "0",
                "FID_ETC_CLS_CODE": "0",
            },
            timeout=15,
        )
        d = r.json()
    except Exception:  # noqa: BLE001 — 네트워크/파싱 실패는 종목 스킵
        return []
    if d.get("rt_cd") != "0":
        if "초당" in d.get("msg1", "") and _retry < 2:
            time.sleep(1.0 + _retry * 0.5)
            return fetch_raw(ticker, base_date, _retry + 1)
        return []
    return d.get("output2", []) or []


def resolve_target_dates(days: int, base_date: str) -> set[str]:
    """시장 공통 거래일 집합을 대표 대형주로 확정 (C-1 해결).

    첫 종목 하나에 의존하지 않고, 대표 대형주(거래정지 대비 다단 fallback)의
    실제 응답 거래일에서 최근 N개를 취한다. 모두 실패하면 빈 집합.
    """
    for ref in REF_TICKERS:
        out = fetch_raw(ref, base_date)
        dates = sorted({r.get("stck_bsop_date") for r in out if r.get("stck_bsop_date")},
                       reverse=True)
        if dates:
            return set(dates[:days])
    return set()


def rows_for_ticker(ticker: str, name: str, out: list[dict], target: set[str]) -> list[tuple]:
    """raw output2 → investor_daily 행(11주체 × 대상 거래일). tr_pbmn(백만원)→원."""
    recs = []
    for r in out:
        ds = r.get("stck_bsop_date")
        if ds not in target:
            continue
        for ko, p, netf in INVESTORS:
            recs.append((
                ds, ticker, name, ko,
                _i(r.get(f"{p}_seln_vol")),
                _i(r.get(f"{p}_shnu_vol")),
                _i(r.get(f"{p}_{netf}")),
                _i(r.get(f"{p}_seln_tr_pbmn")) * 1_000_000,
                _i(r.get(f"{p}_shnu_tr_pbmn")) * 1_000_000,
                _i(r.get(f"{p}_ntby_tr_pbmn")) * 1_000_000,
            ))
    return recs


def upsert(conn: sqlite3.Connection, ticker: str, target: set[str], recs: list[tuple]) -> None:
    """대상 거래일×11주체 기존 행 삭제 후 재적재 (재실행 안전).

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
    ap = argparse.ArgumentParser(description="KIS 종목별 투자자 세분(11주체) 수급 → investor_daily.db")
    ap.add_argument("--days", type=int, default=3, help="최근 N거래일 (기본 3, 최대 30, --dates 없을 때)")
    ap.add_argument("--dates", nargs="*", default=None, help="대상 거래일 명시 (YYYYMMDD ...)")
    ap.add_argument("--limit", type=int, default=0, help="앞 N종목만 (테스트용, 0=전체)")
    ap.add_argument("--db", type=str, default=str(DB_PATH), help="investor_daily.db 경로")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error("investor_daily.db 없음: %s", db_path)
        return 1

    # 조회 기준일 = 오늘(KST). KIS가 그 이하 최근 거래일을 반환 (주말/휴일도 안전)
    base_date = datetime.now(KST).strftime("%Y%m%d")

    # 거래일 확정 (대표 대형주 기반 — 종목 루프 전에 1회) — C-1
    if args.dates:
        target = set(args.dates)
        logger.info("대상 거래일(명시): %s", sorted(target))
    else:
        days = min(max(args.days, 1), 30)  # 한 호출 최대 30거래일
        target = resolve_target_dates(days, base_date)
        if not target:
            logger.error("거래일 확정 실패 (대표 종목 전부 빈 응답 — KIS 장애/휴장 의심). 중단.")
            return 1
        logger.info("대상 거래일(자동·대표종목, base=%s): %s", base_date, sorted(target))

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
        logger.info("대상 %d종목 — KIS 11주체 세분 수급 수집 (TR=%s)", len(tickers), TR_ID)

        t0 = time.time()
        for i, (tk, nm) in enumerate(tickers):
            try:
                out = fetch_raw(tk, base_date)
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
    logger.info("%s: ok=%d empty=%d err=%d insert=%d행 (%d주체)",
                status, ok, empty, err, ins, len(INVESTORS))

    # 무결성 자가검증: 대표종목 최신일 7세분합 == 기관합계 (KIS 필드명/명세 변경=침묵실패 감지)
    if not aborted and ins:
        try:
            c2 = sqlite3.connect(str(db_path), timeout=30)
            mx = c2.execute("select max(date) from investor_daily").fetchone()[0]
            chk = dict(c2.execute(
                "select investor, net_vol from investor_daily where ticker=? and date=?",
                ("005930", mx)).fetchall())
            c2.close()
            parts = ["금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금"]
            if all(p in chk for p in parts) and chk.get("기관합계"):
                s, orgn = sum(chk[p] for p in parts), chk["기관합계"]
                if s == orgn:
                    logger.info("무결성 OK: 005930 %s 7세분합=기관합계=%d주", mx, orgn)
                else:
                    logger.warning("⚠️ 무결성 불일치(KIS 필드 변경 의심): 005930 %s "
                                   "7세분합=%d ≠ 기관합계=%d", mx, s, orgn)
        except Exception as e:  # noqa: BLE001 — 검증 실패가 수집 성공을 무효화하지 않음
            logger.warning("무결성 체크 스킵: %s", e)

    return 1 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
