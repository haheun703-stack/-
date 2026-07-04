"""공급계약 공시 이벤트 스터디 — 매출대비(%) 구간별 forward 수익률.

미래가치 엔진 v1 O축(수주 모멘텀) 가점 규칙의 백테스트 근거 산출.
- DART list API로 기간 내 '단일판매ㆍ공급계약체결' 원공시(정정 제외) 전수 수집
- dart_contract_parser로 계약금액/매출대비 파싱
- 공시일 종가 기준 D+1/D+5/D+20 forward (stock_data_daily CSV, 전종목 커버)
- 결과: 매출대비 구간별 평균/승률 → 가점 규칙 결정

사용: ./venv/bin/python3.11 scripts/research/contract_event_study.py --months 3
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.adapters.dart_contract_parser import fetch_contract_detail  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("contract_event_study")

LIST_URL = "https://opendart.fss.or.kr/api/list.json"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
OUT_PATH = PROJECT_ROOT / "data" / "research" / "contract_event_study.json"

# 매출대비(%) 구간 — 분위 대신 해석 가능한 고정 구간(튜닝 아님, 보고용 절단점)
RATIO_BUCKETS = [(0, 5, "<5%"), (5, 15, "5~15%"), (15, 50, "15~50%"), (50, 1e9, "50%+")]


def collect_contract_events(api_key: str, bgn: str, end: str) -> list[dict]:
    """기간 내 공급계약 원공시(정정 제외) 목록."""
    events, page = [], 1
    while True:
        try:
            r = requests.get(LIST_URL, params={
                "crtfc_key": api_key, "bgn_de": bgn, "end_de": end,
                "pblntf_ty": "I", "page_no": page, "page_count": 100}, timeout=30)
            d = r.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("list p%d 실패: %s", page, e)
            break
        if d.get("status") != "000":
            break
        for x in d.get("list", []):
            nm = x.get("report_nm", "")
            if "공급계약" in nm and "기재정정" not in nm and x.get("stock_code"):
                events.append({"ticker": x["stock_code"], "name": x["corp_name"],
                               "date": x["rcept_dt"], "rcept_no": x["rcept_no"],
                               "market": x.get("corp_cls", "")})
        total_page = int(d.get("total_page", 1))
        if page >= total_page:
            break
        page += 1
        time.sleep(0.12)
    return events


_price_cache: dict[str, pd.Series | None] = {}


def _load_close(ticker: str) -> pd.Series | None:
    if ticker in _price_cache:
        return _price_cache[ticker]
    ser = None
    try:
        hits = globmod.glob(str(CSV_DIR / f"*_{ticker}.csv"))
        if hits:
            df = pd.read_csv(hits[0], parse_dates=[0])
            df = df.sort_values(df.columns[0])
            col = "Close" if "Close" in df.columns else "close"
            ser = pd.Series(df[col].values, index=df[df.columns[0]])
    except Exception:
        ser = None
    _price_cache[ticker] = ser
    return ser


def fwd_return(ticker: str, date_compact: str, n: int) -> float | None:
    close = _load_close(ticker)
    if close is None or close.empty:
        return None
    d = pd.Timestamp(f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:]}")
    i = close.index.searchsorted(d, side="right") - 1
    if i < 0 or i + n >= len(close):
        return None
    base = close.iloc[i]
    if not base or base <= 0:
        return None
    return float((close.iloc[i + n] / base - 1) * 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--max-parse", type=int, default=800, help="원문 파싱 상한(부하 가드)")
    args = ap.parse_args()

    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        logger.warning("DART_API_KEY 없음 — 종료(graceful)")
        return 0

    end = datetime.now()
    bgn = end - timedelta(days=args.months * 30)
    # list API 안정성 위해 2주 단위 분할 수집
    events: list[dict] = []
    cursor = bgn
    while cursor < end:
        seg_end = min(cursor + timedelta(days=13), end)
        seg = collect_contract_events(api_key, cursor.strftime("%Y%m%d"),
                                      seg_end.strftime("%Y%m%d"))
        events.extend(seg)
        logger.info("수집 %s~%s: %d건 (누적 %d)", cursor.date(), seg_end.date(),
                    len(seg), len(events))
        cursor = seg_end + timedelta(days=1)

    # 원문 파싱
    parsed = []
    sess = requests.Session()
    for i, ev in enumerate(events[:args.max_parse]):
        det = fetch_contract_detail(ev["rcept_no"], api_key, session=sess)
        if det and det.get("revenue_ratio_pct") is not None:
            parsed.append({**ev, **det})
        if (i + 1) % 50 == 0:
            logger.info("파싱 %d/%d (유효 %d)", i + 1, len(events), len(parsed))
        time.sleep(0.12)
    logger.info("파싱 완료: %d/%d 유효(매출대비 확보)", len(parsed), len(events))

    # forward 수익률 결합
    rows = []
    for p in parsed:
        row = dict(p)
        for n in (1, 5, 20):
            row[f"fwd{n}"] = fwd_return(p["ticker"], p["date"], n)
        rows.append(row)

    # 구간별 통계
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "period": f"{bgn.date()}~{end.date()}", "n_events": len(events),
              "n_parsed": len(parsed), "buckets": []}
    print(f"\n══ 공급계약 이벤트 스터디 ({bgn.date()}~{end.date()}, "
          f"원공시 {len(events)}건 · 매출대비 파싱 {len(parsed)}건) ══")
    for lo, hi, label in RATIO_BUCKETS:
        grp = [r for r in rows if lo <= (r.get("revenue_ratio_pct") or 0) < hi]
        stat = {"bucket": label, "n": len(grp)}
        line = f"  매출대비 {label:<7} n={len(grp):<4}"
        for n in (1, 5, 20):
            vals = pd.Series([r[f"fwd{n}"] for r in grp if r.get(f"fwd{n}") is not None])
            if len(vals):
                stat[f"d{n}_mean"] = round(float(vals.mean()), 2)
                stat[f"d{n}_win"] = round(float((vals > 0).mean() * 100), 1)
                stat[f"d{n}_median"] = round(float(vals.median()), 2)
                line += (f" | D+{n} {vals.mean():+.2f}% 승률{(vals > 0).mean()*100:.0f}%"
                         f" 중앙{vals.median():+.2f}%")
        report["buckets"].append(stat)
        print(line)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report["events"] = rows  # 원자료 보존(재현성)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n저장: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
