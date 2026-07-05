"""공급계약 공시 이벤트 스터디 — 매출대비(%) 구간별 forward 수익률.

미래가치 엔진 v1 O축(수주 모멘텀) 가점 규칙의 백테스트 근거 산출.
- DART list API로 기간 내 '단일판매ㆍ공급계약체결' 원공시(정정·해지 제외) 전수 수집
- dart_contract_parser로 계약금액/매출대비 파싱
- 지표 3종 (7/4 적대검수 반영):
  raw    : 공시일 종가 → D+N 종가 (참고용 — 시장 드리프트 포함)
  excess : raw − 동일창 KOSPI (시장 보정 — 코드 내장 = 재현성)
  exec   : D+1 시가 매수 → D+N 종가 (★실행가능 수익 — 공시 시각 미상이라
           당일 종가 체결은 낙관 편향. BAT-D 16:35 태그 → 익일 시가가 실제 최선)
- 호라이즌별 유효 표본 n 별도 기록(창 끝 D+20 절단로 인한 표본 불일치 명시)

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
            if d.get("status") != "013":  # 013=조회 결과 없음(정상)
                logger.warning("list API 비정상 status=%s msg=%s (p%d)",
                               d.get("status"), d.get("message"), page)
            break
        for x in d.get("list", []):
            nm = x.get("report_nm", "")
            if ("공급계약" in nm and "기재정정" not in nm and "해지" not in nm
                    and x.get("stock_code")):
                events.append({"ticker": x["stock_code"], "name": x["corp_name"],
                               "date": x["rcept_dt"], "rcept_no": x["rcept_no"],
                               "market": x.get("corp_cls", ""),
                               "subsidiary": "종속회사" in nm})
        total_page = int(d.get("total_page", 1))
        if page >= total_page:
            break
        page += 1
        time.sleep(0.12)
    return events


_price_cache: dict[str, pd.DataFrame | None] = {}
_kospi_close: pd.Series | None = None


def _load_prices(ticker: str) -> pd.DataFrame | None:
    """종목 close(+open) — 인덱스 date. open 없으면 컬럼 미포함."""
    if ticker in _price_cache:
        return _price_cache[ticker]
    df = None
    try:
        hits = globmod.glob(str(CSV_DIR / f"*_{ticker}.csv"))
        if hits:
            raw = pd.read_csv(hits[0], parse_dates=[0])
            date_col = raw.columns[0]
            raw = raw.sort_values(date_col)
            cols = {c.lower(): c for c in raw.columns}
            data = {"close": raw[cols.get("close", raw.columns[1])].values}
            if "open" in cols:
                data["open"] = raw[cols["open"]].values
            df = pd.DataFrame(data, index=raw[date_col])
    except Exception:
        df = None
    _price_cache[ticker] = df
    return df


def _kospi() -> pd.Series | None:
    global _kospi_close
    if _kospi_close is None:
        try:
            k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv",
                            parse_dates=["Date"]).sort_values("Date")
            _kospi_close = k.set_index("Date")["close"]
        except Exception:
            _kospi_close = pd.Series(dtype=float)
    return _kospi_close if len(_kospi_close) else None


def _anchor_idx(index: pd.Index, date_compact: str) -> int:
    d = pd.Timestamp(f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:]}")
    return index.searchsorted(d, side="right") - 1


def fwd_metrics(ticker: str, date_compact: str, n: int) -> dict[str, float | None]:
    """raw(공시일 종가→D+N 종가) / excess(raw−KOSPI) / exec(D+1 시가→D+N 종가)."""
    out: dict[str, float | None] = {"raw": None, "excess": None, "exec": None}
    px = _load_prices(ticker)
    if px is None or px.empty:
        return out
    i = _anchor_idx(px.index, date_compact)
    if i < 0 or i + n >= len(px):
        return out
    base = px["close"].iloc[i]
    if not base or base <= 0:
        return out
    out["raw"] = float((px["close"].iloc[i + n] / base - 1) * 100)
    kospi = _kospi()
    if kospi is not None:
        ki = _anchor_idx(kospi.index, date_compact)
        if ki >= 0 and ki + n < len(kospi):
            out["excess"] = out["raw"] - float(
                (kospi.iloc[ki + n] / kospi.iloc[ki] - 1) * 100)
    # 실행가능: D+1 시가 진입 (BAT-D 16:35 태그 → 익일 시가가 실제 최선)
    if "open" in px.columns and i + 1 < len(px) and i + n < len(px):
        entry = px["open"].iloc[i + 1]
        if entry and entry > 0:
            out["exec"] = float((px["close"].iloc[i + n] / entry - 1) * 100)
    return out


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

    # forward 수익률 결합 (raw / excess=KOSPI보정 / exec=D+1시가 진입)
    rows = []
    for p in parsed:
        row = dict(p)
        for n in (1, 5, 20):
            m = fwd_metrics(p["ticker"], p["date"], n)
            row[f"raw{n}"], row[f"ex{n}"], row[f"exec{n}"] = m["raw"], m["excess"], m["exec"]
        rows.append(row)

    # 구간별 × 지표별 통계 (호라이즌별 유효 n 명시 — D+20은 창 끝 절단으로 표본 다름)
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "period": f"{bgn.date()}~{end.date()}", "n_events": len(events),
              "n_parsed": len(parsed), "buckets": []}
    print(f"\n══ 공급계약 이벤트 스터디 ({bgn.date()}~{end.date()}, "
          f"원공시 {len(events)}건 · 매출대비 파싱 {len(parsed)}건) ══")
    METRICS = [("raw", "raw(공시일종가)"), ("ex", "KOSPI보정"), ("exec", "★실행가능(D+1시가진입)")]
    for lo, hi, label in RATIO_BUCKETS:
        grp = [r for r in rows if lo <= (r.get("revenue_ratio_pct") or 0) < hi]
        stat = {"bucket": label, "n": len(grp)}
        print(f"  매출대비 {label:<7} 이벤트 {len(grp)}건")
        for mkey, mlabel in METRICS:
            line = f"    {mlabel:<22}"
            for n in (1, 5, 20):
                vals = pd.Series([r[f"{mkey}{n}"] for r in grp
                                  if r.get(f"{mkey}{n}") is not None])
                stat[f"{mkey}_d{n}_n"] = len(vals)
                if len(vals):
                    stat[f"{mkey}_d{n}_mean"] = round(float(vals.mean()), 2)
                    stat[f"{mkey}_d{n}_win"] = round(float((vals > 0).mean() * 100), 1)
                    stat[f"{mkey}_d{n}_median"] = round(float(vals.median()), 2)
                    line += (f"| D+{n} {vals.mean():+.2f}% 승{(vals > 0).mean()*100:.0f}%"
                             f" (n={len(vals)}) ")
            print(line)
        report["buckets"].append(stat)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report["events"] = rows  # 원자료 보존(재현성)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n저장: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
