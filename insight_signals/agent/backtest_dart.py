# -*- coding: utf-8 -*-
"""DART D002 (임원·주요주주 매수) 과거 백테스트 — 진입점.

30픽을 기다릴 필요 없이 과거 공시로 즉시 검증한다 (백테스트 우선 원칙).

실행:
    python -u -X utf8 -m insight_signals.agent.backtest_dart --months 6
옵션:
    --months N            조회 기간 (기본 6개월)
    --baseline-sample K   이벤트당 무작위 기저선 종목 수 (기본 5)
    --max-corps N         상세 조회 회사 수 상한 (0=무제한, 기본 0)
    --seed N              기저선 샘플링 시드 (재현용, 기본 42)

산출물:
    data/insight_signals/backtest_dart_YYYYMMDD.csv      이벤트별 수익률
    data/insight_signals/backtest_dart_YYYYMMDD_summary.md
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import random
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from insight_signals.adapters import history_client          # noqa: E402
from insight_signals.adapters.dart_client import DartClient  # noqa: E402
from insight_signals.agent import _env                       # noqa: E402
from insight_signals.use_cases import backtest               # noqa: E402

log = logging.getLogger("insight_signals.backtest")

FCHART_SLEEP = 0.2   # 네이버 fchart 예의상 간격
DART_CHUNK_DAYS = 85  # list.json 조회 기간 분할


def _chunks(bgn: dt.date, end: dt.date):
    cur = bgn
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=DART_CHUNK_DAYS), end)
        yield cur.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")
        cur = nxt + dt.timedelta(days=1)


class PriceCache:
    def __init__(self, count: int):
        self.count = count
        self._cache: dict = {}

    def closes(self, code: str):
        if code not in self._cache:
            self._cache[code] = history_client.daily_closes(code, count=self.count)
            time.sleep(FCHART_SLEEP)
        return self._cache[code]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--baseline-sample", type=int, default=5)
    parser.add_argument("--max-corps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = _env.project_root()
    _env.setup_logging(root, "insight_signals_backtest")
    _env.load_dotenv_manual(root)
    cfg = _env.load_config(root)
    data_dir = os.path.join(root, cfg["paths"]["data_dir"])
    os.makedirs(data_dir, exist_ok=True)

    dart_key = os.environ.get(cfg["env_names"]["dart_key"], "")
    if not dart_key:
        log.error("%s 미설정 — 백테스트는 DART 키가 필요합니다", cfg["env_names"]["dart_key"])
        return 1
    dart = DartClient(dart_key, cache_dir=data_dir)

    end = dt.date.today()
    bgn = end - dt.timedelta(days=args.months * 30)
    log.info("D002 백테스트: %s ~ %s", bgn, end)

    # 1) 기간 내 D002 공시 회사 수집
    corps = []
    for b, e in _chunks(bgn, end):
        corps += dart.insider_filing_corps(b, e)
    corp_codes = sorted({c["corp_code"] for c in corps if c.get("stock_code")})
    if args.max_corps:
        corp_codes = corp_codes[: args.max_corps]
    log.info("상장사 %d곳 상세 조회 시작 (약 %.1f분 예상)", len(corp_codes), len(corp_codes) * 0.3 / 60)

    # 2) 회사별 증감 상세 -> 순매수 이벤트
    filings = []
    since = bgn.strftime("%Y%m%d")
    for i, cc in enumerate(corp_codes, 1):
        try:
            filings += dart.insider_details(cc, since_yyyymmdd=since)
        except Exception as e:  # noqa: BLE001
            log.warning("상세 조회 실패 [%s]: %s", cc, e)
        if i % 100 == 0:
            log.info("  ... %d/%d", i, len(corp_codes))
        time.sleep(0.15)
    events = backtest.build_events(filings)
    log.info("순매수 이벤트 %d건", len(events))
    if not events:
        log.error("이벤트가 없습니다 — 기간을 늘려보세요")
        return 1

    # 3) 이벤트 수익률
    need_days = args.months * 22 + max(backtest.HORIZONS) + 40
    prices = PriceCache(count=need_days)
    ev_rows, skipped = [], 0
    for code, name, edate, qty in events:
        r = backtest.horizon_returns(prices.closes(code), edate)
        if r is None:
            skipped += 1
            continue
        ev_rows.append(
            {"kind": "event", "stock_code": code, "stock_name": name,
             "event_date": edate, "net_qty": qty, **r}
        )
    log.info("이벤트 수익률 계산 %d건 (시세 없음 스킵 %d건)", len(ev_rows), skipped)

    # 4) 기저선: 같은 날 유니버스 무작위 진입 (비이벤트 기저선)
    rng = random.Random(args.seed)
    universe = sorted(set(dart.listed_name_map().values()))
    event_codes = {c for c, _, _, _ in events}
    pool = [c for c in universe if c not in event_codes]
    # 요청량 제한: 기저선 후보를 미리 최대 400종목으로 압축해 시세 캐시 재사용
    pool = rng.sample(pool, min(400, len(pool)))
    base_rows = []
    for code, name, edate, _ in events:
        picked = rng.sample(pool, min(args.baseline_sample, len(pool)))
        for bc in picked:
            r = backtest.horizon_returns(prices.closes(bc), edate)
            if r is not None:
                base_rows.append(
                    {"kind": "baseline", "stock_code": bc, "stock_name": "",
                     "event_date": edate, "net_qty": 0, **r}
                )
    log.info("기저선 표본 %d건", len(base_rows))

    # 5) 요약 + 저장
    ev_sum = backtest.summarize(ev_rows)
    base_sum = backtest.summarize(base_rows)
    excess = backtest.compare(ev_sum, base_sum)

    today = end.isoformat()
    csv_path = os.path.join(data_dir, f"backtest_dart_{today}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["kind", "stock_code", "stock_name", "event_date", "net_qty",
                    "entry_date", "entry_price"] + [f"ret_{h}d" for h in backtest.HORIZONS])
        for r in ev_rows + base_rows:
            w.writerow([r["kind"], r["stock_code"], r["stock_name"], r["event_date"],
                        r["net_qty"], r["entry_date"], r["entry_price"]]
                       + [r["returns"].get(h, "") for h in backtest.HORIZONS])

    md = [f"# D002 임원 매수 백테스트 — {bgn} ~ {end}", "",
          f"- 이벤트 {len(ev_rows)}건 / 기저선 {len(base_rows)}건 "
          f"(이벤트당 무작위 {args.baseline_sample}종목, seed={args.seed})",
          "- 진입: 공시일 T+1 종가 / 수익률: 거래일 기준", "",
          "| 구간 | 이벤트 평균 | 기저선 평균 | 초과 | 이벤트 승률 | 기저선 승률 | 이벤트 중앙값 |",
          "|---|---|---|---|---|---|---|"]
    for h in backtest.HORIZONS:
        e, b, x = ev_sum.get(h), base_sum.get(h), excess.get(h)
        if e and b and x:
            md.append(f"| +{h}일 | {e['avg']:+.2f}% | {b['avg']:+.2f}% | "
                      f"**{x['excess_avg']:+.2f}%p** | {e['win_rate']}% | "
                      f"{b['win_rate']}% | {e['median']:+.2f}% |")
    md += ["", "## 판정 가이드",
           "- 초과 평균과 초과 중앙값이 **모두 양수**이고 승률 차이가 +5%p 이상이면 승격 후보",
           "- 한 구간만 좋으면 보류 — 세 구간 방향이 일치해야 신뢰",
           "- 이벤트 n < 30이면 기간을 늘려 재실행 (--months 12)"]
    md_path = os.path.join(data_dir, f"backtest_dart_{today}_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    log.info("완료 — %s / %s", csv_path, md_path)
    for h in backtest.HORIZONS:
        if h in excess:
            log.info("  +%d일 초과성과: %+.2f%%p (이벤트 %d건)",
                     h, excess[h]["excess_avg"], ev_sum[h]["n"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
