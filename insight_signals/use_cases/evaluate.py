# -*- coding: utf-8 -*-
"""성과 평가 — 관찰 픽이 실제로 벌었는지 검증.

v0.2.1 (7/10 전체검수 픽스):
- (구버전 결함) +N일 수익률을 '평가일(오늘) 가격'으로 계산해 오래된 픽이 영원히
  같은 버킷에서 재계산되는 드리프트가 있었음 → **horizon 도달일의 실제 종가로
  1회 계산해 원장(perf_ledger.csv)에 고정 저장**하는 방식으로 재설계.
- horizon은 달력일이 아니라 **거래일** 기준 (백테스트 backtest.py와 동일 잣대).
- 부수 픽스: 0바이트 picks_log 헤더 누락, 오염 행 1개로 전체 크래시,
  같은 날 재실행 시 picks_log 중복 append, 평가 API 호출 무한 성장
  (원장에 없는 미완결 픽만 시세 조회).
"""
from __future__ import annotations

import bisect
import csv
import datetime as dt
import logging
import os

log = logging.getLogger("insight_signals.evaluate")

PICKS_LOG_FIELDS = [
    "date", "stock_code", "stock_name", "combined_score",
    "sources", "price_at_pick",
]
LEDGER_FIELDS = [
    "date", "stock_code", "stock_name", "sources", "horizon",
    "price_at_pick", "price_h", "return_pct", "computed_on",
]

HORIZONS = (5, 10, 20)  # 거래일 기준


def _new_or_empty(path: str) -> bool:
    return (not os.path.exists(path)) or os.path.getsize(path) == 0


def append_picks(log_path: str, picks) -> None:
    """픽 로그 append. 같은 (date, stock_code)가 이미 있으면 스킵 (재실행 안전)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    existing = set()
    if not _new_or_empty(log_path):
        try:
            with open(log_path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    existing.add((row.get("date", ""), row.get("stock_code", "")))
        except Exception as e:  # noqa: BLE001 — 로그가 깨져도 신규 기록은 계속
            log.warning("picks_log 읽기 실패(중복검사 생략): %s", e)

    new_file = _new_or_empty(log_path)
    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=PICKS_LOG_FIELDS)
        if new_file:
            w.writeheader()
        for p in picks:
            if (p.date, p.stock_code) in existing:
                log.info("picks_log 중복 스킵: %s %s", p.date, p.stock_code)
                continue
            w.writerow(
                {
                    "date": p.date,
                    "stock_code": p.stock_code,
                    "stock_name": p.stock_name,
                    "combined_score": p.combined_score,
                    "sources": "|".join(p.sources),
                    "price_at_pick": p.price_at_pick or "",
                }
            )


def _load_ledger(ledger_path: str) -> list:
    if _new_or_empty(ledger_path):
        return []
    try:
        with open(ledger_path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception as e:  # noqa: BLE001
        log.warning("perf_ledger 읽기 실패(빈 원장으로 진행): %s", e)
        return []


def _append_ledger(ledger_path: str, rows: list) -> None:
    if not rows:
        return
    new_file = _new_or_empty(ledger_path)
    with open(ledger_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)


def evaluate(log_path: str, ledger_path: str, closes_fn, today: dt.date | None = None) -> dict:
    """미완결 (픽, horizon)만 계산해 원장에 고정 저장하고, 원장 전체로 요약.

    closes_fn(code) -> [(YYYYMMDD, close), ...] 오름차순 (예: history_client.daily_closes).
    Returns {"rows": 이번에 새로 계산된 행들, "summary": {h: {n, avg, median, win_rate}}}
    """
    today = today or dt.date.today()
    ledger = _load_ledger(ledger_path)
    done = {(r["date"], r["stock_code"], int(r["horizon"])) for r in ledger
            if str(r.get("horizon", "")).strip().isdigit()}

    picks = []
    if not _new_or_empty(log_path):
        try:
            with open(log_path, encoding="utf-8-sig") as f:
                picks = list(csv.DictReader(f))
        except Exception as e:  # noqa: BLE001
            log.warning("picks_log 읽기 실패(평가 생략): %s", e)

    closes_cache: dict = {}
    new_rows = []
    for p in picks:
        try:
            pick_date = p.get("date", "")
            code = p.get("stock_code", "")
            base = float(p.get("price_at_pick") or 0)
            if not pick_date or not code or base <= 0:
                continue
            pending = [h for h in HORIZONS if (pick_date, code, h) not in done]
            if not pending:
                continue
            if code not in closes_cache:
                closes_cache[code] = closes_fn(code) or []
            closes = closes_cache[code]
            if not closes:
                continue
            dates = [d for d, _ in closes]
            key = pick_date.replace("-", "")
            idx = bisect.bisect_right(dates, key)  # 픽일 다음 거래일부터 1..h
            for h in pending:
                j = idx + h - 1
                if j >= len(closes):
                    continue  # 아직 h거래일 미도달 — 다음 실행에서 재시도
                price_h = closes[j][1]
                new_rows.append({
                    "date": pick_date, "stock_code": code,
                    "stock_name": p.get("stock_name", ""),
                    "sources": p.get("sources", ""), "horizon": h,
                    "price_at_pick": base, "price_h": price_h,
                    "return_pct": round((price_h - base) / base * 100.0, 2),
                    "computed_on": today.isoformat(),
                })
        except Exception as e:  # noqa: BLE001 — 오염 행 1개가 전체를 멈추지 않게
            log.warning("픽 평가 스킵 (%s): %s", p, e)

    _append_ledger(ledger_path, new_rows)
    ledger += new_rows

    summary = {}
    for h in HORIZONS:
        vals = []
        for r in ledger:
            try:
                if int(r["horizon"]) == h:
                    vals.append(float(r["return_pct"]))
            except (KeyError, ValueError, TypeError):
                continue
        if vals:
            vals_sorted = sorted(vals)
            n = len(vals)
            median = vals_sorted[n // 2] if n % 2 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2
            summary[h] = {
                "n": n,
                "avg": round(sum(vals) / n, 2),
                "median": round(median, 2),
                "win_rate": round(100.0 * sum(1 for v in vals if v > 0) / n, 1),
            }
    return {"rows": new_rows, "summary": summary}
