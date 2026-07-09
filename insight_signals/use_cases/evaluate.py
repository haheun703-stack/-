# -*- coding: utf-8 -*-
"""성과 평가 — 관찰 픽이 실제로 벌었는지 검증.

picks_log.csv에 쌓인 픽들에 대해 +5/+10/+20 (달력)일 경과 시점의
수익률을 계산한다. 이 숫자가 '스코어링 승격' 판단의 근거가 된다.
"""
from __future__ import annotations

import csv
import datetime as dt
import os

PICKS_LOG_FIELDS = [
    "date", "stock_code", "stock_name", "combined_score",
    "sources", "price_at_pick",
]
PERF_FIELDS = PICKS_LOG_FIELDS + ["eval_date", "days_elapsed", "price_now", "return_pct"]

HORIZONS = (5, 10, 20)  # 달력일 기준 (근사)


def append_picks(log_path: str, picks) -> None:
    new_file = not os.path.exists(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=PICKS_LOG_FIELDS)
        if new_file:
            w.writeheader()
        for p in picks:
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


def evaluate(log_path: str, price_fn, today: dt.date | None = None) -> dict:
    """픽 로그 전체를 평가. price_fn(code) -> float|None.

    Returns {"rows": [...], "summary": {horizon: {"n":, "avg":, "win_rate":}}}
    """
    today = today or dt.date.today()
    if not os.path.exists(log_path):
        return {"rows": [], "summary": {}}

    with open(log_path, encoding="utf-8-sig") as f:
        picks = list(csv.DictReader(f))

    rows = []
    price_cache: dict = {}
    for p in picks:
        try:
            base = float(p.get("price_at_pick") or 0)
        except ValueError:
            base = 0
        if not base:
            continue
        pick_date = dt.date.fromisoformat(p["date"])
        elapsed = (today - pick_date).days
        # 도달한 가장 긴 구간으로 평가
        horizon = max((h for h in HORIZONS if elapsed >= h), default=None)
        if horizon is None:
            continue
        code = p["stock_code"]
        if code not in price_cache:
            price_cache[code] = price_fn(code)
        now = price_cache[code]
        if not now:
            continue
        ret = (now - base) / base * 100.0
        rows.append(
            {
                **p,
                "eval_date": today.isoformat(),
                "days_elapsed": elapsed,
                "horizon": horizon,
                "price_now": now,
                "return_pct": round(ret, 2),
            }
        )

    summary = {}
    for h in HORIZONS:
        rs = [r["return_pct"] for r in rows if r["horizon"] == h]
        if rs:
            summary[h] = {
                "n": len(rs),
                "avg": round(sum(rs) / len(rs), 2),
                "win_rate": round(100.0 * sum(1 for x in rs if x > 0) / len(rs), 1),
            }
    return {"rows": rows, "summary": summary}
