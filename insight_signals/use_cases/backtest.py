# -*- coding: utf-8 -*-
"""백테스트 순수 로직 — 이벤트 수익률 vs 비이벤트 기저선.

기저선 원칙 (퀀트봇 검수 반영):
  이벤트(임원 매수 공시) 수익률을 '페이퍼 평균'이 아니라
  **같은 날 유니버스 무작위 진입** 수익률과 비교한다.
  이래야 시장 전체가 오른 장(레짐 베타)에 속지 않는다.

진입 규칙: 공시일(T) 다음 거래일(T+1) 종가 진입 — 선견 편향 방지.
수익률: 진입 후 +5/+10/+20 '거래일' 종가 기준.
"""
from __future__ import annotations

import bisect
import statistics
from collections import defaultdict

HORIZONS = (5, 10, 20)


def horizon_returns(closes, event_date: str, horizons=HORIZONS):
    """closes: [(YYYYMMDD, close)] 오름차순. event_date 다음 거래일 진입.

    Returns {"entry_date":, "entry_price":, "returns": {h: pct|None}} 또는 None.
    """
    if not closes:
        return None
    dates = [d for d, _ in closes]
    idx = bisect.bisect_right(dates, event_date)  # 첫 '초과' 거래일 = T+1
    if idx >= len(closes):
        return None
    entry_date, entry = closes[idx]
    if entry <= 0:
        return None
    rets = {}
    for h in horizons:
        j = idx + h
        rets[h] = (
            round((closes[j][1] - entry) / entry * 100.0, 2) if j < len(closes) else None
        )
    return {"entry_date": entry_date, "entry_price": entry, "returns": rets}


def summarize(rows, horizons=HORIZONS) -> dict:
    """rows: [{"returns": {h: pct|None}}] -> 구간별 n/평균/중앙값/승률."""
    out = {}
    for h in horizons:
        vals = [r["returns"][h] for r in rows if r.get("returns", {}).get(h) is not None]
        if vals:
            out[h] = {
                "n": len(vals),
                "avg": round(sum(vals) / len(vals), 2),
                "median": round(statistics.median(vals), 2),
                "win_rate": round(100.0 * sum(1 for v in vals if v > 0) / len(vals), 1),
            }
    return out


def compare(event_summary: dict, baseline_summary: dict, horizons=HORIZONS) -> dict:
    """이벤트 - 기저선 초과 성과 (알파 추정)."""
    out = {}
    for h in horizons:
        e, b = event_summary.get(h), baseline_summary.get(h)
        if e and b:
            out[h] = {
                "excess_avg": round(e["avg"] - b["avg"], 2),
                "excess_median": round(e["median"] - b["median"], 2),
                "excess_win_rate": round(e["win_rate"] - b["win_rate"], 1),
            }
    return out


def build_events(filings) -> list:
    """InsiderFiling 리스트 -> 순매수 이벤트 [(stock_code, corp_name, rcept_dt, net_qty)].

    같은 종목·같은 공시일의 보고를 합산, 순매수(>0)만 이벤트로 인정.
    (스톡옵션 행사 등 혼입은 v0.2에서도 한계 — 문서에 고지)
    """
    agg = defaultdict(int)
    names = {}
    for f in filings:
        if not f.stock_code:
            continue
        key = (f.stock_code, f.rcept_dt)
        agg[key] += f.change_qty
        names[f.stock_code] = f.corp_name
    return [
        (code, names.get(code, ""), dt, qty)
        for (code, dt), qty in sorted(agg.items(), key=lambda x: x[0][1])
        if qty > 0
    ]
