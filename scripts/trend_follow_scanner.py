#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""추세추종 스캐너 (Trend-Following Scanner).

풀백(눌림목) 스캐너가 구조적으로 못 잡는 '안 떨어지고 오르는 강추세주'를 발굴한다.
사장님 3원칙(2026-06-22): 5일선 위 · 정배열 · 신고가 · 5일 모멘텀.

★freeze 양립: 신호 생성/관측 전용. 매매 경로 배선 0. 실주문과 무관.
  (signal_engine과 동일 — 자동매매 OFF, 신호만 생성/저장)
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

PROCESSED_DIR = "data/processed"
INVESTOR_DB = "data/investor_flow/investor_daily.db"
OUT_JSON = "data/trend_following_signals.json"

# --- 발굴 기준 (사장님 3원칙) ---
MOM5_MIN = 8.0        # 5일 모멘텀 최소 +8%
HIGH_LOOKBACK = 60    # 신고가 판정 기간(거래일)
HIGH_RATIO = 0.98     # 60일 고가의 98% 이상 = 신고가권

# --- 과열 이상치 분리 (추격 위험 → 제외 아닌 '과열' 태그) ---
OVERHEAT_MOM5 = 60.0  # 5일 +60% 초과
OVERHEAT_VOLR = 10.0  # 거래량 20일평균 10배 초과
OVERHEAT_RSI = 85.0


def load_names() -> dict:
    """ticker -> 종목명 (investor_daily.db)."""
    try:
        con = sqlite3.connect(INVESTOR_DB)
        rows = con.execute(
            "SELECT DISTINCT ticker, name FROM investor_daily"
        ).fetchall()
        con.close()
        return {str(t): n for t, n in rows if n}
    except Exception:
        return {}


def load_supply(date_compact: str) -> dict:
    """해당일 외국인/기관 순매수액(억원)."""
    out: dict = {}
    if not date_compact:
        return out
    try:
        con = sqlite3.connect(INVESTOR_DB)
        for tk, inv, nv in con.execute(
            "SELECT ticker, investor, net_val FROM investor_daily WHERE date=?",
            [date_compact],
        ).fetchall():
            d = out.setdefault(str(tk), {})
            if inv == "외국인":
                d["foreign"] = (nv or 0) / 1e8
            elif inv == "기관합계":
                d["inst"] = (nv or 0) / 1e8
        con.close()
    except Exception:
        pass
    return out


def scan() -> tuple[str | None, list]:
    names = load_names()
    latest: str | None = None
    hits: list = []
    for path in glob.glob(os.path.join(PROCESSED_DIR, "*.parquet")):
        tk = os.path.basename(path)[:-8]
        try:
            df = pd.read_parquet(path)
            if len(df) < HIGH_LOOKBACK + 1:
                continue
            d0 = str(df.index[-1])[:10]
            if latest is None:
                latest = d0
            if d0 != latest:
                continue  # 오늘 종가 없는 종목 제외
            last = df.iloc[-1]
            close = float(last["close"])
            s5, s20, s60 = last.get("sma_5"), last.get("sma_20"), last.get("sma_60")
            if any(pd.isna(x) for x in (s5, s20, s60)):
                continue
            if not (close > s5 > s20 > s60):            # 정배열 + 5일선 위
                continue
            hi = float(df["high"].iloc[-HIGH_LOOKBACK:].max())
            if close < hi * HIGH_RATIO:                  # 신고가권
                continue
            mom5 = (close / float(df["close"].iloc[-6]) - 1) * 100
            if mom5 < MOM5_MIN:                          # 5일 모멘텀
                continue
            rsi = float(last.get("rsi_14", 0) or 0)
            volavg = float(df["volume"].iloc[-20:-1].mean()) or 1.0
            volr = float(last["volume"]) / volavg
            overheat = mom5 > OVERHEAT_MOM5 or volr > OVERHEAT_VOLR or rsi > OVERHEAT_RSI
            hits.append({
                "ticker": tk,
                "name": names.get(tk, tk),
                "close": int(close),
                "mom5_pct": round(mom5, 1),
                "gap_sma5_pct": round((close / float(s5) - 1) * 100, 1),
                "rsi": round(rsi, 0),
                "vol_ratio": round(volr, 1),
                "stop_price": int(s5),                   # 5일선 = 손절선
                "grade": "과열" if overheat else "정상",
            })
        except Exception:
            continue

    supply = load_supply((latest or "").replace("-", ""))
    for h in hits:
        s = supply.get(h["ticker"], {})
        h["foreign_eok"] = round(s.get("foreign", 0.0))
        h["inst_eok"] = round(s.get("inst", 0.0))
    hits.sort(key=lambda x: -x["mom5_pct"])
    return latest, hits


def main() -> None:
    latest, hits = scan()
    normal = [h for h in hits if h["grade"] == "정상"]
    over = [h for h in hits if h["grade"] == "과열"]
    print(f"[추세추종] {latest} | 포착 {len(hits)}종 (정상 {len(normal)} / 과열 {len(over)})")
    print(f"{'종목명':<14}{'종가':>10}{'5일%':>7}{'5선이격':>8}{'RSI':>5}"
          f"{'거래량x':>7}{'손절(5선)':>11}{'외인억':>8}{'기관억':>8}  등급")
    for h in hits:
        print(f"{h['name']:<14}{h['close']:>10,}{h['mom5_pct']:>7}{h['gap_sma5_pct']:>8}"
              f"{h['rsi']:>5.0f}{h['vol_ratio']:>7}{h['stop_price']:>11,}"
              f"{h['foreign_eok']:>8.0f}{h['inst_eok']:>8.0f}  {h['grade']}")
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump({"date": latest, "signals": hits}, fp, ensure_ascii=False, indent=2)
    print(f"-> 저장: {OUT_JSON} ({len(hits)}종)")


if __name__ == "__main__":
    main()
