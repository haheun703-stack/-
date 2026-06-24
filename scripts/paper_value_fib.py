#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""밸류-피보나치 페이퍼 (4번째 독립포트) — 매일 1회 라이브 실행

전략(6/24 백테스트 확정: 승률 77%, +33%/년, MDD -9%):
  종목풀   시총 Top100
  펀더멘털 ROE>8% AND 부채<200% AND 영업익YoY>0
           (정보봇 company_card 우선 → 없으면 valuation_gap 폴백)
  진입     60일고점 -10%↓ + RSI<40 + 반등확인(RSI상승/SAR반전/양봉) + 수급(외인or기관)
  가점     미국 peer 저평가 (quant_us_peer, 대표 ~10종목)
  청산     60일 전고점 OR +25% 익절 / -20% 손절 — ★회전금지(보유)
  동시보유 10 | 시드 1억 | 페이퍼(dry, 실주문 X)

출력: data/paper_portfolio_vf.json (메인/B안/Bluechip 동일 구조)
실행(BAT-D 장후): QM_ROOT=$HOME/quantum-master ./venv/bin/python3.11 -u scripts/paper_value_fib.py
"""
import sys
import os
import json
import glob
from datetime import datetime

ROOT = os.environ.get("QM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.expanduser(ROOT)
sys.path.insert(0, ROOT)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass
import pandas as pd

PROCESSED = os.path.join(ROOT, "data", "processed")
PF_PATH = os.path.join(ROOT, "data", "paper_portfolio_vf.json")
INITIAL = 100_000_000
MAX_POS = 10
SLOT = INITIAL / MAX_POS
ENTRY_DROP, ENTRY_RSI = -10.0, 40.0
ADD2_DROP, ADD3_DROP = -7.0, -13.0
F1, F2, F3 = 0.50, 0.30, 0.20
STOP, TAKE_TARGET = -20.0, 25.0
ROE_MIN, DEBT_MAX = 8.0, 200.0
COST = 0.004


def _default_pf():
    return {"created": datetime.now().strftime("%Y-%m-%d %H:%M"), "initial_capital": INITIAL,
            "capital": INITIAL, "positions": {}, "closed_trades": [], "daily_equity": [],
            "stats": {"total_trades": 0, "wins": 0, "losses": 0, "max_equity": INITIAL, "mdd": 0.0},
            "updated": ""}


def load_pf():
    if os.path.exists(PF_PATH):
        try:
            return json.load(open(PF_PATH, encoding="utf-8"))
        except Exception:
            pass
    return _default_pf()


def save_pf(pf):
    pf["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    json.dump(pf, open(PF_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def top_codes(n=100):
    u = pd.read_csv(os.path.join(ROOT, "data", "universe.csv"), dtype={"ticker": str})
    u = u.sort_values("market_cap", ascending=False)
    name = dict(zip(u["ticker"].str.zfill(6), u["name"]))
    return [str(t).zfill(6) for t in u["ticker"].head(n)], name


_VG = None
def _vg():
    global _VG
    if _VG is None:
        _VG = {}
        fs = sorted(glob.glob(os.path.join(ROOT, "data", "valuation_gap_*.json")))
        if fs:
            try:
                for x in json.load(open(fs[-1])):
                    _VG[str(x.get("ticker")).zfill(6)] = x
            except Exception:
                pass
    return _VG


def fundamental_ok(ticker):
    """ROE>8 & 부채<200 & 영업익YoY>0. 정보봇 company_card 우선 → valuation_gap 폴백."""
    try:
        from src.adapters.quant_supabase_reader import get_company_card
        cc = get_company_card(ticker)
    except Exception:
        cc = None
    if cc and cc.get("roe") is not None:
        roe = cc["roe"] * 100 if abs(cc["roe"]) < 1 else cc["roe"]
        debt = cc.get("debt_ratio") or 0
        debt = debt * 100 if debt < 10 else debt
        yoy = cc.get("quarter_op_yoy")
        return (roe > ROE_MIN and debt < DEBT_MAX and (yoy is None or yoy > 0)), "card"
    v = _vg().get(ticker)
    if v:
        yoy = v.get("oi_yoy")
        ok = (yoy is not None and yoy > 0) and (v.get("grade") in ("S", "A"))
        return ok, "vg폴백"
    return False, "무데이터"


def us_peer_bonus(ticker):
    try:
        from src.adapters.quant_supabase_reader import get_us_peer
        p = get_us_peer(ticker)
        if p and str(p.get("vs_us_valuation", "")).startswith(("저평가", "under", "UNDER")):
            return True
    except Exception:
        pass
    return False


def run():
    pf = load_pf()
    codes, name = top_codes(100)
    today = ""
    # ── 1) 보유 포지션 업데이트 (분할매수 + 청산) ──
    for code in list(pf["positions"]):
        pq = os.path.join(PROCESSED, f"{code}.parquet")
        if not os.path.exists(pq):
            continue
        df = pd.read_parquet(pq)
        r = df.iloc[-1]
        today = df.index[-1].strftime("%Y-%m-%d")
        P = pf["positions"][code]
        hi, lo, cl = float(r["high"]), float(r["low"]), float(r["close"])
        P["peak_price"] = max(P.get("peak_price", P["avg_price"]), hi)
        entry = P["entry_px"]
        dec = (lo / entry - 1) * 100
        if not P.get("b2") and dec <= ADD2_DROP and pf["capital"] >= SLOT * F2:
            px = entry * (1 + ADD2_DROP / 100); q = int(SLOT * F2 / px)
            if q > 0:
                P["cost"] += q * px; P["qty"] += q; P["avg_price"] = P["cost"] / P["qty"]
                pf["capital"] -= q * px; P["b2"] = True
        if not P.get("b3") and dec <= ADD3_DROP and pf["capital"] >= SLOT * F3:
            px = entry * (1 + ADD3_DROP / 100); q = int(SLOT * F3 / px)
            if q > 0:
                P["cost"] += q * px; P["qty"] += q; P["avg_price"] = P["cost"] / P["qty"]
                pf["capital"] -= q * px; P["b3"] = True
        avg = P["avg_price"]
        exit_px, reason = None, None
        if (lo / avg - 1) * 100 <= STOP:
            exit_px, reason = avg * (1 + STOP / 100), "STOP"
        elif hi >= P["high60"] or (hi / avg - 1) * 100 >= TAKE_TARGET:
            exit_px, reason = min(hi, max(P["high60"], avg * (1 + TAKE_TARGET / 100))), "TARGET"
        if exit_px:
            pf["capital"] += P["qty"] * exit_px * (1 - COST)
            pnl = (exit_px / avg - 1) * 100
            pf["closed_trades"].append({
                "ticker": code, "name": P.get("name"), "strategy": "VALUE_FIB",
                "entry_date": P["entry_date"], "exit_date": today,
                "avg_price": round(avg), "exit_price": round(exit_px), "qty": P["qty"],
                "pnl_pct": round(pnl, 2), "exit_reason": reason})
            pf["stats"]["total_trades"] += 1
            pf["stats"]["wins" if pnl > 0 else "losses"] += 1
            del pf["positions"][code]
    # ── 2) 신규 진입 ──
    slots = MAX_POS - len(pf["positions"])
    if slots > 0:
        cands = []
        for code in codes:
            if code in pf["positions"]:
                continue
            pq = os.path.join(PROCESSED, f"{code}.parquet")
            if not os.path.exists(pq):
                continue
            try:
                df = pd.read_parquet(pq)
            except Exception:
                continue
            if not all(k in df.columns for k in ("close", "rsi_14", "high_60", "high", "low", "open")):
                continue
            r = df.iloc[-1]
            today = df.index[-1].strftime("%Y-%m-%d")
            h60, cl, rsi = float(r["high_60"]), float(r["close"]), r["rsi_14"]
            if h60 <= 0 or pd.isna(cl) or pd.isna(rsi):
                continue
            drop = (cl / h60 - 1) * 100
            fs = r.get("foreign_consecutive_buy", 0); fs = 0 if pd.isna(fs) else fs
            iss = r.get("inst_consecutive_buy", 0); iss = 0 if pd.isna(iss) else iss
            if not (drop <= ENTRY_DROP and rsi < ENTRY_RSI and (fs > 0 or iss > 0)):
                continue
            rsi_pv = df["rsi_14"].iloc[-2] if len(df) > 1 else None
            op = float(r["open"]); sar = r.get("sar_reversal_up", 0); sar = 0 if pd.isna(sar) else sar
            if not ((rsi_pv is not None and not pd.isna(rsi_pv) and rsi > rsi_pv) or sar == 1 or cl > op):
                continue
            ok, src = fundamental_ok(code)
            if not ok:
                continue
            cands.append((drop, code, cl, h60, src, us_peer_bonus(code)))
        # 미국peer 가점 우선 → drop 깊은(싼) 순
        cands.sort(key=lambda x: (not x[5], x[0]))
        for drop, code, cl, h60, src, bonus in cands[:slots]:
            first = SLOT * F1
            q = int(first / cl)
            if q <= 0 or pf["capital"] < q * cl:
                continue
            pf["capital"] -= q * cl
            pf["positions"][code] = {
                "name": name.get(code, code), "ticker": code, "entry_date": today,
                "entry_px": cl, "avg_price": cl, "qty": q, "cost": q * cl,
                "peak_price": cl, "high60": h60, "strategy": "VALUE_FIB",
                "fund_src": src, "us_peer": bonus, "b2": False, "b3": False}
    # ── 3) equity / stats ──
    eq = pf["capital"]
    for code, P in pf["positions"].items():
        pq = os.path.join(PROCESSED, f"{code}.parquet")
        try:
            eq += P["qty"] * float(pd.read_parquet(pq)["close"].iloc[-1])
        except Exception:
            eq += P["cost"]
    pf["daily_equity"].append({"date": today, "equity": round(eq),
                               "capital": round(pf["capital"]), "positions": len(pf["positions"])})
    pf["stats"]["max_equity"] = max(pf["stats"]["max_equity"], eq)
    if pf["stats"]["max_equity"] > 0:
        pf["stats"]["mdd"] = min(pf["stats"]["mdd"], round((eq / pf["stats"]["max_equity"] - 1) * 100, 2))
    save_pf(pf)
    # 요약
    print(f"=== 밸류-피보나치 페이퍼 [{today}] ===")
    print(f"  평가액 {eq:,.0f} | 현금 {pf['capital']:,.0f} | 보유 {len(pf['positions'])} | "
          f"누적거래 {pf['stats']['total_trades']}(승{pf['stats']['wins']}/패{pf['stats']['losses']}) | MDD {pf['stats']['mdd']}%")
    for code, P in pf["positions"].items():
        print(f"    {code} {str(P['name'])[:10]:<10} {P['qty']}주 @{P['avg_price']:,.0f} "
              f"[{P.get('fund_src')}{'·US가점' if P.get('us_peer') else ''}]")


if __name__ == "__main__":
    run()
