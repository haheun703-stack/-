#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""지주사 NAV 디스카운트 페이퍼 (5번째 독립포트) — 매일 1회 라이브 실행

전략(6/25 백테스트+정련 확정: 정련 [1,2]밴드 승률81%/평균+23%/D+60, 최악진입 -37%→손절방어):
  종목풀   검증통과 4 지주사(㈜LG·SK·두산·삼성물산). 사업지주(한화)·단일베타(SK스퀘어)·버그(CJ) 제외
  신호     z∈[1.0,2.0] (할인 좁혀지는 중·과열前) AND NAVmom5d>0 (자산 상승) AND 할인거래(disc<0)
  진입     신호 + 빈슬롯 + 쿨다운(청산 후 20거래일 재진입 금지)
  청산     D+60 보유만기(TARGET) OR -15% 손절(STOP, 꼬리위험 -37% 방어) — ★회전금지(보유)
  동시보유 4 | 시드 1억(종목당 2500만) | 페이퍼(dry, 실주문 X)

NAV = Σ(상장자회사 시총×지분율)+비상장+자체사업−순부채 (config/holding_nav.yaml, DART 실측)
할인율 z-score = 자기 252일 롤링 대비. 시총 = 자회사 종가×발행주식수(universe 현재시총/최근종가 역산).

출력: data/paper_portfolio_holdnav.json
실행(BAT-D 장후): QM_ROOT=$HOME/quantum-master ./venv/bin/python3.11 -u scripts/paper_holding_nav.py
"""
import csv
import glob  # noqa: F401  (다른 페이퍼와 구조 통일)
import json
import os
import sys
from datetime import datetime

ROOT = os.environ.get("QM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.expanduser(ROOT)
sys.path.insert(0, ROOT)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.use_cases.holding_nav import EOK, Holding  # noqa: E402

PROCESSED = os.path.join(ROOT, "data", "processed")
PF_PATH = os.path.join(ROOT, "data", "paper_portfolio_holdnav.json")
YAML_PATH = os.path.join(ROOT, "config", "holding_nav.yaml")

INITIAL = 100_000_000
INCLUDE = ["003550", "034730", "000150", "028260"]  # 검증통과 4종목
MAX_POS = len(INCLUDE)
SLOT = INITIAL / MAX_POS
ROLL = 252
Z_LO, Z_HI = 1.0, 2.0
HOLD_DAYS = 60      # D+60 보유만기(거래일)
STOP = -15.0        # 손절(%)
COOLDOWN = 20       # 청산 후 재진입 금지(거래일)
COST = 0.004


def _default_pf():
    return {"created": datetime.now().strftime("%Y-%m-%d %H:%M"), "initial_capital": INITIAL,
            "capital": INITIAL, "positions": {}, "closed_trades": [], "daily_equity": [],
            "cooldown": {}, "stats": {"total_trades": 0, "wins": 0, "losses": 0,
                                      "max_equity": INITIAL, "mdd": 0.0}, "updated": ""}


def load_pf():
    if os.path.exists(PF_PATH):
        try:
            pf = json.load(open(PF_PATH, encoding="utf-8"))
            pf.setdefault("cooldown", {})
            return pf
        except Exception:
            pass
    return _default_pf()


def save_pf(pf):
    pf["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    json.dump(pf, open(PF_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _load_caps():
    caps = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "universe.csv"), encoding="utf-8")):
        try:
            caps[str(r["ticker"]).zfill(6)] = float(r["market_cap"])
        except (ValueError, KeyError):
            pass
    return caps


def _close(tk):
    p = os.path.join(PROCESSED, f"{tk}.parquet")
    return pd.read_parquet(p, columns=["close"])["close"].astype(float) if os.path.exists(p) else None


def compute_signals():
    """검증 4종목의 현재 NAV 신호. {ticker: {name, z, mom, disc, price, on, date}}."""
    cfg = yaml.safe_load(open(YAML_PATH, encoding="utf-8"))
    holdings = [Holding.from_dict(tk, d) for tk, d in cfg["holdings"].items()
                if str(tk).zfill(6) in INCLUDE]
    caps = _load_caps()
    need = set()
    for h in holdings:
        need.add(h.ticker)
        need.update(s.ticker for s in h.listed_stakes)
    cl, sh = {}, {}
    for tk in need:
        c = _close(tk)
        if c is not None and tk in caps and c.iloc[-1] > 0:
            cl[tk], sh[tk] = c, caps[tk] / c.iloc[-1]
    mc = pd.DataFrame({tk: cl[tk] * sh[tk] for tk in cl}).sort_index()
    out = {}
    for h in holdings:
        if h.ticker not in mc.columns:
            continue
        st = pd.Series(0.0, index=mc.index)
        for s in h.listed_stakes:
            if s.ticker in mc.columns:
                st = st.add(mc[s.ticker] * (s.pct / 100), fill_value=0.0)
        fx = (h.other_nav_eok + h.own_business_eok - h.net_debt_eok) * EOK
        nav = st + fx
        df = pd.DataFrame({"nav": nav, "hold": mc[h.ticker]}).dropna()
        df = df[df["nav"] > 0]
        if len(df) < ROLL + 5:
            continue
        disc = (df["hold"] - df["nav"]) / df["nav"]
        z = (disc - disc.rolling(ROLL).median()) / disc.rolling(ROLL).std()
        mom = df["nav"].pct_change(5)
        zv, mv, dv = z.iloc[-1], mom.iloc[-1], disc.iloc[-1]
        on = bool(Z_LO <= zv <= Z_HI and mv > 0 and dv < 0)
        # 지주사 현재가 = 종가(시총/주식수와 동일 스케일 아님 → 실제 종가 사용)
        price = float(cl[h.ticker].iloc[-1])
        out[h.ticker] = {"name": h.name, "z": round(float(zv), 2), "mom": round(float(mv) * 100, 2),
                         "disc": round(float(dv) * 100, 1), "price": price, "on": on,
                         "date": df.index[-1].strftime("%Y-%m-%d")}
    return out


def run():
    pf = load_pf()
    sig = compute_signals()
    today = max((s["date"] for s in sig.values()), default=datetime.now().strftime("%Y-%m-%d"))

    # 쿨다운 1일 감소
    for k in list(pf["cooldown"]):
        pf["cooldown"][k] -= 1
        if pf["cooldown"][k] <= 0:
            del pf["cooldown"][k]

    # ── 1) 보유 업데이트 (보유일+1, 만기/손절 청산) ──
    for code in list(pf["positions"]):
        s = sig.get(code)
        if s is None:
            continue
        P = pf["positions"][code]
        P["hold_days"] = P.get("hold_days", 0) + 1
        px = s["price"]
        ret = (px / P["entry_px"] - 1) * 100
        exit_px, reason = None, None
        if ret <= STOP:
            exit_px, reason = px, "STOP"
        elif P["hold_days"] >= HOLD_DAYS:
            exit_px, reason = px, "TARGET"
        if exit_px:
            pf["capital"] += P["qty"] * exit_px * (1 - COST)
            pnl = (exit_px / P["entry_px"] - 1) * 100 - COST * 100
            pf["closed_trades"].append({
                "ticker": code, "name": P.get("name"), "strategy": "HOLDING_NAV",
                "entry_date": P["entry_date"], "exit_date": today,
                "entry_px": round(P["entry_px"]), "exit_price": round(exit_px),
                "qty": P["qty"], "hold_days": P["hold_days"],
                "pnl_pct": round(pnl, 2), "exit_reason": reason})
            pf["stats"]["total_trades"] += 1
            pf["stats"]["wins" if pnl > 0 else "losses"] += 1
            pf["cooldown"][code] = COOLDOWN
            del pf["positions"][code]

    # ── 2) 신규 진입 (신호 ON + 빈슬롯 + 쿨다운 아님) ──
    slots = MAX_POS - len(pf["positions"])
    if slots > 0:
        cands = [(code, s) for code, s in sig.items()
                 if s["on"] and code not in pf["positions"] and code not in pf["cooldown"]]
        cands.sort(key=lambda x: x[1]["z"])  # z 낮은(할인 더 깊은) 순 우선
        for code, s in cands[:slots]:
            px = s["price"]
            q = int(SLOT / px)
            if q <= 0 or pf["capital"] < q * px:
                continue
            pf["capital"] -= q * px
            pf["positions"][code] = {
                "name": s["name"], "ticker": code, "entry_date": today,
                "entry_px": px, "qty": q, "cost": q * px, "hold_days": 0,
                "strategy": "HOLDING_NAV", "entry_z": s["z"], "entry_disc": s["disc"]}

    # ── 3) equity / stats ──
    eq = pf["capital"]
    for code, P in pf["positions"].items():
        s = sig.get(code)
        eq += P["qty"] * s["price"] if s else P["cost"]
    pf["daily_equity"].append({"date": today, "equity": round(eq),
                               "capital": round(pf["capital"]), "positions": len(pf["positions"])})
    pf["stats"]["max_equity"] = max(pf["stats"]["max_equity"], eq)
    if pf["stats"]["max_equity"] > 0:
        pf["stats"]["mdd"] = min(pf["stats"]["mdd"],
                                 round((eq / pf["stats"]["max_equity"] - 1) * 100, 2))
    save_pf(pf)

    # 요약
    print(f"=== 지주사 NAV 페이퍼 [{today}] ===")
    print(f"  평가액 {eq:,.0f} | 현금 {pf['capital']:,.0f} | 보유 {len(pf['positions'])} | "
          f"누적 {pf['stats']['total_trades']}(승{pf['stats']['wins']}/패{pf['stats']['losses']}) | MDD {pf['stats']['mdd']}%")
    print("  [신호판] " + " | ".join(
        f"{s['name']} z{s['z']:+.1f}/mom{s['mom']:+.1f}%/{s['disc']:+.0f}%{'★ON' if s['on'] else ''}"
        for s in sig.values()))
    for code, P in pf["positions"].items():
        print(f"    {code} {str(P['name'])[:8]:<8} {P['qty']}주 @{P['entry_px']:,.0f} "
              f"D+{P['hold_days']} (진입z{P.get('entry_z')})")


if __name__ == "__main__":
    if datetime.now().weekday() >= 5:  # 주말 실행 금지(MEMORY 교훈)
        print("주말 — 지주사 NAV 페이퍼 실행 skip")
        sys.exit(0)
    run()
