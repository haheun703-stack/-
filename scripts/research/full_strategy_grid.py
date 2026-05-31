"""완결 매매전략 그리드 백테스트 (사장님 5/31 — 진입~보유~장중~매도 전체).

코워크 supply_alpha_portfolio_backtest는 '진입 + D+40 무조건 시간청산'만 → 매도/장중 공백.
여기서 진입신호 × 보유기간 × 청산룰(손절/트레일/익절/시간)을 포트폴리오로 그리드 검증.
"어떤 종목을 · 며칠 · 어떻게 팔아야 수익이 좋은가" 답 탐색.

청산(보유 중 매일, 일봉 고저가로 장중 근사, 우선순위 손절>트레일>익절>시간):
  손절: 저가 ≤ entry*(1-stop)         트레일: 고점갱신 후 저가 ≤ peak*(1-trail)
  익절: 고가 ≥ entry*(1+take)          시간: 보유 HOLD일 경과 → 종가
진입: 전일 신호 + KOSPI 과열(120선+15%) 아님 → 당일 시가. 슬롯 N, 현금재활용.
비용: 매수 슬리피지 0.15% / 매도 슬리피지 0.15% + 세금 0.18%.

★ 생존편향(data/processed=현재 생존종목, 상폐 미포함) 미보정 → 절대수익 상방편향.
  판정은 'KOSPI 대비 초과' + '슬롯/구간 안정성'으로. PASS 전 가설. 실매수 HOLD.
"""
from __future__ import annotations

import glob
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

PROCESSED = PROJECT_ROOT / "data" / "processed"
KOSPI = PROJECT_ROOT / "data" / "kospi_index.csv"
MIN_TV = 1e9
OVERHEAT_MULT = 1.15
START, END = pd.Timestamp("2023-06-01"), pd.Timestamp("2026-05-29")
CAP = 100_000_000
BUY_SLIP, SELL_SLIP, SELL_TAX = 0.0015, 0.0015, 0.0018
SLOTS = 10


def load_kospi():
    k = pd.read_csv(KOSPI)
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"])
    k = k.set_index("date").sort_index()
    k["ma120"] = k["close"].rolling(120).mean()
    k["overheat"] = k["close"] > k["ma120"] * OVERHEAT_MULT
    return k


# 진입 신호 정의 (parquet 행 r)
def sig_supply(r):
    return float(r.get("supply_divergence", 0) or 0) > 0

def sig_supply_inst(r):
    return float(r.get("supply_divergence", 0) or 0) > 0 and int(r.get("inst_consecutive_buy", 0) or 0) >= 3

SIGNALS = {"supply_div": sig_supply, "supply+inst3": sig_supply_inst}


def build(kospi, include_delisted=False):
    cal = list(kospi.index)
    cal_pos = {d: i for i, d in enumerate(cal)}
    overheat = {d: bool(v) for d, v in kospi["overheat"].items()}
    files = glob.glob(str(PROCESSED / "*.parquet"))
    if include_delisted:  # 생존편향 보정: 상폐 종목 유니버스 추가
        files += glob.glob(str(PROJECT_ROOT / "data" / "delisted" / "*.parquet"))
    sig_by = {name: defaultdict(list) for name in SIGNALS}
    C_raw, O_raw, H_raw, L_raw = {}, {}, {}, {}
    for f in files:
        code = Path(f).stem.zfill(6)
        try:
            df = pd.read_parquet(f).sort_index()
        except Exception:
            continue
        if "supply_divergence" not in df.columns or "trading_value" not in df.columns:
            continue
        df = df[(df.index >= START) & (df.index <= END)]
        df = df[(df["close"] > 0) & (df["open"] > 0)]
        if len(df) < 2:
            continue
        C_raw[code] = df["close"]; O_raw[code] = df["open"]
        H_raw[code] = df["high"]; L_raw[code] = df["low"]
        tv = df["trading_value"]
        recs = df.to_dict("index")
        for t, r in recs.items():
            if float(tv.get(t, 0) or 0) < MIN_TV:
                continue
            sdv = float(r.get("supply_divergence", 0) or 0)
            for name, fn in SIGNALS.items():
                if fn(r):
                    sig_by[name][t].append((code, sdv))
    C = pd.DataFrame(C_raw).reindex(cal).ffill()
    O = pd.DataFrame(O_raw).reindex(cal)
    H = pd.DataFrame(H_raw).reindex(cal).ffill()
    L = pd.DataFrame(L_raw).reindex(cal).ffill()
    return cal, cal_pos, overheat, sig_by, C, O, H, L


def simulate(cal, overheat, sig_dates, C, O, H, L, hold, stop, trail, take, last_valid=None):
    idxs = list(range(len(cal)))
    cash = float(CAP)
    pos = {}  # code -> dict(shares, entry_i, entry_px, peak)
    entries = 0; hold_sum = 0; wins = 0; closed = 0
    peak_eq, mdd, mv = CAP, 0.0, CAP
    for i in idxs:
        d = cal[i]
        # 청산 점검
        for code in list(pos):
            p = pos[code]
            if code not in C.columns:
                continue
            # 상폐 강제청산 (보정 모드): 종목 마지막 거래일 경과 → 직전 종가로 청산
            if last_valid is not None and i >= last_valid.get(code, 10**9):
                cl = C.at[d, code]
                if not pd.isna(cl):
                    cash += p["shares"] * cl * (1 - SELL_SLIP - SELL_TAX)
                    hold_sum += (i - p["entry_i"]); closed += 1
                    if cl > p["entry_px"]:
                        wins += 1
                    del pos[code]
                continue
            hi = H.at[d, code]; lo = L.at[d, code]; cl = C.at[d, code]
            if pd.isna(cl):
                continue
            if not pd.isna(hi):
                p["peak"] = max(p["peak"], hi)
            exit_px = None
            if stop and not pd.isna(lo) and lo <= p["entry_px"] * (1 - stop):
                exit_px = p["entry_px"] * (1 - stop)
            elif trail and not pd.isna(lo) and lo <= p["peak"] * (1 - trail):
                exit_px = p["peak"] * (1 - trail)
            elif take and not pd.isna(hi) and hi >= p["entry_px"] * (1 + take):
                exit_px = p["entry_px"] * (1 + take)
            elif i >= p["entry_i"] + hold:
                exit_px = cl
            if exit_px is not None:
                cash += p["shares"] * exit_px * (1 - SELL_SLIP - SELL_TAX)
                hold_sum += (i - p["entry_i"]); closed += 1
                if exit_px > p["entry_px"]:
                    wins += 1
                del pos[code]
        # 진입 (전일 신호, 과열 아님)
        if i > 0:
            dp = cal[i - 1]
            if not overheat.get(dp, False):
                cands = sorted(sig_dates.get(dp, []), key=lambda x: -x[1])
                free = SLOTS - len(pos)
                for code, _ in cands:
                    if free <= 0:
                        break
                    if code in pos or code not in O.columns:
                        continue
                    eo = O.at[d, code]
                    if pd.isna(eo) or eo <= 0:
                        continue
                    alloc = cash / free
                    if alloc < 300_000:
                        continue
                    sh = int(alloc / (eo * (1 + BUY_SLIP)))
                    if sh <= 0:
                        continue
                    cash -= sh * eo * (1 + BUY_SLIP)
                    pos[code] = {"shares": sh, "entry_i": i, "entry_px": eo, "peak": eo}
                    free -= 1; entries += 1
        # 평가
        held = 0.0
        for code, p in pos.items():
            if code in C.columns:
                px = C.at[d, code]
                if not pd.isna(px):
                    held += p["shares"] * px
        mv = cash + held
        peak_eq = max(peak_eq, mv)
        mdd = max(mdd, (peak_eq - mv) / peak_eq)
    ret = mv / CAP - 1
    return {"ret": ret, "entries": entries, "mdd": mdd,
            "avg_hold": hold_sum / max(closed, 1), "winrate": wins / max(closed, 1) * 100, "closed": closed}


def main() -> int:
    kospi = load_kospi()
    cal, cal_pos, overheat, sig_by, C, O, H, L = build(kospi)
    kclose = {d: float(v) for d, v in kospi["close"].items()}
    kos_ret = (kclose[cal[-1]] / kclose[cal[0]] - 1) * 100
    print(f"캘린더 {len(cal)}일 / 슬롯 {SLOTS} / KOSPI buy&hold {kos_ret:+.1f}% / 과열일 {sum(overheat.values())}")
    print(f"신호 발생일수: " + " / ".join(f"{n}={len(sig_by[n])}" for n in SIGNALS) + "\n")

    # 청산룰 세트 (stop, trail, take)
    exits = [
        ("시간만(D+H)", 0.0, 0.0, 0.0),
        ("손절8", 0.08, 0.0, 0.0),
        ("손절8+트레일7", 0.08, 0.07, 0.0),
        ("손절8+익절20", 0.08, 0.0, 0.20),
        ("손절8+트레일12+익절25", 0.08, 0.12, 0.25),
    ]
    holds = [20, 40]
    print(f'{"진입":<13}{"보유":>5}{"청산룰":<22}{"수익":>9}{"vs KOSPI":>10}{"승률":>6}{"MaxDD":>7}{"평균보유":>8}{"진입수":>7}')
    rows = []
    for signame in SIGNALS:
        for hold in holds:
            for ename, stop, trail, take in exits:
                r = simulate(cal, overheat, sig_by[signame], C, O, H, L, hold, stop, trail, take)
                rows.append((signame, hold, ename, r))
                print(f'{signame:<13}{hold:>5}{ename:<22}{r["ret"]*100:>+8.1f}%{(r["ret"]*100-kos_ret):>+9.1f}%p'
                      f'{r["winrate"]:>5.0f}%{r["mdd"]*100:>6.1f}%{r["avg_hold"]:>7.0f}일{r["entries"]:>7}')
    best = max(rows, key=lambda x: x[3]["ret"] - kos_ret / 100)
    print(f'\n★ 최고(vs KOSPI): {best[0]} / D+{best[1]} / {best[2]} → 초과 {(best[3]["ret"]*100-kos_ret):+.1f}%p')
    print("★ 생존편향 미보정(상방편향). KOSPI 대비 초과 + 슬롯/구간 안정성으로 해석. PASS 전 가설.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
