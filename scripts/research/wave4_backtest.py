"""3파4파 과거 PF 백테스트 (코덱스 3단계 검증 — 5/31 사장님 6조건).

★ lookahead 안전: 각 거래일 i에서 analyze(df.iloc[:i+1]) 재호출 (미래 미참조).
6조건:
  ① signal_date ≠ entry_date: i일 신호 → i+1 시가 진입 (당일 미완성 캔들 진입 금지)
  ② 유동성 필터: i일 거래대금 ≥ 하한 (시나리오: 0 / 10억 / 50억)
  ③ 거래비용: 왕복 0.5% (수수료 0.015%×2 + 거래세 0.18% + 슬리피지 ~0.3%)
  ④ 손절/청산 고정(사전 고정, 사후 최적화 금지):
       손절 = 종가가 1파 고점 하회 (파동중첩 무효)
       트레일 = 진입 후 고점 대비 -3%
       시간 = D+20 종가 (5파 목표 시간)
  ⑤ 워크포워드: 23하 / 24 / 25 / 26 구간 분리 (전체 일괄 최적화 금지)
  ⑥ 리포트: PF/승률/평균수익·손실/MDD/거래수/종목편중 + 유동성 필터 전후 비교

룰은 scan_wave4_pullback.py와 동일(사전 고정). 무거움 → --sample N (거래대금 상위).
퀀트봇=연구자. 실매수 HOLD. PASS 전까지 전략 아님=가설.
"""
from __future__ import annotations

import argparse
import glob
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.geometric_engine import ElliottWaveAnalyzer, WaveType

HOLD_MAX = 20
TRAIL = 0.03
COST = 0.005
CONF_MIN = 60.0
RMIN, RMAX = 0.236, 0.618
WARMUP = 60  # 최소 누적 봉 (지표/파동 안정)


def is_signal(wave) -> dict | None:
    if wave is None or wave.wave_type != WaveType.IMPULSE or wave.direction != "up":
        return None
    if wave.current_wave not in ("4", "4~5 전환"):
        return None
    if wave.confidence < CONF_MIN:
        return None
    if not any(r.startswith("규칙3") for r in wave.rules_passed):
        return None
    w = wave.waves
    try:
        w1h = float(w["1"][3])
        w3s = float(w["3"][2])
        w3e = float(w["3"][3])
        w4l = float(w["4"][3])
    except (KeyError, IndexError, TypeError):
        return None
    length = w3e - w3s
    if length <= 0:
        return None
    rr = (w3e - w4l) / length
    if not (RMIN <= rr <= RMAX):
        return None
    return {"w1_high": w1h, "retrace": rr}


def backtest(dfs, s, e, min_value, analyzer) -> list:
    trades = []
    for code, df in dfs:
        d = df[(df.index >= s) & (df.index <= e)]
        if len(d) < WARMUP + 25:
            continue
        o = d["open"].values
        c = d["close"].values
        hi = d["high"].values
        lo = d["low"].values
        tv = d["trading_value"].values if "trading_value" in d.columns else np.full(len(d), 1e12)
        N = len(d)
        i = WARMUP
        while i < N - 1:
            if tv[i] < min_value:
                i += 1
                continue
            sub = d.iloc[: i + 1]
            try:
                wave = analyzer.analyze(sub, lookback=200)
            except Exception:
                i += 1
                continue
            sg = is_signal(wave)
            if sg is None:
                i += 1
                continue
            entry = o[i + 1]
            if entry <= 0 or np.isnan(entry):
                i += 1
                continue
            stop_price = sg["w1_high"]
            peak = entry
            exit_price = None
            exit_j = None
            for j in range(i + 1, min(i + 1 + HOLD_MAX, N)):
                peak = max(peak, hi[j])
                if c[j] < stop_price:  # 손절: 1파 고점 하회
                    exit_price = c[j]
                    exit_j = j
                    break
                if peak > entry and lo[j] <= peak * (1 - TRAIL):  # 트레일 -3%
                    exit_price = peak * (1 - TRAIL)
                    exit_j = j
                    break
            if exit_price is None:  # 시간 청산 D+20
                exit_j = min(i + HOLD_MAX, N - 1)
                exit_price = c[exit_j]
            ret = exit_price / entry - 1 - COST
            if not np.isnan(ret):
                trades.append({"code": code, "ret": ret, "bars": exit_j - (i + 1)})
            i = exit_j + 1  # 청산 후 재탐색 (중복 보유 금지)
    return trades


def report(trades, label):
    if not trades:
        print(f"{label:<10}{'거래 0':>8}")
        return
    rets = np.array([t["ret"] for t in trades])
    n = len(rets)
    win = rets[rets > 0]
    loss = rets[rets <= 0]
    pf = win.sum() / (-loss.sum()) if loss.sum() < 0 else 99.0
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min() * 100
    cc = Counter(t["code"] for t in trades)
    top_code, top_n = cc.most_common(1)[0]
    conc = top_n / n * 100  # 최다 종목 편중%
    print(f"{label:<10}{n:>6}{len(win)/n*100:>6.0f}%{rets.mean()*100:>+8.2f}%"
          f"{(win.mean()*100 if len(win) else 0):>+7.2f}%{(loss.mean()*100 if len(loss) else 0):>+7.2f}%"
          f"{pf:>6.2f}{mdd:>7.1f}%{conc:>6.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="거래대금 상위 N종목 (0=전체)")
    args = ap.parse_args()

    analyzer = ElliottWaveAnalyzer(zigzag_pct=3.0, min_bars=3)
    files = glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet"))
    dfs = []
    for f in files:
        try:
            code = Path(f).stem
            df = pd.read_parquet(f).sort_index()
            if len(df) < 90:
                continue
            avg_tv = float(df["trading_value"].tail(250).mean()) if "trading_value" in df.columns else 0.0
            dfs.append((code, df, avg_tv))
        except Exception:
            pass
    if args.sample > 0:
        dfs.sort(key=lambda x: -x[2])
        dfs = dfs[: args.sample]
    dfs = [(c, d) for c, d, _ in dfs]
    print(f"종목 {len(dfs)}개 (D+{HOLD_MAX} max / 트레일 -{TRAIL*100:.0f}% / 손절=1파고점 / 비용 {COST*100:.1f}%)\n")

    periods = [
        ("2023-06-01", "2023-12-31", "23하"),
        ("2024-01-01", "2024-12-31", "24약세"),
        ("2025-01-01", "2025-12-31", "25년"),
        ("2026-01-01", "2026-05-29", "26최근"),
        ("2023-06-01", "2026-05-29", "전체"),
    ]
    for label, min_value in [("유동성필터 OFF (거래대금 0)", 0), ("유동성필터 ON (거래대금≥10억)", 1e9), ("유동성필터 STRONG (≥50억)", 5e9)]:
        print(f"=== {label} ===")
        print(f'{"구간":<10}{"거래":>6}{"승률":>6}{"평균":>8}{"평익":>7}{"평손":>7}{"PF":>6}{"MDD":>8}{"편중":>6}')
        for s, e, lbl in periods:
            report(backtest(dfs, s, e, min_value, analyzer), lbl)
        print()

    print("★ lookahead 안전(df.iloc[:i+1]). 생존자편향=현재 생존종목만(상폐 미포함, PF 상방편향).")
    print("  PASS 전까지 가설. 사후 최적화 금지(룰 사전 고정).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
