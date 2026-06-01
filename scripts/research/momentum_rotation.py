"""모멘텀 로테이션 — "강한 놈이 더 강해진다 + 엉덩이로 돈 번다" (사장님 6/1).

8번 실패 = 단기신호로 사고 손절/청산으로 자주 팔아서(휩소). 정반대로:
  중기 상대강도(룩백 수익률) 상위 K종목 동일가중 보유 → 손절 없음 →
  정기 리밸런싱(월/격주)에 순위 밖 종목만 더 강한 놈으로 교체.
초강세장 주도주 올라타기의 정석. ★2025.6~2026.5만. vs KOSPI. 절대수익+MDD.
look-ahead 0(리밸일 종가까지 룩백으로만 선정, 다음날 시가 진입).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

CAP = 100_000_000
BUY_SLIP, SELL_SLIP, SELL_TAX = 0.0015, 0.0015, 0.0018
MIN_TV = 1e9   # 거래대금 10억 (유동성)


def load():
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); cal = list(k.sort_values("date")["date"])
    C, O, TV = {}, {}, {}
    for f in glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet")):
        try:
            df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
        except Exception:
            continue
        if len(df) < 200 or "trading_value" not in df.columns:
            continue
        code = Path(f).stem
        C[code] = df["close"]; O[code] = df["open"]; TV[code] = df["trading_value"]
    Cm = pd.DataFrame(C).reindex(cal).ffill()
    Om = pd.DataFrame(O).reindex(cal)
    TVm = pd.DataFrame(TV).reindex(cal).ffill()
    return cal, k.set_index("date")["close"], Cm, Om, TVm


def rebal_days(cal, S, E, freq):
    days = [d for d in cal if S <= d <= E]
    out = []
    if freq == "M":
        seen = set()
        for i, d in enumerate(days):
            key = (d.year, d.month)
            nxt = days[i + 1] if i + 1 < len(days) else None
            if nxt is None or (nxt.year, nxt.month) != key:  # 그 달 마지막 거래일
                out.append(d)
    elif freq == "2W":
        out = days[::10]
    return out


def sim(cal, Cm, Om, TVm, S, E, lookback, K, freq):
    cols = list(Cm.columns)
    rb = set(rebal_days(cal, S, E, freq))
    idxs = [i for i, d in enumerate(cal) if S <= d <= E]
    cash = float(CAP); pos = {}; peak = CAP; mdd = 0.0; mv = CAP
    target = None
    for i in idxs:
        d = cal[i]; dp = cal[i - 1] if i > 0 else None
        # 전 거래일이 리밸일이면 오늘 시가로 재조정
        if dp is not None and dp in rb and target is not None:
            # 매도: 목표에 없는 종목 전량
            for code in list(pos):
                if code not in target:
                    px = Om.at[d, code]
                    if pd.isna(px) or px <= 0:
                        px = Cm.at[d, code]
                    cash += pos[code] * px * (1 - SELL_SLIP - SELL_TAX); del pos[code]
            # 동일가중 목표 금액
            if target:
                tgt_val = mv / len(target)
                for code in target:
                    px = Om.at[d, code]
                    if pd.isna(px) or px <= 0:
                        continue
                    cur = pos.get(code, 0) * px
                    if cur < tgt_val:  # 추가 매수
                        buy = min(tgt_val - cur, cash)
                        sh = int(buy / (px * (1 + BUY_SLIP)))
                        if sh > 0:
                            cash -= sh * px * (1 + BUY_SLIP); pos[code] = pos.get(code, 0) + sh
                    elif cur > tgt_val * 1.3:  # 과대 → 일부 매도
                        sh = int((cur - tgt_val) / px)
                        if 0 < sh <= pos.get(code, 0):
                            cash += sh * px * (1 - SELL_SLIP - SELL_TAX); pos[code] -= sh
        # 리밸일이면 다음날 목표 선정 (오늘 종가까지 룩백 = look-ahead 0)
        if d in rb:
            j = i - lookback
            if j >= 0:
                ret = {}
                for code in cols:
                    p0 = Cm.at[cal[j], code]; p1 = Cm.at[d, code]
                    tv = TVm.at[d, code]
                    if not pd.isna(p0) and not pd.isna(p1) and p0 > 0 and not pd.isna(tv) and tv >= MIN_TV:
                        ret[code] = p1 / p0 - 1
                ranked = sorted(ret, key=ret.get, reverse=True)
                target = ranked[:K]
        held = sum(sh * Cm.at[d, code] for code, sh in pos.items() if not pd.isna(Cm.at[d, code]))
        mv = cash + held; peak = max(peak, mv); mdd = max(mdd, (peak - mv) / peak)
    return mv / CAP - 1, mdd


def main() -> int:
    cal, kclose, Cm, Om, TVm = load()
    S, E = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-05-29")
    ksub = kclose[(kclose.index >= S) & (kclose.index <= E)]
    kos = (ksub.iloc[-1] / ksub.iloc[0] - 1) * 100
    print(f"모멘텀 로테이션 — 2025.6~2026.5 초강세장 (KOSPI {kos:+.1f}%)\n")
    print(f'{"룩백":>5}{"종목수":>6}{"리밸":>6}{"수익":>10}{"vs KOSPI":>10}{"MDD":>8}')
    for lookback, lbl in [(60, "3개월"), (120, "6개월"), (20, "1개월")]:
        for K in (5, 10, 20):
            for freq in ("M", "2W"):
                ret, mdd = sim(cal, Cm, Om, TVm, S, E, lookback, K, freq)
                print(f'{lbl:>5}{K:>6}{freq:>6}{ret*100:>+9.1f}%{(ret*100-kos):>+9.1f}%p{mdd*100:>7.1f}%')
    print(f'\n{"KOSPI buy&hold":<20}{kos:>+8.1f}%{0:>+9.1f}%p')
    print("★ 모멘텀이 KOSPI 초과(+) = 초강세장 주도주 올라타기 성공. 처음으로 이기면 진짜 무기.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
