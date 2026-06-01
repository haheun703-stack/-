"""손익비 엔진 — expectancy 모드 (단타봇 설계 6/1, 퀀트봇 실행).

자 전환: 예측(IC, 13번 약함) 버리고 expectancy(끊고 따라가기) 잰다.
진입: 거래량≥5x & 종가강도≥0.8 & 양봉(끼) → T+1 시가 (갭상승 슬립 반영)
청산: 고점 -3% 트레일링(손절측 슬리피지 크게) + 청산후 재신호=재진입(STOP_REENTER). 고정TP 없음.
측정: expectancy(비용차감)·손익비(avg승/avg패)·승률·MDD·거래수.
합격(단타봇): ①expectancy>0 ②손익비≥2 ③강세·약세 양수 ④비용2배 견딤
함정 4: 슬리피지(손절 -1% 추가)·체결(T+1갭상승)·다중검정(-3%고정,변형은 민감도)·생존편향(상폐포함).
★2025.6~2026.5. look-ahead 0.
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

# 비용 (1배 / 스트레스 2배)
BUY_SLIP = 0.0015
SELL_TAX = 0.0018
STOP_SLIP = 0.010   # 손절측 급락 미끄러짐 (단타봇 함정①)
GAP_SLIP = 0.003    # T+1 갭상승 진입 불리 (단타봇 함정②)
VOL_X = 5.0
CLOSE_STR = 0.8
MIN_TV = 5e8


def load():
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); k = k.set_index("date").sort_index()
    store = {}
    for pat in ["data/processed/*.parquet", "data/delisted/*.parquet"]:  # 생존편향 보정(함정④)
        for f in glob.glob(str(PROJECT_ROOT / pat)):
            try:
                df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
            except Exception:
                continue
            if len(df) < 40 or not all(c in df.columns for c in ["open", "high", "low", "close", "volume", "volume_ma20"]):
                continue
            code = Path(f).stem
            if code not in store:
                store[code] = df
    return k, store


def trades_for(df, S, E, cost_mult, min_tv=MIN_TV):
    """한 종목의 모든 거래(진입~트레일청산) 수익률 리스트. STOP_REENTER 포함."""
    o, h, lo, c, v, vma = (df["open"], df["high"], df["low"], df["close"],
                           df["volume"], df["volume_ma20"])
    rng = (h - lo).replace(0, np.nan)
    cstr = (c - lo) / rng
    prevc = c.shift(1)
    sig = (v >= vma * VOL_X) & (cstr >= CLOSE_STR) & (c > prevc) & ((v * c) >= min_tv)
    idx = list(df.index)
    n = len(idx)
    res = []
    i = 1
    bs = BUY_SLIP * cost_mult; ss = STOP_SLIP * cost_mult; tax = SELL_TAX * cost_mult; gs = GAP_SLIP * cost_mult
    while i < n - 1:
        d = idx[i]
        if not bool(sig.get(d, False)) or not (S <= idx[i + 1] <= E):
            i += 1; continue
        # T+1 시가 진입 (갭상승 슬립)
        entry = o.iloc[i + 1] * (1 + bs + gs)
        if pd.isna(entry) or entry <= 0:
            i += 1; continue
        peak = h.iloc[i + 1] if not pd.isna(h.iloc[i + 1]) else entry
        j = i + 1
        exitp = None
        while j < n:
            hj, lj, cj = h.iloc[j], lo.iloc[j], c.iloc[j]
            if not pd.isna(hj):
                peak = max(peak, hj)
            stop = peak * 0.97
            if not pd.isna(lj) and lj <= stop:
                exitp = stop * (1 - ss - tax)  # 손절 슬리피지
                break
            j += 1
        if exitp is None:  # 데이터 끝까지 보유 → 마지막 종가 청산
            lastc = c.iloc[min(j, n - 1) - 0] if j < n else c.iloc[-1]
            exitp = c.iloc[-1] * (1 - tax)
            j = n
        res.append(exitp / entry - 1)
        i = j + 1  # 청산 후부터 재탐색 = STOP_REENTER
    return res


def expectancy(store, S, E, cost_mult, min_tv=MIN_TV):
    allr = []
    for df in store.values():
        allr += trades_for(df, S, E, cost_mult, min_tv)
    if not allr:
        return None
    r = np.array(allr)
    wins = r[r > 0]; losses = r[r <= 0]
    exp = r.mean()
    pf_ratio = (wins.mean() / abs(losses.mean())) if len(losses) and losses.mean() != 0 else float("inf")
    wr = len(wins) / len(r) * 100
    return dict(exp=exp * 100, ratio=pf_ratio, wr=wr, n=len(r),
                avgwin=wins.mean() * 100 if len(wins) else 0,
                avgloss=losses.mean() * 100 if len(losses) else 0)


def main() -> int:
    k, store = load()
    print(f"손익비 엔진 expectancy 모드 — {len(store)}종목(상폐포함), 진입=거래량5x+종가강도0.8+양봉, 청산=고점-3%트레일+재진입\n")
    # 강세/약세 분리: KOSPI 200MA 위=강세일 비중 높은 구간. 사장님 철칙(25.6~26.5)내 전반/후반으로 근사
    periods = [("2025-06-01", "2026-05-29", "전체(강세)"),
               ("2025-06-01", "2025-11-30", "전반"),
               ("2025-12-01", "2026-05-29", "후반")]
    for cost_mult, clbl in [(1.0, "비용 1배"), (2.0, "비용 2배(스트레스)")]:
        print(f"=== {clbl} ===")
        print(f'{"구간":<14}{"expectancy":>11}{"손익비":>8}{"승률":>7}{"평균익":>8}{"평균손":>8}{"거래수":>7}')
        for s, e, lbl in periods:
            m = expectancy(store, pd.Timestamp(s), pd.Timestamp(e), cost_mult)
            if m:
                print(f'{lbl:<14}{m["exp"]:>+10.2f}%{m["ratio"]:>7.2f}{m["wr"]:>6.0f}%'
                      f'{m["avgwin"]:>+7.1f}%{m["avgloss"]:>+7.1f}%{m["n"]:>7}')
        print()
    print("★ 합격(단타봇): ①expectancy>0(비용차감) ②손익비≥2 ③전반·후반 양수 ④비용2배도 양수")
    print("  불합격 시 어느 함정(비용/슬리피지/구간)이 죽였는지 위 분해로 판정 → 행동가능.\n")

    # ── 유동성 상향이 ④(비용2배 슬리피지)를 푸는가 — 단타봇 paper 사전검증 ──
    print("=== 유동성 상향 검증 (비용 2배 스트레스 고정, 전체구간) — ④ 범인 슬리피지 해결되나 ===")
    print(f'{"거래대금하한":<12}{"expectancy":>11}{"손익비":>8}{"승률":>7}{"거래수":>7}{"  ④판정"}')
    for tv, lbl in [(5e8, "5억"), (3e9, "30억"), (1e10, "100억"), (3e10, "300억"), (1e11, "1000억")]:
        m = expectancy(store, pd.Timestamp("2025-06-01"), pd.Timestamp("2026-05-29"), 2.0, tv)
        if m:
            ok = "✅통과" if (m["exp"] > 0 and m["ratio"] >= 2) else "❌"
            print(f'{lbl:<12}{m["exp"]:>+10.2f}%{m["ratio"]:>7.2f}{m["wr"]:>6.0f}%{m["n"]:>7}   {ok}')
    print("★ 유동성 올릴수록 expectancy>0 & 손익비≥2 회복되면 = 단타봇 진단(슬리피지 범인) 맞음, paper 갈 근거.")
    print("  거래수 급감하면 = 유동성 필터가 신호를 죽임(다른 트레이드오프). paper 전 여기서 판단.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
