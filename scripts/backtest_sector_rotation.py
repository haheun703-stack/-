# -*- coding: utf-8 -*-
"""규칙 B 백테스트 — 주도 섹터 로테이션의 지속성 심판.

배경 (2026-07-10, TRADING_PRINCIPLES_퀀트봇.md 규칙 B):
- "돈이 쏠리는 주도주에 타고, 끝물엔 나온다" — 일일 리포트(주도 섹터 Top3/이탈 경고)
  배선 전에 '주도 섹터가 실제로 지속되는지'를 과거 데이터로 먼저 심판한다.
- 코워크 지시서는 KIS 재호출을 권했으나 기존 자산(parquet 수급 컬럼 7.5년 100% 채움
  + krx_full_sectors.json 29업종)이 더 깊어 재사용 (7/10 실측: 기관/외인 = 원 단위).

사전 명시 설계 (그리드 탐색 금지):
- 섹터 = krx_full_sectors.json (구성 3종목 미만 제외). 동일가중 일일수익.
- 기저선 = 유니버스(로드된 전 종목) 동일가중 평균 → 초과수익 = 섹터 - 유니버스.
- 윈도우 = 비중첩 20거래일 (중첩 t 과대 교훈 반영 — 윈도우 단위 t는 깨끗).
- 거래대금 = close×volume 근사 (parquet trading_value가 2025부터 결측이라 전 기간 통일).
- 가설1 모멘텀 지속: 윈도우 w 초과수익 Top3 섹터 → w+1 평균 초과수익 > 0?  (Bottom3 대칭)
- 가설2 이탈 신호: w-1에서 Top3였던 섹터가 w에서 [초과수익<0 AND 거래대금<직전 6윈도우
  피크의 60%] 이면 '이탈' → w+1에서 잔류군보다 언더퍼폼?
- 가설3 수급 강도: w의 (기관합계+외국인합계 순매수합)/(거래대금합) Top3 → w+1 초과수익 > 0?
- 가설4 콤보: 모멘텀 순위 + 수급 순위 합산 Top3 → w+1 초과수익 > 0?
- 판정: 해당 축 평균 초과 > 0 & t ≥ 2 & 연도별 부호 대체로 일관 → 리포트 배선 채택.

한계 (정직 고지): 섹터 구성 = 현재 스냅샷 (생존·소속이동 편향, 축 간 비교엔 공통).

★판정 (2026-07-10 실행, 윈도우 95개): 기각 — 주도 섹터 리포트 배선 안 함.
- 모멘텀Top3 -0.94%p(t-1.66)·수급Top3 -0.54%p(t-1.24)·콤보 -0.92%p(t-2.06):
  KR 20일 호흡에서 섹터 주도성은 지속이 아니라 평균회귀. 2026 폭락기 -5.36%p(모멘텀 크래시).
- 이탈 신호(D-b): 이탈군 -1.18 vs 잔류군 -0.39 — 방향은 지시서와 일치하나 n=28 무의미.
- 콤보의 역방향 유의는 '주도 섹터 페이드' 후보로만 기록 (같은 데이터 부호반전 채택 금지 —
  후속 사전등록 검증 필요). 단일 사전 설계(20일)라 다른 호흡(5일/60일)은 미검증.

실행:
    python -u -X utf8 scripts/backtest_sector_rotation.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SECTORS_JSON = PROJECT_ROOT / "data" / "sector_rotation" / "krx_full_sectors.json"
OUT_DIR = PROJECT_ROOT / "data" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2019-01-01"
WIN = 20      # 비중첩 윈도우 길이 (거래일)
TOP_N = 3
EXIT_VOL_RATIO = 0.6   # 이탈: 거래대금 < 직전 6윈도우 피크의 60% (사전 고정)
EXIT_LOOKBACK = 6


def load_matrices():
    sectors = json.load(open(SECTORS_JSON, encoding="utf-8"))
    members = {name: [s["code"] for s in v["stocks"]] for name, v in sectors.items()}

    ret, flow, tval = {}, {}, {}
    for f in glob.glob(str(RAW_DIR / "*.parquet")):
        code = Path(f).stem
        try:
            df = pd.read_parquet(f, columns=["close", "volume", "기관합계", "외국인합계"])
        except Exception:
            continue
        df = df[(df.index >= START) & (df["close"] > 0)]
        if len(df) < 100:
            continue
        ret[code] = df["close"].pct_change() * 100.0
        flow[code] = df["기관합계"].fillna(0) + df["외국인합계"].fillna(0)
        tval[code] = df["close"] * df["volume"]
    return members, pd.DataFrame(ret), pd.DataFrame(flow), pd.DataFrame(tval)


def main() -> None:
    print("[1/3] 데이터 로드")
    members, ret, flow, tval = load_matrices()
    print(f"  종목 {ret.shape[1]}개, 거래일 {ret.shape[0]}일")

    mkt = ret.mean(axis=1)
    sec_x, sec_flow, sec_tval = {}, {}, {}
    for name, codes in members.items():
        cols = [c for c in codes if c in ret.columns]
        if len(cols) < 3:
            continue
        sec_x[name] = ret[cols].mean(axis=1) - mkt          # 일일 초과수익
        sec_flow[name] = flow[cols].sum(axis=1)
        sec_tval[name] = tval[cols].sum(axis=1)
    X = pd.DataFrame(sec_x)         # 일일 섹터 초과수익 (%p)
    F = pd.DataFrame(sec_flow)
    V = pd.DataFrame(sec_tval)
    print(f"  섹터 {X.shape[1]}개")

    # 비중첩 윈도우 집계
    n_win = len(X) // WIN
    idx = [X.index[i * WIN] for i in range(n_win)]
    wx = pd.DataFrame({s: [X[s].iloc[i*WIN:(i+1)*WIN].sum() for i in range(n_win)] for s in X.columns}, index=idx)
    wf = pd.DataFrame({s: [F[s].iloc[i*WIN:(i+1)*WIN].sum() for i in range(n_win)] for s in F.columns}, index=idx)
    wv = pd.DataFrame({s: [V[s].iloc[i*WIN:(i+1)*WIN].sum() for i in range(n_win)] for s in V.columns}, index=idx)
    wflow_int = wf / wv.replace(0, np.nan)   # 수급 강도 (원/원, 무차원)
    print(f"[2/3] 비중첩 {WIN}일 윈도우 {n_win}개 ({idx[0].date()} ~ {idx[-1].date()})")

    def t_stat(s):
        s = pd.Series(s).dropna()
        n = len(s)
        if n < 8:
            return np.nan, np.nan, n
        return float(s.mean()), float(s.mean() / (s.std(ddof=1) / np.sqrt(n))), n

    results = {}

    # 가설1: 모멘텀 지속 (+ Bottom3 대칭)
    top_next, bot_next = [], []
    for i in range(n_win - 1):
        r = wx.iloc[i].dropna().sort_values(ascending=False)
        top_next.append(wx.iloc[i + 1][r.index[:TOP_N]].mean())
        bot_next.append(wx.iloc[i + 1][r.index[-TOP_N:]].mean())
    results["가설1 모멘텀Top3"] = (t_stat(top_next), pd.Series(top_next, index=idx[1:]))
    results["가설1 모멘텀Bot3"] = (t_stat(bot_next), pd.Series(bot_next, index=idx[1:]))

    # 가설2: 이탈 신호 (w-1 Top3 → w에서 이탈조건 충족 vs 미충족 → w+1 비교)
    exit_next, stay_next = [], []
    for i in range(1, n_win - 1):
        prev_top = wx.iloc[i - 1].dropna().sort_values(ascending=False).index[:TOP_N]
        for s in prev_top:
            vol_peak = wv[s].iloc[max(0, i - EXIT_LOOKBACK):i].max()
            exited = (wx.iloc[i][s] < 0) and (wv[s].iloc[i] < EXIT_VOL_RATIO * vol_peak)
            (exit_next if exited else stay_next).append(wx.iloc[i + 1][s])
    results["가설2 이탈군"] = (t_stat(exit_next), None)
    results["가설2 잔류군"] = (t_stat(stay_next), None)

    # 가설3: 수급 강도 Top3
    fl_next = []
    for i in range(n_win - 1):
        r = wflow_int.iloc[i].dropna().sort_values(ascending=False)
        if len(r) < TOP_N * 2:
            fl_next.append(np.nan)
            continue
        fl_next.append(wx.iloc[i + 1][r.index[:TOP_N]].mean())
    results["가설3 수급Top3"] = (t_stat(fl_next), pd.Series(fl_next, index=idx[1:]))

    # 가설4: 콤보 (모멘텀 순위 + 수급 순위)
    co_next = []
    for i in range(n_win - 1):
        rm = wx.iloc[i].rank(ascending=False)
        rf = wflow_int.iloc[i].rank(ascending=False)
        combo = (rm + rf).dropna().sort_values()
        if len(combo) < TOP_N * 2:
            co_next.append(np.nan)
            continue
        co_next.append(wx.iloc[i + 1][combo.index[:TOP_N]].mean())
    results["가설4 콤보Top3"] = (t_stat(co_next), pd.Series(co_next, index=idx[1:]))

    print("[3/3] 판정 (다음 윈도우 20일 초과수익 %p, 기저선=유니버스)")
    print()
    for name, ((m, t, n), series) in results.items():
        print(f"  {name:12s}: 평균 {m:+.2f}%p | t {t:+.2f} | n={n}")

    print("\n===== 연도별 (다음 윈도우 초과 %p) =====")
    yearly = pd.DataFrame({k: v for k, (_, v) in results.items() if v is not None})
    print(yearly.groupby(yearly.index.year).mean().round(2).to_string())

    # 저장
    yearly.to_parquet(OUT_DIR / "sector_rotation_windows.parquet")
    print(f"\n[저장] {OUT_DIR / 'sector_rotation_windows.parquet'}")
    print("판정 기준: 평균>0 & t≥2 & 연도별 부호 대체로 일관 → 해당 축만 리포트 배선")


if __name__ == "__main__":
    main()
