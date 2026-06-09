"""KODEX 인버스 헤지 — 타이밍 신호 비교 (C60 휩쏘 대안 검증, read-only).

코덱스 "돈 버는 핵심은 타이밍"의 진짜 시험: C60(가격 후행, 휩쏘)보다 나은
헤지 on/off 신호가 있는가?
  - 고정 헤지(baseline)
  - 변동성 확대(vol20 > vol60) → 헤지 on  (변동성 군집은 선행 가능)
  - 환율 급등(USD/KRW 20일 신고가) → 헤지 on  (외국인 디레버리징)
look-ahead 회피: 모든 신호 전일(shift 1) 기준, 윈도우는 과거만.

별도 트랙. 매매 무관·저장 0(콘솔). 데이터 로드는 explore_kodex_leverage_hedge 재사용.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.explore_kodex_leverage_hedge import load_close  # noqa: E402


def _bt(port_r: pd.Series, label: str) -> None:
    cum = (1 + port_r).cumprod()
    ret = (cum.iloc[-1] - 1) * 100
    mdd = ((cum / cum.cummax() - 1).min()) * 100
    print(f"  {label:<34} {ret:+7.1f}% / MDD {mdd:6.1f}%")


def fixed(lev, inv, w, seg):
    lr = lev.pct_change().fillna(0)
    ir = inv.pct_change().fillna(0)
    _bt((lr + ir * (w / 100)).loc[seg], f"고정 인버스{w}")


def vol_hedge(lev, inv, w, seg):
    lr = lev.pct_change()
    ir = inv.pct_change().fillna(0)
    v20 = lr.rolling(20).std()
    v60 = lr.rolling(60).std()
    on = (v20.shift(1) > v60.shift(1)).astype(float).fillna(0)  # 변동성 확대 = 헤지
    port = lr.fillna(0) + ir * (w / 100) * on
    n = int(on.loc[seg].sum())
    _bt(port.loc[seg], f"변동성확대 인버스{w} ({n}일on)")


def fx_hedge(lev, inv, fx, w, seg):
    lr = lev.pct_change()
    ir = inv.pct_change().fillna(0)
    fxa = fx.reindex(lev.index).ffill()
    hi = fxa.rolling(20).max()
    on = (fxa.shift(1) >= hi.shift(1)).astype(float).fillna(0)  # 환율 20일 신고가 = 헤지
    port = lr.fillna(0) + ir * (w / 100) * on
    n = int(on.loc[seg].sum())
    _bt(port.loc[seg], f"환율신고가 인버스{w} ({n}일on)")


def main():
    import FinanceDataReader as fdr

    fx = fdr.DataReader("USD/KRW")["Close"]
    fx.index = pd.to_datetime(fx.index)

    # ── 2022 약세장 ──
    print("=" * 64)
    print("[2022 약세장] 신호별 헤지 (인버스80% 기준)")
    lev22 = load_close("122630", "20211001", "20221231")
    inv22 = load_close("114800", "20211001", "20221231")
    seg22 = slice("2022-01-01", "2022-12-31")
    fixed(lev22, inv22, 80, seg22)
    vol_hedge(lev22, inv22, 80, seg22)
    fx_hedge(lev22, inv22, fx, 80, seg22)
    print("  (참고: C60동적 인버스80 = -39.5/-41.2 · 레버단독 -46.6/-50.6 · 고정80 -30.1/-33.9)")

    # ── 최근 60거래일 강세장 ──
    print("=" * 64)
    print("[60일 강세장] 신호별 헤지 (인버스80% 기준)")
    lev26 = load_close("122630", "20251001", "20260609")
    inv26 = load_close("114800", "20251001", "20260609")
    seg26 = slice(lev26.index[-60], lev26.index[-1])
    fixed(lev26, inv26, 80, seg26)
    vol_hedge(lev26, inv26, 80, seg26)
    fx_hedge(lev26, inv26, fx, 80, seg26)
    print("  (참고: 레버단독 +125.0/-30.2)")

    print("=" * 64)
    print("판정: 약세장에서 변동성/환율 신호가 고정(-30.1)·C60(-39.5)을 이기면")
    print("'C60보다 나은 타이밍 신호 존재'. 못 이기면 고정이 현실적 답.")


if __name__ == "__main__":
    main()
