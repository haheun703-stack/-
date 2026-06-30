"""레짐-게이트 배분 백테스트 — "강세엔 지수/레버, 약세엔 현금/인버스"가 종목질을 이기는가.

배경(2026-06-30 페이퍼 부검): 강세장(KOSPI +51%)에서 봇의 종목질 전 전략이 지수에 대패.
  최고(밸류피보나치) +1.8% << KOSPI +51%. → 가설: "강세레짐엔 종목 고르지 말고 베타(지수/레버)를 먹어라."
핵심 질문: BRAIN 레짐 분류(여기선 가격기반 재현)가 실시간에 충분히 정확해서 **레버를 믿고 태워도 되는가?**
  = 레짐-게이트가 ①Buy&Hold 지수를 이기고 ②무지성 2x레버의 약세장 붕괴를 피하는가.

방법(lookahead 차단):
  - 레짐은 t일 종가까지의 데이터로만 산출(rolling MA/RV) → t+1일 수익에 적용(1일 시프트).
  - 레버리지/인버스는 KOSPI 일일수익률에서 합성(2x/-1x 일일복리) → 변동성 drag 정직 반영.
  - 거래비용/슬리피지: 레짐 전환 시 0.1% 1회 차감(보수적).

레짐(가격기반 4단계, SYSTEM_MAP 근사):
  BULL    : 정배열(close>MA20>MA60) & 저변동(RV20<중앙값)  → 레버리지(2x)
  CAUTION : 정배열 & 고변동(RV20>=중앙값)                 → 지수(1x)
  NEUTRAL : close>MA60 이나 정배열 깨짐                    → 현금
  CRISIS  : close<MA60                                    → 현금(기본) / 인버스(옵션)

사용:
    python -u -X utf8 scripts/backtest_regime_gated_allocation.py
    python -u -X utf8 scripts/backtest_regime_gated_allocation.py --inverse   # CRISIS시 인버스
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CSV = PROJECT_ROOT / "data" / "kospi_index_bak_3_20.csv"
FEE = 0.001   # 레짐 전환 1회 거래비용


def load_kospi() -> pd.DataFrame:
    d = pd.read_csv(CSV)
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").set_index("Date")
    col = "close" if "close" in d.columns else "adj close"
    d = d[[col]].rename(columns={col: "close"}).dropna()
    d = d[d["close"] > 0]
    return d


def classify_regime(d: pd.DataFrame) -> pd.Series:
    """★시스템 실제 레짐 분류기 1:1 재현 (regime_macro_signal.classify_regime + calc_rv_percentile).

    if close>ma20:  BULL(rv_pct<50) / CAUTION(rv_pct>=50)
    elif close>ma60: BEAR
    else: CRISIS
    rv_pct = RV20의 252일 롤링 백분위 순위(과거만, lookahead無).
    """
    close = d["close"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ret = close.pct_change()
    rv = ret.rolling(20).std() * np.sqrt(252) * 100
    rv_pct = (rv.rolling(252, min_periods=60).rank(pct=True) * 100).fillna(50)

    reg = pd.Series("CRISIS", index=d.index)                 # else: close<=ma60
    reg[(close <= ma20) & (close > ma60)] = "BEAR"           # elif: ma60<close<=ma20
    reg[(close > ma20) & (rv_pct >= 50)] = "CAUTION"         # if: close>ma20 고변동
    reg[(close > ma20) & (rv_pct < 50)] = "BULL"             # if: close>ma20 저변동
    reg[ma60.isna()] = "NEUTRAL"                             # warmup
    return reg


def run_policy(d: pd.DataFrame, reg: pd.Series, weights: dict, use_inverse: bool) -> pd.Series:
    """레짐별 자산배분 정책 → 일별 전략수익률 시계열(거래비용 포함)."""
    ret = d["close"].pct_change().fillna(0.0)
    # 자산 일일수익(합성): 지수1x / 레버2x(일일복리) / 인버스-1x / 현금0
    asset_ret = {
        "INDEX": ret,
        "LEV": 2.0 * ret,
        "INV": -1.0 * ret,
        "CASH": pd.Series(0.0, index=d.index),
    }
    # t일 레짐 → t+1일 적용(lookahead 차단)
    reg_lag = reg.shift(1).fillna("NEUTRAL")
    target = reg_lag.map(weights).fillna("CASH")
    if not use_inverse:
        target = target.replace("INV", "CASH")
    # 전략 일일수익
    strat = pd.Series(0.0, index=d.index)
    for asset in ("INDEX", "LEV", "INV", "CASH"):
        strat[target == asset] = asset_ret[asset][target == asset]
    # 거래비용: 타깃자산 바뀌는 날 FEE 차감
    switch = target != target.shift(1)
    strat = strat - switch.astype(float) * FEE
    return strat


def metrics(strat: pd.Series, d: pd.DataFrame) -> dict:
    eq = (1 + strat).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = strat.mean() / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
    return {"total": eq.iloc[-1] - 1, "cagr": cagr, "mdd": mdd, "sharpe": sharpe, "eq": eq}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inverse", action="store_true", help="CRISIS 레짐에 인버스(-1x) 사용")
    ap.add_argument("--start", default="2018-01-01", help="백테스트 시작일")
    args = ap.parse_args()

    d = load_kospi()
    d = d[d.index >= pd.to_datetime(args.start)]
    reg = classify_regime(d)

    # 비교 정책들
    _crisis = "INV" if args.inverse else "CASH"
    policies = {
        "Buy&Hold 지수(1x)": {"BULL": "INDEX", "CAUTION": "INDEX", "BEAR": "INDEX", "NEUTRAL": "INDEX", "CRISIS": "INDEX"},
        "무지성 레버(2x 상시)": {"BULL": "LEV", "CAUTION": "LEV", "BEAR": "LEV", "NEUTRAL": "LEV", "CRISIS": "LEV"},
        "★레짐게이트": {"BULL": "LEV", "CAUTION": "INDEX", "BEAR": "CASH", "NEUTRAL": "CASH", "CRISIS": _crisis},
    }

    print(f"=== 레짐-게이트 배분 백테스트 (KOSPI {d.index[0].date()}~{d.index[-1].date()}, "
          f"{'인버스ON' if args.inverse else '인버스OFF'}) ===")
    print(f"레짐 분포: {dict(reg.value_counts())}\n")

    results = {}
    for name, w in policies.items():
        strat = run_policy(d, reg, w, args.inverse)
        m = metrics(strat, d)
        results[name] = m
        print(f"  {name:20s} 총수익 {m['total']*100:>+7.1f}%  CAGR {m['cagr']*100:>+6.1f}%  "
              f"MDD {m['mdd']*100:>6.1f}%  Sharpe {m['sharpe']:>4.2f}")

    # 연도별 수익 비교
    print("\n=== 연도별 수익률 (레짐게이트가 약세장 방어 + 강세장 참여하는가) ===")
    print(f"  {'연도':6} {'지수1x':>9} {'레버2x':>9} {'레짐게이트':>10}  {'그해 레짐(주된)':>16}")
    eqs = {k: v["eq"] for k, v in results.items()}
    rets = {k: eq.resample("YE").last().pct_change() for k, eq in eqs.items()}
    yr_reg = reg.groupby(reg.index.year).agg(lambda s: s.value_counts().idxmax())
    for yr in sorted(set(d.index.year)):
        try:
            r1 = rets["Buy&Hold 지수(1x)"].get(pd.Timestamp(f"{yr}-12-31"))
            r2 = rets["무지성 레버(2x 상시)"].get(pd.Timestamp(f"{yr}-12-31"))
            r3 = rets["★레짐게이트"].get(pd.Timestamp(f"{yr}-12-31"))
            def fmt(x): return f"{x*100:>+8.1f}%" if pd.notna(x) else "      -  "
            print(f"  {yr:6} {fmt(r1)} {fmt(r2)} {fmt(r3)}   {yr_reg.get(yr,'?'):>16}")
        except Exception:
            pass

    print("\n  ※ 결론 판정: 레짐게이트가 (1)지수1x보다 CAGR↑ & MDD개선 (2)무지성레버의 약세장 붕괴 회피")
    print("     → 둘 다 만족하면 'BRAIN 레짐 믿고 레버 태워도 된다'가 데이터로 지지됨.")
    print("  ※ 한계: 가격기반 레짐(NW 선행신호 미포함)·일일종가 체결가정·세금제외. in-sample 경고.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
