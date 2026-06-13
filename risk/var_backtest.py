"""risk/var_backtest.py — VaR 모델 검증(백테스트). RISK_ENGINE Phase 2d / 스펙 §4.5.

'모델을 의심하는 루프': 각 시점에서 과거 데이터로 산출한 VaR95를 다음날 실현 손익이 초과
(더 큰 손실)했나 기록한다. 모델이 옳으면 VaR95 초과는 평균 5%(95% 신뢰) 빈도로 나타난다.

판정:
  - 초과율 목표밴드 = 3~8% (★Phase 2 완료 기준, 스펙 line 369). 벗어나면 var_limit/t_dist_df
    조정 검토(파라미터 분기 1회 룰 — 손실 직후 감정적 변경 금지).
  - Kupiec POF(proportion of failures) LR 검정: 관측 초과율이 기대(5%)와 통계적으로 다른가.
    LR > 3.84(χ² 1df, 95%)면 모델 기각(초과율 비정상 — 과소/과대 추정).

순수 계산(var_engine 재사용) + CLI(로컬 parquet 종목 백테스트). 파일 write 0.
사용: python -u -X utf8 risk/var_backtest.py --ticker 005930 --window 250
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# 직접 실행(python risk/var_backtest.py) 시 프로젝트 루트를 path에 (CLAUDE.md 규칙)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.config import RISK_CONFIG, RiskConfig  # noqa: E402
from risk.var_engine import compute_portfolio_var  # noqa: E402

_CHI2_1DF_95 = 3.841459  # χ²(1df, 95%) 임계


@dataclass(frozen=True)
class BacktestResult:
    n_eval: int
    n_exceed: int
    exceed_rate: float
    expected_rate: float
    kupiec_lr: float
    kupiec_reject: bool    # True = 모델 기각(초과율이 기대와 유의하게 다름)
    in_target_band: bool   # 3~8% (Phase 2 완료 기준)
    reason: str = "ok"


def _kupiec_lr(n: int, k: int, p: float) -> float:
    """Kupiec POF likelihood ratio. n 관측 중 k 초과, 기대 확률 p.

    LR = -2[ln L(p) - ln L(k/n)], L(q) = (1-q)^(n-k) · q^k. χ²(1df) 분포.
    k=0 또는 경계는 0(검정 불능)으로 반환(샘플 적을 때 보수).
    """
    if n <= 0 or k <= 0 or k >= n:
        return 0.0
    pi = k / n
    ln_lp = (n - k) * math.log(1.0 - p) + k * math.log(p)
    ln_lpi = (n - k) * math.log(1.0 - pi) + k * math.log(pi)
    return -2.0 * (ln_lp - ln_lpi)


def backtest_var(
    returns: pd.Series,
    *,
    window: int = 250,
    cfg: RiskConfig = RISK_CONFIG,
    target_band: tuple = (0.03, 0.08),
) -> BacktestResult:
    """롤링 VaR95 백테스트 (단일 종목/시계열).

    각 시점 t(≥window): returns[t-window:t]로 VaR95 산출 → returns[t] 실현이 -VaR95보다
    작으면(손실이 더 큼) 초과(exception). var_engine ok=False 구간은 건너뛴다(평가 제외).
    """
    r = returns.dropna()
    if len(r) <= window:
        return BacktestResult(0, 0, float("nan"), 0.05, 0.0, False, False,
                              reason=f"insufficient:{len(r)}<=window{window}")

    exceeds = 0
    evals = 0
    for t in range(window, len(r)):
        hist = r.iloc[t - window:t]
        vr = compute_portfolio_var({"X": hist}, {"X": 1.0}, cfg=cfg, lookback=window)
        if not vr.ok:
            continue
        evals += 1
        if float(r.iloc[t]) < -vr.var95:
            exceeds += 1

    if evals == 0:
        return BacktestResult(0, 0, float("nan"), 0.05, 0.0, False, False, reason="no_valid_windows")

    rate = exceeds / evals
    expected = 0.05  # VaR95
    lr = _kupiec_lr(evals, exceeds, expected)
    return BacktestResult(
        n_eval=evals, n_exceed=exceeds, exceed_rate=rate, expected_rate=expected,
        kupiec_lr=lr, kupiec_reject=lr > _CHI2_1DF_95,
        in_target_band=(target_band[0] <= rate <= target_band[1]),
    )


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="VaR95 롤링 백테스트 (Phase 2d)")
    parser.add_argument("--ticker", default="005930")
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--days", type=int, default=1260)
    args = parser.parse_args()

    from src.etf.samsung_single_leverage_shadow import load_daily_ohlcv
    from src.use_cases.gate_wiring import _returns_from_ohlcv

    df = load_daily_ohlcv(args.ticker, days=args.days)
    r = _returns_from_ohlcv(df)
    if r is None or len(r) <= args.window:
        n = 0 if r is None else len(r)
        print(f"데이터 부족: {args.ticker} (returns={n}, window={args.window}). "
              f"로컬 parquet 없거나 짧음(pykrx는 KRX 만료로 차단).")
        return 1

    res = backtest_var(r, window=args.window)
    print(f"=== VaR95 백테스트: {args.ticker} (window={args.window}, returns={len(r)}) ===")
    print(f"  평가일 {res.n_eval} / 초과 {res.n_exceed}")
    print(f"  초과율 {res.exceed_rate * 100:.2f}%  (기대 5%, 목표밴드 3~8%)")
    print(f"  Kupiec LR {res.kupiec_lr:.2f}  ({'기각=모델 비정상' if res.kupiec_reject else '통과=정상'})")
    print(f"  Phase 2 완료 기준(초과율 3~8%): {'OK' if res.in_target_band else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
