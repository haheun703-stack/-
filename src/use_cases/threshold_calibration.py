"""§13-3 임계값 백테스트 캘리브레이션 (P2 §13-7 #11, 자기반성 #5 해소)

설계 문서: docs/02-design/threshold-calibration-design.md (2026-05-18 작성)
일정: 5/18 골격 → 5/19~5/21 sweep 실행 → 5/22 결과 + §13-3 교체 PR

캘리브레이션 4종:
  (1) 체결강도 임계값 (현재 ≥150, 분봉 근사로 검증)
  (2) 거래량 임계값 (현재 5분 ≥5x, 5분봉 sweep)
  (3) D+1 적중률 승급 기준 (현재 ≥50%, 일봉 sweep)
  (4) 진입 가드 매트릭스 (Vol_Ratio × 외인 × 기관 = 27조합)

데이터:
  - 정보봇 일봉 39컬럼: data/external/jgis_ohlcv/*.csv (2,632종목, ~250일)
  - 자체 5분봉: data/intraday/5min/{date}/{ticker}.parquet (≈30거래일)
  - 자체 15분봉: data/intraday/15min/{date}/{ticker}.parquet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JGIS_OHLCV_DIR = PROJECT_ROOT / "data" / "external" / "jgis_ohlcv"
INTRADAY_5MIN_DIR = PROJECT_ROOT / "data" / "intraday" / "5min"
INTRADAY_15MIN_DIR = PROJECT_ROOT / "data" / "intraday" / "15min"
CALIBRATION_OUT_DIR = PROJECT_ROOT / "data" / "calibration"

# 수수료/세금 (KIS 키움 기준, 2026 현행)
FEE_RATE = 0.00015  # 매수·매도 각 0.015%
TAX_RATE_KOSPI = 0.0018  # 매도 시 코스피 0.18%
TAX_RATE_KOSDAQ = 0.0020  # 매도 시 코스닥 0.20%

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Entity: 캘리브레이션 결과
# ─────────────────────────────────────────────


@dataclass
class CalibrationResult:
    """단일 임계값 조합의 캘리브레이션 결과."""

    sweep_id: str  # "D1_acc_50pct_strong_bull"
    target: str  # "(3)_D1_accuracy" | "(2)_volume_threshold" | "(4)_entry_guard" | "(1)_strength"
    params: dict  # 임계값 파라미터 (e.g., {"d1_threshold": 0.50, "regime": "STRONG_BULL"})
    n_samples: int  # 표본 수
    n_hits: int  # 적중 건수
    accuracy_d1: float  # D+1 적중률
    accuracy_d3: float  # D+3 적중률
    accuracy_d5: float  # D+5 적중률
    mean_return_d1: float  # 평균 D+1 수익률 (수수료/세금 차감 후)
    mean_return_d3: float
    mean_return_d5: float
    profit_factor: float  # PF (수익 합 / 손실 합)
    max_drawdown: float  # MDD
    note: str = ""


@dataclass
class SweepResult:
    """전체 sweep 결과 (여러 CalibrationResult 묶음)."""

    run_id: str  # ISO timestamp
    target: str  # (1)~(4) 중 하나
    results: list[CalibrationResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None


# ─────────────────────────────────────────────
# 일봉 로더 (정보봇 39컬럼 CSV)
# ─────────────────────────────────────────────


def load_jgis_daily(ticker: str, min_rows: int = 60) -> pd.DataFrame | None:
    """정보봇 39컬럼 일봉 CSV 로드.

    Args:
        ticker: 6자리 종목코드 (예: "005930")
        min_rows: 최소 행 수 (이보다 적으면 None 반환)

    Returns:
        pd.DataFrame with Date as index, 또는 None
    """
    pattern = f"*_{ticker}.csv"
    matches = list(JGIS_OHLCV_DIR.glob(pattern))
    if not matches:
        logger.debug("CSV 없음: %s", ticker)
        return None

    try:
        df = pd.read_csv(matches[0], parse_dates=["Date"])
        if len(df) < min_rows:
            logger.debug("데이터 부족: %s (%d행)", ticker, len(df))
            return None
        df = df.sort_values("Date").set_index("Date")
        return df
    except Exception as e:
        logger.debug("CSV 로드 실패: %s — %s", ticker, e)
        return None


def list_available_tickers(min_rows: int = 60) -> list[str]:
    """정보봇 CSV에 존재하는 모든 종목코드 추출."""
    tickers = []
    for csv_path in JGIS_OHLCV_DIR.glob("*.csv"):
        # 파일명 패턴: "종목명_티커.csv"
        stem = csv_path.stem
        if "_" not in stem:
            continue
        ticker = stem.rsplit("_", 1)[-1]
        if not ticker.isalnum():
            continue
        tickers.append(ticker)
    return tickers


# ─────────────────────────────────────────────
# 분봉 로더 (5min / 15min parquet)
# ─────────────────────────────────────────────


def load_intraday_5min(ticker: str, date: str) -> pd.DataFrame | None:
    """5분봉 parquet 로드.

    Args:
        ticker: 6자리 종목코드
        date: "YYYY-MM-DD"

    Returns:
        pd.DataFrame with columns [time, open, high, low, close, volume], 또는 None
    """
    path = INTRADAY_5MIN_DIR / date / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.debug("5min parquet 로드 실패: %s/%s — %s", date, ticker, e)
        return None


def load_intraday_15min(ticker: str, date: str) -> pd.DataFrame | None:
    """15분봉 parquet 로드. 5분봉과 동일 인터페이스."""
    path = INTRADAY_15MIN_DIR / date / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.debug("15min parquet 로드 실패: %s/%s — %s", date, ticker, e)
        return None


# ─────────────────────────────────────────────
# D+1/D+3/D+5 라벨러
# ─────────────────────────────────────────────


def label_forward_returns(
    df: pd.DataFrame,
    apply_cost: bool = True,
    market: str = "KOSPI",
) -> pd.DataFrame:
    """각 일자에 대해 D+1/D+3/D+5 종가 기준 수익률 라벨링.

    Args:
        df: load_jgis_daily 결과 (Date index + 'Close' 컬럼 포함)
        apply_cost: 수수료/세금 차감 여부 (기본 True)
        market: "KOSPI" | "KOSDAQ" (세율 차이)

    Returns:
        df + [ret_d1, ret_d3, ret_d5, hit_d1, hit_d3, hit_d5] 컬럼 추가
    """
    if df is None or "Close" not in df.columns:
        return df

    out = df.copy()
    tax = TAX_RATE_KOSDAQ if market == "KOSDAQ" else TAX_RATE_KOSPI
    cost = (2 * FEE_RATE + tax) if apply_cost else 0.0

    for d, col_ret, col_hit in [(1, "ret_d1", "hit_d1"), (3, "ret_d3", "hit_d3"), (5, "ret_d5", "hit_d5")]:
        future_close = out["Close"].shift(-d)
        gross = (future_close - out["Close"]) / out["Close"]
        out[col_ret] = gross - cost  # 거래 비용 차감
        out[col_hit] = out[col_ret] > 0

    return out


# ─────────────────────────────────────────────
# 시그널 평가 인터페이스 (각 sweep이 구현)
# ─────────────────────────────────────────────


def evaluate_signal_d1_accuracy(df: pd.DataFrame, **params) -> pd.Series:
    """[(3)] D+1 적중률 임계값 평가 — placeholder.

    5/20 구현 예정. 현재는 모든 행 False 반환.
    """
    logger.warning("[(3) sweep] 미구현 — 5/20 예정")
    return pd.Series(False, index=df.index)


def evaluate_signal_entry_guard(df: pd.DataFrame, **params) -> pd.Series:
    """[(4)] 진입 가드 매트릭스 평가 — placeholder.

    5/20 구현 예정. params: vol_ratio_min, foreign_min, inst_min
    """
    logger.warning("[(4) sweep] 미구현 — 5/20 예정")
    return pd.Series(False, index=df.index)


def evaluate_signal_volume_5min(df_5min: pd.DataFrame, **params) -> pd.Series:
    """[(2)] 5분봉 거래량 임계값 평가 — placeholder.

    5/21 구현 예정. params: vol_multiplier
    """
    logger.warning("[(2) sweep] 미구현 — 5/21 예정")
    return pd.Series(False, index=df_5min.index)


def evaluate_signal_strength_proxy(df_5min: pd.DataFrame, **params) -> pd.Series:
    """[(1)] 체결강도 분봉 근사 평가 — placeholder.

    5/21 구현 예정. params: strength_pct_min
    """
    logger.warning("[(1) sweep] 미구현 — 5/21 예정")
    return pd.Series(False, index=df_5min.index)


# ─────────────────────────────────────────────
# 통계 집계
# ─────────────────────────────────────────────


def summarize_signal(
    df_labeled: pd.DataFrame,
    signal: pd.Series,
    sweep_id: str,
    target: str,
    params: dict,
) -> CalibrationResult:
    """시그널이 True인 행에 대해 D+1/D+3/D+5 통계 산출."""
    triggered = df_labeled[signal.fillna(False)]
    n = len(triggered)

    if n == 0:
        return CalibrationResult(
            sweep_id=sweep_id,
            target=target,
            params=params,
            n_samples=0,
            n_hits=0,
            accuracy_d1=0.0,
            accuracy_d3=0.0,
            accuracy_d5=0.0,
            mean_return_d1=0.0,
            mean_return_d3=0.0,
            mean_return_d5=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            note="표본 0건",
        )

    ret_d1 = triggered["ret_d1"].dropna()
    ret_d3 = triggered["ret_d3"].dropna()
    ret_d5 = triggered["ret_d5"].dropna()

    profits = ret_d1[ret_d1 > 0].sum()
    losses = abs(ret_d1[ret_d1 < 0].sum())
    pf = float(profits / losses) if losses > 0 else float("inf")

    cum = (1 + ret_d1.fillna(0)).cumprod()
    mdd = float((cum / cum.cummax() - 1).min()) if len(cum) > 0 else 0.0

    return CalibrationResult(
        sweep_id=sweep_id,
        target=target,
        params=params,
        n_samples=n,
        n_hits=int(triggered["hit_d1"].sum()),
        accuracy_d1=float(triggered["hit_d1"].mean()),
        accuracy_d3=float(triggered["hit_d3"].mean()),
        accuracy_d5=float(triggered["hit_d5"].mean()),
        mean_return_d1=float(ret_d1.mean()) if len(ret_d1) > 0 else 0.0,
        mean_return_d3=float(ret_d3.mean()) if len(ret_d3) > 0 else 0.0,
        mean_return_d5=float(ret_d5.mean()) if len(ret_d5) > 0 else 0.0,
        profit_factor=pf,
        max_drawdown=mdd,
    )


# ─────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────


def save_sweep_result(sweep: SweepResult) -> Path:
    """SweepResult를 parquet + json 2개 형식으로 저장."""
    CALIBRATION_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = CALIBRATION_OUT_DIR / f"threshold_sweep_{sweep.run_id}.parquet"
    out_json = CALIBRATION_OUT_DIR / f"threshold_summary_{sweep.run_id}.json"

    rows = []
    for r in sweep.results:
        row = {
            "sweep_id": r.sweep_id,
            "target": r.target,
            "n_samples": r.n_samples,
            "n_hits": r.n_hits,
            "accuracy_d1": r.accuracy_d1,
            "accuracy_d3": r.accuracy_d3,
            "accuracy_d5": r.accuracy_d5,
            "mean_return_d1": r.mean_return_d1,
            "mean_return_d3": r.mean_return_d3,
            "mean_return_d5": r.mean_return_d5,
            "profit_factor": r.profit_factor,
            "max_drawdown": r.max_drawdown,
            "note": r.note,
            **{f"param_{k}": v for k, v in r.params.items()},
        }
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_parquet(out_parquet, index=False)

    import json

    summary = {
        "run_id": sweep.run_id,
        "target": sweep.target,
        "started_at": sweep.started_at.isoformat(),
        "finished_at": (sweep.finished_at or datetime.now()).isoformat(),
        "n_combinations": len(sweep.results),
        "best_pf": max((r.profit_factor for r in sweep.results if r.n_samples >= 50), default=0.0),
        "best_acc_d1": max((r.accuracy_d1 for r in sweep.results if r.n_samples >= 50), default=0.0),
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("저장 완료: %s, %s", out_parquet.name, out_json.name)
    return out_parquet
