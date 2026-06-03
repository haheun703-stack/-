"""Regime monitor — C60-only hard gate with a wide observation log.

사장님 결정 (2026-06-03): "판단은 단순하게, 관측은 넓게."

- Hard gate (실전 국면 판단, 종목별 개별):
    close > MA60  -> "BULL"            (강세장 → buyhold/분할)
    close <= MA60 -> "BEAR_TRANSITION" (약세전환 → C60 스위치 ON)
  어제(6/2) robust 3각검증을 통과한 유일한 신호가 C60 하나뿐이므로 hard gate는
  C60 단독으로 고정한다.

- Observations (로그만 — gate 판단에 절대 사용하지 않음):
    * 20일 realized volatility + 변동성 클러스터 경고 (종가 기반, 과거치 완비)
    * KOSPI MA60/MA120 위치 (kospi_index.csv)
    * 외국인 순매수 (kospi_investor_flow.csv, 가용 구간만)
  미검증 지표를 hard gate에 섞으면 룰이 흐려지므로 분리한다. 대신 버리지 않고
  "C60보다 며칠 먼저 경고했는가"만 기록해, 추후 보조 게이트 승격 여부를 판단한다.

This module is read-only with respect to trading. It never imports or calls any
order adapter (실주문 0 / shadow only).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.etf.c60_shadow import normalize_ohlcv
from src.etf.samsung_single_leverage_shadow import load_daily_ohlcv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SHADOW_DIR = PROJECT_ROOT / "data" / "shadow"
KOSPI_CSV = PROJECT_ROOT / "data" / "kospi_index.csv"
INVESTOR_CSV = PROJECT_ROOT / "data" / "kospi_investor_flow.csv"

MA_PERIOD = 60
KOSPI_MA_SHORT = 60
KOSPI_MA_LONG = 120
VOL_WINDOW = 20
VOL_CLUSTER_LOOKBACK = 60
VOL_CLUSTER_MULT = 1.5
TRADING_DAYS = 252
MERGE_TOLERANCE = pd.Timedelta("4D")  # 주말/공휴일은 끌어오되 그 이상 stale은 NaN
LEADLAG_WINDOW = 20  # 국면 전환 전 몇 거래일까지 보조 관측 경고를 추적할지

# 신호원(기초자산) 티커. 사장님: 삼성/SK/반도체 각각 자기 60일선으로 개별 판단.
UNDERLYINGS: dict[str, str] = {
    "488080": "반도체레버 기초자산",
    "005930": "삼성전자",
    "000660": "SK하이닉스",
}

REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR_TRANSITION"


@dataclass(frozen=True)
class RegimeRow:
    date: str
    ticker: str
    name: str
    close: float
    ma60: float
    # --- hard gate (C60 단독) ---
    regime: str
    regime_change: bool
    days_in_regime: int
    # --- observations (로그만, gate 미사용) ---
    realized_vol_20: float | None
    vol_cluster_warn: bool
    kospi_close: float | None
    kospi_ma60: float | None
    kospi_ma120: float | None
    kospi_above_ma60: bool | None
    kospi_above_ma120: bool | None
    kospi_warn: bool
    foreign_net: float | None
    foreign_warn: bool


def _round(value, digits: int = 4):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return round(float(value), digits)


def _safe_bool(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(value)


def _opt_bool(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return bool(value)


def load_kospi() -> pd.DataFrame:
    """KOSPI 지수 + MA60/MA120. 종가 이탈 경고(kospi_warn) 포함."""
    if not KOSPI_CSV.exists():
        logger.warning("kospi csv not found: %s", KOSPI_CSV)
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(KOSPI_CSV)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["kospi_close"] = pd.to_numeric(df["close"], errors="coerce")
    df["kospi_ma60"] = df["kospi_close"].rolling(KOSPI_MA_SHORT).mean()
    df["kospi_ma120"] = df["kospi_close"].rolling(KOSPI_MA_LONG).mean()
    df["kospi_above_ma60"] = df["kospi_close"] > df["kospi_ma60"]
    df["kospi_above_ma120"] = df["kospi_close"] > df["kospi_ma120"]
    df["kospi_warn"] = df["kospi_close"] <= df["kospi_ma60"]
    return df[
        [
            "date",
            "kospi_close",
            "kospi_ma60",
            "kospi_ma120",
            "kospi_above_ma60",
            "kospi_above_ma120",
            "kospi_warn",
        ]
    ]


def load_investor() -> pd.DataFrame:
    """외국인 순매수(KOSPI 전체). 종목별은 추후 KIS adapter로 보강."""
    if not INVESTOR_CSV.exists():
        logger.warning("investor csv not found: %s", INVESTOR_CSV)
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(INVESTOR_CSV)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["foreign_net"] = pd.to_numeric(df["foreign_net"], errors="coerce")
    df["foreign_warn"] = df["foreign_net"] < 0  # 순매도 = 경고
    return df[["date", "foreign_net", "foreign_warn"]]


def _attach_observations(df: pd.DataFrame) -> pd.DataFrame:
    """종목 일봉(df, datetime index)에 KOSPI/외국인 관측을 날짜정렬 병합."""
    base = df.reset_index()
    base = base.rename(columns={base.columns[0]: "date"})
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values("date")

    kospi = load_kospi()
    investor = load_investor()

    if not kospi.empty:
        base = pd.merge_asof(
            base, kospi, on="date", direction="backward", tolerance=MERGE_TOLERANCE
        )
    if not investor.empty:
        base = pd.merge_asof(
            base, investor, on="date", direction="backward", tolerance=MERGE_TOLERANCE
        )
    return base.set_index("date")


def build_regime_ledger(
    ticker: str,
    days: int = 1300,
    prefer_remote: bool = True,
    prices: pd.DataFrame | None = None,
) -> list[RegimeRow]:
    """C60 단독 hard gate + 보조 관측 로그 ledger 생성.

    prefer_remote=True 기본: data/raw parquet가 stale(005930/000660는 3/27 멈춤)
    일 수 있어 매일 도는 국면 판단기는 pykrx 최신을 강제한다.
    prices: 테스트/백테스트용 가격 주입(없으면 load_daily_ohlcv로 로드).
    """
    name = UNDERLYINGS.get(ticker, ticker)
    raw = prices if prices is not None else load_daily_ohlcv(
        ticker, days=days, prefer_remote=prefer_remote
    )
    df = normalize_ohlcv(raw)
    if df.empty or len(df) < MA_PERIOD + 1:
        logger.warning("insufficient data for %s (%d rows)", ticker, 0 if df.empty else len(df))
        return []

    df = df.copy()
    df["ma60"] = df["close"].rolling(MA_PERIOD).mean()
    df["ret1"] = df["close"].pct_change()
    df["realized_vol_20"] = df["ret1"].rolling(VOL_WINDOW).std() * np.sqrt(TRADING_DAYS)
    df["vol_baseline"] = df["realized_vol_20"].rolling(VOL_CLUSTER_LOOKBACK).mean()
    df["vol_cluster_warn"] = df["realized_vol_20"] > df["vol_baseline"] * VOL_CLUSTER_MULT
    df = df.dropna(subset=["ma60"])
    if df.empty:
        return []

    # hard gate: C60 단독
    df["regime"] = np.where(df["close"] > df["ma60"], REGIME_BULL, REGIME_BEAR)
    df["regime_change"] = df["regime"] != df["regime"].shift(1)
    group = (df["regime"] != df["regime"].shift(1)).cumsum()
    df["days_in_regime"] = df.groupby(group).cumcount() + 1

    df = _attach_observations(df)

    rows: list[RegimeRow] = []
    for dt, row in df.iterrows():
        rows.append(
            RegimeRow(
                date=pd.Timestamp(dt).strftime("%Y-%m-%d"),
                ticker=ticker,
                name=name,
                close=_round(row["close"]),
                ma60=_round(row["ma60"]),
                regime=str(row["regime"]),
                regime_change=_safe_bool(row["regime_change"]),
                days_in_regime=int(row["days_in_regime"]),
                realized_vol_20=_round(row.get("realized_vol_20"), 4),
                vol_cluster_warn=_safe_bool(row.get("vol_cluster_warn")),
                kospi_close=_round(row.get("kospi_close")),
                kospi_ma60=_round(row.get("kospi_ma60")),
                kospi_ma120=_round(row.get("kospi_ma120")),
                kospi_above_ma60=_opt_bool(row.get("kospi_above_ma60")),
                kospi_above_ma120=_opt_bool(row.get("kospi_above_ma120")),
                kospi_warn=_safe_bool(row.get("kospi_warn")),
                foreign_net=_round(row.get("foreign_net"), 1),
                foreign_warn=_safe_bool(row.get("foreign_warn")),
            )
        )
    return rows


def lead_lag_summary(rows: list[RegimeRow]) -> list[dict]:
    """각 BULL->BEAR_TRANSITION 전환에 대한 RAW(미보정) lead.

    ⚠️ 착시주의: 이 lead는 base-rate 미보정이다. 보조 경고가 상시 점등이면
    (예: 외국인 순매도 base_rate~0.63) 윈도우 시작부터 켜져 있어 lead가 윈도우
    최대값으로 고정되는 착시가 발생한다. 2026-06-03 적대검증에서 외국인은
    NOISE(precision@5=0.0)로 gate 부적합 확정됨.

    정직한 선행성 판정(rising edge / base_rate / precision)은 반드시
    scripts/research/regime_obs_adversarial.py 로 확인할 것.

    raw_lead(거래일) > 0 : 윈도우 안에서 경고가 (먼저) 켜져 있었음 — 선행 아님 주의
    None                 : 전환 전 LEADLAG_WINDOW 거래일 안에 경고 없음
    """
    if not rows:
        return []
    df = pd.DataFrame([asdict(r) for r in rows]).reset_index(drop=True)
    switches = df.index[(df["regime"] == REGIME_BEAR) & (df["regime_change"])].tolist()

    def first_warn_lead(col: str, sw_idx: int):
        lo = max(0, sw_idx - LEADLAG_WINDOW)
        window = df.iloc[lo : sw_idx + 1]
        warned = window.index[window[col]]
        if len(warned) == 0:
            return None
        return int(sw_idx - warned[0])

    out = []
    for sw_idx in switches:
        out.append(
            {
                "switch_date": df.iloc[sw_idx]["date"],
                "vol_cluster_raw_lead_unadjusted": first_warn_lead("vol_cluster_warn", sw_idx),
                "kospi_raw_lead_unadjusted": first_warn_lead("kospi_warn", sw_idx),
                "foreign_raw_lead_unadjusted": first_warn_lead("foreign_warn", sw_idx),
            }
        )
    return out


# 보조 관측 gate 상태 (2026-06-03 적대검증 확정). hard gate=C60 단독 유지.
OBSERVATION_GATE_STATUS = {
    "foreign_net": "NOISE / gate 제외 (base_rate~0.63 상시점등, precision@5=0.0)",
    "vol_cluster": "WEAK / 로그 유지 (희소하나 전환 예고력 낮음)",
    "kospi_ma60": "CANDIDATE / 로그 추적 — gate 미승격 (삼성 precision 0.53·recall 0.54, SK 0.32 편차)",
}


def build_report(ticker: str, rows: list[RegimeRow]) -> dict:
    if not rows:
        return {"ticker": ticker, "rows": 0}
    last = rows[-1]
    leadlag = lead_lag_summary(rows)
    # 선행성 집계 (값이 있는 전환만)
    def avg_lead(key):
        vals = [s[key] for s in leadlag if s[key] is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "ticker": ticker,
        "name": last.name,
        "rows": len(rows),
        "first_date": rows[0].date,
        "last_date": last.date,
        "hard_gate": "C60 단독 (close vs MA60). 보조 관측은 gate 미사용.",
        "current_regime": last.regime,
        "current_close": last.close,
        "current_ma60": last.ma60,
        "days_in_current_regime": last.days_in_regime,
        "current_observations": {
            "realized_vol_20": last.realized_vol_20,
            "vol_cluster_warn": last.vol_cluster_warn,
            "kospi_above_ma60": last.kospi_above_ma60,
            "kospi_above_ma120": last.kospi_above_ma120,
            "kospi_warn": last.kospi_warn,
            "foreign_net": last.foreign_net,
            "foreign_warn": last.foreign_warn,
        },
        "observation_gate_status": OBSERVATION_GATE_STATUS,
        "bear_switch_count": len(leadlag),
        "raw_lead_unadjusted": {
            "_caution": (
                "base-rate 미보정 RAW lead. 상시점등 관측은 착시(외국인=NOISE). "
                "정직 판정은 scripts/research/regime_obs_adversarial.py 참조."
            ),
            "vol_cluster_raw_lead_unadjusted": avg_lead("vol_cluster_raw_lead_unadjusted"),
            "kospi_raw_lead_unadjusted": avg_lead("kospi_raw_lead_unadjusted"),
            "foreign_raw_lead_unadjusted": avg_lead("foreign_raw_lead_unadjusted"),
        },
        "raw_lead_detail": leadlag,
    }


def save_ledger(ticker: str, rows: list[RegimeRow]) -> tuple[Path, Path]:
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = SHADOW_DIR / f"{ticker}_regime_ledger.json"
    report_path = SHADOW_DIR / f"{ticker}_regime_report.json"
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(build_report(ticker, rows), f, ensure_ascii=False, indent=2)
    return ledger_path, report_path


def run_all(days: int = 1300, prefer_remote: bool = True) -> dict[str, dict]:
    """전체 기초자산 국면 ledger 생성 + 저장. 보고용 dict 반환."""
    reports: dict[str, dict] = {}
    for ticker in UNDERLYINGS:
        rows = build_regime_ledger(ticker, days=days, prefer_remote=prefer_remote)
        if not rows:
            reports[ticker] = {"ticker": ticker, "rows": 0, "error": "no data"}
            continue
        save_ledger(ticker, rows)
        reports[ticker] = build_report(ticker, rows)
    return reports
