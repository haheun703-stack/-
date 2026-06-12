"""L1 트레이드 레벨 — 리스크 예산 사이징 + 한국 갭 보정 + 유동성 한도 (스펙 §2, Phase 1a).

설계 철학 (§0 — 협상 불가):
1. 수익은 결과, 생존이 목표. 포지션 크기는 신호 강도가 아니라 손절폭이 결정한다.
2. 리스크 관리는 사후 보고서가 아니라 사전 게이트. 사이징은 주문 이전에 끝난다.
3. 모든 모델은 틀린다. 한국 장 손절의 상당수는 장중이 아니라 시초 갭에서 발생하므로
   장중 손절 주문은 보장 수단이 아니다 → 252일 최악 하방 갭으로 사이즈를 재계산한다(§2.2).
4. 노출(exposure)이 아니라 리스크(risk)를 본다. R = 계좌자산 × risk_per_trade가 출발점.

쓰기 3원칙 (6/11 레포 표준) 준수:
- 이 모듈은 순수 계산만 한다. 파일쓰기/네트워크/주문 경로 접촉 = 0 (부작용 0).
- 기록이 필요하면 호출자가 `audit_record()`로 KST ISO 타임스탬프가 박힌 dict를 받아
  명시적으로 저장한다(기본 호출은 아무것도 쓰지 않음).
- fail-closed: ATR/갭/ADV를 계산할 수 없으면(데이터 부족·오염) 사이즈를 키우지 않고
  rejected=True로 차단한다. 모르면 차단.

주의: 하한가 생존 조건(max_single_weight ≤ |daily_kill_limit|/limit_down_pct)은
`risk.config.limit_down_survival_ok`가 담당 — 이 모듈은 캡 적용만 한다.
신호 점수는 사이즈에 사용 금지(진입 여부/우선순위는 이 모듈 밖 관심사).
"""
from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from risk.config import KST, RISK_CONFIG, RiskConfig


def _is_real(x: object) -> bool:
    """numpy 스칼라(np.int64/np.float64) 포함 유한 실수 판정.

    isinstance(x, (int, float))는 np.int64를 놓쳐 정상 입력을 invalid_input으로
    오거부한다 — pandas/numpy 파이프라인 산출값을 받으려면 numbers.Real로 본다.
    bool은 int 서브클래스라 명시 제외(금액에 True/False 유입 차단).
    """
    if isinstance(x, bool):
        return False
    return isinstance(x, numbers.Real) and math.isfinite(float(x))

__all__ = [
    "compute_atr",
    "worst_overnight_gap",
    "adv_krw",
    "SizingResult",
    "size_position",
    "audit_record",
]


# ──────────────────────────────────────────────────────────────────────────────
# 결과 컨테이너 — 모든 중간값을 담아 감사(audit) 가능하게 한다.
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SizingResult:
    """사이징 결과 + 감사용 중간값 전부.

    rejected=True여도 그 시점까지 계산된 중간값(stop_price 등)은 채워서
    '왜 차단됐는가'를 숫자로 재구성할 수 있게 한다.
    """

    size_krw: float                  # 최종 포지션 예산(캡 적용 후, 주식수 절사 전)
    shares: int                      # floor(size_krw / 진입가)
    risk_budget_krw: float           # R = 계좌자산 × risk_per_trade
    stop_price: float                # 최종 손절가 (ATR손절 vs 수급이탈가 중 높은 쪽)
    stop_width: float                # (진입가 − stop_price) / 진입가
    gap_worst: float                 # 최근 252일 최대 하방 오버나이트 갭(양수 비율)
    effective_stop_width: float      # max(stop_width, gap_worst, [sparse_gap_floor]) — 갭 보정 손절폭
    caps_applied: tuple[str, ...]    # 적용된 캡 이름들 ('single_weight_cap', 'adv_cap')
    gap_sample_count: int            # 유효 갭 표본 수 (< gap_min_samples면 sparse_gap_floor 적용·sparse_gap_floor 캡)
    rejected: bool                   # True면 진입 금지 (fail-closed)
    reason: str | None               # 차단 사유 코드 (성공 시 None)


def _rejected(
    reason: str,
    *,
    risk_budget_krw: float = 0.0,
    stop_price: float = 0.0,
    stop_width: float = 0.0,
    gap_worst: float = 0.0,
    effective_stop_width: float = 0.0,
    size_krw: float = 0.0,
    shares: int = 0,
    caps_applied: tuple[str, ...] = (),
    gap_sample_count: int = 0,
) -> SizingResult:
    """차단 결과 생성 헬퍼 — 계산된 중간값은 최대한 보존(감사 목적)."""
    return SizingResult(
        size_krw=size_krw,
        shares=shares,
        risk_budget_krw=risk_budget_krw,
        stop_price=stop_price,
        stop_width=stop_width,
        gap_worst=gap_worst,
        effective_stop_width=effective_stop_width,
        caps_applied=caps_applied,
        gap_sample_count=gap_sample_count,
        rejected=True,
        reason=reason,
    )


# ──────────────────────────────────────────────────────────────────────────────
# §2.1 — Wilder ATR
# ──────────────────────────────────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR(period) — 마지막 시점의 ATR 하나를 반환.

    df: 소문자 high/low/close 컬럼 필수(open은 ATR 계산에 불사용), 행=시간 오름차순.
    TR = max(고저폭, |고가−전일종가|, |저가−전일종가|),
    초기 ATR = 첫 period개 TR 단순평균 → 이후 Wilder 평활 (ATR×(p−1)+TR)/p.

    Raises:
        ValueError: 필수 컬럼 누락, period<1, 유효 행 < period+1, 비유한값 포함.
            (size_position이 이를 잡아 fail-closed로 차단한다.)
    """
    if period < 1:
        raise ValueError(f"period는 1 이상이어야 함: {period}")
    required = ("high", "low", "close")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV 컬럼 누락: {missing}")

    data = df[list(required)].dropna()
    if len(data) < period + 1:
        raise ValueError(f"행 부족: {len(data)}행 < period+1={period + 1}행 (ATR 계산 불가)")

    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    if not (np.isfinite(high).all() and np.isfinite(low).all() and np.isfinite(close).all()):
        raise ValueError("가격에 비유한값(inf 등) 포함 — 데이터 오염, 계산 차단")

    prev_close = close[:-1]
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
    )
    # Wilder 평활: SMA 시드 후 재귀
    atr = float(tr[:period].mean())
    for t in tr[period:]:
        atr = (atr * (period - 1) + float(t)) / period
    return atr


# ──────────────────────────────────────────────────────────────────────────────
# §2.2 — 한국 갭 리스크: 최근 lookback일 최악 하방 오버나이트 갭
# ──────────────────────────────────────────────────────────────────────────────
def _valid_down_gaps(df: pd.DataFrame, lookback: int = 252) -> np.ndarray:
    """최근 lookback 거래일의 하방 오버나이트 갭(양수 비율) 배열. 표본수 = 길이.

    갭 = max(0, −(시가 − 전일종가) / 전일종가). 상승/보합 갭은 0.0으로 포함(표본에는 셈).
    ★dropna로 행을 통째 제거하지 않는다 — NaN이 섞인 행이 실제 최악 갭이었을 때 그 행이
      삭제돼 사이즈가 부풀던 버그(적대리뷰 P1) 방지. 대신 valid 마스크로 해당 갭만 뺀다.
    ★today_open ≤ 0(유령행 OHLCV=0)도 제외 — 시가 0이 만드는 가짜 100% 갭 방지(P2).
    전일종가 ≤ 0·비유한값도 제외(오염 방어).

    Raises:
        ValueError: open/close 컬럼 누락 (size_position이 fail-closed로 흡수).
    """
    required = ("open", "close")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"갭 계산 컬럼 누락: {missing}")
    if lookback < 1:
        raise ValueError(f"lookback은 1 이상이어야 함: {lookback}")

    data = df[list(required)].tail(lookback + 1)  # ★dropna 제거 — valid 마스크로 처리
    if len(data) < 2:
        return np.empty(0, dtype=float)

    open_ = data["open"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    prev_close = close[:-1]
    today_open = open_[1:]

    valid = (
        np.isfinite(prev_close)
        & np.isfinite(today_open)
        & (prev_close > 0)
        & (today_open > 0)  # ★유령행(시가 0) 방어
    )
    if not valid.any():
        return np.empty(0, dtype=float)
    gaps = -(today_open[valid] - prev_close[valid]) / prev_close[valid]
    return np.maximum(0.0, gaps)


def worst_overnight_gap(df: pd.DataFrame, lookback: int = 252) -> float:
    """최근 lookback 거래일 중 최대 하방 오버나이트 갭(양수 비율). 하방 갭 없으면 0.0.

    Raises:
        ValueError: open/close 컬럼 누락 (size_position이 fail-closed로 흡수).
    """
    gaps = _valid_down_gaps(df, lookback)
    return float(gaps.max()) if gaps.size else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# §2.3 — 유동성: 20일 평균 거래대금(ADV)
# ──────────────────────────────────────────────────────────────────────────────
def adv_krw(df: pd.DataFrame, window: int = 20) -> float:
    """최근 window일 평균 거래대금(원). 'trading_value' 컬럼 우선, 없으면 close×volume.

    행 부족 시 있는 만큼만 평균(스펙 명시). 계산 불능(컬럼 전무·전부 NaN·결과 ≤0)이면
    0.0 반환 → size_position의 ADV 캡이 사이즈를 0으로 눌러 fail-closed 차단으로 이어진다.
    """
    if window < 1:
        return 0.0
    if "trading_value" in df.columns:
        series = pd.to_numeric(df["trading_value"], errors="coerce").dropna()
    elif "close" in df.columns and "volume" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")
        series = (close * volume).dropna()
    else:
        return 0.0  # 거래대금을 알 수 없음 → 모르면 차단(상류 캡에서 0)

    series = series.tail(window)
    if series.empty:
        return 0.0
    value = float(series.mean())
    if not math.isfinite(value) or value <= 0:
        return 0.0
    return value


# ──────────────────────────────────────────────────────────────────────────────
# §2 통합 — 포지션 사이징 (순수 계산, 부작용 0)
# ──────────────────────────────────────────────────────────────────────────────
def size_position(
    equity_krw: float,
    entry_price: float,
    ohlcv_df: pd.DataFrame,
    supply_stop_price: float | None = None,
    cfg: RiskConfig = RISK_CONFIG,
) -> SizingResult:
    """리스크 예산 사이징(§2.1) + 갭 보정(§2.2) + 유동성 한도(§2.3).

    절차:
        ① ATR 손절가 = 진입가 − atr_stop_mult × ATR(atr_period)
        ② 수급이탈가가 주어지면 stop = max(ATR손절가, 수급이탈가) — 진입가에 가까운(높은) 쪽
        ③ stop ≥ 진입가 또는 stop_width ≤ 0 → rejected 'invalid_stop'
        ④ effective_stop_width = max(stop_width, 252일 최악 하방 갭)
        ⑤ size = R / effective_stop_width  (R = equity × risk_per_trade)
        ⑥ 캡: equity×max_single_weight 초과 시 축소('single_weight_cap'),
              ADV×adv_limit_ratio 초과 시 축소('adv_cap')
        ⑦ shares = floor(size/진입가), shares < 1 → rejected 'below_min_unit'
        ⑧ 입력 가드: equity≤0·entry≤0·비유한값 → rejected 'invalid_input'

    fail-closed 추가 사유(계약 외 명시): OHLCV 행 부족/컬럼 누락/오염으로
    ATR·갭을 계산할 수 없으면 rejected 'insufficient_data' (모르면 차단).
    신호 점수는 입력에 없다 — 사이즈는 손절폭만으로 결정된다.
    """
    # ⑧ 입력 가드 (가장 먼저 — 쓰레기 입력으로 아래 산식 오염 방지)
    #    _is_real: np.int64/np.float64 등 numpy 스칼라도 허용(정상 입력 오거부 방지),
    #    bool·NaN·inf는 거부 (fail-closed 입력 위생).
    if not _is_real(equity_krw) or not _is_real(entry_price) or equity_krw <= 0 or entry_price <= 0:
        return _rejected("invalid_input")
    if supply_stop_price is not None and not _is_real(supply_stop_price):
        return _rejected("invalid_input")

    equity = float(equity_krw)
    entry = float(entry_price)
    risk_budget = equity * cfg.risk_per_trade  # R

    # ①② 손절가 — ATR·갭 계산 불능이면 fail-closed
    try:
        atr = compute_atr(ohlcv_df, period=cfg.atr_period)
        gaps = _valid_down_gaps(ohlcv_df, lookback=cfg.gap_lookback_days)
    except ValueError:
        return _rejected("insufficient_data", risk_budget_krw=risk_budget)
    gap_worst = float(gaps.max()) if gaps.size else 0.0
    gap_sample_count = int(gaps.size)

    stop_price = entry - cfg.atr_stop_mult * atr
    if supply_stop_price is not None:
        # 둘 중 진입가에 가까운(=더 높은) 쪽 — 더 보수적인 손절
        stop_price = max(stop_price, float(supply_stop_price))

    stop_width = (entry - stop_price) / entry

    # ③ 손절 무효 (stop ≥ entry 또는 폭 ≤ 0 — ATR=0 횡보 데이터 포함)
    if stop_price >= entry or stop_width <= 0:
        return _rejected(
            "invalid_stop",
            risk_budget_krw=risk_budget,
            stop_price=stop_price,
            stop_width=stop_width,
            gap_worst=gap_worst,
            gap_sample_count=gap_sample_count,
        )

    # ④⑤ 갭 보정 후 사이즈
    #     ★갭 표본 부족(짧은 이력=신규상장 등, 갭 위험이 가장 큰 부류)이면 보수 floor —
    #       "모르면 보수적으로 크게 잡는다"(적대리뷰 P1). 충분한 이력이면 floor 미적용.
    caps: list[str] = []
    effective_gap = gap_worst
    if gap_sample_count < cfg.gap_min_samples and cfg.sparse_gap_floor > gap_worst:
        effective_gap = cfg.sparse_gap_floor
        caps.append("sparse_gap_floor")  # 짧은 이력 → 보수 갭 floor 적용(감사 흔적)
    effective_stop_width = max(stop_width, effective_gap)
    size_krw = risk_budget / effective_stop_width

    # ⑥ 캡 (적용 순서: 종목 비중 캡 → 유동성 캡, 둘 다 기록)
    single_cap = equity * cfg.max_single_weight
    if size_krw > single_cap:
        size_krw = single_cap
        caps.append("single_weight_cap")
    adv_cap = adv_krw(ohlcv_df, window=cfg.adv_window) * cfg.adv_limit_ratio
    if size_krw > adv_cap:
        size_krw = adv_cap  # ADV 미산출(0.0) 포함 — 유동성 모르면 사이즈 0 (fail-closed)
        caps.append("adv_cap")

    # ⑦ 주식수 절사
    shares = math.floor(size_krw / entry)
    if shares < 1:
        return _rejected(
            "below_min_unit",
            risk_budget_krw=risk_budget,
            stop_price=stop_price,
            stop_width=stop_width,
            gap_worst=gap_worst,
            effective_stop_width=effective_stop_width,
            size_krw=size_krw,
            shares=0,
            caps_applied=tuple(caps),
            gap_sample_count=gap_sample_count,
        )

    return SizingResult(
        size_krw=size_krw,
        shares=shares,
        risk_budget_krw=risk_budget,
        stop_price=stop_price,
        stop_width=stop_width,
        gap_worst=gap_worst,
        effective_stop_width=effective_stop_width,
        caps_applied=tuple(caps),
        gap_sample_count=gap_sample_count,
        rejected=False,
        reason=None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 감사 기록 헬퍼 — 쓰기 3원칙 ②: 기록은 KST ISO 타임스탬프로 자기상태 선언.
# 이 함수는 dict만 만든다(파일쓰기 없음) — 저장은 호출자가 명시적으로.
# ──────────────────────────────────────────────────────────────────────────────
def audit_record(result: SizingResult, now_kst: datetime | None = None) -> dict:
    """SizingResult → 감사용 dict (KST ISO 타임스탬프 포함, 부작용 0).

    now_kst 주입 가능(테스트 용이성). naive datetime이 주입되면 KST로 간주해
    tz를 부착한다(naive datetime.now() 금지 — VPS UTC 9시간 어긋남 방어).
    """
    ts = now_kst if now_kst is not None else datetime.now(KST)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=KST)
    else:
        ts = ts.astimezone(KST)
    return {"ts_kst": ts.isoformat(), **asdict(result)}
