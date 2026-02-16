"""
Mode B: 포물선 시작점 탐지기 (Parabola Detector)

기존 Mode A(추세추종)와 병행하는 독립 탐지 모드.
"아직 안 움직인 종목"에서 포물선 직전 패턴을 찾는다.

핵심 조건 (P1~P6):
  P1: BB Squeeze     — 변동성 수축 (bb_width 20일 하위 20%)
  P2: Volume Dry     — 거래량 고갈 (volume < 20일 평균의 60%)
  P3: Curvature Flip — 가격 곡률 음→양 전환 (ema_curvature)
  P4: OBV Divergence — 가격 횡보인데 OBV 상승 (숨은 매집)
  P5: Base Tight     — 20일 가격 범위가 ATR 대비 타이트
  P6: Supply Shift   — 외국인/기관 20일 누적 순매수 양전환

진입: P1+P2+P3 필수 (코어), P4~P6은 부스터
손절: 베이스 하단 (rolling_low_10)
목표: 베이스 높이 × 3
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_parabola(
    ticker: str,
    df: pd.DataFrame,
    idx: int,
    min_core: int = 3,
) -> dict | None:
    """단일 종목의 포물선 시작점 탐지.

    Args:
        ticker: 종목코드
        df: OHLCV + 지표 DataFrame
        idx: 현재 봉 인덱스
        min_core: 최소 코어 조건 충족 수 (기본 3 = P1+P2+P3 모두)

    Returns:
        시그널 dict (탐지 시) 또는 None
    """
    if idx < 30:  # 최소 데이터 필요
        return None

    row = df.iloc[idx]
    close = float(row.get("close", 0))
    if close <= 0:
        return None

    # ── P1: BB Squeeze (변동성 수축) ──
    bb_width = float(row.get("bb_width", 0) or 0)
    if bb_width <= 0:
        p1 = False
    else:
        # 최근 20일 bb_width 중 하위 20% 이내인지
        bb_hist = df["bb_width"].iloc[max(0, idx - 19) : idx + 1].dropna()
        if len(bb_hist) >= 10:
            pct_rank = (bb_hist < bb_width).sum() / len(bb_hist)
            p1 = pct_rank <= 0.15  # 하위 15%
        else:
            p1 = False

    # ── P2: Volume Dry (거래량 고갈) ──
    volume = float(row.get("volume", 0) or 0)
    vol_ma20 = float(row.get("volume_ma20", 0) or 0)
    if vol_ma20 > 0:
        p2 = volume < vol_ma20 * 0.50  # 거래량 고갈 (평균의 50% 미만)
    else:
        p2 = False

    # ── P3: Curvature Flip + 가속도 (곡률 반전: 음→양, 가속도 > 1σ) ──
    curvature = float(row.get("ema_curvature", 0) or 0)
    curvature_prev = float(row.get("ema_curvature_prev", 0) or 0)
    if curvature > 0 and curvature_prev <= 0:
        # 가속도 = 전환 강도
        accel = curvature - curvature_prev
        # 최근 20일 curvature 변동성 대비 정규화
        curv_hist = df["ema_curvature"].iloc[max(0, idx - 19) : idx + 1].dropna()
        curv_std = curv_hist.std() if len(curv_hist) >= 10 else 0
        if curv_std > 0:
            p3 = (accel / curv_std) > 1.0  # 1σ 이상 가속만 인정
        else:
            p3 = True  # std 계산 불가 시 부호 전환만으로 판정
    else:
        p3 = False

    # ── P4: OBV-가격 괴리 (숨은 매집) ──
    obv_trend = str(row.get("obv_trend_5d", ""))
    price_trend = str(row.get("price_trend_5d", ""))
    p4 = obv_trend == "up" and price_trend in ("flat", "down", "sideways", "")

    # ── P5: Base Tight (베이스 타이트) ──
    high_20 = float(row.get("high_20", 0) or 0)
    low_10 = float(row.get("rolling_low_10", 0) or 0)
    atr = float(row.get("atr_14", 0) or 0)
    base_range = high_20 - low_10 if high_20 > 0 and low_10 > 0 else 0
    if base_range > 0 and atr > 0:
        p5 = base_range < atr * 2.0
    else:
        p5 = False

    # ── P6: 수급 전환 (외국인/기관 누적 양전환) — 방향 필터 ──
    foreign_net_20d = float(row.get("foreign_net_20d", 0) or 0)
    foreign_consec = int(row.get("foreign_consecutive_buy", 0) or 0)
    p6 = foreign_net_20d > 0 or foreign_consec >= 3

    # ── 코어 조건 체크: P1+P2+P3 필수 + P6 필수 (방향 필터) ──
    core_flags = [p1, p2, p3]
    core_count = sum(core_flags)
    if core_count < min_core:
        return None
    if not p6:  # 수급 전환 없으면 방향 불명 → 패스
        return None

    # ── 스코어 계산 ──
    booster_count = sum([p4, p5])  # P4, P5만 부스터
    total_score = core_count + 1 + booster_count  # P6 코어(1) + 부스터(0~2) = 4~6

    # 최소 5점 이상 (P1+P2+P3+P6 + 부스터 1개 이상)
    if total_score < 5:
        return None

    # ── 진입/손절/목표 계산 ──
    entry_price = close
    base_low = low_10 if low_10 > 0 else close * 0.95
    stop_loss = base_low * 0.99  # 베이스 하단 약간 아래

    # 손절폭
    risk = entry_price - stop_loss
    if risk <= 0:
        return None

    # 목표: 베이스 높이 × 3 (포물선 기대)
    base_height = high_20 - base_low if high_20 > base_low else atr * 2
    target_price = entry_price + base_height * 3.0
    reward = target_price - entry_price
    rr_ratio = reward / risk if risk > 0 else 0

    if rr_ratio < 2.0:
        return None

    # ── 유동성 필터 ──
    trading_value = float(row.get("trading_value", 0) or 0)
    if trading_value < 1_000_000_000:  # 10억 미만 제외
        return None

    # ── 시그널 생성 ──
    return {
        "ticker": ticker,
        "signal": True,
        "mode": "parabola",
        "date": str(df.index[idx].date()) if hasattr(df.index[idx], "date") else str(df.index[idx]),
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "risk_reward_ratio": round(rr_ratio, 2),
        "atr_value": atr,
        "grade": "B",
        "zone_score": total_score / 6.0,  # 4~6 → 0.67~1.0 스케일
        "trigger_type": "parabola",
        "trigger_confidence": total_score / 6.0,  # 4~6 → 0.67~1.0
        "position_ratio": 0.5,  # Mode B는 작은 포지션
        "entry_stage_pct": 0.30,  # 30%만 초기 진입
        "stop_loss_pct": risk / entry_price,
        "parabola_detail": {
            "p1_bb_squeeze": p1,
            "p2_volume_dry": p2,
            "p3_curvature_flip": p3,
            "p4_obv_diverge": p4,
            "p5_base_tight": p5,
            "p6_supply_shift": p6,
            "core_count": core_count,
            "booster_count": booster_count,
            "total_score": total_score,
            "bb_width": bb_width,
            "vol_ratio": volume / vol_ma20 if vol_ma20 > 0 else 0,
            "curvature": curvature,
            "base_range_atr": base_range / atr if atr > 0 else 0,
            "rr_ratio": rr_ratio,
        },
    }


def scan_parabola_universe(
    data_dict: dict[str, pd.DataFrame],
    idx: int,
    held_tickers: set[str] | None = None,
) -> list[dict]:
    """전종목 포물선 스캔.

    Args:
        data_dict: {ticker: DataFrame} 전종목 데이터
        idx: 현재 봉 인덱스
        held_tickers: 이미 보유 중인 종목 (제외용)

    Returns:
        포물선 시그널 리스트 (total_score 높은 순)
    """
    held = held_tickers or set()
    signals = []

    for ticker, df in data_dict.items():
        if ticker in held:
            continue
        if idx >= len(df):
            continue

        try:
            sig = detect_parabola(ticker, df, idx)
            if sig:
                signals.append(sig)
        except Exception:
            continue

    # total_score 높은 순 → R:R 높은 순
    signals.sort(
        key=lambda s: (
            s["parabola_detail"]["total_score"],
            s["risk_reward_ratio"],
        ),
        reverse=True,
    )

    return signals
