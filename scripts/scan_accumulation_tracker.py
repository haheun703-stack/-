"""
세력 매집 추적기 — Volume Explosion → Multi-Phase Rally Detector

차트 패턴 실증:
  거래량 폭발 → 점진적 상승 → 조정 → 급상승 → 반복

접근법:
  1. 과거 60일간 모든 parquet에서 vol_z >= 3.0 또는 VSR >= 3.0 발생 이력 탐색
  2. 폭발 이후 OBV/TRIX/기관매수/가격구조를 분석하여 매집 진행 여부 판정
  3. 페이즈 분류:
     Phase 1 (폭발): 폭발일 0~5일 이내
     Phase 2 (매집): 5~25일, 거래량↓ + OBV↑ + 가격 횡보/상승
     Phase 3 (재돌파): TRIX 0선 돌파 + MACD 상승 + 거래량 재증가 조짐
     Phase 4 (가속): 2차 거래량 폭발 + 파라볼릭 상승
  4. 매집 점수 (0~100) 기반 추천

출력: data/accumulation_tracker.json
BAT-D 12.7단계 (수급폭발 이후)

Usage:
    python scripts/scan_accumulation_tracker.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
OUTPUT_PATH = DATA_DIR / "accumulation_tracker.json"

# ── 파라미터 ──
SPIKE_VOL_Z = 3.0           # 폭발 임계치 (vol_z)
SPIKE_VSR = 3.0             # 폭발 임계치 (volume_surge_ratio)
LOOKBACK_DAYS = 60          # 과거 몇일까지 폭발 이력 탐색
MIN_SPIKE_AGE = 3           # 폭발 후 최소 경과일 (너무 이른건 Phase1)
MIN_SCORE = 40              # 최소 점수 (이하는 출력 안함)


def build_name_map() -> dict[str, str]:
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name_map[parts[1]] = parts[0]
    return name_map


def find_historical_spikes(df: pd.DataFrame, lookback: int = LOOKBACK_DAYS) -> list[dict]:
    """과거 lookback일 내 거래량 폭발 이력 탐색.

    Returns: [{idx, date, close, high, vol_z, vsr, volume}, ...]
    """
    if len(df) < lookback + 5:
        lookback = max(len(df) - 5, 10)

    recent = df.tail(lookback)
    spikes = []

    for i, (date_idx, row) in enumerate(recent.iterrows()):
        vol_z = float(row.get("vol_z", 0) or 0)
        vsr = float(row.get("volume_surge_ratio", 0) or 0)
        if pd.isna(vol_z):
            vol_z = 0
        if pd.isna(vsr):
            vsr = 0

        if vol_z >= SPIKE_VOL_Z or vsr >= SPIKE_VSR:
            spikes.append({
                "row_idx": len(df) - len(recent) + i,
                "date": str(date_idx)[:10] if hasattr(date_idx, "strftime") else str(date_idx)[:10],
                "close": float(row.get("close", 0)),
                "high": float(row.get("high", 0)),
                "vol_z": round(vol_z, 2),
                "vsr": round(vsr, 2),
                "volume": int(row.get("volume", 0)),
            })

    return spikes


def analyze_post_spike(df: pd.DataFrame, spike: dict) -> dict | None:
    """폭발 이후 매집 진행 상황 분석.

    Returns: 분석 결과 딕셔너리 또는 None (분석 불가)
    """
    spike_idx = spike["row_idx"]
    total_rows = len(df)
    days_since = total_rows - 1 - spike_idx  # 폭발 후 경과 거래일

    if days_since < MIN_SPIKE_AGE:
        return None  # 너무 이르면 스킵

    # 폭발 이후 데이터
    post_df = df.iloc[spike_idx:]
    if len(post_df) < 3:
        return None

    last = df.iloc[-1]
    spike_close = spike["close"]
    spike_high = spike["high"]
    current_close = float(last.get("close", 0))

    if spike_close <= 0 or current_close <= 0:
        return None

    # 폭발 이후 수익률
    return_since_spike = (current_close / spike_close - 1) * 100

    # -30% 이상 급락한 종목은 매집 실패로 판정
    if return_since_spike < -30:
        return None

    # ── 4축 매집 점수 계산 ──

    # 축1: OBV 추세 (25점)
    obv_score = _score_obv_trend(df, spike_idx)

    # 축2: 기관/외인 매집 패턴 (25점)
    flow_score = _score_institutional_flow(df, spike_idx)

    # 축3: TRIX/MACD 모멘텀 (25점)
    momentum_score = _score_momentum(df, spike_idx)

    # 축4: 가격 구조 (25점)
    structure_score = _score_price_structure(df, spike_idx)

    total = obv_score + flow_score + momentum_score + structure_score

    # 페이즈 분류
    phase = _classify_phase(days_since, df, spike_idx)

    return {
        "days_since_spike": days_since,
        "spike_date": spike["date"],
        "spike_close": int(spike_close),
        "spike_high": int(spike_high),
        "spike_vol_z": spike["vol_z"],
        "current_close": int(current_close),
        "return_since_spike": round(return_since_spike, 1),
        "phase": phase,
        "total_score": round(total, 1),
        "obv_score": obv_score,
        "flow_score": flow_score,
        "momentum_score": momentum_score,
        "structure_score": structure_score,
    }


def _score_obv_trend(df: pd.DataFrame, spike_idx: int) -> int:
    """축1: OBV 추세 점수 (0~25).

    핵심: 폭발 이후 OBV가 하락하지 않으면 매집 진행 중.
    - OBV가 spike 시점 대비 상승: 20~25점
    - OBV 횡보(±5%): 10~15점
    - OBV 하락: 0~5점
    """
    if "obv" not in df.columns:
        return 10  # 데이터 없으면 중립

    post_df = df.iloc[spike_idx:]
    if len(post_df) < 3:
        return 10

    spike_obv = float(post_df.iloc[0]["obv"])
    current_obv = float(post_df.iloc[-1]["obv"])

    if spike_obv == 0:
        return 10

    obv_change_pct = (current_obv / spike_obv - 1) * 100

    # OBV 5일 추세 (최근)
    obv_5d_trend = float(df.iloc[-1].get("obv_trend_5d", 0) or 0)

    score = 0

    # OBV 절대 변화
    if obv_change_pct > 10:
        score += 18
    elif obv_change_pct > 5:
        score += 15
    elif obv_change_pct > 0:
        score += 12
    elif obv_change_pct > -5:
        score += 8
    else:
        score += 3

    # 최근 5일 OBV 추세 보너스
    if obv_5d_trend > 0:
        score += 7
    elif obv_5d_trend == 0:
        score += 4
    else:
        score += 0

    return min(score, 25)


def _score_institutional_flow(df: pd.DataFrame, spike_idx: int) -> int:
    """축2: 기관/외인 매집 패턴 (0~25).

    핵심: 폭발 이후 외인/기관이 계속 사는가?
    - 외인 연속매수: 높은 점수
    - 기관 5일 순매수: 보너스
    - 동시매수: 추가 보너스
    """
    last = df.iloc[-1]
    score = 0

    # 외인 연속매수일
    foreign_consec = int(last.get("foreign_consecutive_buy", 0) or 0)
    if foreign_consec >= 5:
        score += 12
    elif foreign_consec >= 3:
        score += 8
    elif foreign_consec >= 1:
        score += 4

    # 외인 5일 순매수
    foreign_5d = float(last.get("foreign_net_5d", 0) or 0)
    if pd.isna(foreign_5d):
        foreign_5d = 0
    if foreign_5d > 0:
        score += 5
    elif foreign_5d > -1e6:
        score += 2

    # 기관 5일 순매수 (수동 계산)
    if "기관합계" in df.columns:
        inst_5d = float(df["기관합계"].tail(5).sum())
        if pd.isna(inst_5d):
            inst_5d = 0
        if inst_5d > 0:
            score += 5
        elif inst_5d > -1e6:
            score += 2
    else:
        inst_5d = 0

    # 동시매수 보너스
    if foreign_5d > 0 and inst_5d > 0:
        score += 3

    return min(score, 25)


def _score_momentum(df: pd.DataFrame, spike_idx: int) -> int:
    """축3: TRIX/MACD 모멘텀 (0~25).

    핵심: TRIX 0선 돌파 = 중기 상승 확정, MACD 확장 = 단기 가속
    """
    last = df.iloc[-1]
    score = 0

    # TRIX 0선 위치
    trix = float(last.get("trix", 0) or 0)
    trix_signal = float(last.get("trix_signal", 0) or 0)
    trix_gx = bool(last.get("trix_golden_cross", 0))

    if trix > 0 and trix > trix_signal:
        score += 12  # TRIX 0선 위 + 시그널 위 = 강세
    elif trix > 0:
        score += 8   # TRIX 0선 위
    elif trix > trix_signal:
        score += 5   # TRIX 상승 중이지만 아직 0선 아래
    else:
        score += 0

    if trix_gx:
        score += 3  # 골든크로스 보너스

    # MACD 히스토그램 방향
    macd_hist = float(last.get("macd_histogram", 0) or 0)
    macd_hist_prev = float(last.get("macd_histogram_prev", 0) or 0)

    if macd_hist > 0 and macd_hist > macd_hist_prev:
        score += 7  # MACD 양수 + 확장
    elif macd_hist > 0:
        score += 5  # MACD 양수
    elif macd_hist > macd_hist_prev:
        score += 3  # MACD 개선 중
    else:
        score += 0

    # RSI 적정대 보너스 (40~65)
    rsi = float(last.get("rsi_14", 50) or 50)
    if 40 <= rsi <= 65:
        score += 3
    elif 35 <= rsi <= 70:
        score += 1

    return min(score, 25)


def _score_price_structure(df: pd.DataFrame, spike_idx: int) -> int:
    """축4: 가격 구조 (0~25).

    핵심: Higher Low + MA 지지 + 적정 조정폭
    """
    last = df.iloc[-1]
    post_df = df.iloc[spike_idx:]
    if len(post_df) < 3:
        return 10

    score = 0
    close = float(last.get("close", 0))
    ma20 = float(last.get("sma_20", 0) or 0)
    ma60 = float(last.get("sma_60", 0) or 0)

    # MA20 위
    if close > ma20 > 0:
        score += 5
    # MA60 위
    if close > ma60 > 0:
        score += 5

    # Higher Low 구조: 최근 10일 최저가 > 폭발 후 5일내 최저가
    spike_close = float(post_df.iloc[0]["close"])
    early_low = float(post_df.head(min(10, len(post_df)))["low"].min())
    recent_low = float(df.tail(10)["low"].min()) if len(df) >= 10 else early_low

    if recent_low > early_low:
        score += 5  # Higher Low
    elif recent_low >= early_low * 0.97:
        score += 3  # 유사 수준 유지

    # 폭발 후 최고점 대비 조정폭
    post_high = float(post_df["high"].max())
    if post_high > 0:
        drawdown_from_peak = (close / post_high - 1) * 100
        # -3~-15% 조정이 이상적 (눌림목)
        if -15 <= drawdown_from_peak <= -3:
            score += 7  # 적정 조정 구간
        elif -3 < drawdown_from_peak <= 0:
            score += 5  # 고점 근처 (강세)
        elif -20 <= drawdown_from_peak < -15:
            score += 3  # 약간 깊은 조정
        else:
            score += 0  # 과도한 하락

    # 추세선 기울기 (linreg_slope_20)
    slope = float(last.get("linreg_slope_20", 0) or 0)
    if slope > 0:
        score += 3

    return min(score, 25)


def _classify_phase(days_since: int, df: pd.DataFrame, spike_idx: int) -> str:
    """폭발 이후 현재 페이즈 분류."""
    last = df.iloc[-1]
    trix = float(last.get("trix", 0) or 0)
    vol_z = float(last.get("vol_z", 0) or 0)
    if pd.isna(vol_z):
        vol_z = 0
    vsr = float(last.get("volume_surge_ratio", 0) or 0)
    if pd.isna(vsr):
        vsr = 0

    # Phase 4: 2차 거래량 폭발 (다시 vol_z >= 2.5)
    if days_since >= 10 and (vol_z >= 2.5 or vsr >= 2.5):
        return "가속"

    # Phase 3: TRIX 0선 돌파 + 10일 이상 경과
    if days_since >= 8 and trix > 0:
        return "재돌파"

    # Phase 2: 매집 구간
    if days_since >= MIN_SPIKE_AGE:
        return "매집"

    return "폭발"


def scan_all_stocks(name_map: dict) -> list[dict]:
    """전체 parquet 스캔하여 매집 진행 중인 종목 탐지."""
    results = []

    for pq in sorted(PROCESSED_DIR.glob("*.parquet")):
        ticker = pq.stem
        try:
            df = pd.read_parquet(pq)
            if len(df) < 30:
                continue

            # 과거 폭발 이력 탐색
            spikes = find_historical_spikes(df, LOOKBACK_DAYS)
            if not spikes:
                continue

            # 가장 최근 폭발만 사용 (가장 관련성 높음)
            latest_spike = spikes[-1]

            # 폭발 이후 매집 분석
            analysis = analyze_post_spike(df, latest_spike)
            if analysis is None:
                continue

            if analysis["total_score"] < MIN_SCORE:
                continue

            # 멀티 스파이크 보너스 (60일 내 2회 이상 폭발 = 지속적 관심)
            multi_spike = len(spikes) >= 2
            if multi_spike:
                analysis["total_score"] = min(analysis["total_score"] + 5, 100)

            name = name_map.get(ticker, ticker)
            last = df.iloc[-1]

            results.append({
                "ticker": ticker,
                "name": name,
                "phase": analysis["phase"],
                "total_score": analysis["total_score"],
                "days_since_spike": analysis["days_since_spike"],
                "spike_date": analysis["spike_date"],
                "spike_close": analysis["spike_close"],
                "current_close": analysis["current_close"],
                "return_since_spike": analysis["return_since_spike"],
                "spike_vol_z": analysis["spike_vol_z"],
                "obv_score": analysis["obv_score"],
                "flow_score": analysis["flow_score"],
                "momentum_score": analysis["momentum_score"],
                "structure_score": analysis["structure_score"],
                "n_spikes_60d": len(spikes),
                "rsi": round(float(last.get("rsi_14", 50) or 50), 1),
                "trix": round(float(last.get("trix", 0) or 0), 3),
                "foreign_consec": int(last.get("foreign_consecutive_buy", 0) or 0),
            })

        except Exception as e:
            logger.debug("분석 실패 %s: %s", ticker, e)

    # 점수 내림차순
    results.sort(key=lambda x: -x["total_score"])
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    name_map = build_name_map()

    print("[매집 추적기] 전체 parquet 스캔 시작...")
    results = scan_all_stocks(name_map)

    # 페이즈별 통계
    phase_stats = {}
    for r in results:
        p = r["phase"]
        phase_stats[p] = phase_stats.get(p, 0) + 1

    print(f"\n{'='*60}")
    print(f"[매집 추적기] 탐지: {len(results)}종목")
    for phase in ["폭발", "매집", "재돌파", "가속"]:
        cnt = phase_stats.get(phase, 0)
        if cnt:
            print(f"  {phase}: {cnt}종목")
    print(f"{'='*60}")

    # 상위 20개 출력
    top = results[:20]
    if top:
        print(f"\n{'─'*65}")
        print(f"  ★ 세력 매집 추적 — 상위 20 ★")
        print(f"{'─'*65}")
        for i, r in enumerate(top, 1):
            phase_icon = {
                "폭발": "💥",
                "매집": "🔄",
                "재돌파": "🚀",
                "가속": "⚡",
            }.get(r["phase"], "?")

            print(
                f"  {i:2d}. {phase_icon} [{r['phase']}] {r['name']}({r['ticker']}) "
                f"{r['total_score']}점 | "
                f"{r['days_since_spike']}일전폭발 "
                f"수익:{r['return_since_spike']:+.1f}% "
                f"RSI:{r['rsi']:.0f}"
            )
            print(
                f"      OBV:{r['obv_score']} 수급:{r['flow_score']} "
                f"모멘텀:{r['momentum_score']} 구조:{r['structure_score']} "
                f"| TRIX:{r['trix']:.3f} 외인연속:{r['foreign_consec']}일 "
                f"폭발{r['n_spikes_60d']}회"
            )
    else:
        print("  현재 매집 진행 중 종목 없음")

    # JSON 저장
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_detected": len(results),
        "phase_stats": phase_stats,
        "top20": [r["ticker"] for r in top],
        "items": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[저장] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
