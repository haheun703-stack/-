"""
파생 시그널 수집기 — 선물 베이시스 + 풋/콜 프록시 + 프로그램매매 추정

KRX 직접 API가 세션 제한이 있어, ETF 가격/거래량에서 역산하는 방식:
  1. 선물 베이시스 ≈ KODEX200선물ETF / KOSPI200 ETF - 1 (콘탱고/백워데이션)
  2. 풋/콜 프록시 ≈ 인버스ETF 거래대금 / (인버스 + 레버리지) 거래대금
  3. 레버리지/인버스 거래량 비율 → 시장 심리 (외국인 선물 포지션 대리변수)
  4. 프로그램매매 추정 ≈ KOSPI200 ETF + 레버리지 대량 거래 시 차익거래 추정

출력: data/derivatives/derivatives_signal.json
사용: python -u -X utf8 scripts/derivatives_collector.py
BAT: schedule_D_after_close.bat (장마감 후 실행)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "derivatives"
SIGNAL_PATH = DATA_DIR / "derivatives_signal.json"
HISTORY_PATH = DATA_DIR / "derivatives_history.csv"

# ETF 심볼 (yfinance)
ETF_SYMBOLS = {
    "KODEX200":       "069500.KS",  # KOSPI200 현물 추종
    "KODEX_LEV":      "122630.KS",  # KODEX 레버리지 (2X 롱)
    "KODEX_INV2X":    "252670.KS",  # KODEX 인버스2X (2X 숏)
    "KODEX_INV":      "114800.KS",  # KODEX 인버스 (1X 숏)
    "KODEX200_FUT":   "261240.KS",  # KODEX 200선물
    "KODEX_KQ_LEV":   "233740.KS",  # KODEX 코스닥150레버리지
    "KODEX_KQ_INV":   "251340.KS",  # KODEX 코스닥150선물인버스
}

# KOSPI200 지수
KOSPI200_SYMBOL = "^KS200"
LOOKBACK_DAYS = 60  # 60일 이력


def _fetch_etf_data(period: str = "3mo") -> dict[str, pd.DataFrame]:
    """ETF 가격/거래량 수집."""
    results = {}
    all_symbols = {**ETF_SYMBOLS, "KOSPI200_IDX": KOSPI200_SYMBOL}

    for name, sym in all_symbols.items():
        try:
            t = yf.Ticker(sym)
            h = t.history(period=period)
            if not h.empty:
                results[name] = h
                logger.debug(f"  {name}: {len(h)}일")
        except Exception as e:
            logger.warning(f"  {name}({sym}) 수집 실패: {e}")

    return results


def _compute_futures_basis(data: dict) -> dict:
    """선물 베이시스 계산.

    베이시스 = 선물ETF가격 / 현물ETF가격 - 1
    양수(콘탱고): 선물 프리미엄 → 상승 기대
    음수(백워데이션): 선물 디스카운트 → 하락 기대
    """
    result = {"available": False}

    fut = data.get("KODEX200_FUT")
    spot = data.get("KODEX200")
    if fut is None or spot is None:
        return result

    # 날짜 정렬 (공통 날짜만)
    common = fut.index.intersection(spot.index)
    if len(common) < 5:
        return result

    fut_c = fut.loc[common, "Close"]
    spot_c = spot.loc[common, "Close"]

    # 선물/현물 비율 (절대값이 아닌 비율 추세)
    # KODEX200선물은 15000원대, KODEX200은 86000원대 → 직접 비교 불가
    # 대신 수익률 비율로 베이시스 추정
    fut_ret = fut_c.pct_change()
    spot_ret = spot_c.pct_change()
    basis_spread = fut_ret - spot_ret  # 선물이 현물보다 더 많이 움직이면 양수

    latest_basis = float(basis_spread.iloc[-1]) if len(basis_spread) > 0 else 0
    basis_5d = float(basis_spread.tail(5).mean()) if len(basis_spread) >= 5 else 0
    basis_20d = float(basis_spread.tail(20).mean()) if len(basis_spread) >= 20 else 0

    # 콘탱고/백워데이션 판정
    if basis_5d > 0.002:
        status = "CONTANGO"  # 상승 기대
    elif basis_5d < -0.002:
        status = "BACKWARDATION"  # 하락 기대
    else:
        status = "FLAT"

    result = {
        "available": True,
        "basis_1d": round(latest_basis * 100, 4),  # %
        "basis_5d_avg": round(basis_5d * 100, 4),
        "basis_20d_avg": round(basis_20d * 100, 4),
        "status": status,
    }
    return result


def _compute_put_call_proxy(data: dict) -> dict:
    """풋/콜 프록시 계산.

    인버스2X 거래대금 / (인버스2X + 레버리지) 거래대금
    > 0.5 → 약세 심리 (풋 과다)
    < 0.3 → 강세 심리 (콜 과다)
    0.3~0.5 → 중립
    """
    result = {"available": False}

    inv = data.get("KODEX_INV2X")
    lev = data.get("KODEX_LEV")
    if inv is None or lev is None:
        return result

    common = inv.index.intersection(lev.index)
    if len(common) < 5:
        return result

    # 거래대금 = 종가 × 거래량
    inv_val = inv.loc[common, "Close"] * inv.loc[common, "Volume"]
    lev_val = lev.loc[common, "Close"] * lev.loc[common, "Volume"]
    total = inv_val + lev_val

    # 0 나누기 방지
    pc_ratio = pd.Series(0.5, index=common)
    mask = total > 0
    pc_ratio[mask] = inv_val[mask] / total[mask]

    latest = float(pc_ratio.iloc[-1])
    avg_5d = float(pc_ratio.tail(5).mean())
    avg_20d = float(pc_ratio.tail(20).mean())

    # 5일 대비 변화 (심리 전환 감지)
    change_5d = latest - avg_5d

    # 상태 판정
    if latest > 0.55:
        status = "EXTREME_BEARISH"
    elif latest > 0.45:
        status = "BEARISH"
    elif latest > 0.35:
        status = "NEUTRAL"
    elif latest > 0.25:
        status = "BULLISH"
    else:
        status = "EXTREME_BULLISH"

    # 반전 시그널: 극단치에서 반대 방향 움직임
    reversal = ""
    if latest > 0.55 and change_5d < -0.05:
        reversal = "BEARISH_EXHAUSTION"  # 공포 정점 후 반등 가능
    elif latest < 0.25 and change_5d > 0.05:
        reversal = "BULLISH_EXHAUSTION"  # 탐욕 정점 후 조정 가능

    # 코스닥도 있으면 추가
    kq_detail = {}
    kq_inv = data.get("KODEX_KQ_INV")
    kq_lev = data.get("KODEX_KQ_LEV")
    if kq_inv is not None and kq_lev is not None:
        kq_common = kq_inv.index.intersection(kq_lev.index)
        if len(kq_common) >= 3:
            kq_inv_val = kq_inv.loc[kq_common, "Close"] * kq_inv.loc[kq_common, "Volume"]
            kq_lev_val = kq_lev.loc[kq_common, "Close"] * kq_lev.loc[kq_common, "Volume"]
            kq_total = kq_inv_val + kq_lev_val
            kq_mask = kq_total > 0
            kq_pc = pd.Series(0.5, index=kq_common)
            kq_pc[kq_mask] = kq_inv_val[kq_mask] / kq_total[kq_mask]
            kq_detail = {
                "kosdaq_pc_latest": round(float(kq_pc.iloc[-1]), 4),
                "kosdaq_pc_5d": round(float(kq_pc.tail(5).mean()), 4),
            }

    result = {
        "available": True,
        "pc_ratio": round(latest, 4),
        "pc_5d_avg": round(avg_5d, 4),
        "pc_20d_avg": round(avg_20d, 4),
        "pc_change_5d": round(change_5d, 4),
        "status": status,
        "reversal": reversal,
        **kq_detail,
    }
    return result


def _compute_leverage_flow(data: dict) -> dict:
    """레버리지/인버스 자금 흐름 분석.

    외국인과 기관은 선물 대신 레버리지 ETF를 사용하기도 함.
    레버리지 거래량 급증 = 프로그램 매수 동반 가능성
    인버스 거래량 급증 = 프로그램 매도 동반 가능성
    """
    result = {"available": False}

    lev = data.get("KODEX_LEV")
    inv = data.get("KODEX_INV2X")
    spot = data.get("KODEX200")
    if lev is None or inv is None or spot is None:
        return result

    common = lev.index.intersection(inv.index).intersection(spot.index)
    if len(common) < 10:
        return result

    # 거래량 Z-score (20일 기준)
    lev_vol = lev.loc[common, "Volume"]
    inv_vol = inv.loc[common, "Volume"]
    spot_vol = spot.loc[common, "Volume"]

    def _zscore(s):
        mean = s.rolling(20).mean()
        std = s.rolling(20).std()
        return (s - mean) / std.replace(0, 1)

    lev_z = _zscore(lev_vol)
    inv_z = _zscore(inv_vol)
    spot_z = _zscore(spot_vol)

    latest_lev_z = float(lev_z.iloc[-1]) if not np.isnan(lev_z.iloc[-1]) else 0
    latest_inv_z = float(inv_z.iloc[-1]) if not np.isnan(inv_z.iloc[-1]) else 0
    latest_spot_z = float(spot_z.iloc[-1]) if not np.isnan(spot_z.iloc[-1]) else 0

    # 프로그램매매 추정: 레버리지 + 현물 동시 급증 → 차익거래
    program_buy_signal = latest_lev_z > 1.5 and latest_spot_z > 1.0
    program_sell_signal = latest_inv_z > 1.5 and latest_spot_z > 1.0

    # 순유입 (레버리지 - 인버스 거래대금)
    lev_val = lev.loc[common, "Close"] * lev_vol
    inv_val = inv.loc[common, "Close"] * inv_vol
    net_flow = lev_val - inv_val  # 양수 = 롱 우세

    net_flow_1d = float(net_flow.iloc[-1])
    net_flow_5d = float(net_flow.tail(5).sum())

    result = {
        "available": True,
        "leverage_vol_z": round(latest_lev_z, 2),
        "inverse_vol_z": round(latest_inv_z, 2),
        "spot_vol_z": round(latest_spot_z, 2),
        "program_buy_est": program_buy_signal,
        "program_sell_est": program_sell_signal,
        "net_flow_1d_억": round(net_flow_1d / 1e8, 0),
        "net_flow_5d_억": round(net_flow_5d / 1e8, 0),
    }
    return result


def _compute_composite_score(basis: dict, pc: dict, flow: dict) -> dict:
    """3개 축 종합 → 파생 시그널 스코어 (-100 ~ +100)."""
    score = 0.0
    breakdown = {}

    # 1. 선물 베이시스 (30%)
    if basis.get("available"):
        b5 = basis.get("basis_5d_avg", 0)
        # +0.1% 이상 = 상승 기대, -0.1% = 하락 기대
        basis_score = max(-30, min(30, b5 * 300))
        score += basis_score
        breakdown["basis"] = round(basis_score, 1)

    # 2. 풋/콜 프록시 (40%) — 가장 중요
    if pc.get("available"):
        pc_val = pc.get("pc_ratio", 0.5)
        # 0.5 = 중립(0점), 0.7 = 극도 약세(-40), 0.2 = 극도 강세(+40)
        pc_score = (0.5 - pc_val) * 130  # 0.2→+39, 0.5→0, 0.7→-26
        pc_score = max(-40, min(40, pc_score))

        # 반전 시그널 보정
        if pc.get("reversal") == "BEARISH_EXHAUSTION":
            pc_score += 10  # 공포 정점 → 반등 신호
        elif pc.get("reversal") == "BULLISH_EXHAUSTION":
            pc_score -= 10  # 탐욕 정점 → 조정 신호

        score += pc_score
        breakdown["put_call"] = round(pc_score, 1)

    # 3. 레버리지 흐름 (30%)
    if flow.get("available"):
        net_5d = flow.get("net_flow_5d_억", 0)
        # 순유입 양수 = 롱 우세
        flow_score = max(-30, min(30, net_5d / 500))

        # 프로그램매매 추정
        if flow.get("program_buy_est"):
            flow_score += 5
        if flow.get("program_sell_est"):
            flow_score -= 5

        score += flow_score
        breakdown["flow"] = round(flow_score, 1)

    score = max(-100, min(100, score))

    # 등급
    if score >= 40:
        grade = "STRONG_BULL"
    elif score >= 15:
        grade = "MILD_BULL"
    elif score > -15:
        grade = "NEUTRAL"
    elif score > -40:
        grade = "MILD_BEAR"
    else:
        grade = "STRONG_BEAR"

    return {
        "score": round(score, 1),
        "grade": grade,
        "breakdown": breakdown,
    }


def _save_history(signal: dict):
    """CSV 히스토리에 추가 (검증용)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    row = {
        "date": signal.get("date", ""),
        "score": signal["composite"]["score"],
        "grade": signal["composite"]["grade"],
        "pc_ratio": signal["put_call_proxy"].get("pc_ratio", ""),
        "pc_status": signal["put_call_proxy"].get("status", ""),
        "basis_5d": signal["futures_basis"].get("basis_5d_avg", ""),
        "basis_status": signal["futures_basis"].get("status", ""),
        "net_flow_5d": signal["leverage_flow"].get("net_flow_5d_억", ""),
        "lev_z": signal["leverage_flow"].get("leverage_vol_z", ""),
        "inv_z": signal["leverage_flow"].get("inverse_vol_z", ""),
    }

    if HISTORY_PATH.exists():
        df = pd.read_csv(HISTORY_PATH)
        # 중복 방지
        if row["date"] not in df["date"].values:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(HISTORY_PATH, index=False)


def collect_derivatives_signal() -> dict:
    """파생 시그널 수집 + 종합 신호 생성."""
    logger.info("=== 파생 시그널 수집 시작 ===")

    # 1. ETF 데이터 수집
    data = _fetch_etf_data(period="3mo")
    logger.info(f"ETF {len(data)}종 수집 완료")

    # 2. 3축 계산
    basis = _compute_futures_basis(data)
    pc = _compute_put_call_proxy(data)
    flow = _compute_leverage_flow(data)

    # 3. 종합 스코어
    composite = _compute_composite_score(basis, pc, flow)

    # 4. 신호 구성
    signal = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "futures_basis": basis,
        "put_call_proxy": pc,
        "leverage_flow": flow,
        "composite": composite,
    }

    # 5. 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_PATH.write_text(
        json.dumps(signal, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"저장: {SIGNAL_PATH}")

    # 히스토리 추가
    _save_history(signal)

    # 6. 로그 출력
    c = composite
    logger.info(
        f"파생 시그널: {c['grade']} ({c['score']:+.1f}) "
        f"| 베이시스: {basis.get('status', '?')} "
        f"| P/C: {pc.get('status', '?')} ({pc.get('pc_ratio', 0):.3f}) "
        f"| 순유입: {flow.get('net_flow_5d_억', 0):+,.0f}억"
    )

    if pc.get("reversal"):
        logger.warning(f"⚠️ 반전 시그널: {pc['reversal']}")

    return signal


def main():
    signal = collect_derivatives_signal()

    c = signal["composite"]
    print(f"\n=== 파생 시그널 ===")
    print(f"등급: {c['grade']} (스코어: {c['score']:+.1f})")
    print(f"내역: {c['breakdown']}")

    pc = signal["put_call_proxy"]
    if pc.get("available"):
        print(f"\n풋/콜 프록시: {pc['pc_ratio']:.3f} ({pc['status']})")
        print(f"  5일평균: {pc['pc_5d_avg']:.3f}, 20일평균: {pc['pc_20d_avg']:.3f}")
        if pc.get("reversal"):
            print(f"  ⚠️ 반전: {pc['reversal']}")

    basis = signal["futures_basis"]
    if basis.get("available"):
        print(f"\n선물 베이시스: {basis['basis_5d_avg']:+.4f}% ({basis['status']})")

    flow = signal["leverage_flow"]
    if flow.get("available"):
        print(f"\n레버리지 흐름:")
        print(f"  순유입 5D: {flow['net_flow_5d_억']:+,.0f}억")
        print(f"  레버리지 Z: {flow['leverage_vol_z']:+.2f}, 인버스 Z: {flow['inverse_vol_z']:+.2f}")


if __name__ == "__main__":
    main()
