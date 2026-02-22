"""
세력감지 스캐너 — 전종목 대상 이상 패턴 탐지

전체 processed parquet (662+종목) + CSV 수급 데이터를 스캔하여
5가지 세력/이상 패턴을 탐지합니다.

패턴:
  P1. 거래량 폭발    — volume_surge_ratio > 3x (20일 평균 대비)
  P2. 수급 반전      — 5일 매도 → 오늘 대량 매수 (외인/기관)
  P3. BB Squeeze     — bb_width 수축 후 급확장 + 거래량 동반
  P4. OBV 다이버전스  — 가격 횡보/하락 + OBV 상승 (은밀 매집)
  P5. 수급 이탈 경고  — 외인+기관 동시 대량 매도 (탈출 시그널)

핵심필터:
  세력 포착  — 2개 이상 패턴 동시 발생 (P5 제외)
  이상 감지  — 1개 강한 패턴
  매집 의심  — OBV 다이버전스 + 거래량 수축
  이탈 경고  — P5 (동시 대량 매도)

Usage:
    python scripts/scan_whale_detect.py
    python scripts/scan_whale_detect.py --top 30
"""

from __future__ import annotations

import argparse
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

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
OUTPUT_PATH = PROJECT_ROOT / "data" / "whale_detect.json"


# ──────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────

def _sf(val, default=0):
    """NaN/Inf 안전 변환"""
    v = float(val)
    return default if (np.isnan(v) or np.isinf(v)) else round(v, 2)


def build_name_map() -> dict[str, str]:
    """CSV 파일명에서 종목코드 → 종목명 매핑"""
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name, ticker = parts
            name_map[ticker] = name
    return name_map


def get_flow_from_csv(ticker: str) -> dict:
    """CSV에서 최근 수급 상세 데이터 추출"""
    csvs = list(CSV_DIR.glob(f"*_{ticker}.csv"))
    if not csvs:
        return {}

    try:
        df = pd.read_csv(csvs[0], parse_dates=["Date"])
        df = df.dropna(subset=["Close"]).sort_values("Date")
        if len(df) < 10:
            return {}

        tail = df.tail(10)
        f_net = tail["Foreign_Net"].values if "Foreign_Net" in tail.columns else np.zeros(10)
        i_net = tail["Inst_Net"].values if "Inst_Net" in tail.columns else np.zeros(10)

        # 5일 전 (index 0~4)과 최근 (index 5~9)
        f_prev5 = float(np.nansum(f_net[:5]))
        f_last5 = float(np.nansum(f_net[5:]))
        i_prev5 = float(np.nansum(i_net[:5]))
        i_last5 = float(np.nansum(i_net[5:]))

        f_today = float(f_net[-1]) if len(f_net) > 0 else 0
        i_today = float(i_net[-1]) if len(i_net) > 0 else 0

        return {
            "f_prev5": f_prev5,
            "f_last5": f_last5,
            "i_prev5": i_prev5,
            "i_last5": i_last5,
            "f_today": f_today,
            "i_today": i_today,
        }
    except Exception:
        return {}


# ──────────────────────────────────────────
# 패턴 탐지 함수
# ──────────────────────────────────────────

def detect_volume_explosion(df: pd.DataFrame) -> dict | None:
    """P1: 거래량 폭발 (volume_surge_ratio > 2.5x)"""
    if "volume_surge_ratio" not in df.columns:
        return None

    last = df.iloc[-1]
    vsr = float(last["volume_surge_ratio"])
    if np.isnan(vsr) or vsr < 2.5:
        return None

    # 가격 방향 (상승/하락 폭발 구분)
    pct = float(last.get("price_change", 0)) if "price_change" in df.columns else 0
    direction = "상승폭발" if pct > 1.0 else ("하락폭발" if pct < -1.0 else "횡보폭발")

    return {
        "pattern": "P1_거래량폭발",
        "strength": min(vsr / 3.0, 3.0),  # 1.0 ~ 3.0
        "vsr": round(vsr, 1),
        "direction": direction,
        "price_change": _sf(pct),
        "desc": f"거래량 {vsr:.1f}배 폭발 ({direction})",
    }


def detect_flow_reversal(df: pd.DataFrame, csv_flow: dict) -> dict | None:
    """P2: 수급 반전 — 5일 매도 → 최근 대량 매수 전환"""
    if not csv_flow:
        # parquet에서 대체
        if "외국인합계" not in df.columns:
            return None

        tail = df.tail(10)
        if len(tail) < 10:
            return None

        f_vals = tail["외국인합계"].values
        i_vals = tail["기관합계"].values if "기관합계" in tail.columns else np.zeros(10)

        f_prev5 = float(np.nansum(f_vals[:5]))
        f_last5 = float(np.nansum(f_vals[5:]))
        i_prev5 = float(np.nansum(i_vals[:5]))
        i_last5 = float(np.nansum(i_vals[5:]))
        f_today = float(f_vals[-1]) if not np.isnan(f_vals[-1]) else 0
        i_today = float(i_vals[-1]) if not np.isnan(i_vals[-1]) else 0
    else:
        f_prev5 = csv_flow["f_prev5"]
        f_last5 = csv_flow["f_last5"]
        i_prev5 = csv_flow["i_prev5"]
        i_last5 = csv_flow["i_last5"]
        f_today = csv_flow["f_today"]
        i_today = csv_flow["i_today"]

    reasons = []
    strength = 0

    # 외인 반전: 이전 5일 순매도 → 최근 5일 순매수
    if f_prev5 < 0 and f_last5 > 0 and abs(f_last5) > abs(f_prev5) * 0.5:
        reasons.append(f"외인 매도→매수 반전 ({f_prev5:+,.0f}→{f_last5:+,.0f})")
        strength += 1.5

    # 기관 반전
    if i_prev5 < 0 and i_last5 > 0 and abs(i_last5) > abs(i_prev5) * 0.5:
        reasons.append(f"기관 매도→매수 반전 ({i_prev5:+,.0f}→{i_last5:+,.0f})")
        strength += 1.5

    # 오늘 대량 매수 (외인+기관 동시)
    if f_today > 0 and i_today > 0:
        reasons.append(f"금일 외인({f_today:+,.0f})+기관({i_today:+,.0f}) 동시매수")
        strength += 0.5

    if strength < 1.0:
        return None

    return {
        "pattern": "P2_수급반전",
        "strength": min(strength, 3.0),
        "f_prev5": _sf(f_prev5),
        "f_last5": _sf(f_last5),
        "i_prev5": _sf(i_prev5),
        "i_last5": _sf(i_last5),
        "f_today": _sf(f_today),
        "i_today": _sf(i_today),
        "reasons": reasons,
        "desc": " / ".join(reasons),
    }


def detect_bb_squeeze(df: pd.DataFrame) -> dict | None:
    """P3: BB Squeeze → 폭발 — bb_width 수축 후 급확장"""
    if "bb_width" not in df.columns or len(df) < 30:
        return None

    bw = df["bb_width"].dropna()
    if len(bw) < 30:
        return None

    # 최근 20일 bb_width의 최소 → 현재 대비
    recent_20 = bw.iloc[-20:]
    min_bw = float(recent_20.min())
    curr_bw = float(bw.iloc[-1])
    prev_bw = float(bw.iloc[-2]) if len(bw) >= 2 else curr_bw
    avg_bw = float(recent_20.mean())

    if np.isnan(min_bw) or np.isnan(curr_bw):
        return None

    # 조건: 이전에 수축(bb_width < 평균의 70%) → 현재 확장(bb_width > 최소*1.5)
    was_squeezed = min_bw < avg_bw * 0.7
    is_expanding = curr_bw > min_bw * 1.5 and curr_bw > prev_bw * 1.1

    if not (was_squeezed and is_expanding):
        return None

    # 거래량 동반 여부
    vsr = float(df.iloc[-1].get("volume_surge_ratio", 1.0))
    vol_confirm = vsr > 1.5

    expansion = curr_bw / min_bw if min_bw > 0 else 1.0
    strength = min(expansion / 2.0, 2.5)
    if vol_confirm:
        strength += 0.5

    return {
        "pattern": "P3_BB스퀴즈",
        "strength": min(strength, 3.0),
        "min_bb_width": _sf(min_bw, 0),
        "curr_bb_width": _sf(curr_bw, 0),
        "expansion": _sf(expansion),
        "vol_confirm": vol_confirm,
        "desc": f"BB 수축→폭발 ({expansion:.1f}배)" + (" +거래량" if vol_confirm else ""),
    }


def detect_obv_divergence(df: pd.DataFrame) -> dict | None:
    """P4: OBV 다이버전스 — 가격 횡보/하락 + OBV 상승 (은밀 매집)"""
    if "obv" not in df.columns or len(df) < 20:
        return None

    tail = df.tail(20)
    close = tail["close"].values
    obv = tail["obv"].values

    if np.any(np.isnan(close[-5:])) or np.any(np.isnan(obv[-5:])):
        return None

    # 가격 변화 (최근 10일)
    price_chg_10 = (close[-1] / close[-10] - 1) * 100 if close[-10] > 0 else 0

    # OBV 변화 (최근 10일) — 정규화
    obv_start = obv[-10]
    obv_end = obv[-1]
    if obv_start == 0:
        return None
    obv_chg_pct = (obv_end / obv_start - 1) * 100

    # 다이버전스: 가격 약보합(-5% ~ +3%) + OBV 상승(>3%)
    price_weak = -5 < price_chg_10 < 3
    obv_rising = obv_chg_pct > 3

    if not (price_weak and obv_rising):
        return None

    # OBV 추세 강도
    obv_trend = float(df.iloc[-1].get("obv_trend_5d", 0))
    strength = min(obv_chg_pct / 10.0, 2.5)
    if obv_trend > 0:
        strength += 0.3

    # 거래량 수축 동반 = 매집 의심 강화
    vsr = float(df.iloc[-1].get("volume_surge_ratio", 1.0))
    vol_contracted = vsr < 0.7
    if vol_contracted:
        strength += 0.3

    return {
        "pattern": "P4_OBV다이버전스",
        "strength": min(strength, 3.0),
        "price_chg_10d": _sf(price_chg_10),
        "obv_chg_10d": _sf(obv_chg_pct),
        "obv_trend_5d": _sf(obv_trend),
        "vol_contracted": vol_contracted,
        "desc": f"가격 {price_chg_10:+.1f}% vs OBV +{obv_chg_pct:.1f}% (은밀매집"
                + (" +저거래량)" if vol_contracted else ")"),
    }


def detect_flow_exit(df: pd.DataFrame, csv_flow: dict) -> dict | None:
    """P5: 수급 이탈 경고 — 외인+기관 동시 대량 매도"""
    if not csv_flow:
        if "외국인합계" not in df.columns:
            return None
        tail = df.tail(5)
        if len(tail) < 5:
            return None
        f_5d = float(np.nansum(tail["외국인합계"].values))
        i_5d = float(np.nansum(tail["기관합계"].values)) if "기관합계" in tail.columns else 0
        f_today = float(tail["외국인합계"].iloc[-1]) if not np.isnan(tail["외국인합계"].iloc[-1]) else 0
        i_today = float(tail["기관합계"].iloc[-1]) if "기관합계" in tail.columns and not np.isnan(tail["기관합계"].iloc[-1]) else 0
    else:
        f_5d = csv_flow["f_last5"]
        i_5d = csv_flow["i_last5"]
        f_today = csv_flow["f_today"]
        i_today = csv_flow["i_today"]

    # 동시 매도 조건: 외인+기관 모두 5일 순매도 + 오늘도 매도
    both_selling_5d = f_5d < 0 and i_5d < 0
    both_selling_today = f_today < 0 and i_today < 0

    if not (both_selling_5d and both_selling_today):
        return None

    total = abs(f_5d) + abs(i_5d)
    # 거래대금 대비 규모 판단 (간접)
    strength = min(1.0 + (total / 1e9), 3.0)  # 10억 단위

    return {
        "pattern": "P5_수급이탈",
        "strength": min(strength, 3.0),
        "f_5d": _sf(f_5d),
        "i_5d": _sf(i_5d),
        "f_today": _sf(f_today),
        "i_today": _sf(i_today),
        "desc": f"외인({f_5d:+,.0f})+기관({i_5d:+,.0f}) 동시매도 5일 지속",
    }


# ──────────────────────────────────────────
# 핵심필터 분류
# ──────────────────────────────────────────

def classify_whale(patterns: list[dict]) -> str:
    """패턴 조합으로 핵심필터 등급 분류"""
    p_names = [p["pattern"] for p in patterns]
    has_exit = "P5_수급이탈" in p_names
    positive_patterns = [p for p in patterns if p["pattern"] != "P5_수급이탈"]

    # 이탈 경고 우선
    if has_exit and not positive_patterns:
        return "이탈경고"

    # OBV 다이버전스 + 거래량 수축 = 매집 의심
    obv_p = next((p for p in patterns if p["pattern"] == "P4_OBV다이버전스"), None)
    if obv_p and obv_p.get("vol_contracted"):
        return "매집의심"

    # 2개 이상 긍정 패턴 = 세력 포착
    if len(positive_patterns) >= 2:
        return "세력포착"

    # 1개 패턴
    if len(positive_patterns) == 1:
        return "이상감지"

    # 이탈 + 다른 패턴 동시
    if has_exit and positive_patterns:
        return "혼합시그널"

    return "이상감지"


# ──────────────────────────────────────────
# 메인 스캔
# ──────────────────────────────────────────

def scan_all() -> list[dict]:
    """전체 processed parquet 스캔"""
    name_map = build_name_map()
    parquets = sorted(PROCESSED_DIR.glob("*.parquet"))
    print(f"[스캔] {len(parquets)}개 종목 대상 세력감지 시작...")

    results = []
    errors = 0

    for pq in parquets:
        ticker = pq.stem
        try:
            df = pd.read_parquet(pq)
            if len(df) < 30:
                continue

            # 최근 데이터만 사용 (성능)
            df = df.tail(60)
            last = df.iloc[-1]

            # 기본 필터: 종가 1000원 이상, 거래량 1000주 이상
            close = float(last.get("close", 0))
            vol = float(last.get("volume", 0))
            if close < 1000 or vol < 1000:
                continue

            # 거래대금 추정 (close * volume), 5억 미만 스킵
            trading_val = close * vol
            if trading_val < 5e8:
                continue

            # CSV 수급 조회
            csv_flow = get_flow_from_csv(ticker)

            # 5개 패턴 탐지
            detected = []
            p1 = detect_volume_explosion(df)
            if p1:
                detected.append(p1)
            p2 = detect_flow_reversal(df, csv_flow)
            if p2:
                detected.append(p2)
            p3 = detect_bb_squeeze(df)
            if p3:
                detected.append(p3)
            p4 = detect_obv_divergence(df)
            if p4:
                detected.append(p4)
            p5 = detect_flow_exit(df, csv_flow)
            if p5:
                detected.append(p5)

            if not detected:
                continue

            # 핵심필터 분류
            grade = classify_whale(detected)

            # 종합 강도 (패턴 strength 합산)
            total_strength = sum(p["strength"] for p in detected)

            # 기본 지표 수집
            name = name_map.get(ticker, ticker)
            rsi = _sf(last.get("rsi_14", 50))
            adx = _sf(last.get("adx_14", 0))
            bb_pos = _sf(last.get("bb_position", 50))
            vsr = _sf(last.get("volume_surge_ratio", 1))
            ma20 = _sf(last.get("sma_20", 0))
            ma60 = _sf(last.get("sma_60", 0))
            above_ma20 = close > ma20 if ma20 > 0 else False
            above_ma60 = close > ma60 if ma60 > 0 else False
            pct_chg = _sf(last.get("price_change", 0))
            foreign_5d = _sf(last.get("foreign_net_5d", 0))

            rec = {
                "ticker": ticker,
                "name": name,
                "close": int(close),
                "price_change": pct_chg,
                "volume": int(vol),
                "trading_value_m": _sf(trading_val / 1e6),  # 백만원 (추정)
                "rsi": rsi,
                "adx": adx,
                "bb_position": bb_pos,
                "volume_surge_ratio": vsr,
                "above_ma20": above_ma20,
                "above_ma60": above_ma60,
                "foreign_5d": foreign_5d,
                "grade": grade,
                "strength": _sf(total_strength),
                "pattern_count": len(detected),
                "patterns": detected,
            }
            results.append(rec)

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning("종목 %s 처리 실패: %s", ticker, e)

    # 정렬: 세력포착 > 매집의심 > 이상감지 > 혼합시그널 > 이탈경고, 같은 등급 내 강도순
    grade_order = {"세력포착": 0, "매집의심": 1, "이상감지": 2, "혼합시그널": 3, "이탈경고": 4}
    results.sort(key=lambda x: (grade_order.get(x["grade"], 9), -x["strength"]))

    return results


def main():
    parser = argparse.ArgumentParser(description="세력감지 스캐너")
    parser.add_argument("--top", type=int, default=50, help="출력 종목 수")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = scan_all()

    # 통계
    grades = {}
    for r in results:
        g = r["grade"]
        grades[g] = grades.get(g, 0) + 1

    print(f"\n{'='*60}")
    print(f"[세력감지] 총 {len(results)}건 탐지")
    for g in ["세력포착", "매집의심", "이상감지", "혼합시그널", "이탈경고"]:
        cnt = grades.get(g, 0)
        if cnt:
            print(f"  {g}: {cnt}건")
    print(f"{'='*60}\n")

    # 상위 종목 출력
    for i, r in enumerate(results[:args.top], 1):
        pats = " + ".join(p["pattern"].split("_")[1] for p in r["patterns"])
        print(f"  {i:2d}. [{r['grade']}] {r['name']}({r['ticker']}) "
              f"종가 {r['close']:,} ({r['price_change']:+.1f}%) "
              f"강도 {r['strength']:.1f} — {pats}")
        for p in r["patterns"]:
            print(f"      ↳ {p['desc']}")

    # JSON 저장
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_scanned": len(list(PROCESSED_DIR.glob("*.parquet"))),
        "total_detected": len(results),
        "stats": grades,
        "items": results[:args.top],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
