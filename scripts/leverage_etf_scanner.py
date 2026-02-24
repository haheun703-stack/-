"""레버리지/인버스 ETF 로테이션 스캐너 — 5축 100점 스코어링.

KOSPI 레짐 + US Overnight + 기술지표 + 수급 + 매크로 통합 분석.
기존 update_etf_master.py 지표 함수를 재사용한다.

사용법:
  python scripts/leverage_etf_scanner.py           # 기본 (수급 포함)
  python scripts/leverage_etf_scanner.py --no-flow  # 수급 생략 (오프라인)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 기존 ETF 마스터에서 지표 함수 재사용
from scripts.update_etf_master import (
    calc_rsi,
    calc_bb_pct,
    calc_adx,
    calc_stoch,
    calc_trix,
    calc_ma_gap,
    calc_returns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
LEVERAGE_DIR = DATA_DIR / "leverage_etf"
UNIVERSE_PATH = LEVERAGE_DIR / "leverage_universe.json"
OUT_PATH = LEVERAGE_DIR / "leverage_etf_scan.json"
REGIME_PATH = DATA_DIR / "kospi_regime.json"
US_SIGNAL_PATH = DATA_DIR / "overnight_signal.json"


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def load_universe() -> dict:
    with open(UNIVERSE_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_regime() -> dict:
    """kospi_regime.json → regime, close, slots 등."""
    if REGIME_PATH.exists():
        with open(REGIME_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"regime": "CAUTION", "close": 0, "slots": 3}


def load_us_signal() -> dict:
    """overnight_signal.json → grade, l1_score, vix 등."""
    if US_SIGNAL_PATH.exists():
        with open(US_SIGNAL_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"grade": "NEUTRAL"}


def fetch_etf_ohlcv(etf_code: str, days: int = 120) -> pd.DataFrame | None:
    """pykrx로 ETF OHLCV 수집."""
    try:
        from pykrx import stock as pykrx_stock

        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df = pykrx_stock.get_etf_ohlcv_by_date(start, end, etf_code)
        if df is None or df.empty:
            return None

        df.columns = [c.lower().replace("시가", "open").replace("고가", "high")
                      .replace("저가", "low").replace("종가", "close")
                      .replace("거래량", "volume").replace("거래대금", "value")
                      .replace("nav", "nav").replace("기초지수", "base_idx")
                      for c in df.columns]
        # 한글 컬럼이 남아있으면 매핑
        rename = {"시가": "open", "고가": "high", "저가": "low",
                  "종가": "close", "거래량": "volume", "거래대금": "value"}
        df.rename(columns=rename, inplace=True)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.warning("OHLCV 수집 실패 (%s): %s", etf_code, e)
        return None


def get_etf_flow(etf_code: str, df: pd.DataFrame, days: int = 5) -> tuple[float, float, float, float]:
    """pykrx ETF 수급 조회 (5일 누적 + 당일)."""
    try:
        from pykrx import stock as pykrx_stock

        trading_dates = [d.strftime("%Y%m%d") for d in df.index[-max(days + 2, 7):]]
        if len(trading_dates) < 2:
            return 0, 0, 0, 0

        recent = trading_dates[-days:]
        fromdate = recent[0]
        todate = recent[-1]
        col = ("거래대금", "순매수")

        df5 = pykrx_stock.get_etf_trading_volume_and_value(fromdate, todate, etf_code)
        f5, i5 = 0.0, 0.0
        if df5 is not None and not df5.empty and len(df5) >= 13:
            seg = df5.iloc[:13]
            if "외국인" in seg.index and col in seg.columns:
                f5 = float(seg.loc["외국인", col]) / 1e8
                i5 = float(seg.loc["기관합계", col]) / 1e8

        time.sleep(0.2)

        df1 = pykrx_stock.get_etf_trading_volume_and_value(todate, todate, etf_code)
        ft, it = 0.0, 0.0
        if df1 is not None and not df1.empty and len(df1) >= 13:
            seg = df1.iloc[:13]
            if "외국인" in seg.index and col in seg.columns:
                ft = float(seg.loc["외국인", col]) / 1e8
                it = float(seg.loc["기관합계", col]) / 1e8

        return round(f5, 1), round(i5, 1), round(ft, 1), round(it, 1)
    except Exception as e:
        logger.warning("수급 조회 실패 (%s): %s", etf_code, e)
        return 0, 0, 0, 0


# ─────────────────────────────────────────────
# 변동성 감가 리스크
# ─────────────────────────────────────────────

def calc_vol_decay_risk(close: pd.Series, period: int = 20) -> str:
    """20일 실현변동성 → 감가 리스크 등급."""
    if len(close) < period + 1:
        return "N/A"
    log_ret = np.log(close / close.shift(1)).dropna()
    rv = float(log_ret.tail(period).std() * np.sqrt(252) * 100)
    if rv < 20:
        return "Low"
    elif rv < 35:
        return "Mid"
    else:
        return "High"


# ─────────────────────────────────────────────
# 5축 100점 스코어링
# ─────────────────────────────────────────────

# 레짐적합성 매트릭스 (30점 만점)
REGIME_SCORE = {
    ("BULL", "LONG"): 30,
    ("BULL", "SHORT"): 0,
    ("CAUTION", "LONG"): 20,
    ("CAUTION", "SHORT"): 10,
    ("BEAR", "LONG"): 5,
    ("BEAR", "SHORT"): 25,
    ("CRISIS", "LONG"): 0,
    ("CRISIS", "SHORT"): 30,
}


def calc_leverage_score(
    e: dict,
    regime: str,
    us_signal: dict,
) -> tuple[float, str, list[str], dict]:
    """5축 100점 레버리지 ETF 추천 점수.

    축1: 레짐적합성(30) + 축2: 진입타이밍(25) + 축3: 추세강도(20)
    축4: 수급(15) + 축5: 매크로(10)
    """
    direction = e["direction"]
    reasons: list[str] = []
    breakdown = {}

    # ── 축1: 레짐적합성 (30점) ──
    axis1 = REGIME_SCORE.get((regime, direction), 10)
    breakdown["regime"] = axis1
    if axis1 >= 25:
        reasons.append(f"{regime} 레짐+{direction} 최적")
    elif axis1 <= 5:
        reasons.append(f"{regime} 레짐+{direction} 방향 불일치")

    # ── 축2: 진입타이밍 (25점) ──
    axis2 = 0
    rsi = e["rsi"]
    bb_pct = e["bb_pct"]

    # RSI (10점): 과매도 영역이 매수 적기, 과매수 영역이 매도 적기
    if direction == "LONG":
        if rsi < 30:
            axis2 += 10; reasons.append(f"RSI {rsi:.0f} 과매도 (절호)")
        elif rsi < 40:
            axis2 += 8
        elif 40 <= rsi <= 60:
            axis2 += 6; reasons.append("RSI 적정")
        elif rsi < 70:
            axis2 += 3
        else:
            axis2 += 0; reasons.append(f"RSI {rsi:.0f} 과열 주의")
    else:  # SHORT
        if rsi > 70:
            axis2 += 10; reasons.append(f"RSI {rsi:.0f} 과매수 (인버스 적기)")
        elif rsi > 60:
            axis2 += 7
        elif 40 <= rsi <= 60:
            axis2 += 4
        else:
            axis2 += 1

    # BB% (10점): LONG이면 낮을수록, SHORT이면 높을수록 좋음
    if direction == "LONG":
        if bb_pct < 20:
            axis2 += 10; reasons.append("BB% 하단 접근 (매수 기회)")
        elif bb_pct < 40:
            axis2 += 7
        elif bb_pct < 60:
            axis2 += 5
        elif bb_pct < 80:
            axis2 += 2
        else:
            axis2 += 0
    else:  # SHORT
        if bb_pct > 80:
            axis2 += 10; reasons.append("BB% 상단 과열 (인버스 적기)")
        elif bb_pct > 60:
            axis2 += 6
        elif bb_pct > 40:
            axis2 += 3
        else:
            axis2 += 0

    # Stoch 골든크로스 (5점)
    if e["stoch_gx"] and direction == "LONG":
        axis2 += 5; reasons.append("Stoch 골든크로스")
    elif not e["stoch_gx"] and direction == "SHORT":
        axis2 += 5
    elif e["stoch_k"] < 20 and direction == "LONG":
        axis2 += 3
    elif e["stoch_k"] > 80 and direction == "SHORT":
        axis2 += 3

    breakdown["entry_timing"] = axis2

    # ── 축3: 추세강도 (20점) ──
    axis3 = 0
    adx = e["adx"]

    # ADX (8점)
    if 25 <= adx <= 50:
        axis3 += 8; reasons.append(f"ADX {adx:.0f} 강한 추세")
    elif 20 <= adx < 25:
        axis3 += 5
    elif adx > 50:
        axis3 += 4  # 과열 추세
    else:
        axis3 += 2

    # TRIX 방향 (7점)
    if e["trix_bull"] and direction == "LONG":
        axis3 += 7; reasons.append("TRIX 상승전환")
    elif not e["trix_bull"] and direction == "SHORT":
        axis3 += 7; reasons.append("TRIX 하락전환 (인버스 유리)")
    elif e["trix_bull"] and direction == "SHORT":
        axis3 += 0
    else:
        axis3 += 3

    # MA20 배열 (5점)
    ma20_gap = e["ma20_gap"]
    if direction == "LONG" and ma20_gap > 0:
        axis3 += min(5, int(ma20_gap))
    elif direction == "SHORT" and ma20_gap < 0:
        axis3 += min(5, int(abs(ma20_gap)))
    elif direction == "LONG" and -3 < ma20_gap <= 0:
        axis3 += 2  # 눌림목
    else:
        axis3 += 0

    breakdown["trend"] = axis3

    # ── 축4: 수급 (15점) ──
    axis4 = 0
    f5 = e["foreign_5d"]
    i5 = e["inst_5d"]

    # 외인 5일 (6점)
    if direction == "LONG":
        if f5 > 0:
            axis4 += min(6, int(f5 * 0.5))
        # SHORT에서 외인 매도는 긍정
    else:
        if f5 < 0:
            axis4 += min(6, int(abs(f5) * 0.5))

    # 기관 5일 (6점)
    if direction == "LONG":
        if i5 > 0:
            axis4 += min(6, int(i5 * 0.5))
    else:
        if i5 < 0:
            axis4 += min(6, int(abs(i5) * 0.5))

    # 동시매수 (3점)
    is_smart = e["is_smart"]
    if is_smart and direction == "LONG":
        axis4 += 3; reasons.append("외인+기관 동시매수")
    elif f5 > 0 and direction == "LONG":
        reasons.append(f"외인 순매수 {f5:.0f}억")
    elif i5 > 0 and direction == "LONG":
        reasons.append(f"기관 순매수 {i5:.0f}억")

    breakdown["flow"] = axis4

    # ── 축5: 매크로 (10점) ──
    axis5 = 0
    us_grade = us_signal.get("grade", "NEUTRAL")

    # US Overnight 등급 (5점)
    us_map_long = {"STRONG_BULL": 5, "MILD_BULL": 3, "NEUTRAL": 2, "MILD_BEAR": 0, "STRONG_BEAR": 0}
    us_map_short = {"STRONG_BEAR": 5, "MILD_BEAR": 3, "NEUTRAL": 2, "MILD_BULL": 0, "STRONG_BULL": 0}
    if direction == "LONG":
        axis5 += us_map_long.get(us_grade, 2)
    else:
        axis5 += us_map_short.get(us_grade, 2)

    # VIX 수준 (3점)
    vix = us_signal.get("vix_close", us_signal.get("vix", 18))
    if isinstance(vix, (int, float)):
        if direction == "LONG":
            if vix < 18:
                axis5 += 3
            elif vix < 25:
                axis5 += 1
        else:  # SHORT
            if vix > 25:
                axis5 += 3
            elif vix > 20:
                axis5 += 1

    # EWY 방향 (2점)
    ewy_ret = us_signal.get("ewy_ret", us_signal.get("ewy_change", 0))
    if isinstance(ewy_ret, (int, float)):
        if direction == "LONG" and ewy_ret > 0:
            axis5 += 2
        elif direction == "SHORT" and ewy_ret < 0:
            axis5 += 2

    if us_grade in ("STRONG_BULL", "MILD_BULL") and direction == "LONG":
        reasons.append(f"US {us_grade}")
    elif us_grade in ("STRONG_BEAR", "MILD_BEAR") and direction == "SHORT":
        reasons.append(f"US {us_grade} (인버스 유리)")

    breakdown["macro"] = axis5

    # ── 합계 ──
    total = axis1 + axis2 + axis3 + axis4 + axis5

    # ── 등급 ──
    if axis1 <= 5:
        grade = "방향불일치"
    elif total >= 75:
        grade = "적극매수"
    elif total >= 60:
        grade = "분할매수"
    elif total >= 45:
        grade = "관심"
    elif total >= 30:
        grade = "대기"
    else:
        grade = "부적합"

    return round(total, 1), grade, reasons, breakdown


# ─────────────────────────────────────────────
# 분할매수 + 동적 상한선
# ─────────────────────────────────────────────

def calc_split_buy(close: int) -> dict:
    """4단계 분할매수 가격 계산."""
    return {
        "entry_1": close,
        "entry_2": int(close * 0.95),
        "entry_3": int(close * 0.90),
        "entry_4": int(close * 0.85),
    }


def calc_upside(regime: dict, direction: str, leverage: int) -> tuple[float, float]:
    """지수 ETF의 PER 기반 동적 상한선 수익률 추정."""
    kospi_close = regime.get("close", 0)
    if kospi_close <= 0 or direction == "SHORT":
        return 0.0, 0.0

    per_16x = 6080  # 보수적 (PER 16x)
    per_18x = 6840  # 낙관적 (PER 18x)

    upside_cons = round((per_16x / kospi_close - 1) * 100 * leverage, 1)
    upside_opt = round((per_18x / kospi_close - 1) * 100 * leverage, 1)

    return max(0, upside_cons), max(0, upside_opt)


# ─────────────────────────────────────────────
# 전체 방향 추천 생성
# ─────────────────────────────────────────────

def generate_recommendation(regime: str, us_grade: str) -> str:
    """KOSPI 레짐 + US 시그널 기반 전체 방향 추천."""
    if regime == "CRISIS":
        return "SHORT 전환 (위기 레짐, 인버스 위주)"
    elif regime == "BEAR":
        if us_grade in ("STRONG_BEAR", "MILD_BEAR"):
            return "SHORT 위주 (하락 레짐 + US 약세)"
        return "관망 (하락 레짐이나 US 혼조)"
    elif regime == "CAUTION":
        if us_grade in ("STRONG_BULL", "MILD_BULL"):
            return "LONG 위주 분할진입 (과열 주의)"
        elif us_grade in ("STRONG_BEAR", "MILD_BEAR"):
            return "관망 (국내 주의 + US 약세)"
        return "소규모 LONG (과열 주의, 분할 필수)"
    else:  # BULL
        if us_grade in ("STRONG_BULL", "MILD_BULL"):
            return "LONG 적극 (최적 환경)"
        return "LONG 위주 (레짐 강세)"


# ─────────────────────────────────────────────
# 메인 스캐너
# ─────────────────────────────────────────────

def scan_leverage_etfs(skip_flow: bool = False) -> dict:
    """레버리지/인버스 ETF 전체 스캔."""
    universe = load_universe()
    regime_data = load_regime()
    us_signal = load_us_signal()

    regime = regime_data.get("regime", "CAUTION")
    us_grade = us_signal.get("grade", "NEUTRAL")

    logger.info("=" * 55)
    logger.info("  레버리지/인버스 ETF 로테이션 스캐너")
    logger.info("  KOSPI 레짐: %s | US Overnight: %s", regime, us_grade)
    logger.info("=" * 55)

    etfs: list[dict] = []

    for etf_name, info in universe.items():
        etf_code = info["etf_code"]
        direction = info["direction"]
        leverage = info["leverage"]
        category = info["category"]

        # OHLCV 수집
        df = fetch_etf_ohlcv(etf_code)
        if df is None or len(df) < 30:
            logger.warning("  %s (%s): 데이터 부족 — 건너뜀", etf_name, etf_code)
            continue

        close = df["close"].astype(float)

        # 기술지표
        rsi = calc_rsi(close)
        bb_pct = calc_bb_pct(close)
        adx_now, adx_prev, plus_di, minus_di = calc_adx(df)
        stoch = calc_stoch(df)
        trix_val, trix_sig, trix_bull = calc_trix(close)
        ma20_gap = calc_ma_gap(close, 20)
        ma60_gap = calc_ma_gap(close, 60)
        ret_1, ret_5, ret_20 = calc_returns(close)

        # 수급
        if skip_flow:
            foreign_5d, inst_5d, foreign_today, inst_today = 0, 0, 0, 0
        else:
            foreign_5d, inst_5d, foreign_today, inst_today = get_etf_flow(etf_code, df)

        is_smart = foreign_5d > 0 and inst_5d > 0

        # 감가 리스크
        vol_risk = calc_vol_decay_risk(close)

        entry = {
            "etf_name": etf_name,
            "etf_code": etf_code,
            "direction": direction,
            "leverage": leverage,
            "category": category,
            "underlying": info.get("underlying", ""),
            "close": int(close.iloc[-1]),
            "volume": int(df["volume"].iloc[-1]),
            "ret_1": ret_1,
            "ret_5": ret_5,
            "ret_20": ret_20,
            "rsi": rsi,
            "bb_pct": bb_pct,
            "adx": adx_now,
            "adx_rising": adx_now > adx_prev,
            "stoch_k": stoch["slow_k"],
            "stoch_d": stoch["slow_d"],
            "stoch_gx": stoch["golden_cross"],
            "trix": trix_val,
            "trix_signal": trix_sig,
            "trix_bull": trix_bull,
            "ma20_gap": ma20_gap,
            "ma60_gap": ma60_gap,
            "foreign_5d": foreign_5d,
            "inst_5d": inst_5d,
            "foreign_today": foreign_today,
            "inst_today": inst_today,
            "is_smart": is_smart,
            "vol_decay_risk": vol_risk,
        }

        # 5축 스코어링
        score, grade, reasons, breakdown = calc_leverage_score(
            entry, regime, us_signal
        )
        entry["score"] = score
        entry["grade"] = grade
        entry["reasons"] = reasons
        entry["score_breakdown"] = breakdown

        # 분할매수 (적극매수/분할매수 등급만)
        if grade in ("적극매수", "분할매수"):
            entry["split_buy"] = calc_split_buy(entry["close"])

        # 동적 상한선 (지수 LONG ETF만)
        if category == "지수" and direction == "LONG":
            up_cons, up_opt = calc_upside(regime_data, direction, leverage)
            entry["upside_conservative"] = up_cons
            entry["upside_optimistic"] = up_opt

        etfs.append(entry)

        dir_icon = "📈" if direction == "LONG" else "📉"
        logger.info(
            "  %s %s (%s): %s점 [%s] | %s",
            dir_icon, etf_name, etf_code, score, grade,
            ", ".join(reasons[:3]) if reasons else "-",
        )

        time.sleep(0.15)

    # 점수순 정렬
    etfs.sort(key=lambda x: x["score"], reverse=True)

    recommendation = generate_recommendation(regime, us_grade)

    result = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "kospi_regime": regime,
        "kospi_close": regime_data.get("close", 0),
        "us_overnight": us_grade,
        "recommendation": recommendation,
        "etf_count": len(etfs),
        "etfs": etfs,
    }

    # 저장
    LEVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 요약 출력
    grades: dict[str, int] = {}
    for e in etfs:
        grades[e["grade"]] = grades.get(e["grade"], 0) + 1

    print(f"\n{'=' * 55}")
    print(f"  레버리지/인버스 ETF 스캔 결과")
    print(f"  KOSPI: {regime} ({regime_data.get('close', 0):,.0f}) | US: {us_grade}")
    print(f"  방향: {recommendation}")
    print(f"{'=' * 55}")
    print(f"  전체: {len(etfs)}개 ETF")

    for g in ["적극매수", "분할매수", "관심", "대기", "부적합", "방향불일치"]:
        cnt = grades.get(g, 0)
        if cnt > 0:
            print(f"  {g}: {cnt}개")

    # LONG TOP 5
    long_etfs = [e for e in etfs if e["direction"] == "LONG"]
    if long_etfs:
        print(f"\n  ── LONG TOP 5 ──")
        for i, e in enumerate(long_etfs[:5], 1):
            print(f"  {i}. [{e['grade']}] {e['etf_name']} — {e['score']}점")
            if e.get("split_buy"):
                sb = e["split_buy"]
                print(f"     분할매수: {sb['entry_1']:,} → {sb['entry_2']:,} → {sb['entry_3']:,} → {sb['entry_4']:,}")
            if e.get("upside_conservative"):
                print(f"     예상수익: +{e['upside_conservative']:.1f}% (보수) ~ +{e['upside_optimistic']:.1f}% (낙관)")
            if e["reasons"]:
                print(f"     사유: {', '.join(e['reasons'][:3])}")

    # SHORT TOP 3
    short_etfs = [e for e in etfs if e["direction"] == "SHORT"]
    if short_etfs:
        print(f"\n  ── SHORT TOP 3 ──")
        for i, e in enumerate(short_etfs[:3], 1):
            print(f"  {i}. [{e['grade']}] {e['etf_name']} — {e['score']}점")
            if e["reasons"]:
                print(f"     사유: {', '.join(e['reasons'][:3])}")

    print(f"\n  저장: {OUT_PATH}")

    return result


def main():
    parser = argparse.ArgumentParser(description="레버리지/인버스 ETF 로테이션 스캐너")
    parser.add_argument("--no-flow", action="store_true", help="pykrx 수급 조회 생략")
    args = parser.parse_args()

    scan_leverage_etfs(skip_flow=args.no_flow)


if __name__ == "__main__":
    main()
