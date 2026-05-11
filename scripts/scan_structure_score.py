#!/usr/bin/env python3
"""Structure Score 스캔 — S1(연간돌파) + S2(주봉StochRSI) + S3(시장레짐)

차트영 전략(ETF 연봉 돌파 → 섹터 선정)과
차자 전략(주봉 Stochastic RSI + Fear&Greed 분할매수)을 결합.

출력: data/structure_score_{YYYYMMDD}.json
병합: scan_sector_fire.py에서 로드 → FIRE×0.6 + Structure×0.4 = Composite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import yaml

from src.adapters.etf_history_adapter import EtfHistoryAdapter
from src.adapters.macro_adapter import MacroAdapter
from src.indicators import IndicatorEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "sector_fire_map.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data"


# ─────────────────────────────────────────────
# 1. 설정 로더
# ─────────────────────────────────────────────

def load_sector_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("sectors", {})


# ─────────────────────────────────────────────
# 2. S1 — 연간돌파 (40점 만점)
# ─────────────────────────────────────────────

def calc_s1(etf_adapter: EtfHistoryAdapter, etf_code: str) -> dict:
    """ETF 월봉 → 연봉 리샘플 → 과거4년 연고가 저항선 대비 현재가 비율.

    ratio >= 1.0  : 40점 (돌파 확인)
    0.95 ~ 1.0    : 30점 (근접)
    0.90 ~ 0.95   : 20점 (접근 중)
    0.85 ~ 0.90   : 10점 (회복세)
    < 0.85        :  0점 (하락 중)
    """
    if not etf_code:
        return {"s1_score": 0, "s1_ratio": 0.0, "s1_detail": "ETF 없음"}

    monthly = etf_adapter.fetch_monthly(etf_code, years=5)
    if monthly.empty or len(monthly) < 12:
        return {"s1_score": 0, "s1_ratio": 0.0, "s1_detail": "월봉 데이터 부족"}

    yearly = etf_adapter.get_yearly_from_monthly(monthly)
    if yearly.empty:
        return {"s1_score": 0, "s1_ratio": 0.0, "s1_detail": "연봉 변환 실패"}

    # 현재 연도 제외, 과거 4년 연고가 최대값 = 저항선
    current_year = datetime.now().year
    past_yearly = yearly[yearly.index.year < current_year]

    if past_yearly.empty:
        return {"s1_score": 0, "s1_ratio": 0.0, "s1_detail": "과거 데이터 부족"}

    resistance = past_yearly["High"].max()
    current_price = monthly["Close"].iloc[-1]

    if resistance <= 0:
        return {"s1_score": 0, "s1_ratio": 0.0, "s1_detail": "저항선 0"}

    ratio = current_price / resistance

    if ratio >= 1.0:
        score = 40
        detail = f"돌파확인 (ratio={ratio:.3f})"
    elif ratio >= 0.95:
        score = 30
        detail = f"저항선근접 (ratio={ratio:.3f})"
    elif ratio >= 0.90:
        score = 20
        detail = f"접근중 (ratio={ratio:.3f})"
    elif ratio >= 0.85:
        score = 10
        detail = f"회복세 (ratio={ratio:.3f})"
    else:
        score = 0
        detail = f"하락중 (ratio={ratio:.3f})"

    return {"s1_score": score, "s1_ratio": round(ratio, 3), "s1_detail": detail}


# ─────────────────────────────────────────────
# 3. S2 — 주봉 StochRSI (30점 만점)
# ─────────────────────────────────────────────

def calc_s2(etf_adapter: EtfHistoryAdapter, ticker: str) -> dict:
    """주봉 close → RSI(14) → StochRSI(14,3,3).

    K <= 20      : 30점 (깊은 과매도 — 최적 매수 구간)
    20 < K <= 30 : 25점 (과매도)
    30 < K <= 50 : 15점 (중립 하단)
    50 < K <= 70 : 10점 (중립 상단)
    70 < K <= 80 :  5점 (과매수 진입)
    K > 80       :  0점 (과매수 — 매수 자제)
    """
    if not ticker:
        return {"s2_score": 0, "s2_stoch_k": None, "s2_detail": "티커 없음"}

    weekly = etf_adapter.fetch_weekly(ticker, weeks=52)
    if weekly.empty or len(weekly) < 30:
        return {"s2_score": 0, "s2_stoch_k": None, "s2_detail": "주봉 데이터 부족"}

    close = weekly["Close"]
    rsi = IndicatorEngine.calc_rsi(close, period=14)
    stoch = IndicatorEngine.calc_stoch_rsi(rsi, period=14, smooth_k=3, smooth_d=3)

    k_val = stoch["stoch_rsi_k"].dropna()
    if k_val.empty:
        return {"s2_score": 0, "s2_stoch_k": None, "s2_detail": "StochRSI 계산 불가"}

    k = k_val.iloc[-1]

    if k <= 20:
        score = 30
        detail = f"깊은과매도 (K={k:.1f})"
    elif k <= 30:
        score = 25
        detail = f"과매도 (K={k:.1f})"
    elif k <= 50:
        score = 15
        detail = f"중립하단 (K={k:.1f})"
    elif k <= 70:
        score = 10
        detail = f"중립상단 (K={k:.1f})"
    elif k <= 80:
        score = 5
        detail = f"과매수진입 (K={k:.1f})"
    else:
        score = 0
        detail = f"과매수 (K={k:.1f})"

    return {"s2_score": score, "s2_stoch_k": round(k, 1), "s2_detail": detail}


# ─────────────────────────────────────────────
# 4. S3 — 시장레짐 (30점 만점, 전 섹터 공통)
# ─────────────────────────────────────────────

def calc_s3(macro: MacroAdapter, etf_adapter: EtfHistoryAdapter) -> dict:
    """S3 = KOSPI주봉StochRSI(12) + VIX(10) + 공포탐욕프록시(8).

    KOSPI 주봉 StochRSI:
      K <= 20: 12점 / 20<K<=40: 8점 / 40<K<=60: 5점 / 60<K<=80: 2점 / >80: 0점

    VIX:
      <= 15: 10점 / 15~20: 7점 / 20~25: 4점 / 25~30: 2점 / >30: 0점

    공포탐욕 프록시 (KOSPI 20일 이격도):
      <= 95%: 8점 / 95~98: 5점 / 98~102: 3점 / 102~105: 1점 / >105: 0점
    """
    result = {
        "s3_score": 0,
        "kospi_stoch_k": None,
        "kospi_stoch_sub": 0,
        "vix_close": None,
        "vix_sub": 0,
        "disparity": None,
        "disparity_sub": 0,
        "s3_detail": "",
    }

    # ── KOSPI 주봉 StochRSI ── (^KS11 for yfinance, KS11 for FDR)
    kospi_weekly = etf_adapter.fetch_weekly("^KS11", weeks=52)
    if kospi_weekly.empty:
        kospi_weekly = etf_adapter.fetch_weekly("KS11", weeks=52)
    if not kospi_weekly.empty and len(kospi_weekly) >= 30:
        close = kospi_weekly["Close"]
        rsi = IndicatorEngine.calc_rsi(close, period=14)
        stoch = IndicatorEngine.calc_stoch_rsi(rsi, period=14, smooth_k=3, smooth_d=3)
        k_vals = stoch["stoch_rsi_k"].dropna()
        if not k_vals.empty:
            k = k_vals.iloc[-1]
            result["kospi_stoch_k"] = round(k, 1)
            if k <= 20:
                result["kospi_stoch_sub"] = 12
            elif k <= 40:
                result["kospi_stoch_sub"] = 8
            elif k <= 60:
                result["kospi_stoch_sub"] = 5
            elif k <= 80:
                result["kospi_stoch_sub"] = 2
            else:
                result["kospi_stoch_sub"] = 0

    # ── VIX ──
    end_str = datetime.now().strftime("%Y-%m-%d")
    start_str = (datetime.now() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    vix_df = macro.fetch_vix(start_str, end_str)
    if not vix_df.empty:
        vix = vix_df["vix_close"].iloc[-1]
        result["vix_close"] = round(float(vix), 1)
        if vix <= 15:
            result["vix_sub"] = 10
        elif vix <= 20:
            result["vix_sub"] = 7
        elif vix <= 25:
            result["vix_sub"] = 4
        elif vix <= 30:
            result["vix_sub"] = 2
        else:
            result["vix_sub"] = 0

    # ── 공포탐욕 프록시 (KOSPI 20일 이격도) ──
    # 방법1: 주봉 데이터에서 마지막 close로 일봉 대체
    # 방법2: macro_adapter FDR (일봉)
    kospi_start = (datetime.now() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    kospi_df = macro.fetch_kospi(kospi_start, end_str)
    if kospi_df.empty:
        # FDR 실패 시 EtfHistoryAdapter 일봉 raw 사용
        kospi_daily = etf_adapter._fetch_daily_raw(
            "^KS11", datetime.now() - pd.Timedelta(days=40), datetime.now()
        )
        if not kospi_daily.empty:
            kospi_df = kospi_daily[["Close"]].rename(columns={"Close": "kospi_close"})
    if not kospi_df.empty and len(kospi_df) >= 20:
        close_series = kospi_df["kospi_close"]
        ma20 = close_series.rolling(20).mean()
        disparity = (close_series / ma20 * 100).iloc[-1]
        result["disparity"] = round(float(disparity), 1)
        if disparity <= 95:
            result["disparity_sub"] = 8
        elif disparity <= 98:
            result["disparity_sub"] = 5
        elif disparity <= 102:
            result["disparity_sub"] = 3
        elif disparity <= 105:
            result["disparity_sub"] = 1
        else:
            result["disparity_sub"] = 0

    result["s3_score"] = result["kospi_stoch_sub"] + result["vix_sub"] + result["disparity_sub"]
    parts = []
    if result["kospi_stoch_k"] is not None:
        parts.append(f"KOSPI_K={result['kospi_stoch_k']}")
    if result["vix_close"] is not None:
        parts.append(f"VIX={result['vix_close']}")
    if result["disparity"] is not None:
        parts.append(f"이격도={result['disparity']}%")
    result["s3_detail"] = " / ".join(parts)

    return result


# ─────────────────────────────────────────────
# 5. 등급 산출
# ─────────────────────────────────────────────

def score_to_grade(score: float) -> str:
    if score >= 85:
        return "S+"
    elif score >= 70:
        return "S"
    elif score >= 55:
        return "A"
    elif score >= 40:
        return "B"
    elif score >= 25:
        return "C"
    else:
        return "D"


# ─────────────────────────────────────────────
# 6. 메인 스캔
# ─────────────────────────────────────────────

def scan_structure_score() -> dict:
    """16개 섹터 Structure Score 스캔."""
    sector_cfg = load_sector_config()
    if not sector_cfg:
        logger.error("섹터 설정 로드 실패: %s", CONFIG_PATH)
        return {}

    etf_adapter = EtfHistoryAdapter()
    macro = MacroAdapter()

    # S3: 시장 레짐 (전 섹터 공통, 1회만)
    logger.info("━━ S3 시장레짐 계산 ━━")
    s3 = calc_s3(macro, etf_adapter)
    logger.info("  S3=%d점 (%s)", s3["s3_score"], s3["s3_detail"])

    market_regime = {
        "kospi_stoch_k": s3["kospi_stoch_k"],
        "vix": s3["vix_close"],
        "disparity": s3["disparity"],
        "s3_score": s3["s3_score"],
        "s3_detail": s3["s3_detail"],
    }

    # 섹터별 S1 + S2
    logger.info("━━ 섹터별 S1/S2 계산 ━━")
    sectors = {}

    for sector_name, cfg in sector_cfg.items():
        etf_info = cfg.get("etf")
        etf_code = etf_info["code"] if isinstance(etf_info, dict) else None
        fallback_ticker = cfg.get("fallback_ticker")

        # S1: 연간돌파 — ETF 기준
        s1 = calc_s1(etf_adapter, etf_code)

        # S2: 주봉 StochRSI — ETF 우선, 없으면 fallback_ticker
        s2_ticker = etf_code or fallback_ticker
        s2 = calc_s2(etf_adapter, s2_ticker)

        total = s1["s1_score"] + s2["s2_score"] + s3["s3_score"]
        grade = score_to_grade(total)

        sectors[sector_name] = {
            "s1_score": s1["s1_score"],
            "s1_ratio": s1["s1_ratio"],
            "s1_detail": s1["s1_detail"],
            "s2_score": s2["s2_score"],
            "s2_stoch_k": s2["s2_stoch_k"],
            "s2_ticker": s2_ticker or "",
            "s2_detail": s2["s2_detail"],
            "s3_score": s3["s3_score"],
            "total": total,
            "grade": grade,
        }

        logger.info("  %-10s S1=%2d S2=%2d S3=%2d → Total=%2d (%s)",
                     sector_name, s1["s1_score"], s2["s2_score"], s3["s3_score"],
                     total, grade)

    return {
        "date": datetime.now().strftime("%Y%m%d"),
        "market_regime": market_regime,
        "sectors": sectors,
    }


# ─────────────────────────────────────────────
# 7. 리포트 + 저장
# ─────────────────────────────────────────────

def print_report(result: dict):
    """콘솔 리포트."""
    SEP = "=" * 100
    today = result.get("date", "")
    mr = result.get("market_regime", {})

    print(f"\n{SEP}")
    print(f"  Structure Score — {today}")
    print(f"  전략: 차트영(ETF 연봉돌파) + 차자(주봉 StochRSI + Fear&Greed)")
    print(SEP)

    print(f"\n  [시장 레짐 — S3 = {mr.get('s3_score', 0)}점]")
    print(f"    KOSPI StochRSI K = {mr.get('kospi_stoch_k', 'N/A')}")
    print(f"    VIX             = {mr.get('vix', 'N/A')}")
    print(f"    KOSPI 이격도     = {mr.get('disparity', 'N/A')}%")

    print(f"\n  {'섹터':>12} {'S1연간':>6} {'비율':>6} {'S2StRSI':>7} {'StK':>5} {'S3시장':>6} {'합계':>5} {'등급':>4}")
    print(f"  {'─' * 60}")

    sectors = result.get("sectors", {})
    sorted_sectors = sorted(sectors.items(), key=lambda x: x[1]["total"], reverse=True)

    for name, s in sorted_sectors:
        stoch_k = f"{s['s2_stoch_k']:.0f}" if s['s2_stoch_k'] is not None else "N/A"
        print(f"  {name:>12} {s['s1_score']:>6} {s['s1_ratio']:>6.3f} "
              f"{s['s2_score']:>7} {stoch_k:>5} {s['s3_score']:>6} "
              f"{s['total']:>5} {s['grade']:>4}")

    print(f"\n  [등급 기준] S+(85+) / S(70+) / A(55+) / B(40+) / C(25+) / D(<25)")
    print(SEP)


def save_output(result: dict):
    """JSON 저장."""
    date_str = result.get("date", datetime.now().strftime("%Y%m%d"))
    json_path = OUTPUT_DIR / f"structure_score_{date_str}.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Structure Score 저장: %s", json_path)


# ─────────────────────────────────────────────
# 8. 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Structure Score 스캔")
    parser.add_argument("--date", type=str, default="", help="날짜 YYYYMMDD (기본: 오늘)")
    args = parser.parse_args()

    result = scan_structure_score()
    if not result:
        logger.error("Structure Score 스캔 실패")
        return

    if args.date:
        result["date"] = args.date

    print_report(result)
    save_output(result)


if __name__ == "__main__":
    main()
