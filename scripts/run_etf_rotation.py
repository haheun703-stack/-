#!/usr/bin/env python3
"""
ETF 3축 로테이션 일일 실행 스크립트
=============================================
사용법:
    python -u -X utf8 scripts/run_etf_rotation.py [--dry-run] [--no-telegram]

    --dry-run       : 결과 저장만 (텔레그램 발송 안 함)
    --no-telegram   : 텔레그램 발송 안 함

데이터 소스 (기존 파이프라인에서 생성됨):
    data/sector_rotation/sector_momentum.json
    data/sector_rotation/etf_trading_signal.json
    data/sector_rotation/investor_flow.json
    data/us_market/overnight_signal.json
    data/leverage_etf/leverage_etf_scan.json
    data/kospi_index.csv
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# BAT 실행 대비 PYTHONPATH 안전장치
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.etf.orchestrator import ETFOrchestrator
from src.etf.config import build_sector_universe

logger = logging.getLogger(__name__)

# ============================================================
# 경로 상수
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"
MOMENTUM_PATH = DATA_DIR / "sector_rotation" / "sector_momentum.json"
SIGNAL_PATH = DATA_DIR / "sector_rotation" / "etf_trading_signal.json"
FLOW_PATH = DATA_DIR / "sector_rotation" / "investor_flow.json"
OVERNIGHT_PATH = DATA_DIR / "us_market" / "overnight_signal.json"
LEVERAGE_SCAN_PATH = DATA_DIR / "leverage_etf" / "leverage_etf_scan.json"
KOSPI_PATH = DATA_DIR / "kospi_index.csv"
OUTPUT_PATH = DATA_DIR / "etf_rotation_result.json"


# ============================================================
# 데이터 로드 + 형식 변환
# ============================================================
def _load_momentum() -> dict:
    """sector_momentum.json → {sector: {"5d", "20d", "60d", "rank"}}.

    원본 키: ret_5, ret_20, ret_60, rank
    """
    if not MOMENTUM_PATH.exists():
        logger.warning("sector_momentum.json 없음 — 빈 데이터 사용")
        return {}

    raw = json.loads(MOMENTUM_PATH.read_text(encoding="utf-8"))
    sectors = raw.get("sectors", [])
    result = {}
    for s in sectors:
        sector_name = s.get("sector", "")
        result[sector_name] = {
            "5d": s.get("ret_5", 0),
            "20d": s.get("ret_20", 0),
            "60d": s.get("ret_60", 0),
            "rank": s.get("rank", 99),
        }
    return result


def _load_smart_money() -> dict:
    """etf_trading_signal.json → {etf_code: {"type", "score", "sector"}}.

    smart_money_etf → type="smart_money"
    theme_money_etf → type="theme_money"
    smart_sectors 리스트의 섹터는 해당 ETF를 universe에서 찾아 smart_money 매핑
    """
    if not SIGNAL_PATH.exists():
        logger.warning("etf_trading_signal.json 없음 — 빈 데이터 사용")
        return {}

    raw = json.loads(SIGNAL_PATH.read_text(encoding="utf-8"))
    result = {}

    # smart_money_etf 배열 처리
    for entry in raw.get("smart_money_etf", []):
        code = entry.get("etf_code", "")
        if code:
            result[code] = {
                "type": "smart_money",
                "score": entry.get("momentum_score", 0),
                "sector": entry.get("sector", ""),
            }

    # theme_money_etf 배열 처리
    for entry in raw.get("theme_money_etf", []):
        code = entry.get("etf_code", "")
        if code and code not in result:
            result[code] = {
                "type": "theme_money",
                "score": entry.get("momentum_score", 0),
                "sector": entry.get("sector", ""),
            }

    # smart_sectors: 섹터명 리스트 → universe에서 해당 섹터 ETF 찾아서 매핑
    smart_sectors = set(raw.get("smart_sectors", []))
    if smart_sectors:
        universe = build_sector_universe()
        for code, info in universe.items():
            if info["sector"] in smart_sectors and code not in result:
                result[code] = {
                    "type": "smart_money",
                    "score": 70,  # 기본 점수
                    "sector": info["sector"],
                }

    return result


def _load_supply() -> dict:
    """investor_flow.json → {etf_code: {"foreign_net_5d", "inst_net_5d", "score"}}.

    원본은 섹터 기반 → ETF 코드로 매핑 필요.
    수급 점수: foreign_cum + inst_cum 기반 정규화.
    """
    if not FLOW_PATH.exists():
        logger.warning("investor_flow.json 없음 — 빈 데이터 사용")
        return {}

    raw = json.loads(FLOW_PATH.read_text(encoding="utf-8"))
    sectors_data = raw.get("sectors", [])

    if not sectors_data:
        return {}

    # 수급 점수 계산: (외인+기관 누적 합) 기준 0~100 정규화
    totals = []
    for s in sectors_data:
        foreign = s.get("foreign_cum_bil", 0)
        inst = s.get("inst_cum_bil", 0)
        totals.append(foreign + inst)

    max_total = max(totals) if totals else 1
    min_total = min(totals) if totals else 0
    spread = max_total - min_total if max_total != min_total else 1

    # 섹터 → ETF 코드 매핑 (investor_flow의 etf_code + universe에서 매칭)
    universe = build_sector_universe()
    sector_to_codes = {}
    for code, info in universe.items():
        sector = info["sector"]
        if sector not in sector_to_codes:
            sector_to_codes[sector] = []
        sector_to_codes[sector].append(code)

    result = {}
    for i, s in enumerate(sectors_data):
        sector = s.get("sector", "")
        foreign_bil = s.get("foreign_cum_bil", 0)
        inst_bil = s.get("inst_cum_bil", 0)
        total = foreign_bil + inst_bil
        score = max(0, min(100, ((total - min_total) / spread) * 100))

        # investor_flow 자체의 etf_code
        flow_code = s.get("etf_code", "")
        if flow_code:
            result[flow_code] = {
                "foreign_net_5d": foreign_bil * 1e9,  # 억→원 (대략)
                "inst_net_5d": inst_bil * 1e9,
                "score": round(score),
            }

        # universe에서 동일 섹터의 ETF 코드에도 매핑
        for code in sector_to_codes.get(sector, []):
            if code not in result:
                result[code] = {
                    "foreign_net_5d": foreign_bil * 1e9,
                    "inst_net_5d": inst_bil * 1e9,
                    "score": round(score),
                }

    return result


def _load_us_overnight() -> dict:
    """overnight_signal.json → {"grade": 1~5, "signal": str}.

    원본 grade: "STRONG_BULL"/"MILD_BULL"/"NEUTRAL"/"MILD_BEAR"/"STRONG_BEAR"
    엔진 기대: grade=1~5 (숫자), signal=문자열
    """
    if not OVERNIGHT_PATH.exists():
        logger.warning("overnight_signal.json 없음 — 기본값 사용")
        return {"grade": 3, "signal": "neutral"}

    raw = json.loads(OVERNIGHT_PATH.read_text(encoding="utf-8"))
    grade_str = raw.get("grade", "NEUTRAL").upper()

    grade_map = {
        "STRONG_BULL": 1,
        "MILD_BULL": 2,
        "NEUTRAL": 3,
        "MILD_BEAR": 4,
        "STRONG_BEAR": 5,
    }
    signal_map = {
        "STRONG_BULL": "strong_positive",
        "MILD_BULL": "positive",
        "NEUTRAL": "neutral",
        "MILD_BEAR": "negative",
        "STRONG_BEAR": "strong_negative",
    }

    return {
        "grade": grade_map.get(grade_str, 3),
        "signal": signal_map.get(grade_str, "neutral"),
    }


def _load_leverage_5axis() -> float:
    """leverage_etf_scan.json → 최고 점수 ETF의 score (0~100)."""
    if not LEVERAGE_SCAN_PATH.exists():
        logger.warning("leverage_etf_scan.json 없음 — 기본 0점")
        return 0

    raw = json.loads(LEVERAGE_SCAN_PATH.read_text(encoding="utf-8"))
    etfs = raw.get("etfs", [])
    if not etfs:
        return 0

    # 최고 점수 ETF의 score 반환
    best = max(etfs, key=lambda x: x.get("score", 0))
    return float(best.get("score", 0))


def _calc_kospi_regime() -> dict:
    """KOSPI 레짐 계산 (dashboard_data_provider 로직 재사용).

    Returns:
        {
            "regime": "BULL"/"CAUTION"/"BEAR"/"CRISIS",
            "close": float,
            "ma20": float,
            "ma60": float,
            "ma20_above": bool,
            "ma60_above": bool,
        }
    """
    if not KOSPI_PATH.exists():
        logger.warning("kospi_index.csv 없음 — 기본 CAUTION")
        return {
            "regime": "CAUTION",
            "close": 0,
            "ma20": 0,
            "ma60": 0,
            "ma20_above": True,
            "ma60_above": False,
        }

    try:
        df = pd.read_csv(KOSPI_PATH, index_col="Date", parse_dates=True).sort_index()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()

        if len(df) < 60:
            return {
                "regime": "CAUTION", "close": 0,
                "ma20": 0, "ma60": 0,
                "ma20_above": True, "ma60_above": False,
            }

        row = df.iloc[-1]
        close = float(row["close"])
        ma20 = float(row["ma20"]) if not pd.isna(row["ma20"]) else 0
        ma60 = float(row["ma60"]) if not pd.isna(row["ma60"]) else 0

        # 실현 변동성 백분위
        log_ret = np.log(df["close"] / df["close"].shift(1))
        rv20 = log_ret.rolling(20).std() * np.sqrt(252) * 100
        rv20_pct = rv20.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        rv_pct = float(rv20_pct.iloc[-1]) if not pd.isna(rv20_pct.iloc[-1]) else 0.5

        # 레짐 판정
        if ma20 == 0 or ma60 == 0:
            regime = "CAUTION"
        elif close > ma20:
            regime = "BULL" if rv_pct < 0.50 else "CAUTION"
        elif close > ma60:
            regime = "BEAR"
        else:
            regime = "CRISIS"

        return {
            "regime": regime,
            "close": round(close, 2),
            "ma20": round(ma20, 2),
            "ma60": round(ma60, 2),
            "ma20_above": close > ma20 if ma20 > 0 else False,
            "ma60_above": close > ma60 if ma60 > 0 else False,
        }
    except Exception as e:
        logger.error("KOSPI 레짐 계산 실패: %s", e)
        return {
            "regime": "CAUTION", "close": 0,
            "ma20": 0, "ma60": 0,
            "ma20_above": True, "ma60_above": False,
        }


# ============================================================
# 메인 실행
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ETF 3축 로테이션 일일 실행")
    parser.add_argument("--dry-run", action="store_true", help="텔레그램 발송 안 함")
    parser.add_argument("--no-telegram", action="store_true", help="텔레그램 발송 안 함")
    args = parser.parse_args()

    send_telegram = not (args.dry_run or args.no_telegram)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print(f"\n🚀 ETF 3축 로테이션 시작 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- 1. 데이터 로드 ----
    print("\n📂 데이터 로드 중...")
    momentum_data = _load_momentum()
    smart_money_data = _load_smart_money()
    supply_data = _load_supply()
    us_overnight = _load_us_overnight()
    five_axis_score = _load_leverage_5axis()
    kospi = _calc_kospi_regime()

    print(f"  📊 KOSPI 레짐: {kospi['regime']} (종가 {kospi['close']:,.0f})")
    print(f"  📊 모멘텀 섹터: {len(momentum_data)}개")
    print(f"  📊 Smart Money: {len(smart_money_data)}개")
    print(f"  📊 수급 데이터: {len(supply_data)}개")
    print(f"  📊 US Overnight: {us_overnight['grade']}등급 ({us_overnight['signal']})")
    print(f"  📊 레버리지 5축: {five_axis_score:.0f}점")

    # ---- 2. 오케스트레이터 실행 ----
    orchestrator = ETFOrchestrator()
    result = orchestrator.run(
        regime=kospi["regime"],
        kospi_ma20_above=kospi["ma20_above"],
        kospi_ma60_above=kospi["ma60_above"],
        momentum_data=momentum_data,
        smart_money_data=smart_money_data,
        supply_data=supply_data,
        us_overnight=us_overnight,
        five_axis_score=five_axis_score,
    )

    # ---- 3. 결과 JSON 저장 ----
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in result.items() if k != "telegram_report"}
    OUTPUT_PATH.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n💾 결과 저장: {OUTPUT_PATH}")

    # ---- 4. 텔레그램 발송 ----
    if send_telegram:
        try:
            from src.telegram_sender import send_message
            report = result.get("telegram_report", "")
            if report:
                ok = send_message(report)
                if ok:
                    print("📨 텔레그램 발송 완료")
                else:
                    print("⚠️ 텔레그램 발송 실패")
            else:
                print("⚠️ 리포트 내용 없음 — 텔레그램 스킵")
        except Exception as e:
            print(f"⚠️ 텔레그램 발송 오류: {e}")
    else:
        print("📭 텔레그램 발송 스킵 (--dry-run / --no-telegram)")

    print(f"\n✅ ETF 3축 로테이션 완료 — {datetime.now().strftime('%H:%M:%S')}")
    return result


if __name__ == "__main__":
    main()
