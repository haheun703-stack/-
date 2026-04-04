"""
급락반등 포착 스캐너 — 백테스트 검증 STRONG_ALPHA 시그널

백테스트 결과 (2019~2026, 1071종목):
  볼린저급락 반등 : D+5 +3.38%, 승률 60.2%, 손익비 2.64 (n=1034)
  거래량폭발 반등 : D+5 +3.31%, 승률 62.5%, 손익비 2.61 (n=75)

조건:
  ● 볼린저급락 반등
    - 20일 이동평균 대비 -15% 이상 급락
    - 볼린저밴드 하단(2σ) 아래로 이탈
    - 기관 또는 외국인 당일 순매수
    → 해석: "극도로 빠진 상태 + 통계적 바닥 + 큰손이 사기 시작"

  ● 거래량폭발 반등
    - 20일 이동평균 대비 -15% 이상 급락
    - 거래량이 20일 평균의 3배 이상 폭발
    - 기관 또는 외국인 당일 순매수
    → 해석: "급락 후 거래가 터지면서 큰손이 담는 중"

  ● 겹치기 (두 조건 동시 충족 → 최고등급)

등급:
  ★ 적극매수 — 두 시그널 동시 또는 쌍끌이(기관+외인 동시) 수급
  ◎ 매수     — 단일 시그널 + 기관 또는 외인 순매수
  ○ 관심     — 조건 근접 (이격 -12%~-15% 등)

Usage:
    python scripts/scan_crash_bounce.py
    python scripts/scan_crash_bounce.py --top 20
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

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "crash_bounce_scan.json"

logger = logging.getLogger(__name__)


def _sf(val, default=0):
    """NaN/Inf 안전 변환"""
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


# ──────────────────────────────────────────
# 종목명 매핑
# ──────────────────────────────────────────

def build_name_map() -> dict[str, str]:
    """종목코드 → 종목명 매핑 (universe.csv 우선 → CSV 폴백 → pykrx 폴백)."""
    name_map = {}

    # 1) universe.csv — rebuild_universe.py가 매일 G1에서 갱신 (가장 신뢰)
    universe_path = PROJECT_ROOT / "data" / "universe.csv"
    if universe_path.exists():
        try:
            import csv as csv_mod
            with open(universe_path, encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    t = row.get("ticker", "").strip()
                    n = row.get("name", "").strip()
                    if t and n:
                        name_map[t] = n
            if name_map:
                logger.info("종목명 매핑: universe.csv %d건", len(name_map))
                return name_map
        except Exception as e:
            logger.warning("universe.csv 로드 실패: %s", e)

    # 2) stock_data_daily CSV 파일명 (로컬 전용)
    csv_dir = PROJECT_ROOT / "stock_data_daily"
    if csv_dir.exists():
        for csv in csv_dir.glob("*.csv"):
            parts = csv.stem.rsplit("_", 1)
            if len(parts) == 2:
                name, ticker = parts
                name_map[ticker] = name
        if name_map:
            logger.info("종목명 매핑: stock_data_daily %d건", len(name_map))
            return name_map

    # 3) pykrx 폴백 (장중에만 안정적)
    try:
        from pykrx import stock as krx
        from datetime import timedelta
        date = datetime.now()
        for _ in range(7):
            date_str = date.strftime("%Y%m%d")
            try:
                for market in ["KOSPI", "KOSDAQ"]:
                    tickers = krx.get_market_ticker_list(date_str, market=market)
                    for t in tickers:
                        if t not in name_map:
                            name_map[t] = krx.get_market_ticker_name(t)
                if name_map:
                    return name_map
            except Exception:
                pass
            date -= timedelta(days=1)
    except ImportError:
        pass

    return name_map


# ──────────────────────────────────────────
# 스캔 로직
# ──────────────────────────────────────────

def scan_crash_bounce(top_n: int = 30) -> dict:
    """전종목 급락반등 스캔"""
    name_map = build_name_map()
    parquets = sorted(RAW_DIR.glob("*.parquet"))
    logger.info("원본 데이터 %d개 종목 스캔 시작", len(parquets))

    results = []

    for pq in parquets:
        ticker = pq.stem
        try:
            df = pd.read_parquet(pq)
            if len(df) < 60:
                continue

            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            if "기관합계" not in df.columns or "외국인합계" not in df.columns:
                continue

            close = df["close"]
            volume = df["volume"].fillna(0)
            inst = df["기관합계"].fillna(0)
            foreign = df["외국인합계"].fillna(0)

            # 최신 데이터 (오늘)
            today = df.iloc[-1]
            today_close = _sf(today["close"])
            if today_close <= 0:
                continue

            # ── 지표 계산 ──
            ma20 = close.rolling(20).mean()
            today_ma20 = _sf(ma20.iloc[-1])
            if today_ma20 <= 0:
                continue

            # 이격도 (20일 이동평균 대비 %)
            이격도 = round((today_close - today_ma20) / today_ma20 * 100, 2)

            # 볼린저밴드
            bb_std = close.rolling(20).std()
            bb_lower = ma20 - 2 * bb_std
            today_bb_lower = _sf(bb_lower.iloc[-1])
            볼린저하단_이탈 = today_close < today_bb_lower

            # 볼린저 위치 (0~100%, 하단=0 상단=100)
            bb_upper = ma20 + 2 * bb_std
            today_bb_upper = _sf(bb_upper.iloc[-1])
            bb_range = today_bb_upper - today_bb_lower
            볼린저위치 = round((today_close - today_bb_lower) / bb_range * 100, 1) if bb_range > 0 else 50

            # 거래량 배수
            vol_ma20 = volume.rolling(20).mean()
            today_vol = _sf(volume.iloc[-1])
            today_vol_ma = _sf(vol_ma20.iloc[-1])
            거래량배수 = round(today_vol / today_vol_ma, 1) if today_vol_ma > 0 else 0

            # 수급 (당일)
            기관_당일 = _sf(inst.iloc[-1])
            외인_당일 = _sf(foreign.iloc[-1])
            수급_있음 = 기관_당일 > 0 or 외인_당일 > 0
            쌍끌이 = 기관_당일 > 0 and 외인_당일 > 0

            # 수급 연속 일수
            def 연속매수일(series):
                vals = series.values
                streak = 0
                for v in reversed(vals):
                    if _sf(v) > 0:
                        streak += 1
                    else:
                        break
                return streak

            외인연속 = 연속매수일(foreign.tail(10))
            기관연속 = 연속매수일(inst.tail(10))

            # 전일 대비 등락률
            전일대비 = round((today_close / _sf(close.iloc[-2], today_close) - 1) * 100, 2) if len(df) >= 2 else 0

            # 5일 수익률
            수익률_5d = round((today_close / _sf(close.iloc[-6], today_close) - 1) * 100, 2) if len(df) >= 6 else 0

            # ── 시그널 판정 ──
            급락_15 = 이격도 <= -15
            급락_12 = 이격도 <= -12  # 관심 등급용

            시그널 = []
            if 급락_15 and 볼린저하단_이탈 and 수급_있음:
                시그널.append("볼린저급락 반등")
            if 급락_15 and 거래량배수 >= 3 and 수급_있음:
                시그널.append("거래량폭발 반등")

            # 관심 등급: 조건에 근접
            관심 = False
            if not 시그널 and 급락_12 and 수급_있음:
                if 볼린저위치 < 15 or 거래량배수 >= 2:
                    관심 = True

            if not 시그널 and not 관심:
                continue

            # ── 등급 ──
            if len(시그널) >= 2 or (시그널 and 쌍끌이):
                등급 = "적극매수"
            elif 시그널:
                등급 = "매수"
            else:
                등급 = "관심"

            # ── 이유 생성 (주린이 친화) ──
            이유 = []
            이유.append(f"20일 평균 대비 {이격도:+.1f}% 급락")
            if 볼린저하단_이탈:
                이유.append(f"볼린저밴드 하단 이탈 (위치 {볼린저위치:.0f}%)")
            if 거래량배수 >= 3:
                이유.append(f"거래량 평소의 {거래량배수:.1f}배 폭발")
            elif 거래량배수 >= 2:
                이유.append(f"거래량 평소의 {거래량배수:.1f}배")
            if 쌍끌이:
                이유.append(f"기관+외인 동시 매수 (쌍끌이)")
            elif 외인_당일 > 0:
                이유.append(f"외국인 매수 (연속 {외인연속}일)" if 외인연속 > 1 else "외국인 당일 매수")
            elif 기관_당일 > 0:
                이유.append(f"기관 매수 (연속 {기관연속}일)" if 기관연속 > 1 else "기관 당일 매수")
            if 수익률_5d <= -10:
                이유.append(f"5일간 {수익률_5d:+.1f}% 하락 (낙폭과대)")

            # 점수 (100점 만점)
            점수 = 0
            if "볼린저급락 반등" in 시그널:
                점수 += 40
            if "거래량폭발 반등" in 시그널:
                점수 += 40
            if 쌍끌이:
                점수 += 20
            elif 수급_있음:
                점수 += 10
            if 이격도 <= -20:
                점수 += 10
            elif 이격도 <= -15:
                점수 += 5
            if 거래량배수 >= 5:
                점수 += 10
            elif 거래량배수 >= 3:
                점수 += 5
            점수 = min(점수, 100)

            # 관심 등급은 낮은 점수
            if 등급 == "관심":
                점수 = min(점수, 30)

            results.append({
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "close": int(today_close),
                "전일대비": 전일대비,
                "수익률_5일": 수익률_5d,
                "이격도_20일": 이격도,
                "볼린저위치": 볼린저위치,
                "볼린저하단": int(today_bb_lower),
                "거래량배수": 거래량배수,
                "기관_당일": round(기관_당일 / 1e8, 1),  # 억원 단위
                "외인_당일": round(외인_당일 / 1e8, 1),
                "기관연속매수": 기관연속,
                "외인연속매수": 외인연속,
                "시그널": 시그널 if 시그널 else ["관심"],
                "등급": 등급,
                "점수": 점수,
                "이유": 이유,
                "백테스트": {
                    "볼린저급락": {"승률": 60.2, "평균수익": 3.38, "손익비": 2.64, "표본": 1034},
                    "거래량폭발": {"승률": 62.5, "평균수익": 3.31, "손익비": 2.61, "표본": 75},
                },
            })

        except Exception as e:
            logger.debug("스캔 실패 %s: %s", ticker, e)
            continue

    # 점수 내림차순 정렬
    results.sort(key=lambda x: -x["점수"])

    # 등급별 카운트
    등급_집계 = {}
    for r in results:
        g = r["등급"]
        등급_집계[g] = 등급_집계.get(g, 0) + 1

    report = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scanner": "급락반등 포착기",
        "설명": "극도로 빠진 종목 중 큰손(기관/외국인)이 매수하기 시작한 종목. "
                "백테스트 승률 60%+, 5일 평균수익 +3.3%",
        "total_scanned": len(parquets),
        "detected": len(results),
        "등급별": 등급_집계,
        "candidates": results[:top_n],
        "all": results,
    }

    return report


def save_report(report: dict) -> Path:
    """JSON 저장"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("저장: %s (%d건)", OUTPUT_PATH, report["detected"])
    return OUTPUT_PATH


def print_report(report: dict) -> None:
    """콘솔 출력"""
    print()
    print("=" * 75)
    print("  급락반등 포착기 — 백테스트 검증 STRONG_ALPHA 시그널")
    print(f"  {report['updated_at']}  |  전체 {report['total_scanned']}종목 → {report['detected']}건 포착")
    print(f"  등급: {report['등급별']}")
    print("=" * 75)
    print()
    print("  %-4s %-10s %8s %7s %7s %6s %6s %s" % (
        "등급", "종목명", "현재가", "이격도", "거래량", "기관", "외인", "시그널"))
    print("  " + "-" * 68)

    for r in report["candidates"]:
        등급표시 = {"적극매수": "★", "매수": "◎", "관심": "○"}.get(r["등급"], " ")
        시그널_짧게 = "+".join(s.replace(" 반등", "") for s in r["시그널"])
        기관 = f"{r['기관_당일']:+.0f}억" if r['기관_당일'] != 0 else "-"
        외인 = f"{r['외인_당일']:+.0f}억" if r['외인_당일'] != 0 else "-"
        print("  %s%-3s %-10s %8s %6.1f%% %5.1f배 %6s %6s  %s" % (
            등급표시, r["등급"], r["name"][:10],
            f"{r['close']:,}", r["이격도_20일"],
            r["거래량배수"], 기관, 외인, 시그널_짧게))

    print()
    for r in report["candidates"]:
        if r["등급"] in ("적극매수", "매수"):
            print(f"  [{r['등급']}] {r['name']} ({r['ticker']})")
            for reason in r["이유"]:
                print(f"    → {reason}")
            print()


def main():
    parser = argparse.ArgumentParser(description="급락반등 포착 스캐너")
    parser.add_argument("--top", type=int, default=30, help="상위 N개 (기본: 30)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    report = scan_crash_bounce(top_n=args.top)
    save_report(report)
    print_report(report)


if __name__ == "__main__":
    main()
