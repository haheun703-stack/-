"""
바이오/CDMO 급락 감시 스캐너

목적: 관세 수혜 바이오/CDMO 종목 중 급락+수급 반전 포인트를 포착.
"현대바이오 같은 한번씩 날아가는" 종목을 미리 골라 대기.

핵심 로직:
  1. 바이오/CDMO 30종목 유니버스 정의
  2. RSI < 35 + BB 하단 접근 = 급락 구간 탐지
  3. 외인/기관 수급 반전 (순매도→순매수) 신호
  4. PB15_BB 패턴 (우리 STRONG_ALPHA PF=2.64) 일치 확인
  5. 출력: data/bio_cdmo_watch.json → NXT 전략 연동

사용: python -u -X utf8 scripts/scan_bio_cdmo.py [--date 2026-04-07]
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_PATH = DATA_DIR / "bio_cdmo_watch.json"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 바이오/CDMO 유니버스 — 관세 수혜 + 폭발 잠재력
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BIO_UNIVERSE = {
    # CDMO/CMO (미국 제약 관세 100% → 한국 위탁생산 수혜)
    "207940": {"name": "삼성바이오로직스", "theme": "CDMO", "tier": "대형"},
    "302440": {"name": "SK바이오사이언스", "theme": "CDMO", "tier": "중형"},
    "328130": {"name": "루닛", "theme": "AI의료", "tier": "중형"},

    # 바이오시밀러/혁신신약 (글로벌 공급망 재편 수혜)
    "068270": {"name": "셀트리온", "theme": "바이오시밀러", "tier": "대형"},
    "009420": {"name": "한올바이오파마", "theme": "혁신신약", "tier": "중형"},
    "196170": {"name": "알테오젠", "theme": "SC플랫폼", "tier": "중형"},
    "141080": {"name": "리가켐바이오", "theme": "ADC", "tier": "중형"},
    "298380": {"name": "에이비엘바이오", "theme": "ADC", "tier": "중형"},
    "145020": {"name": "휴젤", "theme": "보톡스", "tier": "중형"},

    # 폭발 잠재력 (변동성 높은 테마주)
    "048410": {"name": "현대바이오", "theme": "치료제", "tier": "소형"},
    "137950": {"name": "제이씨케미칼", "theme": "바이오디젤", "tier": "소형"},
    "950160": {"name": "코오롱티슈진", "theme": "유전자치료", "tier": "소형"},
    "226950": {"name": "올릭스", "theme": "RNAi", "tier": "소형"},
    "323990": {"name": "박셀바이오", "theme": "세포치료", "tier": "소형"},
    "263750": {"name": "펄어비스", "theme": "게임", "tier": "중형"},  # 바이오 아님, 변동성 비교군

    # 진단/장비
    "950130": {"name": "엑세스바이오", "theme": "진단", "tier": "소형"},
    "041510": {"name": "에스엠", "theme": "에스엠", "tier": "중형"},  # 비교군

    # 제약 대형 (안정적 + 수급 관찰)
    "000100": {"name": "유한양행", "theme": "제약대형", "tier": "대형"},
    "185750": {"name": "종근당", "theme": "제약대형", "tier": "대형"},
    "128940": {"name": "한미약품", "theme": "제약대형", "tier": "대형"},
    "006280": {"name": "녹십자", "theme": "혈액제제", "tier": "대형"},

    # CMO/CDMO 밸류체인
    "237690": {"name": "에스티팜", "theme": "올리고API", "tier": "중형"},
    "950170": {"name": "JW생명과학", "theme": "수액CMO", "tier": "중형"},
    "253840": {"name": "수젠텍", "theme": "진단CMO", "tier": "소형"},

    # 최근 급등 이력 (한번씩 날아가는 유형)
    "066570": {"name": "LG전자", "theme": "비교군", "tier": "대형"},  # 비교군
    "338840": {"name": "와이바이오로직스", "theme": "방사선치료", "tier": "소형"},
    "365340": {"name": "성일하이텍", "theme": "2차전지리사이클", "tier": "중형"},
    "058470": {"name": "리노공업", "theme": "반도체소재", "tier": "중형"},
}


def _load_ohlcv(ticker: str, days: int = 60) -> pd.DataFrame | None:
    """data/raw/{ticker}.parquet에서 OHLCV 로드."""
    pq_path = RAW_DIR / f"{ticker}.parquet"
    if not pq_path.exists():
        return None
    try:
        df = pd.read_parquet(pq_path)
        df.index = pd.to_datetime(df.index)
        return df.tail(days)
    except Exception:
        return None


def _calc_rsi(series: pd.Series, period: int = 14) -> float:
    """RSI 계산."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else 50.0


def _calc_bb_pct(close: pd.Series, period: int = 20) -> float:
    """볼린저밴드 %B (0=하단, 100=상단)."""
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    last = close.iloc[-1]
    band_width = upper.iloc[-1] - lower.iloc[-1]
    if band_width <= 0:
        return 50.0
    return float((last - lower.iloc[-1]) / band_width * 100)


def _calc_change_from_high(close: pd.Series, lookback: int = 20) -> float:
    """최근 N일 고점 대비 하락률."""
    high = close.rolling(lookback).max()
    if high.iloc[-1] <= 0:
        return 0.0
    return float((close.iloc[-1] / high.iloc[-1] - 1) * 100)


def _load_supply_data(ticker: str) -> dict:
    """수급 데이터 로드 (institutional_flow 등)."""
    result = {"foreign_5d": 0, "inst_5d": 0, "foreign_today": 0, "inst_today": 0,
              "foreign_consec": 0, "inst_consec": 0, "dual_buy": False}

    # accumulation_tracker에서 조회
    acc_path = DATA_DIR / "institutional_flow" / "accumulation_alert.json"
    if acc_path.exists():
        try:
            acc = json.loads(acc_path.read_text(encoding="utf-8"))
            for item in acc.get("alerts", []):
                if item.get("ticker") == ticker:
                    result["foreign_5d"] = item.get("foreign_5d_억", 0)
                    result["inst_5d"] = item.get("inst_5d_억", 0)
                    result["foreign_consec"] = item.get("foreign_consecutive", 0)
                    result["inst_consec"] = item.get("inst_consecutive", 0)
                    break
        except Exception:
            pass

    # dual_buying_watch에서 조회
    dual_path = DATA_DIR / "dual_buying_watch.json"
    if dual_path.exists():
        try:
            dual = json.loads(dual_path.read_text(encoding="utf-8"))
            for item in dual.get("items", dual.get("stocks", [])):
                if item.get("ticker") == ticker:
                    result["dual_buy"] = True
                    result["foreign_today"] = item.get("foreign_net", 0)
                    result["inst_today"] = item.get("inst_net", 0)
                    break
        except Exception:
            pass

    return result


def scan_bio_cdmo(target_date: str | None = None) -> dict:
    """바이오/CDMO 급락 감시 스캔."""
    today = target_date or date.today().strftime("%Y-%m-%d")
    logger.info("바이오/CDMO 급락 감시 시작 (%s, %d종목)", today, len(BIO_UNIVERSE))

    alerts = []

    for ticker, meta in BIO_UNIVERSE.items():
        df = _load_ohlcv(ticker, days=60)
        if df is None or len(df) < 20:
            continue

        close = df["close"] if "close" in df.columns else df.get("Close", pd.Series())
        if close.empty:
            continue

        price = float(close.iloc[-1])
        rsi = _calc_rsi(close)
        bb_pct = _calc_bb_pct(close)
        chg_from_high_20d = _calc_change_from_high(close, 20)
        chg_from_high_60d = _calc_change_from_high(close, 60)

        # 20일 이동평균 대비
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma20_gap = (price / ma20 - 1) * 100 if ma20 > 0 else 0

        # 수급 데이터
        supply = _load_supply_data(ticker)

        # ━━━ 스코어링 ━━━
        score = 0
        signals = []

        # (1) RSI 급락: RSI < 30 = 최고, < 35 = 양호
        if rsi < 25:
            score += 30
            signals.append(f"RSI 극과매도({rsi:.0f})")
        elif rsi < 30:
            score += 25
            signals.append(f"RSI 과매도({rsi:.0f})")
        elif rsi < 35:
            score += 15
            signals.append(f"RSI 약세({rsi:.0f})")

        # (2) BB 하단 접근/이탈
        if bb_pct < 5:
            score += 25
            signals.append(f"BB 하단이탈({bb_pct:.0f}%)")
        elif bb_pct < 20:
            score += 15
            signals.append(f"BB 하단접근({bb_pct:.0f}%)")

        # (3) 20일 고점 대비 급락 (-15% 이상)
        if chg_from_high_20d <= -20:
            score += 20
            signals.append(f"20일고점 -{abs(chg_from_high_20d):.0f}%")
        elif chg_from_high_20d <= -15:
            score += 15
            signals.append(f"20일고점 -{abs(chg_from_high_20d):.0f}%")
        elif chg_from_high_20d <= -10:
            score += 8
            signals.append(f"20일고점 -{abs(chg_from_high_20d):.0f}%")

        # (4) 수급 반전 (핵심 — PB15_BB 패턴)
        if supply["dual_buy"]:
            score += 30
            signals.append("쌍끌이유입!")
        elif supply["foreign_5d"] > 0 and supply["inst_5d"] > 0:
            score += 25
            signals.append(f"외인+기관 5일순유입")
        elif supply["foreign_consec"] >= 3 or supply["inst_consec"] >= 3:
            score += 15
            signals.append(f"연속유입(외{supply['foreign_consec']}일/기{supply['inst_consec']}일)")
        elif supply["foreign_5d"] > 0 or supply["inst_5d"] > 0:
            score += 8
            signals.append("수급 전환 징후")

        # (5) MA20 이하 (눌림 확인)
        if ma20_gap <= -15:
            score += 10
            signals.append(f"MA20 -{abs(ma20_gap):.0f}% 이탈")
        elif ma20_gap <= -5:
            score += 5

        # ━━━ 등급 판정 ━━━
        if score >= 70:
            grade = "강력 포착"
        elif score >= 50:
            grade = "포착"
        elif score >= 30:
            grade = "관심"
        else:
            grade = "관찰"

        # 관심 이상만 출력
        if score >= 25:
            alerts.append({
                "ticker": ticker,
                "name": meta["name"],
                "theme": meta["theme"],
                "tier": meta["tier"],
                "price": int(price),
                "rsi": round(rsi, 1),
                "bb_pct": round(bb_pct, 1),
                "chg_20d_high": round(chg_from_high_20d, 1),
                "chg_60d_high": round(chg_from_high_60d, 1),
                "ma20_gap": round(ma20_gap, 1),
                "supply": supply,
                "score": score,
                "grade": grade,
                "signals": signals,
            })

    # 점수순 정렬
    alerts.sort(key=lambda x: -x["score"])

    result = {
        "date": today,
        "generated_at": datetime.now().isoformat(),
        "total_scanned": len(BIO_UNIVERSE),
        "alerts_count": len(alerts),
        "alerts": alerts,
        "summary": {
            "강력 포착": sum(1 for a in alerts if a["grade"] == "강력 포착"),
            "포착": sum(1 for a in alerts if a["grade"] == "포착"),
            "관심": sum(1 for a in alerts if a["grade"] == "관심"),
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("바이오/CDMO 감시 완료: %d건 (강력포착 %d, 포착 %d, 관심 %d)",
                len(alerts),
                result["summary"]["강력 포착"],
                result["summary"]["포착"],
                result["summary"]["관심"])

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="바이오/CDMO 급락 감시 스캐너")
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    result = scan_bio_cdmo(target_date=args.date)

    alerts = result["alerts"]
    print(f"\n=== 바이오/CDMO 급락 감시 ({result['date']}) ===")
    print(f"  스캔: {result['total_scanned']}종목 → 감시: {len(alerts)}건")
    print(f"  강력포착: {result['summary']['강력 포착']} | 포착: {result['summary']['포착']} | 관심: {result['summary']['관심']}")

    for i, a in enumerate(alerts[:15], 1):
        supply_tag = ""
        if a["supply"]["dual_buy"]:
            supply_tag = " [쌍끌이]"
        elif a["supply"]["foreign_5d"] > 0:
            supply_tag = f" [외인+{a['supply']['foreign_5d']:.0f}억]"

        print(f"  {i:2d}. [{a['grade']:4s}] {a['name']:12s} ({a['theme']:8s}) "
              f"{a['price']:>8,}원 RSI={a['rsi']:4.1f} BB={a['bb_pct']:4.1f}% "
              f"고점-{abs(a['chg_20d_high']):.0f}% 점수={a['score']}{supply_tag}")
        if a["signals"]:
            print(f"         → {' | '.join(a['signals'])}")


if __name__ == "__main__":
    main()
