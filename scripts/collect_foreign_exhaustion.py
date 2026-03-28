"""외국인 보유한도 소진율 수집기

종목별 외국인 보유한도/보유수량/소진율을 KRX에서 수집하여
수급 팩터 보정과 한도 임박 종목 감지에 활용한다.

데이터 소스:
  pykrx get_exhaustion_rates_of_foreign_investment()
  → 상장주식수, 보유수량, 지분율, 한도수량, 한도소진율

시그널:
  FE1: 소진율 90% 이상 (한도 임박 — 추가 매수 제한 우려)
  FE2: 소진율 5일 급등 (+5%p 이상 — 외국인 집중 매수)
  FE3: 소진율 70~89% + 연속매수 (잠재적 한도 근접)

출력: data/foreign_exhaustion/daily_exhaustion.json
히스토리: data/foreign_exhaustion/exhaustion_history.csv
사용: python -u -X utf8 scripts/collect_foreign_exhaustion.py [--send]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "foreign_exhaustion"
SIGNAL_PATH = DATA_DIR / "daily_exhaustion.json"
HISTORY_PATH = DATA_DIR / "exhaustion_history.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 시그널 임계값
THRESHOLD_HIGH = 90       # FE1: 한도 임박
THRESHOLD_RISING = 5.0    # FE2: 5일 급등 %p
THRESHOLD_WATCH = 70      # FE3: 감시 구간 하한


def _get_universe() -> list[str]:
    """processed parquet 기준 종목 목록."""
    return [f.stem for f in sorted(PROCESSED_DIR.glob("*.parquet"))]


def _is_trading_day() -> bool:
    """오늘이 거래일인지 확인 (주말 제외)."""
    return datetime.now().weekday() < 5  # 0=월 ~ 4=금


def _fetch_exhaustion(date_str: str) -> pd.DataFrame:
    """pykrx로 KOSPI + KOSDAQ 외인소진율 조회.

    Returns:
        DataFrame(index=ticker, columns=[상장주식수, 보유수량, 지분율, 한도수량, 한도소진율])
    """
    from pykrx import stock as krx

    frames = []
    for market in ("KOSPI", "KOSDAQ"):
        try:
            df = krx.get_exhaustion_rates_of_foreign_investment(date_str, market)
            if df is not None and not df.empty:
                frames.append(df)
                logger.info(f"  {market}: {len(df)}종목")
            else:
                logger.warning(f"  {market}: 데이터 없음")
        except (ValueError, KeyError) as e:
            logger.warning(f"  {market} 조회 실패 (데이터 없음): {e}")
        except Exception as e:
            # JSON decode error 등 — 주말/비거래일에 KRX가 빈 응답 반환
            err_msg = str(e)
            if "Expecting value" in err_msg or "JSONDecode" in err_msg:
                logger.warning(f"  {market}: KRX 빈 응답 (비거래일 가능)")
            else:
                logger.warning(f"  {market} 조회 실패: {e}")
        time.sleep(1)  # KRX rate limit 방어

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def _fetch_prev_exhaustion(prev_date_str: str) -> dict[str, float]:
    """5일 전 소진율 조회 (변화율 계산용).

    Returns:
        {ticker: exhaustion_rate}
    """
    from pykrx import stock as krx

    result = {}
    for market in ("KOSPI", "KOSDAQ"):
        try:
            df = krx.get_exhaustion_rates_of_foreign_investment(prev_date_str, market)
            if df is not None and not df.empty:
                col = "한도소진율"
                if col in df.columns:
                    for ticker in df.index:
                        val = df.loc[ticker, col]
                        if pd.notna(val):
                            result[ticker] = float(val)
        except Exception:
            pass
        time.sleep(1)

    return result


def _compute_signals(
    df: pd.DataFrame,
    prev_rates: dict[str, float],
    universe: set[str],
) -> list[dict]:
    """종목별 시그널 계산."""
    signals = []

    for ticker in df.index:
        if ticker not in universe:
            continue

        row = df.loc[ticker]
        listed = int(row.get("상장주식수", 0))
        holding = int(row.get("보유수량", 0))
        ratio = float(row.get("지분율", 0))
        limit_qty = int(row.get("한도수량", 0))
        exhaustion = float(row.get("한도소진율", 0))

        if limit_qty <= 0 or listed <= 0:
            continue

        # 5일 변화
        prev_rate = prev_rates.get(ticker)
        change_5d = round(exhaustion - prev_rate, 2) if prev_rate is not None else None

        # 시그널 판정
        signal_type = None
        score = 0

        if exhaustion >= THRESHOLD_HIGH:
            signal_type = "FE1"
            score = min(100, int(50 + (exhaustion - THRESHOLD_HIGH) * 5))
        elif change_5d is not None and change_5d >= THRESHOLD_RISING:
            signal_type = "FE2"
            score = min(100, int(40 + change_5d * 6))
        elif THRESHOLD_WATCH <= exhaustion < THRESHOLD_HIGH and change_5d is not None and change_5d > 1.0:
            signal_type = "FE3"
            score = min(80, int(30 + exhaustion * 0.3 + change_5d * 3))

        entry = {
            "ticker": ticker,
            "listed_shares": listed,
            "foreign_limit": limit_qty,
            "foreign_holding": holding,
            "holding_ratio": round(ratio, 2),
            "exhaustion_rate": round(exhaustion, 2),
            "exhaustion_5d_change": change_5d,
            "signal": signal_type,
            "score": score,
        }
        signals.append(entry)

    return signals


def _save_signal(signal: dict) -> None:
    """JSON 저장."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_PATH.write_text(
        json.dumps(signal, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"저장: {SIGNAL_PATH}")


def _save_history(signal: dict) -> None:
    """CSV 히스토리 추가."""
    summary = signal.get("summary", {})
    row = {
        "date": signal["date"],
        "total_analyzed": signal.get("total_analyzed", 0),
        "high90_count": summary.get("high90_count", 0),
        "high80_count": summary.get("high80_count", 0),
        "avg_exhaustion": summary.get("avg_exhaustion", 0),
        "rising_count": summary.get("rising_count", 0),
        "data_source": "pykrx",
    }

    if HISTORY_PATH.exists():
        df = pd.read_csv(HISTORY_PATH)
        if row["date"] not in df["date"].values:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(HISTORY_PATH, index=False)


def _build_telegram_message(signal: dict) -> str:
    """텔레그램 알림 메시지 생성."""
    lines = [f"외인소진율 리포트 ({signal['date']})"]
    lines.append(f"분석: {signal['total_analyzed']}종목")

    s = signal.get("summary", {})
    lines.append(f"90%+: {s.get('high90_count', 0)}개 | 80%+: {s.get('high80_count', 0)}개")

    # 한도 임박 TOP
    top = signal.get("top_exhaustion", [])
    if top:
        lines.append("\n한도 임박 TOP:")
        for t in top[:5]:
            chg = f" ({t['exhaustion_5d_change']:+.1f}%p)" if t.get("exhaustion_5d_change") is not None else ""
            lines.append(f"  {t['ticker']} {t['exhaustion_rate']:.1f}%{chg}")

    # 급등 TOP
    rising = signal.get("rising_exhaustion", [])
    if rising:
        lines.append("\n소진율 급등:")
        for t in rising[:5]:
            lines.append(f"  {t['ticker']} {t['exhaustion_rate']:.1f}% (+{t['exhaustion_5d_change']:.1f}%p)")

    return "\n".join(lines)


def collect_foreign_exhaustion() -> dict:
    """외인소진율 수집 + 시그널 생성."""
    from src.pipeline_alert import PipelineErrorTracker
    tracker = PipelineErrorTracker("collect_foreign_exhaustion")
    logger.info("=== 외인소진율 수집 시작 ===")

    today = datetime.now()
    today_str = today.strftime("%Y%m%d")
    prev_date = today - timedelta(days=7)  # 5영업일 ~ 7달력일
    prev_str = prev_date.strftime("%Y%m%d")

    # 1. 유니버스
    universe = set(_get_universe())
    logger.info(f"유니버스: {len(universe)}종목")

    # 2. 오늘 소진율 수집
    logger.info("오늘 외인소진율 수집...")
    df = _fetch_exhaustion(today_str)

    if df.empty:
        # 오늘 데이터 없으면 직전 거래일 시도
        for back in range(1, 4):
            alt_date = today - timedelta(days=back)
            alt_str = alt_date.strftime("%Y%m%d")
            logger.info(f"  {alt_str} 재시도...")
            df = _fetch_exhaustion(alt_str)
            if not df.empty:
                today_str = alt_str
                break

    if df.empty:
        logger.error("외인소진율 데이터 수집 실패")
        tracker.finalize(total=0)
        return {"date": today.strftime("%Y-%m-%d"), "error": "데이터 없음"}

    logger.info(f"수집 완료: {len(df)}종목 (유니버스 매칭 전)")

    # 3. 5일 전 소진율 (변화율 계산)
    logger.info("5일 전 소진율 수집...")
    prev_rates = _fetch_prev_exhaustion(prev_str)
    logger.info(f"  비교 대상: {len(prev_rates)}종목")

    # 4. 시그널 계산
    all_signals = _compute_signals(df, prev_rates, universe)
    logger.info(f"시그널 계산: {len(all_signals)}종목")

    # 5. 분류
    fe1 = [s for s in all_signals if s["signal"] == "FE1"]
    fe2 = [s for s in all_signals if s["signal"] == "FE2"]
    fe3 = [s for s in all_signals if s["signal"] == "FE3"]

    high90 = [s for s in all_signals if s["exhaustion_rate"] >= 90]
    high80 = [s for s in all_signals if 80 <= s["exhaustion_rate"] < 90]
    rising = [s for s in all_signals
              if s["exhaustion_5d_change"] is not None and s["exhaustion_5d_change"] >= THRESHOLD_RISING]

    avg_rate = (
        sum(s["exhaustion_rate"] for s in all_signals) / len(all_signals)
        if all_signals else 0
    )

    # TOP 정렬
    top_exhaustion = sorted(all_signals, key=lambda x: x["exhaustion_rate"], reverse=True)[:15]
    rising_exhaustion = sorted(
        [s for s in all_signals if s["exhaustion_5d_change"] is not None],
        key=lambda x: x["exhaustion_5d_change"],
        reverse=True,
    )[:10]

    # 6. 결과 구성
    date_label = datetime.strptime(today_str, "%Y%m%d").strftime("%Y-%m-%d")
    signal = {
        "date": date_label,
        "generated_at": today.strftime("%Y-%m-%d %H:%M"),
        "data_source": "pykrx",
        "total_analyzed": len(all_signals),
        "summary": {
            "high90_count": len(high90),
            "high80_count": len(high80),
            "avg_exhaustion": round(avg_rate, 2),
            "rising_count": len(rising),
            "fe1_count": len(fe1),
            "fe2_count": len(fe2),
            "fe3_count": len(fe3),
        },
        "top_exhaustion": top_exhaustion,
        "rising_exhaustion": rising_exhaustion,
        "fe1_signals": sorted(fe1, key=lambda x: x["exhaustion_rate"], reverse=True),
        "fe2_signals": sorted(fe2, key=lambda x: x["exhaustion_5d_change"] or 0, reverse=True),
        "fe3_signals": sorted(fe3, key=lambda x: x["score"], reverse=True),
    }

    # 7. 저장
    _save_signal(signal)
    _save_history(signal)

    # 8. 에러 집계
    tracker.finalize(total=len(universe))

    # 9. 로그
    logger.info(
        f"외인소진율: 90%+ {len(high90)}개 | 80%+ {len(high80)}개 | "
        f"급등 {len(rising)}개 | 평균 {avg_rate:.1f}%"
    )
    logger.info(f"시그널: FE1={len(fe1)} FE2={len(fe2)} FE3={len(fe3)}")

    return signal


def main():
    parser = argparse.ArgumentParser(description="외인소진율 수집기")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송")
    args = parser.parse_args()

    # 비거래일(주말) 스킵
    if not _is_trading_day():
        logger.info("비거래일(주말) — 스킵")
        print("비거래일(주말) — 스킵")
        return

    signal = collect_foreign_exhaustion()

    if signal.get("error"):
        logger.error(signal["error"])
        sys.exit(1)

    # 콘솔 출력
    print(f"\n=== 외인소진율 리포트 ===")
    print(f"분석: {signal['total_analyzed']}종목")
    s = signal["summary"]
    print(f"  90%+ (한도임박): {s['high90_count']}개")
    print(f"  80%+ (감시구간): {s['high80_count']}개")
    print(f"  급등 (+5%p↑):  {s['rising_count']}개")
    print(f"  평균 소진율:    {s['avg_exhaustion']:.1f}%")

    if signal.get("top_exhaustion"):
        print(f"\n한도 임박 TOP:")
        for t in signal["top_exhaustion"][:7]:
            chg = f" ({t['exhaustion_5d_change']:+.1f}%p)" if t.get("exhaustion_5d_change") is not None else ""
            print(f"  {t['ticker']} {t['exhaustion_rate']:6.1f}%{chg}")

    if signal.get("rising_exhaustion"):
        print(f"\n소진율 급등 TOP:")
        for t in signal["rising_exhaustion"][:5]:
            if t["exhaustion_5d_change"] is not None:
                print(f"  {t['ticker']} {t['exhaustion_rate']:6.1f}% (+{t['exhaustion_5d_change']:.1f}%p)")

    # 텔레그램 발송
    if args.send and (s["high90_count"] > 0 or s["rising_count"] > 0):
        try:
            from src.telegram_sender import send_message
            msg = _build_telegram_message(signal)
            send_message(msg)
            logger.info("텔레그램 발송 완료")
        except Exception as e:
            logger.warning("텔레그램 발송 실패: %s", e)


if __name__ == "__main__":
    main()
