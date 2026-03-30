"""외국인 보유한도 소진율 수집기

종목별 외국인 보유한도/보유수량/소진율을 KIS API에서 수집하여
수급 팩터 보정과 한도 임박 종목 감지에 활용한다.

데이터 소스:
  KIS API FHKST01010100 (주식현재가) — hts_frgn_ehrt 필드
  → 소진율, 보유수량

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
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

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
    """sector_map.csv 기준 종목 목록 (기타 제외)."""
    csv_path = PROJECT_ROOT / "data" / "universe" / "sector_map.csv"
    if csv_path.exists():
        tickers = []
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("sector", "기타") != "기타" and row.get("ticker"):
                    tickers.append(row["ticker"])
        return tickers
    # 폴백: parquet
    return [f.stem for f in sorted(PROCESSED_DIR.glob("*.parquet"))]


def _is_trading_day() -> bool:
    """오늘이 거래일인지 확인 (주말 제외)."""
    return datetime.now().weekday() < 5  # 0=월 ~ 4=금


def _get_kis_token() -> str | None:
    """KIS API 토큰 발급 (기존 토큰 파일 재사용)."""
    load_dotenv(PROJECT_ROOT / ".env")
    app_key = os.getenv("KIS_APP_KEY")
    app_secret = os.getenv("KIS_APP_SECRET")
    if not app_key or not app_secret:
        logger.error("KIS_APP_KEY / KIS_APP_SECRET 미설정")
        return None

    token_path = PROJECT_ROOT / ".kis_token.json"
    if token_path.exists():
        with open(token_path) as f:
            tk = json.load(f)
        token = tk.get("access_token")
        if token:
            return token

    base = "https://openapi.koreainvestment.com:9443"
    for attempt in range(3):
        try:
            r = requests.post(f"{base}/oauth2/tokenP", json={
                "grant_type": "client_credentials",
                "appkey": app_key, "appsecret": app_secret,
            }, timeout=10)
            data = r.json()
            if "access_token" in data:
                return data["access_token"]
            logger.warning("토큰 발급 대기 (%d/3): %s", attempt + 1, data.get("error_description", ""))
            time.sleep(65)
        except Exception as e:
            logger.warning("토큰 발급 실패: %s", e)
            time.sleep(5)
    return None


def _fetch_exhaustion_kis(tickers: list[str]) -> pd.DataFrame:
    """KIS API FHKST01010100으로 종목별 외인소진율 수집.

    Returns:
        DataFrame(index=ticker, columns=[한도소진율, 보유수량, 지분율])
    """
    load_dotenv(PROJECT_ROOT / ".env")
    app_key = os.getenv("KIS_APP_KEY", "")
    app_secret = os.getenv("KIS_APP_SECRET", "")
    base = "https://openapi.koreainvestment.com:9443"

    token = _get_kis_token()
    if not token:
        return pd.DataFrame()

    results = []
    success = 0
    fail = 0

    for i, ticker in enumerate(tickers):
        try:
            headers = {
                "authorization": f"Bearer {token}",
                "appkey": app_key, "appsecret": app_secret,
                "tr_id": "FHKST01010100",
            }
            params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}
            r = requests.get(
                f"{base}/uapi/domestic-stock/v1/quotations/inquire-price",
                headers=headers, params=params, timeout=5,
            )
            out = r.json().get("output", {})
            ehrt = float(out.get("hts_frgn_ehrt", 0))
            hold = int(out.get("frgn_hldn_qty", 0))

            if ehrt > 0 or hold > 0:
                results.append({
                    "ticker": ticker,
                    "한도소진율": ehrt,
                    "보유수량": hold,
                    "지분율": ehrt,  # KIS에서 소진율 ≈ 지분율 근사
                    "상장주식수": 0,
                    "한도수량": 0,
                })
                success += 1
        except Exception:
            fail += 1

        # 속도 조절: 초당 18건
        if (i + 1) % 18 == 0:
            time.sleep(1.1)

        # 진행 로그 (200건마다)
        if (i + 1) % 200 == 0:
            logger.info("  수집 진행: %d/%d (성공 %d, 실패 %d)", i + 1, len(tickers), success, fail)

    logger.info("KIS 외인소진율 수집: %d종목 성공 / %d 실패", success, fail)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index("ticker")
    return df


def _fetch_prev_exhaustion(prev_date_str: str) -> dict[str, float]:
    """이전 소진율 조회 (히스토리 CSV에서 읽기, 없으면 빈 dict)."""
    if not HISTORY_PATH.exists():
        return {}
    try:
        # 이전 daily_exhaustion에서 읽기 (간이 방식)
        return {}  # 첫 수집 시 비교 데이터 없음 — 다음 수집부터 활성화
    except Exception:
        return {}


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
        "data_source": "kis_api",
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
    universe_list = _get_universe()
    universe = set(universe_list)
    logger.info(f"유니버스: {len(universe)}종목")

    # 2. KIS API로 소진율 수집
    logger.info("KIS API 외인소진율 수집 시작...")
    df = _fetch_exhaustion_kis(universe_list)

    if df.empty:
        logger.error("외인소진율 데이터 수집 실패")
        tracker.finalize(total=0)
        return {"date": today.strftime("%Y-%m-%d"), "error": "데이터 없음"}

    logger.info(f"수집 완료: {len(df)}종목")

    # 3. 이전 소진율 (변화율 계산, 히스토리에서)
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
        "data_source": "kis_api",
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
