"""공매도/대차잔고 시그널 수집기

종목별 공매도 잔고 + 시장 전체 공매도 압력을 분석하여
수급 팩터(SD) 보정과 LENS 밸류트랩 판별에 활용한다.

데이터 소스:
  1차: pykrx 공매도 API (KRX → short_balance, short_volume)
  2차: raw parquet 기존 데이터 (2019~2026-02-11)
  3차: 인버스 ETF 프록시 (derivatives_collector 연동)

시그널:
  SH1: 공매도 잔고율 급증 (종목별) → 하방 압력
  SH2: 공매도 잔고 급감 (종목별) → 숏커버 반등 후보
  SH3: 잔고율 역대 극단 (종목별) → 숏커버 기회 or 밸류트랩
  SH4: 시장 전체 공매도 비중 이상 (매크로)

출력: data/short_selling/daily_short.json
사용: python -u -X utf8 scripts/collect_short_selling.py
BAT: schedule_D_after_close.bat (장마감 후 실행)
"""

from __future__ import annotations

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "short_selling"
SIGNAL_PATH = DATA_DIR / "daily_short.json"
HISTORY_PATH = DATA_DIR / "short_history.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DERIVATIVES_PATH = PROJECT_ROOT / "data" / "derivatives" / "derivatives_signal.json"

# 유니버스: processed parquet 기준 (시가총액 상위)
UNIVERSE_FILE = PROJECT_ROOT / "config" / "universe.txt"

# settings
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# 공매도 금지 기간
SHORT_BAN_PERIODS = [
    ("2020-03-16", "2021-05-02"),
    ("2023-11-06", "2025-03-30"),
]


def _is_short_banned(date) -> bool:
    """해당 날짜가 공매도 금지 기간인지 확인."""
    d = pd.Timestamp(date)
    for start, end in SHORT_BAN_PERIODS:
        if pd.Timestamp(start) <= d <= pd.Timestamp(end):
            return True
    return False


def _load_settings() -> dict:
    """settings.yaml에서 short_selling 설정 로드."""
    try:
        import yaml
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("short_selling", {})
    except Exception:
        return {}


def _get_universe() -> list[str]:
    """분석 대상 종목 목록 (processed parquet 기준)."""
    tickers = []
    for f in sorted(PROCESSED_DIR.glob("*.parquet")):
        tickers.append(f.stem)
    return tickers


def _try_pykrx_update(tickers: list[str], date_str: str) -> dict[str, dict]:
    """pykrx로 최신 공매도 데이터 수집 시도.

    Returns:
        {ticker: {"short_balance": int, "short_volume": int}} or empty if failed
    """
    try:
        from pykrx import stock as krx
    except ImportError:
        return {}

    result = {}

    # 1. 전체 종목 공매도 잔고 (by_ticker)
    try:
        df = krx.get_shorting_balance_by_ticker(date_str, "KOSPI")
        if not df.empty and len(df) > 10:
            for ticker in tickers:
                if ticker in df.index:
                    row = df.loc[ticker]
                    result[ticker] = {
                        "short_balance": int(row.get("공매도잔고", 0)),
                    }
            logger.info(f"pykrx KOSPI 공매도 잔고: {len(result)}종목")
    except Exception:
        pass

    if not result:
        # KOSDAQ도 시도
        try:
            df = krx.get_shorting_balance_by_ticker(date_str, "KOSDAQ")
            if not df.empty:
                for ticker in tickers:
                    if ticker in df.index:
                        row = df.loc[ticker]
                        result[ticker] = {
                            "short_balance": int(row.get("공매도잔고", 0)),
                        }
        except Exception:
            pass

    return result


def _load_from_parquet(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """raw parquet에서 공매도 이력 로드."""
    data = {}
    for ticker in tickers:
        f = RAW_DIR / f"{ticker}.parquet"
        if not f.exists():
            continue
        try:
            df = pd.read_parquet(f)
            if "short_balance" in df.columns:
                data[ticker] = df
        except Exception:
            continue
    return data


def _compute_stock_signals(
    ticker: str,
    df: pd.DataFrame,
    cfg: dict,
) -> dict | None:
    """종목별 공매도 시그널 계산.

    Returns:
        시그널 dict or None (데이터 부족 시)
    """
    if "short_balance" not in df.columns:
        return None

    sb = df["short_balance"].copy()
    sb = sb.fillna(0)

    # 최근 유효 데이터 찾기
    valid = sb[sb > 0]
    if len(valid) < 20:
        return None

    latest_balance = float(valid.iloc[-1])
    latest_date = valid.index[-1]

    # 5일 전 / 20일 전 잔고
    idx = valid.index.get_loc(latest_date)
    balance_5d = float(valid.iloc[max(0, idx - 5)]) if idx >= 5 else latest_balance
    balance_20d = float(valid.iloc[max(0, idx - 20)]) if idx >= 20 else latest_balance

    # 변화율
    change_5d = (latest_balance / balance_5d - 1) if balance_5d > 0 else 0
    change_20d = (latest_balance / balance_20d - 1) if balance_20d > 0 else 0

    # 잔고율 (공매도 잔고 / 거래량 20일 평균)
    vol_20d = df["volume"].tail(20).mean()
    balance_ratio = latest_balance / vol_20d if vol_20d > 0 else 0

    # 역대 백분위
    percentile = float((valid < latest_balance).mean())

    # ─── SH1: 잔고율 급증 ───
    sh1_threshold = cfg.get("sh1_threshold", 0.50)
    sh1_triggered = change_5d >= sh1_threshold and latest_balance > 10000

    # ─── SH2: 잔고율 급감 (숏커버) ───
    sh2_triggered = change_5d <= -0.30 and latest_balance > 10000

    # ─── SH3: 잔고율 역대 극단 ───
    sh3_percentile = cfg.get("sh3_percentile", 0.90)
    sh3_triggered = percentile >= sh3_percentile

    # 종합 short_risk 점수 (-1 ~ +1)
    # 양수 = 하방 압력, 음수 = 숏커버 반등 가능
    risk = 0.0
    if sh1_triggered:
        risk += 0.6
    if sh3_triggered:
        risk += 0.3  # 극단이지만 방향은 불확실
    if sh2_triggered:
        risk -= 0.5  # 숏커버 → 반등 기대

    # 5일/20일 추세 반영
    if change_5d > 0.2:
        risk += 0.2
    elif change_5d < -0.2:
        risk -= 0.2

    risk = max(-1.0, min(1.0, risk))

    return {
        "ticker": ticker,
        "latest_balance": int(latest_balance),
        "latest_date": str(latest_date.date()),
        "balance_5d_ago": int(balance_5d),
        "change_5d_pct": round(change_5d * 100, 1),
        "change_20d_pct": round(change_20d * 100, 1),
        "balance_ratio": round(balance_ratio, 2),
        "percentile": round(percentile * 100, 1),
        "sh1_surge": sh1_triggered,
        "sh2_short_cover": sh2_triggered,
        "sh3_extreme": sh3_triggered,
        "short_risk": round(risk, 2),
    }


def _compute_market_signal(stock_signals: list[dict], cfg: dict) -> dict:
    """시장 전체 공매도 압력 (SH4).

    - 전체 종목 공매도 잔고율 평균
    - 급증 종목 비율
    - 파생 시그널 연동
    """
    if not stock_signals:
        return {
            "available": False,
            "sh4_triggered": False,
        }

    # 잔고율 급증 종목 비율
    surge_count = sum(1 for s in stock_signals if s["sh1_surge"])
    extreme_count = sum(1 for s in stock_signals if s["sh3_extreme"])
    cover_count = sum(1 for s in stock_signals if s["sh2_short_cover"])
    total = len(stock_signals)

    surge_ratio = surge_count / total if total > 0 else 0
    avg_risk = np.mean([s["short_risk"] for s in stock_signals])

    # SH4: 시장 전체 공매도 이상
    sh4_ratio = cfg.get("sh4_market_ratio", 1.5)
    sh4_triggered = surge_ratio > 0.15  # 15% 이상 종목에서 급증

    # 파생 시그널 연동
    deriv_score = 0
    try:
        with open(DERIVATIVES_PATH, "r", encoding="utf-8") as f:
            deriv = json.load(f)
        deriv_score = deriv.get("composite", {}).get("score", 0)
    except Exception:
        pass

    # 종합 매크로 공매도 점수 (-10 ~ +10)
    # 양수 = 하방 압력, 음수 = 반등 기대
    macro_penalty = 0
    if sh4_triggered:
        macro_penalty = -cfg.get("regime_penalty_sh4", 5)  # -5 = 하방 보정
    if cover_count > surge_count * 2:
        macro_penalty += 3  # 숏커버 > 급증 → 반등 분위기

    return {
        "available": True,
        "total_stocks": total,
        "surge_count": surge_count,
        "extreme_count": extreme_count,
        "cover_count": cover_count,
        "surge_ratio_pct": round(surge_ratio * 100, 1),
        "avg_risk": round(avg_risk, 2),
        "sh4_triggered": sh4_triggered,
        "macro_score_adj": macro_penalty,
        "derivatives_score": deriv_score,
    }


def _generate_alerts(
    stock_signals: list[dict],
    market_signal: dict,
    portfolio_tickers: set[str] | None = None,
) -> list[dict]:
    """텔레그램 알림 대상 생성."""
    alerts = []

    # 보유 종목 공매도 급증 경고
    if portfolio_tickers:
        for s in stock_signals:
            if s["ticker"] in portfolio_tickers and s["sh1_surge"]:
                alerts.append({
                    "type": "SHORT_SURGE_HOLDING",
                    "ticker": s["ticker"],
                    "message": (
                        f"공매도 잔고 급증 ({s['change_5d_pct']:+.0f}%, 5일)\n"
                        f"잔고: {s['latest_balance']:,}주\n"
                        f"역대 백분위: {s['percentile']:.0f}%"
                    ),
                })

    # 숏커버 반등 후보 TOP 5
    cover_candidates = [s for s in stock_signals if s["sh2_short_cover"]]
    cover_candidates.sort(key=lambda x: x["change_5d_pct"])
    for s in cover_candidates[:5]:
        alerts.append({
            "type": "SHORT_COVER_CANDIDATE",
            "ticker": s["ticker"],
            "message": (
                f"숏커버 반등 후보 (잔고 {s['change_5d_pct']:+.0f}%, 5일)\n"
                f"역대 백분위: {s['percentile']:.0f}%"
            ),
        })

    # SH4 시장 전체 경고
    if market_signal.get("sh4_triggered"):
        alerts.append({
            "type": "MARKET_SHORT_SURGE",
            "ticker": "MARKET",
            "message": (
                f"시장 전체 공매도 급증 경고\n"
                f"급증 종목: {market_signal['surge_count']}개 "
                f"({market_signal['surge_ratio_pct']:.0f}%)"
            ),
        })

    return alerts


def collect_short_selling_signal() -> dict:
    """공매도 시그널 수집 + 종합 신호 생성."""
    logger.info("=== 공매도/대차잔고 시그널 수집 시작 ===")

    cfg = _load_settings().get("signals", {})
    today = datetime.now()
    today_str = today.strftime("%Y%m%d")

    # 공매도 금지 기간 체크
    if _is_short_banned(today):
        logger.warning("현재 공매도 금지 기간 — 데이터 없음")
        signal = {
            "date": today.strftime("%Y-%m-%d"),
            "generated_at": today.strftime("%Y-%m-%d %H:%M"),
            "short_banned": True,
            "stock_signals": [],
            "market_signal": {"available": False},
            "alerts": [],
        }
        _save_signal(signal)
        return signal

    # 유니버스 로드
    tickers = _get_universe()
    logger.info(f"유니버스: {len(tickers)}종목")

    # 1. pykrx로 최신 데이터 수집 시도
    pykrx_data = _try_pykrx_update(tickers, today_str)
    if pykrx_data:
        logger.info(f"pykrx 최신 데이터: {len(pykrx_data)}종목")
    else:
        logger.warning("pykrx 공매도 데이터 없음 — parquet 기존 데이터 활용")

    # 2. raw parquet에서 공매도 이력 로드
    parquet_data = _load_from_parquet(tickers)
    logger.info(f"parquet 공매도 이력: {len(parquet_data)}종목")

    # 3. 종목별 시그널 계산
    stock_signals = []
    for ticker in tickers:
        df = parquet_data.get(ticker)
        if df is None:
            continue

        sig = _compute_stock_signals(ticker, df, cfg)
        if sig:
            stock_signals.append(sig)

    logger.info(f"종목별 시그널: {len(stock_signals)}종목 분석 완료")

    # 4. 시장 전체 시그널
    scoring_cfg = _load_settings().get("scoring", {})
    market_signal = _compute_market_signal(stock_signals, scoring_cfg)

    # 5. 알림 생성
    portfolio_tickers = _load_portfolio_tickers()
    alerts = _generate_alerts(stock_signals, market_signal, portfolio_tickers)

    # 6. TOP 종목 정렬
    # 공매도 급증 TOP (하방 위험)
    surge_top = sorted(
        [s for s in stock_signals if s["sh1_surge"]],
        key=lambda x: x["change_5d_pct"],
        reverse=True,
    )[:10]

    # 숏커버 후보 TOP (반등 기대)
    cover_top = sorted(
        [s for s in stock_signals if s["sh2_short_cover"]],
        key=lambda x: x["change_5d_pct"],
    )[:10]

    # 극단 잔고 TOP
    extreme_top = sorted(
        [s for s in stock_signals if s["sh3_extreme"]],
        key=lambda x: x["percentile"],
        reverse=True,
    )[:10]

    # 7. 시그널 구성
    signal = {
        "date": today.strftime("%Y-%m-%d"),
        "generated_at": today.strftime("%Y-%m-%d %H:%M"),
        "short_banned": False,
        "data_source": "pykrx" if pykrx_data else "parquet_history",
        "total_analyzed": len(stock_signals),
        "surge_top": surge_top,
        "cover_top": cover_top,
        "extreme_top": extreme_top,
        "market_signal": market_signal,
        "alerts": alerts,
        "summary": {
            "surge_count": market_signal.get("surge_count", 0),
            "cover_count": market_signal.get("cover_count", 0),
            "extreme_count": market_signal.get("extreme_count", 0),
            "avg_risk": market_signal.get("avg_risk", 0),
            "macro_adj": market_signal.get("macro_score_adj", 0),
        },
    }

    # 8. 저장
    _save_signal(signal)

    # 9. 히스토리 추가
    _save_history(signal)

    # 10. 로그
    ms = market_signal
    logger.info(
        f"공매도 시그널: 급증 {ms.get('surge_count', 0)}개 | "
        f"숏커버 {ms.get('cover_count', 0)}개 | "
        f"극단 {ms.get('extreme_count', 0)}개 | "
        f"리스크 {ms.get('avg_risk', 0):+.2f}"
    )

    if alerts:
        logger.info(f"알림: {len(alerts)}건")
        for a in alerts[:3]:
            logger.info(f"  [{a['type']}] {a['ticker']}")

    return signal


def _load_portfolio_tickers() -> set[str]:
    """현재 보유 종목 로드."""
    try:
        path = PROJECT_ROOT / "data" / "equity_tracker.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {p["ticker"] for p in data.get("positions", [])}
    except Exception:
        return set()


def _save_signal(signal: dict) -> None:
    """시그널 JSON 저장."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_PATH.write_text(
        json.dumps(signal, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"저장: {SIGNAL_PATH}")


def _save_history(signal: dict) -> None:
    """CSV 히스토리에 추가."""
    row = {
        "date": signal["date"],
        "surge_count": signal["summary"]["surge_count"],
        "cover_count": signal["summary"]["cover_count"],
        "extreme_count": signal["summary"]["extreme_count"],
        "avg_risk": signal["summary"]["avg_risk"],
        "macro_adj": signal["summary"]["macro_adj"],
        "data_source": signal.get("data_source", ""),
    }

    if HISTORY_PATH.exists():
        df = pd.read_csv(HISTORY_PATH)
        if row["date"] not in df["date"].values:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(HISTORY_PATH, index=False)


def main():
    signal = collect_short_selling_signal()

    print(f"\n=== 공매도 시그널 ===")

    ms = signal["market_signal"]
    if ms.get("available"):
        print(f"분석 종목: {signal['total_analyzed']}개")
        print(f"  급증(SH1): {ms['surge_count']}개 ({ms['surge_ratio_pct']:.0f}%)")
        print(f"  숏커버(SH2): {ms['cover_count']}개")
        print(f"  극단(SH3): {ms['extreme_count']}개")
        print(f"  평균 리스크: {ms['avg_risk']:+.2f}")
        print(f"  매크로 보정: {ms['macro_score_adj']:+d}점")

    if signal.get("surge_top"):
        print(f"\n🔴 공매도 급증 TOP:")
        for s in signal["surge_top"][:5]:
            print(
                f"  {s['ticker']}: "
                f"잔고 {s['latest_balance']:>12,}주 "
                f"({s['change_5d_pct']:+.0f}% 5일) "
                f"백분위 {s['percentile']:.0f}%"
            )

    if signal.get("cover_top"):
        print(f"\n🟢 숏커버 반등 후보:")
        for s in signal["cover_top"][:5]:
            print(
                f"  {s['ticker']}: "
                f"잔고 {s['latest_balance']:>12,}주 "
                f"({s['change_5d_pct']:+.0f}% 5일) "
                f"백분위 {s['percentile']:.0f}%"
            )

    if signal.get("alerts"):
        print(f"\n📢 알림 {len(signal['alerts'])}건:")
        for a in signal["alerts"][:5]:
            print(f"  [{a['type']}] {a['ticker']}: {a['message'][:50]}")


if __name__ == "__main__":
    main()
