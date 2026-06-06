"""외국인 보유한도 소진율 수집기

종목별 외국인 보유한도/보유수량/소진율을 일별 거래일 기준으로 수집하여
수급 팩터 보정과 한도 임박 종목 감지에 활용한다.

데이터 소스:
  1) 네이버 금융 frgn 일별 페이지 — 날짜/종가/보유주식수/보유율
     → 거래일 행만 내려와 휴장 ghost 원천 차단
  2) KIS API FHKST01010100 (주식현재가) — hts_frgn_ehrt 필드
     → 네이버 실패 시 fallback. 현재가 스냅샷이라 장전/휴장 as-of 보정 불가

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
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

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
NAVER_FRGN_URL = "https://finance.naver.com/item/frgn.naver"
NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
}

# 시그널 임계값
THRESHOLD_HIGH = 90       # FE1: 한도 임박
THRESHOLD_RISING = 5.0    # FE2: 5일 급등 %p
THRESHOLD_WATCH = 70      # FE3: 감시 구간 하한


def _get_universe() -> list[str]:
    """sector_map.csv + sector_fire ETF 기준 종목 목록."""
    tickers: list[str] = []
    csv_path = PROJECT_ROOT / "data" / "universe" / "sector_map.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("sector", "기타") != "기타" and row.get("ticker"):
                    tickers.append(row["ticker"])
    else:
        # 폴백: parquet
        tickers.extend(f.stem for f in sorted(PROCESSED_DIR.glob("*.parquet")))

    tickers.extend(_get_sector_fire_etfs())
    return sorted(set(tickers))


def _get_sector_fire_etfs() -> list[str]:
    """sector_fire_map.yaml의 ETF/레버리지 ETF 코드를 외인소진율 유니버스에 포함."""
    yaml_path = PROJECT_ROOT / "config" / "sector_fire_map.yaml"
    if not yaml_path.exists():
        return []
    try:
        import yaml
    except ImportError:
        return []

    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.debug("sector_fire_map ETF 로드 실패: %s", e)
        return []

    tickers: list[str] = []
    sectors = data.get("sectors") or {}
    for sector in sectors.values():
        for key in ("etf", "leverage_etf"):
            item = (sector or {}).get(key)
            if isinstance(item, dict) and item.get("code"):
                tickers.append(str(item["code"]).zfill(6))
    return tickers


def _is_trading_day(d: date | None = None) -> bool:
    """한국장 거래일인지 확인 (주말 + 공휴일/임시휴장 제외)."""
    from src.trading_calendar import is_kr_trading_day

    return is_kr_trading_day(d or datetime.now().date())


def _safe_int(value: Any, default: int = 0) -> int:
    """쉼표/기호가 섞인 숫자를 int로 변환."""
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    cleaned = (
        text.replace(",", "")
        .replace("%", "")
        .replace("+", "")
        .replace("−", "-")
        .replace(" ", "")
    )
    try:
        return int(float(cleaned))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """쉼표/퍼센트가 섞인 숫자를 float로 변환."""
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    cleaned = (
        text.replace(",", "")
        .replace("%", "")
        .replace("+", "")
        .replace("−", "-")
        .replace(" ", "")
    )
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return default


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """네이버 read_html 결과 컬럼을 단순 문자열로 정규화."""
    out = df.copy()
    cols: list[str] = []
    for col in out.columns:
        if isinstance(col, tuple):
            parts = [str(p).strip() for p in col if str(p).strip() and not str(p).startswith("Unnamed")]
            uniq_parts: list[str] = []
            for part in parts:
                if part not in uniq_parts:
                    uniq_parts.append(part)
            cols.append(" ".join(uniq_parts) if uniq_parts else "")
        else:
            cols.append(str(col).strip())
    out.columns = cols
    return out


def _find_col(columns: list[str], *keywords: str) -> str | None:
    """컬럼명에 모든 키워드가 포함된 첫 컬럼 반환."""
    for col in columns:
        if all(k in col for k in keywords):
            return col
    return None


def _parse_naver_date(value: Any) -> str | None:
    """YYYY.MM.DD → YYYY-MM-DD."""
    text = str(value).strip()
    try:
        return datetime.strptime(text, "%Y.%m.%d").strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _parse_naver_frgn_html(ticker: str, html: str) -> list[dict]:
    """네이버 frgn 페이지 HTML에서 거래일별 외국인 보유율 행을 추출."""
    try:
        tables = pd.read_html(StringIO(html))
    except (ImportError, ValueError) as e:
        logger.debug("네이버 frgn HTML 파싱 실패 %s: %s", ticker, e)
        return []

    parsed: list[dict] = []
    for table in tables:
        df = _normalize_columns(table).dropna(how="all")
        columns = [str(c).strip() for c in df.columns]
        date_col = _find_col(columns, "날짜")
        close_col = _find_col(columns, "종가")
        volume_col = _find_col(columns, "거래량")
        inst_col = _find_col(columns, "기관", "순매매")
        foreign_col = _find_col(columns, "외국인", "순매매")
        holding_col = _find_col(columns, "보유주식수") or _find_col(columns, "보유주수")
        ratio_col = _find_col(columns, "보유율")

        if not all([date_col, close_col, holding_col, ratio_col]):
            continue

        for _, row in df.iterrows():
            date = _parse_naver_date(row.get(date_col))
            if not date:
                continue
            holding = _safe_int(row.get(holding_col))
            ratio = _safe_float(row.get(ratio_col))
            close = _safe_int(row.get(close_col))
            if holding <= 0 and ratio <= 0:
                continue
            parsed.append({
                "ticker": ticker,
                "date": date,
                "close": close,
                "volume": _safe_int(row.get(volume_col)) if volume_col else 0,
                "institution_net": _safe_int(row.get(inst_col)) if inst_col else 0,
                "foreign_net": _safe_int(row.get(foreign_col)) if foreign_col else 0,
                "foreign_holding": holding,
                "holding_ratio": ratio,
            })

    parsed.sort(key=lambda x: x["date"], reverse=True)
    return parsed


def _fetch_naver_frgn_rows(
    ticker: str,
    session: requests.Session,
    pages: int = 1,
) -> list[dict]:
    """네이버 frgn 일별 행 수집."""
    rows: list[dict] = []
    for page in range(1, max(1, pages) + 1):
        try:
            resp = session.get(
                NAVER_FRGN_URL,
                params={"code": ticker, "page": page},
                headers=NAVER_HEADERS,
                timeout=7,
            )
            resp.raise_for_status()
            if not resp.encoding:
                resp.encoding = "euc-kr"
            rows.extend(_parse_naver_frgn_html(ticker, resp.text))
        except Exception as e:
            logger.debug("네이버 frgn 수집 실패 %s page=%s: %s", ticker, page, e)
            break

    # 페이지 중복 방어
    dedup: dict[str, dict] = {}
    for row in rows:
        dedup[row["date"]] = row
    return sorted(dedup.values(), key=lambda x: x["date"], reverse=True)


def _fetch_exhaustion_naver(
    tickers: list[str],
    pages: int = 1,
    session: requests.Session | None = None,
    sleep_sec: float = 0.08,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """네이버 frgn 일별 페이지로 최신 거래일 외인소진율 수집.

    Returns:
        (DataFrame(index=ticker), prev_rates)
    """
    session = session or requests.Session()
    results: list[dict] = []
    prev_rates: dict[str, float] = {}
    success = 0
    fail = 0

    for i, ticker in enumerate(tickers):
        rows = _fetch_naver_frgn_rows(ticker, session=session, pages=pages)
        if rows:
            latest = rows[0]
            if len(rows) >= 6:
                prev_rates[ticker] = float(rows[5]["holding_ratio"])
            results.append({
                "ticker": ticker,
                "기준일": latest["date"],
                "종가": latest["close"],
                "거래량": latest["volume"],
                "기관": latest["institution_net"],
                "외국인": latest["foreign_net"],
                "한도소진율": latest["holding_ratio"],
                "보유수량": latest["foreign_holding"],
                "지분율": latest["holding_ratio"],
                "상장주식수": 0,
                "한도수량": 0,
            })
            success += 1
        else:
            fail += 1

        if sleep_sec > 0:
            time.sleep(sleep_sec)
        if (i + 1) % 200 == 0:
            logger.info("  네이버 수집 진행: %d/%d (성공 %d, 실패 %d)", i + 1, len(tickers), success, fail)

    logger.info("네이버 외인소진율 수집: %d종목 성공 / %d 실패", success, fail)

    if not results:
        return pd.DataFrame(), prev_rates

    df = pd.DataFrame(results).set_index("ticker")
    return df, prev_rates


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
        as_of_date = row.get("기준일")
        close_price = int(row.get("종가", 0) or 0)

        # KIS API는 상장주식수/한도수량 미제공 → 소진율 0이면 스킵
        if exhaustion <= 0:
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
            "as_of_date": as_of_date,
            "close": close_price,
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
        "data_source": signal.get("data_source", "unknown"),
        "as_of_date": signal.get("as_of_date", signal.get("date")),
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
    lines = [f"외인소진율 리포트 ({signal.get('as_of_date', signal['date'])})"]
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


def _dominant_as_of_date(df: pd.DataFrame, fallback: str) -> str:
    """수집 결과의 대표 거래일 결정."""
    if "기준일" not in df.columns or df.empty:
        return fallback
    dates = [str(d) for d in df["기준일"].dropna().tolist() if str(d).strip()]
    if not dates:
        return fallback
    return pd.Series(dates).mode().iloc[0]


def collect_foreign_exhaustion(
    source: str = "naver",
    naver_pages: int = 1,
    tickers: list[str] | None = None,
) -> dict:
    """외인소진율 수집 + 시그널 생성."""
    from src.pipeline_alert import PipelineErrorTracker
    tracker = PipelineErrorTracker("collect_foreign_exhaustion")
    logger.info("=== 외인소진율 수집 시작 ===")

    today = datetime.now()
    today_str = today.strftime("%Y%m%d")
    prev_date = today - timedelta(days=7)  # 5영업일 ~ 7달력일
    prev_str = prev_date.strftime("%Y%m%d")

    # 1. 유니버스
    universe_list = tickers or _get_universe()
    universe = set(universe_list)
    logger.info(f"유니버스: {len(universe)}종목")

    # 2. 소진율 수집: 네이버 일별 primary, KIS 현재가 fallback
    df = pd.DataFrame()
    prev_rates: dict[str, float] = {}
    data_source = source

    if source in {"naver", "auto"}:
        logger.info("네이버 frgn 일별 외인소진율 수집 시작...")
        df, prev_rates = _fetch_exhaustion_naver(universe_list, pages=naver_pages)
        if not df.empty:
            data_source = "naver_frgn_daily"

    if df.empty and source in {"kis", "auto"}:
        logger.warning("네이버 일별 수집 실패 또는 미사용 — KIS 현재가 fallback 실행")
        logger.info("KIS API 외인소진율 수집 시작...")
        df = _fetch_exhaustion_kis(universe_list)
        data_source = "kis_api_current_snapshot"

    if df.empty:
        logger.error("외인소진율 데이터 수집 실패")
        tracker.finalize(total=0)
        return {"date": today.strftime("%Y-%m-%d"), "error": "데이터 없음"}

    date_label = datetime.strptime(today_str, "%Y%m%d").strftime("%Y-%m-%d")
    as_of_date = _dominant_as_of_date(df, date_label)
    raw_as_of_counts = (
        df["기준일"].value_counts().to_dict()
        if "기준일" in df.columns
        else {}
    )
    stale_count = 0
    if "기준일" in df.columns:
        stale_count = int((df["기준일"] != as_of_date).sum())
        if stale_count:
            logger.warning("대표 거래일(%s) 불일치 stale 행 %d건 제외", as_of_date, stale_count)
            df = df[df["기준일"] == as_of_date].copy()

    logger.info(f"수집 완료: {len(df)}종목")

    # 3. 이전 소진율 (네이버는 페이지 내 5거래일 전, KIS fallback은 히스토리)
    if not prev_rates:
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
    signal = {
        "date": as_of_date,
        "as_of_date": as_of_date,
        "generated_at": today.strftime("%Y-%m-%d %H:%M"),
        "data_source": data_source,
        "source_notes": (
            "네이버 frgn 거래일별 날짜/종가/보유주식수/보유율 기준"
            if data_source == "naver_frgn_daily"
            else "KIS 현재가 스냅샷 fallback: 장전/휴장 as-of 보정 불가"
        ),
        "total_analyzed": len(all_signals),
        "summary": {
            "high90_count": len(high90),
            "high80_count": len(high80),
            "avg_exhaustion": round(avg_rate, 2),
            "rising_count": len(rising),
            "fe1_count": len(fe1),
            "fe2_count": len(fe2),
            "fe3_count": len(fe3),
            "as_of_counts": raw_as_of_counts,
            "stale_excluded_count": stale_count,
        },
        "top_exhaustion": top_exhaustion,
        "rising_exhaustion": rising_exhaustion,
        "fe1_signals": sorted(fe1, key=lambda x: x["exhaustion_rate"], reverse=True),
        "fe2_signals": sorted(fe2, key=lambda x: x["exhaustion_5d_change"] or 0, reverse=True),
        "fe3_signals": sorted(fe3, key=lambda x: x["score"], reverse=True),
        "all_signals": all_signals,
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
    parser.add_argument(
        "--source",
        choices=["naver", "kis", "auto"],
        default="naver",
        help="수집 소스 (기본: naver, auto는 네이버 실패 시 KIS fallback)",
    )
    parser.add_argument("--naver-pages", type=int, default=1, help="네이버 frgn 조회 페이지 수")
    parser.add_argument("--tickers", type=str, default="", help="쉼표 구분 테스트/복구 대상 종목")
    args = parser.parse_args()

    # 비거래일(주말/휴장) 스킵
    if not _is_trading_day():
        logger.info("비거래일(주말/휴장) — 스킵")
        print("비거래일(주말/휴장) — 스킵")
        return

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or None
    signal = collect_foreign_exhaustion(
        source=args.source,
        naver_pages=args.naver_pages,
        tickers=tickers,
    )

    if signal.get("error"):
        logger.error(signal["error"])
        sys.exit(1)

    # 콘솔 출력
    print(f"\n=== 외인소진율 리포트 ===")
    print(f"기준일: {signal.get('as_of_date', signal['date'])} | 소스: {signal.get('data_source')}")
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
