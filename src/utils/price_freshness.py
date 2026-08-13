"""종가 parquet 신선도 판정 — B-65/B-73 공용 단일 소스.

★기준을 여기 한 곳에만 둔다.
  8/13 B-67에서 소비자(`run_scenario_v1`)가 생산자의 어휘를 하드코딩해
  **배포 이래 필터가 한 번도 동작하지 않았던** 실패를 겪었다. 임계·기준일
  산정이 두 벌 존재하면 한쪽만 고쳐져도 아무 에러 없이 조용히 어긋난다.
  그래서 페이퍼 진입 차단(B-65)과 픽 생성 필터(B-73)가 같은 함수를 쓴다.

판정 규칙
  - 기준일 = 시장 최신 거래일(`data/kospi_index.csv` 마지막 행).
  - 개별 종목 parquet의 마지막 봉이 기준일보다 STALE_PRICE_MAX_DAYS(달력일)를
    **초과해** 뒤처지면 stale.
  - 기준일 소스가 깨지면 오늘로 폴백한다 = 더 엄격해지는 방향으로 실패한다
    (fail-open이 아니라 fail-closed).

왜 필요한가 (B-65 실발생)
  봉이 멈춘 종목은 진입가·청산가가 모두 같은 옛 종가에서 파생돼 손익률이
  상수가 된다. 손절·익절·트레일링이 구조적으로 발동할 수 없고 MAX_HOLD로만
  빠지며 슬롯을 점유한다. 003410(마지막 봉 2024-12-30)이 실제로 그렇게
  진입 18,796 → 청산 18,755로 원장에 남았다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
KOSPI_CSV_PATH = DATA_DIR / "kospi_index.csv"

#: 시장 기준일 대비 허용 지연(달력일). 초과 시 stale.
#: 연휴·일시 거래정지에서 오탐이 나지 않도록 여유를 둔 값이다.
STALE_PRICE_MAX_DAYS = 7

_ref_cache: list = []          # [pd.Timestamp]
_stale_cache: dict[str, bool] = {}


def market_reference_date() -> pd.Timestamp:
    """시장 최신 거래일. 소스가 없거나 깨지면 오늘로 폴백(더 엄격)."""
    if _ref_cache:
        return _ref_cache[0]
    ref = None
    try:
        df = pd.read_csv(KOSPI_CSV_PATH, usecols=["Date"])
        if len(df) > 0:
            ref = pd.Timestamp(df["Date"].iloc[-1])
    except Exception:  # noqa: BLE001  (소스 파손 — 폴백으로 흡수)
        ref = None
    if ref is None or pd.isna(ref):
        ref = pd.Timestamp(datetime.now().date())
    _ref_cache.append(ref)
    return ref


def last_bar_date(ticker: str) -> str:
    """종목 parquet의 마지막 봉 날짜(YYYY-MM-DD). 없으면 빈 문자열."""
    pq = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq.exists():
        pq = DATA_DIR / "raw" / f"{ticker}.parquet"
    if not pq.exists():
        return ""
    try:
        df = pd.read_parquet(pq, columns=["close"])
        if len(df) == 0:
            return ""
        return df.index[-1].strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return ""


def stale_lag_days(ticker: str) -> int | None:
    """기준일 대비 지연 일수. 판정 불가면 None."""
    d = last_bar_date(ticker)
    if not d:
        return None
    return int((market_reference_date() - pd.Timestamp(d)).days)


def is_ticker_stale(ticker: str) -> bool:
    """봉이 멈춘 종목인가. 판정 불가(parquet 없음)는 False.

    ★판정 불가를 True로 만들지 않는 이유: 호출부들은 이미 그 앞에서
      `price <= 0`으로 걸러낸다. 여기서까지 막으면 원인이 두 겹이 돼
      어느 가드가 걸렀는지 로그로 구분되지 않는다.
    """
    if ticker in _stale_cache:
        return _stale_cache[ticker]
    lag = stale_lag_days(ticker)
    stale = lag is not None and lag > STALE_PRICE_MAX_DAYS
    _stale_cache[ticker] = stale
    return stale


def reset_cache() -> None:
    """장기 실행 프로세스/테스트용 캐시 초기화."""
    _ref_cache.clear()
    _stale_cache.clear()
