"""시가총액 조회 — 살아 있는 소스(universe.csv) 단일 창구 (B-77 ⑵, 2026-09-07).

배경
  `data/market_cap_cache.json`은 생성기가 저장소 어디에도 없어 **`hts_avls` 값이
  2026-03-04에 굳어** 있었는데, `value_intrinsic`·`value_ebitda_ev`가 이를 EV/EBITDA와
  내재가치의 **분모**로 매일 썼다(`scan_nugget` → BAT-D 매일, 9/4 로그 `ValueIntrinsic
  엔진 초기화 완료` 확인).

  9/7 실측: 표본 495종목의 3월→9월 주가배수 중앙값 0.858, **21.6%가 ±50% 이상 이동**
  (최대 7.26배). 즉 다섯 종목 중 하나는 반년 전 시총으로 밸류를 재고 있었다.

교체 근거 (B-91 방식 — 근사를 쓰기 전에 대조한다)
  `data/universe.csv`에 `market_cap`(원 단위) 컬럼이 있고 **매일 갱신**된다(9/4 11:31,
  899종목). 캐시와 교집합 830종목에서 `universe(억환산) / 캐시` 비율의 **중앙값 0.8609**로,
  같은 기간 **주가배수 중앙값 0.858**과 일치했다. 서로 독립인 두 경로가 같은 값을 주므로
  단위(원 → 억)와 규모가 맞다고 판정했다.

원칙
  - **캐시로 폴백하지 않는다.** 폴백을 두면 낡은 값이 다시 섞이고, 그건 이 파일이
    없애려는 상태 그대로다(B-14 "출처 없으면 빈 상태가 정직").
  - universe에 없는 종목은 **0이 아니라 미수록**으로 둔다 — 0으로 채우면 EV가 0이 되어
    "가장 싼 종목"으로 뒤집힌다(8/21 B-91 `trading_value` 0 채움과 같은 함정).
  - universe.csv 자체가 낡으면(기본 7일) 전부 비운다.
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UNIVERSE_CSV = PROJECT_ROOT / "data" / "universe.csv"

#: universe.csv 신선도 임계(일). 주 1회 전체 재구성 + 증분이라 7일이면 충분하다.
UNIVERSE_MAX_AGE_DAYS = 7


def load_market_cap_eok(max_age_days: int = UNIVERSE_MAX_AGE_DAYS) -> dict[str, int]:
    """{ticker: 시가총액(억원)}. 소스가 없거나 낡으면 빈 dict.

    반환 단위는 **억원** — 구 `market_cap_cache.json`의 `hts_avls`와 같은 단위라
    소비처 계산식을 바꾸지 않는다.
    """
    if not UNIVERSE_CSV.exists():
        logger.warning("[market_cap] universe.csv 없음 — 시총 미제공")
        return {}
    try:
        age_days = (datetime.now()
                    - datetime.fromtimestamp(UNIVERSE_CSV.stat().st_mtime)).days
    except OSError:
        logger.warning("[market_cap] universe.csv mtime 확인 불가 — 시총 미제공")
        return {}
    if age_days > max_age_days:
        logger.warning("[market_cap] universe.csv %d일 낡음(임계 %d) — 시총 미제공",
                       age_days, max_age_days)
        return {}

    result: dict[str, int] = {}
    try:
        with open(UNIVERSE_CSV, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                ticker = (row.get("ticker") or "").strip().zfill(6)
                raw = (row.get("market_cap") or "").strip()
                if not ticker or not raw:
                    continue
                try:
                    won = float(raw)
                except ValueError:
                    continue
                if won <= 0:      # 0은 값이 아니라 미수집이다 — 담지 않는다
                    continue
                result[ticker] = int(won / 1e8)   # 원 → 억원
    except OSError as e:
        logger.warning("[market_cap] universe.csv 읽기 실패: %s", e)
        return {}

    logger.info("[market_cap] universe.csv에서 %d종목 시총 로드(억원)", len(result))
    return result


def load_market_cap_legacy_shape() -> dict[str, dict]:
    """구 `market_cap_cache.json` 형태 `{ticker: {"hts_avls": 억원}}`로 반환.

    소비처가 `.get("hts_avls")`로 읽고 있어 호출부 수정을 최소화하기 위한 어댑터다.
    """
    return {t: {"hts_avls": v} for t, v in load_market_cap_eok().items()}
