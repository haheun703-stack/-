#!/usr/bin/env python3
"""원자재 가격 수집기 v2 (7/17 B-12 부활).

배경: 구 수집기(Alpha Vantage 기반)가 4/21 아카이브된 후 `data/commodity_prices.json`이
3/26 유물로 방치 — 건강검진은 통과시키고(7/16 신선도 게이트로 봉쇄) 시나리오 대시보드는
석 달 전 원가갭을 현재처럼 노출하던 상태. yfinance 선물 시세로 재구축.

출력: data/commodity_prices.json
    {date, updated_at, source, commodities: {key: {name, price, unit, ret_1d_pct}}}

★cost_gap(원가 갭) 미포함 — 원가 기준값(production_cost)은 출처 확보 전까지 창작 금지.
  소비처는 cost_gap 없으면 전부 우아하게 스킵:
    - dashboard_data.build_zone_scenario: `if not cg: continue` (원자재 섹션 빈 상태 = 정직)
    - scan_tomorrow_picks 전략 M risk_reward: `if gap:` 가드
    - data_health_check 원자재 체크: 날짜 신선도만 봄 → ✅ 복원

실행:
    python scripts/collect_commodity_prices.py
cron: BAT-A (06:10, 미장 마감 후) — run_bat.sh
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "commodity_prices.json"

# key: (yfinance 티커, 표시명, 단위) — key는 기존 소비처 계약 유지 (wti/natural_gas/copper/gold)
TICKERS = {
    "wti": ("CL=F", "WTI 원유", "USD/bbl"),
    "natural_gas": ("NG=F", "천연가스", "USD/MMBtu"),
    "copper": ("HG=F", "구리", "USD/lb"),
    "gold": ("GC=F", "금", "USD/oz"),
    "dxy": ("DX-Y.NYB", "달러인덱스", "index"),
}


def fetch_one(yf_ticker: str) -> tuple[float, float] | None:
    """최근 종가 + 전일 대비 % 반환. 실패 시 None."""
    import yfinance as yf

    h = yf.Ticker(yf_ticker).history(period="5d", interval="1d")
    if h is None or h.empty or "Close" not in h:
        return None
    closes = h["Close"].dropna()
    if closes.empty:
        return None
    price = float(closes.iloc[-1])
    ret_1d = 0.0
    if len(closes) >= 2 and float(closes.iloc[-2]) > 0:
        ret_1d = (price - float(closes.iloc[-2])) / float(closes.iloc[-2]) * 100
    return price, ret_1d


def main() -> None:
    results: dict[str, dict] = {}
    for key, (yf_ticker, name, unit) in TICKERS.items():
        try:
            fetched = fetch_one(yf_ticker)
        except Exception as e:
            logger.warning("[원자재] %s(%s) 조회 실패: %s", name, yf_ticker, e)
            fetched = None
        if fetched is None:
            logger.warning("[원자재] %s(%s) 데이터 없음 — 스킵", name, yf_ticker)
            continue
        price, ret_1d = fetched
        results[key] = {
            "name": name,
            "price": round(price, 3),
            "unit": unit,
            "ret_1d_pct": round(ret_1d, 2),
        }
        logger.info("[원자재] %s = %.3f %s (%+.2f%%)", name, price, unit, ret_1d)

    if not results:
        # 조용한 실패 방지: 전량 실패면 유물 파일을 덮지 않고 exit 1 (FAIL_COUNT 가시화)
        logger.error("[원자재] 수집 0종 — 기존 파일 보존, 중단 (exit 1)")
        sys.exit(1)

    out = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "yfinance futures (v2, 7/17 재구축)",
        "commodities": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    tmp.replace(OUTPUT_PATH)  # 원자 쓰기 (부분 파일 방지)

    logger.info("[원자재] 저장 완료: %s (%d종)", OUTPUT_PATH, len(results))


if __name__ == "__main__":
    main()
