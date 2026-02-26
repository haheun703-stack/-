"""DART 이벤트 드리븐 시그널 — Timefolio 벤치마크 전략 3

기존 crawl_dart_disclosure.py의 출력(dart_disclosures.json)을
이벤트 드리븐 매매 시그널로 변환한다.

Tier1/Tier2 공시에서 매매 액션 가능한 시그널을 생성하고,
기술적 지표로 보정하여 최종 점수를 산출한다.

입력:
  - data/dart_disclosures.json (crawl_dart_disclosure.py 출력)
  - data/processed/*.parquet (기술적 지표)
  - stock_data_daily/*.csv (종목명 매핑)

출력:
  - data/dart_event_signals.json

Usage:
    python scripts/dart_event_signal.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
DART_JSON = DATA_DIR / "dart_disclosures.json"
OUTPUT_PATH = DATA_DIR / "dart_event_signals.json"

# ── 이벤트→점수 매핑 ──
BUY_EVENTS = {
    "자기주식취득": 20,
    "자기주식처분": -5,   # 처분은 마이너스
    "공급계약체결": 15,
    "수주": 15,
    "대규모내부거래": 10,
    "타법인주식및출자증권취득결정": 10,
    "배당": 10,
    "매출액또는손익구조": 12,
    "잠정실적": 12,
    "사업보고서": 8,
    "반기보고서": 8,
    "분기보고서": 8,
    "특허": 10,
    "라이선스": 10,
    "기술이전": 10,
    "FDA": 15,
    "임상": 12,
    "승인": 12,
    "무상증자": 8,
}

WATCH_EVENTS = {
    "최대주주변경": 5,
    "합병": 5,
    "분할": 5,
    "인수": 5,
    "공개매수": 5,
    "주식교환": 5,
    "주식이전": 5,
}

AVOID_EVENTS = {
    "유상증자": -15,
    "전환사채": -10,
    "신주인수권부사채": -10,
    "관리종목": -30,
    "거래정지": -30,
    "상장폐지": -30,
    "회생절차": -25,
    "파산": -30,
}


def build_name_map() -> dict[str, str]:
    """CSV 파일명에서 종목코드→이름 매핑"""
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name_map[parts[1]] = parts[0]
    return name_map


def get_parquet_data(ticker: str) -> dict | None:
    """parquet에서 최신 기술적 지표 추출 (간소화 버전)"""
    pq_path = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq_path.exists():
        return None
    try:
        df = pd.read_parquet(pq_path).tail(10)
        if len(df) < 3:
            return None
        last = df.iloc[-1]
        close = float(last.get("close", 0))

        # 외인/기관 5일 합산
        f5 = float(np.nansum(df.tail(5)["외국인합계"].values)) if "외국인합계" in df.columns else 0
        i5 = float(np.nansum(df.tail(5)["기관합계"].values)) if "기관합계" in df.columns else 0

        return {
            "close": close,
            "rsi": float(last.get("rsi_14", 50)),
            "adx": float(last.get("adx_14", 20)),
            "bb_pos": float(last.get("bb_position", 50)),
            "foreign_5d": f5,
            "inst_5d": i5,
        }
    except Exception as e:
        logger.warning("parquet 읽기 실패 %s: %s", ticker, e)
        return None


def score_event(disclosure: dict, parquet_data: dict | None) -> dict | None:
    """DART 공시 → 매매 시그널 + 점수"""
    keyword = disclosure.get("keyword", "")
    tier = disclosure.get("tier", "")
    stock_code = disclosure.get("stock_code", "").strip()
    corp_name = disclosure.get("corp_name", "")

    if not keyword or not stock_code:
        return None

    # 이벤트 기본 점수 + 액션 판정
    if keyword in AVOID_EVENTS:
        base_score = AVOID_EVENTS[keyword]
        action = "AVOID"
    elif keyword in BUY_EVENTS:
        base_score = BUY_EVENTS[keyword]
        action = "BUY"
    elif keyword in WATCH_EVENTS:
        base_score = WATCH_EVENTS[keyword]
        action = "WATCH"
    else:
        # 매핑되지 않은 키워드 — tier에 따라 기본값
        if tier == "tier1_즉시":
            base_score = 8
            action = "WATCH"
        elif tier == "tier2_중요":
            base_score = 5
            action = "WATCH"
        else:
            return None  # tier3 이하 무시

    event_score = base_score

    # 기술적 보정 (BUY/WATCH만)
    rsi = 50
    if parquet_data and action in ("BUY", "WATCH"):
        rsi = parquet_data.get("rsi", 50)
        # RSI 과열이면 매수 시그널 감점
        if rsi > 70:
            event_score = max(event_score - 10, 0)
        elif rsi > 60:
            event_score = max(event_score - 3, 0)

        # 수급 보정: 외인+기관 순매수면 가산
        if parquet_data.get("foreign_5d", 0) > 0 and parquet_data.get("inst_5d", 0) > 0:
            event_score += 5
        elif parquet_data.get("foreign_5d", 0) > 0:
            event_score += 2

    return {
        "ticker": stock_code,
        "name": corp_name,
        "event": keyword,
        "report_nm": disclosure.get("report_nm", "")[:60],
        "tier": tier,
        "action": action,
        "event_score": event_score,
        "rsi": round(rsi, 1),
        "url": disclosure.get("url", ""),
    }


def main():
    logger.info("=" * 60)
    logger.info("  DART 이벤트 드리븐 시그널 — 전략 3")
    logger.info("=" * 60)

    # DART 공시 로드
    if not DART_JSON.exists():
        logger.warning("DART 공시 파일 없음: %s", DART_JSON)
        output = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "source_period": "",
            "total_events": 0,
            "actionable_count": 0,
            "signals": [],
            "avoid_list": [],
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("  빈 결과 저장: %s", OUTPUT_PATH)
        return

    with open(DART_JSON, encoding="utf-8") as f:
        dart_data = json.load(f)

    name_map = build_name_map()
    source_period = dart_data.get("period", "")

    # tier1 + tier2 + universe_hits 통합
    all_disclosures = []
    seen_keys = set()

    for tier_key in ("tier1", "tier2", "universe_hits"):
        for item in dart_data.get(tier_key, []):
            # 중복 제거 (같은 종목+같은 키워드)
            dedup_key = f"{item.get('stock_code')}_{item.get('keyword')}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            all_disclosures.append(item)

    logger.info("  공시 %d건 (tier1:%d, tier2:%d, universe:%d)",
                len(all_disclosures),
                dart_data.get("tier1_count", 0),
                dart_data.get("tier2_count", 0),
                dart_data.get("universe_hit_count", 0))

    # 이벤트 스코어링
    buy_signals = []
    watch_signals = []
    avoid_list = []

    for disc in all_disclosures:
        stock_code = disc.get("stock_code", "").strip()
        pq_data = get_parquet_data(stock_code) if stock_code else None

        result = score_event(disc, pq_data)
        if not result:
            continue

        # 종목명 보정
        if not result["name"] and stock_code in name_map:
            result["name"] = name_map[stock_code]

        if result["action"] == "BUY":
            buy_signals.append(result)
        elif result["action"] == "WATCH":
            watch_signals.append(result)
        elif result["action"] == "AVOID":
            avoid_list.append(result)

    # 점수 내림차순 정렬
    buy_signals.sort(key=lambda x: -x["event_score"])
    watch_signals.sort(key=lambda x: -x["event_score"])

    # 통합 시그널 (BUY + WATCH, 상위 20건)
    signals = buy_signals + watch_signals
    signals = signals[:20]

    actionable_count = len(buy_signals)

    # 출력
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source_period": source_period,
        "total_events": len(all_disclosures),
        "actionable_count": actionable_count,
        "signals": signals,
        "avoid_list": avoid_list[:10],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("── 결과 ──")
    logger.info("  BUY 시그널: %d건", len(buy_signals))
    logger.info("  WATCH 시그널: %d건", len(watch_signals))
    logger.info("  AVOID 종목: %d건", len(avoid_list))

    for sig in buy_signals[:5]:
        logger.info("  🟢 BUY  %s(%s) +%d점 — %s",
                     sig["name"], sig["ticker"], sig["event_score"], sig["event"])
    for sig in avoid_list[:3]:
        logger.info("  🔴 AVOID %s(%s) %d점 — %s",
                     sig["name"], sig["ticker"], sig["event_score"], sig["event"])

    logger.info("  저장: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
