"""공급계약 공시 일일 수집기 — 계약금액/매출대비 파싱 → contract_history.jsonl 누적.

미래가치 엔진 O축(수주 모멘텀)의 이력 데이터. BAT-D 배선(FV 엔진보다 앞).
- 최근 N일(기본 3, 주말/휴일 커버) '단일판매ㆍ공급계약체결' 원공시(정정 제외) 수집
- dart_contract_parser로 정량 필드 파싱
- data/contract_history.jsonl 에 rcept_no 기준 중복 없이 append
graceful: API 키 없음/실패 모두 exit 0 (BAT 무손상).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.adapters.dart_contract_parser import fetch_contract_detail  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("collect_contract_history")

LIST_URL = "https://opendart.fss.or.kr/api/list.json"
HISTORY_PATH = PROJECT_ROOT / "data" / "contract_history.jsonl"


def load_seen() -> set[str]:
    seen = set()
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["rcept_no"])
                except Exception:  # noqa: BLE001
                    continue
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3, help="조회 기간(일)")
    args = ap.parse_args()

    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        logger.warning("[contract] DART_API_KEY 없음 — 스킵(graceful)")
        return 0

    end = datetime.now()
    bgn = end - timedelta(days=args.days)
    events, page = [], 1
    try:
        while True:
            r = requests.get(LIST_URL, params={
                "crtfc_key": api_key, "bgn_de": bgn.strftime("%Y%m%d"),
                "end_de": end.strftime("%Y%m%d"), "pblntf_ty": "I",
                "page_no": page, "page_count": 100}, timeout=30)
            d = r.json()
            if d.get("status") != "000":
                break
            for x in d.get("list", []):
                nm = x.get("report_nm", "")
                if "공급계약" in nm and "기재정정" not in nm and x.get("stock_code"):
                    events.append({"ticker": x["stock_code"], "name": x["corp_name"],
                                   "date": x["rcept_dt"], "rcept_no": x["rcept_no"],
                                   "market": x.get("corp_cls", "")})
            if page >= int(d.get("total_page", 1)):
                break
            page += 1
            time.sleep(0.12)
    except Exception as e:  # noqa: BLE001
        logger.warning("[contract] 목록 수집 실패(부분 진행): %s", e)

    seen = load_seen()
    new_events = [e for e in events if e["rcept_no"] not in seen]
    logger.info("[contract] 공급계약 원공시 %d건 중 신규 %d건", len(events), len(new_events))

    added = 0
    sess = requests.Session()
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        for ev in new_events:
            det = fetch_contract_detail(ev["rcept_no"], api_key, session=sess)
            if not det:
                continue
            rec = {**ev, **det, "collected_at": datetime.now().isoformat(timespec="seconds")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            added += 1
            ratio = det.get("revenue_ratio_pct")
            amt = det.get("contract_amount")
            logger.info("  + %s(%s) 계약 %s원 매출대비 %s%%", ev["name"], ev["ticker"],
                        f"{amt:,.0f}" if amt else "?", ratio if ratio is not None else "?")
            time.sleep(0.12)
    logger.info("[contract] 신규 적재 %d건 → %s", added, HISTORY_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
