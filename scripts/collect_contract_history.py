"""공급계약 공시 일일 수집기 — 계약금액/매출대비 파싱 → contract_history.jsonl 누적.

미래가치 엔진 O축(수주 모멘텀)의 이력 데이터. BAT-D 배선(FV 엔진보다 앞).
- 최근 N일(기본 3, 주말/휴일 커버) '단일판매ㆍ공급계약체결' 원공시 수집
  (기재정정·해지 공시 제외, 종속회사 공시는 subsidiary 플래그)
- dart_contract_parser로 정량 필드 파싱
- data/contract_history.jsonl 에 append. dedup은 '파싱 성공(rcept_no)' 기준 —
  매출대비 파싱 실패 레코드는 이후 파서 개선 시 재파싱 허용(7/4 적대검수).
graceful: 전 구간 try — 어떤 실패도 exit 0 (BAT 무손상, 7/4 적대검수 HIGH 반영).
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


def load_seen() -> tuple[set[str], set[str]]:
    """(파싱 성공 rcept_no, 파싱 실패 rcept_no). 손상 라인은 조용히 무시."""
    ok: set[str] = set()
    failed: set[str] = set()
    if HISTORY_PATH.exists():
        # errors="replace": 절단/손상 바이트가 있어도 순회가 죽지 않게 (poison-pill 방지)
        with open(HISTORY_PATH, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    rno = rec["rcept_no"]
                except Exception:  # noqa: BLE001 — 손상 라인 스킵
                    continue
                if rec.get("revenue_ratio_pct") is not None:
                    ok.add(rno)
                else:
                    failed.add(rno)
    return ok, failed


def collect_events(api_key: str, bgn: datetime, end: datetime) -> list[dict]:
    events, page = [], 1
    while True:
        r = requests.get(LIST_URL, params={
            "crtfc_key": api_key, "bgn_de": bgn.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"), "pblntf_ty": "I",
            "page_no": page, "page_count": 100}, timeout=30)
        d = r.json()
        status = d.get("status")
        if status != "000":
            # 013=조회 결과 없음(휴일 등 정상). 그 외(010 키오류·020 한도초과 등)는 경고 —
            # 무음 결측 방지(7/4 적대검수): 키 만료가 '0건 정상'으로 위장되지 않게.
            if status != "013":
                logger.warning("[contract] list API 비정상 status=%s msg=%s",
                               status, d.get("message"))
            break
        for x in d.get("list", []):
            nm = x.get("report_nm", "")
            if ("공급계약" in nm and "기재정정" not in nm and "해지" not in nm
                    and x.get("stock_code")):
                events.append({"ticker": x["stock_code"], "name": x["corp_name"],
                               "date": x["rcept_dt"], "rcept_no": x["rcept_no"],
                               "market": x.get("corp_cls", ""),
                               "subsidiary": "종속회사" in nm})
        if page >= int(d.get("total_page", 1)):
            break
        page += 1
        time.sleep(0.12)
    return events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3, help="조회 기간(일)")
    args = ap.parse_args()

    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        logger.warning("[contract] DART_API_KEY 없음 — 스킵(graceful)")
        return 0

    end = datetime.now()
    events: list[dict] = []
    try:
        events = collect_events(api_key, end - timedelta(days=args.days), end)
    except Exception as e:  # noqa: BLE001
        logger.warning("[contract] 목록 수집 실패(부분 진행): %s", e)

    seen_ok, seen_failed = load_seen()
    new_events = [e for e in events if e["rcept_no"] not in seen_ok]
    logger.info("[contract] 공급계약 원공시 %d건 중 미파싱 %d건", len(events), len(new_events))

    added = 0
    sess = requests.Session()
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        for ev in new_events:
            det = fetch_contract_detail(ev["rcept_no"], api_key, session=sess)
            if not det:
                continue
            if det.get("revenue_ratio_pct") is None and ev["rcept_no"] in seen_failed:
                continue  # 이미 실패 기록 있음 — 성공 시에만 갱신 레코드 추가
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
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — graceful 계약: 어떤 실패도 BAT 무손상
        logger.warning("[contract] 실패(exit 0 유지): %s", exc)
        sys.exit(0)
