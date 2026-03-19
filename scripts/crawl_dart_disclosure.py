"""DART 전자공시 크롤링 → data/dart_disclosures.json

OpenDART API에서 당일 공시를 수집하고,
고영향 공시를 티어별로 분류하여 저장한다.

뉴스보다 30분~수시간 선행하는 정보원:
  공시 발생 → (30분~수시간) → 기자 기사 작성 → RSS 수집

기능:
  1. 당일 전체 공시 목록 수집 (DART list API)
  2. 3-Tier 영향도 분류 (tier1 즉시 / tier2 중요 / tier3 참고)
  3. 우리 유니버스(84종목) 관련 공시 자동 매칭
  4. 텔레그램 알림 연동 (tier1 즉시 전송)

사용법:
  python scripts/crawl_dart_disclosure.py           # 당일 공시 수집
  python scripts/crawl_dart_disclosure.py --days 3  # 최근 3일
  python scripts/crawl_dart_disclosure.py --alert    # 텔레그램 알림 포함
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUT_PATH = PROJECT_ROOT / "data" / "dart_disclosures.json"
UNIVERSE_DIR = PROJECT_ROOT / "data" / "processed"

# ── DART API ──
DART_API_URL = "https://opendart.fss.or.kr/api/list.json"
DART_VIEWER_URL = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={}"


def _get_api_key() -> str:
    """DART API 키 로드 (.env → 환경변수)"""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    key = os.getenv("DART_API_KEY", "")
    if not key:
        raise ValueError("DART_API_KEY가 .env에 설정되지 않았습니다")
    return key


# ══════════════════════════════════════════
# 공시 영향도 분류
# ══════════════════════════════════════════

TIER1_KEYWORDS = [
    # 지배구조 변경 (즉시 반응)
    "최대주주변경", "합병", "분할", "유상증자", "무상증자",
    "자기주식취득", "자기주식처분", "전환사채", "신주인수권부사채",
    "공개매수", "주식교환", "주식이전",
    # 거래정지/폐지 (즉시 반응)
    "상장폐지", "관리종목", "거래정지", "회생절차", "파산",
    # 대규모 이벤트
    "공급계약체결", "대규모내부거래", "타법인주식및출자증권취득결정",
]

TIER2_KEYWORDS = [
    # 실적/재무
    "매출액또는손익구조", "영업이익", "당기순이익", "실적",
    "매출액", "잠정실적", "사업보고서", "반기보고서", "분기보고서",
    # 사업/계약
    "수주", "공급계약", "특허", "라이선스", "기술이전",
    "FDA", "임상", "승인", "인수", "투자",
    # 주주환원
    "배당", "자사주", "주주총회소집",
    # 구조변경
    "분할합병", "사업양수도", "영업양수도",
]

TIER3_KEYWORDS = [
    # 내부자 거래
    "임원", "주요주주", "특정증권등소유상황",
    # 기타
    "소송", "횡령", "배임", "정정",
    "불성실공시", "조회공시",
]

# 무시할 일반 공시 (노이즈 필터)
IGNORE_KEYWORDS = [
    "증권신고서", "투자설명서", "주요사항보고서(자율공시)",
    "기업설명회", "공정공시",
]


def classify_disclosure(report_nm: str) -> tuple[str, str | None]:
    """공시 제목 → (tier, matched_keyword) 분류"""
    # 무시 목록 체크
    for kw in IGNORE_KEYWORDS:
        if kw in report_nm:
            return "tier4_일반", None

    for kw in TIER1_KEYWORDS:
        if kw in report_nm:
            return "tier1_즉시", kw

    for kw in TIER2_KEYWORDS:
        if kw in report_nm:
            return "tier2_중요", kw

    for kw in TIER3_KEYWORDS:
        if kw in report_nm:
            return "tier3_참고", kw

    return "tier4_일반", None


# ══════════════════════════════════════════
# 유니버스 매칭
# ══════════════════════════════════════════

def build_universe_codes() -> set[str]:
    """우리 유니버스(84종목) 종목코드 세트"""
    codes = set()
    for pq in UNIVERSE_DIR.glob("*.parquet"):
        code = pq.stem
        if len(code) == 6 and code.isdigit():
            codes.add(code)
    return codes


# ══════════════════════════════════════════
# DART API 호출
# ══════════════════════════════════════════

def fetch_dart_disclosures(api_key: str, bgn_de: str, end_de: str,
                           page_count: int = 100) -> list[dict]:
    """DART 공시 목록 전체 수집 (페이지네이션 처리)"""
    all_items = []
    page_no = 1

    while True:
        params = {
            "crtfc_key": api_key,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "page_no": page_no,
            "page_count": page_count,
            "sort": "date",
            "sort_mth": "desc",
        }

        try:
            resp = requests.get(DART_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("DART API 호출 실패 (page %d): %s", page_no, e)
            break

        status = data.get("status", "")
        if status == "013":
            # 013 = 조회 결과 없음
            logger.info("DART: 해당 기간 공시 없음")
            break
        if status != "000":
            logger.error("DART API 오류: status=%s, message=%s",
                         status, data.get("message", ""))
            break

        items = data.get("list", [])
        all_items.extend(items)

        total_page = data.get("total_page", 1)
        logger.info("  DART 페이지 %d/%d (%d건)", page_no, total_page, len(items))

        if page_no >= total_page:
            break
        page_no += 1

    return all_items


def process_disclosures(raw_items: list[dict],
                        universe_codes: set[str]) -> dict:
    """공시 목록을 분류하고 유니버스 매칭"""
    tier1 = []
    tier2 = []
    tier3 = []
    universe_hits = []

    for item in raw_items:
        corp_name = item.get("corp_name", "")
        stock_code = item.get("stock_code", "").strip()
        report_nm = item.get("report_nm", "").strip()
        rcept_no = item.get("rcept_no", "")
        rcept_dt = item.get("rcept_dt", "")
        corp_cls = item.get("corp_cls", "")  # Y:유가 K:코스닥 N:코넥스 E:기타

        tier, keyword = classify_disclosure(report_nm)

        entry = {
            "corp_name": corp_name,
            "stock_code": stock_code,
            "report_nm": report_nm,
            "rcept_dt": rcept_dt,
            "tier": tier,
            "keyword": keyword,
            "market": {"Y": "유가증권", "K": "코스닥", "N": "코넥스", "E": "기타"}.get(corp_cls, ""),
            "url": DART_VIEWER_URL.format(rcept_no),
        }

        if tier == "tier1_즉시":
            tier1.append(entry)
        elif tier == "tier2_중요":
            tier2.append(entry)
        elif tier == "tier3_참고":
            tier3.append(entry)

        # 유니버스 매칭
        if stock_code and stock_code in universe_codes and tier != "tier4_일반":
            entry_copy = dict(entry)
            entry_copy["in_universe"] = True
            universe_hits.append(entry_copy)

    return {
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "universe_hits": universe_hits,
    }


# ══════════════════════════════════════════
# 텔레그램 알림
# ══════════════════════════════════════════

def send_dart_alerts(tier1_items: list[dict], universe_hits: list[dict]):
    """tier1 공시 + 유니버스 관련 공시 텔레그램 알림

    NOTE(2026-03-20): 개별 즉시발송 비활성화.
    저녁 통합 리포트(send_evening_summary.py)에 DART 섹션이 포함되어
    이중 발송 방지. --alert 플래그도 어떤 BAT에서도 사용하지 않음.
    자동매매 재개 후 필요하면 다시 활성화.
    """
    logger.info("DART 텔레그램 개별 발송 비활성화 (저녁 통합 리포트로 통합)")
    return


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def crawl_dart(days: int = 1, send_alert: bool = False) -> dict:
    """DART 공시 크롤링 메인 함수 (외부에서 import 가능)

    Args:
        days: 최근 N일 공시 수집 (기본 당일)
        send_alert: 텔레그램 알림 전송 여부

    Returns:
        전체 결과 dict (dart_disclosures.json과 동일 구조)
    """
    api_key = _get_api_key()
    universe_codes = build_universe_codes()

    end_de = datetime.now().strftime("%Y%m%d")
    bgn_de = (datetime.now() - timedelta(days=days - 1)).strftime("%Y%m%d")

    logger.info("DART 공시 크롤링: %s ~ %s (유니버스 %d종목)", bgn_de, end_de, len(universe_codes))

    # API 호출
    raw_items = fetch_dart_disclosures(api_key, bgn_de, end_de)
    logger.info("DART 총 %d건 수집", len(raw_items))

    if not raw_items:
        output = {
            "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "period": f"{bgn_de}~{end_de}",
            "total_count": 0,
            "tier1_count": 0,
            "tier2_count": 0,
            "tier3_count": 0,
            "universe_hit_count": 0,
            "tier1": [],
            "tier2": [],
            "tier3": [],
            "universe_hits": [],
        }
        _save_output(output)
        return output

    # 분류 + 유니버스 매칭
    result = process_disclosures(raw_items, universe_codes)

    output = {
        "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "period": f"{bgn_de}~{end_de}",
        "total_count": len(raw_items),
        "tier1_count": len(result["tier1"]),
        "tier2_count": len(result["tier2"]),
        "tier3_count": len(result["tier3"]),
        "universe_hit_count": len(result["universe_hits"]),
        "tier1": result["tier1"],
        "tier2": result["tier2"][:30],  # 최대 30건
        "tier3": result["tier3"][:20],  # 최대 20건
        "universe_hits": result["universe_hits"],
    }

    _save_output(output)

    # 통계 출력
    logger.info("── DART 공시 분류 ──")
    logger.info("  tier1 즉시: %d건", output["tier1_count"])
    logger.info("  tier2 중요: %d건", output["tier2_count"])
    logger.info("  tier3 참고: %d건", output["tier3_count"])
    logger.info("  유니버스 관련: %d건", output["universe_hit_count"])

    if result["tier1"]:
        logger.info("── tier1 즉시 공시 ──")
        for t in result["tier1"][:10]:
            logger.info("  🔴 %s [%s] — %s",
                        t["corp_name"], t["keyword"], t["report_nm"][:50])

    if result["universe_hits"]:
        logger.info("── 유니버스 관련 공시 ──")
        for u in result["universe_hits"][:10]:
            logger.info("  🎯 %s(%s) [%s] — %s",
                        u["corp_name"], u["stock_code"], u["tier"],
                        u["report_nm"][:40])

    # 텔레그램 알림
    if send_alert and (result["tier1"] or result["universe_hits"]):
        logger.info("텔레그램 알림 전송 중...")
        send_dart_alerts(result["tier1"], result["universe_hits"])

    return output


def _save_output(output: dict):
    """결과 JSON 저장"""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("저장: %s", OUT_PATH)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DART 전자공시 크롤링")
    parser.add_argument("--days", type=int, default=1,
                        help="최근 N일 공시 수집 (기본: 당일)")
    parser.add_argument("--alert", action="store_true",
                        help="텔레그램 알림 전송")
    args = parser.parse_args()

    print("=" * 60)
    print("  DART 전자공시 크롤링 — 뉴스 선행 감지 Step 1")
    print("=" * 60)

    crawl_dart(days=args.days, send_alert=args.alert)


if __name__ == "__main__":
    main()
