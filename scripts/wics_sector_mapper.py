"""WICS(Wise Industry Classification Standard) 슈퍼섹터 매핑 엔진.

wiseindex.com 공개 API로 전 상장종목의 3층 섹터 매핑 테이블을 생성한다.
10개 대분류(슈퍼섹터) → 27개 중분류(섹터) → 종목 매핑.

출력:
  - data/sector_rotation/wics_mapping.csv   : 전종목 3층 매핑
  - data/sector_rotation/wics_etf_bridge.csv: WICS 중분류 ↔ TIGER ETF 브릿지

사용법:
  python scripts/wics_sector_mapper.py              # 매핑 갱신 (전일 기준)
  python scripts/wics_sector_mapper.py --date 20260219  # 특정 날짜
  python scripts/wics_sector_mapper.py --dry-run    # 금융(4개 중분류)만 테스트
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# WICS 10대 슈퍼섹터
# ─────────────────────────────────────────────

WICS_SUPERSECTORS = {
    "G10": "에너지",
    "G15": "소재",
    "G20": "산업재",
    "G25": "경기소비재",
    "G30": "필수소비재",
    "G35": "건강관리",
    "G40": "금융",
    "G45": "IT",
    "G50": "커뮤니케이션서비스",
    "G55": "유틸리티",
}

# ─────────────────────────────────────────────
# WICS 27개 중분류 코드 (wiseindex.com API 검증 완료)
# (중분류코드): (슈퍼섹터코드, 슈퍼섹터명, 중분류명)
# ─────────────────────────────────────────────

WICS_SUBSECTORS = {
    "G1010": ("G10", "에너지", "에너지"),
    "G1510": ("G15", "소재", "소재"),
    "G2010": ("G20", "산업재", "자본재"),
    "G2020": ("G20", "산업재", "상업서비스와공급품"),
    "G2030": ("G20", "산업재", "운송"),
    "G2510": ("G25", "경기소비재", "자동차와부품"),
    "G2520": ("G25", "경기소비재", "내구소비재와의류"),
    "G2530": ("G25", "경기소비재", "호텔,레스토랑,레저등"),
    "G2550": ("G25", "경기소비재", "소매(유통)"),
    "G2560": ("G25", "경기소비재", "교육서비스"),
    "G3010": ("G30", "필수소비재", "식품과기본식료품소매"),
    "G3020": ("G30", "필수소비재", "식품,음료,담배"),
    "G3030": ("G30", "필수소비재", "가정용품과개인용품"),
    "G3510": ("G35", "건강관리", "건강관리장비와서비스"),
    "G3520": ("G35", "건강관리", "제약과생물공학"),
    "G4010": ("G40", "금융", "은행"),
    "G4020": ("G40", "금융", "증권"),
    "G4030": ("G40", "금융", "다각화된금융"),
    "G4040": ("G40", "금융", "보험"),
    "G4050": ("G40", "금융", "부동산"),
    "G4510": ("G45", "IT", "소프트웨어와서비스"),
    "G4520": ("G45", "IT", "기술하드웨어와장비"),
    "G4530": ("G45", "IT", "반도체와반도체장비"),
    "G4540": ("G45", "IT", "디스플레이"),
    "G5010": ("G50", "커뮤니케이션서비스", "전기통신서비스"),
    "G5020": ("G50", "커뮤니케이션서비스", "미디어와엔터테인먼트"),
    "G5510": ("G55", "유틸리티", "유틸리티"),
}

WICS_API_URL = (
    "https://www.wiseindex.com/Index/GetIndexComponets"
    "?ceil_yn=0&dt={date}&sec_cd={code}"
)

# ─────────────────────────────────────────────
# WICS 중분류 → TIGER ETF 브릿지
# ─────────────────────────────────────────────

WICS_ETF_BRIDGE = {
    # 금융 슈퍼섹터 (릴레이 핵심)
    "은행": ("091220", "TIGER 은행"),
    "증권": ("157500", "TIGER 증권"),
    "보험": ("140710", "TIGER 보험"),
    # IT 슈퍼섹터
    "반도체와반도체장비": ("091230", "TIGER 반도체"),
    "소프트웨어와서비스": ("157490", "TIGER 소프트웨어"),
    # 산업재
    "자본재": ("139220", "TIGER 200 건설"),
    # 소재
    "소재": ("139240", "TIGER 200 철강소재"),
    # 에너지
    "에너지": ("139250", "TIGER 200 에너지화학"),
    # 건강관리
    "건강관리장비와서비스": ("143860", "TIGER 헬스케어"),
    "제약과생물공학": ("364970", "TIGER 바이오TOP10"),
    # 커뮤니케이션서비스
    "미디어와엔터테인먼트": ("228810", "TIGER 미디어컨텐츠"),
    # 경기소비재
    "자동차와부품": ("138540", "TIGER 현대차그룹플러스"),
}


# ─────────────────────────────────────────────
# WICS API 호출 (중분류 단위)
# ─────────────────────────────────────────────

def fetch_wics_mapping(date: str | None = None, dry_run: bool = False) -> pd.DataFrame:
    """WICS API에서 전 종목 섹터 매핑을 중분류 단위로 가져온다.

    Args:
        date: YYYYMMDD 형식. None이면 전 영업일 추정.
        dry_run: True면 금융(G40) 중분류 4개만 호출.

    Returns:
        DataFrame[stock_code, stock_name, subsector_code, sector_name,
                  super_sector_code, super_sector_name, market_cap, sector_weight]
    """
    if date is None:
        # 오늘 데이터가 없을 수 있으므로 전일 사용
        yesterday = datetime.now() - timedelta(days=1)
        # 주말 보정
        if yesterday.weekday() == 6:  # 일요일
            yesterday -= timedelta(days=2)
        elif yesterday.weekday() == 5:  # 토요일
            yesterday -= timedelta(days=1)
        date = yesterday.strftime("%Y%m%d")

    # dry-run: 금융 중분류만
    if dry_run:
        codes = {k: v for k, v in WICS_SUBSECTORS.items() if k.startswith("G40")}
    else:
        codes = WICS_SUBSECTORS

    all_rows = []
    logger.info("WICS 매핑 조회 시작 (기준일: %s, %d개 중분류)", date, len(codes))

    for sub_code, (sup_code, sup_name, sector_name) in codes.items():
        url = WICS_API_URL.format(date=date, code=sub_code)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("[WICS] %s(%s) 호출 실패: %s", sub_code, sector_name, e)
            time.sleep(2)
            continue

        items = data.get("list", [])
        if not items:
            logger.warning("[WICS] %s(%s): 데이터 없음", sub_code, sector_name)
            time.sleep(2)
            continue

        for item in items:
            all_rows.append({
                "stock_code": item.get("CMP_CD", ""),
                "stock_name": item.get("CMP_KOR", ""),
                "subsector_code": sub_code,
                "sector_name": sector_name,
                "super_sector_code": sup_code,
                "super_sector_name": sup_name,
                "market_cap": item.get("MKT_VAL", 0),
                "sector_weight": item.get("WGT", 0),
            })

        logger.info("  %s %s(%s): %d종목", sup_name, sector_name, sub_code, len(items))
        time.sleep(1.5)  # rate limit 존중

    if not all_rows:
        logger.error("WICS 매핑 데이터가 비어있음")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # 중복 제거 (동일 종목이 여러 중분류에 나올 가능성은 낮지만 안전장치)
    df = df.drop_duplicates(subset=["stock_code"], keep="first")
    df = df.sort_values(
        ["super_sector_code", "subsector_code", "market_cap"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────

def save_mapping(df: pd.DataFrame) -> Path:
    """wics_mapping.csv 저장."""
    out_path = DATA_DIR / "wics_mapping.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("WICS 매핑 저장: %s (%d종목)", out_path, len(df))
    return out_path


def save_etf_bridge() -> Path:
    """wics_etf_bridge.csv 저장 (정적 매핑)."""
    rows = []
    for sector_name, (etf_code, etf_name) in WICS_ETF_BRIDGE.items():
        # 해당 중분류의 슈퍼섹터 찾기
        sup_code = sup_name = ""
        for sub_code, (sc, sn, sname) in WICS_SUBSECTORS.items():
            if sname == sector_name:
                sup_code, sup_name = sc, sn
                break

        rows.append({
            "wics_sector": sector_name,
            "super_sector_code": sup_code,
            "super_sector_name": sup_name,
            "etf_code": etf_code,
            "etf_name": etf_name,
        })

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "wics_etf_bridge.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("WICS-ETF 브릿지 저장: %s (%d매핑)", out_path, len(df))
    return out_path


# ─────────────────────────────────────────────
# 슈퍼섹터 요약 출력
# ─────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    """슈퍼섹터별 종목 수 + 중분류 리스트 출력."""
    print(f"\n{'=' * 60}")
    print(f"  WICS 슈퍼섹터 매핑 요약 ({len(df)}종목)")
    print(f"{'=' * 60}")

    for code, super_name in WICS_SUPERSECTORS.items():
        sub = df[df["super_sector_code"] == code]
        if sub.empty:
            continue

        sectors = sub["sector_name"].unique()

        print(f"\n  [{super_name}] ({code}) — {len(sub)}종목, {len(sectors)}개 중분류")
        for s in sorted(sectors):
            etf_mark = " ← ETF" if s in WICS_ETF_BRIDGE else ""
            count = len(sub[sub["sector_name"] == s])
            print(f"    · {s} ({count}종목){etf_mark}")

    # 릴레이 감지 가능 슈퍼섹터
    print(f"\n{'─' * 60}")
    print("  릴레이 감지 대상 (ETF 있는 형제 섹터 2개+ 보유):")
    for code, super_name in WICS_SUPERSECTORS.items():
        sub = df[df["super_sector_code"] == code]
        sectors_with_etf = [s for s in sub["sector_name"].unique() if s in WICS_ETF_BRIDGE]
        if len(sectors_with_etf) >= 2:
            print(f"    {super_name}: {' / '.join(sectors_with_etf)}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WICS 슈퍼섹터 매핑 엔진")
    parser.add_argument("--date", type=str, default=None,
                        help="기준일 (YYYYMMDD, 기본=전일)")
    parser.add_argument("--dry-run", action="store_true",
                        help="금융(G40) 중분류만 테스트 호출")
    args = parser.parse_args()

    # 1. WICS API 호출 (중분류 단위)
    df = fetch_wics_mapping(date=args.date, dry_run=args.dry_run)
    if df.empty:
        logger.error("WICS 매핑 실패 — 종료")
        sys.exit(1)

    # 2. 저장
    save_mapping(df)
    save_etf_bridge()

    # 3. 요약 출력
    print_summary(df)

    print(f"\n완료! {len(df)}종목 매핑됨")


if __name__ == "__main__":
    main()
