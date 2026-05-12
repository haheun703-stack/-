#!/usr/bin/env python3
"""
상한가 눌림목 분할매수 엔진 v2.0
═══════════════════════════════════════════════

2층 구조:
  Layer 1: v10c 큐레이션 93종목 (백테스트 검증, 고정)
  Layer 2: 일일 자동 발굴 (전 종목 스캔 → 품질 필터)

v2.0 변경 (2026-05-12):
  - 연속급등(모멘텀) 시그널: 편입 후 추가 +10% 상승 시 즉시 시그널
    → "눌림 없이 급등" 패턴 포착 (가온전선/한솔테크닉스류)
  - 감시 기간 연장: 3→5일 (급등>25%면 7일)
  - 동적 눌림 임계값: 급등폭 비례 (15~20%→-8%, 20~30%→-7%, 30%+→-6%)
  - 수급 stale 차단: 데이터 7일+ 오래되면 시그널 차단 (기존: 통과)
  - 만료 후 성과 추적: expired 종목 5일간 추적 → 놓친 기회 학습

v2.0.1 변경 (2026-05-12):
  - 수급 stale 날짜 계산 버그 수정: int(YYYYMMDD) 뺄셈 → datetime 사용
    (월 경계 넘을 때 71일 등 오류값 방지)
  - stale 기본값 5→7일 (주말+공휴일 고려)

v1.2 변경:
  - 보유비중 프로파일 추가: KIS API 외국인 보유비중 + DB 누적 순매수 결합
  - 수급 탄력성(supply elasticity) 점수 계산

v1.1 변경:
  - 수급 확인 필터 추가: 외국인/기관/금투/연기금 순매수 확인

파라미터:
  - 급등 기준: 15%+
  - 눌림 기준: 동적 (급등폭 비례 6~8%)
  - 감시 기간: 5~7 거래일 (동적)
  - 모멘텀 기준: 편입 후 추가 +10%
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import sqlite3

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WATCHLIST_PATH = PROJECT_ROOT / "data" / "surge_pullback_watchlist.json"
SIGNAL_PATH = PROJECT_ROOT / "data" / "surge_pullback_signals.json"
INVESTOR_DB_PATH = PROJECT_ROOT / "data" / "investor_flow" / "investor_daily.db"
SECTOR_FIRE_DIR = PROJECT_ROOT / "data"

# 수급 확인 대상 투자자 (이 중 1명이라도 순매수면 통과)
SMART_INVESTORS = ["외국인", "기관합계", "금융투자", "연기금"]

# ═══════════════════════════════════════════════════════
# v1.3: 소분류 → 섹터발화 대분류 매핑
# 눌림목 종목의 세부 섹터를 섹터발화(FIRE) 15개 대분류로 연결
# ═══════════════════════════════════════════════════════
SECTOR_TO_FIRE = {
    # AI반도체 (FIRE 대분류)
    "반도체": "AI반도체",
    "AI데이터센터": "AI반도체",
    # 전력기기
    "전기전선": "전력기기",
    "광통신": "전력기기",
    "변압기": "전력기기",
    "원전가스": "전력기기",
    # 자동차
    "자동차": "자동차",
    # 로봇자동화
    "로봇": "로봇자동화",
    # 2차전지
    "2차전지": "2차전지",
    # 건설인프라
    "건설": "건설인프라",
    # 조선해운
    "조선": "조선해운",
    # 방산
    "방산": "방산",
    # 바이오
    "바이오": "바이오",
    # 금융
    "증권": "금융",
    "은행": "금융",
    # 철강소재
    "철강": "철강소재",
    # 정유에너지
    "석유화학": "정유에너지",
    "태양광": "정유에너지",
    "풍력": "정유에너지",
    # 액침냉각
    "액침냉각": "액침냉각",
    # 화장품소비재
    "화장품": "화장품소비재",
    # IT플랫폼
    "게임": "IT플랫폼",
    "통신": "IT플랫폼",
    "사이버보안": "IT플랫폼",
}

# ═══════════════════════════════════════════════════════
# Layer 1: v10c 큐레이션 93종목 (백테스트 91.7% 승률)
# ═══════════════════════════════════════════════════════
LAYER1_UNIVERSE = {
    "반도체": [
        "005930", "000660", "042700", "058470", "039030", "403870",
        "036930", "240810", "095340", "047050", "079370", "357780", "141080",
    ],
    "건설": ["000720", "047040", "006360", "375500", "028050"],
    "로봇": ["454910", "277810", "090360", "012450", "058610", "108860"],
    "조선": ["009540", "042660", "010140", "329180", "010620"],
    "태양광": ["009830", "010060"],
    "풍력": ["112610", "018000"],
    "석유화학": ["096770", "010950", "051910", "011170", "011790"],
    "원전가스": ["052690", "034020", "083650"],
    "방산": ["012450", "079550", "047810", "064350", "103140", "272110"],
    "2차전지": [
        "373220", "006400", "247540", "086520", "003670",
        "066570", "365340", "005070", "121600",
    ],
    "통신": ["017670", "030200", "032640"],
    "AI데이터센터": ["005930", "000660", "035420", "035720", "403870"],
    "항공": ["003490"],
    "우주스페이스": ["012450", "047810", "099190", "304100"],
    "전기전선": [
        "006260", "010120", "267260", "298040", "033100",
        "001440", "006340", "000500", "103590",
    ],
    "광통신": ["010170", "267260", "298040"],
    "변압기": ["267260", "298040", "033100", "103590"],
    "철강": ["005490", "004020", "001230"],
    "증권": ["006800", "016360", "039490", "005940"],
    "은행": ["105560", "055550", "086790", "316140"],
    "사이버보안": ["053800"],
    "화장품": ["090430", "051900", "192820"],
    "게임": ["036570", "251270", "259960", "293490", "263750"],
}

# ═══════════════════════════════════════════════════════
# 섹터 키워드 (Layer 2 자동 발굴 시 섹터 태깅)
# ═══════════════════════════════════════════════════════
SECTOR_KEYWORDS = {
    "반도체": ["반도체", "메모리", "웨이퍼", "칩", "팹리스", "파운드리", "테크", "실리콘",
              "세미콘", "옵틱스", "일렉트로", "머트리얼"],
    "건설": ["건설", "건축", "시멘트", "토목", "E&A", "엔지니어링", "산업개발"],
    "로봇": ["로봇", "로보틱스", "로보", "뉴로", "자동화"],
    "조선": ["조선", "해양", "중공업", "오션", "마린", "엔진", "선박"],
    "2차전지": ["에코프로", "배터리", "리튬", "양극", "음극", "전해", "분리막",
              "에너지솔루션", "퓨처엠", "퓨얼셀", "엔켐"],
    "방산": ["에어로", "방산", "국방", "디펜스", "무기", "항공우주", "시스템즈",
            "풍산", "넥스원", "로템"],
    "전기전선": ["전선", "전기", "일렉트릭", "케이블", "변압기", "전력"],
    "AI데이터센터": ["AI", "데이터센터", "클라우드", "GPU", "NPU"],
    "철강": ["철강", "스틸", "제철", "금속", "아연"],
    "태양광": ["태양광", "솔라", "솔루션"],
    "풍력": ["풍력", "윈드", "터빈"],
    "원전가스": ["원전", "원자력", "한전", "에너빌"],
    "석유화학": ["화학", "석유", "유화", "케미칼", "정유"],
    "통신": ["통신", "텔레콤", "광통신"],
    "우주스페이스": ["우주", "스페이스", "위성", "발사체"],
    "증권": ["증권", "투자", "금융지주"],
    "은행": ["금융", "은행", "지주"],
    "액침냉각": ["냉각", "쿨링", "냉방", "테크닉스"],
}

# ═══════════════════════════════════════════════════════
# v10d 확장 섹터 매핑 (Layer1에 없는 종목의 알려진 섹터)
# ═══════════════════════════════════════════════════════
KNOWN_SECTOR_MAP = {
    # 반도체 확장
    "080580": ["반도체"], "080220": ["반도체"], "046890": ["반도체"],
    "200710": ["반도체"], "330860": ["반도체"], "033640": ["반도체"],
    "432720": ["반도체"], "036540": ["반도체"], "084370": ["반도체"],
    "319660": ["반도체"], "254490": ["반도체"], "122640": ["반도체"],
    "399720": ["반도체"], "089030": ["반도체"], "067310": ["반도체"],
    "053610": ["반도체"], "074600": ["반도체"], "104830": ["반도체"],
    "000990": ["반도체"], "089010": ["반도체"], "031980": ["반도체"],
    "030530": ["반도체"], "110990": ["반도체"], "003160": ["반도체"],
    "126730": ["반도체"], "084850": ["반도체"], "440110": ["반도체", "AI데이터센터"],
    "092190": ["반도체"], "007810": ["반도체"], "327260": ["반도체"],
    "078600": ["반도체"], "353200": ["반도체"], "007660": ["반도체"],
    "095610": ["반도체"], "222800": ["반도체"], "356860": ["반도체"],
    "195870": ["반도체"], "161580": ["반도체"], "101490": ["반도체"],
    "036810": ["반도체"], "322310": ["반도체"], "317330": ["반도체"],
    "064290": ["반도체"], "093370": ["반도체"], "059090": ["반도체"],
    "092870": ["반도체"], "086390": ["반도체"], "090460": ["반도체"],
    "394280": ["반도체"], "218410": ["반도체", "방산"], "108320": ["반도체"],
    "323350": ["반도체"], "429270": ["반도체"], "396270": ["반도체"],
    "076610": ["반도체"], "320000": ["반도체"], "321260": ["반도체"],
    "265520": ["반도체"], "061090": ["반도체"],
    # 건설 확장
    "038500": ["건설"], "002780": ["건설"], "452280": ["건설"],
    "267270": ["건설"], "045100": ["건설"], "013580": ["건설"],
    "005960": ["건설"], "003070": ["건설"], "294870": ["건설"],
    "010780": ["건설"],
    # 로봇 확장
    "138360": ["로봇"], "090710": ["로봇"], "098460": ["로봇"],
    "348340": ["로봇"], "056080": ["로봇"], "232680": ["로봇"],
    "459510": ["로봇"], "117730": ["로봇"], "475400": ["로봇"],
    "108490": ["로봇"], "466100": ["로봇"], "484810": ["로봇"],
    "388720": ["로봇"], "079900": ["로봇"], "319400": ["로봇"],
    "048770": ["로봇"],
    # 조선 확장
    "322000": ["조선"], "097230": ["조선"], "443060": ["조선"],
    "082740": ["조선"],
    # 방산 확장
    "010820": ["방산"], "214430": ["방산"], "073490": ["방산"],
    "005810": ["방산"], "272210": ["방산"], "032820": ["방산"],
    "347700": ["방산"], "488900": ["방산"], "289930": ["방산"],
    "474610": ["방산"], "037460": ["방산"], "361390": ["방산"],
    "484590": ["방산"], "484870": ["방산"], "474170": ["방산", "우주스페이스"],
    "065450": ["방산"], "095270": ["방산"], "448710": ["방산"],
    "005870": ["방산"], "065170": ["방산"], "215090": ["방산", "사이버보안"],
    "221840": ["방산", "항공"], "024740": ["방산"], "014970": ["방산"],
    "020760": ["방산"],
    # 2차전지 확장
    "348370": ["2차전지"], "336370": ["2차전지"], "383310": ["2차전지"],
    "278280": ["2차전지"], "243840": ["2차전지"], "137400": ["2차전지"],
    "307180": ["2차전지"], "126340": ["2차전지"], "006110": ["2차전지"],
    "089980": ["2차전지"], "259630": ["2차전지"], "047310": ["2차전지"],
    "382900": ["2차전지"], "025900": ["2차전지"], "452200": ["2차전지"],
    # AI데이터센터 / 액침냉각
    "307950": ["AI데이터센터"], "004710": ["액침냉각"], "053080": ["액침냉각"],
    # 자동차
    "012330": ["자동차"], "005380": ["자동차"], "000270": ["자동차"],
    "018880": ["자동차"], "161390": ["자동차"],
    # 눌림목 종목 섹터 보강
    "002020": ["화장품"], "052330": ["반도체"],  # 코오롱→화장품소비재, 코텍→반도체
    "000540": ["전기전선"],  # 대한광통신→전력기기
    "078890": ["반도체"],  # 덕산테코피아→AI반도체
    "199800": ["바이오"],  # 툴젠→바이오
    "028050": ["건설"],  # 삼성E&A
    "138080": ["태양광"],  # 오이솔루션
    # 전기전선 확장
    "012200": ["전기전선"], "417200": ["전기전선"], "322180": ["전기전선"],
    "009470": ["전기전선"], "062040": ["전기전선"], "060370": ["전기전선"],
    "017510": ["전기전선"],
    # 우주 확장
    "478340": ["우주스페이스"], "462350": ["우주스페이스"], "451760": ["우주스페이스"],
    "189300": ["우주스페이스"], "098120": ["우주스페이스"],
    "211270": ["우주스페이스"], "065680": ["우주스페이스"],
    # 기타
    "336260": ["태양광"], "130660": ["원전가스"],
    "006650": ["석유화학"], "024060": ["석유화학"], "005950": ["석유화학"],
    "011780": ["석유화학"],
    "092790": ["철강"], "001430": ["철강"],
    "100790": ["증권"], "001510": ["증권"], "003530": ["증권"], "003470": ["증권"],
    "192820": ["화장품"], "044820": ["화장품"],
}

# ═══════════════════════════════════════════════════════
# 기본 설정
# ═══════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    "surge_threshold": 15.0,      # 급등 기준 (%)
    "pullback_pct": 10.0,         # 눌림 기준 (피크 대비 %) — v2.0: 동적 임계값 기본값
    "watch_days": 5,              # v2.0: 감시 기간 3→5일 (급등>25%면 7일)
    "min_price": 10_000,          # 최소 주가 (원)
    "min_trading_value": 1_000_000_000,  # 최소 거래대금 (10억원)
    "capital": 50_000_000,        # 시드 (원)
    "max_position_pct": 0.10,     # 종목당 최대 비중 (10%)
    "fee_rate": 0.00315,          # 수수료+세금+슬리피지
    "holding_days": 20,           # 최대 보유일
    "layer2_sector_required": False,  # Layer2에 섹터 매칭 필수 여부
    "supply_check_enabled": True,     # 수급 확인 필터 활성화 (v1.1)
    "supply_lookback_days": 3,        # 수급 확인 기간 (최근 N거래일)
    "ownership_profile_enabled": True,  # 보유비중 프로파일 활성화 (v1.2)
    # v2.0 신규 설정
    "momentum_signal_enabled": True,   # 연속급등 감지 (눌림 없이 추가 +10% 시 즉시 시그널)
    "momentum_threshold": 10.0,        # 연속급등 임계값 (편입 이후 추가 상승 %)
    "dynamic_pullback_enabled": True,  # 급등폭 비례 동적 눌림 임계값
    "supply_stale_block": True,        # 수급 데이터 stale 시 시그널 차단 (v1.1은 통과)
    "supply_stale_max_days": 7,        # 수급 데이터 허용 최대 갭 (달력일, 주말+공휴일 고려)
    "expired_tracking_days": 5,        # 만료 후 성과 추적 기간
}


def _build_layer1_set() -> tuple[set, dict]:
    """Layer1 유니버스 → (전체 티커 set, 티커→섹터 매핑)"""
    all_tickers = set()
    ticker_sectors = {}
    for sector, tickers in LAYER1_UNIVERSE.items():
        for t in tickers:
            all_tickers.add(t)
            if t not in ticker_sectors:
                ticker_sectors[t] = []
            if sector not in ticker_sectors[t]:
                ticker_sectors[t].append(sector)
    return all_tickers, ticker_sectors


class SurgePullbackEngine:
    """
    상한가 눌림목 분할매수 엔진

    사용법:
        engine = SurgePullbackEngine()
        result = engine.run_daily("20260508")
    """

    def __init__(self, config: dict | None = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.layer1_set, self.layer1_sectors = _build_layer1_set()
        self._sector_fire_cache: dict | None = None
        logger.info("Engine 초기화: Layer1 %d종목, 감시%d일, 급등%.0f%%→눌림%.0f%%",
                     len(self.layer1_set), self.config["watch_days"],
                     self.config["surge_threshold"], self.config["pullback_pct"])

    # ──────────────────────────────────────────
    # 0-A. 섹터발화 연동 (v1.3)
    # ──────────────────────────────────────────
    def load_sector_fire(self, date_str: str = "") -> dict:
        """최신 sector_fire JSON 로드 → {섹터명: {fire_score, fire_grade, fgn_5d, inst_5d}}"""
        if self._sector_fire_cache:
            return self._sector_fire_cache

        # 날짜 지정 또는 최신 파일 자동 탐색
        if date_str:
            target = SECTOR_FIRE_DIR / f"sector_fire_{date_str.replace('-', '')}.json"
            candidates = [target] if target.exists() else []
        else:
            candidates = sorted(SECTOR_FIRE_DIR.glob("sector_fire_*.json"), reverse=True)

        if not candidates:
            logger.debug("섹터발화 데이터 없음")
            return {}

        fire_path = candidates[0]
        try:
            with open(fire_path, encoding="utf-8") as f:
                data = json.load(f)
            sectors = data.get("sectors", [])
            result = {}
            for s in sectors:
                result[s["sector"]] = {
                    "fire_score": s.get("fire_score", 0),
                    "fire_grade": s.get("fire_grade", "D"),
                    "fgn_5d": s.get("fgn_5d", 0),
                    "inst_5d": s.get("inst_5d", 0),
                    "rsi_avg": s.get("rsi_avg", 50),
                }
            self._sector_fire_cache = result
            fire_date = data.get("date", fire_path.stem[-8:])
            logger.info("섹터발화 로드: %s (%d섹터)", fire_date, len(result))
            return result
        except Exception as e:
            logger.warning("섹터발화 로드 실패: %s", e)
            return {}

    def get_sector_fire_for_stock(self, ticker: str, sectors: list[str] = None) -> dict:
        """종목의 섹터발화 상태 조회.

        Returns: {fire_sector, fire_grade, fire_score, fgn_5d, inst_5d}
        """
        fire_data = self.load_sector_fire()
        if not fire_data:
            return {"fire_sector": "", "fire_grade": "?", "fire_score": 0,
                    "fgn_5d": 0, "inst_5d": 0}

        # 종목의 소분류 섹터 결정
        if not sectors:
            sectors = KNOWN_SECTOR_MAP.get(ticker, [])
            if not sectors and ticker in self.layer1_sectors:
                sectors = self.layer1_sectors[ticker]

        # 소분류 → 대분류(FIRE) 매핑 후 가장 높은 발화 섹터 선택
        best = {"fire_sector": "", "fire_grade": "D", "fire_score": 0,
                "fgn_5d": 0, "inst_5d": 0}
        for sub_sector in sectors:
            fire_sector = SECTOR_TO_FIRE.get(sub_sector, "")
            if not fire_sector:
                continue
            info = fire_data.get(fire_sector, {})
            score = info.get("fire_score", 0)
            # 매핑된 섹터 중 점수 가장 높은 것 선택 (0점이어도 매핑은 유지)
            if score > best["fire_score"] or (not best["fire_sector"] and fire_sector):
                best = {
                    "fire_sector": fire_sector,
                    "fire_grade": info.get("fire_grade", "D"),
                    "fire_score": score,
                    "fgn_5d": info.get("fgn_5d", 0),
                    "inst_5d": info.get("inst_5d", 0),
                }
        return best

    # ─────────────────────────────────────��────
    # 0-B. ETF 전파 모델 연동 (v1.4)
    # ──────────────────────────────────────────
    _etf_transmission_cache: dict | None = None
    _etf_sector_cache: dict | None = None  # 섹터별 KR ETF 기대수익

    # US ETF → 한국 소분류 섹터 매핑 (섹터 레벨 간접 전파용)
    _US_ETF_TO_SUBSECTORS = {
        "SOXX": ["반도체", "AI데이터센터"],
        "LIT": ["2차전지"],
        "URA": ["원전가스", "전기전선"],
        "BOTZ": ["로봇"],
        "ITA": ["방산"],
        "ICLN": ["태양광", "풍력"],
        "XLF": ["증권", "은행"],
        "XLE": ["석유화학"],
        "XLK": ["AI데이터센터", "통신", "게임"],
    }

    def _get_etf_transmission_boost(self, ticker: str, sectors: list = None) -> dict:
        """ETF 전파 모델에서 해당 종목의 기대수익률 조회.

        1단계: 종목이 ETF 구성종목 TOP10에 직접 있으면 → 정확한 비중 기반 수익
        2단계: 없으면 → 해당 종목의 섹터로 간접 전파 (KR ETF 기대수익의 70%)

        Returns: {"expected_ret_pct": 7.4, "source_etf": "SOXX", ...} or empty dict
        """
        # 캐시 로드 (세션당 1회)
        if self._etf_transmission_cache is None:
            etf_path = PROJECT_ROOT / "data" / "etf_transmission_result.json"
            if etf_path.exists():
                try:
                    data = json.loads(etf_path.read_text(encoding="utf-8"))
                    # 1. ticker→결과 매핑 (직접 전파)
                    cache = {}
                    sector_cache = {}
                    for etf_key, trans in data.get("transmissions", {}).items():
                        for stock in trans.get("stocks", []):
                            t = stock["ticker"]
                            if t not in cache or stock["expected_ret_pct"] > cache[t]["expected_ret_pct"]:
                                cache[t] = {
                                    "expected_ret_pct": stock["expected_ret_pct"],
                                    "source_etf": etf_key,
                                    "kr_etf": trans.get("kr_etf", ""),
                                    "us_ret_1d_pct": trans.get("us_ret_1d_pct", 0),
                                    "type": "direct",
                                }
                        # 2. 섹터별 KR ETF 기대수익 (간접 전파 용)
                        kr_exp = trans.get("kr_etf_expected_pct", 0)
                        subsectors = self._US_ETF_TO_SUBSECTORS.get(etf_key, [])
                        for sub in subsectors:
                            if sub not in sector_cache or kr_exp > sector_cache[sub]["kr_etf_expected_pct"]:
                                sector_cache[sub] = {
                                    "kr_etf_expected_pct": kr_exp,
                                    "source_etf": etf_key,
                                    "kr_etf": trans.get("kr_etf", ""),
                                    "us_ret_1d_pct": trans.get("us_ret_1d_pct", 0),
                                }
                    self._etf_transmission_cache = cache
                    self._etf_sector_cache = sector_cache
                    logger.info("ETF전파 캐시 로드: 직접 %d종목, 섹터 %d개",
                               len(cache), len(sector_cache))
                except Exception as e:
                    logger.warning("ETF전파 데이터 로드 실패: %s", e)
                    self._etf_transmission_cache = {}
                    self._etf_sector_cache = {}
            else:
                self._etf_transmission_cache = {}
                self._etf_sector_cache = {}

        # 1단계: 직접 매핑 (ETF 구성종목)
        direct = self._etf_transmission_cache.get(ticker)
        if direct:
            return direct

        # 2단계: 섹터 간접 전파 (구성종목이 아닌 동일 섹터 종목)
        if not sectors:
            sectors = KNOWN_SECTOR_MAP.get(ticker, [])
            if not sectors and ticker in self.layer1_sectors:
                sectors = self.layer1_sectors[ticker]

        if sectors and self._etf_sector_cache:
            for sub_sector in sectors:
                sec_data = self._etf_sector_cache.get(sub_sector)
                if sec_data and sec_data["kr_etf_expected_pct"] != 0:
                    # 간접 전파: KR ETF 기대수익 × 0.7 (비구성종목 할인)
                    indirect_ret = sec_data["kr_etf_expected_pct"] * 0.7
                    return {
                        "expected_ret_pct": round(indirect_ret, 2),
                        "source_etf": sec_data["source_etf"],
                        "kr_etf": sec_data["kr_etf"],
                        "us_ret_1d_pct": sec_data["us_ret_1d_pct"],
                        "type": "sector_indirect",
                    }

        return {}

    # ──────────────────────────────────────────
    # 0. 수급 확인 (investor_daily.db 조회)
    # ──────────────────────────────────────────
    def check_supply_demand(self, ticker: str, date_str: str, lookback: int = 3) -> dict:
        """
        특정 종목의 최근 N일 투자자별 순매수 확인.
        SMART_INVESTORS 중 1주체 이상 순매수(net_val > 0)면 confirmed=True.

        반환: {
            confirmed: bool,
            buyers: ["외국인", ...],    # 순매수 주체 목록
            detail: {투자자: 순매수금액(억)},
            reason: "설명"
        }
        """
        result = {
            "confirmed": False,
            "buyers": [],
            "detail": {},
            "reason": "수급 데이터 없음",
        }

        if not INVESTOR_DB_PATH.exists():
            logger.warning("investor_daily.db 없음 — 수급 확인 스킵 (시그널 유지)")
            result["confirmed"] = True  # DB 없으면 기존 로직 유지
            result["reason"] = "DB 없음 (수급 필터 스킵)"
            return result

        fmt_date = date_str.replace("-", "")

        try:
            conn = sqlite3.connect(str(INVESTOR_DB_PATH))
            cur = conn.cursor()

            # 최근 lookback일의 투자자별 순매수 합산
            cur.execute("""
                SELECT investor, SUM(net_val) as total_net
                FROM investor_daily
                WHERE ticker = ? AND date <= ?
                  AND investor IN (?, ?, ?, ?)
                GROUP BY investor
                ORDER BY date DESC
                LIMIT ?
            """, (ticker, fmt_date, *SMART_INVESTORS, lookback * len(SMART_INVESTORS)))

            # 최근 N일 날짜를 구해서 합산
            cur.execute("""
                SELECT DISTINCT date FROM investor_daily
                WHERE ticker = ? AND date <= ?
                ORDER BY date DESC LIMIT ?
            """, (ticker, fmt_date, lookback))
            recent_dates = [r[0] for r in cur.fetchall()]

            if not recent_dates:
                conn.close()
                # v2.0: 수급 데이터 없으면 차단 (stale_block 설정 시)
                if self.config.get("supply_stale_block", True):
                    result["confirmed"] = False
                    result["reason"] = f"종목 {ticker} 수급 데이터 없음 — 시그널 차단"
                else:
                    result["confirmed"] = True
                    result["reason"] = f"종목 {ticker} 수급 데이터 없음 (필터 스킵)"
                return result

            # v2.0: 데이터 신선도 체크 — stale 데이터 시 차단
            latest_data_date = recent_dates[0]
            # v2.0.1: 달력일 정확 계산 (월 경계 넘을 때 int 뺄셈 오류 수정)
            try:
                dt_now = datetime.strptime(fmt_date, "%Y%m%d")
                dt_last = datetime.strptime(latest_data_date, "%Y%m%d")
                date_gap = (dt_now - dt_last).days
            except ValueError:
                date_gap = 999  # 파싱 실패 시 stale 처리
            stale_max = self.config.get("supply_stale_max_days", 7)
            if date_gap > stale_max:
                conn.close()
                if self.config.get("supply_stale_block", True):
                    result["confirmed"] = False
                    result["reason"] = (f"수급 데이터 {date_gap}일 오래됨 "
                                       f"({latest_data_date}) — 시그널 차단")
                else:
                    result["confirmed"] = True
                    result["reason"] = (f"수급 데이터 오래됨 "
                                       f"({latest_data_date}) — 필터 스킵")
                return result

            placeholders = ",".join(["?"] * len(recent_dates))
            inv_placeholders = ",".join(["?"] * len(SMART_INVESTORS))
            cur.execute(f"""
                SELECT investor, SUM(net_val) as total_net
                FROM investor_daily
                WHERE ticker = ? AND date IN ({placeholders})
                  AND investor IN ({inv_placeholders})
                GROUP BY investor
            """, (ticker, *recent_dates, *SMART_INVESTORS))

            rows = cur.fetchall()
            conn.close()

            buyers = []
            detail = {}
            for inv, net_val in rows:
                억 = round(net_val / 1e8, 1)
                detail[inv] = 억
                if net_val > 1e8:  # 최소 1억 이상 순매수만 인정
                    buyers.append(inv)

            result["detail"] = detail
            result["buyers"] = buyers

            if buyers:
                result["confirmed"] = True
                result["reason"] = f"{','.join(buyers)} 순매수 ({lookback}일)"
            else:
                result["confirmed"] = False
                result["reason"] = f"스마트머니 순매수 없음 ({lookback}일)"

        except Exception as e:
            logger.warning("수급 확인 오류 %s: %s — 시그널 유지", ticker, str(e)[:60])
            result["confirmed"] = True  # 오류 시 기존 로직 유지
            result["reason"] = f"DB 오류 (필터 스킵)"

        return result

    # ──────────────────────────────────────────
    # 0b. 보유비중 프로파일 (KIS API + DB 누적)
    # ──────────────────────────────────────────
    def calc_ownership_profile(self, ticker: str, date_str: str) -> dict:
        """
        종목의 투자자별 보유비중 프로파일 계산.

        데이터 소스:
          1) KIS API fetch_price → 외국인 보유비중/보유수량/발행주식수 (실시간)
          2) investor_daily.db → 투자자별 누적 순매수량/금액 (이력 기반)

        반환: {
            total_shares: int,        # 발행주식수
            frgn_pct: float,          # 외국인 보유비중 (%)
            frgn_shares: int,         # 외국인 보유 주수
            investors: {              # 투자자별 이력 프로파일
                "외국인": {cum_net_vol, cum_net_val_억, r5d, r10d, r30d, trend},
                "기관합계": ...,
            },
            elasticity_score: float,  # 수급 탄력성 (0~100)
            elasticity_grade: str,    # A/B/C/D 등급
            elasticity_reason: str,   # 판단 근거
        }
        """
        fmt_date = date_str.replace("-", "")
        profile = {
            "total_shares": 0,
            "frgn_pct": 0.0,
            "frgn_shares": 0,
            "investors": {},
            "elasticity_score": 0.0,
            "elasticity_grade": "N/A",
            "elasticity_reason": "데이터 없음",
        }

        # ── 1) KIS API: 외국인 보유비중 + 발행주식수 ──
        kis_data = self._fetch_kis_ownership(ticker)
        profile["total_shares"] = kis_data.get("total_shares", 0)
        profile["frgn_pct"] = kis_data.get("frgn_hldn_rto", 0.0)
        profile["frgn_shares"] = kis_data.get("frgn_hldn_qty", 0)

        # ── 2) DB: 투자자별 누적 순매수 + 기간별 추세 ──
        db_profile = self._calc_db_cumulative(ticker, fmt_date)
        profile["investors"] = db_profile

        # ── 3) 수급 탄력성 점수 계산 ──
        score, grade, reason = self._calc_elasticity(profile)
        profile["elasticity_score"] = score
        profile["elasticity_grade"] = grade
        profile["elasticity_reason"] = reason

        return profile

    def _fetch_kis_ownership(self, ticker: str) -> dict:
        """KIS API로 외국인 보유비중/발행주식수 조회 (선택적)."""
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")

            import mojito
            is_mock = os.getenv("MODEL") != "REAL"
            broker = mojito.KoreaInvestment(
                api_key=os.getenv("KIS_APP_KEY"),
                api_secret=os.getenv("KIS_APP_SECRET"),
                acc_no=os.getenv("KIS_ACC_NO"),
                mock=is_mock,
            )
            price_data = broker.fetch_price(ticker)
            output = price_data.get("output", {})

            total_shares = int(output.get("lstn_stcn", 0) or 0)
            frgn_qty = int(output.get("frgn_hldn_qty", 0) or 0)
            frgn_rto = float(output.get("frgn_hldn_rto", 0) or 0)

            # API가 비중을 0으로 반환하는 경우 → 직접 계산
            if frgn_rto == 0 and frgn_qty > 0 and total_shares > 0:
                frgn_rto = round(frgn_qty / total_shares * 100, 2)

            logger.info("  KIS 보유비중: %s 발행%s주 외국인%.1f%% (%s주)",
                        ticker, f"{total_shares:,}", frgn_rto, f"{frgn_qty:,}")
            return {
                "total_shares": total_shares,
                "frgn_hldn_qty": frgn_qty,
                "frgn_hldn_rto": frgn_rto,
            }
        except Exception as e:
            logger.warning("  KIS 보유비중 조회 실패 %s: %s", ticker, str(e)[:60])
            return {"total_shares": 0, "frgn_hldn_qty": 0, "frgn_hldn_rto": 0.0}

    def _calc_db_cumulative(self, ticker: str, fmt_date: str) -> dict:
        """investor_daily.db에서 투자자별 누적/기간별 순매수 계산."""
        investors = {}
        if not INVESTOR_DB_PATH.exists():
            return investors

        try:
            conn = sqlite3.connect(str(INVESTOR_DB_PATH))
            cur = conn.cursor()

            # 전체 투자자 종류
            all_investors = ["외국인", "기관합계", "기타법인", "개인"]

            for inv in all_investors:
                # 전체 누적 (DB 전 기간)
                cur.execute("""
                    SELECT SUM(net_vol), SUM(net_val), COUNT(DISTINCT date)
                    FROM investor_daily
                    WHERE ticker = ? AND investor = ? AND date <= ?
                """, (ticker, inv, fmt_date))
                row = cur.fetchone()
                cum_vol = row[0] or 0
                cum_val = row[1] or 0
                total_days = row[2] or 0

                # 최근 N일 순매수 (5일, 10일, 30일)
                periods = {"r5d": 5, "r10d": 10, "r30d": 30}
                period_data = {}

                for key, n in periods.items():
                    cur.execute(f"""
                        SELECT SUM(net_vol), SUM(net_val)
                        FROM investor_daily
                        WHERE ticker = ? AND investor = ? AND date IN (
                            SELECT DISTINCT date FROM investor_daily
                            WHERE ticker = ? AND date <= ?
                            ORDER BY date DESC LIMIT ?
                        )
                    """, (ticker, inv, ticker, fmt_date, n))
                    r = cur.fetchone()
                    period_data[key] = {
                        "net_vol": r[0] or 0,
                        "net_val_억": round((r[1] or 0) / 1e8, 1),
                    }

                # 추세 판정: 최근 5일 vs 최근 30일 방향
                r5 = period_data["r5d"]["net_vol"]
                r30 = period_data["r30d"]["net_vol"]
                if r5 > 0 and r30 > 0:
                    trend = "지속매수"
                elif r5 > 0 and r30 <= 0:
                    trend = "매수전환"
                elif r5 <= 0 and r30 > 0:
                    trend = "소폭매도"
                elif r5 < 0 and r30 < 0:
                    trend = "지속매도"
                else:
                    trend = "중립"

                investors[inv] = {
                    "cum_net_vol": cum_vol,
                    "cum_net_val_억": round(cum_val / 1e8, 1),
                    "total_days": total_days,
                    "r5d": period_data["r5d"],
                    "r10d": period_data["r10d"],
                    "r30d": period_data["r30d"],
                    "trend": trend,
                }

            conn.close()
        except Exception as e:
            logger.warning("DB 누적 계산 오류 %s: %s", ticker, str(e)[:60])

        return investors

    def _calc_elasticity(self, profile: dict) -> tuple[float, str, str]:
        """
        수급 탄력성 점수 계산.

        핵심 아이디어:
          - 외국인/기관 보유비중이 크면 → 기본 점수 높음 (안전판 존재)
          - 최근 소량 매도해도 누적 보유가 크면 → 한번 매수 시 반등 가능
          - 최근 매수 전환 시 → 보너스 점수

        점수 기준 (0~100):
          A등급 (80+): 탄력 매우 높음 — 적극 매수 고려
          B등급 (60~79): 탄력 양호 — 매수 가능
          C등급 (40~59): 보통 — 추가 확인 필요
          D등급 (0~39): 탄력 낮음 — 신중 접근
        """
        score = 0.0
        reasons = []

        frgn_pct = profile.get("frgn_pct", 0)
        investors = profile.get("investors", {})
        frgn = investors.get("외국인", {})
        inst = investors.get("기관합계", {})

        # ── 1) 외국인 보유비중 기본 점수 (0~30점) ──
        if frgn_pct >= 40:
            score += 30
            reasons.append(f"외국인{frgn_pct:.1f}%↑↑")
        elif frgn_pct >= 25:
            score += 22
            reasons.append(f"외국인{frgn_pct:.1f}%↑")
        elif frgn_pct >= 10:
            score += 15
            reasons.append(f"외국인{frgn_pct:.1f}%")
        elif frgn_pct > 0:
            score += 5
            reasons.append(f"외국인{frgn_pct:.1f}%↓")

        # ── 2) 외국인 누적 보유 추세 (0~25점) ──
        if frgn:
            r5_val = frgn.get("r5d", {}).get("net_val_억", 0)
            r30_val = frgn.get("r30d", {}).get("net_val_억", 0)
            cum_val = frgn.get("cum_net_val_억", 0)

            # 누적이 양수이고 최근 소폭 매도인 경우 → 탄력성 높음
            if cum_val > 0 and r5_val < 0 and abs(r5_val) < cum_val * 0.05:
                score += 25
                reasons.append(f"외국인 누적+{cum_val:.0f}억 소폭조정({r5_val:+.0f}억)")
            elif cum_val > 0 and r5_val > 0:
                score += 20
                reasons.append(f"외국인 누적+{cum_val:.0f}억 매수지속")
            elif cum_val > 0:
                score += 10
                reasons.append(f"외국인 누적+{cum_val:.0f}억")
            elif r5_val > 0:  # 최근 매수 전환
                score += 15
                reasons.append(f"외국인 매수전환({r5_val:+.0f}억)")

        # ── 3) 기관 보유 추세 (0~25점) ──
        if inst:
            r5_val = inst.get("r5d", {}).get("net_val_억", 0)
            r30_val = inst.get("r30d", {}).get("net_val_억", 0)
            cum_val = inst.get("cum_net_val_억", 0)

            if cum_val > 0 and r5_val < 0 and abs(r5_val) < cum_val * 0.05:
                score += 25
                reasons.append(f"기관 누적+{cum_val:.0f}억 소폭조정")
            elif cum_val > 0 and r5_val > 0:
                score += 20
                reasons.append(f"기관 누적+{cum_val:.0f}억 매수지속")
            elif cum_val > 0:
                score += 10
                reasons.append(f"기관 누적+{cum_val:.0f}억")
            elif r5_val > 0:
                score += 15
                reasons.append(f"기관 매수전환({r5_val:+.0f}억)")

        # ── 4) 동시 매수 보너스 (0~10점) ──
        if frgn and inst:
            f5 = frgn.get("r5d", {}).get("net_val_억", 0)
            i5 = inst.get("r5d", {}).get("net_val_억", 0)
            if f5 > 0 and i5 > 0:
                score += 10
                reasons.append("외국인+기관 동시매수")
            # 역발상: 개인이 대량 매도 중이면 → 스마트머니가 받는 것
            individual = investors.get("개인", {})
            if individual:
                p5 = individual.get("r5d", {}).get("net_val_억", 0)
                if p5 < -10 and (f5 > 0 or i5 > 0):
                    score += 10
                    reasons.append(f"개인투매({p5:.0f}억)→스마트머니흡수")

        # 점수 클램핑
        score = min(100, max(0, score))

        # 등급 판정
        if score >= 80:
            grade = "A"
        elif score >= 60:
            grade = "B"
        elif score >= 40:
            grade = "C"
        else:
            grade = "D"

        reason = " / ".join(reasons) if reasons else "데이터 부족"
        return round(score, 1), grade, reason

    # ──────────────────────────────────────────
    # 1. 급등주 발굴 (parquet 전 종목 스캔)
    # ──────────────────────────────────────────
    def discover_surges(self, date_str: str) -> list[dict]:
        """
        data/raw/*.parquet 전 종목 스캔 → 특정일 15%+ 급등 종목 추출.
        parquet가 없거나 오래되면 pykrx 개별 조회 폴백.

        반환: [{ticker, name, close, change_pct, volume, trading_value}, ...]
        """
        fmt_date = date_str.replace("-", "")
        target_date = pd.Timestamp(fmt_date)
        logger.info("급등주 스캔: %s (parquet 기반)", fmt_date)

        # 종목명 캐시
        name_cache = self._load_name_cache()

        raw_dir = PROJECT_ROOT / "data" / "raw"
        if not raw_dir.exists():
            logger.error("data/raw 디렉토리 없음")
            return []

        surges = []
        scanned = 0
        threshold = self.config["surge_threshold"]

        parquet_files = sorted(raw_dir.glob("*.parquet"))
        logger.info("parquet 파일: %d개", len(parquet_files))

        for pf in parquet_files:
            ticker = pf.stem
            try:
                df = pd.read_parquet(pf)
                df.index = pd.to_datetime(df.index)

                # 컬럼 정규화
                col_map = {}
                for c in df.columns:
                    cl = str(c).lower()
                    if cl in ("close", "종가"):
                        col_map[c] = "close"
                    elif cl in ("high", "고가"):
                        col_map[c] = "high"
                    elif cl in ("low", "저가"):
                        col_map[c] = "low"
                    elif cl in ("open", "시가"):
                        col_map[c] = "open"
                    elif cl in ("volume", "거래량"):
                        col_map[c] = "volume"
                    elif cl in ("trading_value", "거래대금"):
                        col_map[c] = "trading_value"
                if col_map:
                    df = df.rename(columns=col_map)

                if "close" not in df.columns:
                    continue

                # 타겟 날짜 데이터 확인
                if target_date not in df.index:
                    continue

                scanned += 1
                idx = df.index.get_loc(target_date)
                if idx < 1:
                    continue

                today_close = float(df.iloc[idx]["close"])
                prev_close = float(df.iloc[idx - 1]["close"])

                if prev_close <= 0 or today_close <= 0:
                    continue

                change_pct = (today_close / prev_close - 1) * 100

                if change_pct >= threshold:
                    today_high = float(df.iloc[idx].get("high", today_close))
                    today_vol = int(df.iloc[idx].get("volume", 0))
                    today_tv = int(df.iloc[idx].get("trading_value", 0))

                    # 거래대금이 0이면 종가×거래량으로 추정
                    if today_tv == 0 and today_vol > 0:
                        today_tv = int(today_close * today_vol)

                    name = name_cache.get(ticker, ticker)

                    surges.append({
                        "ticker": ticker,
                        "name": name,
                        "close": int(today_close),
                        "high": int(today_high),
                        "change_pct": round(change_pct, 2),
                        "volume": today_vol,
                        "trading_value": today_tv,
                    })

            except Exception as e:
                logger.debug("  %s 스캔 실패: %s", ticker, str(e)[:40])

        # 종목명 보강 (pykrx)
        surges = self._fill_names(surges, name_cache)

        logger.info("스캔 완료: %d종목 중 %d건 %.0f%%+ 급등",
                    scanned, len(surges), threshold)
        return surges

    def _load_name_cache(self) -> dict:
        """종목명 캐시 (universe.csv 또는 pykrx)"""
        cache = {}
        # universe.csv 시도
        upath = PROJECT_ROOT / "data" / "universe.csv"
        if upath.exists():
            try:
                import csv as csv_mod
                with open(upath, encoding="utf-8") as f:
                    reader = csv_mod.DictReader(f)
                    for row in reader:
                        t = row.get("ticker", "").strip()
                        n = row.get("name", "").strip()
                        if t and n:
                            cache[t] = n
                if cache:
                    return cache
            except Exception:
                pass
        return cache

    def _fill_names(self, surges: list[dict], cache: dict) -> list[dict]:
        """이름이 없는 종목에 pykrx로 이름 보강"""
        try:
            from pykrx import stock as pykrx_stock
        except ImportError:
            return surges

        for s in surges:
            if s["name"] == s["ticker"] or not s["name"]:
                try:
                    name = pykrx_stock.get_market_ticker_name(s["ticker"])
                    if name:
                        s["name"] = name
                        cache[s["ticker"]] = name
                    time.sleep(0.05)
                except Exception:
                    pass
        return surges

    # ──────────────────────────────────────────
    # 2. 품질 필터
    # ──────────────────────────────────────────
    def apply_quality_filters(self, candidates: list[dict]) -> list[dict]:
        """주가/거래대금 필터 → 잡주 제거"""
        min_price = self.config["min_price"]
        min_tv = self.config["min_trading_value"]

        filtered = []
        rejected = []
        for c in candidates:
            price = c.get("close", 0)
            tv = c.get("trading_value", 0)

            if price < min_price:
                rejected.append((c["ticker"], c["name"], f"주가 {price:,}원 < {min_price:,}"))
                continue
            if tv < min_tv:
                rejected.append((c["ticker"], c["name"],
                                f"거래대금 {tv/1e8:.1f}억 < {min_tv/1e8:.0f}억"))
                continue
            filtered.append(c)

        if rejected:
            logger.info("필터 탈락 %d건:", len(rejected))
            for t, n, reason in rejected[:10]:
                logger.info("  REJECT %s %s: %s", t, n, reason)

        logger.info("필터 통과: %d / %d건", len(filtered), len(candidates))
        return filtered

    # ──────────────────────────────────────────
    # 3. 섹터 태깅
    # ──────────────────────────────────────────
    def tag_sector(self, ticker: str, name: str) -> list[str]:
        """종목코드/종목명 → 섹터 매칭 (3단계 룩업)"""
        # 1) Layer1 유니버스
        if ticker in self.layer1_sectors:
            return self.layer1_sectors[ticker]

        # 2) v10d 확장 매핑 (알려진 종목)
        if ticker in KNOWN_SECTOR_MAP:
            return KNOWN_SECTOR_MAP[ticker]

        # 3) 키워드 매칭 (폴백)
        matched = []
        for sector, keywords in SECTOR_KEYWORDS.items():
            if any(kw in name for kw in keywords):
                matched.append(sector)
        return matched if matched else ["미분류"]

    # ──────────────────────────────────────────
    # 4. 워치리스트 관리
    # ──────────────────────────────────────────
    def load_watchlist(self) -> dict:
        """워치리스트 JSON 로드"""
        if WATCHLIST_PATH.exists():
            try:
                with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("워치리스트 로드 실패: %s", e)
        return {"updated": None, "entries": [], "history": []}

    def save_watchlist(self, watchlist: dict):
        """워치리스트 JSON 저장"""
        watchlist["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
            json.dump(watchlist, f, ensure_ascii=False, indent=2)
        logger.info("워치리스트 저장: %d건 활성", len(watchlist["entries"]))

    def _make_entry(self, stock: dict, layer: int, date_str: str) -> dict:
        """워치리스트 엔트리 생성 (편입 시 수급 기초데이터 포함)"""
        entry = {
            "ticker": stock["ticker"],
            "name": stock["name"],
            "layer": layer,
            "sectors": self.tag_sector(stock["ticker"], stock["name"]),
            "surge_date": date_str,
            "surge_pct": round(stock.get("change_pct", 0), 2),
            "surge_close": stock.get("close", 0),
            "peak_price": stock.get("high", stock.get("close", 0)),
            "peak_date": date_str,
            "latest_close": stock.get("close", 0),
            "pullback_from_peak": 0.0,
            "watch_day": 1,
            "status": "watching",
            "signal_date": None,
            "trading_value": stock.get("trading_value", 0),
            # v1.2: 수급 기초데이터 (편입 시점 스냅샷)
            "frgn_pct": 0.0,
            "frgn_cum": 0.0,
            "inst_cum": 0.0,
            "frgn_5d": 0.0,
            "inst_5d": 0.0,
            "total_shares": 0,
            # v1.3: 섹터발화 연동
            "fire_sector": "",
            "fire_grade": "?",
            "fire_score": 0,
        }

        # 보유비중 조회 (편입 시점)
        if self.config.get("ownership_profile_enabled", True):
            try:
                p = self.calc_ownership_profile(stock["ticker"], date_str)
                entry["frgn_pct"] = p.get("frgn_pct", 0)
                entry["total_shares"] = p.get("total_shares", 0)
                inv = p.get("investors", {})
                frgn = inv.get("외국인", {})
                inst = inv.get("기관합계", {})
                entry["frgn_cum"] = frgn.get("cum_net_val_억", 0)
                entry["inst_cum"] = inst.get("cum_net_val_억", 0)
                entry["frgn_5d"] = frgn.get("r5d", {}).get("net_val_억", 0)
                entry["inst_5d"] = inst.get("r5d", {}).get("net_val_억", 0)
                time.sleep(0.15)
            except Exception as e:
                logger.debug("편입 수급 조회 실패 %s: %s", stock["ticker"], str(e)[:40])

        # v1.3: 섹터발화 태깅
        fire_info = self.get_sector_fire_for_stock(stock["ticker"], entry["sectors"])
        entry["fire_sector"] = fire_info["fire_sector"]
        entry["fire_grade"] = fire_info["fire_grade"]
        entry["fire_score"] = fire_info["fire_score"]

        return entry

    # ──────────────────────────────────────────
    # 5. 피크 업데이트 + 눌림 체크 (v2.0)
    # ──────────────────────────────────────────
    def _calc_dynamic_pullback(self, surge_pct: float) -> float:
        """v2.0: 급등폭 비례 동적 눌림 임계값.

        강하게 급등한 종목일수록 눌림 기준을 낮춤 (약눌림 반등 포착).
        - 15~20% 급등 → -8% 눌림
        - 20~30% 급등 → -7% 눌림
        - 30%+ 급등  → -6% 눌림
        """
        if not self.config.get("dynamic_pullback_enabled", True):
            return self.config["pullback_pct"]

        if surge_pct >= 30:
            return 6.0
        elif surge_pct >= 20:
            return 7.0
        elif surge_pct >= 15:
            return 8.0
        return self.config["pullback_pct"]

    def _calc_dynamic_watch_days(self, surge_pct: float) -> int:
        """v2.0: 급등폭에 따른 동적 감시 기간.

        대형 급등(25%+)은 눌림에 시간이 더 걸리므로 감시 기간 연장.
        """
        base = self.config["watch_days"]  # 기본 5일
        if surge_pct >= 25:
            return max(base, 7)
        return base

    def _build_signal_dict(self, entry: dict, peak: int, today_close: int,
                           pullback: float, date_str: str,
                           supply: dict, ownership: dict,
                           fire_info: dict, etf_boost: dict,
                           signal_type: str = "pullback") -> dict:
        """시그널 딕셔너리 생성 (pullback / momentum 공통)."""
        return {
            "ticker": entry["ticker"],
            "name": entry["name"],
            "layer": entry["layer"],
            "sectors": entry["sectors"],
            "surge_date": entry["surge_date"],
            "surge_pct": entry["surge_pct"],
            "peak_price": peak,
            "entry_price": today_close,
            "pullback_pct": round(pullback, 2),
            "watch_day": entry["watch_day"],
            "signal_date": date_str,
            "signal_type": signal_type,  # v2.0: "pullback" 또는 "momentum"
            "supply_buyers": supply.get("buyers", []),
            "supply_detail": supply.get("detail", {}),
            "supply_reason": supply.get("reason", ""),
            # 보유비중 정보
            "frgn_pct": ownership.get("frgn_pct", 0),
            "total_shares": ownership.get("total_shares", 0),
            "elasticity_score": ownership.get("elasticity_score", 0),
            "elasticity_grade": ownership.get("elasticity_grade", "N/A"),
            "elasticity_reason": ownership.get("elasticity_reason", ""),
            "ownership_investors": ownership.get("investors", {}),
            # 섹터발화
            "fire_sector": fire_info.get("fire_sector", ""),
            "fire_grade": fire_info.get("fire_grade", ""),
            "fire_score": fire_info.get("fire_score", 0),
            # ETF 전파
            "etf_expected_ret": etf_boost.get("expected_ret_pct", 0),
            "etf_source": etf_boost.get("source_etf", ""),
        }

    def update_and_check(self, watchlist: dict, date_str: str) -> list[dict]:
        """
        워치리스트 내 종목들의 최신 가격 업데이트 → 눌림/모멘텀 시그널 체크.

        v2.0 변경:
          - 연속급등 감지: 편입 이후 추가 +10% 상승 시 "momentum" 시그널
          - 동적 감시 기간: 급등>25%면 7일, 그 외 5일
          - 동적 눌림 임계값: 급등폭에 비례 (15~20%→-8%, 20~30%→-7%, 30%+→-6%)
          - 수급 stale 차단: 데이터 5일+ 오래되면 시그널 차단

        parquet 기반 (pykrx 폴백).
        반환: 매수 시그널 리스트
        """
        fmt_date = date_str.replace("-", "")
        target_date = pd.Timestamp(fmt_date)
        signals = []

        active = [e for e in watchlist["entries"] if e["status"] == "watching"]
        if not active:
            return signals

        logger.info("감시 종목 %d건 피크/눌림/모멘텀 체크", len(active))
        raw_dir = PROJECT_ROOT / "data" / "raw"

        for entry in active:
            ticker = entry["ticker"]
            try:
                # parquet에서 당일 데이터 조회
                pf = raw_dir / f"{ticker}.parquet"
                price_available = False

                if pf.exists():
                    df = pd.read_parquet(pf)
                    df.index = pd.to_datetime(df.index)

                    # 컬럼 정규화
                    col_map = {}
                    for c in df.columns:
                        cl = str(c).lower()
                        if cl in ("close", "종가"): col_map[c] = "close"
                        elif cl in ("high", "고가"): col_map[c] = "high"
                        elif cl in ("low", "저가"): col_map[c] = "low"
                    if col_map:
                        df = df.rename(columns=col_map)

                    if target_date in df.index:
                        row = df.loc[target_date]
                        today_high = int(row.get("high", 0))
                        today_close = int(row.get("close", 0))

                        if today_close > 0:
                            price_available = True
                            # 피크 업데이트
                            if today_high > entry["peak_price"]:
                                entry["peak_price"] = today_high
                                entry["peak_date"] = date_str

                            # 눌림 계산
                            peak = entry["peak_price"]
                            pullback = 0.0
                            if peak > 0:
                                pullback = (today_close - peak) / peak * 100
                                entry["pullback_from_peak"] = round(pullback, 2)
                                entry["latest_close"] = today_close

                # v1.3.1: parquet 데이터 없어도 저장된 pullback으로 판정
                if not price_available:
                    pullback = entry.get("pullback_from_peak", 0.0)
                    if pullback == 0.0:
                        continue
                    peak = entry.get("peak_price", 0)
                    today_close = entry.get("latest_close", 0)
                    logger.info("  %s parquet 미갱신 — 저장값 사용 (%.1f%%)", entry["name"], pullback)

                # v1.2: 수급 기초데이터 갱신 (DB 기반, KIS API는 편입 시만)
                try:
                    db_inv = self._calc_db_cumulative(ticker, fmt_date)
                    frgn = db_inv.get("외국인", {})
                    inst = db_inv.get("기관합계", {})
                    entry["frgn_cum"] = frgn.get("cum_net_val_억", 0)
                    entry["inst_cum"] = inst.get("cum_net_val_억", 0)
                    entry["frgn_5d"] = frgn.get("r5d", {}).get("net_val_억", 0)
                    entry["inst_5d"] = inst.get("r5d", {}).get("net_val_억", 0)
                except Exception:
                    pass

                # 감시일 증가
                entry["watch_day"] += 1

                # v1.3: 섹터발화 갱신
                fire_info = self.get_sector_fire_for_stock(ticker, entry.get("sectors", []))
                entry["fire_sector"] = fire_info["fire_sector"]
                entry["fire_grade"] = fire_info["fire_grade"]
                entry["fire_score"] = fire_info["fire_score"]

                # v1.4: ETF 전파 모델
                etf_boost = self._get_etf_transmission_boost(ticker, entry.get("sectors", []))
                entry["etf_expected_ret"] = etf_boost.get("expected_ret_pct", 0)
                entry["etf_source"] = etf_boost.get("source_etf", "")

                # v2.0: 동적 감시 기간 (급등폭 기반)
                surge_pct = entry.get("surge_pct", 15.0)
                watch_days_limit = self._calc_dynamic_watch_days(surge_pct)

                # ═══════════════════════════════════════════
                # v2.0: 연속급등(모멘텀) 시그널 체크 — 눌림 전에 먼저 판정
                # 편입 이후 추가 +10% 상승 시 즉시 "momentum" 시그널
                # ═══════════════════════════════════════════
                if (self.config.get("momentum_signal_enabled", True)
                        and price_available and today_close > 0):
                    surge_close = entry.get("surge_close", 0)
                    momentum_threshold = self.config.get("momentum_threshold", 10.0)

                    if surge_close > 0:
                        gain_from_surge = (today_close - surge_close) / surge_close * 100
                        entry["gain_from_surge"] = round(gain_from_surge, 2)

                        if gain_from_surge >= momentum_threshold:
                            # 모멘텀 시그널: 눌림 없이 연속 상승
                            entry["status"] = "signal"
                            entry["signal_date"] = date_str

                            ownership = {}
                            if self.config.get("ownership_profile_enabled", True):
                                try:
                                    ownership = self.calc_ownership_profile(ticker, date_str)
                                    entry["ownership_profile"] = ownership
                                    time.sleep(0.15)
                                except Exception as e:
                                    logger.warning("  보유비중 프로파일 실패 %s: %s",
                                                   ticker, str(e)[:40])

                            sig = self._build_signal_dict(
                                entry, peak, today_close, pullback, date_str,
                                {"confirmed": True, "buyers": [],
                                 "detail": {}, "reason": "모멘텀 시그널 (수급 체크 생략)"},
                                ownership, fire_info, etf_boost,
                                signal_type="momentum",
                            )
                            signals.append(sig)
                            logger.info(
                                "  ★ MOMENTUM SIGNAL: %s %s "
                                "(급등종가%d→현재%d, +%.1f%% 추가상승)",
                                ticker, entry["name"],
                                surge_close, today_close, gain_from_surge,
                            )
                            continue  # 모멘텀 시그널 발생 시 눌림 체크 스킵

                # ═══════════════════════════════════════════
                # 눌림 시그널 판정 (v2.0: 동적 임계값 + 수급 stale 차단)
                # ═══════════════════════════════════════════

                # v2.0: 동적 눌림 임계값 (급등폭 비례)
                pullback_threshold = self._calc_dynamic_pullback(surge_pct)

                # 발화 A/B 등급: 눌림 기준 추가 완화
                effective_threshold = pullback_threshold
                if fire_info["fire_grade"] in ("A", "B") and fire_info["fire_score"] >= 40:
                    effective_threshold = pullback_threshold * 0.8
                    if pullback <= -effective_threshold and pullback > -pullback_threshold:
                        logger.info("  🔥 발화 가중치 적용: %s [%s %s등급] — 기준 %.1f%%→%.1f%%",
                                   entry["name"], fire_info["fire_sector"],
                                   fire_info["fire_grade"], pullback_threshold, effective_threshold)

                # v1.4: ETF 전파 모델 — US ETF 급등 시 임계값 절대 완화 (최대 3%p)
                etf_exp = etf_boost.get("expected_ret_pct", 0)
                if etf_exp >= 2.0:
                    if etf_exp >= 5.0:
                        etf_reduction = 3.0
                    elif etf_exp >= 3.0:
                        etf_reduction = 2.0
                    else:
                        etf_reduction = 1.0
                    effective_threshold = max(5.0, effective_threshold - etf_reduction)
                    logger.info("  📡 ETF전파 적용: %s [%s %+.1f%%] — 기준 -%dp → %.1f%%",
                               entry["name"], etf_boost.get("source_etf", "?"),
                               etf_exp, int(etf_reduction), effective_threshold)

                # 오차 허용 (3%)
                tolerance = 0.03
                trigger_threshold = effective_threshold * (1 - tolerance)

                if pullback <= -trigger_threshold:
                    # 수급 확인: 스마트머니 순매수 여부
                    if self.config.get("supply_check_enabled", True):
                        supply = self.check_supply_demand(
                            ticker, date_str,
                            lookback=self.config.get("supply_lookback_days", 3)
                        )
                    else:
                        supply = {"confirmed": True, "buyers": [], "detail": {}, "reason": "필터 비활성"}

                    if supply["confirmed"]:
                        entry["status"] = "signal"
                        entry["signal_date"] = date_str
                        entry["supply_check"] = supply

                        # 보유비중 프로파일 계산
                        ownership = {}
                        if self.config.get("ownership_profile_enabled", True):
                            try:
                                ownership = self.calc_ownership_profile(ticker, date_str)
                                entry["ownership_profile"] = ownership
                                time.sleep(0.15)
                            except Exception as e:
                                logger.warning("  보유비중 프로파일 실패 %s: %s", ticker, str(e)[:40])

                        sig = self._build_signal_dict(
                            entry, peak, today_close, pullback, date_str,
                            supply, ownership, fire_info, etf_boost,
                            signal_type="pullback",
                        )
                        signals.append(sig)
                        e_grade = ownership.get("elasticity_grade", "?")
                        e_score = ownership.get("elasticity_score", 0)
                        f_pct = ownership.get("frgn_pct", 0)
                        logger.info("  ★ BUY SIGNAL: %s %s (피크%d→현재%d, %.1f%% 눌림) "
                                   "[수급: %s] [탄력성: %s %.0f점, 외국인%.1f%%] "
                                   "[동적임계: %.1f%%]",
                                   ticker, entry["name"], peak, today_close, pullback,
                                   supply["reason"], e_grade, e_score, f_pct,
                                   effective_threshold)
                    else:
                        # 눌림은 도달했지만 수급 미확인 → 시그널 보류
                        entry["supply_check"] = supply
                        logger.info("  ⚠ PULLBACK OK but NO SUPPLY: %s %s (%.1f%% 눌림) — %s",
                                   ticker, entry["name"], pullback, supply["reason"])

                # ── 감시 만료 ──
                elif entry["watch_day"] > watch_days_limit:
                    entry["status"] = "expired"
                    entry["expired_date"] = date_str  # v2.0: 만료일 기록
                    logger.info("  EXPIRE: %s %s (%d일 경과, 눌림 %.1f%%, 감시한도 %d일)",
                               ticker, entry["name"],
                               entry["watch_day"], entry.get("pullback_from_peak", 0),
                               watch_days_limit)

            except Exception as e:
                logger.warning("  %s 업데이트 실패: %s", ticker, str(e)[:60])

        return signals

    # ──────────────────────────────────────────
    # 6. 만료/시그널 정리
    # ──────────────────────────────────────────
    def cleanup_watchlist(self, watchlist: dict) -> dict:
        """만료/시그널 발생 종목을 히스토리로 이동"""
        active = []
        for e in watchlist["entries"]:
            if e["status"] in ("expired", "signal"):
                watchlist.setdefault("history", []).append(e)
            else:
                active.append(e)
        watchlist["entries"] = active
        return watchlist

    # ──────────────────────────────────────────
    # 7. 메인 실행 루틴
    # ──────────────────────────────────────────
    def run_daily(self, date_str: str | None = None) -> dict:
        """
        매일 장마감 후 실행하는 메인 루틴.

        1. 오늘 급등주 발굴 (Layer 2)
        2. Layer 1 종목 중 오늘 급등 체크
        3. 품질 필터 적용
        4. 워치리스트 업데이트
        5. 피크/눌림 체크 → 매수 시그널
        6. 만료 정리

        Returns:
            {
                "date": str,
                "new_layer1": [...],
                "new_layer2": [...],
                "signals": [...],
                "active_watchlist": [...],
                "expired": [...],
                "summary": str,
            }
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        fmt_date = date_str.replace("-", "")

        logger.info("="*60)
        logger.info("상한가 눌림목 엔진 실행: %s", fmt_date)
        logger.info("="*60)

        # ── Step 1: 급등주 발굴 ──
        all_surges = self.discover_surges(fmt_date)

        # ── Step 2: Layer 분류 ──
        layer1_surges = []
        layer2_candidates = []
        for s in all_surges:
            if s["ticker"] in self.layer1_set:
                layer1_surges.append(s)
            else:
                layer2_candidates.append(s)

        logger.info("Layer1 급등: %d건, Layer2 후보: %d건",
                    len(layer1_surges), len(layer2_candidates))

        # ── Step 3: Layer2 품질 필터 ──
        layer2_filtered = self.apply_quality_filters(layer2_candidates)

        # ── Step 4: 워치리스트 업데이트 ──
        watchlist = self.load_watchlist()

        # 기존 워치리스트에 이미 있는 종목 중복 방지
        existing_tickers = {e["ticker"] for e in watchlist["entries"]
                          if e["status"] == "watching"}

        new_l1 = []
        for s in layer1_surges:
            if s["ticker"] not in existing_tickers:
                entry = self._make_entry(s, layer=1, date_str=fmt_date)
                watchlist["entries"].append(entry)
                new_l1.append(entry)
                existing_tickers.add(s["ticker"])

        new_l2 = []
        for s in layer2_filtered:
            if s["ticker"] not in existing_tickers:
                entry = self._make_entry(s, layer=2, date_str=fmt_date)
                watchlist["entries"].append(entry)
                new_l2.append(entry)
                existing_tickers.add(s["ticker"])

        logger.info("신규 편입: Layer1 %d건, Layer2 %d건", len(new_l1), len(new_l2))

        # ── Step 5: 기존 감시 종목 피크/눌림 체크 ──
        # (오늘 새로 편입된 종목은 제외 — 급등 당일은 눌림 없음)
        pre_existing = [e for e in watchlist["entries"]
                       if e["status"] == "watching" and e["surge_date"] != fmt_date]
        temp_wl = {"entries": pre_existing}
        signals = self.update_and_check(temp_wl, fmt_date)

        # 업데이트된 상태 반영
        updated_map = {e["ticker"]: e for e in temp_wl["entries"]}
        for i, e in enumerate(watchlist["entries"]):
            if e["ticker"] in updated_map:
                watchlist["entries"][i] = updated_map[e["ticker"]]

        # ── Step 6: 만료 정리 ──
        expired = [e for e in watchlist["entries"] if e["status"] == "expired"]
        watchlist = self.cleanup_watchlist(watchlist)

        # ── Step 7: 저장 ──
        self.save_watchlist(watchlist)

        # ── Step 8: 시그널 저장 ──
        if signals:
            self._save_signals(signals, fmt_date)

        # ── Step 9: 수익률 추적 ──
        try:
            self.update_performance(fmt_date)
        except Exception as e:
            logger.warning("수익률 추적 실패: %s", str(e)[:60])

        # ── Step 10 (v2.0): 만료 종목 후속 성과 추적 ──
        try:
            self._track_expired_performance(fmt_date)
        except Exception as e:
            logger.warning("만료 후 추적 실패: %s", str(e)[:60])

        # ── 결과 요약 ──
        active = [e for e in watchlist["entries"] if e["status"] == "watching"]
        # v2.0: 시그널 타입별 집계
        pullback_sigs = [s for s in signals if s.get("signal_type") == "pullback"]
        momentum_sigs = [s for s in signals if s.get("signal_type") == "momentum"]
        result = {
            "date": fmt_date,
            "new_layer1": new_l1,
            "new_layer2": new_l2,
            "signals": signals,
            "active_watchlist": active,
            "expired": expired,
            "summary": (
                f"[{fmt_date}] "
                f"급등발굴: L1={len(new_l1)} L2={len(new_l2)} | "
                f"매수시그널: {len(signals)}건 "
                f"(눌림{len(pullback_sigs)}/모멘텀{len(momentum_sigs)}) | "
                f"감시중: {len(active)}건 | "
                f"만료: {len(expired)}건"
            ),
        }

        logger.info(result["summary"])
        return result

    def _save_signals(self, signals: list[dict], date_str: str):
        """매수 시그널 JSON 저장 (기존 시그널에 append)"""
        existing = []
        if SIGNAL_PATH.exists():
            try:
                with open(SIGNAL_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                pass

        for s in signals:
            s["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        existing.extend(signals)

        # 최근 100건만 유지
        existing = existing[-100:]

        SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SIGNAL_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        logger.info("매수 시그널 %d건 저장 → %s", len(signals), SIGNAL_PATH)

    # ──────────────────────────────────────────
    # 8. 수익률 추적 (시그널 후 성과)
    # ──────────────────────────────────────────
    def update_performance(self, date_str: str | None = None):
        """
        과거 시그널들의 현재 수익률을 추적.
        시그널 발생 후 5일/10일/20일 수익률 기록.
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        fmt_date = date_str.replace("-", "")
        target_date = pd.Timestamp(fmt_date)

        if not SIGNAL_PATH.exists():
            return

        with open(SIGNAL_PATH, "r", encoding="utf-8") as f:
            signals = json.load(f)

        if not signals:
            return

        raw_dir = PROJECT_ROOT / "data" / "raw"
        perf_path = PROJECT_ROOT / "data" / "surge_pullback_performance.json"

        # 기존 성과 로드
        perf = {}
        if perf_path.exists():
            try:
                with open(perf_path, "r", encoding="utf-8") as f:
                    perf = json.load(f)
            except Exception:
                pass

        updated = 0
        for sig in signals:
            ticker = sig.get("ticker", "")
            entry = sig.get("entry_price", 0)
            sig_date = sig.get("signal_date", "")
            if not ticker or not entry or not sig_date:
                continue

            key = f"{ticker}_{sig_date}"
            if key not in perf:
                perf[key] = {
                    "ticker": ticker,
                    "name": sig.get("name", ""),
                    "layer": sig.get("layer", 2),
                    "signal_date": sig_date,
                    "entry_price": entry,
                    "surge_pct": sig.get("surge_pct", 0),
                    "status": "tracking",
                    "max_gain": 0.0,
                    "max_loss": 0.0,
                    "current_pct": 0.0,
                    "days_held": 0,
                    "daily_prices": {},
                }

            # parquet에서 현재가 조회
            pf = raw_dir / f"{ticker}.parquet"
            if not pf.exists():
                continue

            try:
                df = pd.read_parquet(pf)
                df.index = pd.to_datetime(df.index)

                # 컬럼 정규화
                col_map = {}
                for c in df.columns:
                    cl = str(c).lower()
                    if cl in ("close", "종가"):
                        col_map[c] = "close"
                if col_map:
                    df = df.rename(columns=col_map)

                if target_date not in df.index or "close" not in df.columns:
                    continue

                current_close = int(df.loc[target_date, "close"])
                if current_close == 0:
                    continue

                pct = (current_close - entry) / entry * 100
                p = perf[key]
                p["current_pct"] = round(pct, 2)
                p["current_price"] = current_close
                p["latest_date"] = fmt_date

                if pct > p.get("max_gain", 0):
                    p["max_gain"] = round(pct, 2)
                if pct < p.get("max_loss", 0):
                    p["max_loss"] = round(pct, 2)

                # 시그널 이후 거래일 계산
                sig_ts = pd.Timestamp(sig_date.replace("-", ""))
                mask = (df.index >= sig_ts) & (df.index <= target_date)
                p["days_held"] = int(mask.sum())

                # 일별 가격 기록 (최근 20일)
                daily = p.get("daily_prices", {})
                daily[fmt_date] = {"close": current_close, "pct": round(pct, 2)}
                # 최근 20일만 유지
                if len(daily) > 20:
                    sorted_keys = sorted(daily.keys())
                    for old_key in sorted_keys[:-20]:
                        del daily[old_key]
                p["daily_prices"] = daily

                updated += 1

            except Exception as e:
                logger.debug("수익률 추적 실패 %s: %s", ticker, str(e)[:40])

        # 통계 요약
        active = [p for p in perf.values() if p.get("status") == "tracking"]
        if active:
            wins = sum(1 for p in active if p.get("current_pct", 0) > 0)
            total = len(active)
            avg_pct = sum(p.get("current_pct", 0) for p in active) / total
            perf["_summary"] = {
                "date": fmt_date,
                "total_signals": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                "avg_return": round(avg_pct, 2),
                "best": max((p.get("current_pct", 0) for p in active), default=0),
                "worst": min((p.get("current_pct", 0) for p in active), default=0),
            }

        # 저장
        perf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(perf_path, "w", encoding="utf-8") as f:
            json.dump(perf, f, ensure_ascii=False, indent=2)

        if updated:
            logger.info("수익률 추적 업데이트: %d건", updated)

    # ──────────────────────────────────────────
    # 9 (v2.0). 만료 후 성과 추적 — 놓친 기회 학습
    # ──────────────────────────────────────────
    def _track_expired_performance(self, date_str: str):
        """
        만료된 종목의 이후 성과를 추적하여 '놓친 기회'를 기록.
        시그널 로직 개선을 위한 학습 데이터를 축적.
        """
        tracking_days = self.config.get("expired_tracking_days", 5)
        if tracking_days <= 0:
            return

        fmt_date = date_str.replace("-", "")
        target_date = pd.Timestamp(fmt_date)
        raw_dir = PROJECT_ROOT / "data" / "raw"

        # 워치리스트의 히스토리에서 만료 종목 조회
        watchlist = self.load_watchlist()
        history = watchlist.get("history", [])
        expired_entries = [e for e in history
                          if e.get("status") == "expired"
                          and e.get("expired_date")]

        if not expired_entries:
            return

        expired_perf_path = PROJECT_ROOT / "data" / "surge_pullback_expired_tracking.json"
        expired_perf = {}
        if expired_perf_path.exists():
            try:
                with open(expired_perf_path, "r", encoding="utf-8") as f:
                    expired_perf = json.load(f)
            except Exception:
                pass

        updated = 0
        missed_opportunities = 0

        for entry in expired_entries:
            ticker = entry["ticker"]
            expired_date = entry.get("expired_date", "")
            if not expired_date:
                continue

            key = f"{ticker}_{expired_date}"

            # 이미 추적 완료된 것은 스킵
            if key in expired_perf and expired_perf[key].get("tracking_complete"):
                continue

            # 만료 후 경과일 계산
            try:
                exp_ts = pd.Timestamp(expired_date.replace("-", ""))
                days_since = (target_date - exp_ts).days
            except Exception:
                continue

            if days_since < 0 or days_since > tracking_days * 2:  # 달력일 기준 2배까지
                if key in expired_perf:
                    expired_perf[key]["tracking_complete"] = True
                continue

            # parquet에서 현재가 조회
            pf = raw_dir / f"{ticker}.parquet"
            if not pf.exists():
                continue

            try:
                df = pd.read_parquet(pf)
                df.index = pd.to_datetime(df.index)

                col_map = {}
                for c in df.columns:
                    cl = str(c).lower()
                    if cl in ("close", "종가"):
                        col_map[c] = "close"
                if col_map:
                    df = df.rename(columns=col_map)

                if target_date not in df.index or "close" not in df.columns:
                    continue

                current_close = int(df.loc[target_date, "close"])
                if current_close <= 0:
                    continue

                expired_close = entry.get("latest_close", 0)
                if expired_close <= 0:
                    continue

                pct_after = (current_close - expired_close) / expired_close * 100

                if key not in expired_perf:
                    expired_perf[key] = {
                        "ticker": ticker,
                        "name": entry.get("name", ""),
                        "layer": entry.get("layer", 2),
                        "surge_pct": entry.get("surge_pct", 0),
                        "expired_date": expired_date,
                        "expired_close": expired_close,
                        "pullback_at_expiry": entry.get("pullback_from_peak", 0),
                        "tracking_complete": False,
                        "max_gain_after": 0.0,
                        "max_loss_after": 0.0,
                        "missed_opportunity": False,
                        "daily": {},
                    }

                p = expired_perf[key]
                p["daily"][fmt_date] = {
                    "close": current_close,
                    "pct": round(pct_after, 2),
                }

                if pct_after > p.get("max_gain_after", 0):
                    p["max_gain_after"] = round(pct_after, 2)
                if pct_after < p.get("max_loss_after", 0):
                    p["max_loss_after"] = round(pct_after, 2)

                # 놓친 기회: 만료 후 +10% 이상 상승
                if pct_after >= 10:
                    p["missed_opportunity"] = True
                    missed_opportunities += 1

                updated += 1

            except Exception as e:
                logger.debug("만료 추적 실패 %s: %s", ticker, str(e)[:40])

        if updated > 0:
            expired_perf_path.parent.mkdir(parents=True, exist_ok=True)
            with open(expired_perf_path, "w", encoding="utf-8") as f:
                json.dump(expired_perf, f, ensure_ascii=False, indent=2)

            if missed_opportunities > 0:
                logger.warning("  ⚠ 놓친 기회 %d건 감지 (만료 후 +10%% 이상 상승)",
                              missed_opportunities)
            logger.info("만료 후 추적: %d건 업데이트", updated)

    # ──────────────────────────────────────────
    # 유틸: 워치리스트 현황 출력
    # ──────────────────────────────────────────
    def print_status(self, watchlist: dict | None = None):
        """현재 워치리스트 상태 출력"""
        if watchlist is None:
            watchlist = self.load_watchlist()

        entries = watchlist.get("entries", [])
        active = [e for e in entries if e["status"] == "watching"]

        print(f"\n{'='*70}")
        print(f" 상한가 눌림목 워치리스트 ({watchlist.get('updated', 'N/A')})")
        print(f"{'='*70}")
        print(f" 활성 감시: {len(active)}건")
        print(f"{'─'*70}")

        if not active:
            print(" (감시 종목 없음)")
            return

        # Layer별 분류
        l1 = [e for e in active if e["layer"] == 1]
        l2 = [e for e in active if e["layer"] == 2]

        for label, items in [("Layer1 (큐레이션)", l1), ("Layer2 (발굴)", l2)]:
            if not items:
                continue
            print(f"\n ■ {label}: {len(items)}건")
            for e in sorted(items, key=lambda x: x.get("pullback_from_peak", 0)):
                pb = e.get("pullback_from_peak", 0)
                status = "🔴" if pb <= -8 else "🟡" if pb <= -5 else "🟢"
                sectors = ",".join(e.get("sectors", [])[:2])
                print(f"   {status} {e['ticker']} {e['name']:12s} "
                      f"급등{e['surge_pct']:+.1f}% "
                      f"피크→{pb:+.1f}% "
                      f"D{e['watch_day']} "
                      f"[{sectors}]")

        print(f"{'='*70}")
