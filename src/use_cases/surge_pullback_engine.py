#!/usr/bin/env python3
"""
상한가 눌림목 분할매수 엔진 v1.0
═══════════════════════════════════════════════

2층 구조:
  Layer 1: v10c 큐레이션 93종목 (백테스트 검증, 고정)
  Layer 2: 일일 자동 발굴 (전 종목 스캔 → 품질 필터)

백테스트 최적 파라미터 (2025-11~2026-05, 6개월):
  - 급등 기준: 15%+
  - 눌림 기준: 피크 대비 -10%
  - 감시 기간: 3 거래일
  - 최소 주가: 10,000원
  - 최소 거래대금: 10억원/일
  - v10c 결과: 91.7% 승률, +170% 수익 (5천만 시드)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WATCHLIST_PATH = PROJECT_ROOT / "data" / "surge_pullback_watchlist.json"
SIGNAL_PATH = PROJECT_ROOT / "data" / "surge_pullback_signals.json"

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
    # AI데이터센터
    "307950": ["AI데이터센터"], "004710": ["액침냉각"],
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
    "pullback_pct": 10.0,         # 눌림 기준 (피크 대비 %)
    "watch_days": 3,              # 감시 기간 (거래일)
    "min_price": 10_000,          # 최소 주가 (원)
    "min_trading_value": 1_000_000_000,  # 최소 거래대금 (10억원)
    "capital": 50_000_000,        # 시드 (원)
    "max_position_pct": 0.10,     # 종목당 최대 비중 (10%)
    "fee_rate": 0.00315,          # 수수료+세금+슬리피지
    "holding_days": 20,           # 최대 보유일
    "layer2_sector_required": False,  # Layer2에 섹터 매칭 필수 여부
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
        logger.info("Engine 초기화: Layer1 %d종목, 감시%d일, 급등%.0f%%→눌림%.0f%%",
                     len(self.layer1_set), self.config["watch_days"],
                     self.config["surge_threshold"], self.config["pullback_pct"])

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
        """워치리스트 엔트리 생성"""
        return {
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
        }

    # ──────────────────────────────────────────
    # 5. 피크 업데이트 + 눌림 체크
    # ──────────────────────────────────────────
    def update_and_check(self, watchlist: dict, date_str: str) -> list[dict]:
        """
        워치리스트 내 종목들의 최신 가격 업데이트 → 눌림 시그널 체크.
        parquet 기반 (pykrx 폴백).
        반환: 매수 시그널 리스트
        """
        fmt_date = date_str.replace("-", "")
        target_date = pd.Timestamp(fmt_date)
        signals = []
        watch_days_limit = self.config["watch_days"]
        pullback_threshold = self.config["pullback_pct"]

        active = [e for e in watchlist["entries"] if e["status"] == "watching"]
        if not active:
            return signals

        logger.info("감시 종목 %d건 피크/눌림 체크", len(active))
        raw_dir = PROJECT_ROOT / "data" / "raw"

        for entry in active:
            ticker = entry["ticker"]
            try:
                # parquet에서 당일 데이터 조회
                pf = raw_dir / f"{ticker}.parquet"
                if not pf.exists():
                    logger.debug("  %s parquet 없음 — 스킵", ticker)
                    continue

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

                if target_date not in df.index:
                    continue

                row = df.loc[target_date]
                today_high = int(row.get("high", 0))
                today_close = int(row.get("close", 0))

                if today_close == 0:
                    continue

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

                # 감시일 증가
                entry["watch_day"] += 1

                # ── 눌림 시그널 판정 ──
                if pullback <= -pullback_threshold:
                    entry["status"] = "signal"
                    entry["signal_date"] = date_str
                    signals.append({
                        "ticker": ticker,
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
                    })
                    logger.info("  ★ BUY SIGNAL: %s %s (피크%d→현재%d, %.1f%% 눌림)",
                               ticker, entry["name"], peak, today_close, pullback)

                # ── 감시 만료 ──
                elif entry["watch_day"] > watch_days_limit:
                    entry["status"] = "expired"
                    logger.info("  EXPIRE: %s %s (%d일 경과, 눌림 %.1f%%)",
                               ticker, entry["name"],
                               entry["watch_day"], entry.get("pullback_from_peak", 0))

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

        # ── 결과 요약 ──
        active = [e for e in watchlist["entries"] if e["status"] == "watching"]
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
                f"매수시그널: {len(signals)}건 | "
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
