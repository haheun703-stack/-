"""2026 Q1 실적 기반 밸류에이션 괴리 분석기

핵심 아이디어:
  영업이익 YoY 증가율 vs 주가 6M 수익률 → 괴리율
  괴리율 양수 = "실적 대비 덜 오른 종목" (상방 여력)

데이터 파이프라인:
  1. DART 2026 Q1 실적 → 캐시(financial_quarterly.json) 폴백
  2. pykrx PER/PBR/EPS 전종목 (1회 호출)
  3. stock_data_daily CSV → 3M/6M/1Y 수익률

출력:
  data/reports/valuation_gap_YYYYMMDD.html (인터랙티브 리포트)
  data/valuation_gap_YYYYMMDD.json (원시 데이터)

실행:
  python -u -X utf8 scripts/scan_valuation_gap.py
  python -u -X utf8 scripts/scan_valuation_gap.py --no-dart   # DART 스킵 (캐시만)
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VG] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("valuation_gap")

DATA_DIR = PROJECT_ROOT / "data"
STOCK_DAILY_DIR = PROJECT_ROOT / "stock_data_daily"
REPORT_DIR = DATA_DIR / "reports"
SECTOR_MAP_PATH = PROJECT_ROOT / "config" / "sector_fire_map.yaml"
FINANCIAL_CACHE_PATH = DATA_DIR / "v2_migration" / "financial_quarterly.json"
NAME_MAP_PATH = DATA_DIR / "universe" / "name_map.json"

# ═══════════════════════════════════════════════════
# 섹터 × 종목 매핑 (sector_fire_map.yaml 외 추가 섹터)
# ═══════════════════════════════════════════════════

EXTRA_SECTORS = {
    "원전가스": {
        "tickers": [
            "034020",  # 두산에너빌리티
            "051600",  # 한전KPS
            "052690",  # 한전기술
            "036460",  # 한국가스공사
            "009540",  # 한국전력
        ],
    },
    "태양광풍력": {
        "tickers": [
            "009830",  # 한화솔루션
            "322000",  # HD현대에너지솔루션
            "112610",  # 씨에스윈드
            "010060",  # OCI홀딩스
        ],
    },
    "AI데이터센터": {
        "tickers": [
            "267260",  # HD현대일렉트릭 (변압기+DC전력)
            "298040",  # 효성중공업 (변압기+DC전력)
            "030200",  # KT (IDC)
            "017670",  # SK텔레콤 (AI인프라)
            "032640",  # LG유플러스 (IDC)
        ],
    },
}


# ═══════════════════════════════════════════════════
# 1. 유니버스 구성
# ═══════════════════════════════════════════════════

def load_name_map() -> dict[str, str]:
    """ticker → 종목명 매핑 로드."""
    if NAME_MAP_PATH.exists():
        with open(NAME_MAP_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_sector_config() -> dict[str, list[str]]:
    """sector_fire_map.yaml + EXTRA_SECTORS → {섹터명: [ticker, ...]}."""
    import yaml

    sectors: dict[str, list[str]] = {}

    if SECTOR_MAP_PATH.exists():
        with open(SECTOR_MAP_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for sector_name, info in cfg.get("sectors", {}).items():
            tickers = info.get("tickers", [])
            if tickers:
                sectors[sector_name] = tickers

    for sector_name, info in EXTRA_SECTORS.items():
        sectors[sector_name] = info["tickers"]

    return sectors


def build_universe(sectors: dict[str, list[str]], name_map: dict[str, str]) -> list[dict]:
    """섹터별 종목 → 분석 대상 리스트."""
    seen = set()
    universe = []

    for sector, tickers in sectors.items():
        for ticker in tickers:
            if ticker in seen:
                continue
            seen.add(ticker)
            universe.append({
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "sector": sector,
            })

    logger.info("유니버스 구성: %d종목 / %d섹터", len(universe), len(sectors))
    return universe


# ═══════════════════════════════════════════════════
# 2. DART 실적 수집
# ═══════════════════════════════════════════════════

def collect_dart_q1(tickers: list[str], year: int = 2026) -> dict[str, dict]:
    """DART 2026 Q1 다중회사 조회 → {ticker: {op_income, net_income, revenue, prev_*}}."""
    try:
        from src.adapters.dart_adapter import DartAdapter
    except ImportError:
        logger.warning("DartAdapter import 실패 — DART 수집 스킵")
        return {}

    dart = DartAdapter()
    if not dart.is_available:
        logger.warning("DART API 키 미설정 — DART 수집 스킵")
        return {}

    result: dict[str, dict] = {}

    # 100개씩 배치 처리
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        logger.info("DART 배치 %d/%d (%d종목)", i // 100 + 1, (len(tickers) - 1) // 100 + 1, len(batch))

        df = dart.fetch_multi_financials(batch, year=year, reprt_code="11013")
        if df is None or len(df) == 0:
            logger.info("  → 2026 Q1 데이터 없음 (미공시 또는 API 에러)")
            continue

        # stock_code별 그룹핑
        for stock_code, group in df.groupby("stock_code"):
            ticker = str(stock_code).zfill(6)
            data = _extract_earnings_from_dart(group)
            if data:
                result[ticker] = data

    logger.info("DART 2026 Q1 수집: %d종목 성공", len(result))
    return result


def _extract_earnings_from_dart(group) -> dict | None:
    """DART 다중회사 응답 DataFrame 그룹에서 실적 추출."""
    import pandas as pd

    def _get_amount(acct_name: str, col: str = "thstrm_amount") -> float | None:
        # 연결(CFS) 우선
        for fs_div in ["CFS", "OFS"]:
            if "fs_div" in group.columns:
                rows = group[(group["account_nm"] == acct_name) & (group["fs_div"] == fs_div)]
            else:
                rows = group[group["account_nm"] == acct_name]

            if len(rows) > 0:
                val = rows.iloc[0].get(col, "")
                return _parse_amount(val)
        return None

    op_income = _get_amount("영업이익")
    net_income = _get_amount("당기순이익")
    revenue = _get_amount("매출액")

    if op_income is None and net_income is None and revenue is None:
        return None

    # 전년동기 (frmtrm_amount)
    prev_op = _get_amount("영업이익", "frmtrm_amount")
    prev_net = _get_amount("당기순이익", "frmtrm_amount")
    prev_rev = _get_amount("매출액", "frmtrm_amount")

    return {
        "source": "DART_2026Q1",
        "op_income": op_income,
        "net_income": net_income,
        "revenue": revenue,
        "prev_op_income": prev_op,
        "prev_net_income": prev_net,
        "prev_revenue": prev_rev,
        "annualize_factor": 4,  # Q1 → 연환산
    }


def _parse_amount(value) -> float | None:
    """금액 문자열 → float (쉼표 제거)."""
    import pandas as pd
    if pd.isna(value) or value == "" or value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════
# 3. 캐시 (financial_quarterly.json) 폴백
# ═══════════════════════════════════════════════════

def load_financial_cache() -> dict:
    """financial_quarterly.json 로드."""
    if not FINANCIAL_CACHE_PATH.exists():
        logger.warning("재무 캐시 없음: %s", FINANCIAL_CACHE_PATH)
        return {}
    with open(FINANCIAL_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_earnings_from_cache(ticker: str, cache: dict) -> dict | None:
    """캐시에서 최신 분기 실적 추출 (폴백)."""
    bs_data = cache.get("bs_data", {}).get(ticker, {})
    if not bs_data:
        return None

    # 최신 분기 순서로 탐색
    quarter_order = ["2025Q3", "2025Q2", "2025Q1", "2024Q4", "2024Q3"]
    annualize_map = {"Q1": 4, "Q2": 2, "Q3": 4 / 3, "Q4": 1}

    for q in quarter_order:
        qdata = bs_data.get(q)
        if qdata and qdata.get("op_income_cum") is not None:
            q_suffix = q[-2:]  # "Q3" etc.
            factor = annualize_map.get(q_suffix, 1)

            # 전년동기 탐색
            prev_year = str(int(q[:4]) - 1) + q[-2:]  # "2025Q3" → "2024Q3"
            prev_data = bs_data.get(prev_year, {})

            return {
                "source": f"CACHE_{q}",
                "op_income": qdata.get("op_income_cum"),
                "net_income": qdata.get("net_income_cum"),
                "revenue": qdata.get("revenue_cum"),
                "prev_op_income": prev_data.get("op_income_cum"),
                "prev_net_income": prev_data.get("net_income_cum"),
                "prev_revenue": prev_data.get("revenue_cum"),
                "annualize_factor": factor,
            }

    return None


def get_quality_from_cache(ticker: str, cache: dict) -> dict:
    """캐시에서 재무건전성 메트릭 추출."""
    quality = cache.get("quality", {}).get(ticker, {})
    return {
        "debt_ratio": quality.get("debt_ratio"),
        "roe_mean": quality.get("roe_mean"),
        "roe_stability": quality.get("roe_stability"),
        "fcf": quality.get("fcf"),
    }


# ═══════════════════════════════════════════════════
# 4. 시장 데이터 (parquet + universe.csv)
# ═══════════════════════════════════════════════════

PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
UNIVERSE_CSV = DATA_DIR / "universe.csv"
VPS_PRICES_PATH = DATA_DIR / "vps_close_prices.json"


def load_latest_prices(tickers: list[str]) -> dict[str, dict]:
    """최신 종가 로드. JSON 캐시 우선 → raw parquet 폴백.

    Returns: {ticker: {close, date, close_6m, close_3m, close_1y}}
    """
    # 1차: JSON 캐시 (로컬에서 VPS 데이터를 SCP로 가져온 경우)
    if VPS_PRICES_PATH.exists():
        with open(VPS_PRICES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        sample_date = next(iter(data.values()), {}).get("date", "?")
        logger.info("종가 JSON 로드: %d종목 (기준일: %s)", len(data), sample_date)
        return data

    # 2차: raw parquet 직접 읽기 (VPS 환경)
    if not RAW_DIR.exists():
        logger.warning("종가 데이터 없음: JSON도 raw parquet도 없음")
        return {}

    import pandas as pd

    result = {}
    target_6m = pd.Timestamp.now() - pd.DateOffset(months=6)
    target_3m = pd.Timestamp.now() - pd.DateOffset(months=3)
    target_1y = pd.Timestamp.now() - pd.DateOffset(years=1)

    ticker_set = set(tickers)
    loaded = 0

    for f in RAW_DIR.iterdir():
        if not f.suffix == ".parquet":
            continue
        ticker = f.stem
        if ticker_set and ticker not in ticker_set:
            continue
        try:
            df = pd.read_parquet(f)
            if "close" not in df.columns or len(df) < 5:
                continue
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            latest = float(df["close"].dropna().iloc[-1])
            latest_date = str(df.index[-1].date())

            mask_6m = df.index <= target_6m
            c6m = float(df.loc[mask_6m, "close"].iloc[-1]) if mask_6m.any() else None
            mask_3m = df.index <= target_3m
            c3m = float(df.loc[mask_3m, "close"].iloc[-1]) if mask_3m.any() else None
            mask_1y = df.index <= target_1y
            c1y = float(df.loc[mask_1y, "close"].iloc[-1]) if mask_1y.any() else None

            result[ticker] = {
                "close": latest,
                "date": latest_date,
                "close_6m": c6m,
                "close_3m": c3m,
                "close_1y": c1y,
            }
            loaded += 1
        except Exception as e:
            logger.debug("raw parquet 읽기 실패 %s: %s", ticker, e)

    if result:
        sample_date = next(iter(result.values()), {}).get("date", "?")
        logger.info("raw parquet 종가 로드: %d종목 (기준일: %s)", loaded, sample_date)
    return result


def load_universe_market_cap() -> dict[str, float]:
    """universe.csv → {ticker: market_cap}."""
    if not UNIVERSE_CSV.exists():
        logger.warning("universe.csv 없음")
        return {}
    import pandas as pd
    df = pd.read_csv(UNIVERSE_CSV, encoding="utf-8-sig", dtype={"ticker": str})
    df["ticker"] = df["ticker"].str.zfill(6)
    result = {}
    for _, row in df.iterrows():
        mc = row.get("market_cap", 0)
        if mc and mc > 0:
            result[row["ticker"]] = float(mc)
    logger.info("universe.csv 시총 로드: %d종목", len(result))
    return result


def load_parquet_data(ticker: str) -> dict:
    """processed parquet → close, fund_PER/EPS, 수익률."""
    import pandas as pd

    parquet_path = PROCESSED_DIR / f"{ticker}.parquet"
    if not parquet_path.exists():
        return {}

    try:
        df = pd.read_parquet(parquet_path)
        if len(df) < 5 or "close" not in df.columns:
            return {}

        closes = df["close"].dropna().values
        current = float(closes[-1])

        # 수익률
        def _ret(n_days: int) -> float | None:
            if len(closes) >= n_days + 1:
                past = float(closes[-(n_days + 1)])
                if past > 0:
                    return round((current / past - 1) * 100, 2)
            return None

        # trailing PER/EPS (마지막 비영값)
        trailing_per = 0.0
        trailing_eps = 0.0
        trailing_pbr = 0.0
        if "fund_PER" in df.columns:
            per_valid = df[df["fund_PER"] > 0]["fund_PER"]
            if len(per_valid) > 0:
                trailing_per = float(per_valid.iloc[-1])
        if "fund_EPS" in df.columns:
            eps_valid = df[df["fund_EPS"] > 0]["fund_EPS"]
            if len(eps_valid) > 0:
                trailing_eps = float(eps_valid.iloc[-1])
        if "fund_PBR" in df.columns:
            pbr_valid = df[df["fund_PBR"] > 0]["fund_PBR"]
            if len(pbr_valid) > 0:
                trailing_pbr = float(pbr_valid.iloc[-1])

        return {
            "close": current,
            "ret_3m": _ret(63),
            "ret_6m": _ret(126),
            "ret_1y": _ret(252),
            "trailing_per": trailing_per,
            "trailing_eps": trailing_eps,
            "trailing_pbr": trailing_pbr,
        }
    except Exception as e:
        logger.debug("parquet 로드 실패 %s: %s", ticker, e)
        return {}


# ═══════════════════════════════════════════════════
# 5. 주가 수익률 (stock_data_daily CSV — parquet 폴백)
# ═══════════════════════════════════════════════════

def load_price_returns(ticker: str, name: str) -> dict:
    """stock_data_daily CSV에서 3M/6M/1Y 수익률 + 현재가 + 시총."""
    import pandas as pd

    # 파일 찾기: {종목명}_{ticker}.csv
    pattern = str(STOCK_DAILY_DIR / f"*_{ticker}.csv")
    files = glob.glob(pattern)
    if not files:
        return {"close": None, "market_cap": None, "ret_3m": None, "ret_6m": None, "ret_1y": None}

    try:
        df = pd.read_csv(files[0], encoding="utf-8-sig")
        if "Close" not in df.columns or len(df) < 5:
            return {"close": None, "market_cap": None, "ret_3m": None, "ret_6m": None, "ret_1y": None}

        df = df.dropna(subset=["Close"])
        closes = df["Close"].values
        current = float(closes[-1])

        market_cap = None
        if "MarketCap" in df.columns:
            mc_vals = df["MarketCap"].dropna().values
            if len(mc_vals) > 0:
                market_cap = float(mc_vals[-1])

        def _ret(n_days: int) -> float | None:
            if len(closes) >= n_days + 1:
                past = float(closes[-(n_days + 1)])
                if past > 0:
                    return round((current / past - 1) * 100, 2)
            return None

        return {
            "close": current,
            "market_cap": market_cap,
            "ret_3m": _ret(63),
            "ret_6m": _ret(126),
            "ret_1y": _ret(252),
        }
    except Exception as e:
        logger.debug("CSV 로드 실패 %s: %s", ticker, e)
        return {"close": None, "market_cap": None, "ret_3m": None, "ret_6m": None, "ret_1y": None}


# ═══════════════════════════════════════════════════
# 6. 분석 로직
# ═══════════════════════════════════════════════════

def analyze_stock(
    ticker: str,
    name: str,
    sector: str,
    earnings: dict | None,
    price: dict,
    quality: dict,
) -> dict | None:
    """단일 종목 종합 분석."""
    if earnings is None:
        return None

    close = price.get("close")
    if not close or close <= 0:
        return None

    # ── 영업이익 YoY ──
    oi = earnings.get("op_income")
    prev_oi = earnings.get("prev_op_income")
    oi_yoy = None
    turnaround = False
    if oi is not None and prev_oi is not None and prev_oi != 0:
        oi_yoy = round((oi - prev_oi) / abs(prev_oi) * 100, 1)
        if prev_oi < 0 and oi > 0:
            turnaround = True

    # ── 매출액 YoY ──
    rev = earnings.get("revenue")
    prev_rev = earnings.get("prev_revenue")
    rev_yoy = None
    if rev is not None and prev_rev is not None and prev_rev != 0:
        rev_yoy = round((rev - prev_rev) / abs(prev_rev) * 100, 1)

    # ── EPS 계산 ──
    net = earnings.get("net_income")
    factor = earnings.get("annualize_factor", 4)
    market_cap = price.get("market_cap")

    shares = None
    if market_cap and close > 0:
        shares = int(market_cap / close)

    eps_q1 = None
    eps_annualized = None
    calc_per = None

    if net is not None and shares and shares > 0:
        eps_q1 = round(net / shares, 0)
        eps_annualized = eps_q1 * factor
        if eps_annualized > 0:
            calc_per = round(close / eps_annualized, 2)

    # ── trailing PER/PBR (parquet에서) ──
    trailing_per = price.get("trailing_per", 0) or 0
    trailing_pbr = price.get("trailing_pbr", 0) or 0
    trailing_eps = price.get("trailing_eps", 0) or 0

    # ── 주가 수익률 ──
    ret_3m = price.get("ret_3m")
    ret_6m = price.get("ret_6m")
    ret_1y = price.get("ret_1y")

    # ── 괴리율 (핵심) ──
    gap = None
    if oi_yoy is not None and ret_6m is not None:
        gap = round(oi_yoy - ret_6m, 1)

    # ── 재무건전성 ──
    debt_ratio = quality.get("debt_ratio")
    roe = quality.get("roe_mean")

    # ── 상방 여력 스코어 ──
    score = calc_upside_score(gap, oi_yoy, calc_per, trailing_per, debt_ratio, roe)
    grade = assign_grade(score)

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "grade": grade,
        "score": score,
        "source": earnings.get("source", "?"),
        # 실적
        "op_income": oi,
        "net_income": net,
        "revenue": rev,
        "oi_yoy": oi_yoy,
        "rev_yoy": rev_yoy,
        "turnaround": turnaround,
        # EPS & PER
        "eps_q1": eps_q1,
        "eps_annualized": eps_annualized,
        "calc_per": calc_per,
        "trailing_per": trailing_per if trailing_per > 0 else None,
        "trailing_pbr": trailing_pbr if trailing_pbr > 0 else None,
        "trailing_eps": trailing_eps if trailing_eps > 0 else None,
        # 주가
        "close": close,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "ret_1y": ret_1y,
        # 괴리율
        "gap": gap,
        # 재무건전성
        "debt_ratio": debt_ratio,
        "roe": roe,
    }


def calc_upside_score(
    gap: float | None,
    oi_yoy: float | None,
    calc_per: float | None,
    trailing_per: float | None,
    debt_ratio: float | None,
    roe: float | None,
) -> float:
    """상방 여력 스코어 (0~100)."""
    score = 0.0

    # 괴리율 (40점) — 핵심
    if gap is not None and gap > 0:
        score += min(gap * 0.8, 40)

    # 영업이익 성장률 (25점)
    if oi_yoy is not None and oi_yoy > 0:
        score += min(oi_yoy * 0.25, 25)

    # PER 할인 (15점) — 연환산 PER이 낮을수록 유리
    if calc_per is not None and calc_per > 0:
        if calc_per < 5:
            score += 15
        elif calc_per < 10:
            score += 12
        elif calc_per < 15:
            score += 8
        elif calc_per < 20:
            score += 4

    # 재무건전성 (10점) — 부채비율 낮을수록
    if debt_ratio is not None:
        health = max(0, 1 - debt_ratio)  # 0~1
        score += min(health * 15, 10)

    # ROE (10점) — 높을수록
    if roe is not None and roe > 0:
        score += min(roe * 60, 10)

    return round(min(score, 100), 1)


def assign_grade(score: float) -> str:
    if score >= 75:
        return "S"
    if score >= 55:
        return "A"
    if score >= 35:
        return "B"
    if score >= 15:
        return "C"
    return "D"


# ═══════════════════════════════════════════════════
# 7. HTML 리포트 생성
# ═══════════════════════════════════════════════════

def _fmt(val, fmt_type="num"):
    """숫자 포맷팅."""
    if val is None:
        return '<span class="na">-</span>'
    if fmt_type == "pct":
        cls = "up" if val > 0 else "down" if val < 0 else ""
        sign = "+" if val > 0 else ""
        return f'<span class="{cls}">{sign}{val:.1f}%</span>'
    if fmt_type == "per":
        if val <= 0:
            return '<span class="na">-</span>'
        return f"{val:.1f}"
    if fmt_type == "ratio":
        return f"{val:.1%}" if val is not None else "-"
    if fmt_type == "roe":
        if val is None:
            return "-"
        return f"{val:.1%}"
    if fmt_type == "won":
        if abs(val) >= 1e12:
            return f"{val/1e12:.1f}조"
        if abs(val) >= 1e8:
            return f"{val/1e8:.0f}억"
        return f"{val:,.0f}"
    if fmt_type == "eps":
        return f"{val:,.0f}원"
    return f"{val}"


def build_html_report(results: list[dict], sectors: dict[str, list[str]], date_str: str) -> str:
    """인터랙티브 HTML 리포트 생성."""

    # 섹터별 그룹
    sector_groups: dict[str, list[dict]] = {}
    for r in results:
        s = r["sector"]
        if s not in sector_groups:
            sector_groups[s] = []
        sector_groups[s].append(r)

    # 각 섹터 내 스코어 내림차순
    for s in sector_groups:
        sector_groups[s].sort(key=lambda x: x["score"], reverse=True)

    # 통계
    total = len(results)
    sa_count = sum(1 for r in results if r["grade"] in ("S", "A"))
    gaps = [r["gap"] for r in results if r["gap"] is not None]
    avg_gap = round(sum(gaps) / len(gaps), 1) if gaps else 0

    # DART 공시 vs 캐시
    dart_count = sum(1 for r in results if r["source"].startswith("DART"))
    cache_count = total - dart_count

    # TOP 10 (괴리율 내림차순)
    top10 = sorted([r for r in results if r["gap"] is not None], key=lambda x: x["gap"], reverse=True)[:10]

    # 턴어라운드 종목
    turnarounds = [r for r in results if r.get("turnaround")]

    # 섹터별 평균 괴리율
    sector_avg = {}
    for s, items in sector_groups.items():
        g = [r["gap"] for r in items if r["gap"] is not None]
        sector_avg[s] = round(sum(g) / len(g), 1) if g else 0
    sector_avg_sorted = sorted(sector_avg.items(), key=lambda x: x[1], reverse=True)

    # 섹터 목록 (정렬: 평균 괴리율 높은 순)
    sector_names = [s for s, _ in sector_avg_sorted if s in sector_groups]

    # 테이블 행 생성
    def _table_rows(items: list[dict]) -> str:
        rows = []
        for r in items:
            gap_cls = ""
            if r["gap"] is not None and r["gap"] >= 30:
                gap_cls = " gold-row"
            elif r["gap"] is not None and r["gap"] >= 15:
                gap_cls = " highlight-row"

            grade_cls = {"S": "grade-s", "A": "grade-a", "B": "grade-b", "C": "grade-c", "D": "grade-d"}.get(r["grade"], "")
            source_cls = "cache-source" if r["source"].startswith("CACHE") else ""
            turn_tag = ' <span class="turn-tag">턴어라운드</span>' if r.get("turnaround") else ""

            rows.append(f"""<tr class="{gap_cls}">
<td class="sticky-col">{r['name']}{turn_tag}</td>
<td>{r['ticker']}</td>
<td class="{grade_cls}">{r['grade']}</td>
<td><b>{r['score']}</b></td>
<td>{_fmt(r['oi_yoy'], 'pct')}</td>
<td>{_fmt(r['rev_yoy'], 'pct')}</td>
<td>{_fmt(r['ret_6m'], 'pct')}</td>
<td>{_fmt(r['gap'], 'pct')}</td>
<td>{_fmt(r['eps_q1'], 'eps')}</td>
<td>{_fmt(r['calc_per'], 'per')}</td>
<td>{_fmt(r['trailing_per'], 'per')}</td>
<td>{_fmt(r['op_income'], 'won')}</td>
<td>{_fmt(r.get('debt_ratio'), 'ratio')}</td>
<td>{_fmt(r.get('roe'), 'roe')}</td>
<td class="{source_cls}">{r['source']}</td>
</tr>""")
        return "\n".join(rows)

    # 섹터 탭 HTML
    tab_buttons = ['<a class="tab active" onclick="showTab(\'all\', this)">전체</a>']
    for s in sector_names:
        tab_buttons.append(f'<a class="tab" onclick="showTab(\'{s}\', this)">{s} ({len(sector_groups[s])})</a>')

    # 섹터별 테이블
    all_rows = _table_rows(sorted(results, key=lambda x: x["score"], reverse=True))
    sector_tables = [f'<div id="all" class="sector-table" style="display:block;">{_make_table(all_rows)}</div>']
    for s in sector_names:
        rows = _table_rows(sector_groups[s])
        sector_tables.append(f'<div id="{s}" class="sector-table" style="display:none;">{_make_table(rows)}</div>')

    # TOP 10 테이블
    top10_rows = _table_rows(top10)

    # 턴어라운드 테이블
    turn_rows = _table_rows(turnarounds) if turnarounds else "<tr><td colspan='15'>해당 종목 없음</td></tr>"

    # 섹터 바 차트 (CSS only)
    max_gap = max(abs(v) for _, v in sector_avg_sorted) if sector_avg_sorted else 1
    bar_items = []
    for s, v in sector_avg_sorted:
        width = abs(v) / max(max_gap, 1) * 80
        bar_cls = "bar-positive" if v >= 0 else "bar-negative"
        sign = "+" if v > 0 else ""
        bar_items.append(f"""<div class="bar-row">
<span class="bar-label">{s}</span>
<div class="bar-track"><div class="{bar_cls}" style="width:{width}%"></div></div>
<span class="bar-value">{sign}{v:.1f}%p</span>
</div>""")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>2026 Q1 밸류에이션 괴리 분석 — {date_str}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0e17; color:#c8d6e5; font-family:'Noto Sans KR','Malgun Gothic',sans-serif; padding:20px; }}
h1 {{ color:#e8eaed; font-size:1.5em; margin-bottom:5px; }}
.subtitle {{ color:#8b949e; font-size:0.9em; margin-bottom:20px; }}

/* 요약 카드 */
.cards {{ display:flex; gap:15px; margin-bottom:25px; flex-wrap:wrap; }}
.card {{ background:#111820; border:1px solid #1e2d3d; border-radius:10px; padding:18px 22px; min-width:180px; flex:1; }}
.card-label {{ color:#8b949e; font-size:0.8em; margin-bottom:4px; }}
.card-value {{ color:#e8eaed; font-size:1.8em; font-weight:700; }}
.card-value.up {{ color:#3fb950; }}

/* 탭 네비게이션 */
.tabs {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:18px; }}
.tab {{ background:#111820; border:1px solid #1e2d3d; border-radius:6px; padding:6px 14px; color:#8b949e; cursor:pointer; font-size:0.85em; text-decoration:none; }}
.tab:hover {{ background:#1a2332; color:#c8d6e5; }}
.tab.active {{ background:#1f6feb; color:#fff; border-color:#1f6feb; }}

/* 테이블 */
.table-wrap {{ overflow-x:auto; margin-bottom:30px; }}
table {{ width:100%; border-collapse:collapse; font-size:0.82em; }}
th {{ background:#111820; color:#8b949e; padding:8px 10px; text-align:left; border-bottom:2px solid #1e2d3d; cursor:pointer; white-space:nowrap; position:sticky; top:0; z-index:2; }}
th:hover {{ color:#c8d6e5; }}
td {{ padding:7px 10px; border-bottom:1px solid #151d27; white-space:nowrap; }}
tr:hover {{ background:#111820; }}
.sticky-col {{ position:sticky; left:0; background:#0a0e17; z-index:1; }}
tr:hover .sticky-col {{ background:#111820; }}

/* 색상 */
.up {{ color:#3fb950; }}
.down {{ color:#f85149; }}
.na {{ color:#484f58; }}
.grade-s {{ color:#f0883e; font-weight:700; font-size:1.1em; }}
.grade-a {{ color:#3fb950; font-weight:600; }}
.grade-b {{ color:#58a6ff; }}
.grade-c {{ color:#8b949e; }}
.grade-d {{ color:#484f58; }}
.cache-source {{ color:#484f58; font-size:0.85em; }}
.turn-tag {{ background:#f8514920; color:#f85149; padding:1px 5px; border-radius:3px; font-size:0.75em; margin-left:4px; }}

/* 하이라이트 행 */
.gold-row {{ background:#ffc10710 !important; }}
.gold-row .sticky-col {{ background:#ffc10708 !important; }}
.highlight-row {{ background:#3fb95008 !important; }}

/* 섹션 */
.section {{ background:#111820; border:1px solid #1e2d3d; border-radius:10px; padding:20px; margin-bottom:25px; }}
.section h2 {{ color:#e8eaed; font-size:1.15em; margin-bottom:12px; }}

/* 바 차트 */
.bar-row {{ display:flex; align-items:center; margin-bottom:6px; }}
.bar-label {{ width:120px; font-size:0.82em; color:#8b949e; text-align:right; padding-right:10px; flex-shrink:0; }}
.bar-track {{ flex:1; height:18px; background:#151d27; border-radius:3px; overflow:hidden; }}
.bar-positive {{ height:100%; background:linear-gradient(90deg,#1f6feb,#3fb950); border-radius:3px; }}
.bar-negative {{ height:100%; background:linear-gradient(90deg,#f85149,#da3633); border-radius:3px; }}
.bar-value {{ width:70px; font-size:0.8em; padding-left:8px; }}

/* 반응형 */
@media(max-width:768px) {{
    .cards {{ flex-direction:column; }}
    .bar-label {{ width:80px; font-size:0.75em; }}
}}

/* 면책 */
.disclaimer {{ color:#484f58; font-size:0.75em; margin-top:30px; padding-top:15px; border-top:1px solid #1e2d3d; }}
</style>
</head>
<body>

<h1>2026 Q1 실적 기반 밸류에이션 괴리 분석</h1>
<div class="subtitle">
    분석일: {date_str} | 총 {total}종목 |
    DART Q1 공시: {dart_count}개 | 캐시 폴백: {cache_count}개
</div>

<!-- 요약 카드 -->
<div class="cards">
    <div class="card">
        <div class="card-label">분석 종목</div>
        <div class="card-value">{total}</div>
    </div>
    <div class="card">
        <div class="card-label">S+A 등급</div>
        <div class="card-value up">{sa_count}</div>
    </div>
    <div class="card">
        <div class="card-label">평균 괴리율</div>
        <div class="card-value {'up' if avg_gap > 0 else 'down' if avg_gap < 0 else ''}">{'+' if avg_gap > 0 else ''}{avg_gap}%p</div>
    </div>
    <div class="card">
        <div class="card-label">DART Q1 공시</div>
        <div class="card-value">{dart_count}</div>
    </div>
</div>

<!-- 섹터 탭 -->
<div class="tabs">
    {''.join(tab_buttons)}
</div>

<!-- 섹터별 테이블 -->
{''.join(sector_tables)}

<!-- TOP 10 -->
<div class="section">
    <h2>실적 대비 가장 덜 오른 TOP 10 (괴리율 기준)</h2>
    <div class="table-wrap">{_make_table(top10_rows)}</div>
</div>

<!-- 턴어라운드 -->
<div class="section">
    <h2>적자 → 흑자 전환 (턴어라운드)</h2>
    <div class="table-wrap">{_make_table(turn_rows)}</div>
</div>

<!-- 섹터별 평균 괴리율 -->
<div class="section">
    <h2>섹터별 평균 괴리율 (높을수록 저평가)</h2>
    {''.join(bar_items)}
</div>

<div class="disclaimer">
    ※ 본 리포트는 참고 자료이며, 투자 판단의 최종 책임은 투자자 본인에게 있습니다.<br>
    ※ 괴리율 = 영업이익 YoY - 주가 6M 수익률. 양수(+)면 실적 대비 주가가 덜 오른 상태.<br>
    ※ CACHE 표시 종목은 DART 2026 Q1 미공시로 최근 캐시 데이터 사용. 정확도 제한적.<br>
    ※ 연환산 PER = 현재가 / (Q1 EPS × 4). 계절성 미반영 한계 있음.<br>
    생성: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Quantum Master
</div>

<script>
function showTab(id, el) {{
    document.querySelectorAll('.sector-table').forEach(t => t.style.display='none');
    document.getElementById(id).style.display='block';
    document.querySelectorAll('.tab').forEach(a => a.classList.remove('active'));
    el.classList.add('active');
}}

// 테이블 정렬
document.querySelectorAll('th[data-sort]').forEach(th => {{
    th.addEventListener('click', function() {{
        const table = this.closest('table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const col = parseInt(this.dataset.sort);
        const type = this.dataset.type || 'num';
        const asc = this.dataset.asc === '1';
        this.dataset.asc = asc ? '0' : '1';

        rows.sort((a, b) => {{
            let va = a.cells[col]?.textContent?.replace(/[^0-9.\\-]/g, '') || '0';
            let vb = b.cells[col]?.textContent?.replace(/[^0-9.\\-]/g, '') || '0';
            va = parseFloat(va) || 0;
            vb = parseFloat(vb) || 0;
            return asc ? va - vb : vb - va;
        }});

        rows.forEach(r => tbody.appendChild(r));
    }});
}});
</script>
</body>
</html>"""

    return html


def _make_table(rows_html: str) -> str:
    """테이블 래퍼."""
    return f"""<div class="table-wrap">
<table>
<thead><tr>
<th class="sticky-col">종목명</th>
<th>코드</th>
<th data-sort="2" data-type="text">등급</th>
<th data-sort="3">스코어</th>
<th data-sort="4">영업이익 YoY</th>
<th data-sort="5">매출 YoY</th>
<th data-sort="6">주가 6M</th>
<th data-sort="7">괴리율</th>
<th data-sort="8">Q1 EPS</th>
<th data-sort="9">연환산PER</th>
<th data-sort="10">trailing PER</th>
<th data-sort="11">영업이익</th>
<th data-sort="12">부채비율</th>
<th data-sort="13">ROE</th>
<th>데이터출처</th>
</tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>"""


# ═══════════════════════════════════════════════════
# 8. 메인
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="2026 Q1 밸류에이션 괴리 분석")
    parser.add_argument("--no-dart", action="store_true", help="DART API 스킵 (캐시만 사용)")
    parser.add_argument("--dry-run", action="store_true", help="분석만, 파일 저장 안 함")
    parser.add_argument("--upload", action="store_true", help="Supabase FLOWX 업로드")
    args = parser.parse_args()

    date_str = datetime.now().strftime("%Y%m%d")
    logger.info("=" * 60)
    logger.info("2026 Q1 밸류에이션 괴리 분석 시작 (%s)", date_str)
    logger.info("=" * 60)

    # 1. 유니버스
    name_map = load_name_map()
    sectors = load_sector_config()
    universe = build_universe(sectors, name_map)

    # 2. DART Q1 수집
    dart_data: dict[str, dict] = {}
    if not args.no_dart:
        tickers = [s["ticker"] for s in universe]
        dart_data = collect_dart_q1(tickers)
    else:
        logger.info("DART 스킵 (--no-dart)")

    # 3. 캐시 로드
    cache = load_financial_cache()

    # 4. 시장 데이터 (최신 종가 + parquet PER/EPS + universe.csv 시총)
    all_tickers = [s["ticker"] for s in universe]
    latest_prices = load_latest_prices(all_tickers)
    market_caps = load_universe_market_cap()

    # 5. 분석
    results = []
    for stock in universe:
        ticker = stock["ticker"]

        # 실적: DART → 캐시 폴백
        earnings = dart_data.get(ticker)
        if earnings is None:
            earnings = get_earnings_from_cache(ticker, cache)

        # 시장 데이터: 최신 종가 우선 → parquet 폴백
        lp = latest_prices.get(ticker, {})
        pq_data = load_parquet_data(ticker)

        if lp.get("close"):
            # 최신 종가 기반 수익률 계산
            c = lp["close"]
            c6m = lp.get("close_6m")
            c3m = lp.get("close_3m")
            c1y = lp.get("close_1y")
            price = {
                "close": c,
                "ret_6m": round((c / c6m - 1) * 100, 2) if c6m and c6m > 0 else None,
                "ret_3m": round((c / c3m - 1) * 100, 2) if c3m and c3m > 0 else None,
                "ret_1y": round((c / c1y - 1) * 100, 2) if c1y and c1y > 0 else None,
                "trailing_per": pq_data.get("trailing_per", 0) if pq_data else 0,
                "trailing_eps": pq_data.get("trailing_eps", 0) if pq_data else 0,
                "trailing_pbr": pq_data.get("trailing_pbr", 0) if pq_data else 0,
                "market_cap": market_caps.get(ticker),
            }
        elif pq_data and pq_data.get("close"):
            price = pq_data
            price["market_cap"] = market_caps.get(ticker)
        else:
            price = load_price_returns(ticker, stock["name"])
            price["market_cap"] = market_caps.get(ticker)

        quality = get_quality_from_cache(ticker, cache)

        result = analyze_stock(ticker, stock["name"], stock["sector"], earnings, price, quality)
        if result:
            results.append(result)

    # 정렬: 스코어 내림차순
    results.sort(key=lambda x: x["score"], reverse=True)

    logger.info("=" * 60)
    logger.info("분석 완료: %d / %d 종목", len(results), len(universe))
    logger.info("=" * 60)

    # 콘솔 요약
    for i, r in enumerate(results[:15]):
        gap_str = f"{r['gap']:+.1f}%p" if r['gap'] is not None else "N/A"
        oi_str = f"{r['oi_yoy']:+.1f}%" if r['oi_yoy'] is not None else "N/A"
        logger.info(
            "%2d. [%s] %s (%s) 스코어=%s | 영업이익YoY=%s | 주가6M=%s | 괴리율=%s | PER=%s",
            i + 1, r["grade"], r["name"], r["ticker"], r["score"],
            oi_str,
            f"{r['ret_6m']:+.1f}%" if r['ret_6m'] is not None else "N/A",
            gap_str,
            f"{r['calc_per']:.1f}" if r['calc_per'] else "N/A",
        )

    if args.dry_run:
        logger.info("dry-run 모드 — 파일 저장 스킵")
        return

    # 6. 저장
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = DATA_DIR / f"valuation_gap_{date_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("JSON 저장: %s", json_path)

    html_path = REPORT_DIR / f"valuation_gap_{date_str}.html"
    html = build_html_report(results, sectors, date_str)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML 저장: %s", html_path)

    # 7. FLOWX Supabase 업로드
    if args.upload:
        try:
            from src.adapters.flowx_uploader import FlowxUploader
            uploader = FlowxUploader()
            if uploader.is_active:
                date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                ok = uploader.upload_valuation_gap(date_dash)
                logger.info("FLOWX 업로드: %s", "성공" if ok else "실패")
            else:
                logger.warning("FLOWX 비활성 — SUPABASE_URL/KEY 확인")
        except Exception as e:
            logger.error("FLOWX 업로드 에러: %s", e)

    logger.info("완료!")


if __name__ == "__main__":
    main()
