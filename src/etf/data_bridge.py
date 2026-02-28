"""
ETF 데이터 브릿지 — 방탄 설계
====================================
기존 JARVIS 데이터 파이프라인 → ETF 3축 엔진 입력 포맷 변환.

Fallback 체인:
  1) parquet → pandas
  2) JSON → fallback
  3) ETF daily parquet → 직접 모멘텀 계산
  4) 완전 없음 → 안전한 기본값 (CAUTION)

컬럼명 한글/영문 자동 매칭 포함.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.etf.config import build_sector_universe

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 데이터 경로
_PATHS = {
    "momentum_json": DATA_DIR / "sector_rotation" / "sector_momentum.json",
    "signal_json": DATA_DIR / "sector_rotation" / "etf_trading_signal.json",
    "flow_json": DATA_DIR / "sector_rotation" / "investor_flow.json",
    "overnight_json": DATA_DIR / "us_market" / "overnight_signal.json",
    "leverage_json": DATA_DIR / "leverage_etf" / "leverage_etf_scan.json",
    "kospi_csv": DATA_DIR / "kospi_index.csv",
    "etf_daily_dir": DATA_DIR / "sector_rotation" / "etf_daily",
    "universe_json": DATA_DIR / "sector_rotation" / "etf_universe.json",
    "portfolio_json": DATA_DIR / "portfolio.json",
}

# 컬럼명 자동 매칭 맵 (한글/영문 혼재 대응)
_COLUMN_ALIASES = {
    "close": ["close", "종가", "Close", "CLOSE", "adj_close"],
    "volume": ["volume", "거래량", "Volume", "VOLUME"],
    "open": ["open", "시가", "Open", "OPEN"],
    "high": ["high", "고가", "High", "HIGH"],
    "low": ["low", "저가", "Low", "LOW"],
    "date": ["date", "Date", "날짜", "DATE", "일자"],
}


def _resolve_column(df: pd.DataFrame, target: str) -> str | None:
    """컬럼명 자동 매칭 — 한글/영문 혼재 대응."""
    aliases = _COLUMN_ALIASES.get(target, [target])
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in [c.lower() for c in df.columns]:
            for c in df.columns:
                if c.lower() == alias.lower():
                    return c
    return None


def _safe_json_load(path: Path) -> dict | list | None:
    """JSON 안전 로드."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("JSON 로드 실패 (%s): %s", path.name, e)
        return None


# ============================================================
# 1. 모멘텀 데이터 (3-tier fallback)
# ============================================================
def load_momentum() -> dict:
    """섹터 모멘텀 데이터 로드.

    Fallback 체인:
      1) sector_momentum.json (파이프라인 생성)
      2) ETF daily parquet → 직접 수익률 계산
      3) 빈 딕셔너리 (안전)

    Returns:
        {sector: {"5d": float, "20d": float, "60d": float, "rank": int}}
    """
    # Tier 1: JSON
    raw = _safe_json_load(_PATHS["momentum_json"])
    if raw and raw.get("sectors"):
        result = {}
        for s in raw["sectors"]:
            sector = s.get("sector", "")
            result[sector] = {
                "5d": s.get("ret_5", s.get("ret_5d", 0)),
                "20d": s.get("ret_20", s.get("ret_20d", 0)),
                "60d": s.get("ret_60", s.get("ret_60d", 0)),
                "rank": s.get("rank", 99),
            }
        logger.info("모멘텀: JSON에서 %d개 섹터 로드", len(result))
        return result

    # Tier 2: ETF daily parquet → 직접 계산
    etf_dir = _PATHS["etf_daily_dir"]
    if etf_dir.exists():
        result = _calc_momentum_from_parquet(etf_dir)
        if result:
            logger.info("모멘텀: parquet에서 직접 계산 — %d개 섹터", len(result))
            return result

    # Tier 3: 빈 데이터
    logger.warning("모멘텀: 데이터 없음 — 빈 딕셔너리 반환")
    return {}


def _calc_momentum_from_parquet(etf_dir: Path) -> dict:
    """ETF daily parquet에서 직접 모멘텀 수익률 계산."""
    universe_raw = _safe_json_load(_PATHS["universe_json"])
    if not universe_raw:
        return {}

    # sector → etf_code 매핑
    sector_codes = {}
    for sector, info in universe_raw.items():
        code = info.get("etf_code", "")
        sector_codes[sector] = code

    results = {}
    for sector, code in sector_codes.items():
        parquet_path = etf_dir / f"{code}.parquet"
        if not parquet_path.exists():
            continue

        try:
            df = pd.read_parquet(parquet_path)
            close_col = _resolve_column(df, "close")
            if not close_col or len(df) < 60:
                continue

            closes = df[close_col].values
            last = closes[-1]
            ret_5 = (last / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
            ret_20 = (last / closes[-21] - 1) * 100 if len(closes) >= 21 else 0
            ret_60 = (last / closes[-61] - 1) * 100 if len(closes) >= 61 else 0

            results[sector] = {
                "5d": round(ret_5, 2),
                "20d": round(ret_20, 2),
                "60d": round(ret_60, 2),
                "rank": 99,  # 순위는 아래서 재계산
            }
        except Exception as e:
            logger.warning("parquet 모멘텀 계산 실패 (%s): %s", code, e)

    # 가중 모멘텀으로 순위 부여
    if results:
        scored = []
        for sector, m in results.items():
            score = m["5d"] * 0.2 + m["20d"] * 0.5 + m["60d"] * 0.3
            scored.append((sector, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        for rank, (sector, _) in enumerate(scored, 1):
            results[sector]["rank"] = rank

    return results


# ============================================================
# 2. Smart Money 데이터 (2-tier fallback)
# ============================================================
def load_smart_money() -> dict:
    """Smart Money 시그널 로드.

    Fallback 체인:
      1) etf_trading_signal.json (smart_money_etf + theme_money_etf + smart_sectors)
      2) 수급 데이터(investor_flow)에서 자동 분류

    Returns:
        {etf_code: {"type": "smart_money"/"theme_money"/"none", "score": float, "sector": str}}
    """
    # Tier 1: JSON
    raw = _safe_json_load(_PATHS["signal_json"])
    if raw:
        result = {}

        for entry in raw.get("smart_money_etf", []):
            code = entry.get("etf_code", "")
            if code:
                result[code] = {
                    "type": "smart_money",
                    "score": entry.get("momentum_score", 0),
                    "sector": entry.get("sector", ""),
                }

        for entry in raw.get("theme_money_etf", []):
            code = entry.get("etf_code", "")
            if code and code not in result:
                result[code] = {
                    "type": "theme_money",
                    "score": entry.get("momentum_score", 0),
                    "sector": entry.get("sector", ""),
                }

        # smart_sectors → universe에서 매칭
        smart_sectors = set(raw.get("smart_sectors", []))
        if smart_sectors:
            universe = build_sector_universe()
            for code, info in universe.items():
                if info["sector"] in smart_sectors and code not in result:
                    result[code] = {
                        "type": "smart_money",
                        "score": 70,
                        "sector": info["sector"],
                    }

        if result:
            logger.info("Smart Money: JSON에서 %d개 로드", len(result))
            return result

    # Tier 2: 수급 데이터에서 자동 분류
    result = _classify_smart_money_from_flow()
    if result:
        logger.info("Smart Money: 수급 데이터에서 자동 분류 — %d개", len(result))
        return result

    logger.warning("Smart Money: 데이터 없음 — 빈 딕셔너리 반환")
    return {}


def _classify_smart_money_from_flow() -> dict:
    """수급 데이터에서 Smart Money 자동 분류.

    외인+기관 동시 순매수 & 누적 > 0 → smart_money
    외인만 순매수 → theme_money
    """
    raw = _safe_json_load(_PATHS["flow_json"])
    if not raw:
        return {}

    sectors_data = raw.get("sectors", [])
    universe = build_sector_universe()
    sector_to_codes = {}
    for code, info in universe.items():
        sector = info["sector"]
        if sector not in sector_to_codes:
            sector_to_codes[sector] = []
        sector_to_codes[sector].append(code)

    result = {}
    for s in sectors_data:
        sector = s.get("sector", "")
        foreign = s.get("foreign_cum_bil", 0)
        inst = s.get("inst_cum_bil", 0)

        if foreign > 0 and inst > 0:
            sm_type = "smart_money"
        elif foreign > 0:
            sm_type = "theme_money"
        else:
            continue

        for code in sector_to_codes.get(sector, []):
            if code not in result:
                result[code] = {
                    "type": sm_type,
                    "score": min(100, (foreign + inst) / 100),
                    "sector": sector,
                }

    return result


# ============================================================
# 3. 수급 데이터
# ============================================================
def load_supply() -> dict:
    """수급(investor_flow) 데이터 로드.

    섹터 기반 → ETF 코드 매핑 + 0~100 정규화.

    Returns:
        {etf_code: {"foreign_net_5d": float, "inst_net_5d": float, "score": int}}
    """
    raw = _safe_json_load(_PATHS["flow_json"])
    if not raw or not raw.get("sectors"):
        logger.warning("수급: 데이터 없음 — 빈 딕셔너리 반환")
        return {}

    sectors_data = raw["sectors"]

    # 정규화 기준 계산
    totals = [s.get("foreign_cum_bil", 0) + s.get("inst_cum_bil", 0) for s in sectors_data]
    max_total = max(totals) if totals else 1
    min_total = min(totals) if totals else 0
    spread = max_total - min_total if max_total != min_total else 1

    # universe에서 섹터→코드 매핑
    universe = build_sector_universe()
    sector_to_codes = {}
    for code, info in universe.items():
        sector = info["sector"]
        if sector not in sector_to_codes:
            sector_to_codes[sector] = []
        sector_to_codes[sector].append(code)

    result = {}
    for s in sectors_data:
        sector = s.get("sector", "")
        foreign_bil = s.get("foreign_cum_bil", 0)
        inst_bil = s.get("inst_cum_bil", 0)
        total = foreign_bil + inst_bil
        score = max(0, min(100, ((total - min_total) / spread) * 100))

        entry = {
            "foreign_net_5d": foreign_bil * 1e9,
            "inst_net_5d": inst_bil * 1e9,
            "score": round(score),
        }

        # flow 자체 etf_code
        flow_code = s.get("etf_code", "")
        if flow_code:
            result[flow_code] = entry

        # universe 매핑
        for code in sector_to_codes.get(sector, []):
            if code not in result:
                result[code] = entry.copy()

    logger.info("수급: %d개 ETF 매핑", len(result))
    return result


# ============================================================
# 4. US Overnight 시그널
# ============================================================
_GRADE_TO_NUM = {
    "STRONG_BULL": 1, "MILD_BULL": 2, "NEUTRAL": 3,
    "MILD_BEAR": 4, "STRONG_BEAR": 5,
}
_GRADE_TO_SIGNAL = {
    "STRONG_BULL": "strong_positive", "MILD_BULL": "positive",
    "NEUTRAL": "neutral", "MILD_BEAR": "negative",
    "STRONG_BEAR": "strong_negative",
}


def load_us_overnight() -> dict:
    """US Overnight 시그널 로드.

    Returns:
        {"grade": 1~5, "signal": str}
    """
    raw = _safe_json_load(_PATHS["overnight_json"])
    if not raw:
        logger.warning("US Overnight: 데이터 없음 — 기본 NEUTRAL(3등급)")
        return {"grade": 3, "signal": "neutral"}

    grade_str = raw.get("grade", "NEUTRAL").upper()
    return {
        "grade": _GRADE_TO_NUM.get(grade_str, 3),
        "signal": _GRADE_TO_SIGNAL.get(grade_str, "neutral"),
    }


# ============================================================
# 5. 레버리지 5축 스코어 (2-tier fallback)
# ============================================================
_REGIME_SCORE_ESTIMATE = {
    "BULL": 80,
    "CAUTION": 55,
    "BEAR": 65,
    "CRISIS": 75,
}


def load_leverage_5axis(regime: str = "CAUTION") -> float:
    """레버리지 5축 스코어 로드.

    Fallback 체인:
      1) leverage_etf_scan.json → 최고 점수 ETF
      2) 레짐에서 추정 (BULL=80, BEAR=65, CRISIS=75, CAUTION=55)

    Args:
        regime: 현재 KOSPI 레짐 (fallback 추정용)

    Returns:
        0~100 점수
    """
    # Tier 1: leverage_etf_scan.json
    raw = _safe_json_load(_PATHS["leverage_json"])
    if raw and raw.get("etfs"):
        etfs = raw["etfs"]
        best = max(etfs, key=lambda x: x.get("score", 0))
        score = float(best.get("score", 0))
        if score > 0:
            logger.info("5축 스코어: JSON에서 %.0f점 (최고 ETF)", score)
            return score

    # Tier 2: 레짐에서 추정
    regime = regime.upper()
    estimated = _REGIME_SCORE_ESTIMATE.get(regime, 55)
    logger.info("5축 스코어: 파일 없음 → 레짐(%s)에서 추정 %.0f점", regime, estimated)
    return float(estimated)


# ============================================================
# 6. KOSPI 레짐 계산
# ============================================================
def calc_kospi_regime() -> dict:
    """KOSPI 레짐 계산.

    Returns:
        {"regime": str, "close": float, "ma20": float, "ma60": float,
         "ma20_above": bool, "ma60_above": bool}
    """
    kospi_path = _PATHS["kospi_csv"]

    # Fallback: kospi_regime.json
    if not kospi_path.exists():
        regime_path = DATA_DIR / "kospi_regime.json"
        if regime_path.exists():
            raw = _safe_json_load(regime_path)
            if raw:
                close = raw.get("close", 0)
                ma20 = raw.get("ma20", 0)
                ma60 = raw.get("ma60", 0)
                return {
                    "regime": raw.get("regime", "CAUTION"),
                    "close": close,
                    "ma20": ma20,
                    "ma60": ma60,
                    "ma20_above": close > ma20 if ma20 > 0 else False,
                    "ma60_above": close > ma60 if ma60 > 0 else False,
                }
        logger.warning("KOSPI: 데이터 없음 → 기본 CAUTION")
        return _default_regime()

    try:
        df = pd.read_csv(kospi_path, parse_dates=True)

        # 컬럼 자동 매칭
        date_col = _resolve_column(df, "date")
        close_col = _resolve_column(df, "close")

        if date_col and date_col != df.index.name:
            df = df.set_index(date_col)
        df = df.sort_index()

        if not close_col or len(df) < 60:
            return _default_regime()

        df["_ma20"] = df[close_col].rolling(20).mean()
        df["_ma60"] = df[close_col].rolling(60).mean()

        row = df.iloc[-1]
        close = float(row[close_col])
        ma20 = float(row["_ma20"]) if not pd.isna(row["_ma20"]) else 0
        ma60 = float(row["_ma60"]) if not pd.isna(row["_ma60"]) else 0

        # 실현 변동성 백분위
        log_ret = np.log(df[close_col] / df[close_col].shift(1))
        rv20 = log_ret.rolling(20).std() * np.sqrt(252) * 100
        rv20_pct = rv20.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        rv_pct = float(rv20_pct.iloc[-1]) if not pd.isna(rv20_pct.iloc[-1]) else 0.5

        # 레짐 판정
        if ma20 == 0 or ma60 == 0:
            regime = "CAUTION"
        elif close > ma20:
            regime = "BULL" if rv_pct < 0.50 else "CAUTION"
        elif close > ma60:
            regime = "BEAR"
        else:
            regime = "CRISIS"

        return {
            "regime": regime,
            "close": round(close, 2),
            "ma20": round(ma20, 2),
            "ma60": round(ma60, 2),
            "ma20_above": close > ma20 if ma20 > 0 else False,
            "ma60_above": close > ma60 if ma60 > 0 else False,
        }
    except Exception as e:
        logger.error("KOSPI 레짐 계산 실패: %s", e)
        return _default_regime()


def _default_regime() -> dict:
    return {
        "regime": "CAUTION",
        "close": 0,
        "ma20": 0,
        "ma60": 0,
        "ma20_above": True,
        "ma60_above": False,
    }


# ============================================================
# 7. 포트폴리오 (개별주 섹터 추출)
# ============================================================
def load_individual_stock_sectors() -> set[str]:
    """portfolio.json에서 개별주 보유 섹터 추출.

    Returns:
        보유 중인 개별주의 섹터 집합
    """
    raw = _safe_json_load(_PATHS["portfolio_json"])
    if not raw:
        return set()

    sectors = set()
    holdings = raw.get("holdings", raw.get("positions", []))

    # 리스트 형태
    if isinstance(holdings, list):
        for h in holdings:
            sector = h.get("sector", h.get("섹터", ""))
            if sector:
                sectors.add(sector)
    # 딕셔너리 형태
    elif isinstance(holdings, dict):
        for code, info in holdings.items():
            if isinstance(info, dict):
                sector = info.get("sector", info.get("섹터", ""))
                if sector:
                    sectors.add(sector)

    if sectors:
        logger.info("개별주 섹터: %s", sectors)
    return sectors


# ============================================================
# 통합 로더
# ============================================================
def load_all() -> dict:
    """모든 데이터를 한 번에 로드 — 방탄 fallback 적용.

    Returns:
        {
            "momentum": dict,
            "smart_money": dict,
            "supply": dict,
            "us_overnight": dict,
            "regime": dict,
            "five_axis_score": float,
            "individual_sectors": set,
        }
    """
    regime = calc_kospi_regime()

    return {
        "momentum": load_momentum(),
        "smart_money": load_smart_money(),
        "supply": load_supply(),
        "us_overnight": load_us_overnight(),
        "regime": regime,
        "five_axis_score": load_leverage_5axis(regime["regime"]),
        "individual_sectors": load_individual_stock_sectors(),
    }
