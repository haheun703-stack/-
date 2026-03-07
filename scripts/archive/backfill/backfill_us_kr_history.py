"""US-KR Market History Backfill & Level 2 Pattern Matcher

2년치 미국/한국 시장 데이터를 수집하여 패턴 매칭 DB 구축.
US 마감 → 다음 KR 거래일 페어를 만들어 SQLite에 저장.

출력:
    data/us_market/us_kr_history.db (SQLite)
    data/us_market/backfill_report.json (리포트)

사용법:
    python scripts/backfill_us_kr_history.py              # 2년 백필
    python scripts/backfill_us_kr_history.py --years 3    # 3년 백필
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

US_DIR = project_root / "data" / "us_market"
DB_PATH = US_DIR / "us_kr_history.db"
REPORT_PATH = US_DIR / "backfill_report.json"

# ── US 심볼 ──
US_SYMBOLS = {
    # Tier 1: 핵심 지표
    "us_sp500":   "SPY",
    "us_nasdaq":  "QQQ",
    "us_dow":     "DIA",
    "us_vix":     "^VIX",
    "us_soxx":    "SOXX",
    # Tier 2: 매크로
    "us_oil":     "USO",
    "us_gold":    "GLD",
    "us_dollar":  "UUP",
    "us_bond10y": "TLT",
    "us_china":   "FXI",
    # Tier 3: 테마
    "us_tsla":    "TSLA",
    "us_xbi":     "XBI",
    "us_xle":     "XLE",
    "us_xlf":     "XLF",
    # 한국 프록시 (미국 상장 Korea ETF — 가장 강력한 KOSPI 선행지표)
    "us_ewy":     "EWY",
}

# ── KR 섹터 ETF (KODEX 시리즈) ──
KR_SYMBOLS = {
    "kr_kospi":    "^KS11",
    "kr_kosdaq":   "^KQ11",
    "kr_semi":     "091160.KS",   # KODEX 반도체
    "kr_ev":       "305720.KS",   # KODEX 2차전지산업
    "kr_bio":      "244580.KS",   # KODEX 바이오
    "kr_bank":     "091170.KS",   # KODEX 은행
    "kr_steel":    "117680.KS",   # KODEX 철강
    "kr_it":       "315930.KS",   # KODEX IT플러스
    "kr_energy":   "117460.KS",   # KODEX 에너지화학
    "kr_domestic": "069500.KS",   # KODEX 200 (내수 대용)
}

# 실패 시 대체 심볼
KR_FALLBACK = {
    "kr_semi":     ["091230.KS"],
    "kr_ev":       ["364690.KS"],
    "kr_bio":      ["227540.KS"],
    "kr_it":       ["261060.KS"],
    "kr_energy":   ["139230.KS"],
    "kr_domestic": ["229200.KS"],
}


# ================================================================
# 1. 데이터 다운로드
# ================================================================

def download_data(symbols_dict: dict, years: int = 2, label: str = "") -> tuple[dict, list]:
    """yfinance로 데이터 다운로드."""
    import yfinance as yf

    all_data = {}
    failed = []

    logger.info(f"[{label}] {len(symbols_dict)}개 심볼 다운로드 시작...")

    for name, symbol in symbols_dict.items():
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{years}y")

            if df is None or df.empty:
                logger.warning(f"  {name:20s} ({symbol}): 데이터 없음")
                failed.append((name, symbol))
                continue

            # 타임존 제거 + 일별 변화율
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df["change_pct"] = df["Close"].pct_change() * 100
            df["name"] = name
            all_data[name] = df
            logger.info(f"  {name:20s} ({symbol}): {len(df)}일")

        except Exception as e:
            logger.warning(f"  {name:20s} ({symbol}): 실패 - {e}")
            failed.append((name, symbol))

        time.sleep(0.2)

    logger.info(f"  성공: {len(all_data)}/{len(symbols_dict)} | 실패: {len(failed)}")
    return all_data, failed


def try_kr_fallback(failed: list, years: int = 2) -> dict:
    """실패한 KR 심볼 대체 시도."""
    import yfinance as yf

    recovered = {}
    for name, symbol in failed:
        alts = KR_FALLBACK.get(name, [])
        for alt in alts:
            if alt == symbol:
                continue
            try:
                ticker = yf.Ticker(alt)
                df = ticker.history(period=f"{years}y")
                if df is not None and not df.empty:
                    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                    df["change_pct"] = df["Close"].pct_change() * 100
                    df["name"] = name
                    recovered[name] = df
                    logger.info(f"  {name} 대체 성공: {alt} ({len(df)}일)")
                    break
            except Exception:
                pass
    return recovered


# ================================================================
# 2. 날짜 매칭: US 마감일 → 다음 KR 거래일
# ================================================================

def build_date_pairs(us_data: dict, kr_data: dict) -> pd.DataFrame | None:
    """US 마감일과 다음 한국 거래일을 매칭."""

    kr_kospi = kr_data.get("kr_kospi")
    if kr_kospi is None:
        logger.error("KOSPI 데이터 없음 - 매칭 불가")
        return None

    us_spy = us_data.get("us_sp500")
    if us_spy is None:
        logger.error("SPY 데이터 없음 - 매칭 불가")
        return None

    kr_trading_days = set(kr_kospi.index.normalize())
    us_trading_days = sorted(us_spy.index.normalize())

    pairs = []
    for us_date in us_trading_days:
        # 다음날부터 5영업일 내 첫 KR 거래일 찾기
        for offset in range(1, 6):
            candidate = (us_date + timedelta(days=offset)).normalize()
            if candidate in kr_trading_days:
                pairs.append({
                    "us_date": us_date,
                    "kr_date": candidate,
                    "gap_days": offset,
                })
                break

    pairs_df = pd.DataFrame(pairs)
    logger.info(f"날짜 매칭 완료: {len(pairs_df)}쌍 "
                f"({pairs_df['us_date'].min().date()} ~ {pairs_df['us_date'].max().date()})")
    return pairs_df


# ================================================================
# 3. 히스토리 레코드 생성
# ================================================================

def _get_change(data_dict: dict, key: str, date) -> float | None:
    """특정 날짜의 변화율(%) 추출."""
    if key not in data_dict:
        return None
    df = data_dict[key]
    try:
        date_n = pd.Timestamp(date).normalize()
        mask = df.index.normalize() == date_n
        if mask.any():
            val = df.loc[mask, "change_pct"].iloc[0]
            return round(val, 4) if pd.notna(val) else None
    except Exception:
        pass
    return None


def _get_close(data_dict: dict, key: str, date) -> float | None:
    """특정 날짜의 종가."""
    if key not in data_dict:
        return None
    df = data_dict[key]
    try:
        date_n = pd.Timestamp(date).normalize()
        mask = df.index.normalize() == date_n
        if mask.any():
            val = df.loc[mask, "Close"].iloc[0]
            return round(val, 2) if pd.notna(val) else None
    except Exception:
        pass
    return None


def _get_open_gap(data_dict: dict, key: str, date) -> float | None:
    """시가 갭 (전일 종가 대비 당일 시가)."""
    if key not in data_dict:
        return None
    df = data_dict[key]
    try:
        date_n = pd.Timestamp(date).normalize()
        mask = df.index.normalize() == date_n
        if mask.any():
            idx = df.index.get_loc(df.index[mask][0])
            if idx > 0:
                prev_close = df.iloc[idx - 1]["Close"]
                today_open = df.iloc[idx]["Open"]
                if prev_close > 0:
                    return round(((today_open - prev_close) / prev_close) * 100, 4)
    except Exception:
        pass
    return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _calc_overnight_score(record: dict) -> float:
    """Level 1 US Overnight Score (-100 ~ +100).

    가중치: EWY 25% > NASDAQ 20% > SP500 15% = VIX 15% = SOXX 15% > Dollar 10%
    EWY는 미국 상장 한국 ETF로 KOSPI와 직접 연동 → 가장 강력한 선행지표.
    """
    score = 0.0

    # EWY (25%) — 한국 프록시, 가장 높은 가중치
    v = record.get("us_ewy_chg")
    if v is not None:
        score += _clamp(v * 12.5, -25, 25)

    # NASDAQ (20%)
    v = record.get("us_nasdaq_chg")
    if v is not None:
        score += _clamp(v * 10, -20, 20)

    # S&P 500 (15%)
    v = record.get("us_sp500_chg")
    if v is not None:
        score += _clamp(v * 7.5, -15, 15)

    # VIX (15%, 역방향)
    v = record.get("us_vix_chg")
    if v is not None:
        score += _clamp(v * -3.75, -15, 15)

    # SOXX (15%)
    v = record.get("us_soxx_chg")
    if v is not None:
        score += _clamp(v * 10, -15, 15)

    # Dollar (10%, 역방향)
    v = record.get("us_dollar_chg")
    if v is not None:
        score += _clamp(v * -7, -10, 10)

    return round(_clamp(score, -100, 100), 1)


def build_history_records(
    pairs_df: pd.DataFrame,
    us_data: dict,
    kr_data: dict,
) -> list[dict]:
    """날짜 페어 기반 히스토리 레코드 생성."""

    us_fields = {
        "us_sp500_chg":   "us_sp500",
        "us_nasdaq_chg":  "us_nasdaq",
        "us_dow_chg":     "us_dow",
        "us_soxx_chg":    "us_soxx",
        "us_oil_chg":     "us_oil",
        "us_dollar_chg":  "us_dollar",
        "us_bond10y_chg": "us_bond10y",
        "us_gold_chg":    "us_gold",
        "us_china_chg":   "us_china",
        "us_tsla_chg":    "us_tsla",
        "us_xbi_chg":     "us_xbi",
        "us_xle_chg":     "us_xle",
        "us_xlf_chg":     "us_xlf",
        "us_ewy_chg":     "us_ewy",
    }

    kr_fields = {
        "kr_kospi_chg":    "kr_kospi",
        "kr_kosdaq_chg":   "kr_kosdaq",
        "kr_semi_chg":     "kr_semi",
        "kr_ev_chg":       "kr_ev",
        "kr_bio_chg":      "kr_bio",
        "kr_bank_chg":     "kr_bank",
        "kr_steel_chg":    "kr_steel",
        "kr_it_chg":       "kr_it",
        "kr_oil_chg":      "kr_energy",
        "kr_domestic_chg": "kr_domestic",
    }

    records = []
    skipped = 0

    for _, pair in pairs_df.iterrows():
        us_date = pair["us_date"]
        kr_date = pair["kr_date"]

        record = {
            "date": kr_date.strftime("%Y-%m-%d"),
            "us_date": us_date.strftime("%Y-%m-%d"),
            "gap_days": int(pair["gap_days"]),
        }

        # US 데이터
        for field, data_key in us_fields.items():
            record[field] = _get_change(us_data, data_key, us_date)

        # VIX 절대 레벨
        record["us_vix_chg"] = _get_change(us_data, "us_vix", us_date)
        record["us_vix_level"] = _get_close(us_data, "us_vix", us_date)

        # KR 데이터
        for field, data_key in kr_fields.items():
            record[field] = _get_change(kr_data, data_key, kr_date)

        # KOSPI 시가 갭
        record["kr_kospi_open_gap"] = _get_open_gap(kr_data, "kr_kospi", kr_date)

        # Level 1 Score 산출
        record["us_overnight_score"] = _calc_overnight_score(record)

        # 핵심 데이터 누락 체크
        if record["us_sp500_chg"] is not None and record["kr_kospi_chg"] is not None:
            records.append(record)
        else:
            skipped += 1

    logger.info(f"레코드 생성: {len(records)}건 | 스킵: {skipped}건")
    return records


# ================================================================
# 4. SQLite DB
# ================================================================

def create_database() -> sqlite3.Connection:
    """SQLite DB 생성."""
    US_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS us_kr_history")
    cursor.execute("""
    CREATE TABLE us_kr_history (
        date                TEXT PRIMARY KEY,
        us_date             TEXT,
        gap_days            INTEGER,

        -- US Market
        us_sp500_chg        REAL,
        us_nasdaq_chg       REAL,
        us_dow_chg          REAL,
        us_vix_level        REAL,
        us_vix_chg          REAL,
        us_soxx_chg         REAL,
        us_oil_chg          REAL,
        us_gold_chg         REAL,
        us_dollar_chg       REAL,
        us_bond10y_chg      REAL,
        us_china_chg        REAL,
        us_tsla_chg         REAL,
        us_xbi_chg          REAL,
        us_xle_chg          REAL,
        us_xlf_chg          REAL,
        us_ewy_chg          REAL,
        us_overnight_score  REAL,

        -- KR Market (다음 거래일 실적)
        kr_kospi_chg        REAL,
        kr_kospi_open_gap   REAL,
        kr_kosdaq_chg       REAL,
        kr_semi_chg         REAL,
        kr_ev_chg           REAL,
        kr_bio_chg          REAL,
        kr_bank_chg         REAL,
        kr_steel_chg        REAL,
        kr_it_chg           REAL,
        kr_oil_chg          REAL,
        kr_domestic_chg     REAL,

        created_at          TEXT DEFAULT (datetime('now'))
    )
    """)

    cursor.execute("CREATE INDEX idx_us_score ON us_kr_history(us_overnight_score)")
    cursor.execute("CREATE INDEX idx_us_date ON us_kr_history(us_date)")
    cursor.execute("CREATE INDEX idx_vix_level ON us_kr_history(us_vix_level)")

    conn.commit()
    return conn


def insert_records(conn: sqlite3.Connection, records: list[dict]) -> int:
    """레코드 일괄 삽입."""
    if not records:
        return 0

    columns = [k for k in records[0].keys()]
    placeholders = ", ".join(["?" for _ in columns])
    columns_str = ", ".join(columns)

    cursor = conn.cursor()
    inserted = 0

    for record in records:
        try:
            values = [record.get(col) for col in columns]
            cursor.execute(
                f"INSERT OR REPLACE INTO us_kr_history ({columns_str}) VALUES ({placeholders})",
                values,
            )
            inserted += 1
        except Exception as e:
            logger.warning(f"삽입 실패 ({record.get('date')}): {e}")

    conn.commit()
    logger.info(f"DB 저장 완료: {inserted}/{len(records)}건")
    return inserted


# ================================================================
# 5. Level 2 패턴 매칭 엔진
# ================================================================

class PatternMatcher:
    """역사적 패턴 매칭으로 보정값 산출."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or DB_PATH)
        self.min_samples = 15

    def find_similar_patterns(
        self,
        today_us: dict,
        top_pct: int = 20,
    ) -> pd.DataFrame | None:
        """오늘과 유사한 과거 패턴 검색.

        Args:
            today_us: {"us_nasdaq_chg": float, "us_sp500_chg": float, ...}
            top_pct: 상위 몇 % 유사도를 선택할지
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM us_kr_history ORDER BY date DESC", conn)
        conn.close()

        if len(df) < self.min_samples:
            logger.warning(f"데이터 부족: {len(df)}건 (최소 {self.min_samples}건)")
            return None

        features = [
            "us_nasdaq_chg", "us_sp500_chg", "us_vix_chg",
            "us_soxx_chg", "us_dollar_chg", "us_ewy_chg",
        ]

        today_vector = np.array([
            today_us.get("us_nasdaq_chg", 0) or 0,
            today_us.get("us_sp500_chg", 0) or 0,
            today_us.get("us_vix_chg", 0) or 0,
            today_us.get("us_soxx_chg", 0) or 0,
            today_us.get("us_dollar_chg", 0) or 0,
            today_us.get("us_ewy_chg", 0) or 0,
        ])

        feature_df = df[features].fillna(0)
        means = feature_df.mean()
        stds = feature_df.std().replace(0, 1)

        norm_today = (today_vector - means.values) / stds.values
        norm_hist = (feature_df - means) / stds

        distances = np.sqrt(((norm_hist - norm_today) ** 2).sum(axis=1))
        df["distance"] = distances

        threshold = np.percentile(distances.dropna(), top_pct)
        similar = df[df["distance"] <= threshold].copy()

        return similar.sort_values("distance")

    def analyze_patterns(self, similar_days: pd.DataFrame | None) -> dict:
        """유사 패턴 분석 -> 보정값 산출."""
        if similar_days is None or len(similar_days) < self.min_samples:
            return {"status": "insufficient_data", "pattern_adjustment": 0, "confidence": 0}

        result = {"sample_count": len(similar_days), "status": "ok"}

        # KOSPI 예측
        kospi = similar_days["kr_kospi_chg"].dropna()
        result["kospi"] = {
            "mean_chg": round(kospi.mean(), 3),
            "median_chg": round(kospi.median(), 3),
            "std": round(kospi.std(), 3),
            "positive_rate": round((kospi > 0).mean() * 100, 1),
        }

        # 시가 갭 예측
        gap = similar_days["kr_kospi_open_gap"].dropna()
        if len(gap) > 0:
            result["kospi_open_gap"] = {
                "mean_gap": round(gap.mean(), 3),
                "median_gap": round(gap.median(), 3),
            }

        # 섹터별 예측
        sector_cols = {
            "반도체":   "kr_semi_chg",
            "2차전지":  "kr_ev_chg",
            "바이오":   "kr_bio_chg",
            "은행":     "kr_bank_chg",
            "철강":     "kr_steel_chg",
            "IT":       "kr_it_chg",
            "에너지":   "kr_oil_chg",
            "내수":     "kr_domestic_chg",
        }

        result["sectors"] = {}
        for sector_name, col in sector_cols.items():
            if col in similar_days.columns:
                s = similar_days[col].dropna()
                if len(s) >= 10:
                    result["sectors"][sector_name] = {
                        "mean_chg": round(s.mean(), 3),
                        "positive_rate": round((s > 0).mean() * 100, 1),
                        "sample_count": len(s),
                    }

        # 패턴 보정값 (-15 ~ +15)
        confidence = min(len(kospi) / 50, 1.0)
        pattern_adj = kospi.mean() * 5 * confidence
        result["pattern_adjustment"] = round(_clamp(pattern_adj, -15, 15), 1)
        result["confidence"] = round(confidence, 2)

        return result


# ================================================================
# 6. 분석 리포트
# ================================================================

def generate_analysis_report(conn: sqlite3.Connection) -> dict:
    """DB 기반 통계 분석 리포트."""
    df = pd.read_sql("SELECT * FROM us_kr_history ORDER BY date", conn)

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_records": len(df),
        "date_range": {"start": df["date"].min(), "end": df["date"].max()},
    }

    # Score 구간별 KOSPI 반응
    bins = [(-100, -50), (-50, -20), (-20, 20), (20, 50), (50, 100)]
    labels = ["STRONG_BEAR", "MILD_BEAR", "NEUTRAL", "MILD_BULL", "STRONG_BULL"]

    score_analysis = {}
    for (lo, hi), label in zip(bins, labels):
        subset = df[(df["us_overnight_score"] >= lo) & (df["us_overnight_score"] < hi)]
        if len(subset) > 0:
            kospi = subset["kr_kospi_chg"].dropna()
            if len(kospi) > 0:
                score_analysis[label] = {
                    "count": len(subset),
                    "kospi_mean": round(kospi.mean(), 3),
                    "kospi_positive_rate": round((kospi > 0).mean() * 100, 1),
                    "kospi_median": round(kospi.median(), 3),
                }

    report["score_vs_kospi"] = score_analysis

    # 상관관계
    us_cols = ["us_sp500_chg", "us_nasdaq_chg", "us_soxx_chg",
               "us_vix_chg", "us_dollar_chg", "us_oil_chg"]
    kr_cols = ["kr_kospi_chg", "kr_semi_chg", "kr_ev_chg",
               "kr_bio_chg", "kr_bank_chg"]

    correlations = {}
    for uc in us_cols:
        for kc in kr_cols:
            if uc in df.columns and kc in df.columns:
                valid = df[[uc, kc]].dropna()
                if len(valid) >= 30:
                    corr = valid[uc].corr(valid[kc])
                    correlations[f"{uc} -> {kc}"] = round(corr, 3)

    sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
    report["correlations_top15"] = dict(list(sorted_corr.items())[:15])

    # 극단 상황 분석
    extreme = {}

    nasdaq_crash = df[df["us_nasdaq_chg"] <= -3.0]
    if len(nasdaq_crash) > 0:
        k = nasdaq_crash["kr_kospi_chg"].dropna()
        if len(k) > 0:
            extreme["nasdaq_crash_minus3"] = {
                "count": len(nasdaq_crash),
                "kospi_mean": round(k.mean(), 3),
                "kospi_worst": round(k.min(), 3),
                "kospi_positive_rate": round((k > 0).mean() * 100, 1),
            }

    vix_high = df[df["us_vix_level"] >= 30]
    if len(vix_high) > 0:
        k = vix_high["kr_kospi_chg"].dropna()
        if len(k) > 0:
            extreme["vix_above_30"] = {
                "count": len(vix_high),
                "kospi_mean": round(k.mean(), 3),
                "kospi_positive_rate": round((k > 0).mean() * 100, 1),
            }

    soxx_crash = df[df["us_soxx_chg"] <= -3.0]
    if len(soxx_crash) > 0:
        ks = soxx_crash["kr_semi_chg"].dropna()
        if len(ks) > 0:
            extreme["soxx_crash_vs_kr_semi"] = {
                "count": len(soxx_crash),
                "kr_semi_mean": round(ks.mean(), 3),
                "kr_semi_worst": round(ks.min(), 3),
            }

    report["extreme_analysis"] = extreme

    # 시가 갭 상관관계
    gap_data = df[["us_overnight_score", "kr_kospi_open_gap"]].dropna()
    if len(gap_data) > 30:
        gap_corr = gap_data["us_overnight_score"].corr(gap_data["kr_kospi_open_gap"])
        bull_gap = gap_data[gap_data["us_overnight_score"] > 20]["kr_kospi_open_gap"]
        bear_gap = gap_data[gap_data["us_overnight_score"] < -20]["kr_kospi_open_gap"]
        report["gap_analysis"] = {
            "score_gap_correlation": round(gap_corr, 3),
            "mean_positive_gap": round(bull_gap.mean(), 3) if len(bull_gap) > 0 else None,
            "mean_negative_gap": round(bear_gap.mean(), 3) if len(bear_gap) > 0 else None,
        }

    return report


# ================================================================
# 7. 메인
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="US-KR 히스토리 백필 + Level 2 DB 구축")
    parser.add_argument("--years", type=int, default=2, help="백필 년수 (기본: 2)")
    args = parser.parse_args()

    start_time = datetime.now()
    years = args.years

    # Step 1: US 다운로드
    us_data, us_failed = download_data(US_SYMBOLS, years=years, label="US Market")

    # Step 2: KR 다운로드
    kr_data, kr_failed = download_data(KR_SYMBOLS, years=years, label="KR Market")
    if kr_failed:
        logger.info("실패한 KR 심볼 대체 시도...")
        recovered = try_kr_fallback(kr_failed, years=years)
        kr_data.update(recovered)

    # Step 3: 날짜 매칭
    pairs_df = build_date_pairs(us_data, kr_data)
    if pairs_df is None or len(pairs_df) == 0:
        logger.error("날짜 매칭 실패 - 종료")
        return

    # Step 4: 레코드 생성
    records = build_history_records(pairs_df, us_data, kr_data)

    # Step 5: DB 저장
    conn = create_database()
    inserted = insert_records(conn, records)

    # Step 6: 분석 리포트
    report = generate_analysis_report(conn)

    # 리포트 출력
    logger.info(f"\n{'='*55}")
    logger.info(f"총 레코드: {report['total_records']}건")
    logger.info(f"기간: {report['date_range']['start']} ~ {report['date_range']['end']}")

    logger.info(f"\n--- US Score 구간별 KOSPI 반응 ---")
    for grade, stats in report.get("score_vs_kospi", {}).items():
        logger.info(f"  {grade:15s} | {stats['count']:3d}건 | "
                     f"KOSPI {stats['kospi_mean']:+.3f}% | "
                     f"상승확률 {stats['kospi_positive_rate']:.1f}%")

    logger.info(f"\n--- 상관관계 TOP 10 ---")
    for i, (pair, corr) in enumerate(list(report.get("correlations_top15", {}).items())[:10]):
        logger.info(f"  {pair:45s} | {corr:+.3f}")

    logger.info(f"\n--- 극단 상황 분석 ---")
    for event, stats in report.get("extreme_analysis", {}).items():
        logger.info(f"  {event}: {stats}")

    if "gap_analysis" in report:
        gap = report["gap_analysis"]
        logger.info(f"\n--- 시가 갭 분석 ---")
        logger.info(f"  Score<->Gap 상관관계: {gap.get('score_gap_correlation', 'N/A')}")

    # Step 7: Level 2 테스트
    logger.info(f"\n--- Level 2 패턴매칭 테스트 ---")
    matcher = PatternMatcher()

    scenarios = [
        {"name": "나스닥 +2%, 반도체 +3%",
         "us_nasdaq_chg": 2.0, "us_sp500_chg": 1.5, "us_vix_chg": -5.0,
         "us_soxx_chg": 3.0, "us_dollar_chg": -0.3},
        {"name": "나스닥 -2%, VIX 급등",
         "us_nasdaq_chg": -2.0, "us_sp500_chg": -1.8, "us_vix_chg": 15.0,
         "us_soxx_chg": -2.5, "us_dollar_chg": 0.5},
        {"name": "혼조세",
         "us_nasdaq_chg": 0.8, "us_sp500_chg": -0.2, "us_vix_chg": 2.0,
         "us_soxx_chg": 1.5, "us_dollar_chg": 0.1},
    ]

    for scenario in scenarios:
        name = scenario.pop("name")
        logger.info(f"\n  시나리오: {name}")
        similar = matcher.find_similar_patterns(scenario)
        if similar is not None:
            analysis = matcher.analyze_patterns(similar)
            if analysis["status"] == "ok":
                logger.info(f"    유사 패턴: {analysis['sample_count']}건")
                logger.info(f"    KOSPI 예측: {analysis['kospi']['mean_chg']:+.3f}% "
                             f"(상승확률 {analysis['kospi']['positive_rate']:.1f}%)")
                logger.info(f"    패턴 보정값: {analysis['pattern_adjustment']:+.1f} "
                             f"(신뢰도 {analysis['confidence']:.2f})")
                for s_name, s_val in analysis.get("sectors", {}).items():
                    logger.info(f"      {s_name:8s}: {s_val['mean_chg']:+.3f}% "
                                 f"(상승 {s_val['positive_rate']:.0f}%)")
        scenario["name"] = name

    # 리포트 저장
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    elapsed = (datetime.now() - start_time).total_seconds()
    conn.close()

    logger.info(f"\n{'='*55}")
    logger.info(f"백필 완료! ({elapsed:.1f}초)")
    logger.info(f"  DB: {DB_PATH} ({inserted}건)")
    logger.info(f"  리포트: {REPORT_PATH}")
    logger.info(f"{'='*55}")


if __name__ == "__main__":
    main()
