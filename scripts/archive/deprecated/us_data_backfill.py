"""
US Market Data — 백필 + 파생지표 계산

yfinance로 미국 시장 핵심 지표 다운로드 (3년 기본):
  - 대형 지수 (SPY, QQQ, DIA)
  - 섹터 ETF (XLK, XLF, XLE, XLI, XLV, SOXX)
  - 변동성 (VIX, VIX3M)
  - 채권 (TLT, HYG, LQD, ^TNX, ^TYX, ^IRX)
  - 환율 (UUP, JPY=X, KRW=X)
  - 원자재 (GLD, SLV, USO, COPX)
  - 한국 프록시 (EWY)
  - 채권 변동성 (^MOVE)

파생 지표:
  - 1D: 기존 수익률/SMA/상대강도
  - 2D: 크레딧 스프레드 (HY-IG), MOVE z-score, 10Y-3M 커브
  - 3D: 교차자산 60일 롤링 상관관계 (금/달러/채권/유가 vs SPY)

저장: data/us_market/us_daily.parquet

사용법:
    python scripts/us_data_backfill.py [--years 3]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# PYTHONPATH 안전장치
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

US_DIR = Path("data/us_market")
PARQUET_PATH = US_DIR / "us_daily.parquet"

# ================================================================
# 핵심 US 티커 (28개)
# ================================================================
TICKERS = {
    # 대형 지수
    "SPY": "S&P500",
    "QQQ": "NASDAQ100",
    "DIA": "DOW30",
    # 섹터 ETF
    "XLK": "Tech",
    "XLF": "Finance",
    "XLE": "Energy",
    "XLI": "Industrial",
    "XLV": "Healthcare",
    "SOXX": "Semiconductor",
    # 변동성 & 채권 & 달러
    "^VIX": "VIX",
    "TLT": "Treasury20Y",
    "UUP": "DollarIndex",
    # 원자재 (금/은/원유/구리/천연가스/우라늄/농산물/리튬)
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "WTI_Oil",
    "COPX": "Copper_Miners",
    "UNG": "NaturalGas",
    "URA": "Uranium",
    "DBA": "Agriculture",
    "LIT": "Lithium",
    # 한국 프록시
    "EWY": "KoreaETF",
    # ── NIGHTWATCH 기존 ──
    "HYG": "HighYield",         # L0: 하이일드 채권
    "^TNX": "Treasury10Y",      # L1: 10년물 금리
    "^TYX": "Treasury30Y",      # L1: 30년물 금리
    "^VIX3M": "VIX3M",          # L0: VIX 3개월
    "JPY=X": "USDJPY",          # L4: 엔캐리
    "KRW=X": "USDKRW",          # L4: 원/달러
    # ── 2D 신규 (레짐 전환 선행) ──
    "LQD": "InvestmentGrade",   # IG 채권 ETF (크레딧 스프레드)
    "^MOVE": "MOVE_Index",      # 채권시장 변동성
    "^IRX": "Treasury3M",       # 3개월 T-Bill (수익률 커브)
}

# 한국 섹터 매핑 (US ETF → KR 업종)
US_KR_SECTOR_MAP = {
    "XLK": ["반도체", "IT", "소프트웨어", "전자부품"],
    "SOXX": ["반도체", "전자부품"],
    "XLF": ["은행", "증권", "보험", "금융"],
    "XLE": ["에너지", "정유", "화학"],
    "XLI": ["조선", "기계", "건설", "자동차"],
    "XLV": ["제약", "바이오", "의료기기"],
    "GLD": ["철강금속"],
    "USO": ["에너지", "정유", "화학"],
    "COPX": ["조선", "기계", "건설", "자동차", "전자부품"],
}


# ================================================================
# 파생 지표 계산
# ================================================================

def _calc_derived(df: pd.DataFrame) -> pd.DataFrame:
    """파생 지표 (수익률, 이평, 변동성 + 2D/3D 신규)."""

    # ── 1D: 기존 파생지표 ──

    # 주요 지수 일간 수익률
    for prefix in ["spy", "qqq", "dia", "soxx", "ewy"]:
        col = f"{prefix}_close"
        if col in df.columns:
            df[f"{prefix}_ret_1d"] = df[col].pct_change()
            df[f"{prefix}_ret_5d"] = df[col].pct_change(5)
            df[f"{prefix}_sma_20"] = df[col].rolling(20).mean()
            df[f"{prefix}_above_sma20"] = (df[col] > df[f"{prefix}_sma_20"]).astype(int)

    # VIX 파생
    if "vix_close" in df.columns:
        df["vix_sma_20"] = df["vix_close"].rolling(20).mean()
        df["vix_zscore"] = (
            (df["vix_close"] - df["vix_close"].rolling(60).mean())
            / df["vix_close"].rolling(60).std()
        )
        df["vix_spike"] = (df["vix_close"] > df["vix_sma_20"] * 1.2).astype(int)

    # TLT (채권): 상승 = risk-off
    if "tlt_close" in df.columns:
        df["tlt_ret_1d"] = df["tlt_close"].pct_change()

    # 원자재 파생 지표 (금/은/원유/구리/천연가스/우라늄/농산물/리튬)
    for prefix in ["gld", "slv", "uso", "copx", "ung", "ura", "dba", "lit"]:
        col = f"{prefix}_close"
        if col in df.columns:
            df[f"{prefix}_ret_1d"] = df[col].pct_change()
            df[f"{prefix}_ret_5d"] = df[col].pct_change(5)
            df[f"{prefix}_sma_20"] = df[col].rolling(20).mean()
            df[f"{prefix}_above_sma20"] = (df[col] > df[f"{prefix}_sma_20"]).astype(int)

    # 금/은 비율 (Gold-Silver Ratio) — 경기 침체 지표
    if "gld_close" in df.columns and "slv_close" in df.columns:
        df["gold_silver_ratio"] = df["gld_close"] / df["slv_close"].replace(0, pd.NA)

    # 구리/금 비율 (Copper-Gold Ratio) — 경기 확장/수축 지표
    if "copx_close" in df.columns and "gld_close" in df.columns:
        df["copper_gold_ratio"] = df["copx_close"] / df["gld_close"].replace(0, pd.NA)
        df["copper_gold_ratio_sma20"] = df["copper_gold_ratio"].rolling(20).mean()

    # ── NIGHTWATCH 파생지표 ──

    # HYG (하이일드 채권)
    if "hyg_close" in df.columns:
        df["hyg_ret_1d"] = df["hyg_close"].pct_change()
        df["hyg_ret_5d"] = df["hyg_close"].pct_change(5)
        df["hyg_sma_20"] = df["hyg_close"].rolling(20).mean()
        if "spy_close" in df.columns:
            spy_5d = df["spy_close"].pct_change(5)
            hyg_5d = df["hyg_close"].pct_change(5)
            df["hyg_spy_div_5d"] = hyg_5d - spy_5d

    # 10Y/30Y 국채 금리 변화 (basis point)
    for prefix in ["tnx", "tyx"]:
        col = f"{prefix}_close"
        if col in df.columns:
            df[f"{prefix}_change_bp"] = df[col].diff()
            df[f"{prefix}_ret_1d"] = df[col].pct_change()
            df[f"{prefix}_sma_20"] = df[col].rolling(20).mean()

    # 10Y-30Y 스프레드 (정상: 음수, 역전: 양수)
    if "tnx_close" in df.columns and "tyx_close" in df.columns:
        df["yield_spread_10_30"] = df["tnx_close"] - df["tyx_close"]

    # VIX 기간구조 (VIX / VIX3M)
    if "vix_close" in df.columns and "vix3m_close" in df.columns:
        df["vix_term_ratio"] = df["vix_close"] / df["vix3m_close"].replace(0, pd.NA)

    # 환율 수익률
    for prefix in ["jpyx", "krwx"]:
        col = f"{prefix}_close"
        if col in df.columns:
            df[f"{prefix}_ret_1d"] = df[col].pct_change()
            df[f"{prefix}_ret_5d"] = df[col].pct_change(5)

    # 섹터 상대 강도 (vs SPY)
    if "spy_close" in df.columns:
        spy_ret = df["spy_close"].pct_change(5)
        for prefix in ["xlk", "xlf", "xle", "xli", "xlv", "soxx", "ewy",
                        "gld", "slv", "uso", "copx", "ung", "ura", "dba", "lit"]:
            col = f"{prefix}_close"
            if col in df.columns:
                sector_ret = df[col].pct_change(5)
                df[f"{prefix}_rel_spy_5d"] = sector_ret - spy_ret

    # ── 2D: 레짐 전환 선행지표 ──

    # LQD (IG 채권) 수익률
    if "lqd_close" in df.columns:
        df["lqd_ret_1d"] = df["lqd_close"].pct_change()
        df["lqd_ret_5d"] = df["lqd_close"].pct_change(5)

    # 크레딧 스프레드: HY - IG (일간 수익률 차이)
    # 양수 = HY가 IG보다 더 하락 = 크레딧 리스크 확대
    if "hyg_close" in df.columns and "lqd_close" in df.columns:
        hyg_ret = df["hyg_close"].pct_change()
        lqd_ret = df["lqd_close"].pct_change()
        df["credit_spread"] = lqd_ret - hyg_ret  # IG 수익 - HY 수익 (양수=HY 약세)
        df["credit_spread_5d"] = df["credit_spread"].rolling(5).sum()
        cs_mean = df["credit_spread"].rolling(60).mean()
        cs_std = df["credit_spread"].rolling(60).std()
        df["credit_spread_z"] = (df["credit_spread_5d"] - cs_mean) / cs_std.replace(0, pd.NA)

    # 10Y-3M 수익률 커브 (역전 감지)
    if "tnx_close" in df.columns and "irx_close" in df.columns:
        # TNX = 10년물 (% 단위), IRX = 13주 T-Bill (% 단위)
        df["yield_curve_10_3m"] = df["tnx_close"] - df["irx_close"]

    # MOVE 인덱스 (채권시장 변동성)
    if "move_close" in df.columns:
        df["move_level"] = df["move_close"]
        move_mean = df["move_close"].rolling(60).mean()
        move_std = df["move_close"].rolling(60).std()
        df["move_z"] = (df["move_close"] - move_mean) / move_std.replace(0, pd.NA)
        df["move_spike"] = (df["move_z"] > 2.0).astype(int)
    elif "tlt_close" in df.columns:
        # MOVE 대체: TLT 30일 실현변동성
        tlt_ret = df["tlt_close"].pct_change()
        df["move_level"] = tlt_ret.rolling(30).std() * (252 ** 0.5) * 100
        move_mean = df["move_level"].rolling(60).mean()
        move_std = df["move_level"].rolling(60).std()
        df["move_z"] = (df["move_level"] - move_mean) / move_std.replace(0, pd.NA)
        df["move_spike"] = (df["move_z"] > 2.0).astype(int)

    # ── 3D: 교차자산 상관관계 (60일 롤링) ──

    if "spy_close" in df.columns:
        spy_ret = df["spy_close"].pct_change()

        # 금 vs SPY (정상: 음의 상관)
        if "gld_close" in df.columns:
            gld_ret = df["gld_close"].pct_change()
            df["corr_gold_spy_60d"] = spy_ret.rolling(60).corr(gld_ret)

        # 달러 vs SPY (정상: 약한 음의 상관)
        if "uup_close" in df.columns:
            uup_ret = df["uup_close"].pct_change()
            df["corr_dollar_spy_60d"] = spy_ret.rolling(60).corr(uup_ret)

        # 채권 vs SPY (정상: 음의 상관)
        if "tlt_close" in df.columns:
            tlt_ret = df["tlt_close"].pct_change()
            df["corr_bond_spy_60d"] = spy_ret.rolling(60).corr(tlt_ret)

        # 유가 vs SPY (정상: 양의 상관)
        if "uso_close" in df.columns:
            uso_ret = df["uso_close"].pct_change()
            df["corr_oil_spy_60d"] = spy_ret.rolling(60).corr(uso_ret)

    return df


# ================================================================
# 백필 함수
# ================================================================

def backfill(years: int = 3) -> pd.DataFrame:
    """yfinance로 US 시장 데이터 백필."""
    import yfinance as yf

    US_DIR.mkdir(parents=True, exist_ok=True)

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    logger.info(f"US 시장 백필: {start.date()} ~ {end.date()} ({len(TICKERS)} 티커)")

    all_data = {}

    for ticker, label in TICKERS.items():
        try:
            obj = yf.Ticker(ticker)
            df = obj.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if df is None or df.empty:
                logger.warning(f"  {ticker} ({label}): 데이터 없음")
                continue

            # 날짜 인덱스 정리 (timezone 제거)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

            # 필요 컬럼만
            prefix = ticker.replace("^", "").replace("=", "").replace("-", "").lower()
            cols = {
                "Close": f"{prefix}_close",
                "Volume": f"{prefix}_volume",
                "High": f"{prefix}_high",
                "Low": f"{prefix}_low",
            }
            df = df.rename(columns=cols)
            available = [c for c in cols.values() if c in df.columns]
            df = df[available]

            all_data[ticker] = df
            logger.info(f"  {ticker} ({label}): {len(df)}일")
        except Exception as e:
            logger.warning(f"  {ticker} ({label}): 실패 — {e}")

    if not all_data:
        logger.error("수집된 데이터 없음!")
        return pd.DataFrame()

    # 날짜 기준 병합
    merged = pd.DataFrame()
    for ticker, df in all_data.items():
        if merged.empty:
            merged = df
        else:
            merged = merged.join(df, how="outer")

    # 결측값 전방 채움 (휴일 등)
    merged = merged.ffill()

    # 파생 지표 계산
    merged = _calc_derived(merged)

    # 저장
    merged.to_parquet(PARQUET_PATH)
    logger.info(f"저장: {PARQUET_PATH} ({len(merged)}일 × {len(merged.columns)}컬럼)")

    return merged


def main():
    parser = argparse.ArgumentParser(description="US Market Data 백필 (2D/3D 포함)")
    parser.add_argument("--years", type=int, default=3, help="백필 년수 (기본: 3)")
    args = parser.parse_args()

    backfill(args.years)


if __name__ == "__main__":
    main()
