"""ETF 마스터 데이터 빌더 — 22개 ETF 전체 (지표 + 수급 + 추천점수).

22개 섹터 ETF의 기술지표·수급·추천점수를 계산하여
data/etf_master.json 에 저장한다.

사용법:
  python scripts/update_etf_master.py           # 기본 실행
  python scripts/update_etf_master.py --no-flow  # 수급 조회 생략 (오프라인)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
ETF_DAILY_DIR = DATA_DIR / "sector_rotation" / "etf_daily"
UNIVERSE_PATH = DATA_DIR / "sector_rotation" / "etf_universe.json"
OUT_PATH = DATA_DIR / "etf_master.json"


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def load_universe() -> dict:
    """etf_universe.json → {sector: {etf_code, etf_name, ...}}."""
    with open(UNIVERSE_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_etf_ohlcv(etf_code: str) -> pd.DataFrame | None:
    """ETF 일별 parquet 로드."""
    path = ETF_DAILY_DIR / f"{etf_code}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ─────────────────────────────────────────────
# 기술지표 계산
# ─────────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1)


def calc_bb_pct(close: pd.Series, period: int = 20) -> float:
    """Bollinger Band % (0~100)."""
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    bb = (close - lower) / (upper - lower) * 100
    val = bb.iloc[-1]
    return round(float(val), 1) if not np.isnan(val) else 50.0


def calc_adx(df: pd.DataFrame, period: int = 14) -> tuple[float, float, float, float]:
    """ADX + DI+/DI- + 전일 ADX."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    plus_dm = high.diff()
    minus_dm = low.diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()

    adx_now = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0
    adx_prev = float(adx.iloc[-2]) if len(adx) > 1 and not np.isnan(adx.iloc[-2]) else adx_now
    pdi = float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 0
    mdi = float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 0
    return round(adx_now, 1), round(adx_prev, 1), round(pdi, 1), round(mdi, 1)


def calc_stoch(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> dict:
    """Slow Stochastic K/D + 골든크로스."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    fast_k = (close - lowest) / (highest - lowest) * 100

    slow_k = fast_k.rolling(d_period).mean()
    slow_d = slow_k.rolling(d_period).mean()

    k_now = float(slow_k.iloc[-1]) if not np.isnan(slow_k.iloc[-1]) else 50
    d_now = float(slow_d.iloc[-1]) if not np.isnan(slow_d.iloc[-1]) else 50
    k_prev = float(slow_k.iloc[-2]) if len(slow_k) > 1 else k_now
    d_prev = float(slow_d.iloc[-2]) if len(slow_d) > 1 else d_now

    return {
        "slow_k": round(k_now, 1),
        "slow_d": round(d_now, 1),
        "golden_cross": k_prev < d_prev and k_now > d_now,
    }


def calc_trix(close: pd.Series, period: int = 12, sig_period: int = 9) -> tuple[float, float, bool]:
    """TRIX + Signal + Bull 여부."""
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    trix = ema3.pct_change() * 100
    sig = trix.rolling(sig_period).mean()
    t = float(trix.iloc[-1]) if not np.isnan(trix.iloc[-1]) else 0
    s = float(sig.iloc[-1]) if not np.isnan(sig.iloc[-1]) else 0
    return round(t, 4), round(s, 4), t > s


def calc_ma_gap(close: pd.Series, period: int) -> float:
    """종가 vs MA 괴리율(%)."""
    ma = close.rolling(period).mean()
    if np.isnan(ma.iloc[-1]) or ma.iloc[-1] == 0:
        return 0.0
    return round(float((close.iloc[-1] / ma.iloc[-1] - 1) * 100), 1)


def calc_returns(close: pd.Series) -> tuple[float, float, float]:
    """1일/5일/20일 수익률(%)."""
    n = len(close)
    ret_1 = round(float((close.iloc[-1] / close.iloc[-2] - 1) * 100), 2) if n >= 2 else 0
    ret_5 = round(float((close.iloc[-1] / close.iloc[-6] - 1) * 100), 2) if n >= 6 else 0
    ret_20 = round(float((close.iloc[-1] / close.iloc[-21] - 1) * 100), 2) if n >= 21 else 0
    return ret_1, ret_5, ret_20


# ─────────────────────────────────────────────
# 수급 조회 (pykrx)
# ─────────────────────────────────────────────

def get_etf_investor_flow(
    etf_code: str, trading_dates: list[str], days: int = 5
) -> tuple[float, float, float, float]:
    """pykrx get_etf_trading_volume_and_value(from, to, ticker)로 ETF 수급 조회.

    Args:
        etf_code: ETF 종목코드 (6자리)
        trading_dates: parquet에서 추출한 최근 거래일 리스트 (YYYYMMDD)
        days: 누적 일수

    Returns: (foreign_5d_억, inst_5d_억, foreign_today_억, inst_today_억)
    """
    try:
        from pykrx import stock as pykrx_stock

        recent = trading_dates[-days:]
        if not recent:
            return 0, 0, 0, 0

        fromdate = recent[0]
        todate = recent[-1]
        col = ("거래대금", "순매수")

        # 5일 합산 (기간 합계)
        df5 = pykrx_stock.get_etf_trading_volume_and_value(fromdate, todate, etf_code)
        f5, i5 = 0.0, 0.0
        if df5 is not None and not df5.empty and len(df5) >= 13:
            seg = df5.iloc[:13]
            if "외국인" in seg.index and col in seg.columns:
                f5 = float(seg.loc["외국인", col]) / 1e8
                i5 = float(seg.loc["기관합계", col]) / 1e8

        time.sleep(0.2)

        # 당일 (마지막 거래일)
        df1 = pykrx_stock.get_etf_trading_volume_and_value(todate, todate, etf_code)
        ft, it = 0.0, 0.0
        if df1 is not None and not df1.empty and len(df1) >= 13:
            seg = df1.iloc[:13]
            if "외국인" in seg.index and col in seg.columns:
                ft = float(seg.loc["외국인", col]) / 1e8
                it = float(seg.loc["기관합계", col]) / 1e8

        return round(f5, 1), round(i5, 1), round(ft, 1), round(it, 1)
    except Exception as e:
        logger.warning("수급 조회 실패 (%s): %s", etf_code, e)
        return 0, 0, 0, 0


# ─────────────────────────────────────────────
# 추천 점수 (100점, 4축)
# ─────────────────────────────────────────────

def calc_score(e: dict) -> tuple[float, str, list[str]]:
    """4축 100점 추천 점수 계산.

    모멘텀(30) + 기술적(30) + 수급(25) + 안전성(15)
    """
    score = 0.0
    reasons: list[str] = []

    # ── 모멘텀 (30점) ──
    r5 = max(0, min(15, e["ret_5"] * 3))
    r20 = max(0, min(15, e["ret_20"] * 1.5))
    score += r5 + r20
    if e["ret_5"] > 3:
        reasons.append(f"5일 +{e['ret_5']:.1f}% 강세")
    if e["ret_20"] > 5:
        reasons.append(f"20일 +{e['ret_20']:.1f}% 상승추세")

    # ── 기술적 (30점) ──
    rsi = e["rsi"]
    rsi_s = 10 if 35 <= rsi <= 65 else (5 if 30 <= rsi <= 70 else 0)
    trix_s = 10 if e["trix_bull"] else 5
    stoch_s = 10 if e["stoch_gx"] else (7 if e["stoch_k"] > e["stoch_d"] else 3)
    score += rsi_s + trix_s + stoch_s

    if 35 <= rsi <= 65:
        reasons.append("RSI 적정")
    elif rsi > 70:
        reasons.append(f"RSI {rsi:.0f} 과열")
    if e["trix_bull"]:
        reasons.append("TRIX 상승전환")
    if e["stoch_gx"]:
        reasons.append("Stoch 골든크로스")

    # ── 수급 (25점) ──
    f5 = max(0, min(10, e["foreign_5d"] * 2))
    i5 = max(0, min(10, e["inst_5d"] * 2))
    dual = 5 if e["is_smart"] else 0
    score += f5 + i5 + dual

    if e["is_smart"]:
        reasons.append("외인+기관 동시매수")
    elif e["foreign_5d"] > 0:
        reasons.append(f"외인 순매수 {e['foreign_5d']:.0f}억")
    elif e["inst_5d"] > 0:
        reasons.append(f"기관 순매수 {e['inst_5d']:.0f}억")

    # ── 안전성 (15점) ──
    bb_s = 5 if e["bb_pct"] < 80 else 0
    adx_s = 5 if 20 <= e["adx"] <= 50 else (3 if e["adx"] < 20 else 0)
    drop_s = 5 if e["ret_5"] > -5 else 0
    score += bb_s + adx_s + drop_s

    if e["bb_pct"] > 95:
        reasons.append("BB% 과열 주의")

    # ── 등급 ──
    grade = (
        "HOT매수" if score >= 70 else
        "매수" if score >= 55 else
        "진입대기" if score >= 40 else
        "관찰"
    )

    return round(score, 1), grade, reasons


# ─────────────────────────────────────────────
# 메인 빌더
# ─────────────────────────────────────────────

def build_etf_master(skip_flow: bool = False) -> dict:
    """22개 ETF 마스터 데이터 생성."""
    universe = load_universe()
    today_str = datetime.now().strftime("%Y%m%d")

    logger.info("=" * 50)
    logger.info("  ETF 마스터 데이터 빌드 (%d개 ETF)", len(universe))
    logger.info("=" * 50)

    if not skip_flow:
        logger.info("  pykrx 수급 조회 활성화")
    else:
        logger.info("  수급 조회 생략 (--no-flow)")

    etfs: list[dict] = []

    for sector, info in universe.items():
        etf_code = info["etf_code"]
        etf_name = info["etf_name"]
        category = info.get("category", "sector")

        # OHLCV 로드
        df = load_etf_ohlcv(etf_code)
        if df is None or len(df) < 30:
            logger.warning("  %s (%s): 데이터 부족 — 건너뜀", sector, etf_code)
            continue

        close = df["close"].astype(float)

        # 기술지표
        rsi = calc_rsi(close)
        bb_pct = calc_bb_pct(close)
        adx_now, adx_prev, plus_di, minus_di = calc_adx(df)
        stoch = calc_stoch(df)
        trix_val, trix_sig, trix_bull = calc_trix(close)
        ma20_gap = calc_ma_gap(close, 20)
        ma60_gap = calc_ma_gap(close, 60)
        ret_1, ret_5, ret_20 = calc_returns(close)

        # 수급 (parquet 거래일 기반으로 pykrx 조회)
        if skip_flow:
            foreign_5d, inst_5d, foreign_today, inst_today = 0, 0, 0, 0
        else:
            trading_dates = [d.strftime("%Y%m%d") for d in df.index[-10:]]
            foreign_5d, inst_5d, foreign_today, inst_today = get_etf_investor_flow(
                etf_code, trading_dates
            )

        is_smart = foreign_5d > 0 and inst_5d > 0

        entry = {
            "sector": sector,
            "etf_code": etf_code,
            "etf_name": etf_name,
            "category": category,
            "close": int(close.iloc[-1]),
            "volume": int(df["volume"].iloc[-1]),
            "ret_1": ret_1,
            "ret_5": ret_5,
            "ret_20": ret_20,
            "rsi": rsi,
            "bb_pct": bb_pct,
            "adx": adx_now,
            "adx_rising": adx_now > adx_prev,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "stoch_k": stoch["slow_k"],
            "stoch_d": stoch["slow_d"],
            "stoch_gx": stoch["golden_cross"],
            "trix": trix_val,
            "trix_signal": trix_sig,
            "trix_bull": trix_bull,
            "ma20_gap": ma20_gap,
            "ma60_gap": ma60_gap,
            "foreign_5d": foreign_5d,
            "inst_5d": inst_5d,
            "foreign_today": foreign_today,
            "inst_today": inst_today,
            "is_smart": is_smart,
        }

        # 추천 점수
        score, grade, reasons = calc_score(entry)
        entry["score"] = score
        entry["grade"] = grade
        entry["reasons"] = reasons

        etfs.append(entry)
        logger.info("  %s (%s): %s점 [%s]", sector, etf_code, score, grade)

    # 점수순 정렬
    etfs.sort(key=lambda x: x["score"], reverse=True)

    result = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "etf_count": len(etfs),
        "etfs": etfs,
    }

    # 저장
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 요약 출력
    grades: dict[str, int] = {}
    for e in etfs:
        g = e["grade"]
        grades[g] = grades.get(g, 0) + 1

    print(f"\n  ETF 마스터 저장: {OUT_PATH}")
    print(f"  전체: {len(etfs)}개 ETF")
    for g in ["HOT매수", "매수", "진입대기", "관찰"]:
        cnt = grades.get(g, 0)
        if cnt > 0:
            print(f"  {g}: {cnt}개")

    # TOP 5 출력
    print("\n  ── TOP 5 ──")
    for i, e in enumerate(etfs[:5], 1):
        print(f"  {i}. [{e['grade']}] {e['sector']} ({e['etf_name']}) — {e['score']}점")
        if e["reasons"]:
            print(f"     사유: {', '.join(e['reasons'])}")

    return result


def main():
    parser = argparse.ArgumentParser(description="ETF 마스터 데이터 빌더")
    parser.add_argument("--no-flow", action="store_true", help="pykrx 수급 조회 생략")
    args = parser.parse_args()

    build_etf_master(skip_flow=args.no_flow)


if __name__ == "__main__":
    main()
