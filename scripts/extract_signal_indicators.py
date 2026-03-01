"""
포물선 시작점 지표값 자동 추출 스크립트

Usage:
    python scripts/extract_signal_indicators.py
    python scripts/extract_signal_indicators.py --input data/signal_points.csv
    python scripts/extract_signal_indicators.py --analyze   # 패턴 분석만
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROCESSED_DIR = Path("data/processed")
DEFAULT_INPUT = Path("data/signal_points.csv")
OUTPUT_DIR = Path("data/signal_analysis")


# =========================================================
# 1. 데이터 로드
# =========================================================

def load_parquet(stock_code: str) -> pd.DataFrame | None:
    """종목코드로 processed parquet 로드. 없으면 pykrx에서 다운로드+지표 계산."""
    code = str(stock_code).zfill(6)
    path = PROCESSED_DIR / f"{code}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df.attrs["ticker"] = code
        return df

    # processed에 없으면 pykrx에서 다운로드
    print(f"  → pykrx 다운로드 중... ", end="")
    df = _download_and_calc_indicators(code)
    if df is not None:
        df.attrs["ticker"] = code
        print(f"OK ({len(df)}일)")
    return df


def _download_and_calc_indicators(code: str) -> pd.DataFrame | None:
    """pykrx에서 OHLCV + 수급 다운로드 → 기본 지표 계산."""
    try:
        from pykrx import stock as krx
        import time

        # OHLCV (최근 5년)
        df = krx.get_market_ohlcv("20190101", "20260214", code)
        if df is None or df.empty or len(df) < 200:
            print(f"SKIP (데이터 부족: {len(df) if df is not None else 0}일)")
            return None

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]
        # 한글 컬럼명 → 영문
        rename = {"시가": "open", "고가": "high", "저가": "low", "종가": "close",
                  "거래량": "volume", "거래대금": "trading_value"}
        df.rename(columns=rename, inplace=True)
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                return None

        # 투자자별 매매 (외국인, 기관)
        time.sleep(0.3)
        try:
            inv = krx.get_market_trading_value_by_date("20190101", "20260214", code)
            if inv is not None and not inv.empty:
                inv.index = pd.to_datetime(inv.index)
                for col_name in ["외국인합계", "기관합계"]:
                    if col_name in inv.columns:
                        df[col_name] = inv[col_name].reindex(df.index).fillna(0)
        except Exception:
            pass

        # 기본 지표 계산
        _calc_basic_indicators(df)
        return df

    except Exception as e:
        print(f"다운로드 실패: {e}")
        return None


def _calc_basic_indicators(df: pd.DataFrame):
    """기본 기술적 지표 일괄 계산."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # SMA
    for p in [5, 20, 60, 120, 200]:
        df[f"sma_{p}"] = close.rolling(p).mean()

    # EMA
    df["ema_8"] = close.ewm(span=8).mean()
    df["ema_24"] = close.ewm(span=24).mean()

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["ATR_pct"] = df["atr_14"] / close * 100

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Stochastic RSI
    rsi = df["rsi_14"]
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    df["stoch_rsi_k"] = ((rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100).rolling(3).mean()
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(3).mean()

    # ADX
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    cond = plus_dm > minus_dm
    plus_dm = plus_dm.where(cond, 0)
    minus_dm = minus_dm.where(~cond, 0)
    atr14 = df["atr_14"].replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    df["adx_14"] = dx.rolling(14).mean()

    # Bollinger Bands
    sma20 = df["sma_20"]
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"] = bb_range / sma20
    df["bb_position"] = (close - df["bb_lower"]) / bb_range

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    df["macd_histogram_prev"] = df["macd_histogram"].shift(1)

    # TRIX
    ema1 = close.ewm(span=12).mean()
    ema2 = ema1.ewm(span=12).mean()
    ema3 = ema2.ewm(span=12).mean()
    df["trix"] = ema3.pct_change() * 100
    df["trix_signal"] = df["trix"].ewm(span=9).mean()
    df["trix_golden_cross"] = ((df["trix"] > df["trix_signal"]) &
                                (df["trix"].shift(1) <= df["trix_signal"].shift(1))).astype(int)
    df["trix_prev"] = df["trix"].shift(1)
    df["trix_signal_prev"] = df["trix_signal"].shift(1)
    df["rsi_prev"] = df["rsi_14"].shift(1)

    # Volume
    df["volume_ma5"] = volume.rolling(5).mean()
    df["volume_ma20"] = volume.rolling(20).mean()
    df["volume_surge_ratio"] = volume / df["volume_ma20"].replace(0, np.nan)
    vm5 = df["volume_ma5"].replace(0, np.nan)
    df["volume_contraction_ratio"] = df["volume_ma5"] / df["volume_ma20"].replace(0, np.nan)

    # OBV
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv

    # OBV trend
    obv_s = pd.Series(obv, index=df.index, dtype=float)
    df["obv_trend_5d"] = obv_s.diff(5)

    # High/Low
    df["high_20"] = high.rolling(20).max()
    df["high_60"] = high.rolling(60).max()
    df["high_120"] = high.rolling(120).max()
    df["high_252"] = high.rolling(252).max()
    df["pct_of_52w_high"] = close / df["high_252"].replace(0, np.nan)

    # Pullback ATR
    df["pullback_atr"] = (df["high_20"] - close) / df["atr_14"].replace(0, np.nan)

    # EMA Curvature (2차 미분)
    ema24 = df["ema_24"]
    d1 = ema24.diff()
    d2 = d1.diff()
    df["ema_curvature"] = d2
    df["ema_curvature_prev"] = d2.shift(1)

    # Linear regression slopes
    def _linreg_slope(series, window):
        result = pd.Series(index=series.index, dtype=float)
        x = np.arange(window)
        for i in range(window - 1, len(series)):
            y = series.iloc[i - window + 1 : i + 1].values.astype(float)
            valid = ~np.isnan(y)
            if valid.sum() >= window // 2:
                result.iloc[i] = np.polyfit(x[valid], y[valid], 1)[0]
        return result

    df["linreg_slope_20"] = _linreg_slope(close, 20)
    df["linreg_slope_5"] = _linreg_slope(close, 5)

    # Price/OBV trends
    df["price_trend_5d"] = close.pct_change(5) * 100
    df["prev_high"] = high.shift(1)

    # Foreign
    if "외국인합계" in df.columns:
        fn = df["외국인합계"].fillna(0)
        df["foreign_net_5d"] = fn.rolling(5).sum()
        df["foreign_net_20d"] = fn.rolling(20).sum()
        # 연속 매수일
        consec = []
        streak = 0
        for v in fn:
            if v > 0:
                streak += 1
            else:
                streak = 0
            consec.append(streak)
        df["foreign_consecutive_buy"] = consec
    else:
        df["foreign_net_5d"] = 0
        df["foreign_net_20d"] = 0
        df["foreign_consecutive_buy"] = 0

    # Short (기본값)
    for col in ["short_balance_chg_5d", "short_interest_pct", "short_cover_signal"]:
        if col not in df.columns:
            df[col] = 0

    # OU, Martin 등 (기본값 — processed에만 있는 고급 지표)
    for col in ["ou_z", "half_life", "snr", "smart_z", "ema2_norm",
                "martin_dead_zone", "slope_ma60", "days_above_sma20",
                "higher_low_3d", "higher_low_5d", "lower_tail_ratio",
                "is_bullish", "gap_up_pct", "foreign_vol_confirm",
                "sentiment_pessimism", "sentiment_extreme"]:
        if col not in df.columns:
            df[col] = 0


def find_nearest_date(df: pd.DataFrame, target_date: str) -> int:
    """target_date에 가장 가까운 거래일 인덱스 반환."""
    target = pd.Timestamp(target_date)
    # 정확한 날짜가 있으면 사용
    if target in df.index:
        return df.index.get_loc(target)
    # searchsorted로 정확하게 이전 거래일 찾기
    pos = df.index.searchsorted(target, side="right") - 1
    if pos >= 0:
        return pos
    # target이 모든 날짜보다 이전이면 첫 번째 거래일
    return 0


# =========================================================
# 2. 지표 추출 (30+ 지표)
# =========================================================

def extract_indicators(df: pd.DataFrame, idx: int) -> dict:
    """signal_date 시점의 모든 지표값 추출."""
    row = df.iloc[idx]
    result = {}

    # ── 기본 정보 ──
    result["date"] = str(df.index[idx].date())
    result["close"] = row.get("close", 0)
    result["open"] = row.get("open", 0)
    result["high"] = row.get("high", 0)
    result["low"] = row.get("low", 0)

    # ── 가격 관련 ──
    result["bb_width"] = _safe(row, "bb_width")
    result["bb_pctB"] = _safe(row, "bb_position")  # %B
    result["bb_width_pctile_20d"] = _calc_percentile_rank(
        df, "bb_width", idx, window=20
    )
    result["atr_14"] = _safe(row, "atr_14")
    result["atr_pct"] = _safe(row, "ATR_pct")

    # 20일 High-Low Range / ATR (베이스 타이트함)
    if idx >= 19:
        hl_range = df["high"].iloc[idx - 19 : idx + 1].max() - df["low"].iloc[idx - 19 : idx + 1].min()
        atr = _safe(row, "atr_14") or 1.0
        result["range_atr_ratio_20d"] = round(hl_range / atr, 4) if atr > 0 else 0
    else:
        result["range_atr_ratio_20d"] = 0

    # 이평 대비 괴리율
    close = row.get("close", 0)
    sma60 = _safe(row, "sma_60")
    sma120 = _safe(row, "sma_120")
    result["gap_from_sma60_pct"] = _pct_gap(close, sma60)
    result["gap_from_sma120_pct"] = _pct_gap(close, sma120)

    # 52주 고점 대비
    result["pct_of_52w_high"] = _safe(row, "pct_of_52w_high")

    # ── 추세 관련 ──
    result["adx_14"] = _safe(row, "adx_14")

    # SMA 정렬 상태
    sma5 = _safe(row, "sma_5")
    sma20 = _safe(row, "sma_20")
    sma60_val = _safe(row, "sma_60")
    sma120_val = _safe(row, "sma_120")
    result["sma5"] = sma5
    result["sma20"] = sma20
    result["sma60"] = sma60_val
    result["sma120"] = sma120_val

    # 정배열: 5>20>60>120 = 4, 역배열: 120>60>20>5 = -4
    vals = [sma5, sma20, sma60_val, sma120_val]
    if all(v and v > 0 for v in vals):
        alignment = sum(1 for i in range(3) if vals[i] > vals[i + 1])
        result["sma_alignment"] = alignment  # 3=완전정배열, 0=완전역배열
    else:
        result["sma_alignment"] = -1

    result["sma60_vs_sma120"] = "위" if (sma60_val or 0) > (sma120_val or 0) else "아래"

    # TRIX
    result["trix"] = _safe(row, "trix")
    result["trix_signal"] = _safe(row, "trix_signal")
    result["trix_golden_cross"] = int(row.get("trix_golden_cross", 0) or 0)
    trix_prev = _safe(row, "trix_prev")
    trix_sig_prev = _safe(row, "trix_signal_prev")
    trix_val = _safe(row, "trix")
    trix_sig = _safe(row, "trix_signal")
    if trix_val is not None and trix_sig is not None and trix_prev is not None and trix_sig_prev is not None:
        # 교차 상태: 위=trix>signal, 아래=trix<signal
        result["trix_cross_state"] = "위" if trix_val > trix_sig else "아래"
        # 교차 직전: 이전엔 아래였는데 지금 위 = 골든크로스
        if trix_prev < trix_sig_prev and trix_val >= trix_sig:
            result["trix_cross_event"] = "골든크로스"
        elif trix_prev > trix_sig_prev and trix_val <= trix_sig:
            result["trix_cross_event"] = "데드크로스"
        else:
            result["trix_cross_event"] = "없음"
    else:
        result["trix_cross_state"] = "N/A"
        result["trix_cross_event"] = "N/A"

    # RSI
    result["rsi_14"] = _safe(row, "rsi_14")
    result["stoch_rsi_k"] = _safe(row, "stoch_rsi_k")
    result["stoch_rsi_d"] = _safe(row, "stoch_rsi_d")

    # MACD
    result["macd"] = _safe(row, "macd")
    result["macd_signal"] = _safe(row, "macd_signal")
    result["macd_histogram"] = _safe(row, "macd_histogram")

    # ── 거래량 관련 ──
    result["volume"] = int(row.get("volume", 0) or 0)
    result["volume_ma20"] = _safe(row, "volume_ma20")
    result["volume_surge_ratio"] = _safe(row, "volume_surge_ratio")  # 당일/20일평균
    result["volume_pctile_20d"] = _calc_percentile_rank(
        df, "volume", idx, window=20
    )

    # OBV 관련
    result["obv"] = int(row.get("obv", 0) or 0)
    result["obv_trend_5d"] = _safe(row, "obv_trend_5d")

    # OBV 20일 기울기 (선형회귀)
    if idx >= 19:
        obv_20 = df["obv"].iloc[idx - 19 : idx + 1].values.astype(float)
        x = np.arange(20)
        valid = ~np.isnan(obv_20)
        if valid.sum() >= 10:
            slope = np.polyfit(x[valid], obv_20[valid], 1)[0]
            result["obv_slope_20d"] = round(slope, 2)
        else:
            result["obv_slope_20d"] = 0
    else:
        result["obv_slope_20d"] = 0

    # OBV vs 가격 괴리 (가격 횡보인데 OBV 상승)
    if idx >= 19:
        price_20 = df["close"].iloc[idx - 19 : idx + 1].values.astype(float)
        obv_20 = df["obv"].iloc[idx - 19 : idx + 1].values.astype(float)
        x = np.arange(20)
        p_valid = ~np.isnan(price_20)
        o_valid = ~np.isnan(obv_20)
        p_slope = np.polyfit(x[p_valid], price_20[p_valid], 1)[0] if p_valid.sum() >= 10 else 0
        o_slope = np.polyfit(x[o_valid], obv_20[o_valid], 1)[0] if o_valid.sum() >= 10 else 0
        # 정규화: 가격 기울기를 ATR로, OBV를 평균 OBV로
        atr = _safe(row, "atr_14") or 1
        p_norm = p_slope / atr if atr > 0 else 0
        obv_mean = np.mean(np.abs(obv_20[o_valid])) or 1
        o_norm = o_slope / obv_mean
        result["price_slope_20d_norm"] = round(p_norm, 4)
        result["obv_slope_20d_norm"] = round(o_norm, 4)
        # 괴리: 가격 평탄 + OBV 상승 = 양의 괴리 (선행 발산)
        if abs(p_norm) < 0.05 and o_norm > 0.01:
            result["obv_divergence"] = "양성(OBV선행)"
        elif p_norm < -0.05 and o_norm > 0.01:
            result["obv_divergence"] = "강한양성(하락중OBV상승)"
        elif p_norm > 0.05 and o_norm < -0.01:
            result["obv_divergence"] = "음성(상승중OBV하락)"
        else:
            result["obv_divergence"] = "중립"
    else:
        result["price_slope_20d_norm"] = 0
        result["obv_slope_20d_norm"] = 0
        result["obv_divergence"] = "N/A"

    # ── 수급 관련 ──
    result["foreign_net_5d"] = _safe(row, "foreign_net_5d")
    result["foreign_net_20d"] = _safe(row, "foreign_net_20d")
    result["foreign_consecutive_buy"] = int(row.get("foreign_consecutive_buy", 0) or 0)

    # 기관 20일 누적 순매수
    if "기관합계" in df.columns and idx >= 19:
        inst_20d = df["기관합계"].iloc[idx - 19 : idx + 1].fillna(0).sum()
        result["inst_net_20d"] = float(inst_20d)
    else:
        result["inst_net_20d"] = 0

    # ── 곡률 관련 ──
    result["ema_curvature"] = _safe(row, "ema_curvature")
    result["ema_curvature_prev"] = _safe(row, "ema_curvature_prev")
    curv = _safe(row, "ema_curvature")
    curv_prev = _safe(row, "ema_curvature_prev")
    if curv is not None:
        result["curvature_sign"] = "양" if curv > 0 else "음" if curv < 0 else "0"
    else:
        result["curvature_sign"] = "N/A"
    if curv is not None and curv_prev is not None:
        result["curvature_accel"] = round(curv - curv_prev, 6)
    else:
        result["curvature_accel"] = 0

    # ── 심리지표(10) ──
    # Psychological Line = 최근 10일 중 양봉(종가>전일종가) 비율 × 100
    if idx >= 10:
        closes = df["close"].iloc[idx - 10 : idx + 1].values.astype(float)
        up_days = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
        result["psychological_line_10"] = round(up_days / 10 * 100, 1)
    else:
        result["psychological_line_10"] = 50.0

    # ── 공매도 관련 ──
    result["short_balance_chg_5d"] = _safe(row, "short_balance_chg_5d")
    result["short_interest_pct"] = _safe(row, "short_interest_pct")
    result["short_cover_signal"] = int(row.get("short_cover_signal", 0) or 0)

    # ── 변동성 수축 지표 ──
    result["bb_width_pctile_60d"] = _calc_percentile_rank(
        df, "bb_width", idx, window=60
    )
    result["volume_contraction_ratio"] = _safe(row, "volume_contraction_ratio")

    # ── 추가: 선형회귀 기울기 ──
    result["linreg_slope_20"] = _safe(row, "linreg_slope_20")
    result["linreg_slope_5"] = _safe(row, "linreg_slope_5")

    # ── 추가: OU 파라미터 (평균회귀) ──
    result["ou_z"] = _safe(row, "ou_z")
    result["half_life"] = _safe(row, "half_life")
    result["snr"] = _safe(row, "snr")

    # ── 추가: Martin 지표 ──
    result["ema2_norm"] = _safe(row, "ema2_norm")
    result["martin_dead_zone"] = int(row.get("martin_dead_zone", 0) or 0)

    # ── SignalEngine 평가 (Zone Score, R:R 등) ──
    # processed parquet에서만 작동 (pykrx 다운로드 데이터는 OU 등 미포함)
    try:
        from src.signal_engine import SignalEngine
        engine = SignalEngine("config/settings.yaml")
        ticker_code = df.attrs.get("ticker", "000000")
        se_result = engine.calculate_signal(ticker_code, df, idx)
        result["zone_score"] = se_result.get("zone_score", 0)
        result["risk_reward"] = se_result.get("risk_reward_ratio", 0)
        result["trigger_type"] = se_result.get("trigger_type", "none")
        result["grade"] = se_result.get("grade", "D")
        result["entry_price"] = se_result.get("entry_price", 0)
        result["stop_loss"] = se_result.get("stop_loss", 0)
        result["target_price"] = se_result.get("target_price", 0)
    except Exception as e:
        result["zone_score"] = 0
        result["risk_reward"] = 0
        result["trigger_type"] = "N/A"
        result["grade"] = "N/A"
        result["entry_price"] = 0
        result["stop_loss"] = 0
        result["target_price"] = 0

    return result


# =========================================================
# 헬퍼
# =========================================================

def _safe(row, col, default=0):
    """안전하게 값 추출, NaN → default."""
    val = row.get(col, default)
    if pd.isna(val):
        return default
    return round(float(val), 6) if isinstance(val, (float, np.floating)) else val


def _pct_gap(price, ma):
    """이평 대비 괴리율 (%)."""
    if not ma or ma == 0:
        return 0
    return round((price - ma) / ma * 100, 2)


def _calc_percentile_rank(df: pd.DataFrame, col: str, idx: int, window: int = 20) -> float:
    """col의 현재값이 최근 window일 내에서 하위 몇 %인지."""
    if col not in df.columns or idx < window:
        return 50.0
    values = df[col].iloc[max(0, idx - window + 1) : idx + 1].dropna()
    if len(values) < 5:
        return 50.0
    current = values.iloc[-1]
    rank = (values < current).sum()
    return round(rank / len(values) * 100, 1)


# =========================================================
# 3. 패턴 분석
# =========================================================

def analyze_patterns(result_df: pd.DataFrame) -> str:
    """50개 시작점의 지표값 통계 분석."""
    lines = []
    lines.append("=" * 70)
    lines.append("포물선 시작점 지표값 패턴 분석")
    lines.append(f"총 {len(result_df)}개 시작점")
    lines.append("=" * 70)
    lines.append("")

    # 분석할 수치형 컬럼
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    # 제외: 가격 자체 (종목마다 다르므로)
    exclude = {"close", "open", "high", "low", "volume", "obv",
               "sma5", "sma20", "sma60", "sma120", "volume_ma20",
               "foreign_net_5d", "foreign_net_20d", "inst_net_20d",
               "entry_price", "stop_loss", "target_price"}
    analyze_cols = [c for c in numeric_cols if c not in exclude]

    # 핵심 90% 공유 조건 테이블
    lines.append("─" * 70)
    lines.append("[ 핵심 90% 공유 조건 ]")
    lines.append("─" * 70)
    shared_conditions = []

    for col in sorted(analyze_cols):
        values = result_df[col].dropna()
        if len(values) < 3:
            continue

        med = values.median()
        mean = values.mean()
        std = values.std()
        vmin = values.min()
        vmax = values.max()
        q10 = values.quantile(0.10)
        q90 = values.quantile(0.90)

        # 90% 공유 범위
        n_total = len(values)
        range_90 = f"[{q10:.4f}, {q90:.4f}]"

        shared_conditions.append({
            "지표": col,
            "평균": f"{mean:.4f}",
            "중앙값": f"{med:.4f}",
            "표준편차": f"{std:.4f}",
            "최소": f"{vmin:.4f}",
            "최대": f"{vmax:.4f}",
            "90%범위": range_90,
            "n": n_total,
        })

    if shared_conditions:
        cond_df = pd.DataFrame(shared_conditions)
        lines.append(cond_df.to_string(index=False))
    lines.append("")

    # 범주형 분석
    cat_cols = ["sma_alignment", "sma60_vs_sma120", "trix_cross_state",
                "trix_cross_event", "obv_divergence", "curvature_sign",
                "trigger_type", "grade"]
    lines.append("─" * 70)
    lines.append("[ 범주형 지표 분포 ]")
    lines.append("─" * 70)
    for col in cat_cols:
        if col in result_df.columns:
            counts = result_df[col].value_counts()
            total = len(result_df[col].dropna())
            lines.append(f"\n{col} (n={total}):")
            for val, cnt in counts.items():
                pct = cnt / total * 100
                bar = "█" * int(pct / 5)
                lines.append(f"  {val}: {cnt}건 ({pct:.1f}%) {bar}")

    # 핵심 발견 요약
    lines.append("")
    lines.append("=" * 70)
    lines.append("[ 핵심 발견 요약 — 파라미터 역도출 근거 ]")
    lines.append("=" * 70)

    # BB Width 하위 %
    if "bb_width_pctile_20d" in result_df.columns:
        bb_pctile = result_df["bb_width_pctile_20d"].dropna()
        below_30 = (bb_pctile <= 30).sum()
        lines.append(f"BB Width 하위 30% 이내: {below_30}/{len(bb_pctile)} ({below_30/len(bb_pctile)*100:.0f}%)")

    if "bb_width_pctile_60d" in result_df.columns:
        bb_pctile60 = result_df["bb_width_pctile_60d"].dropna()
        below_20 = (bb_pctile60 <= 20).sum()
        lines.append(f"BB Width 60일 하위 20% 이내: {below_20}/{len(bb_pctile60)} ({below_20/len(bb_pctile60)*100:.0f}%)")

    if "adx_14" in result_df.columns:
        adx = result_df["adx_14"].dropna()
        below_25 = ((adx >= 10) & (adx <= 25)).sum()
        lines.append(f"ADX 10~25 범위: {below_25}/{len(adx)} ({below_25/len(adx)*100:.0f}%)")

    if "rsi_14" in result_df.columns:
        rsi = result_df["rsi_14"].dropna()
        in_range = ((rsi >= 30) & (rsi <= 55)).sum()
        lines.append(f"RSI 30~55 범위: {in_range}/{len(rsi)} ({in_range/len(rsi)*100:.0f}%)")

    if "volume_surge_ratio" in result_df.columns:
        vsr = result_df["volume_surge_ratio"].dropna()
        low_vol = (vsr < 1.0).sum()
        lines.append(f"거래량 < 20일 평균: {low_vol}/{len(vsr)} ({low_vol/len(vsr)*100:.0f}%)")

    if "obv_divergence" in result_df.columns:
        obv_div = result_df["obv_divergence"]
        pos = (obv_div.isin(["양성(OBV선행)", "강한양성(하락중OBV상승)"])).sum()
        lines.append(f"OBV 양성 괴리: {pos}/{len(obv_div)} ({pos/len(obv_div)*100:.0f}%)")

    if "psychological_line_10" in result_df.columns:
        psy = result_df["psychological_line_10"].dropna()
        below_40 = (psy <= 40).sum()
        lines.append(f"심리지표 <= 40: {below_40}/{len(psy)} ({below_40/len(psy)*100:.0f}%)")

    return "\n".join(lines)


# =========================================================
# 4. 파라미터 역도출
# =========================================================

def derive_parameters(result_df: pd.DataFrame) -> str:
    """90% 공유 조건에서 게이트/트리거 파라미터 역도출."""
    lines = []
    lines.append("=" * 70)
    lines.append("파라미터 역도출 결과")
    lines.append("=" * 70)
    lines.append("")

    def q_range(col, lo=0.05, hi=0.95):
        vals = result_df[col].dropna()
        if len(vals) < 5:
            return None, None, None
        return vals.quantile(lo), vals.median(), vals.quantile(hi)

    # G1 ADX
    lo, med, hi = q_range("adx_14")
    if lo is not None:
        lines.append(f"[G1] ADX 하한 → {lo:.1f} (현재 18, 시작점 5% = {lo:.1f})")
        lines.append(f"      ADX 중앙값: {med:.1f}, 상한 95%: {hi:.1f}")

    # G2 Pullback
    lo, med, hi = q_range("pct_of_52w_high")
    if lo is not None:
        lines.append(f"[G3] 52주 고점 비율 → 5%: {lo:.3f}, 중앙: {med:.3f}, 95%: {hi:.3f}")

    # BB Width
    lo, med, hi = q_range("bb_width_pctile_20d")
    if lo is not None:
        lines.append(f"[P1] BB Width 20일 Pctile → 중앙: {med:.1f}%, 90%범위: [{lo:.1f}, {hi:.1f}]")

    lo60, med60, hi60 = q_range("bb_width_pctile_60d")
    if lo60 is not None:
        lines.append(f"[P1] BB Width 60일 Pctile → 중앙: {med60:.1f}%, 90%범위: [{lo60:.1f}, {hi60:.1f}]")

    # RSI
    lo, med, hi = q_range("rsi_14")
    if lo is not None:
        lines.append(f"[T2] RSI 범위 → [{lo:.1f}, {hi:.1f}] (중앙: {med:.1f})")

    # Volume
    lo, med, hi = q_range("volume_surge_ratio")
    if lo is not None:
        lines.append(f"[T2] Volume Surge Ratio → 중앙: {med:.2f}, 90%범위: [{lo:.2f}, {hi:.2f}]")

    # EMA Curvature
    lo, med, hi = q_range("ema_curvature")
    if lo is not None:
        lines.append(f"[T3] EMA Curvature → 중앙: {med:.6f}, 90%범위: [{lo:.6f}, {hi:.6f}]")

    # Psychological Line
    lo, med, hi = q_range("psychological_line_10")
    if lo is not None:
        lines.append(f"[심리] 심리지표(10) → 중앙: {med:.0f}%, 90%범위: [{lo:.0f}, {hi:.0f}]")

    # Zone Score
    lo, med, hi = q_range("zone_score")
    if lo is not None:
        lines.append(f"[Zone] Zone Score → 중앙: {med:.3f}, 90%범위: [{lo:.3f}, {hi:.3f}]")

    # R:R
    lo, med, hi = q_range("risk_reward")
    if lo is not None:
        lines.append(f"[R:R] Risk Reward → 중앙: {med:.2f}, 90%범위: [{lo:.2f}, {hi:.2f}]")

    lines.append("")
    lines.append("─" * 70)
    lines.append("위 범위를 settings.yaml 게이트/트리거 파라미터에 적용하세요.")
    lines.append("적용 후 3구간 백테스트(Part 6)로 검증합니다.")

    return "\n".join(lines)


# =========================================================
# 메인
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="포물선 시작점 지표값 추출")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="시작점 CSV 파일 경로")
    parser.add_argument("--analyze", action="store_true",
                        help="기존 추출 결과에서 패턴 분석만 실행")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / "signal_indicators.csv"
    analysis_txt = OUTPUT_DIR / "pattern_analysis.txt"
    params_txt = OUTPUT_DIR / "derived_parameters.txt"

    if args.analyze:
        if not output_csv.exists():
            print("! 추출 결과 없음. 먼저 추출을 실행하세요.")
            return
        result_df = pd.read_csv(output_csv)
        print(f"기존 결과 로드: {len(result_df)}건")
    else:
        # CSV 로드
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"! 입력 파일 없음: {input_path}")
            return

        signals = pd.read_csv(input_path)
        print(f"시작점 CSV 로드: {len(signals)}건")
        print()

        # 지표 추출
        results = []
        for i, row in signals.iterrows():
            code = str(row["stock_code"]).zfill(6)
            name = row["stock_name"]
            date = str(row["signal_date"])
            direction = row.get("direction", "long")
            notes = row.get("notes", "")

            print(f"[{i+1}/{len(signals)}] {name}({code}) {date} ... ", end="")

            df = load_parquet(code)
            if df is None:
                print("SKIP (parquet 없음)")
                continue

            idx = find_nearest_date(df, date)
            if idx < 0:
                print("SKIP (날짜 없음)")
                continue

            actual_date = str(df.index[idx].date())
            if actual_date != date:
                print(f"(날짜 조정: {date} → {actual_date}) ", end="")

            indicators = extract_indicators(df, idx)
            indicators["stock_code"] = code
            indicators["stock_name"] = name
            indicators["signal_date_input"] = date
            indicators["direction"] = direction
            indicators["notes"] = notes

            results.append(indicators)
            print("OK")

        if not results:
            print("\n! 추출된 결과 없음")
            return

        result_df = pd.DataFrame(results)

        # 컬럼 순서: 메타 → 지표
        meta_cols = ["stock_code", "stock_name", "signal_date_input", "date",
                     "direction", "notes"]
        other_cols = [c for c in result_df.columns if c not in meta_cols]
        result_df = result_df[meta_cols + sorted(other_cols)]

        result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n지표 추출 완료: {output_csv} ({len(result_df)}건 × {len(result_df.columns)}컬럼)")

    # 패턴 분석
    print("\n패턴 분석 중...")
    analysis = analyze_patterns(result_df)
    with open(analysis_txt, "w", encoding="utf-8") as f:
        f.write(analysis)
    print(analysis)
    print(f"\n분석 결과 저장: {analysis_txt}")

    # 파라미터 역도출
    print("\n파라미터 역도출 중...")
    params = derive_parameters(result_df)
    with open(params_txt, "w", encoding="utf-8") as f:
        f.write(params)
    print(params)
    print(f"\n역도출 결과 저장: {params_txt}")


if __name__ == "__main__":
    main()
