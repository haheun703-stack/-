"""
개별 종목 분석 스크립트 - v4.0 라이브 트레이딩 파이프라인 분석

사용법: python scripts/analyze_stock.py --ticker 039130
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

from src.market_signal_scanner import MarketSignalScanner

# Windows cp949 인코딩 문제 해결
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_stock_data(ticker: str) -> pd.DataFrame:
    """stock_data_daily에서 종목 데이터 로드."""
    data_dir = Path("stock_data_daily")
    for f in data_dir.glob(f"*_{ticker}.csv"):
        df = pd.read_csv(f, index_col="Date", parse_dates=True)
        # 컬럼명 소문자 변환 (signal_engine 호환)
        df.columns = [c.strip() for c in df.columns]
        return df, f.stem.split("_")[0]  # (DataFrame, 종목명)
    raise FileNotFoundError(f"종목 {ticker} CSV 파일을 찾을 수 없습니다.")


def compute_extra_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """signal_engine에서 필요한 추가 지표 계산."""
    # volume_surge_ratio (20일 평균 대비)
    df["volume_surge_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # slope_ma60 (5일 기울기)
    if "MA60" in df.columns:
        df["slope_ma60"] = df["MA60"].pct_change(5) * 100

    # ATR 14 (이미 ATR 컬럼 있음)
    df["atr_14"] = df["ATR"]

    # close 소문자
    df["close"] = df["Close"]
    df["high"] = df["High"]
    df["low"] = df["Low"]
    df["open"] = df["Open"]
    df["volume"] = df["Volume"]

    # Bollinger Band Width
    if "Upper_Band" in df.columns and "Lower_Band" in df.columns:
        df["bb_width"] = (df["Upper_Band"] - df["Lower_Band"]) / df["Close"] * 100

    # 수급 연속일수
    if "Foreign_Net" in df.columns:
        df["foreign_streak"] = _calc_streak(df["Foreign_Net"])
    if "Inst_Net" in df.columns:
        df["inst_streak"] = _calc_streak(df["Inst_Net"])

    return df


def _calc_streak(series: pd.Series) -> pd.Series:
    """연속 양수/음수 일수 계산."""
    streak = pd.Series(0, index=series.index, dtype=int)
    for i in range(1, len(series)):
        if series.iloc[i] > 0:
            streak.iloc[i] = max(streak.iloc[i-1], 0) + 1
        elif series.iloc[i] < 0:
            streak.iloc[i] = min(streak.iloc[i-1], 0) - 1
        else:
            streak.iloc[i] = 0
    return streak


def analyze_technical(df: pd.DataFrame, idx: int) -> dict:
    """기술적 분석 레이어별 판정."""
    row = df.iloc[idx]
    prev = df.iloc[idx - 1] if idx > 0 else row

    analysis = {}

    # ── 기본 정보 ──
    analysis["date"] = str(df.index[idx].date()) if hasattr(df.index[idx], "date") else str(df.index[idx])
    analysis["close"] = row["Close"]
    analysis["volume"] = row["Volume"]
    analysis["change_pct"] = ((row["Close"] - prev["Close"]) / prev["Close"]) * 100

    # ── 이동평균 분석 ──
    ma_analysis = {}
    for ma_name in ["MA5", "MA20", "MA60", "MA120"]:
        if ma_name in df.columns and not pd.isna(row[ma_name]):
            diff_pct = ((row["Close"] - row[ma_name]) / row[ma_name]) * 100
            ma_analysis[ma_name] = {
                "value": round(row[ma_name], 0),
                "diff_pct": round(diff_pct, 2),
                "above": row["Close"] > row[ma_name],
            }
    analysis["ma"] = ma_analysis

    # ── 추세 판단 ──
    if all(k in ma_analysis for k in ["MA5", "MA20", "MA60"]):
        if ma_analysis["MA5"]["above"] and ma_analysis["MA20"]["above"] and ma_analysis["MA60"]["above"]:
            analysis["trend"] = "강한 상승"
        elif ma_analysis["MA5"]["above"] and ma_analysis["MA20"]["above"]:
            analysis["trend"] = "상승"
        elif ma_analysis["MA60"]["above"]:
            analysis["trend"] = "중립(60일선 위)"
        else:
            analysis["trend"] = "하락"
    else:
        analysis["trend"] = "판단불가"

    # ── RSI ──
    rsi = row.get("RSI", np.nan)
    if not pd.isna(rsi):
        analysis["rsi"] = round(rsi, 1)
        if rsi > 70:
            analysis["rsi_signal"] = "과매수"
        elif rsi > 60:
            analysis["rsi_signal"] = "강세"
        elif rsi > 40:
            analysis["rsi_signal"] = "중립"
        elif rsi > 30:
            analysis["rsi_signal"] = "약세"
        else:
            analysis["rsi_signal"] = "과매도"

    # ── MACD ──
    macd = row.get("MACD", np.nan)
    macd_sig = row.get("MACD_Signal", np.nan)
    if not pd.isna(macd) and not pd.isna(macd_sig):
        analysis["macd"] = round(macd, 1)
        analysis["macd_signal"] = round(macd_sig, 1)
        analysis["macd_histogram"] = round(macd - macd_sig, 1)
        if macd > macd_sig and prev.get("MACD", 0) <= prev.get("MACD_Signal", 0):
            analysis["macd_cross"] = "골든크로스"
        elif macd < macd_sig and prev.get("MACD", 0) >= prev.get("MACD_Signal", 0):
            analysis["macd_cross"] = "데드크로스"
        elif macd > macd_sig:
            analysis["macd_cross"] = "매수 유지"
        else:
            analysis["macd_cross"] = "매도 유지"

    # ── Stochastic ──
    stoch_k = row.get("Stoch_K", np.nan)
    stoch_d = row.get("Stoch_D", np.nan)
    if not pd.isna(stoch_k):
        analysis["stoch_k"] = round(stoch_k, 1)
        analysis["stoch_d"] = round(stoch_d, 1) if not pd.isna(stoch_d) else None
        if stoch_k > 80:
            analysis["stoch_signal"] = "과매수"
        elif stoch_k < 20:
            analysis["stoch_signal"] = "과매도"
        else:
            analysis["stoch_signal"] = "중립"

    # ── Bollinger Band ──
    upper = row.get("Upper_Band", np.nan)
    lower = row.get("Lower_Band", np.nan)
    if not pd.isna(upper) and not pd.isna(lower):
        analysis["bb_upper"] = round(upper, 0)
        analysis["bb_lower"] = round(lower, 0)
        bb_pct = (row["Close"] - lower) / (upper - lower) * 100 if upper != lower else 50
        analysis["bb_pct"] = round(bb_pct, 1)
        analysis["bb_width"] = round((upper - lower) / row["Close"] * 100, 2)
        if bb_pct > 100:
            analysis["bb_signal"] = "상단 돌파 (과열)"
        elif bb_pct > 80:
            analysis["bb_signal"] = "상단 근접"
        elif bb_pct < 0:
            analysis["bb_signal"] = "하단 돌파 (침체)"
        elif bb_pct < 20:
            analysis["bb_signal"] = "하단 근접"
        else:
            analysis["bb_signal"] = "중립"

    # ── ATR (변동성) ──
    atr = row.get("ATR", np.nan)
    if not pd.isna(atr):
        analysis["atr"] = round(atr, 0)
        analysis["atr_pct"] = round(atr / row["Close"] * 100, 2)

    # ── ADX (추세 강도) ──
    adx = row.get("ADX", np.nan)
    plus_di = row.get("Plus_DI", np.nan)
    minus_di = row.get("Minus_DI", np.nan)
    if not pd.isna(adx):
        analysis["adx"] = round(adx, 1)
        analysis["plus_di"] = round(plus_di, 1) if not pd.isna(plus_di) else None
        analysis["minus_di"] = round(minus_di, 1) if not pd.isna(minus_di) else None
        if adx > 25 and not pd.isna(plus_di) and not pd.isna(minus_di):
            if plus_di > minus_di:
                analysis["adx_signal"] = "강한 상승 추세"
            else:
                analysis["adx_signal"] = "강한 하락 추세"
        elif adx < 20:
            analysis["adx_signal"] = "추세 없음 (횡보)"
        else:
            analysis["adx_signal"] = "약한 추세"

    # ── TRIX ──
    trix = row.get("TRIX", np.nan)
    trix_sig = row.get("TRIX_Signal", np.nan)
    if not pd.isna(trix) and not pd.isna(trix_sig):
        analysis["trix"] = round(trix, 4)
        analysis["trix_signal_val"] = round(trix_sig, 4)
        analysis["trix_bullish"] = trix > trix_sig

    # ── OBV ──
    obv = row.get("OBV", np.nan)
    if not pd.isna(obv):
        analysis["obv"] = int(obv)
        # OBV 5일 변화
        if idx >= 5:
            obv_5ago = df.iloc[idx - 5].get("OBV", np.nan)
            if not pd.isna(obv_5ago) and obv_5ago != 0:
                analysis["obv_change_5d"] = round((obv - obv_5ago) / abs(obv_5ago) * 100, 2)

    # ── 거래량 분석 ──
    vol_surge = row.get("volume_surge_ratio", np.nan)
    if not pd.isna(vol_surge):
        analysis["vol_surge_ratio"] = round(vol_surge, 2)
        if vol_surge > 3:
            analysis["vol_signal"] = "폭발적 거래량"
        elif vol_surge > 2:
            analysis["vol_signal"] = "급증"
        elif vol_surge > 1.5:
            analysis["vol_signal"] = "증가"
        elif vol_surge > 0.7:
            analysis["vol_signal"] = "보통"
        else:
            analysis["vol_signal"] = "감소"

    # ── 수급 분석 ──
    foreign = row.get("Foreign_Net", 0)
    inst = row.get("Inst_Net", 0)
    analysis["foreign_net"] = int(foreign) if not pd.isna(foreign) else 0
    analysis["inst_net"] = int(inst) if not pd.isna(inst) else 0

    return analysis


def check_pipeline_layers(df: pd.DataFrame, idx: int, analysis: dict) -> dict:
    """6-Layer Pipeline 시뮬레이션."""
    row = df.iloc[idx]
    layers = {}

    # ── L0: Pre-Gate ──
    market_cap = row.get("MarketCap", 0)
    avg_volume_20 = df["Volume"].iloc[max(0, idx-19):idx+1].mean()
    avg_trading_value = avg_volume_20 * row["Close"]

    l0_passed = True
    l0_reason = []
    if market_cap > 0 and market_cap < 100_000_000_000:  # 시총 1000억 미만
        l0_passed = False
        l0_reason.append(f"시총 {market_cap/1e8:.0f}억 < 1000억")
    if avg_trading_value < 500_000_000:  # 일거래대금 5억 미만
        l0_passed = False
        l0_reason.append(f"일거래대금 {avg_trading_value/1e8:.1f}억 < 5억")

    layers["L0_pre_gate"] = {
        "passed": l0_passed,
        "reason": ", ".join(l0_reason) if l0_reason else "통과",
        "market_cap_억": round(market_cap / 1e8, 0) if market_cap > 0 else "N/A",
        "avg_trading_value_억": round(avg_trading_value / 1e8, 1),
    }

    # ── L0: Zone Score (ATR pullback + Valuation + Supply/Demand) ──
    atr = row.get("ATR", 0)
    close = row["Close"]

    # ATR pullback (MA20 기준)
    ma20 = row.get("MA20", close)
    pullback_atr = (ma20 - close) / atr if atr > 0 else 0
    # Zone score 근사
    if 0.5 <= pullback_atr <= 1.5:
        atr_score = 30  # sweet_spot
    elif 0.25 <= pullback_atr < 0.5 or 1.5 < pullback_atr <= 2.0:
        atr_score = 20  # healthy / deep
    elif 0 <= pullback_atr < 0.25:
        atr_score = 10  # shallow
    else:
        atr_score = 5   # noise or structural

    # RSI score
    rsi = row.get("RSI", 50)
    if 30 <= rsi <= 45:
        rsi_score = 20
    elif 45 < rsi <= 55:
        rsi_score = 15
    elif rsi < 30:
        rsi_score = 10
    else:
        rsi_score = 5

    # Stochastic score
    stoch_k = row.get("Stoch_K", 50)
    if stoch_k < 20:
        stoch_score = 15
    elif stoch_k < 40:
        stoch_score = 10
    else:
        stoch_score = 5

    zone_score = atr_score + rsi_score + stoch_score
    if zone_score >= 55:
        grade = "A"
    elif zone_score >= 40:
        grade = "B"
    elif zone_score >= 25:
        grade = "C"
    else:
        grade = "F"

    layers["L0_grade"] = {
        "passed": grade != "F",
        "zone_score": zone_score,
        "grade": grade,
        "components": {
            "atr_pullback": round(pullback_atr, 2),
            "atr_score": atr_score,
            "rsi_score": rsi_score,
            "stoch_score": stoch_score,
        },
    }

    # ── L1: Regime (Accumulation 확률) ──
    # P_Accum 데이터 없으면 가격/거래량 패턴으로 추정
    # 최근 60일 가격 범위 수축 + 거래량 감소 = 매집 패턴
    recent_60 = df.iloc[max(0, idx-59):idx+1]
    price_range = (recent_60["High"].max() - recent_60["Low"].min()) / close * 100
    vol_trend = df["Volume"].iloc[max(0, idx-19):idx+1].mean() / df["Volume"].iloc[max(0, idx-39):max(1, idx-19)].mean() if idx >= 40 else 1

    accum_likely = price_range < 30 and vol_trend < 1.2
    layers["L1_regime"] = {
        "passed": True,  # 데이터 없으면 통과
        "accum_likely": accum_likely,
        "price_range_60d": round(price_range, 1),
        "vol_trend_ratio": round(vol_trend, 2),
        "note": "매집 패턴 감지" if accum_likely else "비매집 구간",
    }

    # ── L2: OU Filter ──
    # z-score 근사 (MA60 기준)
    ma60 = row.get("MA60", close)
    std_60 = recent_60["Close"].std() if len(recent_60) > 5 else 1
    z_score = (close - ma60) / std_60 if std_60 > 0 else 0

    ou_passed = True
    ou_reason = ""
    layers["L2_ou"] = {
        "passed": ou_passed,
        "z_score": round(z_score, 2),
        "note": "데이터 충분 시 OU 모델로 정밀 분석 필요",
    }

    # ── L3: Momentum ──
    vol_surge = row.get("volume_surge_ratio", 1.0)
    slope_60 = row.get("slope_ma60", 0)
    if pd.isna(vol_surge):
        vol_surge = 1.0
    if pd.isna(slope_60):
        slope_60 = 0

    mom_passed = not (vol_surge < 1.2 and slope_60 < -0.5)
    layers["L3_momentum"] = {
        "passed": mom_passed,
        "vol_surge": round(vol_surge, 2),
        "slope_ma60": round(slope_60, 4),
        "reason": "약한 모멘텀" if not mom_passed else "통과",
    }

    # ── L4: Smart Money ──
    # OBV 추세 + 수급 분석
    obv_now = row.get("OBV", 0)
    obv_20ago = df.iloc[max(0, idx-19)].get("OBV", obv_now)
    obv_trend = "상승" if obv_now > obv_20ago else "하락"

    sm_signal = "중립"
    if obv_trend == "상승" and close > ma20:
        sm_signal = "매집 추정"
    elif obv_trend == "하락" and close < ma20:
        sm_signal = "유출 추정"

    layers["L4_smart_money"] = {
        "passed": sm_signal != "유출 추정",
        "obv_trend": obv_trend,
        "obv_20d_change": int(obv_now - obv_20ago),
        "signal": sm_signal,
    }

    # ── L5: Risk (손익비) ──
    # Impulse 트리거 기준
    swing_low = df["Low"].iloc[max(0, idx-9):idx+1].min()
    stop_price = max(swing_low * 0.995, close * 0.97)  # 3% 정지 손절
    target_price = close + atr * 3  # ATR 3배 목표

    risk = close - stop_price
    reward = target_price - close
    rr_ratio = reward / risk if risk > 0 else 0

    layers["L5_risk"] = {
        "passed": rr_ratio >= 1.5,
        "entry_price": int(close),
        "stop_loss": int(stop_price),
        "target_price": int(target_price),
        "risk_reward": round(rr_ratio, 2),
        "risk_amount": int(risk),
        "reward_amount": int(reward),
    }

    # ── L6: Trigger ──
    # Impulse 트리거 조건
    conditions = {}

    # 1. Close > MA5
    conditions["close_above_ma5"] = close > row.get("MA5", 0) if not pd.isna(row.get("MA5", np.nan)) else False

    # 2. RSI 상승 전환
    if idx >= 1:
        prev_rsi = df.iloc[idx-1].get("RSI", 50)
        conditions["rsi_turning_up"] = rsi > prev_rsi and rsi < 70
    else:
        conditions["rsi_turning_up"] = False

    # 3. MACD 히스토그램 상승
    macd = row.get("MACD", 0)
    macd_sig = row.get("MACD_Signal", 0)
    hist = macd - macd_sig if not pd.isna(macd) and not pd.isna(macd_sig) else 0
    if idx >= 1:
        prev_hist = df.iloc[idx-1].get("MACD", 0) - df.iloc[idx-1].get("MACD_Signal", 0)
        conditions["macd_hist_rising"] = hist > prev_hist
    else:
        conditions["macd_hist_rising"] = False

    # 4. 거래량 서지
    conditions["volume_surge"] = vol_surge > 1.2

    # 5. Stoch K > D 골든크로스
    stoch_d = row.get("Stoch_D", 50)
    conditions["stoch_golden"] = stoch_k > stoch_d if not pd.isna(stoch_d) else False

    met_count = sum(conditions.values())
    impulse_triggered = met_count >= 3

    # Confirm 트리거 (보수적)
    confirm_conditions = {
        "close_above_ma20": close > ma20 if not pd.isna(ma20) else False,
        "rsi_above_50": rsi > 50 if not pd.isna(rsi) else False,
        "adx_rising": row.get("ADX", 0) > 20,
        "plus_di_above": (row.get("Plus_DI", 0) > row.get("Minus_DI", 0)),
    }
    confirm_met = sum(confirm_conditions.values())
    confirm_triggered = confirm_met >= 3

    if impulse_triggered:
        trigger_type = "impulse"
        trigger_confidence = met_count / 5
    elif confirm_triggered:
        trigger_type = "confirm"
        trigger_confidence = confirm_met / 4
    else:
        trigger_type = "none"
        trigger_confidence = 0

    layers["L6_trigger"] = {
        "passed": trigger_type != "none",
        "trigger_type": trigger_type,
        "confidence": round(trigger_confidence, 2),
        "impulse_conditions": conditions,
        "impulse_met": met_count,
        "confirm_conditions": confirm_conditions,
        "confirm_met": confirm_met,
    }

    return layers


def analyze_recent_trend(df: pd.DataFrame, idx: int, days: int = 20) -> dict:
    """최근 N일 가격 추이 분석."""
    start = max(0, idx - days + 1)
    recent = df.iloc[start:idx + 1]

    high = recent["High"].max()
    low = recent["Low"].min()
    close_start = recent["Close"].iloc[0]
    close_end = recent["Close"].iloc[-1]
    change = ((close_end - close_start) / close_start) * 100

    # 연속 상승/하락일
    up_days = 0
    for i in range(len(recent) - 1, 0, -1):
        if recent["Close"].iloc[i] >= recent["Close"].iloc[i - 1]:
            up_days += 1
        else:
            break

    return {
        "period": f"{days}일",
        "high": int(high),
        "low": int(low),
        "change_pct": round(change, 2),
        "up_days_streak": up_days,
        "avg_volume": int(recent["Volume"].mean()),
    }


def print_analysis_report(ticker: str, name: str, analysis: dict, layers: dict,
                          trends: dict, df: pd.DataFrame, idx: int):
    """분석 리포트 출력."""
    row = df.iloc[idx]

    print("=" * 60)
    print(f"  [{name}] {ticker} - 종합 분석 리포트")
    print(f"  분석일: {analysis['date']} | 종가: {analysis['close']:,.0f}원")
    print("=" * 60)

    # ── 1. 가격 동향 ──
    print(f"\n{'─' * 60}")
    print("  [1] 가격 동향")
    print(f"{'─' * 60}")
    print(f"  현재가: {analysis['close']:,.0f}원 ({analysis['change_pct']:+.2f}%)")
    print(f"  거래량: {analysis['volume']:,}주")
    if "vol_signal" in analysis:
        print(f"  거래량 수준: {analysis['vol_signal']} (x{analysis['vol_surge_ratio']:.1f})")
    print(f"  추세: {analysis['trend']}")

    for period, data in trends.items():
        print(f"  {period}: {data['change_pct']:+.1f}% (고가 {data['high']:,} / 저가 {data['low']:,})")

    # ── 2. 이동평균 ──
    print(f"\n{'─' * 60}")
    print("  [2] 이동평균선 배열")
    print(f"{'─' * 60}")
    for ma_name, ma_data in analysis["ma"].items():
        arrow = "▲" if ma_data["above"] else "▼"
        print(f"  {ma_name}: {ma_data['value']:,.0f}원  {arrow} {ma_data['diff_pct']:+.1f}%")

    # ── 3. 기술 지표 ──
    print(f"\n{'─' * 60}")
    print("  [3] 기술적 지표")
    print(f"{'─' * 60}")
    if "rsi" in analysis:
        print(f"  RSI(14): {analysis['rsi']:.1f} - {analysis['rsi_signal']}")
    if "macd" in analysis:
        print(f"  MACD: {analysis['macd']:.1f} | Signal: {analysis['macd_signal']:.1f} | Hist: {analysis['macd_histogram']:.1f}")
        print(f"  MACD 상태: {analysis['macd_cross']}")
    if "stoch_k" in analysis:
        print(f"  Stochastic: K={analysis['stoch_k']:.1f} D={analysis['stoch_d']:.1f} - {analysis['stoch_signal']}")
    if "adx" in analysis:
        print(f"  ADX: {analysis['adx']:.1f} - {analysis['adx_signal']}")
        if analysis.get("plus_di") is not None:
            print(f"  +DI: {analysis['plus_di']:.1f} / -DI: {analysis['minus_di']:.1f}")
    if "bb_pct" in analysis:
        print(f"  Bollinger: {analysis['bb_signal']} (밴드 내 위치: {analysis['bb_pct']:.0f}%)")
        print(f"    상단: {analysis['bb_upper']:,.0f} | 하단: {analysis['bb_lower']:,.0f} | 밴드폭: {analysis['bb_width']:.1f}%")
    if "trix" in analysis:
        trix_dir = "강세" if analysis["trix_bullish"] else "약세"
        print(f"  TRIX: {analysis['trix']:.4f} vs Signal: {analysis['trix_signal_val']:.4f} - {trix_dir}")
    if "atr" in analysis:
        print(f"  ATR(14): {analysis['atr']:,.0f}원 ({analysis['atr_pct']:.1f}%)")
    if "obv_change_5d" in analysis:
        print(f"  OBV 5일 변화: {analysis['obv_change_5d']:+.1f}%")

    # ── 4. Pipeline 결과 ──
    print(f"\n{'─' * 60}")
    print("  [4] 6-Layer Pipeline 분석")
    print(f"{'─' * 60}")

    layer_names = {
        "L0_pre_gate": "L0 Pre-Gate (사전 스크리닝)",
        "L0_grade": "L0 Grade (등급 판정)",
        "L1_regime": "L1 Regime (매집 구간)",
        "L2_ou": "L2 OU (평균회귀)",
        "L3_momentum": "L3 Momentum (모멘텀)",
        "L4_smart_money": "L4 Smart Money (세력 분석)",
        "L5_risk": "L5 Risk (손익비)",
        "L6_trigger": "L6 Trigger (진입 트리거)",
    }

    all_passed = True
    for key, label in layer_names.items():
        layer = layers.get(key, {})
        status = "✅ PASS" if layer.get("passed", False) else "❌ FAIL"
        if not layer.get("passed", False):
            all_passed = False
        print(f"  {status} | {label}")

        # 세부 정보
        if key == "L0_pre_gate":
            print(f"           시총: {layer.get('market_cap_억', 'N/A')}억 | 일거래대금: {layer.get('avg_trading_value_억', 0):.0f}억")
        elif key == "L0_grade":
            print(f"           Zone Score: {layer['zone_score']} | Grade: {layer['grade']}")
            comp = layer["components"]
            print(f"           ATR풀백: {comp['atr_pullback']:.2f} ({comp['atr_score']}p) | RSI: {comp['rsi_score']}p | Stoch: {comp['stoch_score']}p")
        elif key == "L1_regime":
            print(f"           60일 가격범위: {layer['price_range_60d']:.1f}% | 거래량추세: x{layer['vol_trend_ratio']:.2f} | {layer['note']}")
        elif key == "L2_ou":
            print(f"           Z-Score: {layer['z_score']:.2f}")
        elif key == "L3_momentum":
            print(f"           거래량서지: x{layer['vol_surge']:.2f} | MA60기울기: {layer['slope_ma60']:.4f}")
        elif key == "L4_smart_money":
            print(f"           OBV추세: {layer['obv_trend']} ({layer['obv_20d_change']:+,}) | 판정: {layer['signal']}")
        elif key == "L5_risk":
            print(f"           진입: {layer['entry_price']:,}원 | 손절: {layer['stop_loss']:,}원 | 목표: {layer['target_price']:,}원")
            print(f"           손익비: {layer['risk_reward']:.2f} (위험: {layer['risk_amount']:,} / 보상: {layer['reward_amount']:,})")
        elif key == "L6_trigger":
            print(f"           트리거: {layer['trigger_type'].upper()} (신뢰도: {layer['confidence']:.0%})")
            if layer["trigger_type"] == "impulse":
                for cond, met in layer["impulse_conditions"].items():
                    icon = "○" if met else "×"
                    print(f"             {icon} {cond}")
            elif layer["trigger_type"] == "confirm":
                for cond, met in layer["confirm_conditions"].items():
                    icon = "○" if met else "×"
                    print(f"             {icon} {cond}")

    # ── 5. 시장 시그널 알림 ──
    print(f"\n{'=' * 60}")
    print("  [5] 시장 시그널 알림")
    print(f"{'=' * 60}")

    scanner = MarketSignalScanner()
    market_signals = scanner.scan_all(df, idx)

    if market_signals:
        importance_icons = {
            "critical": "🚨", "high": "⚡", "medium": "💡", "low": "ℹ️"
        }
        for sig in market_signals:
            icon = importance_icons.get(sig.importance, "•")
            print(f"\n  {icon} {sig.title}")
            print(f"     {sig.description}")
            print(f"     신뢰도: {sig.confidence:.0f}% | 중요도: {sig.importance.upper()}")
    else:
        print("\n  특이 시그널 없음")

    # ── 6. 종합 판단 ──
    print(f"\n{'=' * 60}")
    print("  [6] 종합 판단 - 내일 매매 전략")
    print(f"{'=' * 60}")

    trigger = layers["L6_trigger"]
    risk = layers["L5_risk"]
    grade = layers["L0_grade"]["grade"]

    if all_passed and trigger["passed"]:
        print(f"\n  [*] 매수 시그널 발동 ({trigger['trigger_type'].upper()})")
        print(f"  등급: {grade} | Zone Score: {layers['L0_grade']['zone_score']}")
        print(f"\n  [매매 계획]")
        print(f"    진입가: {risk['entry_price']:,}원")
        print(f"    손절가: {risk['stop_loss']:,}원 ({((risk['stop_loss']/risk['entry_price'])-1)*100:+.1f}%)")
        print(f"    목표가: {risk['target_price']:,}원 ({((risk['target_price']/risk['entry_price'])-1)*100:+.1f}%)")
        print(f"    손익비: {risk['risk_reward']:.2f}")

        if trigger["trigger_type"] == "impulse":
            print(f"    비중: 40% (1차 시동 진입)")
            print(f"    손절: -3% 정지 손절")
        else:
            print(f"    비중: 40% (1차 확인 진입)")
            print(f"    손절: -5% ATR 손절")

        # 부분청산 계획
        atr = row.get("ATR", 0)
        print(f"\n  [4단계 부분청산 계획]")
        for i, (r_mult, pct) in enumerate([(2, 25), (4, 25), (8, 25), (10, 25)], 1):
            target = risk["entry_price"] + atr * r_mult
            print(f"    {i}차: {target:,.0f}원 ({r_mult}R, +{((target/risk['entry_price'])-1)*100:.1f}%) → {pct}% 청산")

    elif trigger["passed"] and not risk.get("passed", True):
        print(f"\n  [>] 관망 - 트리거 발동했으나 손익비 부족 ({risk['risk_reward']:.2f})")
        print(f"  손익비 1.5 이상 시 매수 가능")

    elif not trigger["passed"]:
        print(f"\n  [>] 대기 - 진입 트리거 미발동")
        imp = trigger["impulse_met"]
        conf = trigger["confirm_met"]
        print(f"  Impulse: {imp}/5 조건 충족 (3개 필요)")
        print(f"  Confirm: {conf}/4 조건 충족 (3개 필요)")
        print(f"\n  [조건 달성 현황 - Impulse]")
        for cond, met in trigger["impulse_conditions"].items():
            icon = "✅" if met else "⬜"
            print(f"    {icon} {cond}")
        print(f"\n  [필요 조건]")
        not_met = [c for c, m in trigger["impulse_conditions"].items() if not m]
        if not_met:
            print(f"    미충족: {', '.join(not_met)}")
    else:
        # 파이프라인 중간에 실패
        failed = [k for k, v in layers.items() if not v.get("passed", True)]
        print(f"\n  [>] 매수 불가 - Pipeline 레이어 실패")
        for f_layer in failed:
            print(f"    ❌ {layer_names.get(f_layer, f_layer)}")

        # v3.2: Grade F이지만 시장 시그널이 강할 때 Trend Continuation 안내
        if "L0_grade" in failed and market_signals:
            critical_signals = [s for s in market_signals if s.importance in ("critical", "high")]
            if critical_signals:
                print(f"\n  [!] v3.2 Trend Continuation 경로 가능")
                print(f"  Grade F이지만 아래 강력 시그널 감지:")
                for cs in critical_signals:
                    print(f"    → {cs.title} (신뢰도 {cs.confidence:.0f}%)")
                print(f"  실제 매매 시 signal_engine의 Trend Continuation 로직 적용됨")

    print(f"\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="개별 종목 분석")
    parser.add_argument("--ticker", default="039130", help="종목코드 (기본: 039130 하나투어)")
    args = parser.parse_args()

    # 데이터 로드
    df, name = load_stock_data(args.ticker)
    df = compute_extra_indicators(df)

    idx = len(df) - 1  # 최신 데이터

    # 분석 실행
    analysis = analyze_technical(df, idx)
    layers = check_pipeline_layers(df, idx, analysis)

    # 최근 추이
    trends = {}
    for days in [5, 10, 20]:
        trends[f"최근 {days}일"] = analyze_recent_trend(df, idx, days)

    # 리포트 출력
    print_analysis_report(args.ticker, name, analysis, layers, trends, df, idx)

    # 파일로도 저장
    output_path = Path(__file__).parent.parent / "data" / f"analysis_{args.ticker}.txt"
    import contextlib
    with open(output_path, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            print_analysis_report(args.ticker, name, analysis, layers, trends, df, idx)
    print(f"\n  >> 분석 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
