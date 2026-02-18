"""
전체 종목 매수 후보 스캔 -> Kill→Rank→Tag -> 텔레그램 발송

v10.0: 4축 100점 제거, Kill 중복 제거, Trap 제거
  - Kill: K3(트리거) + K4(유동성) — K1/K2/K5는 v8 Gate G1/G2/G3과 중복이므로 제거
  - Rank: R:R × Zone × Catalyst (선행 100%)
  - Tag: 수급 streak 기반 (Part 2에서 5D 교차필터로 확장 예정)

사용법:
    python scripts/scan_buy_candidates.py --grade AB
    python scripts/scan_buy_candidates.py --grade AB --no-news --no-send
"""

import argparse
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.market_signal_scanner import MarketSignalScanner

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CSV_DIR = Path(__file__).resolve().parent.parent / "stock_data_daily"


# =========================================================
# 데이터 신선도 검증 (대책 D)
# =========================================================

def validate_data_freshness() -> dict:
    """데이터 최종일 vs 최근 거래일 비교 → 경고 출력."""
    from datetime import date, timedelta

    result = {"fresh": True, "last_data_date": None, "last_trading_date": None, "msg": ""}

    # parquet 최종일 확인
    sample = sorted(PROCESSED_DIR.glob("*.parquet"))[:3]
    last_dates = []
    for pq in sample:
        try:
            _df = pd.read_parquet(pq)
            if "close" in _df.columns and len(_df) > 0:
                last_dates.append(_df.index[-1] if hasattr(_df.index, 'date') else len(_df))
        except Exception:
            pass

    # CSV 최종일 확인 (fallback)
    if not last_dates:
        csvs = sorted(CSV_DIR.glob("*.csv"))[:3]
        for f in csvs:
            try:
                _df = pd.read_csv(f, usecols=["Date"], nrows=10000)
                last_dates.append(pd.to_datetime(_df["Date"]).max())
            except Exception:
                pass

    if not last_dates:
        result["fresh"] = False
        result["msg"] = "데이터 파일을 찾을 수 없습니다"
        return result

    # 최근 거래일 계산 (exchange_calendars 없으면 간이 판단)
    try:
        import exchange_calendars as xcals
        xkrx = xcals.get_calendar("XKRX")
        today = date.today()
        sessions = xkrx.sessions_in_range(
            pd.Timestamp(today - timedelta(days=10)),
            pd.Timestamp(today),
        )
        if len(sessions) > 0:
            last_trading = sessions[-1].date()
        else:
            last_trading = today
    except Exception:
        # exchange_calendars 없으면 간이 판단
        today = date.today()
        wd = today.weekday()
        if wd == 0:
            last_trading = today - timedelta(days=3)
        elif wd == 6:
            last_trading = today - timedelta(days=2)
        else:
            last_trading = today - timedelta(days=1) if wd > 0 else today

    # 데이터 최종일 추출
    ld = max(last_dates)
    if hasattr(ld, 'date'):
        data_date = ld.date()
    elif hasattr(ld, 'to_pydatetime'):
        data_date = ld.to_pydatetime().date()
    else:
        data_date = pd.Timestamp(ld).date()

    result["last_data_date"] = str(data_date)
    result["last_trading_date"] = str(last_trading)

    gap_days = (last_trading - data_date).days
    if gap_days > 1:
        result["fresh"] = False
        result["msg"] = f"데이터 {gap_days}일 지연 (데이터: {data_date}, 최근거래일: {last_trading})"
    else:
        result["msg"] = f"데이터 최신 (데이터: {data_date}, 최근거래일: {last_trading})"

    return result


# =========================================================
# CSV 유니버스 경량 스코어링 (37컬럼 기반)
# =========================================================

def load_csv_stocks(exclude_tickers: set[str] | None = None) -> dict[str, tuple[pd.DataFrame, str]]:
    """stock_data_daily/*.csv 로드 → {ticker: (DataFrame, name)}.

    exclude_tickers: parquet에 이미 있는 종목 제외.
    """
    data = {}
    if not CSV_DIR.exists():
        return data
    for f in sorted(CSV_DIR.glob("*.csv")):
        match = re.search(r"_(\d{6})$", f.stem)
        if not match:
            continue
        ticker = match.group(1)
        if exclude_tickers and ticker in exclude_tickers:
            continue
        name = f.stem[: f.stem.rfind("_")]
        try:
            df = pd.read_csv(f)
            if len(df) >= 200:
                data[ticker] = (df, name)
        except Exception:
            continue
    return data


def _calc_streak_csv(series: pd.Series) -> int:
    """CSV용 연속 순매수 일수 계산."""
    if series.empty:
        return 0
    vals = series.values
    if pd.isna(vals[-1]) or vals[-1] == 0:
        return 0
    direction = 1 if vals[-1] > 0 else -1
    count = 0
    for v in reversed(vals):
        if pd.isna(v) or v == 0:
            break
        if (v > 0 and direction > 0) or (v < 0 and direction < 0):
            count += 1
        else:
            break
    return count * direction


def quick_csv_score(df: pd.DataFrame, idx: int) -> dict | None:
    """CSV 37컬럼 기반 간이 스코어링.

    Zone proxy:  이평선 위치 + BB position + RSI zone
    R:R proxy:   ATR 기반 entry/stop/target
    Trigger:     TRIX + MACD 골든크로스
    """
    if idx < 120:
        return None

    row = df.iloc[idx]
    prev = df.iloc[idx - 1]
    close = row.get("Close", np.nan)
    if pd.isna(close) or close <= 0:
        return None

    # 동전주 제외 (2,000원 미만)
    if close < 2000:
        return None

    # --- Zone Score (0~1) ---
    ma5 = row.get("MA5", np.nan)
    ma20 = row.get("MA20", np.nan)
    ma60 = row.get("MA60", np.nan)
    ma120 = row.get("MA120", np.nan)

    # 이평선 위치 (0~0.4)
    ma_score = 0.0
    if not pd.isna(ma20) and close > ma20:
        ma_score += 0.1
    if not pd.isna(ma60) and close > ma60:
        ma_score += 0.15
    if not pd.isna(ma120) and close > ma120:
        ma_score += 0.15

    # BB 위치 (0~0.3)
    bb_upper = row.get("Upper_Band", np.nan)
    bb_lower = row.get("Lower_Band", np.nan)
    bb_score = 0.0
    if not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_upper > bb_lower:
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower)
        if 0.2 <= bb_pos <= 0.7:
            bb_score = 0.3
        elif bb_pos < 0.2:
            bb_score = 0.2
        else:
            bb_score = 0.1

    # RSI 위치 (0~0.3)
    rsi = row.get("RSI", 50)
    if pd.isna(rsi):
        rsi = 50
    rsi_score = 0.0
    if 35 <= rsi <= 55:
        rsi_score = 0.3
    elif 30 <= rsi < 35:
        rsi_score = 0.25
    elif 55 < rsi <= 65:
        rsi_score = 0.2
    elif rsi < 30:
        rsi_score = 0.15
    else:
        rsi_score = 0.05

    # Stochastic 보너스 (0~0.1)
    stoch_k = row.get("Stoch_K", 50)
    if pd.isna(stoch_k):
        stoch_k = 50
    stoch_bonus = 0.0
    if stoch_k < 30:
        stoch_bonus = 0.1  # 과매도 (반등 기대)
    elif stoch_k < 50:
        stoch_bonus = 0.05

    zone = round(ma_score + bb_score + rsi_score + stoch_bonus, 3)
    zone = min(zone, 1.0)  # 상한 1.0

    # --- R:R (Risk:Reward) — 최근 지지/저항 기반 ---
    atr = row.get("ATR", np.nan)
    if pd.isna(atr) or atr <= 0:
        atr = close * 0.02

    # 손절: 최근 10일 저점 기반 (ATR 하한)
    low_10 = df["Low"].iloc[max(0, idx - 9) : idx + 1].min()
    stop_loss = round(max(low_10 - 0.5 * atr, close - 3 * atr))

    # 목표: 최근 60일 고점 기반
    high_60 = df["High"].iloc[max(0, idx - 59) : idx + 1].max()
    target_price = round(max(high_60, close + 1.5 * atr))

    if close > stop_loss and stop_loss > 0:
        risk = close - stop_loss
        reward = target_price - close
        rr = round(reward / risk, 2) if risk > 0 else 0
    else:
        rr = 0

    # --- Trigger (TRIX + MACD) ---
    trix = row.get("TRIX", np.nan)
    trix_sig = row.get("TRIX_Signal", np.nan)
    trix_prev = prev.get("TRIX", np.nan)
    trix_sig_prev = prev.get("TRIX_Signal", np.nan)
    macd = row.get("MACD", np.nan)
    macd_sig = row.get("MACD_Signal", np.nan)

    trigger = "none"
    trix_above = not pd.isna(trix) and not pd.isna(trix_sig) and trix > trix_sig
    macd_above = not pd.isna(macd) and not pd.isna(macd_sig) and macd > macd_sig
    trix_gc = (
        trix_above
        and not pd.isna(trix_prev) and not pd.isna(trix_sig_prev)
        and trix_prev <= trix_sig_prev
    )

    if trix_above and macd_above:
        trigger = "confirm"
    elif trix_gc:
        trigger = "impulse"
    elif trix_above or macd_above:
        trigger = "setup"

    # --- ADX Gate ---
    adx = row.get("ADX", 0)
    if pd.isna(adx):
        adx = 0
    plus_di = row.get("Plus_DI", 0) or 0
    minus_di = row.get("Minus_DI", 0) or 0
    if pd.isna(plus_di):
        plus_di = 0
    if pd.isna(minus_di):
        minus_di = 0

    # ADX < 15 → 추세 없음, 스킵
    if adx < 15:
        return None

    # +DI < -DI → 하락추세, 스킵
    if plus_di < minus_di and adx > 20:
        return None

    # --- 거래대금 ---
    vol_20d = df["Volume"].iloc[max(0, idx - 19) : idx + 1].mean()
    avg_tv = close * vol_20d

    # --- 수급 ---
    foreign_streak = 0
    inst_streak = 0
    foreign_net_5d = 0
    inst_net_5d = 0

    if "Foreign_Net" in df.columns:
        f_series = df["Foreign_Net"].iloc[max(0, idx - 19) : idx + 1].fillna(0)
        foreign_streak = _calc_streak_csv(f_series)
        foreign_net_5d = int(df["Foreign_Net"].iloc[max(0, idx - 4) : idx + 1].fillna(0).sum())

    if "Inst_Net" in df.columns:
        i_series = df["Inst_Net"].iloc[max(0, idx - 19) : idx + 1].fillna(0)
        inst_streak = _calc_streak_csv(i_series)
        inst_net_5d = int(df["Inst_Net"].iloc[max(0, idx - 4) : idx + 1].fillna(0).sum())

    # OBV 추세
    obv = row.get("OBV", 0) or 0
    obv_20 = df["OBV"].iloc[max(0, idx - 20)] if "OBV" in df.columns else 0
    obv_trend = "up" if obv > (obv_20 or 0) else "down"

    # 거래량 서지
    vol_ma5 = df["Volume"].iloc[max(0, idx - 4) : idx + 1].mean()
    vol_surge = vol_ma5 / vol_20d if vol_20d > 0 else 1.0

    # 52주 고점 대비
    h252 = df["High"].iloc[max(0, idx - 252) : idx + 1].max()
    pct_of_52w = round(close / h252, 4) if h252 > 0 else 0

    return {
        "zone_score": zone,
        "risk_reward": round(rr, 2),
        "trigger_type": trigger,
        "entry_price": int(close),
        "stop_loss": int(stop_loss),
        "target_price": int(target_price),
        "rsi": float(rsi),
        "adx": float(adx),
        "plus_di": float(plus_di),
        "minus_di": float(minus_di),
        "vol_surge": float(vol_surge),
        "obv_trend": obv_trend,
        "avg_trading_value_20d": float(avg_tv),
        "foreign_streak": foreign_streak,
        "inst_streak": inst_streak,
        "foreign_amount_5d": foreign_net_5d,
        "inst_amount_5d": inst_net_5d,
        "foreign_net_5d": float(foreign_net_5d),
        "short_balance_chg_5d": 0.0,
        "pct_of_52w_high": pct_of_52w,
        "confidence": 0,
        "consensus": None,
        "market_signals": [],
        "grade": "CSV",
    }


# =========================================================
# Kill→Rank→Tag 파이프라인
# =========================================================

def detect_regime() -> dict:
    """공매도 상태 판정 (로깅 + 향후 G4용)."""
    import yaml
    from datetime import date

    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    today = date.today()
    calendar = cfg.get("short_selling_calendar", [])
    status = "active"

    for period in calendar:
        start = date.fromisoformat(str(period["start"]))
        end = date.fromisoformat(str(period["end"]))
        if start <= today <= end:
            status = period["status"]
            break

    return {"status": status}


def kill_filters(sig: dict) -> tuple[bool, list[str]]:
    """Kill Filters — K3(트리거) + K4(유동성).

    K1(Zone), K2(R:R), K5(고점근접)은 v8 Gate G1/G2/G3과 중복 → 제거.
    """
    kills = []

    # K3: Trigger 미발동
    trigger = sig.get("trigger_type", "none")
    if trigger in ("none", "waiting", "setup"):
        kills.append(f"K3:Trigger({trigger})")

    # K4: 20일 평균 거래대금 < 10억
    avg_tv = sig.get("avg_trading_value_20d", 0)
    if avg_tv < 1_000_000_000:
        kills.append(f"K4:유동성({avg_tv / 1e8:.0f}억<10억)")

    return len(kills) == 0, kills


def generate_tags(sig: dict) -> list[str]:
    """정보 태그 (참고용).

    수급 streak + SD 교차 + 관찰자 밀도 태그.
    """
    tags = []

    # 외국인/기관 연속 매수
    f_streak = sig.get("foreign_streak", 0)
    i_streak = sig.get("inst_streak", 0)
    if f_streak >= 5:
        tags.append(f"외{f_streak}D연속")
    elif f_streak >= 3:
        tags.append(f"외{f_streak}D")
    if i_streak >= 5:
        tags.append(f"기{i_streak}D연속")
    elif i_streak >= 3:
        tags.append(f"기{i_streak}D")

    # SD 교차 태그
    sd = sig.get("sd_cross", "")
    if sd == "양호":
        tags.append("수급양호")
    elif sd == "경고":
        tags.append("수급경고")

    # 관찰자 밀도 태그
    density = sig.get("density", "")
    if density == "저밀도":
        tags.append("숨은종목")
    elif density == "고밀도":
        tags.append("과밀")

    return tags


def load_overnight_signal() -> dict:
    """US Overnight Signal JSON 로드. 없으면 neutral 반환."""
    import json
    signal_path = Path("data/us_market/overnight_signal.json")
    if not signal_path.exists():
        return {"composite": "neutral", "score": 0.0, "sector_signals": {}}
    try:
        with open(signal_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"composite": "neutral", "score": 0.0, "sector_signals": {}}


def run_pipeline(
    candidates: list[dict],
    use_short_filter: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Kill→Rank→Tag 파이프라인.

    반환: (survivors, killed_list)
    use_short_filter: True면 SD 교차필터(외국인×공매도) 활성, False면 sd_mult=1.0 고정
    """
    # 레짐 감지 (로깅용)
    regime = detect_regime()
    print(f"  Regime: {regime['status']}")

    # US Overnight Signal 로드 (Level 1 + Level 2 + 섹터Kill + 특수룰)
    us_signal = load_overnight_signal()
    us_score = us_signal.get("score", 0.0)
    us_sectors = us_signal.get("sector_signals", {})
    us_kills = us_signal.get("sector_kills", {})
    us_rules = us_signal.get("special_rules", [])
    us_grade = us_signal.get("grade", "NEUTRAL")
    us_combined = us_signal.get("combined_score_100", us_score * 100)

    # 특수 룰에서 global_position_cap 추출 (가장 엄격한 것)
    global_cap = 1.0
    for rule in us_rules:
        cap = rule.get("global_position_cap", 1.0)
        global_cap = min(global_cap, cap)
        boost = rule.get("global_position_boost")
        if boost:
            global_cap = max(global_cap, boost)

    if us_grade != "NEUTRAL":
        print(f"  US Overnight: {us_grade} ({us_combined:+.1f})")
    if us_rules:
        for r in us_rules:
            print(f"    [특수룰] {r['name']}: {r['desc']}")
    killed_by_us = [s for s, v in us_kills.items() if v.get("killed")]
    if killed_by_us:
        print(f"    [섹터Kill] {', '.join(killed_by_us)}")

    killed_list = []
    survivors = []

    for sig in candidates:
        # Kill Filters (K3 + K4)
        passed, kill_reasons = kill_filters(sig)
        if not passed:
            sig["v9_kill_reasons"] = kill_reasons
            killed_list.append(sig)
            continue

        survivors.append(sig)

    # Rank: R:R × zone_score × catalyst × sd_cross × density
    #
    # SD 교차필터 (Part 2):
    #   외국인 매수 + 공매도 커버링 → ×1.05 (양호)
    #   외국인 매도 + 공매도 빌딩  → ×0.90 (경고)
    #   혼재/데이터 없음            → ×1.00 (중립)
    #
    # 관찰자 밀도 (Part 2):
    #   avg_trading_value_20d 기준 분위로 역가중
    #   상위 고밀도(>500억) → ×0.95, 하위 저밀도(<50억) → ×1.05

    # 밀도 분위 계산 (전체 survivor 기준)
    tv_values = [s.get("avg_trading_value_20d", 0) for s in survivors]
    tv_p75 = sorted(tv_values)[int(len(tv_values) * 0.75)] if len(tv_values) >= 4 else 50e9
    tv_p25 = sorted(tv_values)[int(len(tv_values) * 0.25)] if len(tv_values) >= 4 else 5e9

    for sig in survivors:
        zone = sig.get("zone_score", 0)
        rr = sig.get("risk_reward", 0)

        catalyst_boost = 1.0

        # Grok 뉴스 실적 서프라이즈
        news_data = sig.get("news_data")
        if news_data:
            earnings = news_data.get("earnings_estimate", {})
            if earnings.get("surprise_direction") == "beat":
                catalyst_boost = 1.10

        # DART 공시 촉매 부스트
        dart = sig.get("dart_analysis", {})
        if dart.get("catalyst_type") == "catalyst" and dart.get("confidence", 0) >= 0.7:
            catalyst_boost *= 1.10

        # SD 교차필터: 외국인 × 공매도
        # v10.1: 마스터 스위치 OFF → sd_mult 항상 1.0, 공매도 데이터 무시
        sd_mult = 1.0
        if use_short_filter:
            f_net = sig.get("foreign_net_5d", 0)
            s_chg = sig.get("short_balance_chg_5d", 0)
            if f_net > 0 and s_chg < 0:       # 외국인 매수 + 공매도 커버링
                sd_mult = 1.05
                sig["sd_cross"] = "양호"
            elif f_net < 0 and s_chg > 0:      # 외국인 매도 + 공매도 빌딩
                sd_mult = 0.90
                sig["sd_cross"] = "경고"
            else:
                sig["sd_cross"] = "중립"
        else:
            sig["sd_cross"] = "비활성"

        # 관찰자 밀도 (역가중)
        tv = sig.get("avg_trading_value_20d", 0)
        density_mult = 1.0
        if tv > tv_p75:                    # 고밀도 (과밀)
            density_mult = 0.95
            sig["density"] = "고밀도"
        elif tv < tv_p25:                  # 저밀도 (숨은 보석)
            density_mult = 1.05
            sig["density"] = "저밀도"
        else:
            sig["density"] = "보통"

        # US Overnight 섹터 부스트/Kill
        us_mult = 1.0
        sig_sector = sig.get("sector", "")
        sig["us_sector_killed"] = False

        # 1) 섹터 Kill 체크 (최우선)
        for kr_sector, kill_info in us_kills.items():
            if kr_sector and kr_sector in sig_sector and kill_info.get("killed"):
                us_mult = 0.0  # KILL
                sig["us_sector_match"] = kr_sector
                sig["us_sector_killed"] = True
                break

        # 2) Kill 아닌 경우: L1 섹터 부스트
        if not sig["us_sector_killed"]:
            for kr_sector, sec_info in us_sectors.items():
                if kr_sector and kr_sector in sig_sector:
                    sec_score = sec_info.get("score", 0)
                    us_mult = 1.0 + sec_score * 0.10  # bullish +5%, bearish -5%
                    sig["us_sector_match"] = kr_sector
                    break
            # 지수 전체 방향 (섹터 매칭 안 될 때)
            if us_mult == 1.0 and abs(us_score) > 0.15:
                us_mult = 1.0 + us_score * 0.05

            # 3) 특수 룰 global_cap 적용
            if global_cap < 1.0:
                us_mult = min(us_mult, global_cap)

        sig["v9_rank_score"] = round(rr * zone * catalyst_boost * sd_mult * density_mult * us_mult, 4)
        sig["v9_catalyst_boost"] = catalyst_boost
        sig["v9_sd_mult"] = sd_mult
        sig["v9_density_mult"] = density_mult
        sig["v9_us_mult"] = round(us_mult, 3)

    # US 섹터 Kill된 종목을 killed_list로 이동
    us_killed = [s for s in survivors if s.get("us_sector_killed")]
    if us_killed:
        for s in us_killed:
            s["v9_kill_reasons"] = [f"US 섹터 KILL ({s.get('us_sector_match', '?')})"]
        killed_list.extend(us_killed)
        survivors = [s for s in survivors if not s.get("us_sector_killed")]
        print(f"  US 섹터 Kill: {len(us_killed)}종목 제거")

    # Tags (sd_cross, density 태그 포함)
    for sig in survivors:
        sig["v9_tags"] = generate_tags(sig)

    # 순위 정렬
    survivors.sort(key=lambda s: s["v9_rank_score"], reverse=True)

    return survivors, killed_list


# =========================================================
# Grok 뉴스 검색
# =========================================================

def fetch_grok_news(name: str, ticker: str) -> dict | None:
    """Grok API로 종목 심층 분석 (동기 직접 호출)."""
    try:
        import requests as _req
        from dotenv import load_dotenv as _ld
        _ld(Path(__file__).resolve().parent.parent / ".env")

        from src.adapters.grok_news_adapter import GrokNewsAdapter
        adapter = GrokNewsAdapter()
        if not adapter.api_key:
            return None

        prompt = adapter._deep_analysis_prompt(name, ticker)
        payload = {
            "model": "grok-4-1-fast",
            "input": [
                {"role": "system", "content": (
                    "너는 한국 주식시장 전문 리서치 애널리스트다. "
                    "웹과 X(트위터)를 검색해서 종목의 최신 뉴스뿐 아니라, "
                    "아직 해소되지 않은 과거 이슈, 실적 전망, 수급 동향을 "
                    "종합적으로 분석한다. 반드시 요청된 JSON 형식으로만 응답한다."
                )},
                {"role": "user", "content": prompt},
            ],
            "tools": [{"type": "web_search"}, {"type": "x_search"}],
        }
        resp = _req.post(
            "https://api.x.ai/v1/responses",
            headers=adapter.headers,
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            return None
        return adapter._parse_response(resp.json())
    except Exception as e:
        print(f"  ! Grok fail ({name}): {e}")
        return None


# =========================================================
# 데이터 로드 + SignalEngine Pipeline
# =========================================================

def load_all_parquets() -> dict[str, pd.DataFrame]:
    """data/processed/*.parquet 로드 → {ticker: DataFrame}"""
    data = {}
    for pq in sorted(PROCESSED_DIR.glob("*.parquet")):
        df = pd.read_parquet(pq)
        if len(df) >= 200:
            data[pq.stem] = df
    return data


def load_name_map() -> dict:
    """stock_data_daily CSV 파일명에서 종목명 추출."""
    name_map = {}
    csv_dir = Path(__file__).resolve().parent.parent / "stock_data_daily"
    if csv_dir.exists():
        for f in csv_dir.glob("*.csv"):
            match = re.search(r"_(\d{6})$", f.stem)
            if match:
                ticker = match.group(1)
                name = f.stem[: f.stem.rfind("_")]
                name_map[ticker] = name
    return name_map


def _calc_di(df: pd.DataFrame, idx: int, period: int = 14) -> tuple[float, float]:
    """ADX의 +DI, -DI를 직접 계산 (parquet에 미포함이므로)."""
    if len(df) < period + 2 or idx < period + 1:
        return 0.0, 0.0

    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    # 상대적으로 큰 쪽만 살림
    mask_plus = plus_dm < minus_dm
    mask_minus = minus_dm < plus_dm
    plus_dm = plus_dm.where(~mask_plus, 0)
    minus_dm = minus_dm.where(~mask_minus, 0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, min_periods=period).mean()
    p_di = 100 * plus_dm.ewm(span=period, min_periods=period).mean() / atr
    m_di = 100 * minus_dm.ewm(span=period, min_periods=period).mean() / atr

    return float(p_di.iloc[idx] or 0), float(m_di.iloc[idx] or 0)


def _calc_streak(series: pd.Series) -> int:
    """최근 연속 순매수 일수 계산 (음수면 순매도 연속)."""
    if series.empty:
        return 0
    vals = series.values
    if pd.isna(vals[-1]) or vals[-1] == 0:
        return 0
    direction = 1 if vals[-1] > 0 else -1
    count = 0
    for v in reversed(vals):
        if pd.isna(v) or v == 0:
            break
        if (v > 0 and direction > 0) or (v < 0 and direction < 0):
            count += 1
        else:
            break
    return count * direction


# =========================================================
# 메인 스캔
# =========================================================

def scan_all(
    grade_filter: str = "A",
    use_news: bool = True,
    use_dart: bool = False,
    universe: str = "parquet",
) -> tuple[list[dict], dict]:
    """전 종목 스캔 -> Grade 필터 -> Kill→Rank→Tag 반환.

    universe: "parquet" (101종목), "all" (parquet + CSV 전체)
    """
    from src.signal_engine import SignalEngine

    # 데이터 신선도 검증 (대책 D)
    freshness = validate_data_freshness()
    print(f"  데이터 신선도: {freshness['msg']}")
    if not freshness["fresh"]:
        print(f"  ⚠ 경고: {freshness['msg']}")

    # 데이터 로드 (parquet)
    data_dict = load_all_parquets()
    name_map = load_name_map()
    pq_label = f"{len(data_dict)} parquet"
    csv_label = ""
    if universe == "all":
        csv_label = " + CSV 전종목"
    print(f"scan: {pq_label}{csv_label} | grade={grade_filter} | news={'ON' if use_news else 'OFF'}")

    # SignalEngine 초기화
    engine = SignalEngine("config/settings.yaml")
    scanner = MarketSignalScanner()

    candidates = []
    stats = {
        "total": len(data_dict),
        "loaded": 0,
        "passed_pipeline": 0,
        "trigger_impulse": 0,
        "trigger_confirm": 0,
        "grade_A": 0,
        "grade_B": 0,
        "grade_C": 0,
        "after_grade_filter": 0,
    }

    t0 = time.time()

    for i, (ticker, df) in enumerate(data_dict.items()):
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(data_dict)} ({elapsed:.1f}s)")

        stats["loaded"] += 1
        idx = len(df) - 1

        try:
            result = engine.calculate_signal(ticker, df, idx)
        except Exception:
            continue

        if not result["signal"]:
            continue

        grade = result["grade"]
        stats["passed_pipeline"] += 1

        trigger_type = result.get("trigger_type", "none")
        if trigger_type == "impulse":
            stats["trigger_impulse"] += 1
        elif trigger_type == "confirm":
            stats["trigger_confirm"] += 1

        grade_key = f"grade_{grade}"
        stats[grade_key] = stats.get(grade_key, 0) + 1

        if grade not in grade_filter:
            continue

        stats["after_grade_filter"] += 1

        # DataFrame에서 필드 추출
        row = df.iloc[idx]
        name = name_map.get(ticker, ticker)

        # 수급 streak 계산
        foreign_streak = 0
        inst_streak = 0
        foreign_amount_5d = 0
        inst_amount_5d = 0

        if "foreign_net" in df.columns:
            f_series = df["foreign_net"].iloc[max(0, idx - 19) : idx + 1].fillna(0)
            foreign_streak = _calc_streak(f_series)
            foreign_amount_5d = int(df["foreign_net"].iloc[max(0, idx - 4) : idx + 1].fillna(0).sum())

        if "inst_net" in df.columns:
            i_series = df["inst_net"].iloc[max(0, idx - 19) : idx + 1].fillna(0)
            inst_streak = _calc_streak(i_series)
            inst_amount_5d = int(df["inst_net"].iloc[max(0, idx - 4) : idx + 1].fillna(0).sum())

        sig = {
            "ticker": ticker,
            "name": name,
            "grade": grade,
            "zone_score": result["zone_score"],
            "trigger_type": trigger_type,
            "confidence": result.get("trigger_confidence", 0),
            "entry_price": result["entry_price"],
            "stop_loss": result["stop_loss"],
            "target_price": result["target_price"],
            "risk_reward": result.get("risk_reward_ratio", 0),
            # DataFrame에서 직접 추출
            "rsi": float(row.get("rsi_14", 50) or 50),
            "adx": float(row.get("adx_14", 0) or 0),
            "plus_di": 0.0,
            "minus_di": 0.0,
            "vol_surge": float(row.get("volume_surge_ratio", 1.0) or 1.0),
            "obv_trend": (
                "up" if (row.get("obv", 0) or 0) > (df.iloc[max(0, idx - 20)].get("obv", 0) or 0)
                else "down"
            ),
            # 수급
            "foreign_streak": foreign_streak,
            "inst_streak": inst_streak,
            "foreign_amount_5d": foreign_amount_5d,
            "inst_amount_5d": inst_amount_5d,
            # SD 교차필터 (Part 2: 외국인 × 공매도)
            "foreign_net_5d": float(row.get("foreign_net_5d", 0) or 0),
            "short_balance_chg_5d": float(row.get("short_balance_chg_5d", 0) or 0),
            # SignalEngine 고급 필드
            "consensus": result.get("consensus"),
            # 유동성 + 고점 필드
            "avg_trading_value_20d": float(
                (df["close"] * df["volume"]).iloc[max(0, idx - 19) : idx + 1].mean()
            ),
            "pct_of_52w_high": float(row.get("pct_of_52w_high", 0) or 0),
        }

        # +DI/-DI 직접 계산 (parquet에 미포함)
        try:
            p_di, m_di = _calc_di(df, idx)
            sig["plus_di"] = p_di
            sig["minus_di"] = m_di
        except Exception:
            pass

        # Market Signal Scanner
        try:
            market_signals = scanner.scan_all(df, idx)
            sig["market_signals"] = [
                {"title": s.title, "importance": s.importance, "confidence": s.confidence}
                for s in market_signals
            ] if market_signals else []
        except Exception:
            sig["market_signals"] = []

        candidates.append(sig)

    scan_elapsed = time.time() - t0
    stats["scan_sec"] = round(scan_elapsed, 1)

    # -- CSV 유니버스 스캔 (--universe all) --
    if universe == "all":
        pq_tickers = set(data_dict.keys())
        csv_data = load_csv_stocks(exclude_tickers=pq_tickers)
        stats["csv_total"] = len(csv_data)
        stats["csv_passed"] = 0
        print(f"\nCSV 유니버스 스캔: {len(csv_data)}종목 (parquet 제외)...")
        csv_t0 = time.time()

        for j, (ticker, (csv_df, name)) in enumerate(csv_data.items()):
            if (j + 1) % 500 == 0:
                el = time.time() - csv_t0
                print(f"  CSV {j+1}/{len(csv_data)} ({el:.1f}s)")

            idx = len(csv_df) - 1
            scored = quick_csv_score(csv_df, idx)
            if scored is None:
                continue

            # trigger가 none/waiting/setup이면 후보 아님
            if scored["trigger_type"] in ("none", "waiting"):
                continue

            # zone 최소 기준
            if scored["zone_score"] < 0.3:
                continue

            scored["ticker"] = ticker
            scored["name"] = name
            stats["csv_passed"] += 1
            candidates.append(scored)

        csv_elapsed = time.time() - csv_t0
        stats["csv_scan_sec"] = round(csv_elapsed, 1)
        print(f"  CSV 스캔 완료: {stats['csv_passed']}종목 통과 ({csv_elapsed:.1f}s)")

    # 뉴스/DART 초기화
    for sig in candidates:
        sig["news_data"] = None

    # -- Kill→Rank→Tag (1차: 뉴스 없이) --
    # v10.1: 마스터 스위치로 SD 교차필터(외국인×공매도) 제어
    _use_short = engine.config.get("use_short_selling_filter", False)
    survivors, killed = run_pipeline(candidates, use_short_filter=_use_short)
    stats["v9_killed"] = len(killed)
    stats["v9_survivors"] = len(survivors)
    stats["v9_killed_list"] = killed

    # -- Grok 뉴스 적용 (Kill 생존자 상위 15개) --
    NEWS_LIMIT = 15
    if use_news and survivors:
        news_targets = survivors[:NEWS_LIMIT]
        print(f"\nGrok news (생존 상위 {len(news_targets)}/{len(survivors)} stocks)...")
        news_t0 = time.time()
        for sig in news_targets:
            print(f"  {sig['name']}({sig['ticker']})...", end=" ", flush=True)
            news_data = fetch_grok_news(sig["name"], sig["ticker"])
            sig["news_data"] = news_data
            if news_data:
                sentiment = news_data.get("overall_sentiment", "?")
                takeaway = news_data.get("key_takeaway", "")[:30]
                print(f"OK [{sentiment}] {takeaway}")
            else:
                print("- (empty)")
        stats["news_sec"] = round(time.time() - news_t0, 1)

        # 뉴스 촉매 부스트 반영 → 재정렬
        for sig in survivors:
            boost = sig.get("v9_catalyst_boost", 1.0)
            news = sig.get("news_data")
            if news:
                earnings = news.get("earnings_estimate", {})
                if earnings.get("surprise_direction") == "beat":
                    boost = max(boost, 1.10)
                    sig["v9_catalyst_boost"] = boost
                    sig["v9_rank_score"] = round(
                        sig.get("risk_reward", 0)
                        * sig.get("zone_score", 0)
                        * boost
                        * sig.get("v9_sd_mult", 1.0)
                        * sig.get("v9_density_mult", 1.0)
                        * sig.get("v9_us_mult", 1.0),
                        4,
                    )
        survivors.sort(key=lambda s: s["v9_rank_score"], reverse=True)
    else:
        stats["news_sec"] = 0

    # -- DART 공시 분류 --
    if use_dart and survivors:
        try:
            from src.adapters.dart_adapter import DartAdapter
            from src.adapters.openai_classifier import classify_batch

            print(f"\nDART 공시 분류 ({len(survivors)} stocks)...")
            dart_adapter = DartAdapter()
            dart_results = classify_batch(survivors, dart_adapter)

            dart_catalyst_count = 0
            for sig in survivors:
                dr = dart_results.get(sig["ticker"], {})
                sig["dart_analysis"] = dr
                if dr.get("catalyst_type") == "catalyst" and dr.get("confidence", 0) >= 0.7:
                    dart_catalyst_count += 1
                    print(f"  촉매 발견: {sig['name']}({sig['ticker']}) — "
                          f"{dr.get('catalyst_category','')} ({dr.get('reason','')[:40]})")

            stats["dart_catalyst_count"] = dart_catalyst_count
            stats["dart_total"] = len(dart_results)
            print(f"  DART 분류 완료: {dart_catalyst_count}/{len(dart_results)} 촉매 발견")
        except Exception as e:
            print(f"  DART 공시 분류 실패 (fail-safe 계속): {e}")
            stats["dart_catalyst_count"] = 0
    else:
        stats["dart_catalyst_count"] = 0

    stats["elapsed_sec"] = round(time.time() - t0, 1)
    return survivors, stats


# =========================================================
# 텔레그램 메시지 포맷
# =========================================================

def format_telegram_message(candidates: list[dict], stats: dict) -> str:
    """Kill→Rank→Tag 텔레그램 메시지 포맷."""
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    # -- Header --
    lines.append(f"[Quant v10.0] {now} Kill\u2192Rank\u2192Tag")
    lines.append("")

    # -- US Overnight --
    us_signal = load_overnight_signal()
    us_comp = us_signal.get("composite", "neutral").upper()
    us_sc = us_signal.get("score", 0.0)
    us_vix = us_signal.get("vix", {})
    if us_comp != "NEUTRAL" or abs(us_sc) > 0.05:
        lines.append(f"[ US Overnight: {us_comp} ({us_sc:+.2f}) ]")
        idx = us_signal.get("index_direction", {})
        spy_r = idx.get("SPY", {}).get("ret_1d", 0)
        qqq_r = idx.get("QQQ", {}).get("ret_1d", 0)
        lines.append(f"  SPY {spy_r:+.1f}% QQQ {qqq_r:+.1f}% | VIX {us_vix.get('level','?')} [{us_vix.get('status','?')}]")
        lines.append("")

    # -- 파이프라인 설명 --
    lines.append("[ 파이프라인 ]")
    lines.append("Kill(K3+K4) \u2192 Rank(R:R\u00d7Zone\u00d7US\u00d7Den) \u2192 Tag")
    lines.append("  \u00b7 선행: Zone + R:R + Trigger")
    lines.append("  \u00b7 US Overnight: 섹터별 부스트")
    lines.append("  \u00b7 밀도: 고밀도(-5%), 저밀도(+5%)")
    lines.append("")

    # -- 스캔 통계 --
    killed = stats.get("v9_killed", 0)
    survivors = stats.get("v9_survivors", 0)
    lines.append("[ 스캔 통계 ]")
    lines.append(
        f"Parquet: {stats['total']:,}종목 > Pipeline: {stats['passed_pipeline']}종목"
    )
    if stats.get("csv_total"):
        lines.append(
            f"CSV 전종목: {stats['csv_total']:,}종목 > 통과: {stats.get('csv_passed', 0)}종목"
        )
    lines.append(
        f"등급: A:{stats.get('grade_A',0)} B:{stats.get('grade_B',0)} "
        f"C:{stats.get('grade_C',0)} | 필터 후: {stats.get('after_grade_filter',0)}종목"
    )
    lines.append(f"Kill: {killed}종목 | 생존: {survivors}종목")
    scan_time = f"스캔 {stats.get('scan_sec',0)}초"
    if stats.get("csv_scan_sec"):
        scan_time += f" + CSV {stats['csv_scan_sec']}초"
    scan_time += f" + 뉴스 {stats.get('news_sec',0)}초"
    lines.append(f"소요: {scan_time}")
    lines.append("")

    if not candidates:
        lines.append("Kill 필터 통과 종목 없음")
        killed_list = stats.get("v9_killed_list", [])
        if killed_list:
            lines.append("")
            lines.append(f"[ Kill ({len(killed_list)}종목) ]")
            for sig in killed_list:
                reasons = ", ".join(sig.get("v9_kill_reasons", []))
                lines.append(f"  {sig['name']}({sig['ticker']}): {reasons}")
        return "\n".join(lines)

    # -- 1순위 추천 매수 --
    top = candidates[0]
    top_trigger = "확인매수" if top["trigger_type"] == "confirm" else ("IMP" if top["trigger_type"] == "impulse" else "SETUP")
    top_tags = ", ".join(top.get("v9_tags", []))
    boost = top.get("v9_catalyst_boost", 1.0)
    us_m = top.get("v9_us_mult", 1.0)
    den_m = top.get("v9_density_mult", 1.0)
    mods = []
    if boost > 1.0:
        mods.append("촉매")
    if us_m != 1.0:
        mods.append(f"US{us_m:.2f}")
    if den_m != 1.0:
        mods.append(f"밀도{den_m:.2f}")
    mod_str = f" [{','.join(mods)}]" if mods else ""

    lines.append("[ 1순위 추천 매수 ]")
    csv_mark = " [CSV]" if top.get("grade") == "CSV" else ""
    lines.append(f"{top['name']} ({top['ticker']}) [{top_trigger}]{csv_mark}")
    lines.append(
        f"Rank {top['v9_rank_score']:.3f} = "
        f"R:R({top['risk_reward']:.1f}) x Zone({top['zone_score']:.2f}){mod_str}"
    )
    lines.append(
        f"현재 {top['entry_price']:,}원 | "
        f"목표 {top['target_price']:,} (+{((top['target_price']/top['entry_price'])-1)*100:.1f}%) | "
        f"손절 {top['stop_loss']:,} ({((top['stop_loss']/top['entry_price'])-1)*100:.1f}%)"
    )
    if top_tags:
        lines.append(f"태그: {top_tags}")

    # -- 나머지 후보 (상위 20개까지 표시) --
    DISPLAY_LIMIT = 20
    if len(candidates) > 1:
        display = candidates[1:DISPLAY_LIMIT]
        remaining = len(candidates) - DISPLAY_LIMIT
        lines.append("")
        lines.append(f"[ 매수 후보 ({len(candidates)-1}개) ]")
        for i, sig in enumerate(display, start=2):
            tags = ", ".join(sig.get("v9_tags", []))
            b = sig.get("v9_catalyst_boost", 1.0)
            b_str = " x1.10" if b > 1.0 else ""
            csv_m = " [CSV]" if sig.get("grade") == "CSV" else ""
            lines.append(
                f"{i}. {sig['name']}({sig['ticker']}) "
                f"Rank {sig['v9_rank_score']:.3f} "
                f"RR:{sig['risk_reward']:.1f} Zone:{sig['zone_score']:.2f}{b_str}{csv_m}"
            )
            if tags:
                lines.append(f"   [{tags}]")
        if remaining > 0:
            lines.append(f"  ... +{remaining}종목 (scan_result.txt 참조)")

    # -- Kill 요약 --
    killed_list = stats.get("v9_killed_list", [])
    if killed_list:
        lines.append("")
        lines.append(f"[ Kill ({len(killed_list)}종목) ]")
        for sig in killed_list[:5]:
            reasons = ", ".join(sig.get("v9_kill_reasons", []))
            lines.append(f"  {sig['name']}({sig['ticker']}): {reasons}")
        if len(killed_list) > 5:
            lines.append(f"  ... +{len(killed_list)-5}종목")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Quant Buy Scan (Kill→Rank→Tag)")
    parser.add_argument("--no-send", action="store_true", help="No telegram send")
    parser.add_argument("--grade", type=str, default="A", help="Grade filter (A, AB, ABC)")
    parser.add_argument("--no-news", action="store_true", help="Skip Grok news")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report")
    parser.add_argument("--dart", action="store_true", help="DART 공시 + OpenAI 촉매 분류")
    parser.add_argument("--universe", type=str, default="parquet",
                        choices=["parquet", "all"],
                        help="parquet: 101종목만 | all: 전체 KOSPI/KOSDAQ CSV 포함")
    args = parser.parse_args()

    dart_label = " + DART" if args.dart else ""
    uni_label = " [전종목]" if args.universe == "all" else ""
    print("=" * 50)
    print(f"  [Quant v10.0] Kill\u2192Rank\u2192Tag{dart_label}{uni_label}")
    print(f"  Kill(K3+K4) \u2192 Rank(R:R\u00d7Zone) \u2192 Tag")
    print("=" * 50)

    candidates, stats = scan_all(
        grade_filter=args.grade.upper(),
        use_news=not args.no_news,
        use_dart=args.dart,
        universe=args.universe,
    )

    msg = format_telegram_message(candidates, stats)
    print("\n" + msg)

    # HTML 보고서 생성 + PNG 변환
    png_path = None
    if not args.no_html and candidates:
        try:
            from src.html_report import generate_premarket_report
            print("\nHTML 보고서 생성 중...")
            html_path, png_path = generate_premarket_report(candidates, stats)
            print(f"HTML: {html_path}")
            if png_path:
                print(f"PNG:  {png_path}")
        except Exception as e:
            print(f"HTML 보고서 생성 실패: {e}")

    if not args.no_send:
        from src.telegram_sender import send_message

        # 1) PNG 이미지 전송 (보고서)
        if png_path and png_path.exists():
            from src.html_report import send_report_to_telegram
            print("\nSending report image to Telegram...")
            caption = f"[Quant v10.0] 장시작전 분석 | {len(candidates)}종목 | Grade {args.grade.upper()}"
            img_ok = send_report_to_telegram(png_path, caption)
            print("OK - Report image sent" if img_ok else "FAIL - Image send")

        # 2) 텍스트 메시지 전송
        print("Sending text to Telegram...")
        success = send_message(msg)
        print("OK - Text sent" if success else "FAIL - Check .env")
    else:
        print("\n(--no-send: skipped)")

    output_path = Path(__file__).parent.parent / "data" / "scan_result.txt"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(msg, encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
