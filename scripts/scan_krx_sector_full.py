"""KRX 업종분류 기반 전종목 섹터 스캔 + Zone A/B/C 프레임워크.

ETF 구성종목이 아닌 KRX 공식 업종분류로 전종목을 매핑하고,
섹터별 모멘텀 + z-score + 수급 분류를 수행한다.

핵심:
  - ETF 11종목이 아닌 KRX 전체(예: 증권 29종목)를 커버
  - Smart Money: FULL 사이즈, BB% < 40 우선, z_5 > z_20 회복, -7% 손절
  - Theme Money: HALF 사이즈, Zone A만 진입, -3% 손절, 5일 엑싯
  - Zone A (추세 가속): ADX>40 + Zone B 신호 2개 미만 → 진입
  - Zone B (천장 형성): 4개 중 2개 이상 → 진입 불가
    {리젝션캔들, 거래량폭증>4x, OBV 다이버전스, ADX 하락>-3}
  - Zone C (약한 추세): ADX<40 → 대기

사용법:
  python scripts/scan_krx_sector_full.py                # 핵심 섹터 전체
  python scripts/scan_krx_sector_full.py --sector 증권    # 특정 업종만
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
DAILY_DIR = DATA_DIR / "etf_daily"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# KRX 업종 → ETF 매핑 (모멘텀+수급 참조용)
KRX_TO_ETF = {
    "증권": "증권",
    "은행": "은행",
    "보험": "금융",
    "기타금융": "금융",
    "건설": "건설",
    "전기·전자": "반도체",  # 가장 가까운 ETF
    "화학": "에너지화학",
    "금속": "철강소재",
    "기계·장비": "건설",
    "운송장비·부품": "현대차그룹",
    "IT 서비스": "소프트웨어",
    "오락·문화": "미디어",
    "제약": "바이오",
    "의료·정밀기기": "헬스케어",
}

# 핵심 분석 대상 업종
TARGET_SECTORS = [
    "증권", "은행", "보험", "기타금융",
    "건설", "전기·전자", "화학", "금속",
    "기계·장비", "운송장비·부품",
    "IT 서비스", "오락·문화",
    "제약", "의료·정밀기기",
]


def calc_zone_signals(df: pd.DataFrame) -> dict:
    """Zone B 4개 신호를 계산하여 Zone A/B/C 판정.

    Zone B signals (2개 이상 → Zone B = 진입불가):
      1. 리젝션 캔들: 윗꼬리 > 몸통 × 1.5
      2. 거래량 폭증: 당일 거래량 > 20일 평균 × 4
      3. OBV 다이버전스: 가격 신고가 근처인데 OBV는 이전 고점 대비 하락
      4. ADX 방향 전환: 최근 3일 ADX 변화 < -3 (하락 반전)
    """
    if len(df) < 25:
        return {"zone": "C", "adx": 0, "signals": [], "signal_count": 0}

    last = df.iloc[-1]
    adx = float(last.get("adx_14", 0))

    if adx < 40:
        return {"zone": "C", "adx": round(adx, 1), "signals": [], "signal_count": 0}

    signals = []

    # Signal 1: 리젝션 캔들 (최근 3일 중 하나라도)
    for i in range(-3, 0):
        row = df.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        body = abs(c - o)
        upper_wick = h - max(o, c)
        if body > 0 and upper_wick > body * 1.5:
            signals.append("REJECTION")
            break

    # Signal 2: 거래량 폭증 (당일 > 20일 평균 × 4)
    vol = df["volume"].astype(float)
    vol_avg_20 = vol.iloc[-21:-1].mean()
    vol_today = vol.iloc[-1]
    if vol_avg_20 > 0 and vol_today > vol_avg_20 * 4:
        signals.append("VOL_CLIMAX")

    # Signal 3: OBV 다이버전스 (가격 20일 고점 근처인데 OBV 하락)
    close = df["close"].astype(float)
    high_20 = close.iloc[-20:].max()
    close_last = close.iloc[-1]
    if close_last >= high_20 * 0.97:  # 고점 3% 이내
        obv_col = "obv" if "obv" in df.columns else None
        if obv_col:
            obv = df[obv_col].astype(float)
            obv_at_high = obv.iloc[-20:].max()
            obv_now = obv.iloc[-1]
            if obv_now < obv_at_high * 0.95:  # OBV 5% 이상 하락
                signals.append("OBV_DIV")

    # Signal 4: ADX 방향 전환 (3일간 -3 이상 하락)
    if "adx_14" in df.columns and len(df) >= 4:
        adx_3d_ago = float(df["adx_14"].iloc[-4])
        adx_now = float(df["adx_14"].iloc[-1])
        adx_delta = adx_now - adx_3d_ago
        if adx_delta < -3:
            signals.append("ADX_FALL")

    signal_count = len(signals)
    zone = "B" if signal_count >= 2 else "A"

    return {
        "zone": zone,
        "adx": round(adx, 1),
        "signals": signals,
        "signal_count": signal_count,
        "adx_delta_3d": round(adx_now - adx_3d_ago, 1) if "adx_14" in df.columns and len(df) >= 4 else 0,
    }


def calc_extra_indicators(df: pd.DataFrame) -> dict:
    """BB%, 이격도, z_5, OBV 등 추가 지표 계산."""
    if len(df) < 25:
        return {}

    last = df.iloc[-1]
    close = df["close"].astype(float)
    close_val = float(close.iloc[-1])

    # BB% (Bollinger Band %B)
    bb_upper = float(last.get("bb_upper", 0)) if "bb_upper" in df.columns else 0
    bb_lower = float(last.get("bb_lower", 0)) if "bb_lower" in df.columns else 0
    bb_pct = 0
    if bb_upper > bb_lower > 0:
        bb_pct = (close_val - bb_lower) / (bb_upper - bb_lower) * 100

    # 이격도 (20일 이평선 대비 %)
    ma20 = close.rolling(20).mean().iloc[-1]
    disparity = (close_val / ma20 * 100) if ma20 > 0 else 100

    # 연속 상승일
    consec_up = 0
    for i in range(len(close) - 1, 0, -1):
        if close.iloc[i] > close.iloc[i - 1]:
            consec_up += 1
        else:
            break

    # OBV (있으면)
    obv_val = float(last.get("obv", 0)) if "obv" in df.columns else None

    return {
        "bb_pct": round(bb_pct, 1),
        "disparity_20": round(disparity, 1),
        "consecutive_up": consec_up,
        "obv": obv_val,
    }


def find_sector_leader(stock_data: dict, flow_map: dict, etf_sector: str) -> dict | None:
    """섹터 내 시총 최대 종목(=리더)의 ADX 방향 확인."""
    if not stock_data:
        return None
    leader = max(stock_data.items(), key=lambda x: x[1]["market_cap"])
    ticker, data = leader
    df = data["df"]
    if len(df) < 4 or "adx_14" not in df.columns:
        return None
    adx_now = float(df["adx_14"].iloc[-1])
    adx_3d = float(df["adx_14"].iloc[-4])
    return {
        "ticker": ticker,
        "name": data["name"],
        "adx": round(adx_now, 1),
        "adx_delta_3d": round(adx_now - adx_3d, 1),
        "adx_rising": adx_now > adx_3d,
    }


def load_json(filename: str) -> dict | None:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stock_data(ticker: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def load_etf_close(etf_code: str) -> pd.Series | None:
    path = DAILY_DIR / f"{etf_code}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df["close"].astype(float)


def analyze_sector(
    krx_sector_name: str,
    stocks: list[dict],
    momentum_map: dict,
    flow_map: dict,
    etf_universe: dict,
    min_cap_bil: float = 500,
    relay_override_codes: set[str] | None = None,
) -> list[dict]:
    """KRX 업종 내 전종목을 분석하여 후보 반환."""

    # ETF 매핑으로 모멘텀/수급 데이터 참조
    etf_sector = KRX_TO_ETF.get(krx_sector_name, "")
    mom = momentum_map.get(etf_sector, {})
    fl = flow_map.get(etf_sector, {})

    mom_rank = mom.get("rank", 99)
    mom_score = mom.get("momentum_score", 0)
    sector_ret_20 = mom.get("ret_20", 0)

    foreign_cum = fl.get("foreign_cum_bil", 0)
    inst_cum = fl.get("inst_cum_bil", 0)

    # 수급 분류
    is_smart = foreign_cum > 0 and inst_cum > 0
    is_theme = foreign_cum < -500 and (inst_cum > 0 or mom_rank <= 5)

    # ETF 종가 (z-score 벤치마크)
    etf_info = etf_universe.get(etf_sector, {})
    etf_code = etf_info.get("etf_code", "")
    etf_close = load_etf_close(etf_code) if etf_code else None
    etf_ret20 = None
    if etf_close is not None:
        etf_ret20 = etf_close.pct_change(20) * 100

    # 시총 필터 후 종목별 분석
    filtered = [s for s in stocks if s["market_cap"] / 1e8 >= min_cap_bil]

    # 종목별 수익률 수집 (z-score 계산용)
    stock_data = {}
    for s in filtered:
        ticker = s["code"]
        # 우선주 특수코드(00680K 등) 제외
        if not ticker.isdigit():
            continue
        df = load_stock_data(ticker)
        if df is None or len(df) < 25:
            continue
        close = df["close"].astype(float)
        ret20 = float(close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else np.nan
        ret5 = float(close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 6 else np.nan
        # 일평균 거래대금 (20일, 억원)
        vol = df["volume"].astype(float)
        avg_daily_value = vol.iloc[-21:-1].mean() * float(close.iloc[-1]) / 1e8

        stock_data[ticker] = {
            "name": s["name"],
            "market_cap": s["market_cap"],
            "market": s["market"],
            "ret20": ret20,
            "ret5": ret5,
            "avg_daily_value_bil": round(avg_daily_value, 1),
            "df": df,
        }

    if len(stock_data) < 2:
        return []

    # z-score 계산: 섹터 내 상대 수익률 기준 (20일 & 5일)
    rets20 = [v["ret20"] for v in stock_data.values() if not np.isnan(v["ret20"])]
    rets5 = [v["ret5"] for v in stock_data.values() if not np.isnan(v["ret5"])]
    mean_ret20 = np.mean(rets20)
    std_ret20 = np.std(rets20, ddof=1) if len(rets20) > 2 else 1.0
    if std_ret20 < 0.5:
        std_ret20 = 0.5
    mean_ret5 = np.mean(rets5) if rets5 else 0
    std_ret5 = np.std(rets5, ddof=1) if len(rets5) > 2 else 1.0
    if std_ret5 < 0.5:
        std_ret5 = 0.5

    # 섹터 리더 ADX 확인
    leader = find_sector_leader(stock_data, flow_map, etf_sector)

    results = []
    for ticker, data in stock_data.items():
        if np.isnan(data["ret20"]):
            continue

        z_20 = (data["ret20"] - mean_ret20) / std_ret20
        z_5 = (data["ret5"] - mean_ret5) / std_ret5 if not np.isnan(data["ret5"]) else np.nan
        df = data["df"]
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        # Stoch Slow
        stoch_k = last.get("stoch_slow_k")
        stoch_d = last.get("stoch_slow_d")
        prev_k = prev.get("stoch_slow_k")
        prev_d = prev.get("stoch_slow_d")

        stoch_golden = False
        stoch_golden_recent = False
        if all(v is not None and not np.isnan(v) for v in [stoch_k, stoch_d, prev_k, prev_d]):
            stoch_golden = bool(stoch_k > stoch_d and prev_k <= prev_d)
            for i in range(-5, 0):
                if i + 1 >= 0:
                    continue
                r1 = df.iloc[i]
                r2 = df.iloc[i + 1]
                sk1, sd1 = r1.get("stoch_slow_k"), r1.get("stoch_slow_d")
                sk2, sd2 = r2.get("stoch_slow_k"), r2.get("stoch_slow_d")
                if all(v is not None and not np.isnan(v) for v in [sk1, sd1, sk2, sd2]):
                    if sk2 > sd2 and sk1 <= sd1:
                        stoch_golden_recent = True
                        break

        rsi = float(last.get("rsi_14", 50))
        adx = float(last.get("adx_14", 0))
        close_val = float(last.get("close", 0))

        # Zone A/B/C 판정
        zone_info = calc_zone_signals(df)
        extra = calc_extra_indicators(df)

        # 머니 분류 + Zone 기반 사이징
        if is_smart or foreign_cum > 0:
            money_type = "SMART"
            # Smart Money: BB% < 40 우선 FULL, 그 외 HALF
            if is_smart:
                sizing = "FULL"
            else:
                sizing = "HALF"
            # Smart Money는 Zone 무관 (기본 진입)
            entry_ok = True
            stop_pct = -7.0
            hold_days = 30
        elif is_theme:
            money_type = "THEME"
            sizing = "HALF"
            stop_pct = -3.0
            hold_days = 5
            # Theme Money: Zone A만 진입, B/C 불가
            if zone_info["zone"] == "A":
                entry_ok = True
            elif (zone_info["zone"] == "B"
                  and relay_override_codes
                  and ticker in relay_override_codes):
                # 릴레이 오버라이드: Zone B → Zone A (HALF 사이즈, -3% 손절)
                entry_ok = True
                zone_info["zone"] = "A_RELAY"
                zone_info["signals"].append("RELAY_OVERRIDE")
            else:
                entry_ok = False
            # 섹터 리더 ADX 하락 시 주의
            if leader and not leader["adx_rising"] and leader["adx_delta_3d"] < -3:
                entry_ok = False
                zone_info["signals"].append("LEADER_ADX_FALL")
        elif mom_rank <= 5:
            money_type = "THEME"
            sizing = "HALF"
            stop_pct = -3.0
            hold_days = 5
            if zone_info["zone"] == "A":
                entry_ok = True
            elif (zone_info["zone"] == "B"
                  and relay_override_codes
                  and ticker in relay_override_codes):
                entry_ok = True
                zone_info["zone"] = "A_RELAY"
                zone_info["signals"].append("RELAY_OVERRIDE")
            else:
                entry_ok = False
            if leader and not leader["adx_rising"] and leader["adx_delta_3d"] < -3:
                entry_ok = False
                zone_info["signals"].append("LEADER_ADX_FALL")
        else:
            money_type = "NEUTRAL"
            sizing = "OBS"
            entry_ok = False
            stop_pct = 0
            hold_days = 0

        # z_5 > z_20 = 회복 시작 신호
        recovering = False
        if not np.isnan(z_5) and z_5 > z_20:
            recovering = True

        # Smart Money 우선순위 태그
        sm_priority = ""
        if money_type == "SMART":
            bb = extra.get("bb_pct", 50)
            if bb < 40:
                sm_priority = "BUY_ZONE"
            elif bb < 60:
                sm_priority = "FAIR"
            else:
                sm_priority = "HIGH"

        # 유동성 필터: 거래대금 < 50억 → 1단계 다운그레이드
        avg_dv = data["avg_daily_value_bil"]
        liquidity_ok = avg_dv >= 50
        if not liquidity_ok and sizing != "OBS":
            if sizing == "FULL":
                sizing = "HALF"
            elif sizing == "HALF":
                sizing = "QUARTER"

        results.append({
            "krx_sector": krx_sector_name,
            "etf_sector": etf_sector,
            "ticker": ticker,
            "name": data["name"],
            "market": data["market"],
            "market_cap_bil": round(data["market_cap"] / 1e8),
            "ret_20": round(data["ret20"], 2),
            "ret_5": round(data["ret5"], 2) if not np.isnan(data["ret5"]) else None,
            "z_20": round(z_20, 3),
            "z_5": round(z_5, 3) if not np.isnan(z_5) else None,
            "recovering": recovering,
            "sector_mean_ret": round(mean_ret20, 2),
            "rsi": round(rsi, 1),
            "adx": round(adx, 1),
            "bb_pct": extra.get("bb_pct", None),
            "disparity_20": extra.get("disparity_20", None),
            "consecutive_up": extra.get("consecutive_up", 0),
            "stoch_k": round(float(stoch_k), 1) if stoch_k is not None and not np.isnan(stoch_k) else None,
            "stoch_d": round(float(stoch_d), 1) if stoch_d is not None and not np.isnan(stoch_d) else None,
            "stoch_golden": stoch_golden,
            "stoch_golden_recent": stoch_golden_recent,
            "zone": zone_info["zone"],
            "zone_signals": zone_info["signals"],
            "zone_signal_count": zone_info["signal_count"],
            "money_type": money_type,
            "sizing": sizing,
            "entry_ok": entry_ok,
            "stop_pct": stop_pct,
            "hold_days": hold_days,
            "sm_priority": sm_priority,
            "avg_daily_value_bil": avg_dv,
            "liquidity_ok": liquidity_ok,
            "mom_rank": mom_rank,
            "foreign_cum": foreign_cum,
            "inst_cum": inst_cum,
            "sector_leader": leader,
        })

    results.sort(key=lambda x: x["z_20"])
    return results


def main():
    parser = argparse.ArgumentParser(description="KRX 업종 전종목 섹터 스캔")
    parser.add_argument("--sector", type=str, default="",
                        help="특정 업종만 (쉼표 구분)")
    parser.add_argument("--min-cap", type=float, default=500,
                        help="최소 시총 (억, 기본 500)")
    parser.add_argument("--z-threshold", type=float, default=-0.8,
                        help="래깅 z-score 기준 (기본 -0.8)")
    args = parser.parse_args()

    # 데이터 로드
    krx_sectors = load_json("krx_full_sectors.json")
    momentum = load_json("sector_momentum.json")
    flow = load_json("investor_flow.json")
    etf_universe = load_json("etf_universe.json")

    if not krx_sectors:
        logger.error("krx_full_sectors.json 없음")
        return

    # 모멘텀/수급 맵
    momentum_map = {}
    if momentum:
        for s in momentum.get("sectors", []):
            momentum_map[s["sector"]] = s
    # 수급 JSON → ETF 섹터명 매핑 (flow JSON 키 → ETF sector name)
    FLOW_TO_ETF = {
        "반도체": "반도체", "2차전지": "2차전지", "자동차": "현대차그룹",
        "조선": "조선", "금융": "금융", "방산/항공": "방산",
        "바이오": "바이오", "IT/소프트웨어": "소프트웨어",
        "철강/화학": "에너지화학", "유틸리티": "유틸리티", "건설": "건설",
        "증권": "증권", "은행": "은행",
    }
    flow_map = {}
    if flow:
        sectors_data = flow.get("sectors", {})
        if isinstance(sectors_data, dict):
            for sector_name, vals in sectors_data.items():
                etf_name = FLOW_TO_ETF.get(sector_name, sector_name)
                flow_map[etf_name] = {
                    "sector": etf_name,
                    "foreign_cum_bil": vals.get("foreign_cum", vals.get("foreign_cum_bil", 0)),
                    "inst_cum_bil": vals.get("inst_cum", vals.get("inst_cum_bil", 0)),
                }
        elif isinstance(sectors_data, list):
            for s in sectors_data:
                flow_map[s["sector"]] = s

    # 릴레이 오버라이드 코드 로드
    relay_signal = load_json("relay_signal.json")
    relay_override_codes: set[str] = set()
    if relay_signal:
        for stock in relay_signal.get("override_stocks", []):
            relay_override_codes.add(stock["stock_code"])
        if relay_override_codes:
            logger.info("릴레이 오버라이드: %d종목 Zone B→A 대상", len(relay_override_codes))

    # 대상 섹터
    if args.sector:
        targets = [s.strip() for s in args.sector.split(",")]
    else:
        targets = TARGET_SECTORS

    all_theme = []
    all_smart = []
    all_neutral = []

    for krx_sector in targets:
        if krx_sector not in krx_sectors:
            continue

        info = krx_sectors[krx_sector]
        results = analyze_sector(
            krx_sector, info["stocks"],
            momentum_map, flow_map,
            etf_universe or {},
            min_cap_bil=args.min_cap,
            relay_override_codes=relay_override_codes or None,
        )

        lagging = [r for r in results if r["z_20"] <= args.z_threshold]
        for r in lagging:
            if r["money_type"] == "SMART":
                all_smart.append(r)
            elif r["money_type"] == "THEME":
                all_theme.append(r)
            else:
                all_neutral.append(r)

        # 섹터 요약
        etf_sector = KRX_TO_ETF.get(krx_sector, "?")
        mom = momentum_map.get(etf_sector, {})
        fl = flow_map.get(etf_sector, {})
        total_analyzed = len(results)
        lag_count = len(lagging)

        if lag_count > 0:
            print(f"\n{'─' * 80}")
            print(f"  [{krx_sector}] 총 {total_analyzed}종목 분석 → 래깅 {lag_count}종목 (z<{args.z_threshold})")
            if mom:
                print(f"  ETF: {etf_sector} [모멘텀 #{mom.get('rank','?')}] 수급: 외인{fl.get('foreign_cum_bil',0):+,.0f}억 기관{fl.get('inst_cum_bil',0):+,.0f}억")
            # 섹터 리더 ADX 정보
            if lagging and lagging[0].get("sector_leader"):
                ldr = lagging[0]["sector_leader"]
                ldr_dir = "↑" if ldr["adx_rising"] else "↓"
                print(f"  리더: {ldr['name']} ADX {ldr['adx']}{ldr_dir} ({ldr['adx_delta_3d']:+.1f})")

            print(f"  {'종목':<12} {'시총':>6} {'거래대금':>6} {'20D%':>6} {'z20':>5} {'z5':>5} {'RSI':>4} {'BB%':>5} {'이격':>5} {'Stoch':>5} {'Zone':>5} {'판정':>4} {'사이즈'}")
            print(f"  {'─' * 85}")

            for r in lagging:
                stoch_str = ""
                if r["stoch_golden"]:
                    stoch_str = "GX★"
                elif r["stoch_golden_recent"]:
                    stoch_str = "GX(r)"
                elif r["stoch_k"] is not None:
                    stoch_str = f"{r['stoch_k']:.0f}"
                else:
                    stoch_str = "N/A"

                z5_str = f"{r['z_5']:+.2f}" if r.get("z_5") is not None else "  N/A"
                bb_str = f"{r['bb_pct']:.0f}" if r.get("bb_pct") is not None else "N/A"
                disp_str = f"{r['disparity_20']:.0f}" if r.get("disparity_20") is not None else "N/A"
                zone_str = r.get("zone", "?")
                if r["money_type"] == "THEME" and zone_str == "B":
                    zone_str = "B⛔"
                elif r["money_type"] == "THEME" and zone_str == "A":
                    zone_str = "A✅"
                elif zone_str == "C":
                    zone_str = "C⏳"

                entry_mark = "GO" if r.get("entry_ok") else "NO"
                rec_mark = "↗" if r.get("recovering") else " "
                dv_str = f"{r['avg_daily_value_bil']:.0f}억"
                liq_warn = "⚠" if not r.get("liquidity_ok") else " "

                print(
                    f"  {r['name']:<12} {r['market_cap_bil']:>5,}억 {dv_str:>6}{liq_warn}{r['ret_20']:>+6.1f} "
                    f"{r['z_20']:>+5.2f} {z5_str:>5}{rec_mark} {r['rsi']:>4.0f} {bb_str:>5} {disp_str:>5} "
                    f"{stoch_str:>5} {zone_str:>5} {entry_mark:>4} {r['sizing']}"
                )
                # Zone B 신호 상세 / 유동성 경고
                notes = []
                if r.get("zone_signals"):
                    notes.append(f"Zone: {', '.join(r['zone_signals'])}")
                if not r.get("liquidity_ok"):
                    notes.append(f"유동성↓ {r['avg_daily_value_bil']:.0f}억<50억 → 다운그레이드")
                if notes:
                    print(f"    ↳ {' | '.join(notes)}")

    # ── 종합 ──
    print(f"\n{'━' * 80}")
    print(f"  종합 결과 — KRX 전종목 기준 + Zone A/B/C 프레임워크")
    print(f"{'━' * 80}")

    # Smart Money: 진입 가능한 것 우선 정렬 (BB% < 40 BUY_ZONE 우선)
    if all_smart:
        all_smart.sort(key=lambda x: (
            0 if x.get("sm_priority") == "BUY_ZONE" else 1 if x.get("sm_priority") == "FAIR" else 2,
            -1 if x.get("recovering") else 0,
            x["z_20"],
        ))
        go_count = sum(1 for r in all_smart if r.get("entry_ok"))
        print(f"\n  ◆ 스마트머니 래깅 ({len(all_smart)}종목, 진입가능 {go_count}개):")
        print(f"    {'종목':<12} {'섹터':<8} {'z20':>5} {'z5':>5} {'BB%':>5} {'이격':>5} {'Stoch':>5} {'거래대금':>6} {'Stop':>5} {'우선순위'}")
        print(f"    {'─' * 68}")
        for r in all_smart[:20]:
            stoch = "GX★" if r["stoch_golden"] else "GX(r)" if r["stoch_golden_recent"] else ""
            z5_str = f"{r['z_5']:+.2f}" if r.get("z_5") is not None else " N/A"
            bb_str = f"{r['bb_pct']:.0f}" if r.get("bb_pct") is not None else "N/A"
            disp_str = f"{r['disparity_20']:.0f}" if r.get("disparity_20") is not None else "N/A"
            rec = "↗" if r.get("recovering") else " "
            pri = r.get("sm_priority", "")
            entry = "GO" if r.get("entry_ok") else "NO"
            dv = f"{r['avg_daily_value_bil']:.0f}억"
            liq = "⚠" if not r.get("liquidity_ok") else " "
            print(f"    {r['name']:<12} {r['krx_sector']:<8} {r['z_20']:>+5.2f} {z5_str:>5}{rec} {bb_str:>5} {disp_str:>5} {stoch:>5} {dv:>5}{liq} {r['stop_pct']:>+4.0f}% {pri} {r['sizing']} {entry}")

    # Theme Money: Zone A만 진입 가능
    if all_theme:
        # 진입 가능(Zone A) 우선, 그 다음 z_20 순
        all_theme.sort(key=lambda x: (
            0 if x.get("entry_ok") else 1,
            x.get("mom_rank", 99),
            x["z_20"],
        ))
        go_count = sum(1 for r in all_theme if r.get("entry_ok"))
        no_count = len(all_theme) - go_count
        print(f"\n  ★ 테마머니 래깅 ({len(all_theme)}종목, Zone A 진입가능 {go_count}개, Zone B/C 불가 {no_count}개):")
        print(f"    {'종목':<12} {'섹터':<8} {'z20':>5} {'Zone':>5} {'ADX':>4} {'BB%':>5} {'RSI':>4} {'Stoch':>5} {'거래대금':>6} {'사이즈':>6} {'판정':>4} {'신호'}")
        print(f"    {'─' * 80}")
        for r in all_theme[:20]:
            stoch = "GX★" if r["stoch_golden"] else "GX(r)" if r["stoch_golden_recent"] else ""
            zone_str = r.get("zone", "?")
            bb_str = f"{r['bb_pct']:.0f}" if r.get("bb_pct") is not None else "N/A"
            entry = "GO" if r.get("entry_ok") else "NO"
            sigs = ",".join(r.get("zone_signals", [])) if r.get("zone_signals") else "-"
            dv = f"{r['avg_daily_value_bil']:.0f}억"
            liq = "⚠" if not r.get("liquidity_ok") else " "
            print(f"    {r['name']:<12} {r['krx_sector']:<8} {r['z_20']:>+5.2f} {zone_str:>5} {r['adx']:>4.0f} {bb_str:>5} {r['rsi']:>4.0f} {stoch:>5} {dv:>5}{liq} {r['sizing']:>6} {entry:>4} {sigs}")

    if all_neutral:
        all_neutral.sort(key=lambda x: x["z_20"])
        print(f"\n  ○ 수급중립 래깅 ({len(all_neutral)}종목, 관찰만):")
        for r in all_neutral[:10]:
            stoch = "GX★" if r["stoch_golden"] else "GX(r)" if r["stoch_golden_recent"] else ""
            print(f"    [{r['krx_sector']}] {r['name']:<12} z={r['z_20']:+.2f} {r['ret_20']:>+.1f}% OBS {stoch}")

    total = len(all_smart) + len(all_theme) + len(all_neutral)
    go_total = sum(1 for r in all_smart + all_theme if r.get("entry_ok"))
    print(f"\n  총 래깅: {total}개 (스마트 {len(all_smart)} + 테마 {len(all_theme)} + 중립 {len(all_neutral)})")
    print(f"  진입 가능: {go_total}개 (Smart GO + Theme Zone A)")
    print(f"  진입 불가: {total - go_total - len(all_neutral)}개 (Theme Zone B/C)")

    # JSON 저장 (sector_leader는 첫 종목에만 포함, 나머지 제거해서 중복 방지)
    seen_leaders = set()
    for cat in [all_smart, all_theme, all_neutral]:
        for r in cat:
            ldr = r.pop("sector_leader", None)
            if ldr and r["krx_sector"] not in seen_leaders:
                r["sector_leader_name"] = ldr["name"]
                r["sector_leader_adx"] = ldr["adx"]
                r["sector_leader_adx_rising"] = ldr["adx_rising"]
                seen_leaders.add(r["krx_sector"])

    out = {
        "scan_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "smart_money": all_smart,
        "theme_money": all_theme,
        "neutral": all_neutral,
        "summary": {
            "total_lagging": total,
            "entry_ok": go_total,
            "smart_count": len(all_smart),
            "theme_count": len(all_theme),
            "neutral_count": len(all_neutral),
        },
    }
    out_path = DATA_DIR / "krx_sector_scan.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    logger.info("결과 → %s", out_path)


if __name__ == "__main__":
    main()
