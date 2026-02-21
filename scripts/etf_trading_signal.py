"""ETF 매매 시그널 생성기 — Smart Money ETF + Theme Money ETF.

섹터 모멘텀 + 수급 + 기술지표 조합으로 ETF 매매 시그널을 생성한다.

분류 체계:
  Smart Money ETF: SMART 섹터 + BB% < 80 + RSI < 60 + Rank ≤ 12 → FULL매수
                   SMART 섹터 + RSI < 65 → 관찰
  Theme Money ETF: ADX > 40↑ + RSI < 80 → HALF매수
                   Stoch GX + ADX > 35 → GX매수

사용법:
  python scripts/etf_trading_signal.py               # 시그널 생성 + JSON 저장 (기본)
  python scripts/etf_trading_signal.py --send          # 시그널 생성 + 텔레그램 발송
  python scripts/etf_trading_signal.py --json-only     # JSON 저장만 (콘솔 출력 없음)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
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
ETF_DAILY_DIR = DATA_DIR / "etf_daily"
OUT_PATH = DATA_DIR / "etf_trading_signal.json"

# ─────────────────────────────────────────────
# 수급 섹터명 매핑 (investor_flow.json → sector_momentum.json)
# ─────────────────────────────────────────────
FLOW_TO_ETF = {
    "반도체": "반도체",
    "2차전지": "2차전지",
    "자동차": "현대차그룹",
    "조선": "조선",
    "금융": "금융",
    "방산/항공": "방산",
    "바이오": "바이오",
    "IT/소프트웨어": "소프트웨어",
    "철강/화학": "에너지화학",
    "유틸리티": "유틸리티",  # ETF 없으면 무시
    "건설": "건설",
}


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def load_momentum() -> list[dict]:
    """sector_momentum.json 로드."""
    path = DATA_DIR / "sector_momentum.json"
    if not path.exists():
        logger.error("sector_momentum.json 없음 — sector_momentum.py 먼저 실행")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sectors", [])


def load_flow() -> dict[str, dict]:
    """investor_flow.json → {etf_sector_name: {foreign_cum, inst_cum}}."""
    path = DATA_DIR / "investor_flow.json"
    if not path.exists():
        logger.warning("investor_flow.json 없음 — 수급 없이 진행")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flow_map = {}
    sectors_data = data.get("sectors", {})
    if isinstance(sectors_data, dict):
        for sector_name, vals in sectors_data.items():
            etf_name = FLOW_TO_ETF.get(sector_name, sector_name)
            flow_map[etf_name] = {
                "foreign_cum_bil": vals.get("foreign_cum", vals.get("foreign_cum_bil", 0)),
                "inst_cum_bil": vals.get("inst_cum", vals.get("inst_cum_bil", 0)),
            }
    elif isinstance(sectors_data, list):
        for s in sectors_data:
            flow_map[s["sector"]] = {
                "foreign_cum_bil": s.get("foreign_cum_bil", 0),
                "inst_cum_bil": s.get("inst_cum_bil", 0),
            }
    return flow_map


def load_etf_ohlcv(etf_code: str) -> pd.DataFrame | None:
    """ETF 일별 OHLCV parquet 로드."""
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
    """최신 RSI 값 (Wilder's Smoothing)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1)


def calc_bb_pct(close: pd.Series, period: int = 20) -> float:
    """최신 Bollinger Band % (0~100)."""
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    bb = (close - lower) / (upper - lower) * 100
    val = bb.iloc[-1]
    return round(float(val), 1) if not np.isnan(val) else 50.0


def calc_adx(df: pd.DataFrame, period: int = 14) -> tuple[float, float]:
    """최신 ADX 값 + 전일 ADX (상승/하락 판단용)."""
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
    return round(adx_now, 1), round(adx_prev, 1)


def calc_stoch(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> dict:
    """Stoch Slow K/D + 골든크로스 여부."""
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

    golden_cross = k_prev < d_prev and k_now > d_now

    return {
        "slow_k": round(k_now, 1),
        "slow_d": round(d_now, 1),
        "golden_cross": golden_cross,
    }


def calc_ma20_gap(close: pd.Series) -> float:
    """종가 vs MA20 괴리율(%)."""
    ma20 = close.rolling(20).mean()
    gap = (close.iloc[-1] / ma20.iloc[-1] - 1) * 100
    return round(float(gap), 1) if not np.isnan(gap) else 0.0


# ─────────────────────────────────────────────
# SMART 섹터 판정
# ─────────────────────────────────────────────

def classify_smart_sectors(flow_map: dict) -> set[str]:
    """외인+기관 동시 순매수 섹터 → SMART."""
    smart = set()
    for sector, vals in flow_map.items():
        foreign = vals.get("foreign_cum_bil", 0)
        inst = vals.get("inst_cum_bil", 0)
        if foreign > 0 and inst > 0:
            smart.add(sector)
    return smart


# ─────────────────────────────────────────────
# ETF 시그널 생성
# ─────────────────────────────────────────────

def generate_etf_signals() -> dict:
    """전체 ETF 시그널 생성."""
    momentum_list = load_momentum()
    flow_map = load_flow()
    smart_sectors = classify_smart_sectors(flow_map)

    logger.info("모멘텀 ETF: %d개, 수급: %d섹터, SMART: %s",
                len(momentum_list), len(flow_map), smart_sectors or "없음")

    signals = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "smart_sectors": sorted(smart_sectors),
        "smart_money_etf": [],
        "theme_money_etf": [],
        "watch_list": [],
        "summary": {},
    }

    for sec in momentum_list:
        sector = sec["sector"]
        etf_code = sec["etf_code"]
        rank = sec["rank"]

        # ETF 가격 데이터 로드
        df = load_etf_ohlcv(etf_code)
        if df is None or len(df) < 30:
            logger.warning("%s (%s): 가격 데이터 부족", sector, etf_code)
            continue

        close = df["close"].astype(float)

        # 기술지표 계산
        rsi = calc_rsi(close)
        bb_pct = calc_bb_pct(close)
        adx_now, adx_prev = calc_adx(df)
        adx_rising = adx_now > adx_prev
        stoch = calc_stoch(df)
        ma20_gap = calc_ma20_gap(close)

        # 수급 정보
        flow = flow_map.get(sector, {})
        foreign_bil = flow.get("foreign_cum_bil", 0)
        inst_bil = flow.get("inst_cum_bil", 0)
        is_smart = sector in smart_sectors

        entry = {
            "sector": sector,
            "etf_code": etf_code,
            "momentum_rank": rank,
            "momentum_score": sec.get("momentum_score", 0),
            "rsi": rsi,
            "bb_pct": bb_pct,
            "adx": adx_now,
            "adx_rising": adx_rising,
            "stoch_k": stoch["slow_k"],
            "stoch_d": stoch["slow_d"],
            "stoch_gx": stoch["golden_cross"],
            "ma20_gap": ma20_gap,
            "foreign_5d_bil": foreign_bil,
            "inst_5d_bil": inst_bil,
            "is_smart": is_smart,
            "ret_5": sec.get("ret_5", 0),
            "ret_20": sec.get("ret_20", 0),
        }

        # ── Smart Money ETF 시그널 ──
        if is_smart:
            if bb_pct < 80 and rsi < 60 and rank <= 12:
                entry["signal"] = "SMART_BUY"
                entry["sizing"] = "FULL"
                entry["stop_loss"] = "-5%"
                entry["hold_days"] = 30
                entry["reason"] = f"SMART섹터 + BB{bb_pct:.0f}% + RSI{rsi:.0f} + Rank{rank}"
                signals["smart_money_etf"].append(entry)
            elif rsi < 65:
                entry["signal"] = "SMART_WATCH"
                entry["sizing"] = "HALF"
                entry["stop_loss"] = "-5%"
                entry["hold_days"] = 30
                entry["reason"] = f"SMART섹터 + RSI{rsi:.0f} (BB{bb_pct:.0f}% 높음)"
                signals["watch_list"].append(entry)
            continue  # SMART은 Theme과 중복 분류 안함

        # ── Theme Money ETF 시그널 ──
        # GX 매수 (Stoch 골든크로스 + ADX > 35)
        if stoch["golden_cross"] and adx_now > 35:
            entry["signal"] = "THEME_GX"
            entry["sizing"] = "HALF"
            entry["stop_loss"] = "-3%"
            entry["hold_days"] = 10
            entry["reason"] = f"Stoch GX★ + ADX{adx_now:.0f}"
            signals["theme_money_etf"].append(entry)

        # ADX 강추세 + RSI 적정
        elif adx_now > 40 and adx_rising and rsi < 80:
            entry["signal"] = "THEME_TREND"
            entry["sizing"] = "HALF"
            entry["stop_loss"] = "-3%"
            entry["hold_days"] = 10
            entry["reason"] = f"ADX{adx_now:.0f}↑ + RSI{rsi:.0f}"
            signals["theme_money_etf"].append(entry)

        # 모멘텀 상위 + BB 저위치 (관찰)
        elif rank <= 7 and bb_pct < 50 and rsi < 60:
            entry["signal"] = "MOMENTUM_DIP"
            entry["sizing"] = "QUARTER"
            entry["stop_loss"] = "-3%"
            entry["hold_days"] = 5
            entry["reason"] = f"Top{rank} + BB{bb_pct:.0f}% 저위치 + RSI{rsi:.0f}"
            signals["watch_list"].append(entry)

    # 요약
    signals["summary"] = {
        "total_etf": len(momentum_list),
        "smart_buy": len(signals["smart_money_etf"]),
        "theme_buy": len(signals["theme_money_etf"]),
        "watch": len(signals["watch_list"]),
    }

    return signals


# ─────────────────────────────────────────────
# JSON 저장
# ─────────────────────────────────────────────

def save_signals(signals: dict) -> Path:
    """시그널 JSON 저장."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(signals, f, ensure_ascii=False, indent=2)
    logger.info("ETF 시그널 저장 → %s", OUT_PATH)
    return OUT_PATH


# ─────────────────────────────────────────────
# 텔레그램 메시지 생성
# ─────────────────────────────────────────────

def build_telegram_message(signals: dict) -> str:
    """ETF 매매 시그널 텔레그램 메시지 생성."""
    date = signals.get("date", "?")
    lines = []

    lines.append(f"📊 ETF 매매 시그널 — {date}")
    lines.append("━" * 28)
    lines.append("")

    # SMART 섹터 표시
    smart = signals.get("smart_sectors", [])
    if smart:
        lines.append(f"🔵 SMART 섹터: {', '.join(smart)}")
    else:
        lines.append("🔵 SMART 섹터: 없음")
    lines.append("")

    # ── Smart Money ETF ──
    smart_etfs = signals.get("smart_money_etf", [])
    lines.append("💎 Smart Money ETF (FULL매수)")
    lines.append("─" * 28)
    if smart_etfs:
        for e in smart_etfs:
            lines.append(f"  🟢 {e['sector']} ({e['etf_code']})")
            lines.append(f"    {e['reason']}")
            lines.append(f"    RSI {e['rsi']} | BB {e['bb_pct']}% | ADX {e['adx']}")
            f_bil = e.get('foreign_5d_bil', 0)
            i_bil = e.get('inst_5d_bil', 0)
            lines.append(f"    외인 {f_bil:+,}억 | 기관 {i_bil:+,}억")
            lines.append(f"    손절 {e['stop_loss']} | {e['hold_days']}일 보유")
            lines.append("")
    else:
        lines.append("  해당 없음")
        lines.append("")

    # ── Theme Money ETF ──
    theme_etfs = signals.get("theme_money_etf", [])
    lines.append("🔥 Theme Money ETF (HALF매수)")
    lines.append("─" * 28)
    if theme_etfs:
        for e in theme_etfs:
            gx_mark = " ★GX" if e.get("stoch_gx") else ""
            lines.append(f"  🟡 {e['sector']} ({e['etf_code']}){gx_mark}")
            lines.append(f"    {e['reason']}")
            lines.append(f"    RSI {e['rsi']} | BB {e['bb_pct']}% | ADX {e['adx']}")
            lines.append(f"    Stoch K{e['stoch_k']}/D{e['stoch_d']} | MA20 {e['ma20_gap']:+.1f}%")
            lines.append(f"    손절 {e['stop_loss']} | {e['hold_days']}일 보유")
            lines.append("")
    else:
        lines.append("  해당 없음")
        lines.append("")

    # ── 관찰 목록 ──
    watch = signals.get("watch_list", [])
    if watch:
        lines.append("👀 관찰 목록")
        lines.append("─" * 28)
        for e in watch:
            tag = "SMART" if e.get("is_smart") else "DIP"
            lines.append(f"  ⚪ [{tag}] {e['sector']} — {e['reason']}")
        lines.append("")

    # ── 요약 ──
    s = signals.get("summary", {})
    lines.append(f"📋 요약: SMART {s.get('smart_buy', 0)}개 | THEME {s.get('theme_buy', 0)}개 | 관찰 {s.get('watch', 0)}개")
    lines.append("")
    lines.append("⚠️ ETF 매매는 개별종목 시그널과 병행 운용")
    lines.append("⚠️ 투자 판단은 본인 책임 | Quantum Master")

    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    """텔레그램 전송."""
    try:
        from src.telegram_sender import send_message
        return send_message(message)
    except Exception as e:
        logger.error("텔레그램 전송 실패: %s", e)
        return False


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ETF 매매 시그널 생성기")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송 (기본: 저장만)")
    parser.add_argument("--json-only", action="store_true", help="JSON 저장만 (콘솔 출력 없음)")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("  ETF 매매 시그널 생성 시작")
    logger.info("=" * 50)

    # 시그널 생성
    signals = generate_etf_signals()

    # JSON 저장
    save_signals(signals)

    # 콘솔 출력
    s = signals.get("summary", {})
    print(f"\n  Smart Money ETF: {s.get('smart_buy', 0)}개")
    print(f"  Theme Money ETF: {s.get('theme_buy', 0)}개")
    print(f"  관찰 목록:       {s.get('watch', 0)}개\n")

    for label, key in [("Smart", "smart_money_etf"), ("Theme", "theme_money_etf")]:
        for e in signals.get(key, []):
            print(f"  [{label}] {e['sector']}: {e['signal']} — {e['reason']}")
    for e in signals.get("watch_list", []):
        tag = "SMART" if e.get("is_smart") else "DIP"
        print(f"  [관찰/{tag}] {e['sector']}: {e['reason']}")

    if args.json_only:
        return

    # 텔레그램 전송 (--send 명시 시에만)
    if args.send:
        msg = build_telegram_message(signals)
        ok = send_telegram(msg)
        if ok:
            logger.info("텔레그램 전송 완료 (%d자)", len(msg))
        else:
            logger.error("텔레그램 전송 실패")
    else:
        logger.info("텔레그램 미발송 (--send 로 발송 가능)")


if __name__ == "__main__":
    main()
