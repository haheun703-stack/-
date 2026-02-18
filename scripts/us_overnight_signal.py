"""
US Overnight Signal — Step 2: 매일 오전 신호 생성

미국 시장 전일 종가 기반으로 한국 장시작전 신호를 생성:
  1. 지수 방향 (SPY/QQQ/DIA 종합)
  2. VIX 레벨 (공포 / 정상 / 안정)
  3. 섹터 모멘텀 (한국 업종별 영향도)
  4. 종합 신호 (Bullish / Neutral / Bearish)

출력: data/us_market/overnight_signal.json (scan에서 읽어 사용)
      텔레그램 메시지 (선택)

사용법:
    python scripts/us_overnight_signal.py [--send] [--update]
    --send:   텔레그램으로 전송
    --update: yfinance로 최신 1일 추가 후 신호 생성
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

US_DIR = Path("data/us_market")
PARQUET_PATH = US_DIR / "us_daily.parquet"
SIGNAL_PATH = US_DIR / "overnight_signal.json"

# 한국 섹터 매핑 (US ETF → KR 업종 키워드)
US_KR_SECTOR_MAP = {
    "xlk": ["반도체", "IT", "소프트웨어", "전자부품", "디스플레이"],
    "soxx": ["반도체", "전자부품", "디스플레이"],
    "xlf": ["은행", "증권", "보험", "금융"],
    "xle": ["에너지", "정유", "화학"],
    "xli": ["조선", "기계", "건설", "자동차", "운송"],
    "xlv": ["제약", "바이오", "의료기기", "헬스케어"],
}

DB_PATH = US_DIR / "us_kr_history.db"

# ── 섹터 Kill 설정 (US ETF 기준) ──
# kill_col: parquet의 ret_1d 컬럼 (분수 형태, 0.01=1%)
# kill_threshold: 이 이하면 해당 한국 섹터 KILL
SECTOR_KILL_CONFIG = {
    "반도체":   {"kill_col": "soxx_ret_1d", "threshold": -0.03, "sensitivity": 0.95},
    "전자부품": {"kill_col": "soxx_ret_1d", "threshold": -0.03, "sensitivity": 0.95},
    "디스플레이": {"kill_col": "soxx_ret_1d", "threshold": -0.03, "sensitivity": 0.95},
    "IT":       {"kill_col": "qqq_ret_1d",  "threshold": -0.035, "sensitivity": 0.70},
    "소프트웨어": {"kill_col": "qqq_ret_1d",  "threshold": -0.035, "sensitivity": 0.70},
    "에너지":   {"kill_col": "spy_ret_1d",  "threshold": -0.05, "sensitivity": 0.80},
    "정유":     {"kill_col": "spy_ret_1d",  "threshold": -0.05, "sensitivity": 0.80},
    "화학":     {"kill_col": "spy_ret_1d",  "threshold": -0.05, "sensitivity": 0.80},
    "은행":     {"kill_col": "spy_ret_1d",  "threshold": -0.05, "sensitivity": 0.50},
    "증권":     {"kill_col": "spy_ret_1d",  "threshold": -0.05, "sensitivity": 0.50},
    "금융":     {"kill_col": "spy_ret_1d",  "threshold": -0.05, "sensitivity": 0.50},
    "제약":     {"kill_col": "qqq_ret_1d",  "threshold": -0.05, "sensitivity": 0.45},
    "바이오":   {"kill_col": "qqq_ret_1d",  "threshold": -0.05, "sensitivity": 0.45},
    "의료기기": {"kill_col": "qqq_ret_1d",  "threshold": -0.05, "sensitivity": 0.45},
    "헬스케어": {"kill_col": "qqq_ret_1d",  "threshold": -0.05, "sensitivity": 0.45},
    "조선":     {"kill_col": "spy_ret_1d",  "threshold": -0.07, "sensitivity": 0.25},
    "기계":     {"kill_col": "spy_ret_1d",  "threshold": -0.07, "sensitivity": 0.25},
    "건설":     {"kill_col": "spy_ret_1d",  "threshold": -0.07, "sensitivity": 0.25},
    "자동차":   {"kill_col": "spy_ret_1d",  "threshold": -0.07, "sensitivity": 0.25},
    "운송":     {"kill_col": "spy_ret_1d",  "threshold": -0.07, "sensitivity": 0.25},
}


def _compute_sector_kills(latest) -> dict:
    """섹터별 Kill 판정."""
    kills = {}
    for sector, cfg in SECTOR_KILL_CONFIG.items():
        ret = float(latest.get(cfg["kill_col"], 0) or 0)
        killed = ret <= cfg["threshold"]
        kills[sector] = {
            "killed": killed,
            "driver_ret": round(ret * 100, 2),
            "threshold_pct": round(cfg["threshold"] * 100, 1),
            "sensitivity": cfg["sensitivity"],
        }
    return kills


def _check_special_rules(latest, prev) -> list[dict]:
    """특수 상황 룰 체크."""

    def _ret(col):
        return float(latest.get(col, 0) or 0)

    def _close_chg(col):
        cur = float(latest.get(col, 0) or 0)
        prv = float(prev.get(col, 0) or 0)
        if prv > 0 and cur > 0:
            return (cur - prv) / prv
        return 0.0

    vix_chg = _close_chg("vix_close")
    vix_level = float(latest.get("vix_close", 20) or 20)
    qqq_ret = _ret("qqq_ret_1d")
    spy_ret = _ret("spy_ret_1d")
    soxx_ret = _ret("soxx_ret_1d")

    triggered = []

    # R1: VIX 급등 20%+ → 전체 포지션 캡 50%
    if vix_chg > 0.20:
        triggered.append({
            "name": "VIX_SPIKE",
            "desc": f"VIX 급등 {vix_chg*100:+.1f}% -> 전체 포지션 축소",
            "global_position_cap": 0.5,
        })

    # R2: VIX 30 이상 → 경계 모드
    if vix_level >= 30:
        triggered.append({
            "name": "VIX_HIGH",
            "desc": f"VIX {vix_level:.0f} 고공포 -> 전체 포지션 축소",
            "global_position_cap": 0.7,
        })

    # R3: SOXX -5% 폭락 → 반도체 KILL
    if soxx_ret <= -0.05:
        triggered.append({
            "name": "SOXX_CRASH",
            "desc": f"SOXX {soxx_ret*100:+.1f}% 폭락 -> 반도체 KILL",
            "sector_kill": ["반도체", "전자부품", "디스플레이"],
        })

    # R4: NASDAQ -3% → 성장주 직격
    if qqq_ret <= -0.03:
        triggered.append({
            "name": "NASDAQ_CIRCUIT",
            "desc": f"NASDAQ {qqq_ret*100:+.1f}% -> 성장주 경계",
            "global_position_cap": 0.7,
        })

    # R5: 트리플 상승 (NASDAQ+2%, SPY+2%, SOXX+2%) → 리스크온 부스트
    if qqq_ret >= 0.02 and spy_ret >= 0.02 and soxx_ret >= 0.02:
        triggered.append({
            "name": "TRIPLE_BULL",
            "desc": "트리플 상승 -> 리스크온 부스트",
            "global_position_boost": 1.3,
        })

    # R6: SPY -3% → 시장 전체 경계
    if spy_ret <= -0.03:
        triggered.append({
            "name": "MARKET_CRASH",
            "desc": f"SPY {spy_ret*100:+.1f}% -> 전체 시장 경계",
            "global_position_cap": 0.5,
        })

    return triggered


def _run_level2_pattern(df: pd.DataFrame, latest, prev) -> dict:
    """Level 2 패턴매칭 실행. DB 없으면 스킵."""
    try:
        from scripts.backfill_us_kr_history import PatternMatcher
    except ImportError:
        return {"status": "import_error", "pattern_adjustment": 0, "confidence": 0}

    if not DB_PATH.exists():
        return {"status": "no_db", "pattern_adjustment": 0, "confidence": 0}

    # parquet에서 US 변화율(%) 추출
    def _pct(col):
        """ret_1d 컬럼(분수) → 퍼센트 변환."""
        v = float(latest.get(col, 0) or 0)
        return round(v * 100, 4)

    def _close_chg_pct(col):
        """종가 기반 변화율 직접 계산 (ret_1d 없는 경우)."""
        cur = float(latest.get(col, 0) or 0)
        prv = float(prev.get(col, 0) or 0)
        if prv > 0 and cur > 0:
            return round(((cur - prv) / prv) * 100, 4)
        return 0.0

    today_us = {
        "us_nasdaq_chg": _pct("qqq_ret_1d"),
        "us_sp500_chg":  _pct("spy_ret_1d"),
        "us_vix_chg":    _close_chg_pct("vix_close"),
        "us_soxx_chg":   _pct("soxx_ret_1d"),
        "us_dollar_chg": _close_chg_pct("uup_close"),
        "us_ewy_chg":    _pct("ewy_ret_1d"),
    }

    try:
        matcher = PatternMatcher(str(DB_PATH))
        similar = matcher.find_similar_patterns(today_us)
        result = matcher.analyze_patterns(similar)
        result["today_us_vector"] = today_us
        return result
    except Exception as e:
        logger.warning(f"Level 2 패턴매칭 실패: {e}")
        return {"status": "error", "pattern_adjustment": 0, "confidence": 0}


def update_latest() -> pd.DataFrame:
    """yfinance로 최신 데이터 추가 (증분)."""
    import yfinance as yf
    from scripts.us_overnight_backfill import TICKERS, _calc_derived

    if not PARQUET_PATH.exists():
        logger.error(f"백필 데이터 없음: {PARQUET_PATH} → 먼저 us_overnight_backfill.py 실행")
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)
    last_date = df.index.max()
    today = pd.Timestamp.now().normalize()

    if last_date >= today - pd.Timedelta(days=1):
        logger.info(f"이미 최신 (마지막: {last_date.date()})")
        return df

    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    logger.info(f"증분 업데이트: {start} ~ {end}")

    for ticker in TICKERS:
        try:
            obj = yf.Ticker(ticker)
            new = obj.history(start=start, end=end)
            if new is None or new.empty:
                continue

            new.index = pd.to_datetime(new.index).tz_localize(None).normalize()
            prefix = ticker.replace("^", "").lower()
            cols = {
                "Close": f"{prefix}_close",
                "Volume": f"{prefix}_volume",
                "High": f"{prefix}_high",
                "Low": f"{prefix}_low",
            }
            new = new.rename(columns=cols)
            available = [c for c in cols.values() if c in new.columns]
            new = new[available]

            # 기존 데이터에 병합
            for col in available:
                if col in df.columns:
                    for idx in new.index:
                        if idx not in df.index:
                            df.loc[idx, col] = new.loc[idx, col]
                        elif pd.isna(df.loc[idx, col]):
                            df.loc[idx, col] = new.loc[idx, col]
                else:
                    df[col] = pd.NA
                    for idx in new.index:
                        df.loc[idx, col] = new.loc[idx, col]

        except Exception as e:
            logger.warning(f"  {ticker}: {e}")

    df = df.sort_index()
    df = df.ffill()
    df = _calc_derived(df)
    df.to_parquet(PARQUET_PATH)
    logger.info(f"업데이트 완료: {len(df)}일")

    return df


def generate_signal(df: pd.DataFrame | None = None) -> dict:
    """US Overnight Signal 생성.

    Returns:
        {
            "date": "2026-02-18",
            "us_close_date": "2026-02-17",
            "composite": "bullish" | "neutral" | "bearish",
            "score": float (-1.0 ~ +1.0),
            "index_direction": {"spy": +0.5%, "qqq": +0.8%, ...},
            "vix": {"level": 18.5, "zscore": -0.3, "status": "안정"},
            "sector_signals": {
                "반도체": {"signal": "bullish", "driver": "SOXX +2.1%"},
                ...
            },
            "summary": "요약 텍스트"
        }
    """
    if df is None:
        if not PARQUET_PATH.exists():
            logger.error(f"데이터 없음: {PARQUET_PATH}")
            return {"composite": "neutral", "score": 0.0, "error": "데이터 없음"}
        df = pd.read_parquet(PARQUET_PATH)

    if df.empty:
        return {"composite": "neutral", "score": 0.0, "error": "빈 데이터"}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    us_date = df.index[-1]

    signal = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "us_close_date": us_date.strftime("%Y-%m-%d"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    score = 0.0

    # ─── 1. 지수 방향 (가중치: 45%) ───
    # EWY(한국 프록시)가 가장 높은 비중 → KOSPI 직접 선행지표
    index_dir = {}
    index_score = 0.0

    for prefix, weight in [("ewy", 0.30), ("qqq", 0.25), ("spy", 0.25), ("dia", 0.20)]:
        ret_col = f"{prefix}_ret_1d"
        ret_5d_col = f"{prefix}_ret_5d"
        close_col = f"{prefix}_close"

        ret_1d = float(latest.get(ret_col, 0) or 0)
        ret_5d = float(latest.get(ret_5d_col, 0) or 0)
        above_sma = int(latest.get(f"{prefix}_above_sma20", 0) or 0)

        # 1일 수익률 기반 점수 (-1 ~ +1)
        day_score = max(-1.0, min(1.0, ret_1d * 50))  # ±2% → ±1.0
        # 5일 추세 보너스
        trend_bonus = max(-0.3, min(0.3, ret_5d * 10))
        # SMA 위치 보너스
        sma_bonus = 0.1 if above_sma else -0.1

        idx_score = day_score * 0.6 + trend_bonus * 0.25 + sma_bonus * 0.15
        index_score += idx_score * weight

        index_dir[prefix.upper()] = {
            "ret_1d": round(ret_1d * 100, 2),
            "ret_5d": round(ret_5d * 100, 2),
            "above_sma20": bool(above_sma),
        }

    signal["index_direction"] = index_dir
    score += index_score * 0.45

    # ─── 2. VIX (가중치: 20%) ───
    vix = float(latest.get("vix_close", 20) or 20)
    vix_z = float(latest.get("vix_zscore", 0) or 0)
    vix_spike = int(latest.get("vix_spike", 0) or 0)

    if vix_z < -0.5:
        vix_status = "안정"
        vix_score = 0.5
    elif vix_z < 0.5:
        vix_status = "정상"
        vix_score = 0.0
    elif vix_z < 1.5:
        vix_status = "경계"
        vix_score = -0.4
    else:
        vix_status = "공포"
        vix_score = -0.8

    if vix_spike:
        vix_score -= 0.2
        vix_status += " (스파이크)"

    signal["vix"] = {
        "level": round(vix, 1),
        "zscore": round(vix_z, 2),
        "status": vix_status,
    }
    score += vix_score * 0.20

    # ─── 3. 채권/달러 (가중치: 10%) ───
    bond_dollar_score = 0.0
    tlt_ret = float(latest.get("tlt_ret_1d", 0) or 0)
    # TLT 상승 = risk-off → 주식에 약세
    bond_dollar_score -= max(-0.5, min(0.5, tlt_ret * 30))

    score += bond_dollar_score * 0.10

    # ─── 4. 섹터 신호 (가중치: 25%) ───
    sector_signals = {}
    sector_score_sum = 0.0
    sector_count = 0

    for us_etf, kr_sectors in US_KR_SECTOR_MAP.items():
        rel_col = f"{us_etf}_rel_spy_5d"
        ret_col = f"{us_etf}_ret_1d"
        close_col = f"{us_etf}_close"

        rel_5d = float(latest.get(rel_col, 0) or 0)
        ret_1d = float(latest.get(ret_col, 0) or 0)

        # 섹터 판정
        if rel_5d > 0.01 and ret_1d > 0:
            sec_signal = "bullish"
            sec_score = 0.5
        elif rel_5d < -0.01 and ret_1d < 0:
            sec_signal = "bearish"
            sec_score = -0.5
        else:
            sec_signal = "neutral"
            sec_score = 0.0

        driver = f"{us_etf.upper()} 1D:{ret_1d*100:+.1f}%, 상대강도5D:{rel_5d*100:+.1f}%"

        for kr_sector in kr_sectors:
            if kr_sector not in sector_signals:
                sector_signals[kr_sector] = {
                    "signal": sec_signal,
                    "score": round(sec_score, 2),
                    "driver": driver,
                }
                sector_score_sum += sec_score
                sector_count += 1

    signal["sector_signals"] = sector_signals
    if sector_count > 0:
        score += (sector_score_sum / sector_count) * 0.25

    # ─── 종합 판정 ───
    score = max(-1.0, min(1.0, score))

    if score > 0.15:
        composite = "bullish"
    elif score < -0.15:
        composite = "bearish"
    else:
        composite = "neutral"

    signal["composite"] = composite
    signal["score"] = round(score, 3)

    # ─── 5. Level 2: 패턴매칭 보정 ───
    l2 = _run_level2_pattern(df, latest, prev)
    signal["l2_pattern"] = l2

    # ─── 6. 섹터 Kill 판정 ───
    signal["sector_kills"] = _compute_sector_kills(latest)

    # ─── 7. 특수 상황 룰 ───
    signal["special_rules"] = _check_special_rules(latest, prev)

    l1_100 = round(score * 100, 1)
    l2_adj = l2.get("pattern_adjustment", 0)
    combined_100 = max(-100.0, min(100.0, l1_100 + l2_adj))

    if combined_100 >= 50:
        grade = "STRONG_BULL"
    elif combined_100 >= 20:
        grade = "MILD_BULL"
    elif combined_100 > -20:
        grade = "NEUTRAL"
    elif combined_100 > -50:
        grade = "MILD_BEAR"
    else:
        grade = "STRONG_BEAR"

    signal["grade"] = grade
    signal["l1_score_100"] = l1_100
    signal["combined_score_100"] = round(combined_100, 1)

    # 요약 생성
    spy_ret = index_dir.get("SPY", {}).get("ret_1d", 0)
    qqq_ret = index_dir.get("QQQ", {}).get("ret_1d", 0)
    l2_conf = l2.get("confidence", 0)
    l2_adj_str = f" L2:{l2_adj:+.1f}" if l2_conf > 0 else ""
    signal["summary"] = (
        f"US {grade} ({combined_100:+.1f}{l2_adj_str}) | "
        f"SPY {spy_ret:+.1f}% QQQ {qqq_ret:+.1f}% | "
        f"VIX {vix:.0f} ({vix_status})"
    )

    # 저장
    US_DIR.mkdir(parents=True, exist_ok=True)
    with open(SIGNAL_PATH, "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)

    logger.info(f"신호: {signal['summary']}")
    logger.info(f"저장: {SIGNAL_PATH}")

    return signal


def format_telegram_message(signal: dict) -> str:
    """텔레그램 메시지 포맷."""
    grade = signal.get("grade", signal.get("composite", "NEUTRAL").upper())
    combined = signal.get("combined_score_100", signal.get("score", 0) * 100)

    lines = [
        f"[US Overnight] {signal.get('us_close_date', '?')}",
        f"Grade: {grade} (Score: {combined:+.1f})",
        "",
    ]

    # 지수
    idx = signal.get("index_direction", {})
    if idx:
        lines.append("[ 지수 ]")
        for name, info in idx.items():
            sma = ">" if info.get("above_sma20") else "<"
            lines.append(
                f"  {name}: {info['ret_1d']:+.1f}% (5D:{info['ret_5d']:+.1f}%) {sma}SMA20"
            )

    # VIX
    vix = signal.get("vix", {})
    if vix:
        lines.append(f"\nVIX: {vix['level']} (z:{vix['zscore']:+.1f}) [{vix['status']}]")

    # Level 2 패턴매칭
    l2 = signal.get("l2_pattern", {})
    if l2.get("status") == "ok":
        kospi = l2.get("kospi", {})
        lines.append(
            f"\n[ L2 패턴매칭 ] ({l2.get('sample_count', 0)}건, "
            f"신뢰도 {l2.get('confidence', 0):.0%})"
        )
        lines.append(
            f"  KOSPI 예측: {kospi.get('mean_chg', 0):+.2f}% "
            f"(상승확률 {kospi.get('positive_rate', 50):.0f}%)"
        )
        lines.append(f"  보정값: {l2.get('pattern_adjustment', 0):+.1f}")

        # 섹터별 예측 (상위 3개만)
        l2_sectors = l2.get("sectors", {})
        if l2_sectors:
            sorted_s = sorted(l2_sectors.items(), key=lambda x: abs(x[1].get("mean_chg", 0)), reverse=True)
            for s_name, s_val in sorted_s[:3]:
                lines.append(
                    f"  {s_name}: {s_val['mean_chg']:+.2f}% "
                    f"(상승 {s_val['positive_rate']:.0f}%)"
                )

    # 특수 룰
    rules = signal.get("special_rules", [])
    if rules:
        lines.append("\n[ 특수 룰 ]")
        for r in rules:
            lines.append(f"  {r['name']}: {r['desc']}")

    # 섹터 Kill
    kills = signal.get("sector_kills", {})
    killed_sectors = [s for s, v in kills.items() if v.get("killed")]
    if killed_sectors:
        lines.append(f"\n[ 섹터 KILL ] {', '.join(killed_sectors)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="US Overnight Signal 생성")
    parser.add_argument("--send", action="store_true", help="텔레그램 전송")
    parser.add_argument("--update", action="store_true", help="yfinance 최신 데이터 추가 후 생성")
    args = parser.parse_args()

    if args.update:
        df = update_latest()
    else:
        df = None

    signal = generate_signal(df)

    if args.send and signal.get("composite"):
        msg = format_telegram_message(signal)
        try:
            from src.telegram_sender import send_message
            send_message(msg)
            logger.info("텔레그램 전송 완료")
        except Exception as e:
            logger.warning(f"텔레그램 전송 실패: {e}")
            print(msg)
    else:
        print(format_telegram_message(signal))


if __name__ == "__main__":
    main()
