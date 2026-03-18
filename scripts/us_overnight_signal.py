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
    # 원자재 → 한국 섹터
    "gld": ["철강금속"],
    "uso": ["에너지", "정유", "화학"],
    "copx": ["조선", "기계", "건설", "자동차", "전자부품"],
}

DB_PATH = US_DIR / "us_kr_history.db"
CONFIG_PATH = Path("config/settings.yaml")


def _load_settings() -> dict:
    """settings.yaml 로드. 실패 시 빈 dict."""
    try:
        import yaml
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

# ── 섹터 Kill 설정 (US ETF 기준) ──
# kill_col: parquet의 ret_1d 컬럼 (분수 형태, 0.01=1%)
# kill_threshold: 이 이하면 해당 한국 섹터 KILL
SECTOR_KILL_CONFIG = {
    "반도체":   {"kill_col": "soxx_ret_1d", "threshold": -0.03, "sensitivity": 0.95},
    "전자부품": {"kill_col": "soxx_ret_1d", "threshold": -0.03, "sensitivity": 0.95},
    "디스플레이": {"kill_col": "soxx_ret_1d", "threshold": -0.03, "sensitivity": 0.95},
    "IT":       {"kill_col": "qqq_ret_1d",  "threshold": -0.035, "sensitivity": 0.70},
    "소프트웨어": {"kill_col": "qqq_ret_1d",  "threshold": -0.035, "sensitivity": 0.70},
    "에너지":   {"kill_col": "uso_ret_1d",  "threshold": -0.04, "sensitivity": 0.85},
    "정유":     {"kill_col": "uso_ret_1d",  "threshold": -0.04, "sensitivity": 0.85},
    "화학":     {"kill_col": "uso_ret_1d",  "threshold": -0.04, "sensitivity": 0.80},
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

    # R7: 원유 급락 -5%+ → 에너지/정유 KILL
    uso_ret = _ret("uso_ret_1d")
    if uso_ret <= -0.05:
        triggered.append({
            "name": "OIL_CRASH",
            "desc": f"WTI(USO) {uso_ret*100:+.1f}% 급락 -> 에너지/정유 KILL",
            "sector_kill": ["에너지", "정유", "화학"],
        })

    # R8: 금 급등 +3%+ → risk-off 신호 → 전체 포지션 경계
    gld_ret = _ret("gld_ret_1d")
    if gld_ret >= 0.03:
        triggered.append({
            "name": "GOLD_SPIKE",
            "desc": f"금(GLD) {gld_ret*100:+.1f}% 급등 -> risk-off 경계",
            "global_position_cap": 0.8,
        })

    # R9: 구리 급락 -5%+ → 경기 둔화 신호 → 산업재 KILL
    copx_ret = _ret("copx_ret_1d")
    if copx_ret <= -0.05:
        triggered.append({
            "name": "COPPER_CRASH",
            "desc": f"구리(COPX) {copx_ret*100:+.1f}% 급락 -> 경기 둔화 경고",
            "sector_kill": ["조선", "기계", "건설", "자동차"],
            "global_position_cap": 0.8,
        })

    return triggered


def _classify_shock_type(latest, prev, special_rules: list, nightwatch: dict) -> dict:
    """충격 유형 분류 — CORTEX 아이디어 차용.

    기존 데이터(special_rules, commodities, NIGHTWATCH)를 재활용하여
    당일 충격 유형을 태깅한다. 나중에 BEAR/CRISIS 레짐에서 낙폭 게이트
    구현 시 축적된 shock_type 데이터를 활용.

    분류:
        GEOPOLITICAL — 유가 급등 + 금 상승 + VIX 스파이크
        RATE         — 채권 금리 급변 (NIGHTWATCH L1 기반)
        LIQUIDITY    — VIX 스파이크 + 구리 급락 + 다수 섹터 동시 하락
        EARNINGS     — SOXX 급락 but SPY 상대적 방어 (섹터 특정)
        COMPOUND     — 2개 이상 동시 감지
        NONE         — 특이 충격 없음
    """
    def _ret(col):
        return float(latest.get(col, 0) or 0)

    def _close_chg(col):
        cur = float(latest.get(col, 0) or 0)
        prv = float(prev.get(col, 0) or 0)
        if prv > 0 and cur > 0:
            return (cur - prv) / prv
        return 0.0

    scores = {"GEOPOLITICAL": 0, "RATE": 0, "LIQUIDITY": 0, "EARNINGS": 0}
    signals = {}

    uso_ret = _ret("uso_ret_1d")
    gld_ret = _ret("gld_ret_1d")
    spy_ret = _ret("spy_ret_1d")
    qqq_ret = _ret("qqq_ret_1d")
    soxx_ret = _ret("soxx_ret_1d")
    vix_chg = _close_chg("vix_close")
    vix_level = float(latest.get("vix_close", 20) or 20)
    copx_ret = _ret("copx_ret_1d")

    # ── 지정학 시그널 ──
    if uso_ret >= 0.05:  # 유가 +5%+
        scores["GEOPOLITICAL"] += 35
        signals["oil_spike"] = round(uso_ret * 100, 2)
    if gld_ret >= 0.015:  # 금 +1.5%+ (risk-off)
        scores["GEOPOLITICAL"] += 15
        signals["gold_risk_off"] = round(gld_ret * 100, 2)
    if vix_chg >= 0.15 and vix_level >= 25:  # VIX 급등 + 높은 수준
        scores["GEOPOLITICAL"] += 20
        signals["vix_fear"] = round(vix_level, 1)

    # ── 금리 시그널 (NIGHTWATCH L1 기반) ──
    nw_layers = nightwatch.get("layers", {}) if isinstance(nightwatch, dict) else {}
    l1 = nw_layers.get("L1_bond_vigilante", {})
    tnx_bp = abs(float(l1.get("tnx_change_bp", 0) or 0))
    cross_regime = l1.get("cross_regime", "")

    if tnx_bp >= 0.10:  # 10bp+ 금리 변동
        scores["RATE"] += 40
        signals["tnx_spike_bp"] = round(tnx_bp * 100, 1)
    if cross_regime in ("DIVERGENCE", "CORRECTION"):
        scores["RATE"] += 20
        signals["bond_cross_regime"] = cross_regime
    if nightwatch.get("bond_vigilante_veto"):
        scores["RATE"] += 25
        signals["bond_vigilante_veto"] = True

    # ── 유동성 시그널 ──
    if vix_chg >= 0.20:  # VIX +20%+
        scores["LIQUIDITY"] += 30
        signals["vix_spike_pct"] = round(vix_chg * 100, 1)
    if copx_ret <= -0.03:  # 구리 -3%+ (경기 둔화)
        scores["LIQUIDITY"] += 25
        signals["copper_crash"] = round(copx_ret * 100, 2)
    # 다수 특수 룰 발동 = 유동성 경색 징후
    rule_names = {r.get("name") for r in special_rules}
    if len(rule_names & {"VIX_SPIKE", "MARKET_CRASH", "COPPER_CRASH"}) >= 2:
        scores["LIQUIDITY"] += 20

    # ── 실적 시그널 (섹터 특정 급락) ──
    if soxx_ret <= -0.04 and spy_ret > -0.02:  # SOXX 급락 but SPY 방어
        scores["EARNINGS"] += 45
        signals["soxx_isolated_crash"] = round(soxx_ret * 100, 2)
    elif soxx_ret <= -0.05:  # SOXX 폭락 (시장 전체와 무관하게)
        scores["EARNINGS"] += 30
        signals["soxx_crash"] = round(soxx_ret * 100, 2)

    # ── 최종 판정 ──
    max_type = max(scores, key=scores.get)
    max_score = scores[max_type]

    high_scores = [k for k, v in scores.items() if v >= 40]

    if max_score < 25:
        shock_type = "NONE"
        confidence = 0.0
        description = "특이 충격 없음"
    elif len(high_scores) >= 2:
        shock_type = "COMPOUND"
        confidence = min(max_score / 100, 0.95)
        types_str = "+".join(high_scores)
        description = f"복합 충격 ({types_str})"
    else:
        shock_type = max_type
        confidence = min(max_score / 100, 0.95)
        desc_map = {
            "GEOPOLITICAL": "지정학 충격 (유가/VIX/금)",
            "RATE": "금리 충격 (채권/금리 급변)",
            "LIQUIDITY": "유동성 충격 (VIX/구리/다중 경고)",
            "EARNINGS": "실적/섹터 충격 (반도체 등 특정 섹터)",
        }
        description = desc_map.get(shock_type, shock_type)

    return {
        "shock_type": shock_type,
        "confidence": round(confidence, 2),
        "description": description,
        "scores": {k: v for k, v in scores.items() if v > 0},
        "signals": signals,
    }


def _run_level2_pattern(df: pd.DataFrame, latest, prev) -> dict:
    """Level 2 패턴매칭 실행. DB 없으면 스킵."""
    try:
        import importlib.util as _ilu
        _bf_path = Path(__file__).resolve().parent / "archive" / "backfill" / "backfill_us_kr_history.py"
        _sp = _ilu.spec_from_file_location("backfill_us_kr_history", _bf_path)
        _bm = _ilu.module_from_spec(_sp)
        _sp.loader.exec_module(_bm)
        PatternMatcher = _bm.PatternMatcher
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
    from scripts.us_data_backfill import TICKERS, _calc_derived

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
            prefix = ticker.replace("^", "").replace("=", "").replace("-", "").lower()
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


def _compute_sector_momentum(df: pd.DataFrame) -> dict:
    """US ETF 연속 상승/하락 감지 → KR 섹터 부스트 데이터 생성.

    실증: SOXX 3일 연속 상승 → KR 반도체 3일 전 사전진입: 승률 63.7%, +2.73%
    이를 전체 6개 US ETF → 20개 KR 섹터로 확장.

    Returns:
        {"반도체": {"us_etf": "SOXX", "consecutive_up": 3, "boost": 3.0, ...}, ...}
    """
    result = {}

    for prefix in ["xlk", "xlf", "xle", "xli", "xlv", "soxx", "gld", "uso", "copx"]:
        close_col = f"{prefix}_close"
        if close_col not in df.columns:
            continue

        closes = df[close_col].dropna().tail(10)
        if len(closes) < 3:
            continue

        # 연속 상승일 계산
        consecutive_up = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes.iloc[i] > closes.iloc[i - 1]:
                consecutive_up += 1
            else:
                break

        # 연속 하락일 계산
        consecutive_down = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes.iloc[i] < closes.iloc[i - 1]:
                consecutive_down += 1
            else:
                break

        # 5일 수익률
        ret_5d = 0.0
        if len(closes) >= 6:
            c_last = float(closes.iloc[-1])
            c_prev = float(closes.iloc[-6])
            if c_prev > 0:
                ret_5d = (c_last / c_prev - 1) * 100

        # 1일 수익률
        ret_1d = 0.0
        if len(closes) >= 2:
            c_last = float(closes.iloc[-1])
            c_prev = float(closes.iloc[-2])
            if c_prev > 0:
                ret_1d = (c_last / c_prev - 1) * 100

        # 해당 ETF에 연결된 KR 섹터에 부스트 적용
        kr_sectors = US_KR_SECTOR_MAP.get(prefix, [])
        for sector in kr_sectors:
            # 부스트 계산
            boost = 0.0
            if consecutive_up >= 4:
                boost = 5.0
            elif consecutive_up >= 3:
                boost = 3.0
            elif consecutive_up >= 2:
                boost = 1.0

            # 하락 페널티
            if consecutive_down >= 3:
                boost = min(boost, -3.0)
            elif consecutive_down >= 2:
                boost = min(boost, -1.0)

            entry = {
                "us_etf": prefix.upper(),
                "consecutive_up": int(consecutive_up),
                "consecutive_down": int(consecutive_down),
                "ret_1d_pct": round(ret_1d, 2),
                "ret_5d_pct": round(ret_5d, 2),
                "boost": boost,
            }

            # 같은 섹터가 여러 ETF에서 매핑될 수 있음 → 최대 부스트 채택
            if sector in result:
                if abs(boost) > abs(result[sector]["boost"]):
                    result[sector] = entry
            else:
                result[sector] = entry

    return result


NIGHTWATCH_BLIND_DIR = Path("data/us_market/nightwatch_blind")


def _save_nightwatch_blind_log(signal: dict, score: float, ensemble_score: float):
    """NIGHTWATCH 블라인드 테스트 — 기존 vs 앙상블 일별 비교 기록."""
    settings = _load_settings()
    if not settings.get("nightwatch", {}).get("blind_test", {}).get("enabled", False):
        return

    NIGHTWATCH_BLIND_DIR.mkdir(parents=True, exist_ok=True)
    today = signal.get("us_close_date", datetime.now().strftime("%Y-%m-%d"))

    # 기존 로직 등급 (NIGHTWATCH 없이)
    l1_100 = round(score * 100, 1)
    l2_adj = signal.get("l2_pattern", {}).get("pattern_adjustment", 0)
    existing_combined = max(-100.0, min(100.0, l1_100 + l2_adj))
    if existing_combined >= 50:
        existing_grade = "STRONG_BULL"
    elif existing_combined >= 20:
        existing_grade = "MILD_BULL"
    elif existing_combined > -20:
        existing_grade = "NEUTRAL"
    elif existing_combined > -50:
        existing_grade = "MILD_BEAR"
    else:
        existing_grade = "STRONG_BEAR"

    nw = signal.get("nightwatch", {})
    log_entry = {
        "date": today,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # 비교 핵심
        "existing_grade": existing_grade,
        "existing_combined_100": round(existing_combined, 1),
        "ensemble_grade": signal.get("grade", "?"),
        "ensemble_combined_100": signal.get("combined_score_100", 0),
        # NIGHTWATCH 상세
        "nightwatch_score": nw.get("nightwatch_score"),
        "ensemble_score": signal.get("ensemble_score"),
        "bond_vigilante_veto": nw.get("bond_vigilante_veto", False),
        # 레이어별
        "l0_score": nw.get("layers", {}).get("L0_leading", {}).get("score"),
        "l1_score": nw.get("layers", {}).get("L1_bond_vigilante", {}).get("score"),
        "l1_cross_regime": nw.get("layers", {}).get("L1_bond_vigilante", {}).get("cross_regime"),
        "l4_score": nw.get("layers", {}).get("L4_fx_triangle", {}).get("score"),
        # 시장 컨텍스트
        "spy_ret_1d": nw.get("layers", {}).get("L1_bond_vigilante", {}).get("spy_ret_1d"),
        "tnx_change_bp": nw.get("layers", {}).get("L1_bond_vigilante", {}).get("tnx_change_bp"),
        "special_rules_count": len(signal.get("special_rules", [])),
    }

    log_path = NIGHTWATCH_BLIND_DIR / f"{today}.json"
    log_path.write_text(
        json.dumps(log_entry, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # _index.json 업데이트
    index_path = NIGHTWATCH_BLIND_DIR / "_index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index_data = {"start_date": today, "logs": []}
    if today not in index_data["logs"]:
        index_data["logs"].append(today)
    index_data["total_days"] = len(index_data["logs"])
    index_data["last_updated"] = today
    index_path.write_text(
        json.dumps(index_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"NIGHTWATCH 블라인드 로그: {log_path}")


def _send_bond_vigilante_alert(signal: dict, nw_result: dict):
    """채권 자경단 비토 발동 시 별도 긴급 텔레그램 발송."""
    settings = _load_settings()
    if not settings.get("nightwatch", {}).get("bond_vigilante", {}).get("emergency_alert", True):
        return

    layers = nw_result.get("layers", {})
    l1 = layers.get("L1_bond_vigilante", {})

    spy_ret = l1.get("spy_ret_1d", "?")
    tnx_bp = l1.get("tnx_change_bp", "?")
    cross = l1.get("cross_regime", "?")
    spread_inv = l1.get("spread_inversion_veto", False)
    nw_score = nw_result.get("nightwatch_score", 0)
    ensemble = signal.get("ensemble_score", 0)
    grade = signal.get("grade", "?")

    lines = [
        "\U0001f6a8 [\ucc44\uad8c \uc790\uacbd\ub2e8 \ube44\ud1a0 \ubc1c\ub3d9]",
        "\u2501" * 20,
        f"  SPY: {spy_ret}% | 10Y: {tnx_bp}bp",
        f"  \uad50\ucc28 \ub808\uc9d0: {cross}",
    ]
    if spread_inv:
        lines.append("  \u26a0\ufe0f 10Y-30Y \uc2a4\ud504\ub808\ub4dc \uc5ed\uc804 \uac10\uc9c0")
    lines += [
        "\u2501" * 20,
        f"  NW Score: {nw_score:+.4f}",
        f"  Ensemble: {ensemble:+.4f}",
        f"  \ub4f1\uae09: {grade}",
        "\u2501" * 20,
        "  \uc870\uce58: \ud3ec\uc9c0\uc158 30% \uce21 \uc801\uc6a9",
        "  \uc8fc\uc2dd\u2193 + \uae08\ub9ac\u2191 = \ucd5c\uc545 \uc2dc\ub098\ub9ac\uc624",
        "  \uc2e0\uaddc \ub9e4\uc218 \uc790\uc81c, \uae30\uc874 \ud3ec\uc9c0\uc158 \ucd95\uc18c \uac80\ud1a0",
    ]

    try:
        from src.telegram_sender import send_message
        send_message("\n".join(lines))
        logger.info("\ucc44\uad8c \uc790\uacbd\ub2e8 \ube44\ud1a0 \uae34\uae09 \ud154\ub808\uadf8\ub7a8 \ubc1c\uc1a1 \uc644\ub8cc")
    except Exception as e:
        logger.warning(f"\ucc44\uad8c \uc790\uacbd\ub2e8 \uae34\uae09 \ud154\ub808\uadf8\ub7a8 \uc2e4\ud328: {e}")


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
    score += index_score * 0.40  # 45%→40% (원자재 5% 배분)

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

    # ─── 3. 채권/달러 (가중치: 8%) ── (10%→8%, 원자재에 2% 배분)
    bond_dollar_score = 0.0
    tlt_ret = float(latest.get("tlt_ret_1d", 0) or 0)
    # TLT 상승 = risk-off → 주식에 약세
    bond_dollar_score -= max(-0.5, min(0.5, tlt_ret * 30))

    score += bond_dollar_score * 0.08

    # ─── 3.5 원자재 신호 (가중치: 7%) ───
    commodity_data = {}
    commodity_score = 0.0

    # 금(GLD): 상승 = risk-off → 주식 약세 신호 (역상관)
    gld_ret = float(latest.get("gld_ret_1d", 0) or 0)
    gld_ret5 = float(latest.get("gld_ret_5d", 0) or 0)
    gld_above = int(latest.get("gld_above_sma20", 0) or 0)
    if gld_ret != 0:
        # 금 상승 = risk-off = 주식 약세 → 역방향
        gold_signal = -max(-0.5, min(0.5, gld_ret * 30))
        commodity_score += gold_signal * 0.30  # 금 30%
        commodity_data["gold"] = {
            "ret_1d": round(gld_ret * 100, 2),
            "ret_5d": round(gld_ret5 * 100, 2),
            "above_sma20": bool(gld_above),
            "signal": "risk_off" if gld_ret > 0.005 else ("risk_on" if gld_ret < -0.005 else "neutral"),
        }

    # 원유(USO): WTI → 정유 호재 / 화학 부담 → 중립적
    uso_ret = float(latest.get("uso_ret_1d", 0) or 0)
    uso_ret5 = float(latest.get("uso_ret_5d", 0) or 0)
    uso_above = int(latest.get("uso_above_sma20", 0) or 0)
    if uso_ret != 0:
        # 원유 급락 = 수요 둔화 경고
        oil_signal = max(-0.5, min(0.5, uso_ret * 20))
        commodity_score += oil_signal * 0.25  # 원유 25%
        commodity_data["oil"] = {
            "ret_1d": round(uso_ret * 100, 2),
            "ret_5d": round(uso_ret5 * 100, 2),
            "above_sma20": bool(uso_above),
            "signal": "demand_up" if uso_ret > 0.01 else ("demand_down" if uso_ret < -0.01 else "neutral"),
        }

    # 구리(COPX): "닥터 구리" = 경기 선행지표 → 동행
    copx_ret = float(latest.get("copx_ret_1d", 0) or 0)
    copx_ret5 = float(latest.get("copx_ret_5d", 0) or 0)
    copx_above = int(latest.get("copx_above_sma20", 0) or 0)
    if copx_ret != 0:
        copper_signal = max(-0.5, min(0.5, copx_ret * 25))
        commodity_score += copper_signal * 0.30  # 구리 30%
        commodity_data["copper"] = {
            "ret_1d": round(copx_ret * 100, 2),
            "ret_5d": round(copx_ret5 * 100, 2),
            "above_sma20": bool(copx_above),
            "signal": "expansion" if copx_ret > 0.01 else ("contraction" if copx_ret < -0.01 else "neutral"),
        }

    # 은(SLV): 금과 동반 + 산업재 수요
    slv_ret = float(latest.get("slv_ret_1d", 0) or 0)
    slv_ret5 = float(latest.get("slv_ret_5d", 0) or 0)
    if slv_ret != 0:
        commodity_data["silver"] = {
            "ret_1d": round(slv_ret * 100, 2),
            "ret_5d": round(slv_ret5 * 100, 2),
            "signal": "up" if slv_ret > 0.005 else ("down" if slv_ret < -0.005 else "neutral"),
        }
        commodity_score += max(-0.3, min(0.3, slv_ret * 15)) * 0.15  # 은 15%

    # 구리/금 비율 (경기 확장/수축 판단)
    cg_ratio = float(latest.get("copper_gold_ratio", 0) or 0)
    cg_sma = float(latest.get("copper_gold_ratio_sma20", 0) or 0)
    if cg_ratio > 0 and cg_sma > 0:
        cg_above = cg_ratio > cg_sma
        commodity_data["copper_gold_ratio"] = {
            "ratio": round(cg_ratio, 4),
            "sma20": round(cg_sma, 4),
            "regime": "expansion" if cg_above else "contraction",
        }

    signal["commodities"] = commodity_data
    score += commodity_score * 0.07

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

    # ─── 8. NIGHTWATCH 앙상블 ───
    ensemble_score = score  # 기본값: 기존 L1 score
    nw_result = None

    settings = _load_settings()
    nw_enabled = settings.get("nightwatch", {}).get("enabled", False)

    if nw_enabled:
        try:
            from src.nightwatch.engine import NightwatchEngine

            nw_engine = NightwatchEngine(settings)
            nw_result = nw_engine.compute(df, latest, prev)
            signal["nightwatch"] = nw_result

            ens_w = settings.get("nightwatch", {}).get("ensemble_weights", {})
            w_existing = ens_w.get("existing", 0.70)
            w_nw = ens_w.get("nightwatch", 0.30)
            ensemble_score = score * w_existing + nw_result["nightwatch_score"] * w_nw

            # 채권 자경단 비토
            if nw_result.get("bond_vigilante_veto"):
                veto_cap = settings.get("nightwatch", {}).get(
                    "bond_vigilante", {}
                ).get("veto_score_cap", -0.50)
                ensemble_score = min(ensemble_score, veto_cap)

                pos_cap = settings.get("nightwatch", {}).get(
                    "bond_vigilante", {}
                ).get("position_cap", 0.30)
                signal["special_rules"].append({
                    "name": "BOND_VIGILANTE",
                    "desc": "채권 자경단 발동 → 포지션 축소",
                    "global_position_cap": pos_cap,
                })
                logger.warning("NIGHTWATCH: 채권 자경단 비토 발동!")

                # 긴급 텔레그램 별도 발송 (--send 무관, 비토는 항상)
                _send_bond_vigilante_alert(signal, nw_result)

            signal["ensemble_score"] = round(ensemble_score, 4)
            logger.info(
                f"NIGHTWATCH: nw={nw_result['nightwatch_score']:+.4f} "
                f"ensemble={ensemble_score:+.4f} "
                f"veto={nw_result.get('bond_vigilante_veto', False)}"
            )
        except Exception as e:
            logger.warning(f"NIGHTWATCH 실패, 기존 로직만 사용: {e}")
            signal["nightwatch"] = {"error": str(e)}
    else:
        signal["nightwatch"] = {"enabled": False}

    # ─── 8.5. 충격 유형 분류 (CORTEX 차용) ───
    shock = _classify_shock_type(
        latest, prev,
        signal.get("special_rules", []),
        signal.get("nightwatch", {}),
    )
    signal["shock_type"] = shock
    if shock["shock_type"] != "NONE":
        logger.info(
            f"⚡ 충격 분류: {shock['shock_type']} "
            f"(신뢰도 {shock['confidence']:.0%}) — {shock['description']}"
        )

    # ─── 등급 판정 (ensemble 기반) ───
    l1_100 = round(score * 100, 1)
    ensemble_100 = round(ensemble_score * 100, 1)
    l2_adj = l2.get("pattern_adjustment", 0)
    combined_100 = max(-100.0, min(100.0, ensemble_100 + l2_adj))

    if combined_100 >= 40:
        grade = "STRONG_BULL"
    elif combined_100 >= 10:
        grade = "MILD_BULL"
    elif combined_100 > -10:
        grade = "NEUTRAL"
    elif combined_100 > -40:
        grade = "MILD_BEAR"
    else:
        grade = "STRONG_BEAR"

    signal["grade"] = grade
    signal["l1_score_100"] = l1_100
    signal["ensemble_score_100"] = ensemble_100
    signal["combined_score_100"] = round(combined_100, 1)

    # 전략 C: US→KR 섹터 모멘텀 (전 섹터 사전 포지셔닝)
    signal["sector_momentum"] = _compute_sector_momentum(df)

    # 요약 생성
    spy_ret = index_dir.get("SPY", {}).get("ret_1d", 0)
    qqq_ret = index_dir.get("QQQ", {}).get("ret_1d", 0)
    l2_conf = l2.get("confidence", 0)
    l2_adj_str = f" L2:{l2_adj:+.1f}" if l2_conf > 0 else ""

    # 원자재 요약
    commod_parts = []
    for key, sym in [("gold", "Au"), ("oil", "Oil"), ("copper", "Cu")]:
        c = signal.get("commodities", {}).get(key, {})
        if c:
            commod_parts.append(f"{sym}{c['ret_1d']:+.1f}%")
    commod_str = f" | {' '.join(commod_parts)}" if commod_parts else ""

    # NIGHTWATCH 요약
    nw_str = ""
    if nw_result and isinstance(nw_result, dict) and "nightwatch_score" in nw_result:
        nw_s = nw_result["nightwatch_score"]
        veto_mark = " VETO!" if nw_result.get("bond_vigilante_veto") else ""
        nw_str = f" NW:{nw_s:+.2f}{veto_mark}"

    signal["summary"] = (
        f"US {grade} ({combined_100:+.1f}{l2_adj_str}{nw_str}) | "
        f"SPY {spy_ret:+.1f}% QQQ {qqq_ret:+.1f}% | "
        f"VIX {vix:.0f} ({vix_status}){commod_str}"
    )

    # 저장
    US_DIR.mkdir(parents=True, exist_ok=True)
    with open(SIGNAL_PATH, "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)

    logger.info(f"신호: {signal['summary']}")
    logger.info(f"저장: {SIGNAL_PATH}")

    # NIGHTWATCH 블라인드 테스트 로그
    _save_nightwatch_blind_log(signal, score, ensemble_score)

    return signal


def format_telegram_message(signal: dict) -> str:
    """텔레그램 메시지 포맷."""
    grade = signal.get("grade", signal.get("composite", "NEUTRAL").upper())
    combined = signal.get("combined_score_100", signal.get("score", 0) * 100)

    # 충격 유형 헤더
    shock = signal.get("shock_type", {})
    shock_label = shock.get("shock_type", "NONE") if isinstance(shock, dict) else "NONE"
    shock_emoji = {
        "GEOPOLITICAL": "\U0001f30d", "RATE": "\U0001f4c8",
        "LIQUIDITY": "\U0001f4a7", "EARNINGS": "\U0001f4ca",
        "COMPOUND": "\U0001f534", "NONE": "",
    }.get(shock_label, "")
    shock_str = f" | {shock_emoji}{shock_label}" if shock_label != "NONE" else ""

    lines = [
        f"[US Overnight] {signal.get('us_close_date', '?')}",
        f"Grade: {grade} (Score: {combined:+.1f}){shock_str}",
        "",
    ]

    # 충격 상세 (NONE이 아닐 때만)
    if shock_label != "NONE" and isinstance(shock, dict):
        conf = shock.get("confidence", 0)
        desc = shock.get("description", "")
        lines.append(f"[ 충격 ] {shock_emoji} {desc} (신뢰도 {conf:.0%})")
        sigs = shock.get("signals", {})
        if sigs:
            sig_parts = [f"{k}={v}" for k, v in sigs.items()]
            lines.append(f"  시그널: {', '.join(sig_parts)}")
        lines.append("")

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

    # 원자재
    commod = signal.get("commodities", {})
    if commod:
        lines.append("\n[ 원자재 ]")
        for key, label in [("gold", "금(GLD)"), ("silver", "은(SLV)"),
                           ("oil", "유(USO)"), ("copper", "구리(COPX)")]:
            c = commod.get(key, {})
            if c:
                sma = ">" if c.get("above_sma20") else "<"
                lines.append(
                    f"  {label}: {c['ret_1d']:+.1f}% (5D:{c['ret_5d']:+.1f}%) "
                    f"[{c['signal']}]"
                )
        cg = commod.get("copper_gold_ratio", {})
        if cg:
            lines.append(f"  구리/금 비율: {cg['ratio']:.4f} [{cg['regime']}]")

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

    # NIGHTWATCH
    nw = signal.get("nightwatch", {})
    if nw and nw.get("nightwatch_score") is not None:
        layers = nw.get("layers", {})
        l0 = layers.get("L0_leading", {})
        l1 = layers.get("L1_bond_vigilante", {})
        l2 = layers.get("L2_regime_transition", {})
        l4 = layers.get("L4_fx_triangle", {})

        veto_mark = " VETO!" if nw.get("bond_vigilante_veto") else ""
        lines.append(f"\n[ NIGHTWATCH ] (Score: {nw['nightwatch_score']:+.4f}{veto_mark})")

        # L0
        hyg_5d = l0.get("hyg_ret_5d", "N/A")
        hyg_st = l0.get("hyg_status", "?")
        vix_tr = l0.get("vix_term_ratio", "N/A")
        vix_ts = l0.get("vix_term_status", "?")
        lines.append(f"  L0 선행: HYG 5D:{hyg_5d}% [{hyg_st}] | VIX Term:{vix_tr} [{vix_ts}]")

        # L1
        tnx_bp = l1.get("tnx_change_bp", "N/A")
        cross = l1.get("cross_regime", "?")
        lines.append(f"  L1 채권자경단: 10Y {tnx_bp}bp | {cross}")

        # L2 (2D)
        if l2:
            cs_z = l2.get("credit_spread_z", "N/A")
            cs_st = l2.get("credit_status", "?")
            mv_z = l2.get("move_z", "N/A")
            mv_st = l2.get("move_status", "?")
            curve = l2.get("yield_curve_10_3m", "N/A")
            dual = " DUAL!" if l2.get("dual_alarm") else ""
            lines.append(f"  L2 레짐전환: Credit z:{cs_z} [{cs_st}] | MOVE z:{mv_z} [{mv_st}] | 10Y-3M:{curve}{dual}")

        # L4
        jpy_chg = l4.get("usdjpy_change_pct", "N/A")
        krw_chg = l4.get("usdkrw_change_pct", "N/A")
        yen_st = l4.get("yen_carry_status", "?")
        lines.append(f"  L4 환율삼각: JPY {jpy_chg}% | KRW {krw_chg}% | 엔캐리:{yen_st}")

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
