"""
============================================
FLOWX 퀀트봇 — 미국장 매크로 필터
============================================

역할:
  미국장 데이터(us_market_daily)를 기반으로
  퀀트봇의 5~10일 스윙 매매 전략 파라미터를 결정한다.

단타봇과의 차이:
  단타봇  → 내일 갭 방향 예측 (1~3일)
  퀀트봇  → 5~10일 매크로 환경 판단 + 기관 스타일 진입

핵심 지표:
  1. 미국 3년물 금리  → 자금 비용 + 외인 수급 방향
  2. DXY 달러인덱스  → 신흥국(한국) 자금 흐름
  3. VIX 추세        → 위험 선호/회피 방향성
  4. SOXX 모멘텀     → 반도체 섹터 5일 방향
  5. 섹터 ETF 로테이션 → 글로벌 → 한국 섹터 매핑

데이터 소스: Supabase us_market_daily (정보봇 07:55 갱신)
출력: data/us_quant_macro.json + Supabase quant_us_macro
스케줄: BAT-A 후반 (08:10) 또는 BAT-M_US
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

# ── PYTHONPATH 안전장치 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("us_quant_filter")

DATA_DIR = PROJECT_ROOT / "data"

# ── Supabase 연결 (lazy) ──
_sb_client = None


def _get_supabase():
    global _sb_client
    if _sb_client is not None:
        return _sb_client
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        log.warning("SUPABASE_URL/KEY 미설정")
        return None
    try:
        from supabase import create_client
        _sb_client = create_client(url, key)
        return _sb_client
    except Exception as e:
        log.error("Supabase 연결 실패: %s", e)
        return None


# ── US→KR 스윙 섹터 매핑 ──
US_KR_SECTOR_MAP = {
    "XLK":  {"kr_sectors": ["반도체", "IT소프트웨어", "디스플레이"],    "lag_days": 1},
    "XLF":  {"kr_sectors": ["금융", "은행", "증권"],                  "lag_days": 1},
    "XLE":  {"kr_sectors": ["정유화학", "에너지"],                    "lag_days": 1},
    "XLI":  {"kr_sectors": ["조선", "기계", "방산"],                  "lag_days": 2},
    "XLY":  {"kr_sectors": ["자동차", "소비재", "엔터"],              "lag_days": 1},
    "XLV":  {"kr_sectors": ["제약바이오", "헬스케어"],                "lag_days": 2},
    "XLB":  {"kr_sectors": ["철강", "화학소재"],                      "lag_days": 2},
    "XLRE": {"kr_sectors": ["건설부동산"],                            "lag_days": 3},
    "XLC":  {"kr_sectors": ["게임", "인터넷플랫폼", "미디어"],        "lag_days": 1},
}

# ── 미국 3년물 금리 기준선 ──
YIELD_THRESHOLDS = {
    "VERY_HIGH": 5.0,
    "HIGH":      4.5,
    "NEUTRAL":   4.0,
    "LOW":       3.5,
}


# ════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════

def load_us_data(days: int = 5) -> list:
    """최근 N일 미국장 데이터 로드 (추세 판단용)."""
    sb = _get_supabase()
    if not sb:
        return []
    try:
        res = (
            sb.table("us_market_daily")
            .select("*")
            .order("date", desc=True)
            .limit(days)
            .execute()
        )
        return res.data or []
    except Exception as e:
        log.error("데이터 로드 실패: %s", e)
        return []


# ════════════════════════════════════════════════════════════
# 2. 금리 시그널 (퀀트봇 핵심)
# ════════════════════════════════════════════════════════════

def analyze_yield_signal(data_list: list) -> dict:
    """미국 3년물 금리 분석 — 레벨 + 추세 + 역전."""
    if not data_list:
        return {"signal": "UNKNOWN", "level": None, "trend": "UNKNOWN"}

    latest = data_list[0]
    y3     = latest.get("us_3y_yield")
    spread = latest.get("spread_3y_10y")

    if y3 is None:
        return {"signal": "UNKNOWN", "level": None, "trend": "UNKNOWN"}

    # 레벨 판단
    if y3 >= YIELD_THRESHOLDS["VERY_HIGH"]:
        level_signal = "VERY_HIGH"
        level_impact = "성장주 밸류에이션 붕괴 위험. 가치주/배당주 선호."
    elif y3 >= YIELD_THRESHOLDS["HIGH"]:
        level_signal = "HIGH"
        level_impact = "고금리 부담. 외국인 이탈 주의. 방어적 운영."
    elif y3 >= YIELD_THRESHOLDS["NEUTRAL"]:
        level_signal = "NEUTRAL"
        level_impact = "중립 구간. 수급 중심 종목 선별."
    else:
        level_signal = "LOW"
        level_impact = "저금리 기대. 성장주/반도체 우호적."

    # 5일 추세
    trend = "SIDEWAYS"
    if len(data_list) >= 3:
        yields_recent = [d.get("us_3y_yield") for d in data_list[:3]
                         if d.get("us_3y_yield") is not None]
        if len(yields_recent) >= 2:
            delta = yields_recent[0] - yields_recent[-1]
            if delta > 0.1:
                trend = "RISING"
            elif delta < -0.1:
                trend = "FALLING"

    inverted = spread is not None and spread < 0

    return {
        "signal":   level_signal,
        "level":    y3,
        "trend":    trend,
        "inverted": inverted,
        "spread":   spread,
        "impact":   level_impact,
    }


# ════════════════════════════════════════════════════════════
# 3. 달러/외인 수급 시그널
# ════════════════════════════════════════════════════════════

def analyze_dollar_signal(data_list: list) -> dict:
    """DXY 추세 → 외인 수급 방향 판단."""
    if not data_list:
        return {"signal": "NEUTRAL", "dxy": None}

    latest = data_list[0]
    dxy = latest.get("dxy")

    if dxy is None:
        return {"signal": "NEUTRAL", "dxy": None}

    dxy_vals = [d.get("dxy") for d in data_list if d.get("dxy") is not None]
    trend = "SIDEWAYS"
    if len(dxy_vals) >= 2:
        delta = dxy_vals[0] - dxy_vals[-1]
        if delta > 0.5:
            trend = "STRENGTHENING"
        elif delta < -0.5:
            trend = "WEAKENING"

    if dxy >= 106:
        signal = "STRONG_OUTFLOW"
    elif dxy >= 104:
        signal = "OUTFLOW_RISK"
    elif dxy <= 100 and trend == "WEAKENING":
        signal = "INFLOW_FAVORABLE"
    elif dxy <= 102:
        signal = "NEUTRAL_POSITIVE"
    else:
        signal = "NEUTRAL"

    impact_map = {
        "STRONG_OUTFLOW":   "외국인 대거 이탈 → 외인 의존 종목 피해야",
        "OUTFLOW_RISK":     "외국인 이탈 주의 → 기관 수급 위주로",
        "INFLOW_FAVORABLE": "달러 약세 → 외국인 유입 기대, 수출주 수혜",
        "NEUTRAL_POSITIVE": "달러 안정 → 외인 수급 중립~소폭 긍정",
        "NEUTRAL":          "달러 중립 → 수급 관망",
    }

    return {
        "signal": signal,
        "dxy":    dxy,
        "trend":  trend,
        "impact": impact_map.get(signal, ""),
    }


# ════════════════════════════════════════════════════════════
# 4. VIX 환경 분석
# ════════════════════════════════════════════════════════════

def analyze_vix_env(data_list: list) -> dict:
    """VIX 레벨 + 추세 → 리스크 온/오프 판단."""
    if not data_list:
        return {"env": "UNKNOWN", "vix": None}

    latest = data_list[0]
    vix = latest.get("vix")

    if vix is None:
        return {"env": "UNKNOWN", "vix": None}

    vix_vals = [d.get("vix") for d in data_list if d.get("vix") is not None]
    trend = "SIDEWAYS"
    if len(vix_vals) >= 3:
        delta = vix_vals[0] - vix_vals[-1]
        if delta > 2:
            trend = "RISING_FEAR"
        elif delta < -2:
            trend = "FALLING_FEAR"

    if vix >= 35:
        env = "PANIC"
    elif vix >= 25:
        env = "FEAR"
    elif vix >= 18:
        env = "CAUTION"
    elif vix <= 13:
        env = "COMPLACENCY"
    else:
        env = "STABLE"

    note_map = {
        "PANIC":       "VIX 35+ — 기관 강제 청산 구간. 역발상 매수 기회 가능.",
        "FEAR":        "VIX 25+ — 신규 진입 자제. 기존 포지션 모니터링.",
        "CAUTION":     "VIX 18~25 — 선별적 진입. 수급 강한 종목만.",
        "COMPLACENCY": "VIX 13- — 시장 과열 가능성. 익절 준비.",
        "STABLE":      "VIX 13~18 — 정상 구간. 기본 전략 운영.",
    }

    return {
        "env":   env,
        "vix":   vix,
        "trend": trend,
        "note":  note_map.get(env, ""),
    }


# ════════════════════════════════════════════════════════════
# 5. 섹터 로테이션 (미국 → 한국)
# ════════════════════════════════════════════════════════════

def analyze_sector_rotation(data_list: list) -> dict:
    """미국 섹터 ETF 5일 모멘텀 → 한국 연관 섹터 비중 결정."""
    if not data_list:
        return {"overweight": [], "underweight": [], "etf_momentum": {}}

    etf_momentum = {}
    for etf in US_KR_SECTOR_MAP:
        changes = []
        for d in data_list:
            sector_etf = d.get("sector_etf") or {}
            chg = sector_etf.get(etf)
            if chg is not None:
                changes.append(chg)
        if changes:
            etf_momentum[etf] = round(sum(changes), 2)

    overweight  = []
    underweight = []

    for etf, momentum in etf_momentum.items():
        kr_sectors = US_KR_SECTOR_MAP.get(etf, {}).get("kr_sectors", [])
        if momentum >= 3.0:
            overweight.extend(kr_sectors)
        elif momentum <= -3.0:
            underweight.extend(kr_sectors)

    # SOXX 반도체 별도 처리
    soxx_changes = [d.get("soxx_change") for d in data_list
                    if d.get("soxx_change") is not None]
    soxx_5d = sum(soxx_changes) if soxx_changes else 0

    if soxx_5d >= 5:
        if "반도체" not in overweight:
            overweight.insert(0, "반도체")
    elif soxx_5d <= -5:
        if "반도체" not in underweight:
            underweight.insert(0, "반도체")

    return {
        "overweight":   list(dict.fromkeys(overweight))[:5],
        "underweight":  list(dict.fromkeys(underweight))[:5],
        "etf_momentum": etf_momentum,
        "soxx_5d":      round(soxx_5d, 2),
    }


# ════════════════════════════════════════════════════════════
# 6. 종합 매크로 점수 + 전략 모드
# ════════════════════════════════════════════════════════════

def decide_strategy(yield_sig: dict, dollar_sig: dict,
                    vix_env: dict, sector_rot: dict) -> dict:
    """종합 판단 → 전략 모드 + 매크로 점수."""

    score = 50  # 기준선

    # 금리
    yield_scores = {"LOW": +15, "NEUTRAL": +5, "HIGH": -10,
                    "VERY_HIGH": -25, "UNKNOWN": 0}
    score += yield_scores.get(yield_sig["signal"], 0)
    if yield_sig["trend"] == "FALLING":
        score += 8
    elif yield_sig["trend"] == "RISING":
        score -= 8
    if yield_sig.get("inverted"):
        score -= 12

    # 달러
    dollar_scores = {"INFLOW_FAVORABLE": +15, "NEUTRAL_POSITIVE": +5,
                     "NEUTRAL": 0, "OUTFLOW_RISK": -12, "STRONG_OUTFLOW": -25}
    score += dollar_scores.get(dollar_sig["signal"], 0)

    # VIX
    vix_scores = {"STABLE": +10, "CAUTION": 0, "FEAR": -15,
                  "PANIC": -30, "COMPLACENCY": -5, "UNKNOWN": 0}
    score += vix_scores.get(vix_env["env"], 0)
    if vix_env.get("trend") == "FALLING_FEAR":
        score += 5
    elif vix_env.get("trend") == "RISING_FEAR":
        score -= 8

    # SOXX
    soxx_5d = sector_rot.get("soxx_5d", 0)
    if soxx_5d >= 5:
        score += 8
    elif soxx_5d <= -5:
        score -= 8

    score = max(0, min(100, score))

    # 전략 모드
    if score >= 75:
        mode, limit, hold = "BULL_AGGRESSIVE", 5, (5, 10)
    elif score >= 60:
        mode, limit, hold = "BULL_NORMAL", 4, (5, 8)
    elif score >= 45:
        mode, limit, hold = "NEUTRAL", 3, (3, 7)
    elif score >= 30:
        mode, limit, hold = "BEAR_DEFENSIVE", 2, (3, 5)
    else:
        mode, limit, hold = "BEAR_CASH", 1, (0, 3)

    # 진입 조건
    entry_map = {
        "BULL_AGGRESSIVE": {
            "min_grade": "A", "min_foreign_buy_days": 2,
            "min_inst_confirm": False, "stop_loss_pct": -2.5,
            "note": "공격적. A등급 이상, 빠른 선점.",
        },
        "BULL_NORMAL": {
            "min_grade": "A", "min_foreign_buy_days": 3,
            "min_inst_confirm": False, "stop_loss_pct": -2.0,
            "note": "기본. A등급, 외인 3일 확인.",
        },
        "NEUTRAL": {
            "min_grade": "A+", "min_foreign_buy_days": 3,
            "min_inst_confirm": True, "stop_loss_pct": -1.8,
            "note": "선별. A+만, 외인+기관 동반 필수.",
        },
        "BEAR_DEFENSIVE": {
            "min_grade": "A+", "min_foreign_buy_days": 5,
            "min_inst_confirm": True, "stop_loss_pct": -1.5,
            "note": "방어. 조건 최강화, 극소수만 진입.",
        },
        "BEAR_CASH": {
            "min_grade": "A+", "min_foreign_buy_days": 99,
            "min_inst_confirm": True, "stop_loss_pct": -1.0,
            "note": "현금 보유. 신규 진입 금지.",
        },
    }

    return {
        "strategy_mode":    mode,
        "macro_score":      score,
        "position_limit":   limit,
        "hold_days":        hold,
        "entry_conditions": entry_map[mode],
    }


# ════════════════════════════════════════════════════════════
# 7. 이번 주 전략 요약
# ════════════════════════════════════════════════════════════

def build_weekly_outlook(strategy: dict, yield_sig: dict,
                         dollar_sig: dict, sector_rot: dict) -> str:
    mode  = strategy["strategy_mode"]
    score = strategy["macro_score"]
    y3    = yield_sig.get("level")
    dxy   = dollar_sig.get("dxy")
    ow    = sector_rot.get("overweight", [])

    parts = []
    if y3 is not None:
        parts.append(f"3년물 {y3:.2f}%({yield_sig['signal']})")
    if dxy is not None:
        parts.append(f"DXY {dxy:.1f}({dollar_sig['signal']})")
    if ow:
        parts.append(f"주목섹터: {'/'.join(ow[:3])}")

    mode_str = {
        "BULL_AGGRESSIVE": "공격 매수",
        "BULL_NORMAL":     "기본 운영",
        "NEUTRAL":         "선별 진입",
        "BEAR_DEFENSIVE":  "방어 운영",
        "BEAR_CASH":       "현금 보유",
    }[mode]

    summary = " | ".join(parts)
    return f"[{mode_str}({score}점)] {summary}"


# ════════════════════════════════════════════════════════════
# 8. 메인 실행
# ════════════════════════════════════════════════════════════

def run() -> dict:
    """퀀트봇 미국장 매크로 필터 메인."""
    log.info("=== US Quant Filter 시작 ===")

    data_list = load_us_data(days=5)
    if not data_list:
        log.error("us_market_daily 데이터 없음")
        return {}

    latest = data_list[0]
    log.info("기준일: %s", latest.get("date"))

    yield_sig  = analyze_yield_signal(data_list)
    dollar_sig = analyze_dollar_signal(data_list)
    vix_env    = analyze_vix_env(data_list)
    sector_rot = analyze_sector_rotation(data_list)
    strategy   = decide_strategy(yield_sig, dollar_sig, vix_env, sector_rot)
    outlook    = build_weekly_outlook(strategy, yield_sig, dollar_sig, sector_rot)

    report = {
        "date":               latest.get("date", date.today().isoformat()),
        "updated_at":         datetime.now().strftime("%Y-%m-%d %H:%M"),
        **strategy,
        "sector_overweight":  sector_rot["overweight"],
        "sector_underweight": sector_rot["underweight"],
        "etf_momentum":       sector_rot["etf_momentum"],
        "soxx_5d":            sector_rot.get("soxx_5d"),
        "yield_signal":       yield_sig["signal"],
        "yield_level":        yield_sig.get("level"),
        "yield_trend":        yield_sig.get("trend"),
        "yield_inverted":     yield_sig.get("inverted"),
        "yield_impact":       yield_sig.get("impact"),
        "dollar_signal":      dollar_sig["signal"],
        "dxy":                dollar_sig.get("dxy"),
        "dollar_impact":      dollar_sig.get("impact"),
        "vix_env":            vix_env["env"],
        "vix":                vix_env.get("vix"),
        "vix_note":           vix_env.get("note"),
        "weekly_outlook":     outlook,
        "risk_flags":         latest.get("risk_flags") or [],
    }

    # 로컬 JSON 저장 (디버깅용)
    out_path = DATA_DIR / "us_quant_macro.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    log.info("저장: %s", out_path)

    # 콘솔 출력
    print(f"\n{'='*55}")
    print(f"  퀀트봇 미국장 매크로 분석")
    print(f"{'='*55}")
    print(f"  전략 모드:  {report['strategy_mode']}")
    print(f"  매크로 점수: {report['macro_score']}/100")
    print(f"  최대 슬롯:  {report['position_limit']}개")
    print(f"  보유 기간:  {report['hold_days'][0]}~{report['hold_days'][1]}일")
    print(f"  3년물 금리: {report.get('yield_level')} ({report['yield_signal']}, {report['yield_trend']})")
    print(f"  DXY:       {report.get('dxy')} ({report['dollar_signal']})")
    print(f"  VIX:       {report.get('vix')} ({report['vix_env']})")
    print(f"  주목 섹터:  {report['sector_overweight']}")
    print(f"  회피 섹터:  {report['sector_underweight']}")
    print(f"  전략 요약:  {outlook}")
    print(f"{'='*55}\n")

    return report


if __name__ == "__main__":
    run()
