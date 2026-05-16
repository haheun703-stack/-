"""Strategy 코드 → 한국어 풀이 매핑 (주린이 친화).

원칙 (5/17 퐝가님 지시):
- 코드 내부 식별자(strategy)는 영어 유지 (개발 표준)
- 페이지/텔레그램/리포트 사용자 표시는 반드시 한국어 풀이
- 약어(WR/PF 등)도 한국어로 풀어쓰기

사용:
    from src.utils.strategy_kr import strategy_kr, wr_kr, pf_kr
    print(strategy_kr("BLUECHIP_A_쌍끌이"))  # "우량주 쌍끌이 (외인+기관 동시 매수)"
"""

from __future__ import annotations

STRATEGY_KR: dict[str, str] = {
    # ── 청산 사유 (공통) ──
    "TRAILING_STOP": "트레일링 스톱 (고점 -2%)",
    "TAKE_PROFIT_T1": "1차 익절 (+10% 부분 매도)",
    "TAKE_PROFIT_T2": "2차 익절 (+20% 전량)",
    "TAKE_PROFIT": "익절",
    "STOP_LOSS": "손절 (-7%)",
    "MAX_HOLD": "보유 기간 만료",
    "SUPPLY_EXIT": "수급 이탈 (3일 연속)",
    "NEUTRAL_EXIT": "방향 중립 전환",
    "DIRECTION_SWITCH": "JARVIS 방향 전환",

    # ── Bluechip (우량주 TOP 30) ──
    # 5/17 동기화: 웹봇 정본 따라 "매수" → "동반/진입" 일관화
    "BLUECHIP_STOP_LOSS": "우량주 손절",
    "BLUECHIP_SUPPLY_EXIT": "우량주 수급 이탈",
    "BLUECHIP_TRAILING_STOP": "우량주 트레일링",
    "BLUECHIP_TAKE_PROFIT_T1": "우량주 1차 익절",
    "BLUECHIP_TAKE_PROFIT_T2": "우량주 2차 익절",
    "BLUECHIP_MAX_HOLD": "우량주 보유 만료",
    "BLUECHIP_A_쌍끌이": "우량주 쌍끌이 (외인+기관 동반)",
    "BLUECHIP_B_기관연기금": "우량주 기관·연기금 동반",
    "BLUECHIP_C_3주체합류": "우량주 3주체 합류",
    "BLUECHIP_D_외인폭발": "우량주 외인 폭발 진입",
    "BLUECHIP_E_연기금매집": "우량주 연기금 매집",
    "BLUECHIP_F_금투기타": "우량주 금투·기타법인",

    # ── Paper (일반 페이퍼) ──
    "SCAN": "일반 스캔 진입",
    "ALPHA": "알파 시그널 진입",
    "REBALANCE": "리밸런싱",
    "AI_BRAIN": "AI 두뇌 추천",
    "AI_LARGECAP": "AI 대형주 추천",
    "INTRADAY_LEARNED": "장중 학습 시그널",
    "PB15_BB": "15% 눌림목 + 볼린저밴드",
    "PULLBACK15_VOL3x": "눌림목 거래량 3배 폭증",
    "PULLBACK15_DUAL": "눌림목 + 양방향 동반",
    "PHASE5_STAGE3": "3단계 완성 시그널",
    "PHASE8_THEME_DUAL": "테마 양방향 동반",
    "FOREIGN_SURGE_PB": "외인 폭발 + 눌림목",
    "SILENT_GOLD_COMBO": "사일런트 골든 콤보",
    "LAGGARD_FOLLOW": "래거드 추격",

    # ── ETF 방향 (JARVIS) ──
    # 5/17 동기화: BUY 액션 → "진입" (웹봇 가드 회피 + 의미 명료)
    "ETF_LONG_BUY": "ETF 롱 진입 (KODEX 200)",
    "ETF_STRONG_LONG_BUY": "ETF 강한 롱 (KODEX 레버리지)",
    "ETF_SHORT_BUY": "ETF 인버스 진입 (KODEX 인버스)",
    "ETF_STRONG_SHORT_BUY": "ETF 강한 인버스 (200선물인버스2X)",
    "ETF_LONG_SWITCH": "ETF 롱 스위칭",
    "ETF_STRONG_LONG_SWITCH": "ETF 강한 롱 스위칭",
    "ETF_SHORT_SWITCH": "ETF 인버스 스위칭",
    "ETF_STRONG_SHORT_SWITCH": "ETF 강한 인버스 스위칭",
    "ETF_LONG_SELL": "ETF 롱 청산",
    "ETF_STOP_LOSS": "ETF 손절",
    "ETF_TAKE_PROFIT_T2": "ETF 2차 익절 (+10%)",
    "ETF_TRAILING_STOP": "ETF 트레일링 (고점 -2%)",
    "INVERSE_MAX_HOLD": "인버스 강제 청산 (D+2 만료)",
}


def strategy_kr(code: str) -> str:
    """Strategy 코드를 한국어 풀이로 변환. 없으면 코드 그대로 반환."""
    if not code:
        return "미분류"
    return STRATEGY_KR.get(code, code)


# ── 약어 한국어 풀이 ──

ABBR_KR: dict[str, str] = {
    "WR": "승률",
    "PF": "손익비",
    "MDD": "최대 손실폭",
    "PnL": "손익",
    "ROI": "수익률",
    "RSI": "상대 강도 지수",
    "MA": "이동평균",
    "EPS": "주당순이익",
    "PER": "주가수익비율",
    "GAP": "괴리율",
}


def abbr_kr(abbr: str) -> str:
    """약어 → 한국어 풀이."""
    return ABBR_KR.get(abbr, abbr)


# ── 위험 등급 → 친화적 표현 ──

RISK_LEVEL_FRIENDLY: dict[str, str] = {
    "NORMAL": "✅ 시장 정상",
    "CAUTION": "🟢 시장 주의",
    "WARNING": "🟡 시장 경고",
    "DANGER": "🟠 시장 위험",
    "CRISIS": "🔴 시장 위기",
}


def risk_level_friendly(level: str) -> str:
    """위험 등급 코드 → 사용자 친화적 표현."""
    return RISK_LEVEL_FRIENDLY.get(level, level)


# ── 손익비 등급 ──

def pf_grade(pf: float) -> tuple[str, str]:
    """손익비 → (이모지, 한국어 등급)"""
    if pf >= 1.5:
        return ("🟢", "우수")
    elif pf >= 1.0:
        return ("🟡", "보통")
    else:
        return ("🔴", "부진")
