"""Alpha Scoring v3 — 백테스트 입증 팩터 기반 스코어링 엔진.

기존 5축 100점 → 알파 입증 시그널 기반 100점 전환.

백테스트 결과 (2019~2026, 1,071종목):
  STRONG_ALPHA:
    - PULLBACK_15%+VOL3x: PF 2.58, WR 65.2%, D+5 +3.11%
    - PULLBACK_15%:       PF 2.02, WR 55.5%, D+5 +2.25%
  WEAK_ALPHA (상위):
    - BREAKOUT60+VOL3x+DUAL: PF 1.81, D+5 +2.33%
    - PULLBACK15+DUAL:    PF 1.94, WR 54.9%, D+5 +2.05%
    - VOL3x+INST:         PF 1.55, D+5 +1.53%
    - DUAL_5d:            PF 1.74, WR 52.9%, D+5 +1.37%
    - DUAL_3d_BEAR:       PF 1.62, WR 52.8%, D+5 +1.12%
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AlphaSignal:
    """개별 알파 시그널 결과."""
    name: str
    fired: bool = False
    score: float = 0.0
    detail: str = ""


@dataclass
class AlphaScoreResult:
    """종목별 알파 스코어 결과."""
    ticker: str
    name: str = ""
    total_score: float = 0.0
    grade: str = "보류"
    signals: list = field(default_factory=list)
    tier1_count: int = 0
    tier2_count: int = 0
    overheat_penalty: float = 0.0
    regime_bonus: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "total_score": round(self.total_score, 1),
            "grade": self.grade,
            "signals": [s.name for s in self.signals if s.fired],
            "tier1_count": self.tier1_count,
            "tier2_count": self.tier2_count,
            "overheat_penalty": round(self.overheat_penalty, 1),
            "regime_bonus": round(self.regime_bonus, 1),
            "signal_details": {s.name: s.detail for s in self.signals if s.fired},
        }


# ─── 기술적 지표 계산 ───

def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _calc_stoch(close: pd.Series, high: pd.Series, low: pd.Series,
                k_period: int = 14, d_period: int = 3) -> tuple:
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


# ─── 핵심: 알파 스코어 계산 ───

def calc_alpha_score(
    df: pd.DataFrame,
    ticker: str,
    name: str = "",
    regime: str = "NORMAL",
    inst_accum_days: int = 0,
    inst_accum_grade: str = "",
    fe_grade: str = "",
) -> AlphaScoreResult:
    """백테스트 입증 시그널 기반 종목 스코어 계산.

    Args:
        df: 종목 일봉 DataFrame (columns: close, high, low, volume, 기관합계, 외국인합계)
        ticker: 종목코드
        name: 종목명
        regime: 시장 레짐 (BULL/NORMAL/BEAR/CRISIS)
        inst_accum_days: 기관매집 연속일수 (외부 데이터)
        inst_accum_grade: 기관매집 등급 (STRONG/EARLY_SURGE 등)
        fe_grade: 외인소진율 등급 (FE1/FE2)

    Returns:
        AlphaScoreResult
    """
    result = AlphaScoreResult(ticker=ticker, name=name)

    if df is None or len(df) < 80:
        return result

    # 필수 컬럼 체크
    required = ["close", "high", "low", "volume"]
    for col in required:
        if col not in df.columns:
            return result

    has_inst = "기관합계" in df.columns
    has_foreign = "외국인합계" in df.columns

    # 최근 데이터
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    inst = df["기관합계"].fillna(0) if has_inst else pd.Series(0, index=df.index)
    foreign = df["외국인합계"].fillna(0) if has_foreign else pd.Series(0, index=df.index)

    # 파생 지표
    ma20 = close.rolling(20).mean()
    gap20 = (close - ma20) / ma20 * 100
    vol_ma20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_ma20.replace(0, np.nan)
    high60_max = high.rolling(60).max().shift(1)
    rsi = _calc_rsi(close)
    stoch_k, _ = _calc_stoch(close, high, low)

    # 최근일 값 — 수급 stale 방어: 마지막 날 기관+외인 모두 0이면 유효한 마지막 날 사용
    today = -1  # 마지막 날 (기술지표용)
    supply_idx = -1  # 수급 유효 날 (supply_buy/dual_buy 판정용)

    # 마지막 날 수급이 모두 0이면 stale → 유효 마지막 날 탐색 (최대 10일 전까지)
    if (has_inst or has_foreign) and inst.iloc[-1] == 0 and foreign.iloc[-1] == 0:
        for si in range(len(df) - 1, max(len(df) - 10, 0), -1):
            if inst.iloc[si] != 0 or foreign.iloc[si] != 0:
                supply_idx = si - len(df)  # 음수 인덱스
                break

    gap_today = gap20.iloc[today] if pd.notna(gap20.iloc[today]) else 0
    vol_today = vol_ratio.iloc[today] if pd.notna(vol_ratio.iloc[today]) else 0
    inst_today = inst.iloc[supply_idx]
    foreign_today = foreign.iloc[supply_idx]
    close_today = close.iloc[today]
    rsi_today = rsi.iloc[today] if pd.notna(rsi.iloc[today]) else 50
    stoch_today = stoch_k.iloc[today] if pd.notna(stoch_k.iloc[today]) else 50
    high60_today = high60_max.iloc[today] if pd.notna(high60_max.iloc[today]) else float("inf")

    supply_buy = inst_today > 0 or foreign_today > 0
    dual_buy = inst_today > 0 and foreign_today > 0

    # 연속 쌍끌이 계산 (supply_idx 기준)
    start_i = len(df) + supply_idx if supply_idx != -1 else len(df) - 1
    dual_consec = 0
    for i in range(start_i, max(start_i - 20, 0), -1):
        if inst.iloc[i] > 0 and foreign.iloc[i] > 0:
            dual_consec += 1
        else:
            break

    # ════════════════════════════════════════════════════
    # Tier 1: 핵심 알파 시그널 (최대 70점)
    # ════════════════════════════════════════════════════

    # S1: 급락15% + 거래량3배 + 수급 (PF 2.58) → 30점
    s1 = AlphaSignal("PULLBACK15_VOL3x")
    if gap_today <= -15 and vol_today >= 3.0 and supply_buy:
        s1.fired = True
        s1.score = 30
        s1.detail = f"이격{gap_today:.1f}% vol×{vol_today:.1f}"
    result.signals.append(s1)

    # S2: 급락15% + 쌍끌이 (PF 1.94) → 20점
    s2 = AlphaSignal("PULLBACK15_DUAL")
    if gap_today <= -15 and dual_buy:
        s2.fired = True
        s2.score = 20
        s2.detail = f"이격{gap_today:.1f}% 기관+외인"
    result.signals.append(s2)

    # S3: 60일 돌파 + 거래량3배 + 쌍끌이 (PF 1.81) → 20점
    s3 = AlphaSignal("BREAKOUT60_VOL3x_DUAL")
    if close_today > high60_today and vol_today >= 3.0 and dual_buy:
        s3.fired = True
        s3.score = 20
        s3.detail = f"돌파{close_today:.0f}>{high60_today:.0f} vol×{vol_today:.1f}"
    result.signals.append(s3)

    # S4: 급락10% + 수급 (PF 1.48) → 10점
    s4 = AlphaSignal("PULLBACK10_SUPPLY")
    if -15 < gap_today <= -10 and supply_buy:
        s4.fired = True
        s4.score = 10
        s4.detail = f"이격{gap_today:.1f}%"
    result.signals.append(s4)

    # S4b: 급락7% + 수급 → 6점 (빈도 높은 중간 시그널)
    s4b = AlphaSignal("PULLBACK7_SUPPLY")
    if -10 < gap_today <= -7 and supply_buy and not s4.fired:
        s4b.fired = True
        s4b.score = 6
        s4b.detail = f"이격{gap_today:.1f}%"
    result.signals.append(s4b)

    tier1_signals = [s for s in [s1, s2, s3, s4, s4b] if s.fired]
    result.tier1_count = len(tier1_signals)

    # ════════════════════════════════════════════════════
    # Tier 2: 보조 알파 시그널 (최대 45점)
    # ════════════════════════════════════════════════════

    # S5: 연속 쌍끌이 (PF 1.50~1.74) → 4~15점
    s5 = AlphaSignal("DUAL_BUY")
    if dual_consec >= 5:
        s5.fired = True
        s5.score = 15
        s5.detail = f"연속{dual_consec}일"
    elif dual_consec >= 3:
        s5.fired = True
        s5.score = 8
        s5.detail = f"연속{dual_consec}일"
    elif dual_consec >= 2:
        s5.fired = True
        s5.score = 4
        s5.detail = f"연속{dual_consec}일"
    result.signals.append(s5)

    # S5b: 당일 쌍끌이 (기관+외인 동시 순매수) → 5점
    s5b = AlphaSignal("DUAL_BUY_TODAY")
    if dual_buy and not s5.fired:
        s5b.fired = True
        s5b.score = 5
        s5b.detail = "기관+외인 동시매수"
    result.signals.append(s5b)

    # S6: 거래량 3배 + 기관매수 (PF 1.55) → 10점
    s6 = AlphaSignal("VOL_SPIKE_3x_INST")
    if vol_today >= 3.0 and inst_today > 0 and not s1.fired and not s3.fired:
        s6.fired = True
        s6.score = 10
        s6.detail = f"vol×{vol_today:.1f} 기관매수"
    result.signals.append(s6)

    # S6b: 거래량 2배 + 수급 (PF 1.38) → 5점
    s6b = AlphaSignal("VOL_SPIKE_2x_SUPPLY")
    if 2.0 <= vol_today < 3.0 and supply_buy and not s6.fired:
        s6b.fired = True
        s6b.score = 5
        s6b.detail = f"vol×{vol_today:.1f} 수급"
    result.signals.append(s6b)

    # S7: 외인 연속 매수 (PF 1.30~1.43) → 4~8점
    f_start = len(df) + supply_idx if supply_idx != -1 else len(df) - 1
    foreign_consec = 0
    for i in range(f_start, max(f_start - 20, 0), -1):
        if foreign.iloc[i] > 0:
            foreign_consec += 1
        else:
            break

    s7 = AlphaSignal("FOREIGN_ACCUM")
    if foreign_consec >= 5:
        s7.fired = True
        s7.score = 8
        s7.detail = f"외인연속{foreign_consec}일"
    elif foreign_consec >= 3:
        s7.fired = True
        s7.score = 4
        s7.detail = f"외인연속{foreign_consec}일"
    result.signals.append(s7)

    # S7b: RSI 과매도 + 수급 → 7점 (역발상 매수)
    s7b = AlphaSignal("RSI_OVERSOLD_SUPPLY")
    if rsi_today <= 30 and supply_buy:
        s7b.fired = True
        s7b.score = 7 if rsi_today <= 25 else 5
        s7b.detail = f"RSI={rsi_today:.0f} 수급매수"
    result.signals.append(s7b)

    # S8: 기관매집 조기감지 부스트 (기존 전략N 유지)
    s8 = AlphaSignal("INST_ACCUM_BOOST")
    if inst_accum_grade:
        boost_map = {"EARLY_SURGE": 10, "STRONG": 8, "EARLY_ACCEL": 7, "EARLY_DUAL": 5}
        boost = boost_map.get(inst_accum_grade, 3)
        if inst_accum_days >= 5:
            boost += 3
        elif inst_accum_days >= 3:
            boost += 2
        s8.fired = True
        s8.score = min(boost, 12)
        s8.detail = f"{inst_accum_grade} {inst_accum_days}일"
    result.signals.append(s8)

    # S9: 외인소진율 부스트 (기존 전략O 유지)
    s9 = AlphaSignal("FE_BOOST")
    if fe_grade:
        s9.fired = True
        s9.score = 6 if fe_grade == "FE2" else 3
        s9.detail = fe_grade
    result.signals.append(s9)

    tier2_signals = [s for s in [s5, s5b, s6, s6b, s7, s7b, s8, s9] if s.fired]
    result.tier2_count = len(tier2_signals)

    # ════════════════════════════════════════════════════
    # Tier 3: 과열 패널티 + 레짐 보너스
    # ════════════════════════════════════════════════════

    penalty = 0
    if rsi_today > 75:
        penalty -= 10
    elif rsi_today > 70:
        penalty -= 5

    if stoch_today > 90:
        penalty -= 8
    elif stoch_today > 80:
        penalty -= 4

    # 5일 수익률 체크
    if len(df) >= 6:
        ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
        if ret_5d > 15:
            penalty -= 8
        elif ret_5d > 10:
            penalty -= 4

    result.overheat_penalty = penalty

    # 레짐 보너스 — BEAR에서 쌍끌이는 알파가 더 높음
    regime_bonus = 0
    if regime in ("BEAR", "CRISIS") and dual_consec >= 3:
        regime_bonus = 5  # BEAR 쌍끌이 역발상 보너스
    result.regime_bonus = regime_bonus

    # ════════════════════════════════════════════════════
    # 최종 점수 계산
    # ════════════════════════════════════════════════════

    tier1_score = sum(s.score for s in tier1_signals)
    tier2_score = sum(s.score for s in tier2_signals)

    # Tier1 시그널 겹침 보너스 (여러 Tier1이 동시 발화 = 매우 강한 신호)
    overlap_bonus = 0
    if result.tier1_count >= 2:
        overlap_bonus = 10

    raw_score = tier1_score + tier2_score + overlap_bonus + penalty + regime_bonus
    result.total_score = max(0, min(100, raw_score))

    # ════════════════════════════════════════════════════
    # Grade 배정
    # ════════════════════════════════════════════════════

    # 과열 disqualify
    disqualified = (rsi_today >= 78 or stoch_today >= 90 or
                    (len(df) >= 6 and (close.iloc[-1] / close.iloc[-6] - 1) * 100 >= 15))

    if disqualified and result.total_score < 50:
        result.grade = "보류"
    elif result.total_score >= 70 and result.tier1_count >= 1:
        result.grade = "적극매수"
    elif result.total_score >= 50 and (result.tier1_count >= 1 or result.tier2_count >= 3):
        result.grade = "매수"
    elif result.total_score >= 35 and result.tier2_count >= 2:
        result.grade = "매수"
    elif result.total_score >= 25:
        result.grade = "관심매수"
    elif result.total_score >= 12:
        result.grade = "관찰"
    else:
        result.grade = "보류"

    return result


def score_universe(
    universe: dict[str, pd.DataFrame],
    names: dict[str, str] = None,
    regime: str = "NORMAL",
    inst_accum: dict = None,
    fe_data: dict = None,
) -> list[AlphaScoreResult]:
    """전체 유니버스 스코어링.

    Args:
        universe: {ticker: DataFrame} 딕셔너리
        names: {ticker: name} 종목명 매핑
        regime: 시장 레짐
        inst_accum: {ticker: {"grade": str, "days": int}} 기관매집 데이터
        fe_data: {ticker: {"grade": str}} 외인소진율 데이터

    Returns:
        AlphaScoreResult 리스트 (점수 내림차순)
    """
    names = names or {}
    inst_accum = inst_accum or {}
    fe_data = fe_data or {}

    results = []
    for ticker, df in universe.items():
        ia = inst_accum.get(ticker, {})
        fe = fe_data.get(ticker, {})

        r = calc_alpha_score(
            df=df,
            ticker=ticker,
            name=names.get(ticker, ""),
            regime=regime,
            inst_accum_days=ia.get("days", 0),
            inst_accum_grade=ia.get("grade", ""),
            fe_grade=fe.get("grade", ""),
        )

        if r.total_score > 0:
            results.append(r)

    results.sort(key=lambda x: -x.total_score)
    return results
