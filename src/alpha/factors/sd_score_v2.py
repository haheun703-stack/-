"""SD Score V2 — 금액 기반 멀티타임프레임 수급 분석

기존 "연속 매수일 카운팅" → "금액 + 시총 정규화 + 5패턴 분류"로 전면 교체.

5가지 수급 패턴:
  A: 스텔스 매집 — 외국인 20일 순매수 + 가격 횡보/하락 + 거래량 평이
  B: 스마트머니 합류 — 기관 5일 순매수 + 외국인 20일 순매수
  C: 추세 확인 — 외국인 20일/60일 모두 순매수 + 가격 상승
  D: 초기 전환 — 외국인 5일 순매수 + 20일 순매도 (방향 전환 초기)
  F: 물림 — 외국인+기관 20일 순매도 + 개인 순매수 (절대 매수 금지)
  X: 데이터 부족

데이터 소스: parquet 내 기관합계/외국인합계/개인 컬럼 (원 단위, 일별)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# 결과 데이터 클래스
# ═══════════════════════════════════════════════════

@dataclass
class SDFeatures:
    """종목별 멀티타임프레임 수급 피처."""

    ticker: str

    # 외국인 누적 순매수 (억원)
    foreign_net_5d: float = 0.0
    foreign_net_20d: float = 0.0
    foreign_net_60d: float = 0.0

    # 기관 누적 순매수 (억원)
    inst_net_5d: float = 0.0
    inst_net_20d: float = 0.0
    inst_net_60d: float = 0.0

    # 개인 누적 순매수 (억원)
    individual_net_5d: float = 0.0
    individual_net_20d: float = 0.0

    # 기타법인 누적 순매수 (억원)
    etc_corp_net_20d: float = 0.0

    # 시총 대비 강도 (market_cap 있을 때만)
    foreign_intensity_20d: float = 0.0   # 양수=매집, 음수=이탈
    market_cap_bil: float = 0.0          # 시총 (억원)

    # 가격/거래량 추세
    price_change_20d: float = 0.0        # 20일 가격 변화율 (%)
    volume_ratio_5d_20d: float = 1.0     # 5일평균거래량 / 20일평균거래량

    # 패턴 분류
    pattern: str = "X"                   # A, B, C, D, F, X
    pattern_name: str = "데이터부족"
    pattern_score: int = 0               # 패턴 기본 점수
    intensity_bonus: int = 0             # 강도 보너스 점수
    sd_score: float = 0.35              # 정규화 (0.0 ~ 1.0)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "pattern": self.pattern,
            "pattern_name": self.pattern_name,
            "sd_score": round(self.sd_score, 4),
            "foreign_net_5d": round(self.foreign_net_5d, 1),
            "foreign_net_20d": round(self.foreign_net_20d, 1),
            "foreign_net_60d": round(self.foreign_net_60d, 1),
            "inst_net_5d": round(self.inst_net_5d, 1),
            "inst_net_20d": round(self.inst_net_20d, 1),
            "individual_net_5d": round(self.individual_net_5d, 1),
            "individual_net_20d": round(self.individual_net_20d, 1),
            "etc_corp_net_20d": round(self.etc_corp_net_20d, 1),
            "foreign_intensity_20d": round(self.foreign_intensity_20d, 6),
            "price_change_20d": round(self.price_change_20d, 2),
            "pattern_score": self.pattern_score,
            "intensity_bonus": self.intensity_bonus,
        }


# ═══════════════════════════════════════════════════
# 컬럼명 탐색 헬퍼
# ═══════════════════════════════════════════════════

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """DataFrame에서 첫 번째 매칭되는 컬럼명 반환."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ═══════════════════════════════════════════════════
# 핵심 피처 계산
# ═══════════════════════════════════════════════════

def compute_sd_features(
    df: pd.DataFrame,
    idx: int,
    ticker: str,
    market_cap: float = 0.0,
) -> SDFeatures:
    """parquet DataFrame에서 수급 피처를 계산한다.

    Args:
        df: 종목의 전체 DataFrame (기관합계/외국인합계/개인 컬럼 포함)
        idx: 현재 날짜 인덱스
        ticker: 종목코드
        market_cap: 시가총액 (원). 0이면 intensity 계산 스킵.

    Returns:
        SDFeatures 데이터클래스
    """
    feat = SDFeatures(ticker=ticker)

    # 컬럼명 탐색 (parquet: 한국어, CSV: 영어)
    foreign_col = _find_col(df, ["외국인합계", "foreign_net", "Foreign_Net"])
    inst_col = _find_col(df, ["기관합계", "inst_net", "Inst_Net"])
    indiv_col = _find_col(df, ["개인", "individual_net", "Individual_Net"])
    etc_corp_col = _find_col(df, ["기타법인", "etc_corp_net", "Etc_Corp_Net"])
    close_col = _find_col(df, ["close", "Close", "종가"])
    volume_col = _find_col(df, ["volume", "Volume", "거래량"])

    if foreign_col is None and inst_col is None:
        # 수급 데이터 없음
        feat.pattern = "X"
        feat.pattern_name = "데이터부족"
        feat.sd_score = 0.35  # 중립
        return feat

    # 데이터 충분성 검사 (최소 20일)
    if idx < 19:
        feat.pattern = "X"
        feat.pattern_name = "데이터부족"
        feat.sd_score = 0.35
        return feat

    # ── 누적 순매수 계산 (억원 단위) ──
    def _cumsum(col: Optional[str], window: int) -> float:
        if col is None:
            return 0.0
        start = max(0, idx - window + 1)
        series = df[col].iloc[start : idx + 1].fillna(0)
        return float(series.sum()) / 1e8  # 원 → 억원

    feat.foreign_net_5d = _cumsum(foreign_col, 5)
    feat.foreign_net_20d = _cumsum(foreign_col, 20)
    feat.foreign_net_60d = _cumsum(foreign_col, min(60, idx + 1))

    feat.inst_net_5d = _cumsum(inst_col, 5)
    feat.inst_net_20d = _cumsum(inst_col, 20)
    feat.inst_net_60d = _cumsum(inst_col, min(60, idx + 1))

    feat.individual_net_5d = _cumsum(indiv_col, 5)
    feat.individual_net_20d = _cumsum(indiv_col, 20)

    feat.etc_corp_net_20d = _cumsum(etc_corp_col, 20)

    # ── 시총 대비 강도 ──
    if market_cap > 0:
        mcap_bil = market_cap / 1e8  # 원 → 억원
        feat.market_cap_bil = mcap_bil
        if mcap_bil > 0:
            feat.foreign_intensity_20d = feat.foreign_net_20d / mcap_bil

    # ── 가격 추세 (20일) ──
    if close_col is not None and idx >= 20:
        close_now = df[close_col].iloc[idx]
        close_20d_ago = df[close_col].iloc[idx - 20]
        if close_20d_ago > 0:
            feat.price_change_20d = (close_now / close_20d_ago - 1) * 100

    # ── 거래량 비율 (5일/20일) ──
    if volume_col is not None and idx >= 20:
        vol_5d = df[volume_col].iloc[max(0, idx - 4) : idx + 1].mean()
        vol_20d = df[volume_col].iloc[max(0, idx - 19) : idx + 1].mean()
        if vol_20d > 0:
            feat.volume_ratio_5d_20d = vol_5d / vol_20d

    # ── 패턴 분류 ──
    feat.pattern, feat.pattern_name, feat.pattern_score = _classify_pattern(feat)

    # ── 강도 보너스 ──
    feat.intensity_bonus = _calc_intensity_bonus(feat)

    # ── 최종 정규화 점수 ──
    raw = feat.pattern_score + feat.intensity_bonus
    feat.sd_score = max(0.0, min(1.0, (raw + 30) / 90))

    return feat


# ═══════════════════════════════════════════════════
# 패턴 분류
# ═══════════════════════════════════════════════════

_PATTERN_INFO = {
    "A": ("스텔스매집", 40),
    "B": ("스마트머니합류", 30),
    "C": ("추세확인", 20),
    "D": ("초기전환", 15),
    "F": ("물림", -30),
    "X": ("데이터부족", 0),
}


def _classify_pattern(feat: SDFeatures) -> tuple[str, str, int]:
    """5가지 수급 패턴으로 분류.

    우선순위: F(위험) → A(최고) → B → C → D → X

    Returns:
        (pattern_code, pattern_name, pattern_score)
    """
    f20 = feat.foreign_net_20d
    f60 = feat.foreign_net_60d
    f5 = feat.foreign_net_5d
    i5 = feat.inst_net_5d
    i20 = feat.inst_net_20d
    ind20 = feat.individual_net_20d
    price_chg = feat.price_change_20d
    vol_ratio = feat.volume_ratio_5d_20d

    # ── F: 물림 (최우선 체크 — 스마트머니 빠지고 개인이 받는 중) ──
    # 핵심: 외국인+기관 "합산"이 마이너스면 스마트머니 이탈
    # (기관이 약간 플러스여도 외국인 대량 매도면 F)
    # 최소 기준: 스마트머니 합산 매도 ≥ 10억 AND 개인 순매수 ≥ 10억 (잡음 제거)
    smart_money_20d = f20 + i20
    if smart_money_20d < -10 and ind20 > 10:
        # 개인이 스마트머니 매도분의 30% 이상 흡수 중
        if ind20 / abs(smart_money_20d) >= 0.3:
            return "F", "물림", -30

    # ── A: 스텔스 매집 (외국인 매집 + 가격 횡보/하락 + 거래량 평이) ──
    # 최소 기준: 외국인 20일 순매수 ≥ 10억 (잡음 제거)
    if f20 > 10 and price_chg <= 5.0 and vol_ratio < 1.5:
        return "A", "스텔스매집", 40

    # ── B: 스마트머니 합류 (기관 5일 매수 + 외국인 20일 매수) ──
    if i5 > 0 and f20 > 10:
        return "B", "스마트머니합류", 30

    # ── C: 추세 확인 (외국인 20일/60일 모두 매수 + 가격 상승) ──
    if f20 > 0 and f60 > 0 and price_chg > 0:
        return "C", "추세확인", 20

    # ── D: 초기 전환 (외국인 5일 매수 + 20일은 아직 마이너스) ──
    if f5 > 0 and f20 < 0:
        return "D", "초기전환", 15

    # ── 기타: 패턴 미분류 ──
    # 외국인 20일 매수이지만 다른 패턴에 안 맞는 경우 → 약한 양수
    if f20 > 0:
        return "C", "추세확인", 20

    # 외국인 20일 매도이지만 F까지는 아닌 경우 (기관이 버티거나)
    if f20 < 0 and i20 >= 0:
        return "D", "초기전환", 15

    # 그 외: 약한 부정
    return "X", "혼조", 0


# ═══════════════════════════════════════════════════
# 강도 보너스
# ═══════════════════════════════════════════════════

def _calc_intensity_bonus(feat: SDFeatures) -> int:
    """시총 대비 수급 강도 보너스.

    시총 데이터 없으면 절대 금액 기준 사용.
    """
    bonus = 0

    if feat.market_cap_bil > 0:
        # 시총 대비 정규화
        intensity = feat.foreign_intensity_20d
        if intensity > 0.02:
            bonus += 20  # 시총의 2% 이상 매집
        elif intensity > 0.01:
            bonus += 10  # 시총의 1% 이상
        elif intensity < -0.02:
            bonus -= 20  # 시총의 2% 이상 매도
        elif intensity < -0.01:
            bonus -= 10  # 시총의 1% 이상 매도
    else:
        # 시총 없을 때: 절대 금액 기준 (억원)
        f20 = feat.foreign_net_20d
        if f20 > 500:
            bonus += 15  # 500억+ 매집
        elif f20 > 200:
            bonus += 8
        elif f20 < -500:
            bonus -= 15  # 500억+ 매도
        elif f20 < -200:
            bonus -= 8

    return bonus


# ═══════════════════════════════════════════════════
# 일괄 분석 유틸리티
# ═══════════════════════════════════════════════════

def analyze_sd_batch(
    data_dict: dict[str, pd.DataFrame],
    idx_map: dict[str, int],
    market_caps: Optional[dict[str, float]] = None,
) -> dict[str, SDFeatures]:
    """여러 종목을 일괄 분석.

    Args:
        data_dict: {ticker: DataFrame}
        idx_map: {ticker: current_date_index}
        market_caps: {ticker: market_cap_원} (선택)

    Returns:
        {ticker: SDFeatures}
    """
    results = {}
    caps = market_caps or {}
    for ticker, df in data_dict.items():
        idx = idx_map.get(ticker, len(df) - 1)
        mcap = caps.get(ticker, 0.0)
        results[ticker] = compute_sd_features(df, idx, ticker, mcap)
    return results


# ═══════════════════════════════════════════════════
# 포맷 유틸리티 (텔레그램/콘솔)
# ═══════════════════════════════════════════════════

_PATTERN_EMOJI = {
    "A": "\U0001f7e2",   # 🟢
    "B": "\U0001f535",   # 🔵
    "C": "\U0001f7e1",   # 🟡
    "D": "\U0001f7e0",   # 🟠
    "F": "\U0001f534",   # 🔴
    "X": "\u26aa",       # ⚪
}


def format_sd_summary(feat: SDFeatures) -> str:
    """콘솔/HTML용 수급 요약 한 줄."""
    emoji = _PATTERN_EMOJI.get(feat.pattern, "⚪")
    parts = [
        f"{emoji}{feat.pattern}({feat.pattern_name})",
        f"SD={feat.sd_score:.2f}",
        f"외{feat.foreign_net_20d:+,.0f}억(20d)",
        f"기{feat.inst_net_20d:+,.0f}억(20d)",
        f"개{feat.individual_net_20d:+,.0f}억(20d)",
        f"법{feat.etc_corp_net_20d:+,.0f}억(20d)",
    ]
    if feat.foreign_intensity_20d != 0:
        parts.append(f"강도={feat.foreign_intensity_20d:.4f}")
    return " | ".join(parts)
