"""
Quantum Master v8.1 — Holding Horizon 분류기
포물선의 초점 진입 후 보유 기간을 판단한다

판단 기준:
  - OU half_life: 평균회귀 속도 (빠를수록 단기)
  - ADX: 추세 강도 (강할수록 단기 스윙)
  - 곡률 전환 강도: 변곡점 기울기 (급격할수록 단기 반등)
  - MA 배열 상태: 정배열 완성도 (완성될수록 장기 홀딩)

결과:
  SHORT  = 스윙 1~2주 (빠른 평균회귀 + 강한 추세)
  MEDIUM = 스윙 2~4주 (보통 회귀 + 중간 추세)
  LONG   = 포지션 1~3달 (느린 회귀 + MA 정배열 진행 중)
"""

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HorizonResult:
    """보유기간 판정 결과"""
    horizon: str          # "SHORT", "MEDIUM", "LONG"
    horizon_days: int     # 예상 보유일 (중앙값)
    confidence: float     # 0.0 ~ 1.0
    factors: dict         # 판단 근거


class HoldingHorizonClassifier:
    """포물선의 초점 진입 후 보유기간 분류"""

    def __init__(self, config: dict = None):
        cfg = (config or {}).get('v8_hybrid', {}).get('holding_horizon', {})
        # OU half_life 기준
        self.hl_short = cfg.get('half_life_short', 15)     # < 15일 → 단기
        self.hl_medium = cfg.get('half_life_medium', 30)    # 15~30일 → 중기
        # ADX 기준
        self.adx_strong = cfg.get('adx_strong', 25)         # > 25 → 강한 추세
        self.adx_moderate = cfg.get('adx_moderate', 18)     # 18~25 → 보통
        # 곡률 기준
        self.curv_sharp = cfg.get('curvature_sharp', 0.005)  # 급격한 전환
        # MA 정배열 점수
        self.ma_alignment_weight = cfg.get('ma_alignment_weight', 0.25)

    def classify(self, row: pd.Series) -> HorizonResult:
        """단일 종목의 보유기간을 판정합니다."""
        scores = {
            'SHORT': 0.0,
            'MEDIUM': 0.0,
            'LONG': 0.0,
        }
        factors = {}

        # ── Factor 1: OU half_life ──
        hl = row.get('half_life', row.get('ou_half_life', 30))
        if hl < self.hl_short:
            scores['SHORT'] += 0.35
            factors['half_life'] = f'{hl:.0f}일 (빠른 회귀→단기)'
        elif hl < self.hl_medium:
            scores['MEDIUM'] += 0.35
            factors['half_life'] = f'{hl:.0f}일 (보통 회귀→중기)'
        else:
            scores['LONG'] += 0.35
            factors['half_life'] = f'{hl:.0f}일 (느린 회귀→장기)'

        # ── Factor 2: ADX 추세 강도 ──
        adx = row.get('adx_14', row.get('adx', 20))
        if adx > self.adx_strong:
            scores['SHORT'] += 0.25
            factors['adx'] = f'{adx:.1f} (강한 추세→빠른 수익 실현)'
        elif adx > self.adx_moderate:
            scores['MEDIUM'] += 0.25
            factors['adx'] = f'{adx:.1f} (보통 추세→중기 홀딩)'
        else:
            scores['LONG'] += 0.25
            factors['adx'] = f'{adx:.1f} (약한 추세→천천히 회복)'

        # ── Factor 3: 곡률 전환 강도 ──
        curv = abs(row.get('ema_curvature', 0))
        curv_prev = abs(row.get('ema_curvature_prev', 0))
        curv_delta = curv - curv_prev

        if curv > self.curv_sharp:
            scores['SHORT'] += 0.20
            factors['curvature'] = f'{curv:.5f} (급격한 변곡점→단기 반등)'
        elif curv > self.curv_sharp * 0.3:
            scores['MEDIUM'] += 0.20
            factors['curvature'] = f'{curv:.5f} (완만한 변곡점→중기)'
        else:
            scores['LONG'] += 0.20
            factors['curvature'] = f'{curv:.5f} (미약한 전환→장기 바닥 다지기)'

        # ── Factor 4: MA 정배열 완성도 ──
        close = row.get('close', 0)
        sma60 = row.get('sma_60', row.get('ma60', 0))
        sma120 = row.get('sma_120', row.get('ma120', 0))

        ma_aligned = 0
        if close > sma60 > 0:
            ma_aligned += 1
        if sma60 > sma120 > 0:
            ma_aligned += 1
        if close > sma120 > 0:
            ma_aligned += 1

        w = self.ma_alignment_weight
        if ma_aligned == 3:
            # 완전 정배열 → 추세 확립, 장기 가능
            scores['LONG'] += w
            factors['ma_alignment'] = f'{ma_aligned}/3 (완전 정배열→장기 가능)'
        elif ma_aligned >= 1:
            scores['MEDIUM'] += w
            factors['ma_alignment'] = f'{ma_aligned}/3 (부분 정배열→중기)'
        else:
            scores['SHORT'] += w
            factors['ma_alignment'] = f'{ma_aligned}/3 (역배열→단기만 가능)'

        # ── 최종 판정 ──
        horizon = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[horizon] / total if total > 0 else 0.0

        horizon_days_map = {
            'SHORT': 10,    # 중앙값 10일 (7~14)
            'MEDIUM': 21,   # 중앙값 21일 (14~30)
            'LONG': 45,     # 중앙값 45일 (30~60)
        }

        return HorizonResult(
            horizon=horizon,
            horizon_days=horizon_days_map[horizon],
            confidence=round(confidence, 3),
            factors=factors,
        )

    @staticmethod
    def horizon_label(horizon: str) -> str:
        """한글 라벨 반환"""
        labels = {
            'SHORT': '단기 스윙 (1~2주)',
            'MEDIUM': '중기 스윙 (2~4주)',
            'LONG': '장기 포지션 (1~3달)',
        }
        return labels.get(horizon, horizon)
