"""V2 포지션 사이저 — Half Kelly + 상관관계 감산 (STEP 5-2/5-3)

기존 PositionSizer를 래핑하여 두 가지 추가 조정:
1. 상관관계 감산: 기존 포트폴리오와 고상관(>0.7) 시 사이즈 감축
2. Half Kelly: 시그널 적중률 기반 최적 베팅 비율

final_shares = min(기존_shares × corr_penalty × kelly_mult, 기존_shares)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.position_sizer import PositionSizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PositionSizerV2:
    """V2 사이징: 기존 ATR + 상관관계 감산 + Half Kelly"""

    def __init__(self, config: dict):
        self._base_sizer = PositionSizer(config)

        v2_cfg = config.get("alpha_v2", {}).get("sizing", {})
        self._corr_threshold = v2_cfg.get("corr_threshold", 0.7)
        self._corr_penalty = v2_cfg.get("corr_penalty", 0.7)
        self._sector_max_pct = v2_cfg.get("sector_max_pct", 0.30)
        self._use_kelly = v2_cfg.get("use_kelly", False)
        self._kelly_default = v2_cfg.get("kelly_default", 0.5)

        # 적중률 데이터 로드
        self._accuracy = self._load_accuracy()

    @property
    def max_risk_pct(self):
        return self._base_sizer.max_risk_pct

    def calculate(
        self,
        account_balance: float,
        entry_price: float,
        atr_value: float,
        grade_ratio: float,
        current_portfolio_risk: float = 0.0,
        stage_pct: float = 1.0,
        vol_normalized_weight: float = 1.0,
        # V2 추가 파라미터
        corr_penalty: float = 1.0,
        signal_source: str = "",
    ) -> dict:
        """기존 사이징 + V2 조정.

        Args:
            corr_penalty: 상관관계 감산 비율 (0.7 = 30% 감축)
            signal_source: 시그널 소스명 (Half Kelly용)
        """
        # 1. 기존 사이징 먼저 계산
        result = self._base_sizer.calculate(
            account_balance=account_balance,
            entry_price=entry_price,
            atr_value=atr_value,
            grade_ratio=grade_ratio,
            current_portfolio_risk=current_portfolio_risk,
            stage_pct=stage_pct,
            vol_normalized_weight=vol_normalized_weight,
        )

        base_shares = result["shares"]
        if base_shares <= 0:
            return result

        multiplier = 1.0

        # 2. 상관관계 감산
        if corr_penalty < 1.0:
            multiplier *= corr_penalty

        # 3. Half Kelly
        if self._use_kelly and signal_source:
            kelly_mult = self._calc_half_kelly(signal_source)
            multiplier *= kelly_mult

        # 적용
        if multiplier < 1.0:
            adjusted = max(1, int(base_shares * multiplier))
            result["shares"] = adjusted
            result["investment"] = int(adjusted * entry_price)
            result["risk_amount"] = int(adjusted * result["stop_distance"])
            result["pct_of_account"] = round(
                result["investment"] / account_balance * 100, 1
            ) if account_balance > 0 else 0.0
            result["v2_multiplier"] = round(multiplier, 3)

        return result

    def calc_corr_penalty(
        self,
        candidate: str,
        portfolio_tickers: list[str],
        corr_matrix,
    ) -> float:
        """후보 종목의 포트폴리오 상관관계 감산 비율 계산.

        Args:
            candidate: 후보 종목코드
            portfolio_tickers: 현재 보유 종목 리스트
            corr_matrix: CorrelationMatrix 인스턴스

        Returns:
            0.0~1.0 감산 비율 (1.0 = 감산 없음)
        """
        if not portfolio_tickers or corr_matrix is None:
            return 1.0

        max_corr, _ = corr_matrix.get_max_corr_with_portfolio(
            candidate, portfolio_tickers
        )

        if max_corr > self._corr_threshold:
            return self._corr_penalty  # 기본 0.7 (30% 감축)

        return 1.0

    def _calc_half_kelly(self, signal_source: str) -> float:
        """시그널 소스의 Half Kelly 비율 계산.

        Kelly% = (p × b - q) / b
        Half Kelly = Kelly% / 2, clamped to [0.1, 1.0]

        Args:
            signal_source: 시그널 소스명

        Returns:
            0.1~1.0 Kelly 배수
        """
        acc = self._accuracy.get(signal_source)
        if acc is None:
            return self._kelly_default

        hit_rate = acc.get("hit_rate", 0)
        avg_ret = acc.get("avg_ret", 0)
        total = acc.get("total", 0)

        # 샘플 부족
        if total < 10 or hit_rate <= 0:
            return self._kelly_default

        p = hit_rate / 100.0
        q = 1.0 - p

        # avg_ret이 전체 평균이므로, 수익/손실 비율 추정
        # 수익 시 평균 ≈ avg_ret / p, 손실 시 평균 ≈ avg_ret * (p-1) / q
        # 간소화: b = |avg_ret_win / avg_ret_loss| ≈ 1.5 (보수적 가정)
        b = 1.5

        kelly = (p * b - q) / b
        half_kelly = kelly / 2.0

        # 0.1~1.0 범위로 클램핑
        return max(0.1, min(1.0, half_kelly))

    def _load_accuracy(self) -> dict:
        """signal_accuracy.json에서 적중률 로드."""
        path = PROJECT_ROOT / "data" / "market_learning" / "signal_accuracy.json"
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("cumulative", {})
        except Exception:
            return {}
