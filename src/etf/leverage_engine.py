"""
축2: 레버리지/인버스 ETF 엔진 (포병 → 저격 포병)
========================================
레짐 + US Overnight + 5축 스코어 → 방향성 베팅

BULL + 섹터 모멘텀 1위 반도체 → 반도체 레버리지 (정밀 타격)
BULL + 그 외 섹터 1위          → KODEX 레버리지 (지수 기본값)
BEAR                           → KODEX 인버스
CRISIS                         → KODEX 200선물인버스2X
CAUTION                        → 진입 금지
"""

from datetime import datetime
from dataclasses import dataclass

from src.etf.config import LEVERAGE_ETF, SECTOR_LEVERAGE_ETF, load_settings


@dataclass
class LeverageDecision:
    """레버리지 판단 결과."""
    signal: str             # BUY / SELL / HOLD / NO_ENTRY / EMERGENCY_SELL
    etf_code: str = ""
    etf_name: str = ""
    multiplier: float = 0
    confidence: float = 0   # 0~100
    reason: str = ""
    risk_note: str = ""


@dataclass
class LeveragePosition:
    """레버리지 보유 포지션."""
    code: str
    name: str
    multiplier: float
    entry_price: float
    entry_date: str
    current_price: float = 0.0
    pnl_pct: float = 0.0
    hold_days: int = 0


class LeverageEngine:
    """레버리지/인버스 ETF 판단 엔진."""

    def __init__(self, settings: dict = None):
        self.settings = settings or load_settings()
        self.cfg = self.settings.get("leverage", {})
        self.current_position: LeveragePosition | None = None

        self.stop_loss_pct = self.cfg.get("stop_loss_pct", -3.0)
        self.max_hold_days = self.cfg.get("max_hold_days", 5)
        self.min_5axis_score = self.cfg.get("min_5axis_score", 70)
        self.us_overnight_required = self.cfg.get("us_overnight_required", True)

    def run(
        self,
        regime: str,
        us_overnight: dict,
        five_axis_score: float = 0,
        previous_regime: str = None,
        momentum_data: dict = None,
    ) -> dict:
        """
        레버리지 엔진 메인 실행.

        Args:
            regime: KOSPI 레짐 (BULL/CAUTION/BEAR/CRISIS)
            us_overnight: {"grade": 1~5, "signal": str}
            five_axis_score: 레버리지 5축 스코어 (0~100)
            previous_regime: 이전 레짐
            momentum_data: 섹터 모멘텀 {sector: {"rank": int, ...}}
                           — BULL 레짐일 때 섹터 레버리지 선택에 사용
        """
        regime = regime.upper()

        # 레짐 급변 감지 (BULL → CRISIS)
        if previous_regime and previous_regime.upper() == "BULL" and regime == "CRISIS":
            return self._emergency_close(f"레짐 급변 감지: {previous_regime} → {regime}")

        # 기존 포지션 점검
        pos_action = self._check_position(regime)
        if pos_action["action"] == "CLOSE":
            return {
                "decision": self._to_dict(LeverageDecision(
                    signal="SELL",
                    etf_code=self.current_position.code if self.current_position else "",
                    etf_name=self.current_position.name if self.current_position else "",
                    reason=pos_action["reason"],
                )),
                "position_action": "CLOSE",
                "summary": f"[레버리지] 청산: {pos_action['reason']}",
                "timestamp": datetime.now().isoformat(),
            }

        # 신규 진입 판단
        decision = self._evaluate_entry(regime, us_overnight, five_axis_score, momentum_data)
        summary = self._build_summary(decision, regime)

        return {
            "decision": self._to_dict(decision),
            "position_action": decision.signal,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

    def _evaluate_entry(
        self, regime: str, us_overnight: dict, score: float,
        momentum_data: dict = None,
    ) -> LeverageDecision:
        """신규 진입 여부 판단."""
        if regime == "CAUTION":
            return LeverageDecision(signal="NO_ENTRY", reason="CAUTION 레짐 - 레버리지 진입 금지")

        if self.current_position:
            return LeverageDecision(
                signal="HOLD",
                etf_code=self.current_position.code,
                etf_name=self.current_position.name,
                multiplier=self.current_position.multiplier,
                reason="기존 포지션 유지",
            )

        target_etf = LEVERAGE_ETF.get(regime)
        if not target_etf:
            return LeverageDecision(signal="NO_ENTRY", reason=f"{regime} 레짐에 매칭 ETF 없음")

        # US Overnight 체크 (BULL일 때만)
        if regime == "BULL" and self.us_overnight_required:
            us_grade = us_overnight.get("grade", 3)
            if us_grade > 3:
                return LeverageDecision(
                    signal="NO_ENTRY",
                    reason=f"US Overnight 부정적 ({us_grade}등급) - BULL 레버리지 보류",
                )

        # 5축 스코어 체크
        if score < self.min_5axis_score:
            return LeverageDecision(
                signal="NO_ENTRY",
                reason=f"5축 스코어 미달 ({score:.0f} < {self.min_5axis_score})",
            )

        # 섹터 레버리지 업그레이드: BULL + 모멘텀 1위 섹터에 레버리지 ETF 있으면 정밀 타격
        sector_lev_enabled = self.cfg.get("sector_leverage_enabled", True)
        if regime == "BULL" and sector_lev_enabled and momentum_data:
            sector_etf = self._select_sector_leverage(momentum_data)
            if sector_etf:
                target_etf = sector_etf

        confidence = self._confidence(regime, us_overnight, score)
        return LeverageDecision(
            signal="BUY",
            etf_code=target_etf["code"],
            etf_name=target_etf["name"],
            multiplier=target_etf["multiplier"],
            confidence=confidence,
            reason=self._entry_reason(regime, us_overnight, score, target_etf),
            risk_note=self._risk_note(target_etf),
        )

    def _select_sector_leverage(self, momentum_data: dict) -> dict | None:
        """모멘텀 1위 섹터에 유동성 충분한 레버리지 ETF가 있으면 반환.

        조건:
          1. 섹터 모멘텀 rank == 1
          2. SECTOR_LEVERAGE_ETF에 해당 섹터 등록됨 (유동성 검증 완료)

        Returns:
            섹터 레버리지 ETF dict 또는 None (기본값 KODEX 레버리지 사용)
        """
        # 모멘텀 1위 섹터 찾기
        top_sector = None
        for sector, data in momentum_data.items():
            if data.get("rank") == 1:
                top_sector = sector
                break

        if not top_sector:
            return None

        sector_etf = SECTOR_LEVERAGE_ETF.get(top_sector)
        if sector_etf:
            return sector_etf

        return None

    def _confidence(self, regime: str, us_overnight: dict, score: float) -> float:
        """진입 신뢰도 (0~100)."""
        result = 0.0
        regime_scores = {"BULL": 80, "BEAR": 70, "CRISIS": 90}
        result += regime_scores.get(regime, 0) * 0.4

        us_grade = us_overnight.get("grade", 3)
        if regime == "BULL":
            us_map = {1: 100, 2: 80, 3: 50, 4: 20, 5: 0}
        else:
            us_map = {1: 0, 2: 20, 3: 50, 4: 80, 5: 100}
        result += us_map.get(us_grade, 50) * 0.3
        result += score * 0.3
        return round(result, 1)

    def _entry_reason(self, regime: str, us_overnight: dict, score: float, target_etf: dict = None) -> str:
        us_grade = us_overnight.get("grade", 3)
        us_signal = us_overnight.get("signal", "neutral")
        base = f"레짐: {regime} | US야간: {us_grade}등급({us_signal}) | 5축: {score:.0f}점"
        # 섹터 레버리지 선택 시 표시
        if target_etf and target_etf.get("code") != "122630":
            base += f" | 🎯 섹터정밀: {target_etf['name']}"
        return base

    def _risk_note(self, etf: dict) -> str:
        mult = abs(etf["multiplier"])
        if mult >= 2:
            return f"⚠️ {mult}배 레버리지 - 최대 {self.max_hold_days}일 보유 권장"
        return f"⚠️ {mult}배 - 변동성 감가 주의"

    def _check_position(self, current_regime: str) -> dict:
        """보유 포지션 점검."""
        if not self.current_position:
            return {"action": "NONE", "reason": "포지션 없음"}

        pos = self.current_position
        if pos.pnl_pct <= self.stop_loss_pct:
            return {"action": "CLOSE", "reason": f"손절 발동 ({pos.pnl_pct:.1f}% ≤ {self.stop_loss_pct}%)"}
        if pos.hold_days >= self.max_hold_days:
            return {"action": "CLOSE", "reason": f"보유일 초과 ({pos.hold_days}일 ≥ {self.max_hold_days}일)"}
        if pos.multiplier > 0 and current_regime != "BULL":
            return {"action": "CLOSE", "reason": f"레짐 전환 ({current_regime}) - 롱 레버리지 청산"}
        if pos.multiplier < 0 and current_regime == "BULL":
            return {"action": "CLOSE", "reason": "레짐 BULL 전환 - 인버스 청산"}
        return {"action": "HOLD", "reason": "조건 충족 - 유지"}

    def _emergency_close(self, reason: str) -> dict:
        return {
            "decision": self._to_dict(LeverageDecision(
                signal="EMERGENCY_SELL",
                etf_code=self.current_position.code if self.current_position else "",
                etf_name=self.current_position.name if self.current_position else "",
                reason=f"🚨 {reason}",
                risk_note="익일 시가 전량 청산 필요",
            )),
            "position_action": "EMERGENCY_CLOSE",
            "summary": f"🚨 [레버리지] 긴급청산: {reason}",
            "timestamp": datetime.now().isoformat(),
        }

    def _build_summary(self, d: LeverageDecision, regime: str) -> str:
        if d.signal == "BUY":
            return f"[레버리지] 매수: {d.etf_name} (신뢰도 {d.confidence:.0f}%) | {d.reason}"
        elif d.signal == "HOLD":
            return f"[레버리지] 보유유지: {d.etf_name}"
        elif d.signal == "SELL":
            return f"[레버리지] 매도: {d.reason}"
        return f"[레버리지] 진입대기 ({regime}) | {d.reason}"

    def _to_dict(self, d: LeverageDecision) -> dict:
        return {
            "signal": d.signal, "etf_code": d.etf_code, "etf_name": d.etf_name,
            "multiplier": d.multiplier, "confidence": d.confidence,
            "reason": d.reason, "risk_note": d.risk_note,
        }
