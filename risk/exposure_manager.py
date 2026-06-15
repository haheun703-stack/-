"""risk/exposure_manager.py — L3 노출 관리 계층(계수 합성). RISK_ENGINE 스펙 §4.2·§4.3·§4.4.

세 L3 모니터가 각자 내는 총노출 계수를 한 곳에서 곱해 **최종 목표 노출 계수**를 만든다.
drawdown_ladder·vol_targeting·crowding은 서로 격리(config만 import)돼 각자 계수만 낼 뿐,
"호출처가 함께 곱한다"고만 명시돼 있었다 — 이 모듈이 그 곱셈 계층(factor_stress가
factor_exposure↔stress_test를 묶듯, 여기선 세 노출 계수를 묶는다).

  최종 = base × ladder.gross_exposure(§4.2) × vol.scale(§4.3) × crowding.gross_exposure_mult(§4.4)

★세 계수 모두 ≤ 1.0(사다리 0~1·vol ≤1·크라우딩 0.8/1) → 곱도 ≤ 1.0 = **축소 전용**. 노출을
  키우는 방향으로는 절대 작동하지 않는다(§0 "생존이 목표", vol_targeting의 레버리지 확대 금지 일관).

★사다리 이중 적용 주의(혼동 방지): drawdown_ladder는 두 출력을 낸다 —
  - gross_exposure(총노출 한도 0.7/0.4): **여기(노출 관리)** 가 쓴다.
  - new_size_mult/new_entry_allowed(개별 신규 매수 축소/금지): **게이트 G8(pre_trade_gate)** 의 영역.
  적용 대상이 다르므로(포트 총노출 vs 건별 신규 사이즈) 이중 적용이 아니다 — 같은 DD가 총노출도
  줄이고 신규 매수도 더 줄이는 것은 스펙 §4.2 표 그대로의 의도다("노출 70% **AND** 신규 50% 축소").

★게이트(G1~G8)가 아니라 L3 노출 조절 합성기. 순수 계산 — 실주문 경로 접촉 0, write 0,
  risk(config·세 모니터) 외 미import. 세 모니터 자체는 무변경(이 모듈만 셋을 묶는다 = 격리).
  데이터 미주입 컴포넌트는 중립 1.0(과조절 방지) → 전부 미주입이면 최종 1.0 = 현행 노출 불변.
  ★production 미배선: 이 계수를 소비해 사이징을 줄이는 연결은 unfreeze 직전(별 트랙) = freeze 유지.
"""
from __future__ import annotations

from dataclasses import dataclass

from risk.config import RISK_CONFIG, RiskConfig
from risk.crowding import CrowdingState, crowding_state
from risk.drawdown_ladder import LadderState, ladder_state
from risk.vol_targeting import VolTargetState, vol_target_scale

# vol.scale 표시/비교용 — 부동소수 1.0 경계에서 "조절 없음"을 깔끔히 판정.
_NEUTRAL_EPS = 1e-9


@dataclass(frozen=True)
class ExposurePlan:
    """최종 노출 계수 + 컴포넌트 분해(투명성). 각 _mult는 해당 L3 모니터의 기여."""

    gross_exposure_mult: float          # 최종 = ladder × vol × crowding (≤ 1.0)
    ladder_mult: float                  # drawdown_ladder.gross_exposure (미주입 1.0)
    vol_scale: float                    # vol_targeting.scale (미주입 1.0)
    crowding_mult: float                # crowding.gross_exposure_mult (미주입 1.0)
    target_exposure_krw: float | None   # base 주입 시 base × 최종, 아니면 None
    to_kill_switch: bool                # 사다리 step3(DD -10% 초과) = L4 이관 신호
    interpretation: str


def _interpret(final: float, lm: float, vs: float, cm: float, kill: bool) -> str:
    """최종 계수 → 한 줄 해석. 조절에 기여한 컴포넌트만 나열."""
    if kill:
        return "총노출 0% (드로다운 -10% 초과 → L4 킬스위치 이관)"
    parts: list[str] = []
    if lm < 1.0 - _NEUTRAL_EPS:
        parts.append(f"사다리 {lm:.0%}")
    if vs < 1.0 - _NEUTRAL_EPS:
        parts.append(f"변동성 {vs:.0%}")
    if cm < 1.0 - _NEUTRAL_EPS:
        parts.append(f"크라우딩 {cm:.0%}")
    if not parts:
        return "총노출 100% (조절 없음 — 데이터 미주입/정상)"
    return f"총노출 {final * 100:.0f}% (" + " × ".join(parts) + ")"


def combine_exposure_multipliers(
    ladder: LadderState | None,
    vol: VolTargetState | None,
    crowding: CrowdingState | None,
) -> tuple[float, float, float, float]:
    """세 L3 계수 → (ladder_mult, vol_scale, crowding_mult, 최종 곱). None = 중립 1.0.

    곱셈 합성: 각 ≤ 1.0이라 최종도 ≤ 1.0(축소 전용). 컴포넌트가 None(미평가/미주입)이면 그 축은
    1.0으로 흘려보낸다 — 데이터 없는 위험을 0.x로 가정해 과조절하지 않는다(crowding·vol 철학 일관).
    """
    lm = ladder.gross_exposure if ladder is not None else 1.0
    vs = vol.scale if vol is not None else 1.0
    cm = crowding.gross_exposure_mult if crowding is not None else 1.0
    return lm, vs, cm, lm * vs * cm


def compute_exposure_plan(
    *,
    base_exposure_krw: float | None = None,
    current_dd: float | None = None,
    prev_step: int = 0,
    portfolio_returns=None,
    holding_returns=None,
    vkospi_series=None,
    foreign_futures_series=None,
    cfg: RiskConfig = RISK_CONFIG,
) -> ExposurePlan:
    """세 L3 모니터를 평가해 최종 목표 노출 계수를 합성(§4.2·§4.3·§4.4).

    Args:
        base_exposure_krw: 기본(목표) 총노출 원화. 주입 시 target_exposure_krw = base × 최종 계수.
        current_dd: 계좌 고점 대비 DD(음수). None이면 사다리 미적용(1.0). prev_step=복귀 히스테리시스.
        portfolio_returns: 포트 일별 수익률(§4.3 vol 타겟). 미주입/표본부족 → scale 1.0.
        holding_returns: {ticker: 수익률}(§4.4 C1). vkospi_series/foreign_futures_series = C2/C3.
            전부 미주입이면 crowding 1.0. gate_wiring의 returns_by_ticker를 holding_returns로 재사용 가능.
        cfg: ladder_steps·ladder_hysteresis·target_vol_annual·crowding_corr 단일 출처.

    Returns:
        ExposurePlan. 전부 미주입 → gross_exposure_mult 1.0(현행 노출 불변). 사다리 step3 → 0.0 + kill.
    """
    ladder = ladder_state(current_dd, prev_step, cfg) if current_dd is not None else None
    vol = vol_target_scale(portfolio_returns, cfg=cfg)  # 미주입 시 내부에서 scale 1.0 반환
    crowding = crowding_state(holding_returns, vkospi_series, foreign_futures_series, cfg=cfg)

    lm, vs, cm, final = combine_exposure_multipliers(ladder, vol, crowding)
    target = float(base_exposure_krw) * final if base_exposure_krw is not None else None
    kill = ladder.to_kill_switch if ladder is not None else False
    return ExposurePlan(
        gross_exposure_mult=final,
        ladder_mult=lm,
        vol_scale=vs,
        crowding_mult=cm,
        target_exposure_krw=target,
        to_kill_switch=kill,
        interpretation=_interpret(final, lm, vs, cm, kill),
    )
