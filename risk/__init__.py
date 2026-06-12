"""리스크 엔진 — 미니 알라딘 (docs/01-plan/RISK_ENGINE_SPEC_v2.md).

수익은 결과, 생존이 목표. 주문은 사전 게이트를 통과해야만 나간다.
Phase 1a: config + sizing(L1) + pre_trade_gate(L2 정적 G3~G6) — execution 배선은 Phase 1b(unfreeze 직전).
※ kill_switch/는 이 패키지와 의도적으로 분리된 별도 프로세스(§5) — 여기서 import하지 않는다.
"""
from risk.config import KST, RISK_CONFIG, RiskConfig, limit_down_survival_ok

__all__ = ["KST", "RISK_CONFIG", "RiskConfig", "limit_down_survival_ok"]
