"""리스크 엔진 파라미터 단일 정의 (docs/01-plan/RISK_ENGINE_SPEC_v2.md §9).

설계 철학 (§0 — 코드 전체에 적용, 협상 불가):
1. 수익은 결과, 생존이 목표. 이 엔진의 목적은 수익 극대화가 아니라
   "어떤 상황에서도 계좌가 죽지 않는 구조"다.
2. 리스크 관리는 사후 보고서가 아니라 사전 게이트. 주문은 게이트를 통과해야만 나간다.
3. 모든 모델은 틀린다. 그래서 마지막 층은 모델이 아니라 하드 룰(킬스위치)이다.
4. 노출(exposure)이 아니라 리스크(risk)를 본다.

★파라미터 변경은 분기 1회만 허용. 손실 직후의 감정적 변경 금지 — 변경 사유를 일지에 기록.
★daily_kill_limit 변경 시 max_single_weight 재계산 필수 (§2.2 하한가 생존 조건).
★한도를 백테스트로 최적화하지 말 것 — 생존 제약이지 수익 파라미터가 아니다 (§10).
※ 킬스위치(kill_switch/)는 이 모듈을 import하지 않는다(§5 의도적 격리) — 상수 중복은 버그가 아니라 설계.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone

KST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class RiskConfig:
    # ── L1 사이징 (§2, Phase 1a) ──
    risk_per_trade: float = 0.005        # 초기 0.5% — 검증 3개월 후 최대 1.0%, 그 이상 금지
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    gap_lookback_days: int = 252
    gap_min_samples: int = 60         # 갭 표본 이 미만 = 짧은 이력 → 보수 floor (신규상장 갭 위험, §2.2 보강)
    sparse_gap_floor: float = 0.10    # 이력 부족 종목의 보수적 최악 갭 가정 — "모르면 보수적으로 크게 잡는다"
    adv_window: int = 20
    adv_limit_ratio: float = 0.05        # ADV20의 5% — 시장 충격 없이 하루 안에 전량 청산 가능
    # ── L2 정적 게이트 (§3.5, Phase 1a = G3~G6) ──
    max_single_weight: float = 0.12      # ≈|daily_kill_limit|/0.30=11.67% → 스펙 §9가 12%로 캡(명시)
    max_sector_weight: float = 0.30
    corr_cluster_threshold: float = 0.8
    resize_max_iter: int = 5
    min_trade_krw: float = 100_000.0     # RESIZE 하한 — 이하로 축소되면 REJECT
    # ── L2 VaR 게이트 (Phase 2에서 활성화 — 정의만 선등록) ──
    var_limit: float = 0.025             # G1 (1D, 95%)
    stress_var_limit: float = 0.04       # G2
    component_var_limit: float = 0.25    # G7
    ewma_lambda: float = 0.94            # RiskMetrics 표준
    t_dist_df: int = 6                   # Kupiec 실패 시 4로
    # ── L2 팩터 노출 (§3.1, factor_exposure) ──
    factor_ewma_halflife: int = 60       # EWMA 가중 회귀 반감기(일) — 최근성 반영
    factor_lookback: int = 252           # 회귀 표본 상한(일)
    factor_min_obs: int = 60             # 회귀 최소 표본 — 미만이면 베타 추정 생략(VaR/상관과 동일 보수 기준)
    factor_beta_instability: float = 0.5  # EWMA 베타 vs 252일 단순 베타 괴리 ≥50% = "베타 불안정" 경고(§3.1-2)
    # ── L3 (Phase 3 — 정의만 선등록) ──
    target_vol_annual: float = 0.15
    ladder_steps: tuple = ((-0.04, 1.0), (-0.07, 0.7), (-0.10, 0.4))
    ladder_hysteresis: float = 0.015
    crowding_corr: float = 0.70
    # ── L4 킬스위치 한도 (참조용 — 실제 발동은 kill_switch/가 자체 상수로) ──
    daily_kill_limit: float = -0.035     # K1
    total_kill_limit: float = -0.10      # K2
    quote_stale_max_sec: int = 300       # K3
    limit_down_pct: float = 0.30         # 하한가 생존 조건 분모


RISK_CONFIG = RiskConfig()


def limit_down_survival_ok(cfg: RiskConfig = RISK_CONFIG, tolerance: float = 0.005) -> bool:
    """하한가 생존 조건 (§2.2): max_single_weight ≤ |daily_kill_limit|/limit_down_pct.

    단일 종목이 하한가(-30%)를 가도 계좌 손실이 일일 킬스위치 한도를 넘지 않는 비중.
    스펙 §9가 11.67%→12%로 캡했으므로 tolerance(0.5%p)를 허용. daily_kill_limit 변경 시
    이 함수가 False가 되면 max_single_weight를 반드시 재계산할 것.
    """
    derived = abs(cfg.daily_kill_limit) / cfg.limit_down_pct
    return cfg.max_single_weight <= derived + tolerance
