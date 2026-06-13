"""account_peak (G8 DD 배선) 검증 — RISK_ENGINE Phase 3c.

계좌 고점 영속 추적 → DD → ladder_state(히스테리시스). gate_wiring이 이걸 evaluate(ladder=)에 주입.

불변식:
  1. 첫 평가/신고점 → DD 0, step 0(과차단 방지).
  2. 고점 대비 하락 → DD 음수 → 해당 사다리 step.
  3. step 영속(다음 평가 prev_step 히스테리시스).
  4. graceful: 파일 없음/글리치 → 기본값(현재가 고점).
"""
from risk.account_peak import EquityPeakStore


def test_first_eval_no_drawdown(tmp_path):
    ls = EquityPeakStore(tmp_path / "peak.json").update_and_ladder(100_000_000)
    assert ls.step == 0 and ls.dd == 0.0  # 첫 평가 = 고점 → DD 0


def test_peak_monotonic_new_high(tmp_path):
    store = EquityPeakStore(tmp_path / "peak.json")
    store.update_and_ladder(100_000_000)
    ls = store.update_and_ladder(120_000_000)  # 신고점
    assert ls.step == 0 and ls.dd == 0.0


def test_drawdown_triggers_ladder(tmp_path):
    store = EquityPeakStore(tmp_path / "peak.json")
    store.update_and_ladder(100_000_000)       # 고점 1억
    ls = store.update_and_ladder(94_000_000)   # -6% → step 1
    assert ls.step == 1
    assert abs(ls.dd - (-0.06)) < 1e-9


def test_deep_drawdown_kill_switch(tmp_path):
    store = EquityPeakStore(tmp_path / "peak.json")
    store.update_and_ladder(100_000_000)
    ls = store.update_and_ladder(88_000_000)   # -12% → step 3(킬스위치)
    assert ls.step == 3 and ls.to_kill_switch


def test_step_persisted_hysteresis(tmp_path):
    p = tmp_path / "peak.json"
    EquityPeakStore(p).update_and_ladder(100_000_000)  # 고점
    EquityPeakStore(p).update_and_ladder(92_000_000)   # -8% → step 2 영속
    # 새 인스턴스: prev_step=2 로드. -3% 회복 → step2에서 한 단계 복귀(step 1)
    ls = EquityPeakStore(p).update_and_ladder(97_000_000)
    assert ls.step == 1  # 점진 복귀 + 영속 prev_step 사용


def test_graceful_missing_file(tmp_path):
    ls = EquityPeakStore(tmp_path / "nope.json").update_and_ladder(50_000_000)
    assert ls.step == 0  # 기본 peak=0 → 현재가 고점 → DD 0
