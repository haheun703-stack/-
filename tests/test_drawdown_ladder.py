"""drawdown_ladder (G8) 검증 — RISK_ENGINE Phase 3.

불변식:
  1. DD 구간별 step / 노출 / 신규 허용 / 사이즈 배수.
  2. 히스테리시스: 복귀는 진입경계 +1.5%p 회복해야, 한 단계씩(점진).
  3. 악화는 즉시(히스테리시스는 완화 방향에만).
  4. 경계값 정확(<=), 양수 DD 클램프.
"""
from risk.drawdown_ladder import ladder_state


def test_normal_zone():
    s = ladder_state(-0.02)
    assert s.step == 0 and s.gross_exposure == 1.0 and s.new_entry_allowed and s.new_size_mult == 1.0
    assert not s.to_kill_switch


def test_step1_resize():
    s = ladder_state(-0.05)
    assert s.step == 1 and s.gross_exposure == 0.7 and s.new_entry_allowed and s.new_size_mult == 0.5


def test_step2_no_entry():
    s = ladder_state(-0.08)
    assert s.step == 2 and s.gross_exposure == 0.4 and not s.new_entry_allowed and s.new_size_mult == 0.0


def test_step3_kill_switch():
    s = ladder_state(-0.11)
    assert s.step == 3 and s.gross_exposure == 0.0 and s.to_kill_switch and not s.new_entry_allowed


def test_boundary_exact():
    assert ladder_state(-0.04).step == 1   # 정확히 -4% → step 1 (<=)
    assert ladder_state(-0.039).step == 0
    assert ladder_state(-0.07).step == 2
    assert ladder_state(-0.10).step == 3


def test_positive_dd_clamped():
    s = ladder_state(0.05)  # 고점 이상 → DD 0
    assert s.step == 0 and s.dd == 0.0


def test_hysteresis_holds_step():
    # step 1 진입 후 dd -3%(복귀임계 -2.5% 미달) → step 1 유지
    s = ladder_state(-0.03, prev_step=1)
    assert s.step == 1


def test_hysteresis_recovers():
    # step 1에서 dd -2%(> -2.5% 복귀임계) → step 0 복귀
    s = ladder_state(-0.02, prev_step=1)
    assert s.step == 0


def test_worsening_immediate():
    # prev_step 0인데 dd -8% → step 2 즉시(악화는 히스테리시스 무관)
    assert ladder_state(-0.08, prev_step=0).step == 2


def test_recovery_is_gradual():
    # step 2에서 dd -1%(정상 영역)이어도 한 단계만 복귀(step 1)
    assert ladder_state(-0.01, prev_step=2).step == 1
    # 다음 호출 prev_step=1 → step 0
    assert ladder_state(-0.01, prev_step=1).step == 0


def test_recovery_blocked_stays():
    # step 2에서 dd -6%(step2→1 복귀임계 -5.5% 미달) → step 2 유지
    assert ladder_state(-0.06, prev_step=2).step == 2
