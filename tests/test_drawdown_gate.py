"""pre_trade_gate G8 (드로다운 사다리) 통합 — RISK_ENGINE Phase 3b.

ladder 주입 시 G8: 신규 진입 금지(step 2/3) → REJECT, 사이즈 축소(step 1) → 사전 비례 축소.
ladder=None이면 not_active(Phase 1a 호환 — test_pre_trade_gate/var_gate가 별도 보증).
★original(proposed)은 감사용 보존, RESIZE 시작 사이즈만 G8로 조정.
"""
from risk.drawdown_ladder import ladder_state
from risk.pre_trade_gate import GateRequest, evaluate_pre_trade


def _req(size, equity=100_000_000):
    return GateRequest(ticker="A", sector="ETF", proposed_size_krw=size,
                       equity_krw=equity, adv20_krw=1e15)  # adv 큼 → G6 통과


def test_g8_none_not_active():
    res = evaluate_pre_trade(_req(5_000_000), [])
    assert "not_active" in res.checks["G8"]["status"]
    assert res.verdict == "PASS"


def test_g8_normal_zone_passes():
    res = evaluate_pre_trade(_req(5_000_000), [], ladder=ladder_state(-0.02))
    assert res.checks["G8"]["status"] == "pass"
    assert res.checks["G8"]["step"] == 0
    assert res.verdict == "PASS"


def test_g8_step1_resizes_half():
    # DD -5% → step1 → 사이즈 50% 축소(사전)
    res = evaluate_pre_trade(_req(5_000_000), [], ladder=ladder_state(-0.05))
    assert res.verdict == "RESIZE"
    assert res.checks["G8"]["status"] == "resize"
    assert abs(res.final_size_krw - 2_500_000) < 1.0   # 5M × 0.5
    assert res.original_size_krw == 5_000_000          # proposed 보존(감사)


def test_g8_step2_rejects():
    # DD -8% → step2 → 신규 금지 REJECT
    res = evaluate_pre_trade(_req(5_000_000), [], ladder=ladder_state(-0.08))
    assert res.verdict == "REJECT"
    assert any(v.get("gate") == "G8" for v in res.violations)
    assert res.checks["G8"]["status"] == "violation"


def test_g8_kill_switch_zone_rejects():
    # DD -11% → step3 → 게이트는 신규 금지 REJECT(L4 킬스위치 이관은 별도 트랙)
    res = evaluate_pre_trade(_req(5_000_000), [], ladder=ladder_state(-0.11))
    assert res.verdict == "REJECT"
    assert any(v.get("gate") == "G8" for v in res.violations)


def test_g8_resize_still_checks_g3():
    # step1 축소 후에도 G3 등 한도 검증이 돈다(축소된 사이즈로 통과)
    res = evaluate_pre_trade(_req(5_000_000), [], ladder=ladder_state(-0.05))
    assert res.verdict == "RESIZE"
    assert res.checks["G3"]["status"] == "pass"
