# -*- coding: utf-8 -*-
"""tests/test_kill_switch.py — kill_switch/monitor.py L4 킬스위치 테스트.

전부 tmp_path 격리 — 실제 kill_switch/state.json, data/KILL_SWITCH, data_store/ 절대 무접촉.
네트워크 0, 합성 데이터만. release_code는 monkeypatch로 환경변수 고정(결정적).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pytest

from kill_switch.monitor import (
    DAILY_KILL_LIMIT,
    DEFAULT_KILL_FLAG,
    DEFAULT_STATE_PATH,
    KillMetrics,
    KillSwitchState,
    QUOTE_STALE_MAX_SEC,
    TOTAL_KILL_LIMIT,
    evaluate,
    load_state,
    release,
    release_code,
    run_watchdog,
    save_state,
    trip,
)


# ── 픽스처: 격리 경로 + 결정적 해제코드 ─────────────────────────────────

@pytest.fixture
def state_path(tmp_path):
    """격리된 state.json 경로 — 실제 kill_switch/state.json 오염 방지."""
    return tmp_path / "state.json"


@pytest.fixture
def flag_path(tmp_path):
    """격리된 KILL_SWITCH 플래그 경로 — 실제 data/KILL_SWITCH(5/19 발동본) 오염 방지."""
    return tmp_path / "KILL_SWITCH"


@pytest.fixture(autouse=True)
def _fixed_hmac_key(monkeypatch):
    """해제 코드 파생 키 고정 — 환경 따라 코드가 달라져 테스트가 흔들리는 사고 방지."""
    monkeypatch.setenv("ORDER_INTENTS_HMAC_KEY", "test-hmac-key-for-pytest")


def _isolation_guard(state_path, flag_path):
    """격리 자기검증 — 테스트 경로가 실수로 기본(운영) 경로와 같아지는 사고 방지."""
    assert Path(state_path).resolve() != DEFAULT_STATE_PATH.resolve()
    assert Path(flag_path).resolve() != DEFAULT_KILL_FLAG.resolve()


# ── 1~4: evaluate 룰별 발동 ──────────────────────────────────────────────

def test_evaluate_k1_daily_loss_trips():
    """일일 -4% 손실인데 킬스위치가 침묵해 추가 손실이 누적되는 사고를 막는다."""
    state, actions = evaluate(KillMetrics(daily_pnl_pct=-0.04), KillSwitchState())
    assert state.tripped is True
    assert state.rule == "K1"
    assert "LIQUIDATE_ALL" in actions
    assert "HALT_24H" in state.actions


def test_evaluate_k1_boundary():
    """경계값 처리 착오(-0.035 미발동/-0.034 발동)로 한도가 사실상 느슨해지는 사고를 막는다."""
    s_fire, _ = evaluate(KillMetrics(daily_pnl_pct=DAILY_KILL_LIMIT), KillSwitchState())
    assert s_fire.tripped is True and s_fire.rule == "K1"
    s_skip, acts = evaluate(KillMetrics(daily_pnl_pct=-0.034), KillSwitchState())
    assert s_skip.tripped is False and acts == []


def test_evaluate_k2_cumulative_dd_trips_and_beats_k1():
    """누적 DD -12%를 일일 룰(K1)로 오분류해 24시간 뒤 자동 재개되는 사고를 막는다."""
    # K2 단독
    state, actions = evaluate(KillMetrics(cumulative_dd_pct=-0.12), KillSwitchState())
    assert state.tripped is True and state.rule == "K2"
    assert "HALT_INDEFINITE" in actions
    # K1·K2 동시 충족 → 더 치명적인 K2가 우선
    state2, _ = evaluate(
        KillMetrics(daily_pnl_pct=-0.05, cumulative_dd_pct=-0.12), KillSwitchState()
    )
    assert state2.rule == "K2"


def test_evaluate_k3_each_data_anomaly_blocks_orders():
    """깨진 시세(stale/NaN/인증실패) 위에서 신규 주문이 나가는 사고를 막는다."""
    for metrics in (
        KillMetrics(quote_stale_sec=400),
        KillMetrics(price_nan=True),
        KillMetrics(api_auth_fail=True),
    ):
        state, actions = evaluate(metrics, KillSwitchState())
        assert state.tripped is True, f"미발동: {metrics}"
        assert state.rule == "K3"
        assert actions == ["BLOCK_NEW_ORDERS"]
    # 경계: 300s 발동 / 299s 스킵
    s_fire, _ = evaluate(KillMetrics(quote_stale_sec=QUOTE_STALE_MAX_SEC), KillSwitchState())
    assert s_fire.tripped is True
    s_skip, _ = evaluate(KillMetrics(quote_stale_sec=299), KillSwitchState())
    assert s_skip.tripped is False


def test_evaluate_nan_metric_fails_closed_k3():
    """NaN 손익률을 '데이터 없음'으로 착각해 룰을 건너뛰고 거래가 계속되는 사고를 막는다."""
    state, actions = evaluate(KillMetrics(daily_pnl_pct=float("nan")), KillSwitchState())
    assert state.tripped is True and state.rule == "K3"
    assert "BLOCK_NEW_ORDERS" in actions


def test_evaluate_k4_ungated_fill_trips():
    """게이트 로그 없는 체결(통제 밖 주문 경로)이 조용히 지나가는 사고를 막는다."""
    state, actions = evaluate(KillMetrics(fills_without_gate_log=1), KillSwitchState())
    assert state.tripped is True
    assert state.rule == "K4"
    assert "BLOCK_NEW_ORDERS" in actions


# ── 5~6: None 스킵 / 자동 해제 금지 ─────────────────────────────────────

def test_evaluate_all_none_metrics_no_trip():
    """1b 배선 전 미배선(None) 지표만으로 거짓 발동해 멀쩡한 시스템이 멈추는 사고를 막는다."""
    state_in = KillSwitchState()
    state_out, actions = evaluate(KillMetrics(), state_in)
    assert state_out.tripped is False
    assert actions == []
    assert state_out is state_in  # 미발동 시 입력 객체 그대로 반환(불필요한 복제 없음)


def test_evaluate_tripped_state_never_auto_releases():
    """★핵심 안티패턴: 발동 후 지표가 +5%로 회복됐다고 시스템이 스스로 풀어버리는 사고를 막는다."""
    tripped = KillSwitchState(
        tripped=True,
        rule="K1",
        reason="일일 손실",
        activated_at="2026-06-11T09:00:00+09:00",
        actions=["LIQUIDATE_ALL", "HALT_24H"],
    )
    state_out, actions = evaluate(KillMetrics(daily_pnl_pct=0.05), tripped)
    assert state_out.tripped is True
    assert state_out.rule == "K1"
    assert actions == ["LIQUIDATE_ALL", "HALT_24H"]  # 기존 actions 계속 유효


# ── 7~8: 영속화 / fail-closed 로드 ──────────────────────────────────────

def test_save_load_roundtrip_survives_reboot(state_path):
    """서버 리부트(프로세스 재기동)로 발동 기록이 증발해 거래가 재개되는 사고를 막는다."""
    original = KillSwitchState(
        tripped=True,
        rule="K2",
        reason="누적 DD",
        activated_at="2026-06-11T10:00:00+09:00",
        actions=["LIQUIDATE_ALL", "HALT_INDEFINITE"],
        history=[{"event": "test"}],
    )
    save_state(original, state_path)
    # 재부팅 시뮬: 파일에서 다시 로드
    reloaded = load_state(state_path)
    assert reloaded.tripped is True
    assert reloaded.rule == "K2"
    assert reloaded.actions == ["LIQUIDATE_ALL", "HALT_INDEFINITE"]
    assert reloaded.history == [{"event": "test"}]
    # saved_at KST 자기선언이 기록돼 있어야 한다(쓰기 3원칙 ②)
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert "saved_at" in raw and "+09:00" in raw["saved_at"]


def test_load_corrupted_state_fails_closed(state_path):
    """★state.json이 깨졌을 때 '모름=열림'으로 해석해 차단이 풀리는 사고를 막는다."""
    state_path.write_text("{{{ 이것은 JSON이 아니다 garbage ###", encoding="utf-8")
    state = load_state(state_path)
    assert state.tripped is True
    assert state.rule == "K3"
    assert "fail-closed" in (state.reason or "")
    assert "BLOCK_NEW_ORDERS" in state.actions


def test_load_missing_state_returns_untripped(state_path):
    """파일 없음(정상 초기 상태)을 깨짐으로 오판해 첫 가동부터 차단되는 사고를 막는다."""
    assert not state_path.exists()
    state = load_state(state_path)
    assert state.tripped is False


# ── 9: trip — 플래그 append 보존 ────────────────────────────────────────

def test_trip_persists_state_and_appends_flag_preserving_content(state_path, flag_path):
    """trip이 EnvChecker 소유 KILL_SWITCH 플래그를 덮어써 5/19 발동 증거가 증발하는 사고를 막는다."""
    _isolation_guard(state_path, flag_path)
    existing = "# EnvChecker 기존 발동 내용 (2026-05-19)\nreason=기존사유\n"
    flag_path.write_text(existing, encoding="utf-8")

    state = trip("K1", "테스트 발동", state_path=state_path, flag_path=flag_path)

    assert state.tripped is True and state.rule == "K1"
    # state.json 영속화
    reloaded = load_state(state_path)
    assert reloaded.tripped is True and reloaded.rule == "K1"
    # 플래그: 기존 내용 보존 + 구분선/사유 append
    content = flag_path.read_text(encoding="utf-8")
    assert content.startswith(existing)  # 원본 보존(덮어쓰기 금지)
    assert "# --- kill_switch trip " in content
    assert "rule=K1" in content
    assert "reason=테스트 발동" in content


def test_trip_creates_flag_when_missing(state_path, flag_path):
    """플래그 파일이 없을 때 발동 흔적이 어디에도 안 남는 사고를 막는다."""
    assert not flag_path.exists()
    trip("K3", "데이터 이상 테스트", state_path=state_path, flag_path=flag_path)
    assert flag_path.exists()
    assert "rule=K3" in flag_path.read_text(encoding="utf-8")


def test_trip_while_tripped_preserves_first_activation(state_path, flag_path):
    """중복 발동이 첫 발동(rule/activated_at) 증거를 덮어써 원인 추적이 불가능해지는 사고를 막는다."""
    first = trip("K1", "첫 발동", state_path=state_path, flag_path=flag_path)
    second = trip("K3", "두번째 발동 시도", state_path=state_path, flag_path=flag_path)
    assert second.rule == "K1"  # 원발동 보존
    assert second.activated_at == first.activated_at
    assert any(h.get("event") == "trip_while_tripped" for h in second.history)


def test_trip_rejects_unknown_rule(state_path, flag_path):
    """오타 룰("K9")이 조용히 기록돼 actions 매핑이 깨지는 사고를 막는다."""
    with pytest.raises(ValueError):
        trip("K9", "없는 룰", state_path=state_path, flag_path=flag_path)


# ── 10~11: release — 코드 검증 / 플래그 무접촉 ──────────────────────────

def test_release_wrong_code_raises_and_stays_tripped(state_path, flag_path):
    """아무 문자열로 해제가 통과돼 사람 승인 없이 거래가 재개되는 사고를 막는다."""
    trip("K2", "누적 DD 테스트", state_path=state_path, flag_path=flag_path)
    with pytest.raises(ValueError):
        release("000000" if release_code(load_state(state_path)) != "000000" else "999999",
                state_path=state_path)
    assert load_state(state_path).tripped is True  # 여전히 차단 유지


def test_release_correct_code_releases_with_history(state_path, flag_path):
    """정당한 수동 해제가 안 풀리거나, 풀려도 발동 이력이 사라지는 사고를 막는다."""
    trip("K1", "해제 테스트", state_path=state_path, flag_path=flag_path)
    code = release_code(load_state(state_path))
    assert len(code) == 6 and code.isdigit()
    released = release(code, state_path=state_path)
    assert released.tripped is False
    assert released.rule is None
    assert released.released_at is not None
    assert len(released.history) == 1  # 발동~해제 1건 보존
    assert released.history[0]["rule"] == "K1"


def test_release_when_not_tripped_raises(state_path):
    """미발동 상태에서 release가 통과돼 history에 유령 해제 기록이 남는 사고를 막는다."""
    save_state(KillSwitchState(), state_path)
    with pytest.raises(ValueError):
        release("123456", state_path=state_path)


def test_release_does_not_touch_flag_file(state_path, flag_path):
    """release가 EnvChecker 소유 플래그까지 지워 환경 차단이 풀리는 사고를 막는다."""
    trip("K1", "플래그 무접촉 테스트", state_path=state_path, flag_path=flag_path)
    before_bytes = flag_path.read_bytes()
    code = release_code(load_state(state_path))
    release(code, state_path=state_path)
    assert flag_path.read_bytes() == before_bytes  # 바이트 단위 동일(무접촉)


# ── 12: watchdog ────────────────────────────────────────────────────────

def test_run_watchdog_trips_and_persists(state_path, flag_path):
    """watchdog이 한도 초과를 보고도 메모리에서만 발동해 재기동 시 증발하는 사고를 막는다."""
    _isolation_guard(state_path, flag_path)

    def provider():
        return KillMetrics(daily_pnl_pct=-0.05)

    final = run_watchdog(
        provider,
        state_path=state_path,
        flag_path=flag_path,
        interval_sec=0,
        max_loops=2,
    )
    assert final.tripped is True and final.rule == "K1"
    # state 파일에 영속(재부팅 시뮬)
    assert load_state(state_path).tripped is True
    assert flag_path.exists()


def test_run_watchdog_provider_exception_fails_closed(state_path, flag_path):
    """지표 공급자가 죽었는데(킬스위치가 눈을 잃음) 차단 없이 루프만 도는 사고를 막는다."""
    def broken_provider():
        raise RuntimeError("지표 수집 실패")

    final = run_watchdog(
        broken_provider,
        state_path=state_path,
        flag_path=flag_path,
        interval_sec=0,
        max_loops=2,
    )
    assert final.tripped is True and final.rule == "K3"
    assert load_state(state_path).tripped is True


def test_run_watchdog_no_trip_writes_nothing(state_path, flag_path):
    """정상 지표인데 watchdog이 매 루프마다 불필요한 쓰기를 해 운영 파일을 오염시키는 사고를 막는다."""
    final = run_watchdog(
        lambda: KillMetrics(),
        state_path=state_path,
        flag_path=flag_path,
        interval_sec=0,
        max_loops=2,
    )
    assert final.tripped is False
    assert not state_path.exists()  # 미발동이면 state.json 미생성
    assert not flag_path.exists()


# ── 13: ★모의발동 통합 시나리오 (체크리스트 B) ──────────────────────────

def test_full_scenario_trip_reboot_release(state_path, flag_path):
    """발동→재부팅→해제 전 과정 중 어느 한 고리가 끊겨 차단이 새거나 안 풀리는 사고를 막는다."""
    _isolation_guard(state_path, flag_path)
    # ① 발동
    trip("K2", "통합 시나리오: 누적 DD -10%", state_path=state_path, flag_path=flag_path)
    # ② 재부팅 시뮬: 메모리 버리고 파일에서 재로드 → 여전히 tripped
    rebooted = load_state(state_path)
    assert rebooted.tripped is True and rebooted.rule == "K2"
    # ③ 재부팅 후에도 evaluate는 정상 지표로 자동 해제하지 않는다
    still, _ = evaluate(KillMetrics(daily_pnl_pct=0.05, cumulative_dd_pct=0.0), rebooted)
    assert still.tripped is True
    # ④ 정확한 코드로 수동 해제
    code = release_code(rebooted)
    released = release(code, state_path=state_path)
    assert released.tripped is False
    # ⑤ 해제 후 재로드해도 released 유지 + history 보존 + 플래그는 그대로 존재(사람 몫)
    final = load_state(state_path)
    assert final.tripped is False
    assert len(final.history) == 1
    assert flag_path.exists()


# ── 14: evaluate 순수성 ─────────────────────────────────────────────────

def test_evaluate_is_pure_no_fs_writes_no_mutation(tmp_path, monkeypatch):
    """판정 함수(evaluate)가 몰래 파일을 쓰거나 입력 state를 변조하는 사고를 막는다."""
    monkeypatch.chdir(tmp_path)  # 상대경로 쓰기가 있어도 tmp에 잡히도록
    state_in = KillSwitchState(history=[{"event": "기존이력"}])
    new_state, actions = evaluate(KillMetrics(daily_pnl_pct=-0.04), state_in)
    # 발동 결과는 새 객체 — 입력 state 비변경
    assert new_state.tripped is True
    assert state_in.tripped is False
    assert state_in.rule is None
    assert state_in.history == [{"event": "기존이력"}]
    # 파일시스템 무변화(부작용 0)
    assert list(tmp_path.iterdir()) == []


# ── 적대리뷰 회귀: 승급 / fills NaN / 키부재 fail-closed / CLI ───────────────

def test_evaluate_upgrades_more_lethal_rule_while_tripped():
    # ★사고 방지(P1): state.json 손상으로 K3(주문차단)만 걸린 채 진짜 K1(전량청산)이 삼켜진다.
    k3_state = KillSwitchState(
        tripped=True, rule="K3", reason="state.json corrupted — fail-closed",
        activated_at="2026-06-12T09:00:00+09:00", actions=["BLOCK_NEW_ORDERS"],
    )
    upgraded, actions = evaluate(KillMetrics(daily_pnl_pct=-0.04), k3_state)
    assert upgraded.tripped is True and upgraded.rule == "K1"  # K3 → K1 승급
    assert "LIQUIDATE_ALL" in actions
    assert upgraded.activated_at == "2026-06-12T09:00:00+09:00"  # 최초 발동시각 보존


def test_evaluate_no_downgrade_while_tripped():
    # 사고 방지: 이미 K1인데 K3 지표가 들어와도 덜 치명적 룰로 강등되지 않는다.
    k1_state = KillSwitchState(
        tripped=True, rule="K1", activated_at="t", actions=["LIQUIDATE_ALL", "HALT_24H"],
    )
    out, actions = evaluate(KillMetrics(quote_stale_sec=400), k1_state)
    assert out.rule == "K1" and "LIQUIDATE_ALL" in actions


def test_evaluate_fills_nan_fails_closed_k3():
    # ★사고 방지(P2): fills_without_gate_log=NaN이 NaN>0==False로 K4 무발동되던 것 → K3 fail-closed.
    state, actions = evaluate(KillMetrics(fills_without_gate_log=float("nan")), KillSwitchState())
    assert state.tripped is True and state.rule == "K3"
    assert "BLOCK_NEW_ORDERS" in actions


def test_release_code_fails_closed_without_key(monkeypatch, state_path, flag_path):
    # ★사고 방지(P1): 키 미설정 시 state.json 평문만으로 해제코드 파생 → 자동 우회 해제 가능.
    trip("K2", "키부재 테스트", state_path=state_path, flag_path=flag_path)
    st = load_state(state_path)
    monkeypatch.delenv("ORDER_INTENTS_HMAC_KEY", raising=False)
    with pytest.raises(ValueError):
        release_code(st)
    with pytest.raises(ValueError):
        release("123456", state_path=state_path)  # release도 release_code 거쳐 차단


def test_run_watchdog_upgrades_on_state_corruption(state_path, flag_path):
    # ★사고 방지(P1) 통합: state.json 손상(K3) 채 K1 metrics 유입 → watchdog이 K1로 승급·영속화.
    _isolation_guard(state_path, flag_path)
    state_path.write_text("garbage not json", encoding="utf-8")  # 손상 → load가 K3 fail-closed
    final = run_watchdog(
        lambda: KillMetrics(daily_pnl_pct=-0.05),
        state_path=state_path, flag_path=flag_path, interval_sec=0, max_loops=2,
    )
    assert final.tripped is True and final.rule == "K1"  # K3 → K1 승급
    reloaded = load_state(state_path)
    assert reloaded.rule == "K1" and "LIQUIDATE_ALL" in reloaded.actions  # 영속화


def test_cli_status_default_no_side_effects(state_path, flag_path, capsys):
    # 사고 방지: 인자 없는 status 호출이 파일을 쓰면 점검만으로 운영 상태가 오염된다.
    from kill_switch.monitor import main
    rc = main(["--state-path", str(state_path), "--flag-path", str(flag_path)])
    assert rc == 0
    assert not state_path.exists()  # status는 읽기 전용
    assert "KILL SWITCH 상태" in capsys.readouterr().out


def test_cli_test_fire_guard_rejects_default_path():
    # ★사고 방지: --test-fire가 기본(운영) 경로를 건드리면 실상태/EnvChecker 플래그가 오염된다.
    from kill_switch.monitor import main
    assert main(["--test-fire", "K1"]) == 2  # 기본 경로 → 거부(쓰기 전)


def test_cli_test_fire_isolated_ok_and_release(state_path, flag_path, capsys):
    # 사고 방지: 격리 경로 모의발동(B항목 CLI) + 해제코드 표시 → 해제 전 경로가 끊기는 사고.
    from kill_switch.monitor import main
    assert main(["--test-fire", "K1", "--state-path", str(state_path),
                 "--flag-path", str(flag_path)]) == 0
    assert load_state(state_path).rule == "K1"
    capsys.readouterr()  # 버퍼 비우기
    main(["--show-release-code", "--state-path", str(state_path)])
    code = capsys.readouterr().out.strip().split(":")[-1].strip()
    assert main(["--release", code, "--state-path", str(state_path)]) == 0
    assert load_state(state_path).tripped is False
