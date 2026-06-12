# -*- coding: utf-8 -*-
"""risk/pre_trade_gate.py 테스트 — L2 사전 게이트 G3~G6 + RESIZE + 토큰 + 감사 로그.

합성 데이터만 사용. 네트워크 0. 운영 파일(kill_switch/state.json, data/, data_store/) 무접촉.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from datetime import datetime, timedelta

import pytest

from risk.config import KST, RISK_CONFIG
from risk.pre_trade_gate import (
    GateRequest,
    GateResult,
    Holding,
    evaluate_pre_trade,
    verify_gate_token,
)

# 고정 KST aware 시각 — 시계 의존 없는 결정적 테스트
NOW = datetime(2026, 6, 11, 10, 0, 0, tzinfo=KST)
EQUITY = 100_000_000.0  # 1억
BIG_ADV = 10_000_000_000.0  # ADV cap = 5억 → G6 항상 여유


@pytest.fixture(autouse=True)
def _no_env_hmac_key(monkeypatch):
    # 사고 방지: 개발 PC 환경변수 키가 새어들어 '키 없음' 테스트가 거짓 통과/실패하는 것 차단
    monkeypatch.delenv("ORDER_INTENTS_HMAC_KEY", raising=False)


def make_request(
    ticker="005930",
    sector="SEMI",
    size=5_000_000.0,
    equity=EQUITY,
    adv=BIG_ADV,
):
    return GateRequest(
        ticker=ticker,
        sector=sector,
        proposed_size_krw=size,
        equity_krw=equity,
        adv20_krw=adv,
    )


# ── 1. 깨끗한 요청 → PASS ────────────────────────────────────────────────────
def test_clean_request_pass():
    # 사고 방지: 정상 주문 의도가 게이트에서 오탐 차단되어 전략이 마비되는 것
    holdings = [Holding("000660", 5_000_000.0, sector="OTHER", corr_with_new=0.1)]
    r = evaluate_pre_trade(make_request(), holdings, now_kst=NOW)
    assert r.verdict == "PASS"
    assert r.final_size_krw == r.original_size_krw == 5_000_000.0
    assert r.resize_iterations == 0
    assert r.violations == []


def test_clean_request_checks_recorded():
    # 사고 방지: 감사용 checks에 활성/비활성 게이트 기록이 빠져 사후 추적 불가
    r = evaluate_pre_trade(make_request(), [], now_kst=NOW)
    for g in ("G3", "G4", "G5", "G6"):
        assert r.checks[g]["status"] == "pass"
    for g in ("G1", "G2", "G7", "G8"):
        assert "not_active" in r.checks[g]["status"]


def test_corr_below_threshold_excluded_from_cluster():
    # 사고 방지: 임계 미만 상관(0.79)까지 클러스터로 묶어 멀쩡한 주문을 과잉 차단
    holdings = [Holding("000660", 8_000_000.0, sector="OTHER", corr_with_new=0.79)]
    r = evaluate_pre_trade(make_request(size=8_000_000.0), holdings, now_kst=NOW)
    assert r.verdict == "PASS"
    assert r.checks["G5"]["cluster_tickers"] == []


# ── 2. G3 단일 종목 비중 → RESIZE ────────────────────────────────────────────
def test_g3_oversize_resized_to_limit():
    # 사고 방지: 단일 종목 20% 몰빵 주문이 그대로 나가 하한가 한 방에 킬스위치 초과 손실
    r = evaluate_pre_trade(make_request(size=0.20 * EQUITY), [], now_kst=NOW)
    assert r.verdict == "RESIZE"
    assert r.resize_iterations >= 1
    assert r.final_size_krw <= RISK_CONFIG.max_single_weight * EQUITY + 1e-6
    assert r.final_size_krw < r.original_size_krw
    assert r.checks["G3"]["status"] == "pass"  # 축소 후 재검사 통과 상태가 기록돼야


def test_g3_same_ticker_holding_aggregated():
    # 사고 방지: 같은 종목 기보유분을 빼먹고 신규분만 계산해 실질 비중 한도 초과
    holdings = [Holding("005930", 0.08 * EQUITY, sector="SEMI")]
    r = evaluate_pre_trade(make_request(size=0.10 * EQUITY), holdings, now_kst=NOW)
    assert r.verdict == "RESIZE"
    # 허용량 = 12% × equity − 기보유 8% = 4%
    assert r.final_size_krw == pytest.approx(0.04 * EQUITY)


# ── 3. RESIZE 결과 < min_trade_krw → REJECT ─────────────────────────────────
def test_resize_below_min_trade_rejected():
    # 사고 방지: 의미 없는 초소액으로 쪼개진 '먼지 주문'이 체결되어 장부만 오염
    equity = 800_000.0  # 허용량 = 12% × 80만 = 96,000 < min_trade 100,000
    r = evaluate_pre_trade(
        make_request(size=150_000.0, equity=equity), [], now_kst=NOW
    )
    assert r.verdict == "REJECT"
    assert r.final_size_krw == 0.0
    assert any(v.get("reason") == "resize_below_min_trade" for v in r.violations)
    assert r.token is None and r.signed is False


# ── 4. G4 섹터 비중 → REJECT ─────────────────────────────────────────────────
def test_g4_sector_limit_rejected():
    # 사고 방지: 섹터 35% 집중(기존 25%+신규 10%)이 통과되어 섹터 급락 시 동반 폭사
    holdings = [Holding("373220", 0.25 * EQUITY, sector="BATTERY")]
    r = evaluate_pre_trade(
        make_request(ticker="051910", sector="BATTERY", size=0.10 * EQUITY),
        holdings,
        now_kst=NOW,
    )
    assert r.verdict == "REJECT"
    assert r.final_size_krw == 0.0
    g4 = [v for v in r.violations if v["gate"] == "G4"]
    assert g4 and g4[0]["sector"] == "BATTERY"
    assert r.token is None


def test_g4_unknown_sector_bucket_fail_closed():
    # 사고 방지: 섹터 미상끼리는 한도 미적용으로 빠져나가 숨은 집중 리스크 누적
    holdings = [Holding("999999", 0.25 * EQUITY, sector=None)]
    r = evaluate_pre_trade(
        make_request(ticker="888888", sector=None, size=0.10 * EQUITY),
        holdings,
        now_kst=NOW,
    )
    assert r.verdict == "REJECT"
    g4 = [v for v in r.violations if v["gate"] == "G4"]
    assert g4 and g4[0]["sector"] == "UNKNOWN"


# ── 5. G5 상관 클러스터 → REJECT ─────────────────────────────────────────────
def test_g5_corr_cluster_rejected():
    # 사고 방지: ρ=0.9 종목 8%+신규 8%=실질 단일 16% 포지션이 G3을 우회해 통과
    holdings = [Holding("000660", 0.08 * EQUITY, sector="OTHER", corr_with_new=0.9)]
    r = evaluate_pre_trade(
        make_request(ticker="005930", sector="SEMI", size=0.08 * EQUITY),
        holdings,
        now_kst=NOW,
    )
    assert r.verdict == "REJECT"
    g5 = [v for v in r.violations if v["gate"] == "G5"]
    assert g5 and "000660" in g5[0]["cluster_tickers"]
    assert g5[0]["weight"] > RISK_CONFIG.max_single_weight
    assert r.token is None


# ── 6. G6 유동성 → REJECT ────────────────────────────────────────────────────
def test_g6_no_liquidity_data_fail_closed():
    # 사고 방지: ADV 데이터 결측을 '통과'로 처리해 비유동 종목에 갇히는 것
    r = evaluate_pre_trade(make_request(adv=None), [], now_kst=NOW)
    assert r.verdict == "REJECT"
    assert any(v.get("reason") == "no_liquidity_data" for v in r.violations)


def test_g6_nan_adv_fail_closed():
    # 사고 방지: NaN ADV가 비교 연산을 조용히 통과(NaN 비교=False)해 게이트 무력화
    r = evaluate_pre_trade(make_request(adv=float("nan")), [], now_kst=NOW)
    assert r.verdict == "REJECT"
    assert any(v.get("reason") == "no_liquidity_data" for v in r.violations)


def test_g6_size_exceeds_adv_cap_rejected():
    # 사고 방지: ADV 5% 초과 대량 주문이 나가 시장충격·청산불능 발생
    equity = 1_000_000_000.0  # 비중 1%라 G3/G4 무관 — G6만 분리 검증
    r = evaluate_pre_trade(
        make_request(size=10_000_000.0, equity=equity, adv=100_000_000.0),
        [],
        now_kst=NOW,
    )
    assert r.verdict == "REJECT"
    assert any(v.get("reason") == "adv_liquidity_limit" for v in r.violations)


# ── G0 입력 위생 ─────────────────────────────────────────────────────────────
def test_invalid_inputs_rejected():
    # 사고 방지: 빈 ticker/0 자본/NaN 사이즈 같은 쓰레기 입력이 판정 로직까지 흘러드는 것
    bad_requests = [
        make_request(ticker=""),
        make_request(equity=0.0),
        make_request(size=-1.0),
        make_request(size=float("nan")),
    ]
    for req in bad_requests:
        r = evaluate_pre_trade(req, [], now_kst=NOW)
        assert r.verdict == "REJECT"
        assert r.violations[0]["reason"] == "invalid_input"
        assert r.token is None


def test_invalid_holding_rejected():
    # 사고 방지: 음수 보유금액이 섹터/클러스터 합산을 줄여 리스크 과소평가
    holdings = [Holding("000660", -5_000_000.0, sector="OTHER")]
    r = evaluate_pre_trade(make_request(), holdings, now_kst=NOW)
    assert r.verdict == "REJECT"
    assert r.violations[0]["reason"] == "invalid_input"


# ── 7. 감사 로그 ─────────────────────────────────────────────────────────────
def test_log_dir_none_no_side_effects(tmp_path, monkeypatch):
    # 사고 방지: 기본 호출(순수 평가)이 몰래 파일을 만들어 운영 디렉터리 오염
    monkeypatch.chdir(tmp_path)
    r = evaluate_pre_trade(make_request(), [], now_kst=NOW)
    assert r.verdict == "PASS"
    assert list(tmp_path.rglob("*")) == []


def test_audit_log_one_line_then_append(tmp_path):
    # 사고 방지: 감사 로그가 덮어쓰기되어 이전 판정 기록 증발(append 불변식)
    log_dir = tmp_path / "gate_logs"
    evaluate_pre_trade(make_request(), [], log_dir=log_dir, now_kst=NOW)
    files = list(log_dir.glob("*.jsonl"))
    assert len(files) == 1
    assert files[0].name == "gate_log_20260611.jsonl"
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["verdict"] == "PASS"
    assert rec["request"]["ticker"] == "005930"
    assert "token" not in rec  # 토큰은 로그에 남기지 않는다(서명 비밀 유사물)

    evaluate_pre_trade(make_request(size=0.20 * EQUITY), [], log_dir=log_dir, now_kst=NOW)
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["verdict"] == "RESIZE"


def test_audit_log_write_failure_downgrades_to_reject(tmp_path):
    # 사고 방지: '게이트 로그 없는 체결' — 기록 실패한 PASS가 토큰 들고 살아남는 것
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("file blocks mkdir", encoding="utf-8")  # 파일이라 mkdir 실패 유도
    r = evaluate_pre_trade(
        make_request(), [], log_dir=blocker, hmac_key="k", now_kst=NOW
    )
    assert r.verdict == "REJECT"
    assert r.final_size_krw == 0.0
    assert r.token is None and r.signed is False
    assert any(v.get("reason") == "audit_log_write_failed" for v in r.violations)


# ── 8. 토큰 발급·검증 (★감사 로그 기록 성공 후에만 발급) ──────────────────────
def test_token_issued_and_verified(tmp_path):
    # 사고 방지: 정상 발급 토큰이 검증에서 막혀 Phase 1b 배선 전체 불통
    r = evaluate_pre_trade(make_request(), [], hmac_key="secret-key", log_dir=tmp_path, now_kst=NOW)
    assert r.verdict == "PASS"
    assert r.signed is True and r.token
    assert verify_gate_token(r, hmac_key="secret-key", now_kst=NOW) is True


def test_no_token_without_audit_log():
    # ★사고 방지(P1): log_dir 없이(순수 평가) 토큰이 발급되면 감사 추적 없는 PASS가 주문으로 간다.
    r = evaluate_pre_trade(make_request(), [], hmac_key="secret-key", now_kst=NOW)  # log_dir 없음
    assert r.verdict == "PASS"
    assert r.token is None and r.signed is False  # 무감사 → 토큰 없음
    assert verify_gate_token(r, hmac_key="secret-key", now_kst=NOW) is False


def test_resize_token_verified(tmp_path):
    # 사고 방지: RESIZE 판정 토큰이 축소 후 사이즈로 서명되지 않아 검증 불일치
    r = evaluate_pre_trade(
        make_request(size=0.20 * EQUITY), [], hmac_key="secret-key", log_dir=tmp_path, now_kst=NOW
    )
    assert r.verdict == "RESIZE" and r.signed is True
    assert verify_gate_token(r, hmac_key="secret-key", now_kst=NOW) is True


def test_tampered_token_rejected(tmp_path):
    # 사고 방지: 위조 토큰으로 게이트를 우회한 주문이 Phase 1b 검증을 통과
    r = evaluate_pre_trade(make_request(), [], hmac_key="secret-key", log_dir=tmp_path, now_kst=NOW)
    last = "0" if r.token[-1] != "0" else "1"
    r.token = r.token[:-1] + last
    assert verify_gate_token(r, hmac_key="secret-key", now_kst=NOW) is False


def test_tampered_size_rejected(tmp_path):
    # 사고 방지: 토큰 발급 후 final_size를 부풀려도 서명이 그대로 통과하는 것
    r = evaluate_pre_trade(make_request(), [], hmac_key="secret-key", log_dir=tmp_path, now_kst=NOW)
    r.final_size_krw = r.final_size_krw * 2
    assert verify_gate_token(r, hmac_key="secret-key", now_kst=NOW) is False


def test_token_expired_after_max_age(tmp_path):
    # 사고 방지: 아침 발급 토큰을 오후 장에 재사용(시장 상황 변경 후 낡은 승인)
    r = evaluate_pre_trade(make_request(), [], hmac_key="secret-key", log_dir=tmp_path, now_kst=NOW)
    later = NOW + timedelta(seconds=400)  # max_age 기본 300초 초과
    assert verify_gate_token(r, hmac_key="secret-key", now_kst=later) is False


def test_token_future_issued_rejected(tmp_path):
    # 사고 방지: 시계 역행/조작으로 '미래 발급' 토큰이 신선한 것처럼 통과
    r = evaluate_pre_trade(make_request(), [], hmac_key="secret-key", log_dir=tmp_path, now_kst=NOW)
    earlier = NOW - timedelta(seconds=10)
    assert verify_gate_token(r, hmac_key="secret-key", now_kst=earlier) is False


def test_no_hmac_key_unsigned_and_verify_false(monkeypatch, tmp_path):
    # 사고 방지: 키 미설정 환경에서 무서명 결과가 검증을 통과해 주문이 나가는 것
    monkeypatch.delenv("ORDER_INTENTS_HMAC_KEY", raising=False)
    r = evaluate_pre_trade(make_request(), [], log_dir=tmp_path, now_kst=NOW)
    assert r.verdict == "PASS"
    assert r.token is None and r.signed is False
    assert verify_gate_token(r, now_kst=NOW) is False


def test_wrong_key_rejected(tmp_path):
    # 사고 방지: 다른 키로 서명된(=다른 시스템/위조) 토큰이 우리 검증을 통과
    r = evaluate_pre_trade(make_request(), [], hmac_key="key-A", log_dir=tmp_path, now_kst=NOW)
    assert verify_gate_token(r, hmac_key="key-B", now_kst=NOW) is False


def test_naive_issued_at_rejected():
    # 사고 방지: 타임존 없는 타임스탬프로 만료 계산이 어긋난 토큰을 신뢰
    naive_ts = "2026-06-11T10:00:00"
    fake = GateResult(
        verdict="PASS",
        final_size_krw=1_000_000.0,
        original_size_krw=1_000_000.0,
        issued_at=naive_ts,
        token="deadbeef",
        signed=True,
        ticker="005930",
        nonce="n",
    )
    assert verify_gate_token(fake, hmac_key="secret-key", now_kst=NOW) is False


def test_verdict_tamper_breaks_signature(tmp_path):
    # 사고 방지: RESIZE 토큰의 verdict를 PASS로 바꿔 원래 사이즈로 체결 시도
    r = evaluate_pre_trade(
        make_request(size=0.20 * EQUITY), [], hmac_key="secret-key", log_dir=tmp_path, now_kst=NOW
    )
    assert r.verdict == "RESIZE"
    r.verdict = "PASS"
    assert verify_gate_token(r, hmac_key="secret-key", now_kst=NOW) is False


def test_token_replay_blocked_with_seen_nonces(tmp_path):
    # ★사고 방지(P1): 같은 토큰으로 max_age 내 중복 주문 — seen_nonces로 1회성 강제.
    r = evaluate_pre_trade(make_request(), [], hmac_key="secret-key", log_dir=tmp_path, now_kst=NOW)
    seen: set = set()
    assert verify_gate_token(r, hmac_key="secret-key", now_kst=NOW, seen_nonces=seen) is True
    # 같은 토큰 두 번째 검증 → replay 거부
    assert verify_gate_token(r, hmac_key="secret-key", now_kst=NOW, seen_nonces=seen) is False


def test_nonce_unique_per_evaluation(tmp_path):
    # 사고 방지: nonce가 매 평가 고유하지 않으면 replay 추적이 무력화된다.
    r1 = evaluate_pre_trade(make_request(), [], hmac_key="k", log_dir=tmp_path, now_kst=NOW)
    r2 = evaluate_pre_trade(make_request(), [], hmac_key="k", log_dir=tmp_path, now_kst=NOW)
    assert r1.nonce and r2.nonce and r1.nonce != r2.nonce


# ── 9. REJECT는 토큰 검증 절대 불가 ─────────────────────────────────────────
def test_reject_never_verifies(tmp_path):
    # 사고 방지: 차단 판정 결과 객체가 어떤 경로로든 주문 승인으로 둔갑
    r = evaluate_pre_trade(make_request(adv=None), [], hmac_key="secret-key", log_dir=tmp_path, now_kst=NOW)
    assert r.verdict == "REJECT"
    assert r.token is None
    assert verify_gate_token(r, hmac_key="secret-key", now_kst=NOW) is False
    # 토큰을 억지로 꽂아도 verdict가 REJECT면 무조건 False
    r.token = "f" * 64
    r.signed = True
    assert verify_gate_token(r, hmac_key="secret-key", now_kst=NOW) is False
