"""RISK_ENGINE Phase 1b — 게이트 통행증 강제 + 우회불가 증명 (체크리스트 C).

검증 대상 = KisOrderAdapter._enforce_gate_token / _guard (BUY 최종 관문).

핵심 불변식:
  1. REAL(_is_mock=False) BUY는 검증된 게이트 통행증 없이는 PermissionError(차단).
  2. 위조/만료/replay/종목불일치/사이즈초과 토큰 전부 차단(fail-closed).
  3. mock(_is_mock=True) BUY는 차단하지 않되 무토큰이면 경고만(드라이런 증거).
  4. SELL은 토큰 면제(킬스위치 매도 지속 불변식 보존).
  5. 우회 경로 부재 — broker.create_*_buy_order는 _guard 경유 진입점 안에서만 호출.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from risk.config import KST
from risk.pre_trade_gate import GateRequest, GateResult, evaluate_pre_trade

_KEY = "test-hmac-key-1b"


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
def _signed_token(
    ticker: str, size_krw: float, tmp_path: Path, now_kst: datetime | None = None,
) -> GateResult:
    """감사 로그 기록 후 서명된 PASS 통행증 발급(실 발급 경로와 동일)."""
    req = GateRequest(
        ticker=ticker, sector="ETF",
        proposed_size_krw=size_krw, equity_krw=size_krw * 200,
        adv20_krw=size_krw * 1000,  # adv*0.05 = size*50 ≫ size → G6 통과
    )
    res = evaluate_pre_trade(req, [], log_dir=tmp_path, hmac_key=_KEY, now_kst=now_kst)
    assert res.verdict == "PASS" and res.token and res.signed, res
    return res


@pytest.fixture
def real_adapter(monkeypatch, tmp_path):
    """REAL(_is_mock=False) 어댑터 — mojito/네트워크 차단. ORDER_INTENTS_HMAC_KEY=_KEY."""
    monkeypatch.setattr(
        "src.utils.trade_runtime_safety.KILL_SWITCH_PATHS",
        (tmp_path / "missing_KILL_SWITCH", tmp_path / "missing_kill_switch.flag"),
    )
    monkeypatch.setattr("mojito.KoreaInvestment", MagicMock())
    monkeypatch.setenv("MODEL", "REAL")
    monkeypatch.setenv("ORDER_INTENTS_HMAC_KEY", _KEY)
    from src.adapters.kis_order_adapter import KisOrderAdapter
    adp = KisOrderAdapter()
    assert adp._is_mock is False
    adp.fetch_current_price = MagicMock(return_value={"current_price": 10000})
    yield adp


@pytest.fixture
def mock_adapter(monkeypatch, tmp_path):
    """mock(_is_mock=True) 어댑터 — MODEL!=REAL."""
    monkeypatch.setattr(
        "src.utils.trade_runtime_safety.KILL_SWITCH_PATHS",
        (tmp_path / "missing_KILL_SWITCH", tmp_path / "missing_kill_switch.flag"),
    )
    monkeypatch.setattr("mojito.KoreaInvestment", MagicMock())
    monkeypatch.setenv("MODEL", "MOCK")
    monkeypatch.setenv("ORDER_INTENTS_HMAC_KEY", _KEY)
    from src.adapters.kis_order_adapter import KisOrderAdapter
    adp = KisOrderAdapter()
    assert adp._is_mock is True
    adp.fetch_current_price = MagicMock(return_value={"current_price": 10000})
    yield adp


# ── 1. REAL BUY 무토큰 → 차단 ─────────────────────────────────────────────────
def test_real_buy_without_token_blocked(real_adapter):
    with pytest.raises(PermissionError, match=r"\[GATE\].*통행증 검증 실패"):
        real_adapter._enforce_gate_token("487240", 1, 10000, None)


# ── 2. REAL BUY 유효 토큰 → 통과 ──────────────────────────────────────────────
def test_real_buy_with_valid_token_passes(real_adapter, tmp_path):
    tok = _signed_token("487240", 100000, tmp_path)
    # 주문 금액(1주 × 10000 = 10000) ≤ 승인 사이즈(100000) → 통과(예외 없음)
    real_adapter._enforce_gate_token("487240", 1, 10000, tok)


# ── 3. 위조 서명 토큰 → 차단 ──────────────────────────────────────────────────
def test_real_buy_forged_token_blocked(real_adapter, tmp_path):
    tok = _signed_token("487240", 100000, tmp_path)
    tok.token = "deadbeef" * 8  # 서명 위조
    with pytest.raises(PermissionError, match=r"\[GATE\]"):
        real_adapter._enforce_gate_token("487240", 1, 10000, tok)


# ── 4. 만료 토큰(5분 초과) → 차단 ─────────────────────────────────────────────
def test_real_buy_expired_token_blocked(real_adapter, tmp_path):
    past = datetime.now(KST) - timedelta(seconds=600)  # 10분 전 발급
    tok = _signed_token("487240", 100000, tmp_path, now_kst=past)
    with pytest.raises(PermissionError, match=r"\[GATE\]"):
        real_adapter._enforce_gate_token("487240", 1, 10000, tok)


# ── 5. replay(같은 토큰 재사용) → 2번째 차단 ──────────────────────────────────
def test_real_buy_replay_blocked(real_adapter, tmp_path):
    tok = _signed_token("487240", 100000, tmp_path)
    real_adapter._enforce_gate_token("487240", 1, 10000, tok)  # 1회차 통과(nonce 소비)
    with pytest.raises(PermissionError, match=r"\[GATE\]"):
        real_adapter._enforce_gate_token("487240", 1, 10000, tok)  # 2회차 차단


# ── 6. 종목 불일치(다른 종목 통행증 재사용) → 차단 ───────────────────────────
def test_real_buy_ticker_mismatch_blocked(real_adapter, tmp_path):
    tok = _signed_token("487240", 100000, tmp_path)  # 487240용 토큰
    with pytest.raises(PermissionError, match=r"\[GATE\]"):
        real_adapter._enforce_gate_token("395160", 1, 10000, tok)  # 다른 종목에 사용


# ── 7. 주문 금액 > 승인 사이즈 → 차단 ────────────────────────────────────────
def test_real_buy_oversize_blocked(real_adapter, tmp_path):
    tok = _signed_token("487240", 100000, tmp_path)  # 승인 100,000원
    # 20주 × 10,000 = 200,000 > 100,000 → 사이즈 초과
    with pytest.raises(PermissionError, match=r"승인 사이즈"):
        real_adapter._enforce_gate_token("487240", 20, 10000, tok)


# ── 8. REJECT verdict 토큰(미서명) → 차단 ────────────────────────────────────
def test_real_buy_reject_verdict_blocked(real_adapter, tmp_path):
    # G6 유동성 데이터 없음 → REJECT, 토큰 미발급(signed=False)
    req = GateRequest(ticker="487240", sector="ETF",
                      proposed_size_krw=100000, equity_krw=10_000_000, adv20_krw=None)
    res = evaluate_pre_trade(req, [], log_dir=tmp_path, hmac_key=_KEY)
    assert res.verdict == "REJECT" and not res.signed
    with pytest.raises(PermissionError, match=r"\[GATE\]"):
        real_adapter._enforce_gate_token("487240", 1, 10000, res)


# ── 9. mock BUY 무토큰 → 차단 안 함 + 경고 ───────────────────────────────────
def test_mock_buy_without_token_warns_not_blocks(mock_adapter, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        mock_adapter._enforce_gate_token("487240", 1, 10000, None)  # 예외 없어야 함
    assert any("GATE-DRYRUN" in r.message for r in caplog.records)


# ── 10. mock BUY 유효 토큰 → 경고 없음 ───────────────────────────────────────
def test_mock_buy_with_valid_token_no_warn(mock_adapter, caplog, tmp_path):
    import logging
    tok = _signed_token("487240", 100000, tmp_path)
    with caplog.at_level(logging.WARNING):
        mock_adapter._enforce_gate_token("487240", 1, 10000, tok)
    assert not any("GATE-DRYRUN" in r.message for r in caplog.records)


# ── 11. SELL은 _enforce를 호출하지 않음(_guard 구조 확인) ─────────────────────
def test_guard_calls_enforce_only_for_buy():
    """소스 구조 불변식: _guard는 side=='BUY'에서만 _enforce_gate_token을 호출한다."""
    src = Path("src/adapters/kis_order_adapter.py").read_text(encoding="utf-8")
    # _enforce_gate_token 호출은 정확히 'if side == "BUY":' 가드 아래에서만 등장
    calls = [m.start() for m in re.finditer(r"self\._enforce_gate_token\(", src)]
    defs = [m.start() for m in re.finditer(r"def _enforce_gate_token\(", src)]
    # 호출 1곳(=_guard 내부) + 정의 1곳
    assert len(defs) == 1
    assert len(calls) == 1
    guard_buy = src.rfind('if side == "BUY":', 0, calls[0])
    assert guard_buy != -1 and (calls[0] - guard_buy) < 200, "BUY 가드 직하 호출이어야 함"


# ── 12. 우회 경로 부재 — create_*_buy_order는 _guard 경유 진입점에서만 ────────
def test_no_buy_order_bypass_outside_guard():
    """broker.create_*_buy_order 호출이 buy_limit/buy_market(=_guard 경유) 안에서만 등장."""
    src = Path("src/adapters/kis_order_adapter.py").read_text(encoding="utf-8")
    buy_order_calls = re.findall(r"\.create_(?:limit|market)_buy_order\(", src)
    # 매수 주문 raw 호출은 정확히 2곳(지정가 1 + 시장가 1)
    assert len(buy_order_calls) == 2, f"예상 2곳, 실제 {len(buy_order_calls)}"
    # 두 호출 모두 _guard 호출 이후에 위치(같은 함수 내 가드 선행)
    for m in re.finditer(r"\.create_(?:limit|market)_buy_order\(", src):
        preceding = src[:m.start()]
        assert "self._guard(" in preceding, "raw 매수 주문 앞에 _guard 선행 필수"
