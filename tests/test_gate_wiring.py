"""RISK_ENGINE C-ii part2 — gate_wiring.build_gate_result (라이브 매수 게이트 발급 유일 경로).

보강 3종 + happy/RESIZE + sole-caller grep.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from risk.config import KST
from src.use_cases.gate_wiring import build_gate_result, gate_check

_KEY = "gate-wiring-test-key"
_ASOF = date(2026, 6, 10)  # 수요일(거래일)


class FakeBalance:
    def __init__(self, payload):
        self._p = payload

    def fetch_balance(self):
        return self._p


def _ok_balance(cash=10_000_000, holdings=None):
    return {"ok": True, "available_cash": cash, "holdings": holdings or [], "total_eval": 0}


def _fresh_ohlcv(last: date = _ASOF, close=10000, volume=2000, rows=30):
    idx = pd.bdate_range(end=pd.Timestamp(last), periods=rows)
    return pd.DataFrame({"close": [close] * rows, "volume": [volume] * rows}, index=idx)


def _loader(df):
    return lambda ticker: df


def _sector(name="반도체"):
    return lambda ticker: name


def _build(**kw):
    defaults = dict(
        ticker="005930", proposed_size_krw=500_000,
        balance_port=FakeBalance(_ok_balance()),
        ohlcv_loader=_loader(_fresh_ohlcv()),
        sector_resolver=_sector(),
        hmac_key=_KEY, as_of_date=_ASOF,
        now_kst=datetime(2026, 6, 10, 10, 0, tzinfo=KST),
    )
    defaults.update(kw)
    return build_gate_result(**defaults)


# ── R1: 잔고 조회 실패 → REJECT(balance_unavailable) ─────────────────────────
def test_balance_unavailable_rejects(tmp_path):
    r = _build(balance_port=FakeBalance({"ok": False, "available_cash": 0, "holdings": []}),
               log_dir=tmp_path)
    assert r.verdict == "REJECT"
    assert r.violations[0]["reason"] == "balance_unavailable"
    assert r.token is None  # 토큰 없음 → 주문 불가


def test_balance_exception_rejects(tmp_path):
    class Boom:
        def fetch_balance(self):
            raise RuntimeError("network down")
    r = _build(balance_port=Boom(), log_dir=tmp_path)
    assert r.verdict == "REJECT" and r.violations[0]["reason"] == "balance_fetch_exception"


def test_equity_non_positive_rejects(tmp_path):
    r = _build(balance_port=FakeBalance(_ok_balance(cash=0)), log_dir=tmp_path)
    assert r.verdict == "REJECT" and r.violations[0]["reason"] == "equity_non_positive"


# ── R2: adv20 없음/stale → G6 fail-closed REJECT ─────────────────────────────
def test_adv_missing_rejects(tmp_path):
    r = _build(ohlcv_loader=lambda t: None, log_dir=tmp_path)
    assert r.verdict == "REJECT"
    assert any(v.get("gate") == "G6" for v in r.violations)


def test_adv_stale_rejects(tmp_path):
    # 마지막 봉이 기준일보다 ~9거래일 뒤처짐 → stale → G6 REJECT
    stale_df = _fresh_ohlcv(last=_ASOF - timedelta(days=13))
    r = _build(ohlcv_loader=_loader(stale_df), log_dir=tmp_path)
    assert r.verdict == "REJECT"
    assert any(v.get("gate") == "G6" for v in r.violations)


# ── happy path: PASS + 토큰 발급 ─────────────────────────────────────────────
def test_happy_path_pass_with_token(tmp_path):
    r = _build(log_dir=tmp_path)
    assert r.verdict == "PASS"
    assert r.token and r.signed
    assert r.ticker == "005930"
    assert r.final_size_krw == 500_000


# ── RESIZE: G3 단일비중 초과 → 축소 + 토큰 ───────────────────────────────────
def test_resize_shrinks_and_tokenizes(tmp_path):
    # equity 1,000,000 · proposed 200,000 → G3 0.20>0.12(RESIZE), G4 0.20<0.30(ok)
    r = _build(
        proposed_size_krw=200_000,
        balance_port=FakeBalance(_ok_balance(cash=1_000_000)),
        ohlcv_loader=_loader(_fresh_ohlcv(volume=5000)),  # adv 충분
        log_dir=tmp_path,
    )
    assert r.verdict == "RESIZE"
    assert r.final_size_krw == pytest.approx(120_000)  # 0.12 × 1,000,000
    assert r.token and r.signed


# ── 토큰 발급은 모드 무관(양 모드 공통) — 헬퍼에 mode 개념 없음 확인 ─────────
def test_issuance_is_mode_agnostic(tmp_path):
    # 같은 입력은 mode와 무관하게 동일 발급(헬퍼는 강제가 아니라 발급만) → PASS 토큰
    r = _build(log_dir=tmp_path)
    assert r.verdict == "PASS" and r.signed


# ── gate_check 래퍼 (REAL enforce=True): proceed/gate/qty 3튜플 ───────────────
def _gc(balance, ticker, price, qty, **kw):
    kw.setdefault("hmac_key", _KEY)
    kw.setdefault("as_of_date", _ASOF)
    kw.setdefault("now_kst", datetime(2026, 6, 10, 10, 0, tzinfo=KST))
    return gate_check(balance, ticker, price, qty, **kw)


def test_gate_check_pass(tmp_path):
    proceed, g, q = _gc(FakeBalance(_ok_balance()), "005930", 10000, 10, enforce=True,
                        ohlcv_loader=_loader(_fresh_ohlcv()), sector_resolver=_sector(), log_dir=tmp_path)
    assert proceed and g is not None and g.verdict == "PASS" and q == 10


def test_gate_check_reject_blocks_when_enforce(tmp_path):
    proceed, g, q = _gc(FakeBalance({"ok": False, "available_cash": 0, "holdings": []}),
                        "005930", 10000, 10, enforce=True,
                        ohlcv_loader=_loader(_fresh_ohlcv()), sector_resolver=_sector(), log_dir=tmp_path)
    assert proceed is False and g is None and q == 0


def test_gate_check_resize_shrinks_qty(tmp_path):
    # equity 1,000,000 · 단가 10,000 × 20주 = 200,000 → G3 0.20>0.12 RESIZE→120,000→12주
    proceed, g, q = _gc(FakeBalance(_ok_balance(cash=1_000_000)), "005930", 10000, 20, enforce=True,
                        ohlcv_loader=_loader(_fresh_ohlcv(volume=5000)), sector_resolver=_sector(), log_dir=tmp_path)
    assert proceed and g is not None and g.verdict == "RESIZE" and q == 12


def test_gate_check_bad_input_fail_closed_when_enforce(tmp_path):
    assert _gc(FakeBalance(_ok_balance()), "005930", 0, 10, enforce=True, log_dir=tmp_path) == (False, None, 0)
    assert _gc(FakeBalance(_ok_balance()), "005930", 10000, 0, enforce=True, log_dir=tmp_path) == (False, None, 0)


# ── mock/paper(enforce=False): 절대 차단 안 함 ───────────────────────────────
def test_gate_check_mock_never_blocks_on_reject(tmp_path):
    # 잔고 조회 실패라도 enforce=False면 proceed=True(원수량) — 어댑터가 토큰 무시/경고
    proceed, g, q = _gc(FakeBalance({"ok": False, "available_cash": 0, "holdings": []}),
                        "005930", 10000, 10, enforce=False, log_dir=tmp_path)
    assert proceed is True and g is None and q == 10


def test_gate_check_enforce_derivation():
    from src.use_cases.gate_wiring import _derive_enforce

    class RealAdapter:  # KisOrderAdapter REAL 모사
        _is_mock = False

    class MockAdapter:  # KisOrderAdapter 모의 모사
        _is_mock = True

    class PaperLike:    # PaperOrderAdapter(_is_mock 없음)
        pass

    assert _derive_enforce(RealAdapter()) is True     # REAL만 강제
    assert _derive_enforce(MockAdapter()) is False
    assert _derive_enforce(PaperLike()) is False
    assert _derive_enforce(MagicMock()) is False       # 테스트 mock도 비강제


# ── ★C-ii-b 커버리지: 모든 REAL BUY 호출처는 gate_check 경유 또는 문서화 예외 ──
def test_all_real_buy_callers_gated():
    """src/의 .buy_limit/.buy_market 호출처는 gate_check를 import하거나 문서화된 예외여야 한다.

    새 REAL 매수 호출처가 게이트 없이 추가되면 이 테스트가 실패 → 배선 또는 예외 등록 강제.
    어댑터 하드 차단(1b-i)이 백스톱이라 미배선도 REAL에선 fail-closed(안전)지만, 이 테스트는
    가용성 갭(REAL에서 매수 불능)을 조기 검출한다.
    """
    exceptions = {
        "src/adapters/kis_order_adapter.py",      # 어댑터 정의 = 강제 지점(1b-i)
        "src/adapters/paper_order_adapter.py",    # 페이퍼 시뮬(토큰 무시)
        "src/use_cases/paper_mirror.py",          # 페이퍼 미러(PaperOrderAdapter)
        "src/strategies/chart_hero_executor.py",  # 휴면(D확인) — 재배선 시 게이트
        "src/split_order.py",                     # production 호출처 0(orphaned)
    }
    call_re = re.compile(r"\.(buy_limit|buy_market)\s*\(")
    ungated = []
    for p in Path("src").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        if not call_re.search(text):
            continue
        rel = p.as_posix()
        if rel in exceptions:
            continue
        if "gate_check" not in text:
            ungated.append(rel)
    assert not ungated, (
        f"게이트 미배선 REAL BUY 호출처: {ungated} — gate_check 배선 또는 예외 등록 필요"
    )


# ── ★sole issuance: production에서 evaluate_pre_trade 직접호출은 gate_wiring뿐 ─
def test_evaluate_pre_trade_sole_caller():
    src_root = Path("src")
    offenders = []
    for p in src_root.rglob("*.py"):
        if p.name == "gate_wiring.py":
            continue
        text = p.read_text(encoding="utf-8")
        # 호출 패턴 `evaluate_pre_trade(` 검출(import 라인 'import evaluate_pre_trade'는 제외)
        for m in re.finditer(r"evaluate_pre_trade\s*\(", text):
            offenders.append(f"{p}:{text[:m.start()].count(chr(10))+1}")
    assert not offenders, f"evaluate_pre_trade 직접호출은 gate_wiring만 허용. 위반: {offenders}"
