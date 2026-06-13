"""RISK_ENGINE C-ii part2 — gate_wiring.build_gate_result (라이브 매수 게이트 발급 유일 경로).

보강 3종 + happy/RESIZE + sole-caller grep.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path

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


# ── gate_check 래퍼: REJECT→(None,0) / RESIZE→축소 / 입력불가→(None,0) ─────
def test_gate_check_pass(tmp_path):
    g, q = gate_check(FakeBalance(_ok_balance()), "005930", 10000, 10,
                      ohlcv_loader=_loader(_fresh_ohlcv()), sector_resolver=_sector(),
                      hmac_key=_KEY, as_of_date=_ASOF, log_dir=tmp_path,
                      now_kst=datetime(2026, 6, 10, 10, 0, tzinfo=KST))
    assert g is not None and g.verdict == "PASS" and q == 10


def test_gate_check_reject_returns_none(tmp_path):
    g, q = gate_check(FakeBalance({"ok": False, "available_cash": 0, "holdings": []}),
                      "005930", 10000, 10, ohlcv_loader=_loader(_fresh_ohlcv()),
                      sector_resolver=_sector(), hmac_key=_KEY, as_of_date=_ASOF, log_dir=tmp_path)
    assert g is None and q == 0


def test_gate_check_resize_shrinks_qty(tmp_path):
    # equity 1,000,000 · 단가 10,000 × 20주 = 200,000 → G3 0.20>0.12 RESIZE→120,000→12주
    g, q = gate_check(FakeBalance(_ok_balance(cash=1_000_000)), "005930", 10000, 20,
                      ohlcv_loader=_loader(_fresh_ohlcv(volume=5000)), sector_resolver=_sector(),
                      hmac_key=_KEY, as_of_date=_ASOF, log_dir=tmp_path,
                      now_kst=datetime(2026, 6, 10, 10, 0, tzinfo=KST))
    assert g is not None and g.verdict == "RESIZE" and q == 12


def test_gate_check_bad_input_fail_closed(tmp_path):
    assert gate_check(FakeBalance(_ok_balance()), "005930", 0, 10, log_dir=tmp_path) == (None, 0)
    assert gate_check(FakeBalance(_ok_balance()), "005930", 10000, 0, log_dir=tmp_path) == (None, 0)


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
