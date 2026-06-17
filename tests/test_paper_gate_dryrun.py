"""E-0 — 페이퍼 게이트 드라이런 배선 검증.

검증 포인트:
  1. PaperBalancePort.fetch_balance() 모양(ok/available_cash/holdings) 정합.
  2. run_paper_gate_dryrun이 gate_log_*.jsonl에 GATE-DRYRUN 감사로그 1줄을 남긴다(= E 증거).
  3. enforce=False라 verdict와 무관하게 raise 없이 결과/None 반환(페이퍼 흐름 무영향).
  4. graceful — 깨진 pf/잔고0에서도 페이퍼 엔진을 깨뜨리지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.use_cases.paper_gate import PaperBalancePort, run_paper_gate_dryrun


# ── 1. 잔고 어댑터 모양 ──────────────────────────────────────────────
def test_paper_balance_port_shape():
    pf = {
        "capital": 10_000_000,
        "positions": {
            "005930": {"qty": 10, "avg_price": 80_000},
            "000660": {"qty": 5, "avg_price": 200_000},
        },
    }
    bal = PaperBalancePort(pf).fetch_balance()
    assert bal["ok"] is True
    assert bal["available_cash"] == 10_000_000
    tickers = {h["ticker"] for h in bal["holdings"]}
    assert tickers == {"005930", "000660"}
    # eval_amount = qty * avg_price
    by = {h["ticker"]: h["eval_amount"] for h in bal["holdings"]}
    assert by["005930"] == 10 * 80_000
    assert by["000660"] == 5 * 200_000


def test_paper_balance_port_is_mock_true():
    # enforce 도출이 False가 되도록 _is_mock=True (페이퍼는 절대 차단 안 함)
    assert PaperBalancePort({})._is_mock is True


def test_paper_balance_port_skips_malformed_position():
    pf = {"capital": 1_000_000, "positions": {"X": None, "005930": {"qty": 1, "avg_price": 70_000}}}
    bal = PaperBalancePort(pf).fetch_balance()
    assert {h["ticker"] for h in bal["holdings"]} == {"005930"}


# ── 2. 드라이런이 감사로그를 남긴다 (E 증거) ──────────────────────────
def test_dryrun_writes_audit_log(tmp_path):
    pf = {"capital": 10_000_000, "positions": {}}
    log_dir = tmp_path / "gate_logs_paper"
    # 005930: 로컬 parquet 보유 종목 → adv20/VaR 실데이터(없어도 G6 REJECT가 로그됨)
    run_paper_gate_dryrun(pf, "005930", 80_000.0, 10, log_dir=log_dir)

    logs = list(Path(log_dir).glob("gate_log_*.jsonl"))
    assert logs, "감사 로그 파일이 생성돼야 한다(잔고 통과 시 verdict 무관 1줄 기록)"
    lines = [ln for ln in logs[0].read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 1
    rec = json.loads(lines[0])  # 유효 JSON(NaN/Inf 없는 RFC 준수)
    assert rec["request"]["ticker"] == "005930"
    assert rec["verdict"] in ("PASS", "RESIZE", "REJECT")


def test_dryrun_never_raises_and_returns():
    # 잔고 충분 → 결과(GateResult or None) 반환, raise 없음
    pf = {"capital": 10_000_000, "positions": {}}
    # log_dir=None이면 PAPER_GATE_LOG_DIR(실제 경로)에 쓰므로 테스트는 기록 검증을 위에서 끝냄.
    # 여기선 '예외 없이 반환'만 본다(임시 디렉토리 사용).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = run_paper_gate_dryrun(pf, "005930", 80_000.0, 10, log_dir=td)
    # PASS/RESIZE면 GateResult, REJECT/미가용이면 None — 어느 쪽이든 raise 없으면 통과
    assert out is None or hasattr(out, "verdict")


# ── 3·4. graceful — 깨진 입력에도 페이퍼 엔진 무손상 ──────────────────
@pytest.mark.parametrize("bad_pf", [None, {}, {"capital": 0, "positions": {}}, {"positions": None}])
def test_dryrun_graceful_on_bad_pf(bad_pf, tmp_path):
    # 어떤 깨진 pf에서도 raise 없이 None/결과 반환
    out = run_paper_gate_dryrun(bad_pf, "005930", 80_000.0, 10, log_dir=tmp_path / "g")
    assert out is None or hasattr(out, "verdict")


def test_dryrun_graceful_on_bad_inputs(tmp_path):
    pf = {"capital": 10_000_000, "positions": {}}
    # 단가 0·수량 0 → gate_check가 입력부적격으로 비차단(enforce=False) None, raise 없음
    assert run_paper_gate_dryrun(pf, "005930", 0.0, 0, log_dir=tmp_path / "g") is None
    # 존재하지 않는 종목 → OHLCV 없음 → G6 REJECT 로그(또는 None), raise 없음
    out = run_paper_gate_dryrun(pf, "999999", 1_000.0, 1, log_dir=tmp_path / "g2")
    assert out is None or hasattr(out, "verdict")
