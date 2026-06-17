"""E-0 — 페이퍼 트레이딩 게이트 드라이런 배선.

실제 도는 페이퍼 엔진(``scripts/paper_trading_unified.py``)의 가상매수를 리스크엔진 게이트
(``gate_wiring.gate_check``)에 통과시켜 **GATE-DRYRUN 감사 로그**를 남긴다.
unfreeze 체크리스트 §2-E "페이퍼 20거래일" 카운트의 증거(``gate_log_*.jsonl``)가 여기서
쌓이기 시작한다.

설계 원칙(unfreeze-checklist.md §2-E line 99 충족):

* ★**드라이런(enforce=False)** — 게이트 verdict(PASS/RESIZE/REJECT)를 *기록만* 하고
  가상체결은 호출부에서 그대로 진행한다. 페이퍼의 목적은 게이트가 여러 시장상황서
  오작동·과차단 없이 도는지 **관측**하는 것이지 차단이 아니다.
* ★**실매매 무접촉** — gate_check 6호출처(smart_entry·adaptive_*·limit_up·live_trading·
  telegram)는 손대지 않는다. 이 모듈은 페이퍼 엔진에서만 호출된다.
* ★**graceful** — 게이트 import/호출 실패가 페이퍼 엔진을 절대 깨뜨리지 않는다(전부 흡수).
* ★**로그 분리** — 실매매 게이트 로그(``data/risk/gate_logs``)와 섞이지 않도록 페이퍼
  드라이런 증거는 ``data/risk/gate_logs_paper``에 따로 쌓는다.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 실매매 게이트 로그와 분리 — 페이퍼 드라이런(E) 증거 전용.
PAPER_GATE_LOG_DIR = Path("data/risk/gate_logs_paper")


class PaperBalancePort:
    """페이퍼 포트폴리오 dict → ``gate_wiring``가 기대하는 BalancePort 모양 어댑터.

    ``build_gate_result``는 ``balance_port.fetch_balance()``가 반환하는
    ``{"ok", "available_cash", "holdings": [{"ticker", "eval_amount"}]}``를 읽어 equity를
    합성한다. ``_is_mock=True``라 ``gate_check``의 enforce 도출이 항상 False(절대 비차단) —
    호출부에서 ``enforce=False``도 명시해 이중으로 보장한다.
    """

    _is_mock = True  # enforce 도출 → False (페이퍼는 절대 차단 안 함)

    def __init__(self, pf: dict | None):
        self._pf = pf or {}

    def fetch_balance(self) -> dict:
        pf = self._pf or {}
        cash = float(pf.get("capital", 0) or 0)
        holdings = []
        for ticker, pos in (pf.get("positions") or {}).items():
            if not isinstance(pos, dict):
                continue
            qty = float(pos.get("qty", 0) or 0)
            avg = float(pos.get("avg_price", 0) or 0)
            holdings.append({"ticker": str(ticker), "eval_amount": qty * avg})
        return {"ok": True, "available_cash": cash, "holdings": holdings}


def run_paper_gate_dryrun(pf, ticker, unit_price, qty, *, log_dir=None):
    """페이퍼 가상매수 1건을 게이트 드라이런 평가 — GATE-DRYRUN 로그 1줄 남기고 결과 반환.

    ★절대 raise하지 않는다(페이퍼 엔진 보호). 게이트 미가용/예외 시 ``None``.
    enforce=False라 verdict와 무관하게 페이퍼 흐름은 호출부에서 그대로 진행된다(관측 전용).
    감사 로그는 ``build_gate_result`` 내부(``_finalize`` → ``_append_audit_log``)에서 verdict와
    무관하게 잔고 통과 시 1줄 기록된다 — 이 함수의 반환값(REJECT 시 None)과 로그 기록은 독립.

    Args:
        pf: 페이퍼 포트폴리오 dict(``capital``, ``positions``).
        ticker: 매수 후보 종목코드.
        unit_price: 가상 체결 단가(슬리피지·수수료 반영된 buy_price).
        qty: 가상 매수 수량.
        log_dir: 감사 로그 디렉토리(기본 ``PAPER_GATE_LOG_DIR``).

    Returns:
        ``GateResult`` (PASS/RESIZE) 또는 ``None``(REJECT·미가용·예외).
    """
    try:
        from src.use_cases.gate_wiring import gate_check
    except Exception as exc:  # noqa: BLE001 — 게이트 미배포 환경: 드라이런 스킵
        logger.debug("[paper_gate] gate_wiring import 불가 → 드라이런 스킵: %s", exc)
        return None

    ld = Path(log_dir) if log_dir is not None else PAPER_GATE_LOG_DIR
    try:
        port = PaperBalancePort(pf)
        _proceed, gate, _qty = gate_check(
            port,
            str(ticker),
            float(unit_price),
            int(qty),
            enforce=False,  # ★드라이런 — 절대 차단 안 함
            log_dir=ld,
        )
        if gate is not None:
            logger.info(
                "[paper_gate] %s GATE-DRYRUN verdict=%s",
                ticker,
                getattr(gate, "verdict", "?"),
            )
        return gate
    except Exception as exc:  # noqa: BLE001 — 페이퍼 엔진 절대 무손상
        logger.warning("[paper_gate] %s 드라이런 예외(무시): %s", ticker, exc)
        return None
