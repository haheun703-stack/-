"""fetch_balance ok 플래그 — M-4 결함수정 가드 (5/31).

잔고 조회 실패 시 available_cash=0을 '진짜 0'과 구분(ok=False)하여,
run_monitor가 그 0을 총손실로 오판해 전 종목을 긴급청산하는 사고를 방지한다.
"""
from src.adapters.kis_order_adapter import KisOrderAdapter


def test_kis_balance_failure_sets_ok_false():
    """broker 조회 예외 → ok=False (available_cash=0이지만 진짜 0 아님)."""
    class _RaisingBroker:
        def fetch_balance(self):
            raise RuntimeError("API down")

    class _Shim:
        broker = _RaisingBroker()

    bal = KisOrderAdapter.fetch_balance(_Shim())
    assert bal["ok"] is False
    assert bal["available_cash"] == 0


def test_kis_balance_success_sets_ok_true():
    """정상 조회 → ok=True + 실제 예수금."""
    class _OkBroker:
        def fetch_balance(self):
            return {
                "output1": [],
                "output2": [{
                    "dnca_tot_amt": "1000000",
                    "tot_evlu_amt": "0",
                    "evlu_pfls_smtl_amt": "0",
                }],
            }

    class _Shim:
        broker = _OkBroker()

    bal = KisOrderAdapter.fetch_balance(_Shim())
    assert bal["ok"] is True
    assert bal["available_cash"] == 1_000_000
