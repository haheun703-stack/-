"""regime_monitor 단위 테스트.

핵심: ① C60 단독 hard gate 라벨 정확성 ② 착시 보정 라벨(raw_lead_unadjusted)
③ 외국인 NOISE / gate 미사용 명시 ④ read-only(주문 어댑터 미접촉) 보장.
"""

import inspect

import pandas as pd

from src.etf import regime_monitor as rm
from src.etf.regime_monitor import (
    REGIME_BEAR,
    REGIME_BULL,
    build_regime_ledger,
    build_report,
    lead_lag_summary,
)


def _synthetic_prices() -> pd.DataFrame:
    """상승(BULL) 후 하락(BEAR)으로 60일선 교차가 발생하는 합성 가격."""
    dates = pd.date_range("2023-01-02", periods=220, freq="B")
    up = list(range(100, 200))        # 100 거래일 상승
    down = list(range(199, 79, -1))   # 120 거래일 하락
    close = (up + down)[:220]
    return pd.DataFrame({"close": close, "volume": [1000] * 220}, index=dates)


def test_regime_labels_bull_and_bear():
    rows = build_regime_ledger("TEST", prices=_synthetic_prices())
    assert rows, "ledger should not be empty"
    regimes = {r.regime for r in rows}
    assert REGIME_BULL in regimes
    assert REGIME_BEAR in regimes
    # 라벨은 종가 vs ma60 단독으로 결정 (hard gate=C60)
    for r in rows:
        expected = REGIME_BULL if r.close > r.ma60 else REGIME_BEAR
        assert r.regime == expected


def test_days_in_regime_and_change_flag():
    rows = build_regime_ledger("TEST", prices=_synthetic_prices())
    for prev, cur in zip(rows, rows[1:]):
        if cur.regime == prev.regime:
            assert cur.days_in_regime == prev.days_in_regime + 1
            assert cur.regime_change is False
        else:
            assert cur.days_in_regime == 1
            assert cur.regime_change is True


def test_report_caution_labels_present():
    rows = build_regime_ledger("TEST", prices=_synthetic_prices())
    rep = build_report("TEST", rows)
    # 착시 보정: 새 키 존재, 옛 키 부재
    assert "raw_lead_unadjusted" in rep
    assert "leadlag_avg" not in rep
    assert "_caution" in rep["raw_lead_unadjusted"]
    # hard gate = C60 단독 명시
    assert rep["hard_gate"].startswith("C60")
    # 외국인 = NOISE / gate 제외 명시
    assert "NOISE" in rep["observation_gate_status"]["foreign_net"]


def test_leadlag_keys_renamed_to_raw_unadjusted():
    rows = build_regime_ledger("TEST", prices=_synthetic_prices())
    ll = lead_lag_summary(rows)
    if ll:
        keys = set(ll[0].keys())
        assert any("raw_lead_unadjusted" in k for k in keys)
        assert "vol_cluster_lead_days" not in keys  # 옛 착시 키 제거 확인


def test_read_only_no_order_adapter():
    """shadow 안전: 주문 어댑터/주문 호출 심볼이 소스에 없어야 한다."""
    src = inspect.getsource(rm)
    for forbidden in (
        "KisOrderAdapter",
        "PaperOrderAdapter",
        "place_order",
        "send_order",
        "order_adapter",
    ):
        assert forbidden not in src, f"forbidden symbol leaked: {forbidden}"
