"""macro_signals 단위 테스트 (네트워크 비의존 — attach_warnings 순수함수).

핵심: ① 경고 bool 컬럼 생성·임계 로직 ② read-only(주문 어댑터 미접촉) 보장.
load_macro는 fdr FRED 네트워크 의존이라 단위테스트 제외(스크립트 smoke로 커버).
"""

import inspect
from pathlib import Path

import pandas as pd

from src.etf import macro_signals as ms
from src.etf.macro_signals import WARN_LABEL, attach_warnings


def _synthetic_macro() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=200, freq="B")
    n = len(dates)
    return pd.DataFrame(
        {
            "DGS10": [2.0 + i * 0.02 for i in range(n)],          # 꾸준한 금리 상승
            "DGS2": [1.0 + i * 0.025 for i in range(n)],
            "T10Y2Y": [0.5 - i * 0.01 for i in range(n)],          # 후반 음수(역전)
            "VIXCLS": [15 + (20 if i % 5 == 0 else 0) for i in range(n)],  # 일부 >25
            "DTWEXBGS": [100 * (1 + i * 0.001) for i in range(n)], # 달러 점진 상승
        },
        index=dates,
    )


def test_attach_warnings_creates_bool_columns():
    m = attach_warnings(_synthetic_macro())
    for col in ["vix_warn", "curve_invert_warn", "dxy_surge_warn", "rate_surge_warn"]:
        assert col in m, f"missing {col}"
        assert m[col].dtype == bool


def test_vix_threshold_logic():
    df = _synthetic_macro()
    m = attach_warnings(df)
    assert (m["vix_warn"] == (df["VIXCLS"] > 25)).all()


def test_curve_invert_logic():
    df = _synthetic_macro()
    m = attach_warnings(df)
    assert (m["curve_invert_warn"] == (df["T10Y2Y"] < 0)).all()
    # 후반부엔 역전 경고가 실제로 켜져야 함
    assert m["curve_invert_warn"].iloc[-1]


def test_warn_label_keys_present():
    m = attach_warnings(_synthetic_macro())
    for k in WARN_LABEL:
        if k == "hy_widen_warn":
            continue  # HY는 합성 데이터에 없음(2023-06~ 한계)
        assert k in m


def test_read_only_no_order_adapter():
    src = inspect.getsource(ms)
    adv = Path("scripts/research/macro_leadlag_adversarial.py").read_text(encoding="utf-8")
    for forbidden in ("KisOrderAdapter", "PaperOrderAdapter", "place_order",
                      "send_order", "order_adapter"):
        assert forbidden not in src, f"leaked in macro_signals: {forbidden}"
        assert forbidden not in adv, f"leaked in adversarial: {forbidden}"
