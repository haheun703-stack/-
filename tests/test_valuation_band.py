"""밸류밴드 대시보드 적재 헬퍼 테스트 — 순수 함수만(네트워크 I/O 없음).

verdict 표준화 / source / checkup pos 폴백 / dashboard 행 변환을 박제.
관측 적재 전용 — 매매로직 무관.
"""
from __future__ import annotations

from src.use_cases.valuation_band import (
    STANDARD_VERDICTS,
    ValuationSnapshot,
    apply_checkup,
    source_of,
    to_dashboard_row,
    verdict_category,
)


def _snap(**kw) -> ValuationSnapshot:
    base = dict(
        market="US", ticker="META", name="Meta", price=300.0,
        per=20.0, fwd_per=18.0, pbr=5.0, roe=33.0, pos_52w=9.0,
        debt_to_equity=20.0, fcf=25_000.0, net_debt=-10_000.0, market_cap=1_000_000.0,
    )
    base.update(kw)
    return ValuationSnapshot(**base)


# ── verdict_category: 상세 → 표준 5종 ──────────────────────────────

def test_verdict_category_maps_all_to_standard_set():
    details = [
        "데이터부족", "가치함정·FCF음수", "가치함정·이익↓", "고ROE·이미오름",
        "저점후보·이익↑", "저점관찰", "관망",
    ]
    for d in details:
        assert verdict_category(d) in STANDARD_VERDICTS


def test_verdict_category_value_trap():
    assert verdict_category("가치함정·FCF음수") == "가치함정"
    assert verdict_category("가치함정·이익↓") == "가치함정"


def test_verdict_category_jade_and_watch_merge_to_jade():
    assert verdict_category("저점후보·이익↑") == "저점후보"
    assert verdict_category("저점관찰") == "저점후보"


def test_verdict_category_risen_and_data_shortage():
    assert verdict_category("고ROE·이미오름") == "이미오름"
    assert verdict_category("데이터부족") == "데이터부족"
    assert verdict_category("관망") == "관망"


# ── source_of ─────────────────────────────────────────────────────

def test_source_of_us_is_yfinance():
    assert source_of(_snap(market="US")) == "yfinance"


def test_source_of_kr_is_naver():
    assert source_of(_snap(market="KR", ticker="005930", name="삼성전자")) == "naver"


# ── apply_checkup: per/pbr/pos/price 재활용 (데이터계약 하이브리드) ──

def _ck(per=None, pbr=None, position_pct=None, price=None):
    return {"per": per, "pbr": pbr, "position_pct": position_pct, "price": price}


def test_checkup_fills_pos_when_missing():
    snaps = [_snap(market="KR", ticker="005930", pos_52w=None)]
    out = apply_checkup(snaps, {"005930": _ck(position_pct=42.3)})
    assert out[0].pos_52w == 42.3


def test_checkup_does_not_override_existing_pos():
    snaps = [_snap(market="KR", ticker="005930", pos_52w=55.0)]
    out = apply_checkup(snaps, {"005930": _ck(position_pct=42.3)})
    assert out[0].pos_52w == 55.0  # 기존값 보존


def test_checkup_empty_map_noop():
    snaps = [_snap(pos_52w=None)]
    out = apply_checkup(snaps, {})
    assert out is snaps  # 빈 맵이면 그대로 반환


def test_checkup_missing_code_unchanged():
    snaps = [_snap(market="KR", ticker="000660", pos_52w=None, per=None, pbr=None)]
    out = apply_checkup(snaps, {"005930": _ck(per=7.0, pbr=1.0)})
    assert out[0].per is None and out[0].pbr is None


def test_checkup_fills_per_pbr_when_missing():
    # 한국 종목 데이터부족(per/pbr None) → checkup으로 보완
    snaps = [_snap(market="KR", ticker="005930", per=None, pbr=None, roe=None, pos_52w=None)]
    out = apply_checkup(snaps, {"005930": _ck(per=8.0, pbr=1.2, position_pct=30.0)})
    assert out[0].per == 8.0 and out[0].pbr == 1.2
    assert out[0].pos_52w == 30.0
    # per/pbr 채워지면 roe도 pbr/per 근사 (1.2/8.0*100=15.0)
    assert out[0].roe == 15.0


def test_checkup_does_not_override_existing_per():
    snaps = [_snap(market="KR", ticker="005930", per=10.0, pbr=2.0, roe=20.0)]
    out = apply_checkup(snaps, {"005930": _ck(per=8.0, pbr=1.2)})
    assert out[0].per == 10.0 and out[0].pbr == 2.0 and out[0].roe == 20.0  # 전부 보존


def test_checkup_zero_per_is_treated_as_missing():
    # checkup의 per/pbr이 0인 종목(8/30)은 무효 → snapshot 값 유지
    snaps = [_snap(market="KR", ticker="005930", per=12.0, pbr=1.5)]
    out = apply_checkup(snaps, {"005930": _ck(per=0, pbr=0)})
    assert out[0].per == 12.0 and out[0].pbr == 1.5


# ── to_dashboard_row: 데이터계약 §1 컬럼 ──────────────────────────

EXPECTED_COLS = {
    "date", "market", "ticker", "name", "price", "per", "fwd_per", "pbr", "roe",
    "pos_52w", "fcf_yield", "debt_to_equity", "verdict", "earnings_up", "source",
    "snapshot_time",
}


def test_dashboard_row_has_exact_contract_columns():
    row = to_dashboard_row(_snap(), "2026-06-16", "2026-06-16T15:40:00+09:00")
    assert set(row.keys()) == EXPECTED_COLS


def test_dashboard_row_computed_fields():
    # Meta: FCF +25000 / 시총 1,000,000 = +2.5% , fwd_per<per → earnings_up True
    row = to_dashboard_row(_snap(), "2026-06-16", "2026-06-16T15:40:00+09:00")
    assert row["fcf_yield"] == 2.5
    assert row["earnings_up"] is True
    assert row["verdict"] in STANDARD_VERDICTS
    assert row["date"] == "2026-06-16"
    assert row["market"] == "US"


def test_dashboard_row_value_trap_oracle_like():
    # 오라클형: 고ROE + FCF 음수 → 가치함정
    oracle = _snap(market="US", ticker="ORCL", name="Oracle", roe=53.0, fcf=-3000.0, pos_52w=70.0)
    row = to_dashboard_row(oracle, "2026-06-16", "2026-06-16T15:40:00+09:00")
    assert row["verdict"] == "가치함정"
    assert row["fcf_yield"] is not None and row["fcf_yield"] < 0


def test_dashboard_row_nulls_safe():
    # 데이터 부족 종목도 None 안전하게 직렬화
    empty = ValuationSnapshot("KR", "999999", "X", *([None] * 10))
    row = to_dashboard_row(empty, "2026-06-16", "2026-06-16T15:40:00+09:00")
    assert row["verdict"] == "데이터부족"
    assert row["fcf_yield"] is None
    assert row["earnings_up"] is None
