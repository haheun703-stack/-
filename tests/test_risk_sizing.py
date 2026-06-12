"""risk/sizing.py 테스트 — L1 리스크 예산 사이징 + 갭 보정 + 유동성 한도.

전부 합성 OHLCV로 구성 (네트워크 0, 운영 파일 접촉 0, 파일쓰기 0).
각 테스트의 한국어 주석 = 그 테스트가 막는 사고 1줄.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from risk.config import RISK_CONFIG, RiskConfig, limit_down_survival_ok
from risk.sizing import (
    SizingResult,
    adv_krw,
    audit_record,
    compute_atr,
    size_position,
    worst_overnight_gap,
)


# ──────────────────────────────────────────────────────────────────────────────
# 합성 OHLCV 헬퍼 — 상수 가격으로 ATR/갭/ADV를 정확히 통제한다.
# ──────────────────────────────────────────────────────────────────────────────
def make_ohlcv(
    n: int = 300,
    open_: float = 100.0,
    high: float = 101.25,
    low: float = 98.75,
    close: float = 100.0,
    volume: float = 10_000.0,
    trading_value: float = 1e12,
) -> pd.DataFrame:
    """기본 df: TR=2.5 상수 → ATR=2.5, 갭 0, ADV=1e12(캡 비활성 수준)."""
    return pd.DataFrame(
        {
            "open": [open_] * n,
            "high": [high] * n,
            "low": [low] * n,
            "close": [close] * n,
            "volume": [volume] * n,
            "trading_value": [trading_value] * n,
        }
    )


EQUITY = 10_000_000.0  # R = 50,000 (risk_per_trade=0.005), single_cap = 1,200,000
ENTRY = 100.0


# ──────────────────────────────────────────────────────────────────────────────
# 1. compute_atr
# ──────────────────────────────────────────────────────────────────────────────
def test_compute_atr_constant_tr_exact():
    # 사고 방지: ATR 산식이 틀어지면 손절폭→사이즈 전체가 오염된다. 상수 TR=10이면 ATR=10 정확.
    df = make_ohlcv(n=30, high=105.0, low=95.0, close=100.0)
    atr = compute_atr(df, period=14)
    assert atr == pytest.approx(10.0, abs=1e-9)


def test_compute_atr_wilder_manual_recursion():
    # 사고 방지: ewm 근사 등으로 슬쩍 바뀌면 적발 — Wilder SMA시드+재귀 수계산과 1e-9 일치.
    df = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 11.0, 13.0, 14.0],
            "low": [9.0, 9.0, 10.0, 9.0, 11.0, 12.0],
            "close": [9.5, 10.0, 11.0, 10.0, 12.0, 13.0],
        }
    )
    # TR = [2, 2, 2, 3, 2] → seed=mean(2,2,2)=2 → (2*2+3)/3=7/3 → (7/3*2+2)/3=20/9
    assert compute_atr(df, period=3) == pytest.approx(20.0 / 9.0, abs=1e-9)


def test_compute_atr_insufficient_rows_raises():
    # 사고 방지: 행 부족인데 조용히 ATR을 내면 stale/짧은 데이터로 사이즈가 부풀 수 있다.
    df = make_ohlcv(n=14)  # period=14 → 15행 필요
    with pytest.raises(ValueError):
        compute_atr(df, period=14)


def test_compute_atr_missing_column_raises():
    # 사고 방지: 컬럼 누락 df가 조용히 통과하면 잘못된 데이터로 사이징된다.
    df = make_ohlcv(n=30).drop(columns=["high"])
    with pytest.raises(ValueError):
        compute_atr(df, period=14)


# ──────────────────────────────────────────────────────────────────────────────
# 2. worst_overnight_gap
# ──────────────────────────────────────────────────────────────────────────────
def test_worst_gap_planted_down_gap_exact():
    # 사고 방지: 한국장 시초 갭 보정이 갭을 못 잡으면 갭락 시 손실이 R을 초과한다.
    df = make_ohlcv(n=300)
    df.loc[295, "open"] = 93.0  # 전일종가 100 → 시가 93 = 하방 7% 갭
    assert worst_overnight_gap(df, lookback=252) == pytest.approx(0.07, abs=1e-12)


def test_worst_gap_up_gaps_only_returns_zero():
    # 사고 방지: 상승 갭을 하방 갭으로 오인하면 사이즈가 과잉 축소(역방향 오류)된다.
    df = make_ohlcv(n=300, open_=102.0, close=100.0)  # 매일 +2% 상승 갭만
    assert worst_overnight_gap(df, lookback=252) == 0.0


def test_worst_gap_outside_lookback_ignored():
    # 사고 방지: lookback 창을 무시하고 전체 이력을 보면 10년 전 갭이 영원히 사이즈를 짓누른다.
    df = make_ohlcv(n=300)
    df.loc[10, "open"] = 90.0  # tail(253)=행47~299 밖 → 무시되어야 함
    assert worst_overnight_gap(df, lookback=252) == 0.0


def test_worst_gap_missing_column_raises():
    # 사고 방지: open 누락 df가 0.0으로 조용히 통과하면 갭 보정 자체가 무력화된다.
    df = make_ohlcv(n=300).drop(columns=["open"])
    with pytest.raises(ValueError):
        worst_overnight_gap(df, lookback=252)


# ──────────────────────────────────────────────────────────────────────────────
# 3. adv_krw
# ──────────────────────────────────────────────────────────────────────────────
def test_adv_trading_value_column_preferred():
    # 사고 방지: trading_value가 있는데 close×volume으로 계산하면 단위 불일치로 유동성 과대평가.
    df = make_ohlcv(n=30, close=100.0, volume=999_999.0, trading_value=5_000_000.0)
    assert adv_krw(df, window=20) == pytest.approx(5_000_000.0)


def test_adv_fallback_close_times_volume():
    # 사고 방지: trading_value 없는 소스에서 ADV=0 처리되면 전 종목이 부당 차단된다.
    df = make_ohlcv(n=30, close=100.0, volume=50.0).drop(columns=["trading_value"])
    assert adv_krw(df, window=20) == pytest.approx(5_000.0)


def test_adv_no_columns_returns_zero():
    # 사고 방지: 거래대금을 알 수 없는데 0이 아닌 값을 내면 유동성 캡이 뚫린다(fail-closed 확인).
    df = pd.DataFrame({"high": [1.0] * 30, "low": [1.0] * 30})
    assert adv_krw(df, window=20) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 4. size_position — 수급이탈가 vs ATR손절가
# ──────────────────────────────────────────────────────────────────────────────
def test_supply_stop_higher_than_atr_adopted():
    # 사고 방지: 수급이탈가(진입가에 더 가까운 손절)를 무시하면 손절폭 과대→사이즈 과소가 아니라
    # 손절 라인 자체가 틀어진다. ATR손절=95(100−2×2.5)보다 높은 97이 채택되어야 함.
    df = make_ohlcv()
    res = size_position(EQUITY, ENTRY, df, supply_stop_price=97.0)
    assert not res.rejected
    assert res.stop_price == pytest.approx(97.0)
    assert res.stop_width == pytest.approx(0.03)


def test_atr_stop_wins_when_supply_stop_lower():
    # 사고 방지: 더 낮은(느슨한) 수급이탈가가 ATR손절을 끌어내리면 손절폭 과대로 리스크 초과.
    df = make_ohlcv()
    res = size_position(EQUITY, ENTRY, df, supply_stop_price=90.0)
    assert not res.rejected
    assert res.stop_price == pytest.approx(95.0)  # max(95, 90) = ATR쪽
    assert res.stop_width == pytest.approx(0.05)


def test_basic_sizing_arithmetic():
    # 사고 방지: size = R/stop_width 산식이 틀어지면 모든 포지션 크기가 틀린다.
    df = make_ohlcv()  # ATR=2.5 → stop=95, width=0.05, 갭 0
    res = size_position(EQUITY, ENTRY, df)
    assert not res.rejected and res.reason is None
    assert res.risk_budget_krw == pytest.approx(EQUITY * 0.005)  # R=50,000
    assert res.size_krw == pytest.approx(1_000_000.0)  # 50,000 / 0.05
    assert res.shares == 10_000  # floor(1,000,000/100)
    assert res.caps_applied == ()


# ──────────────────────────────────────────────────────────────────────────────
# 5. invalid_stop
# ──────────────────────────────────────────────────────────────────────────────
def test_supply_stop_at_or_above_entry_rejected():
    # 사고 방지: 손절가 ≥ 진입가(폭 0/음수)면 size=R/0 → 무한대 사이즈 폭주. 차단 필수.
    df = make_ohlcv()
    for bad_stop in (ENTRY, 105.0):
        res = size_position(EQUITY, ENTRY, df, supply_stop_price=bad_stop)
        assert res.rejected and res.reason == "invalid_stop"
        assert res.shares == 0 and res.size_krw == 0.0


def test_zero_atr_flat_market_invalid_stop():
    # 사고 방지: 횡보(ATR=0)에서 stop=entry가 되어도 0나눗셈 폭주 없이 차단되어야 함.
    df = make_ohlcv(high=100.0, low=100.0, close=100.0)
    res = size_position(EQUITY, ENTRY, df)
    assert res.rejected and res.reason == "invalid_stop"


# ──────────────────────────────────────────────────────────────────────────────
# 6. 갭 보정 — gap_worst > stop_width면 effective가 갭으로 대체
# ──────────────────────────────────────────────────────────────────────────────
def test_gap_wider_than_stop_shrinks_size():
    # 사고 방지: 갭 보정이 죽으면 −7% 갭락 이력 종목에 stop_width(5%) 기준 과대 사이즈가 나간다.
    df_nogap = make_ohlcv()  # stop_width=0.05
    df_gap = make_ohlcv()
    df_gap.loc[295, "open"] = 93.0  # 하방 7% 갭 주입

    res_nogap = size_position(EQUITY, ENTRY, df_nogap)
    res_gap = size_position(EQUITY, ENTRY, df_gap)

    assert not res_gap.rejected
    assert res_gap.gap_worst == pytest.approx(0.07)
    assert res_gap.stop_width == pytest.approx(0.05)
    assert res_gap.effective_stop_width == pytest.approx(0.07)  # max(0.05, 0.07)
    assert res_gap.size_krw == pytest.approx(50_000.0 / 0.07)
    assert res_gap.size_krw < res_nogap.size_krw  # 갭 보정이 사이즈를 줄였다


# ──────────────────────────────────────────────────────────────────────────────
# 7. single_weight_cap
# ──────────────────────────────────────────────────────────────────────────────
def test_single_weight_cap_applied():
    # 사고 방지: 종목 비중 캡이 뚫리면 단일 종목 하한가(-30%) 한 방에 일일 킬 한도 초과.
    df = make_ohlcv(high=100.5, low=99.5)  # ATR=1 → stop_width=0.02 → raw size 2.5M
    res = size_position(EQUITY, ENTRY, df)
    assert not res.rejected
    assert "single_weight_cap" in res.caps_applied
    assert "adv_cap" not in res.caps_applied
    assert res.size_krw == pytest.approx(EQUITY * 0.12)  # 1,200,000
    assert res.shares == 12_000


# ──────────────────────────────────────────────────────────────────────────────
# 8. adv_cap
# ──────────────────────────────────────────────────────────────────────────────
def test_adv_cap_applied():
    # 사고 방지: 유동성 캡이 죽으면 저유동성 종목에서 하루 안에 청산 불가능한 사이즈가 나간다.
    df = make_ohlcv(trading_value=1_000_000.0)  # ADV=1M → 캡=50,000 (5%)
    res = size_position(EQUITY, ENTRY, df)  # raw size 1M ≫ 50,000
    assert not res.rejected
    assert res.caps_applied == ("adv_cap",)
    assert res.size_krw == pytest.approx(50_000.0)
    assert res.shares == 500


def test_adv_unknown_leads_to_rejection_with_audit_trail():
    # 사고 방지: 거래대금 미산출 시 사이즈 유지되면 유동성 모르는 종목에 진입한다(fail-closed).
    df = make_ohlcv().drop(columns=["trading_value", "volume"])
    res = size_position(EQUITY, ENTRY, df)
    assert res.rejected and res.reason == "below_min_unit"
    assert "adv_cap" in res.caps_applied  # 왜 0이 됐는지 감사 흔적 보존
    assert res.size_krw == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 9. below_min_unit
# ──────────────────────────────────────────────────────────────────────────────
def test_entry_price_too_large_below_min_unit():
    # 사고 방지: 1주도 못 사는 예산으로 shares=0 주문이 흘러가면 하류에서 0주 주문 오류/우회 발생.
    df = make_ohlcv()
    res = size_position(equity_krw=100_000.0, entry_price=200_000.0, ohlcv_df=df)
    # R=500, width≈2.5/200000 → raw size 거대 → single cap 12,000 → floor(12,000/200,000)=0
    assert res.rejected and res.reason == "below_min_unit"
    assert res.shares == 0
    assert "single_weight_cap" in res.caps_applied  # 차단 결과에도 중간값 보존 확인


# ──────────────────────────────────────────────────────────────────────────────
# 10. invalid_input
# ──────────────────────────────────────────────────────────────────────────────
def test_invalid_inputs_rejected():
    # 사고 방지: equity 0/음수/NaN/inf가 산식에 들어가면 음수·NaN 사이즈가 하류로 전파된다.
    df = make_ohlcv()
    bad_cases = [
        dict(equity_krw=0.0, entry_price=ENTRY),
        dict(equity_krw=-5_000_000.0, entry_price=ENTRY),
        dict(equity_krw=EQUITY, entry_price=0.0),
        dict(equity_krw=EQUITY, entry_price=-100.0),
        dict(equity_krw=float("nan"), entry_price=ENTRY),
        dict(equity_krw=float("inf"), entry_price=ENTRY),
        dict(equity_krw=EQUITY, entry_price=float("nan")),
        dict(equity_krw=EQUITY, entry_price=ENTRY, supply_stop_price=float("nan")),
    ]
    for kwargs in bad_cases:
        res = size_position(ohlcv_df=df, **kwargs)
        assert res.rejected and res.reason == "invalid_input", kwargs
        assert res.shares == 0 and res.size_krw == 0.0


def test_insufficient_data_fail_closed():
    # 사고 방지: 데이터 부족/오염 시 예외 전파나 임의 사이즈 대신 명시적 차단(모르면 차단).
    short_df = make_ohlcv(n=10)  # ATR(14) 계산 불가
    res = size_position(EQUITY, ENTRY, short_df)
    assert res.rejected and res.reason == "insufficient_data"

    no_open_df = make_ohlcv().drop(columns=["open"])  # 갭 계산 불가
    res2 = size_position(EQUITY, ENTRY, no_open_df)
    assert res2.rejected and res2.reason == "insufficient_data"


# ──────────────────────────────────────────────────────────────────────────────
# 11. config — 하한가 생존 조건
# ──────────────────────────────────────────────────────────────────────────────
def test_limit_down_survival_ok_default_true():
    # 사고 방지: 기본 파라미터 조합이 하한가 생존 조건을 깨면(캡>한도/30%) 배포 전에 적발.
    assert limit_down_survival_ok() is True
    assert limit_down_survival_ok(RISK_CONFIG) is True


def test_limit_down_survival_detects_violation():
    # 사고 방지: daily_kill_limit만 조이고 max_single_weight 재계산을 잊는 사고를 함수가 잡는지.
    risky = RiskConfig(daily_kill_limit=-0.01)  # 파생 한도 3.33% ≪ 캡 12%
    assert limit_down_survival_ok(risky) is False


# ──────────────────────────────────────────────────────────────────────────────
# 보너스 — audit_record (쓰기 3원칙: dict만, KST 타임스탬프)
# ──────────────────────────────────────────────────────────────────────────────
def test_audit_record_kst_timestamp_and_fields():
    # 사고 방지: VPS(UTC)에서 naive now()를 쓰면 9시간 어긋난 감사 기록이 남는다 — KST 강제 확인.
    res = size_position(EQUITY, ENTRY, make_ohlcv())
    naive = datetime(2026, 6, 11, 9, 30, 0)
    rec = audit_record(res, now_kst=naive)
    assert rec["ts_kst"] == "2026-06-11T09:30:00+09:00"  # naive → KST 부착
    utc = datetime(2026, 6, 11, 0, 30, 0, tzinfo=timezone.utc)
    rec2 = audit_record(res, now_kst=utc)
    assert rec2["ts_kst"] == "2026-06-11T09:30:00+09:00"  # aware → KST 변환
    for field in ("size_krw", "shares", "stop_price", "effective_stop_width", "rejected", "reason"):
        assert field in rec


# ──────────────────────────────────────────────────────────────────────────────
# 적대리뷰 회귀 — 갭 NaN행 보존 / 유령행 / sparse floor / numpy 정수
# ──────────────────────────────────────────────────────────────────────────────
def test_gap_nan_row_does_not_delete_worst_gap():
    # 사고 방지(P1): NaN 섞인 행을 dropna로 통째 제거하면 인덱스가 밀려 실제 최악 갭이 사라진다.
    df = make_ohlcv(n=300)
    df.loc[295, "open"] = 93.0          # 하방 7% 갭 (전일종가 100 → 시가 93)
    df.loc[290, "open"] = float("nan")  # 무관한 행 NaN — dropna였다면 인덱스가 밀려 0.07이 사라짐
    assert worst_overnight_gap(df, lookback=252) == pytest.approx(0.07, abs=1e-9)


def test_gap_ghost_row_zero_open_excluded():
    # 사고 방지(P2): 시가 0인 유령행(휴장 0행 오염)이 가짜 100% 갭을 만들어 사이즈를 폭락시킨다.
    df = make_ohlcv(n=300)
    df.loc[295, "open"] = 0.0           # 유령행 — today_open>0 마스크로 제외되어야
    assert worst_overnight_gap(df, lookback=252) == 0.0


def test_sparse_gap_floor_applied_to_short_history():
    # 사고 방지(P1): 짧은 이력(신규상장 등)은 갭 표본이 적어 갭 위험이 안 보인다 → 보수 floor로 사이즈 축소.
    df_short = make_ohlcv(n=40)         # 갭 표본 39 < gap_min_samples(60)
    res = size_position(EQUITY, ENTRY, df_short)
    assert not res.rejected
    assert res.gap_sample_count == 39
    assert "sparse_gap_floor" in res.caps_applied
    assert res.effective_stop_width == pytest.approx(0.10)  # sparse_gap_floor (> stop_width 0.05)
    res_long = size_position(EQUITY, ENTRY, make_ohlcv(n=300))
    assert res.size_krw < res_long.size_krw  # floor가 짧은 이력 종목 사이즈를 줄였다


def test_sufficient_history_no_floor():
    # 사고 방지: 충분한 이력(>=60)엔 floor가 적용되지 않아 정상 사이징을 방해하지 않는다.
    res = size_position(EQUITY, ENTRY, make_ohlcv(n=300))
    assert res.gap_sample_count >= 60
    assert "sparse_gap_floor" not in res.caps_applied


def test_numpy_scalar_inputs_accepted_bool_rejected():
    # 사고 방지(P2): np.int64/np.float64(pandas 산출)가 invalid_input으로 오거부되면 정상 종목이 차단된다.
    import numpy as np
    df = make_ohlcv()
    res = size_position(np.int64(EQUITY), np.float64(ENTRY), df)
    assert not res.rejected and res.shares > 0
    res_bool = size_position(True, ENTRY, df)  # bool은 금액이 아니다 → 거부
    assert res_bool.rejected and res_bool.reason == "invalid_input"
