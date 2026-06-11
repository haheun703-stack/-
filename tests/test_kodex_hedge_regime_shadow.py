"""KODEX 국면별 헤지 shadow 모듈 테스트.

★검증: regime classifier(BULL/BEAR/UNKNOWN) + 국면별 hedge policy + 7 portfolio +
누적/MDD/whipsaw 재계산. 합성 데이터(네트워크 없음). 실매매 0.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.etf.kodex_hedge_regime_shadow as kshadow  # noqa: E402
from src.etf.kodex_hedge_regime_shadow import (  # noqa: E402
    HEDGE_FIXED_THICK,
    HEDGE_MINIMAL,
    HEDGE_VOL_DYNAMIC,
    KST,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_UNKNOWN,
    _adaptive_ratio,
    _portfolios_1d,
    _recompute,
    build_record,
    classify_regime,
    finality,
    hedge_policy_for,
)


def _idx(vals):
    return pd.Series(vals, index=pd.date_range("2026-01-01", periods=len(vals), freq="B"), dtype=float)


# ── regime classifier ──
def test_regime_bull_uptrend():
    r, _ = classify_regime(_idx(list(range(100, 200))))  # 단조 상승
    assert r == REGIME_BULL


def test_regime_bear_downtrend():
    r, _ = classify_regime(_idx(list(range(200, 100, -1))))  # 단조 하락
    assert r == REGIME_BEAR


def test_regime_unknown_when_short():
    r, _ = classify_regime(_idx([100] * 30))  # <60일
    assert r == REGIME_UNKNOWN


def test_regime_unknown_when_flat():
    # 60일 이상이지만 횡보(추세 불명)
    vals = [100 + (i % 5) for i in range(80)]
    r, _ = classify_regime(_idx(vals))
    assert r == REGIME_UNKNOWN


# ── hedge policy (국면별) ──
def test_hedge_policy_mapping():
    assert hedge_policy_for(REGIME_BULL) == HEDGE_VOL_DYNAMIC
    assert hedge_policy_for(REGIME_BEAR) == HEDGE_FIXED_THICK
    assert hedge_policy_for(REGIME_UNKNOWN) == HEDGE_MINIMAL


# ── 7 portfolio ──
def test_portfolios_hedge_applied():
    p = _portfolios_1d(0.02, -0.01, {"vol_on": True, "fx_on": False, "c60_on": False})
    assert p["leverage_core_only"] == 2.0
    assert p["leverage_plus_fixed_80"] == round((0.02 - 0.01 * 0.8) * 100, 4)
    assert p["leverage_plus_vol_dynamic"] == round((0.02 - 0.01 * 0.8) * 100, 4)  # vol_on
    assert p["leverage_plus_fx_dynamic"] == 2.0  # fx_off → 헤지 0
    assert p["leverage_plus_c60_dynamic"] == 2.0  # c60_off → 헤지 0


# ── adaptive ratio (국면별 권장) ──
def test_adaptive_ratio_by_regime():
    assert _adaptive_ratio(REGIME_BEAR, {"vol_on": False}) == 0.8       # 약세=고정 두껍게
    assert _adaptive_ratio(REGIME_BULL, {"vol_on": True}) == 0.8        # 강세+변동성확대=헤지
    assert _adaptive_ratio(REGIME_BULL, {"vol_on": False}) == 0.0       # 강세+평온=무헤지
    assert _adaptive_ratio(REGIME_UNKNOWN, {"vol_on": True}) == 0.0     # 애매=관망


# ── 누적/MDD/whipsaw 재계산 ──
def test_recompute_cum_and_mdd():
    recs = [
        {"date": "2026-06-09", "portfolio_ret_1d": 2.0, "hedge_ratio": 0.0},
        {"date": "2026-06-10", "portfolio_ret_1d": -3.0, "hedge_ratio": 0.8},
    ]
    out = _recompute(recs)
    assert out[0]["portfolio_cum_ret"] == 2.0
    assert out[1]["portfolio_cum_ret"] == round((1.02 * 0.97 - 1) * 100, 2)
    assert out[1]["mdd"] < 0  # 둘째 날 하락 → MDD 음수


def test_recompute_whipsaw_flag():
    # 헤지 on/off 잦은 전환 → whipsaw True
    recs = [{"date": f"2026-06-{9+i:02d}", "portfolio_ret_1d": 0.0,
             "hedge_ratio": r} for i, r in enumerate([0.8, 0.0, 0.8, 0.0, 0.8])]
    out = _recompute(recs)
    assert out[-1]["whipsaw_flag"] is True


# ── build_record 구조 ──
def test_build_record_has_16_core_fields():
    idx = _idx(list(range(100, 200)))
    lev = _idx(list(range(100, 200)))
    inv = _idx(list(range(200, 100, -1)))
    rec = build_record("2026-06-09", lev, inv, idx, None)
    for f in ("date", "kodex200_close", "leverage_close", "inverse_close",
              "kodex200_ret_1d", "leverage_ret_1d", "inverse_ret_1d", "regime",
              "hedge_policy", "hedge_ratio", "hedge_reason", "portfolio_ret_1d",
              "realized_vol", "notes", "portfolios_ret_1d", "signals"):
        assert f in rec, f"누락 필드: {f}"
    assert rec["regime"] in (REGIME_BULL, REGIME_BEAR, REGIME_UNKNOWN)
    assert len(rec["portfolios_ret_1d"]) == 7


def test_build_record_rejects_mismatched_last_date():
    # 세 ETF 마지막 거래일이 다르면 1일 수익률이 어긋나므로 거부(거래정지 등)
    idx = _idx(list(range(100, 200)))
    lev = _idx(list(range(100, 200)))
    inv = _idx(list(range(200, 100, -1))).iloc[:-2]  # 마지막 2거래일 빠짐 → 종료일 불일치
    try:
        build_record("2026-06-09", lev, inv, idx, None)
        assert False, "마지막 거래일 불일치인데 통과함"
    except RuntimeError as e:
        assert "거래일 불일치" in str(e)


def test_run_snapshot_no_write_fills_cumulative(tmp_path, monkeypatch):
    # ★회귀: --no-write(dry)도 cum/mdd/whipsaw를 채워야 CLI 포맷이 크래시하지 않음
    up = _idx(list(range(100, 200)))
    down = _idx(list(range(200, 100, -1)))

    def fake_close(ticker, start, end):
        return down.rename(ticker) if ticker == kshadow.TICKER_INVERSE else up.rename(ticker)

    monkeypatch.setattr(kshadow, "_load_close", fake_close)
    monkeypatch.setattr(kshadow, "_load_fx", lambda: None)
    monkeypatch.setattr(kshadow, "LEDGER_PATH", tmp_path / "ledger.json")
    monkeypatch.setattr(kshadow, "LEDGER_DIR", tmp_path)
    rec = kshadow.run_snapshot(write=False)
    assert rec["portfolio_cum_ret"] is not None
    assert rec["mdd"] is not None
    assert rec["whipsaw_flag"] is not None
    assert not (tmp_path / "ledger.json").exists()  # dry는 저장 안 함


# ── is_final / snapshot_time: 레코드가 자기 상태를 선언 ──
def test_finality_past_date_is_final():
    # 과거 거래일 = 종가 확정
    st, is_final = finality("2026-06-10", datetime(2026, 6, 11, 12, 0, tzinfo=KST))
    assert is_final is True
    assert st.startswith("2026-06-11T12:00")


def test_finality_today_intraday_is_provisional():
    # 당일 장중(15:30 전) = 미확정
    _, is_final = finality("2026-06-11", datetime(2026, 6, 11, 12, 36, tzinfo=KST))
    assert is_final is False


def test_finality_today_after_close_is_final():
    # 당일 15:30 이후 = 종가 확정
    _, is_final = finality("2026-06-11", datetime(2026, 6, 11, 15, 31, tzinfo=KST))
    assert is_final is True


def test_finality_close_boundary_inclusive():
    # 정확히 15:30 = 확정(>=)
    _, is_final = finality("2026-06-11", datetime(2026, 6, 11, 15, 30, tzinfo=KST))
    assert is_final is True


# ── load_ledger 소비자 가드: 미확정 기본 제외 ──
def _write_ledger(path, records):
    path.write_text(json.dumps({"records": records}, ensure_ascii=False), encoding="utf-8")


def test_load_ledger_excludes_provisional_by_default(tmp_path, monkeypatch):
    p = tmp_path / "ledger.json"
    _write_ledger(p, [{"date": "2026-06-10", "is_final": True},
                      {"date": "2026-06-11", "is_final": False}])
    monkeypatch.setattr(kshadow, "LEDGER_PATH", p)
    assert [r["date"] for r in kshadow.load_ledger()] == ["2026-06-10"]      # 확정만
    assert len(kshadow.load_ledger(include_provisional=True)) == 2            # 내부 누적은 전체


def test_load_ledger_legacy_record_treated_as_final(tmp_path, monkeypatch):
    # is_final 키 부재(레거시) = 과거 종가로 간주(True)
    p = tmp_path / "ledger.json"
    _write_ledger(p, [{"date": "2026-06-09"}])
    monkeypatch.setattr(kshadow, "LEDGER_PATH", p)
    assert len(kshadow.load_ledger()) == 1


# ── stale_provisional_warning: 저녁 재실행 누락 자가 감지 ──
def test_stale_warning_detects_provisional(tmp_path, monkeypatch):
    p = tmp_path / "ledger.json"
    _write_ledger(p, [{"date": "2026-06-10", "is_final": True},
                      {"date": "2026-06-11", "is_final": False,
                       "snapshot_time": "2026-06-11T12:36:00+09:00"}])
    monkeypatch.setattr(kshadow, "LEDGER_PATH", p)
    w = kshadow.stale_provisional_warning()
    assert w is not None and "2026-06-11" in w


def test_stale_warning_none_when_all_final(tmp_path, monkeypatch):
    p = tmp_path / "ledger.json"
    _write_ledger(p, [{"date": "2026-06-10", "is_final": True}])
    monkeypatch.setattr(kshadow, "LEDGER_PATH", p)
    assert kshadow.stale_provisional_warning() is None


# ── run_snapshot: finality 주입 + dry 기본값(안전한 기본값) ──
def _patch_synthetic(tmp_path, monkeypatch):
    up = _idx(list(range(100, 200)))
    down = _idx(list(range(200, 100, -1)))

    def fake_close(ticker, start, end):
        return down.rename(ticker) if ticker == kshadow.TICKER_INVERSE else up.rename(ticker)

    monkeypatch.setattr(kshadow, "_load_close", fake_close)
    monkeypatch.setattr(kshadow, "_load_fx", lambda: None)
    monkeypatch.setattr(kshadow, "LEDGER_PATH", tmp_path / "ledger.json")
    monkeypatch.setattr(kshadow, "LEDGER_DIR", tmp_path)


def test_run_snapshot_injects_finality_fields(tmp_path, monkeypatch):
    _patch_synthetic(tmp_path, monkeypatch)
    rec = kshadow.run_snapshot(write=False)
    assert "snapshot_time" in rec
    assert isinstance(rec["is_final"], bool)


def test_run_snapshot_default_is_dry(tmp_path, monkeypatch):
    # ★사장님 ②: 안전한 기본값 — 인자 없이 실행 = dry, ledger 저장 0
    _patch_synthetic(tmp_path, monkeypatch)
    kshadow.run_snapshot()  # write 명시 안 함
    assert not (tmp_path / "ledger.json").exists()


def test_run_snapshot_write_true_saves(tmp_path, monkeypatch):
    _patch_synthetic(tmp_path, monkeypatch)
    kshadow.run_snapshot(write=True)
    assert (tmp_path / "ledger.json").exists()
