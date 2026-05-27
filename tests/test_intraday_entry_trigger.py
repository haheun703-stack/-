"""5분봉 진입 트리거 단위 테스트 (MVP-6, 2026-05-27 신규)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.use_cases.intraday_entry_trigger import (
    IntradayEntryDecision,
    _aggregate_bucket,
    _compute_rsi,
    _resample_1m_to_5m,
    evaluate_intraday_entry,
    format_for_telegram,
)


def make_1m_row(time_hhmmss: str, o: int, h: int, l: int, c: int, vol: int) -> dict:
    """KIS API 1분봉 row 형식 mock."""
    return {
        "stck_cntg_hour": time_hhmmss,
        "stck_oprc": str(o),
        "stck_hgpr": str(h),
        "stck_lwpr": str(l),
        "stck_prpr": str(c),
        "cntg_vol": str(vol),
    }


def make_broker(rows: list[dict]):
    """fetch_today_1m_ohlcv를 mocking."""
    broker = MagicMock()
    broker.fetch_today_1m_ohlcv.return_value = {"output2": rows}
    return broker


class TestResample:
    def test_resample_1m_to_5m_basic(self):
        """5개 1분봉 → 1개 5분봉 (09:00~09:04)."""
        # KIS API는 최신순(역순)으로 반환
        rows = [
            make_1m_row("090400", 105, 108, 104, 107, 100),
            make_1m_row("090300", 103, 106, 102, 105, 120),
            make_1m_row("090200", 102, 104, 100, 103, 90),
            make_1m_row("090100", 101, 103, 100, 102, 110),
            make_1m_row("090000", 100, 102, 99, 101, 80),
        ]
        bars = _resample_1m_to_5m(rows)
        assert len(bars) == 1
        bar = bars[0]
        assert bar["open"] == 100   # 09:00 open
        assert bar["close"] == 107  # 09:04 close
        assert bar["high"] == 108   # max
        assert bar["low"] == 99     # min
        assert bar["volume"] == 500  # sum

    def test_resample_two_buckets(self):
        """09:00~09:04 + 09:05~09:09 → 2개 5분봉."""
        rows = []
        # 09:05~09:09
        for i in range(5):
            mm = 9 - i
            rows.append(make_1m_row(f"09{mm:02d}00", 110, 115, 108, 112, 50))
        # 09:00~09:04
        for i in range(5):
            mm = 4 - i
            rows.append(make_1m_row(f"09{mm:02d}00", 100, 105, 99, 103, 40))
        bars = _resample_1m_to_5m(rows)
        assert len(bars) == 2
        # 최신순 → bars[0] = 09:05~09:09
        assert bars[0]["slot"] == "09:05"
        assert bars[1]["slot"] == "09:00"


class TestRSI:
    def test_rsi_insufficient_data(self):
        """데이터 부족 시 50 반환."""
        assert _compute_rsi([100, 101]) == 50.0

    def test_rsi_all_gains(self):
        """모두 상승 = RSI 100."""
        closes = [100 + i for i in range(20)]
        assert _compute_rsi(closes) == 100.0


class TestEvaluate:
    def test_no_data(self):
        """1분봉 데이터 없음 → trigger=False."""
        broker = make_broker([])
        dec = evaluate_intraday_entry(broker, "000990", "DB하이텍")
        assert dec.trigger is False
        assert "1분봉 데이터 없음" in dec.reasons_fail

    def test_all_4_conditions_pass(self):
        """4 조건 모두 충족 → trigger=True (pass_count=4)."""
        # 충분한 1분봉 데이터 (20개 = 4개 5분봉)
        rows = []
        # 직전 5분봉 거래량 평균 100 → 현재 200 거래량 (1.5x↑)
        # 양봉 + RSI 정상 + VWAP 위
        # 최신순으로 구성
        # 5분 슬롯 4 (가장 최근): 거래량 200, 양봉
        for i, t in enumerate(["091400", "091300", "091200", "091100", "091000"]):
            rows.append(make_1m_row(t, 100, 105, 100, 105, 40))  # 강한 양봉
        # 5분 슬롯 3: 평균 거래량 100
        for i, t in enumerate(["090900", "090800", "090700", "090600", "090500"]):
            rows.append(make_1m_row(t, 98, 100, 97, 99, 20))
        # 5분 슬롯 2
        for i, t in enumerate(["090400", "090300", "090200", "090100", "090000"]):
            rows.append(make_1m_row(t, 95, 99, 94, 98, 20))

        broker = make_broker(rows)
        dec = evaluate_intraday_entry(broker, "000990", "DB하이텍")

        # 조건 검증
        assert dec.bullish_candle is True
        assert dec.volume_surge is True  # 200 vs 100 평균 = 2x
        assert dec.vwap_recovery is True  # 최근 105가 VWAP 위
        assert dec.pass_count >= 3
        assert dec.trigger is True

    def test_volume_surge_threshold(self):
        """거래량 1.5배 미달 → volume_surge=False."""
        rows = []
        # 현재 5분봉 거래량 100 (직전 100의 1x = 1.5x 미달)
        for t in ["091400", "091300", "091200", "091100", "091000"]:
            rows.append(make_1m_row(t, 100, 102, 99, 101, 20))
        for t in ["090900", "090800", "090700", "090600", "090500"]:
            rows.append(make_1m_row(t, 100, 102, 99, 100, 20))
        broker = make_broker(rows)
        dec = evaluate_intraday_entry(broker, "000990")
        assert dec.volume_surge is False

    def test_telegram_format_no_trigger(self):
        """trigger=False면 빈 문자열 반환 (알림 안 보냄)."""
        dec = IntradayEntryDecision(ticker="000990", name="DB하이텍", trigger=False)
        assert format_for_telegram(dec) == ""

    def test_telegram_format_with_trigger(self):
        """trigger=True면 메시지 포맷팅."""
        dec = IntradayEntryDecision(
            ticker="000990", name="DB하이텍",
            trigger=True, pass_count=3,
            current_price=205000, five_min_open=204000, five_min_close=205000,
            reasons_pass=["양봉", "거래량 급증"],
            reasons_fail=["RSI 과매수"],
        )
        msg = format_for_telegram(dec)
        assert "DB하이텍" in msg
        assert "000990" in msg
        assert "3/4" in msg
        assert "양봉" in msg
