from __future__ import annotations

from datetime import date

from scripts import collect_foreign_exhaustion as cfe


NAVER_FRGN_HTML = """
<html>
  <body>
    <table>
      <thead>
        <tr>
          <th>날짜</th><th>종가</th><th>전일비</th><th>등락률</th><th>거래량</th>
          <th>기관</th><th>외국인</th><th>보유주식수</th><th>보유율</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>2026.06.04</td><td>351,500</td><td>-9,000</td><td>-2.50%</td><td>12,345,678</td><td>10,000</td><td>-20,000</td><td>2,795,254,819</td><td>47.81%</td></tr>
        <tr><td>2026.06.02</td><td>360,500</td><td>11,500</td><td>3.30%</td><td>10,000,000</td><td>20,000</td><td>-10,000</td><td>2,810,201,369</td><td>48.07%</td></tr>
        <tr><td>2026.06.01</td><td>349,000</td><td>3,000</td><td>0.87%</td><td>9,000,000</td><td>30,000</td><td>40,000</td><td>2,823,815,351</td><td>48.30%</td></tr>
        <tr><td>2026.05.29</td><td>346,000</td><td>1,000</td><td>0.29%</td><td>8,000,000</td><td>1,000</td><td>2,000</td><td>2,780,000,000</td><td>47.50%</td></tr>
        <tr><td>2026.05.28</td><td>345,000</td><td>2,000</td><td>0.58%</td><td>7,000,000</td><td>1,000</td><td>2,000</td><td>2,760,000,000</td><td>47.10%</td></tr>
        <tr><td>2026.05.27</td><td>343,000</td><td>1,000</td><td>0.29%</td><td>6,000,000</td><td>1,000</td><td>2,000</td><td>2,500,000,000</td><td>42.70%</td></tr>
      </tbody>
    </table>
  </body>
</html>
"""


class FakeResponse:
    encoding = "euc-kr"
    text = NAVER_FRGN_HTML

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def get(self, *args, **kwargs) -> FakeResponse:
        return FakeResponse()


def test_parse_naver_frgn_html_uses_real_trading_date() -> None:
    rows = cfe._parse_naver_frgn_html("005930", NAVER_FRGN_HTML)

    assert rows[0]["date"] == "2026-06-04"
    assert rows[0]["close"] == 351500
    assert rows[0]["foreign_holding"] == 2795254819
    assert rows[0]["holding_ratio"] == 47.81
    assert "2026-06-03" not in {row["date"] for row in rows}


def test_fetch_naver_builds_prev_rate_and_signal_as_of_date() -> None:
    df, prev_rates = cfe._fetch_exhaustion_naver(
        ["005930"],
        session=FakeSession(),
        sleep_sec=0,
    )
    signals = cfe._compute_signals(df, prev_rates, {"005930"})

    assert df.loc["005930", "기준일"] == "2026-06-04"
    assert prev_rates["005930"] == 42.70
    assert signals[0]["as_of_date"] == "2026-06-04"
    assert signals[0]["exhaustion_rate"] == 47.81
    assert signals[0]["exhaustion_5d_change"] == 5.11


def test_universe_includes_sector_fire_leverage_etf() -> None:
    universe = cfe._get_universe()

    assert "488080" in universe


def test_trading_day_guard_uses_kr_holiday_calendar() -> None:
    assert cfe._is_trading_day(date(2026, 6, 3)) is False  # 지방선거 임시공휴일
    assert cfe._is_trading_day(date(2026, 5, 1)) is False  # 근로자의날 증시 휴장
    assert cfe._is_trading_day(date(2026, 6, 4)) is True
