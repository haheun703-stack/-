"""
Phase 1: KIS 장중 데이터 수집 어댑터

IntradayDataPort 구현체.
mojito 브로커 객체를 재활용하면서, mojito에 없는 API는 직접 호출한다.

수집 대상:
  - 1분 틱/현재가: mojito fetch_price
  - 1분봉: mojito _fetch_today_1m_ohlcv (단일 호출 = 최근 ~30건)
  - 투자자별 매매동향: KIS API FHKST01010900 직접 호출
  - 시장 지수: KIS API FHPUP02100000 / fetch_price(index)
  - 업종별 시세: KIS API 직접 호출
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import mojito
import requests

from src.use_cases.ports import IntradayDataPort

logger = logging.getLogger(__name__)

# KIS API Rate limit 보호 (초당 최대 20건)
_last_call_time = 0.0
MIN_CALL_INTERVAL = 0.06  # ~16 req/sec


def _rate_limit():
    """API Rate limit 보호"""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - elapsed)
    _last_call_time = time.time()


class KisIntradayAdapter(IntradayDataPort):
    """KIS REST API 기반 장중 데이터 수집 어댑터"""

    def __init__(self, broker: mojito.KoreaInvestment | None = None):
        if broker is not None:
            self.broker = broker
        else:
            is_mock = os.getenv("MODEL") != "REAL"
            self.broker = mojito.KoreaInvestment(
                api_key=os.getenv("KIS_APP_KEY", ""),
                api_secret=os.getenv("KIS_APP_SECRET", ""),
                acc_no=os.getenv("KIS_ACC_NO", ""),
                mock=is_mock,
            )
        logger.info("[KIS장중] 어댑터 초기화 완료")

    # ──────────────────────────────────────────
    # 내부 헬퍼: 직접 API 호출
    # ──────────────────────────────────────────

    def _api_get(self, path: str, tr_id: str, params: dict) -> dict:
        """KIS REST API GET 호출 (mojito에 없는 엔드포인트용)"""
        _rate_limit()
        url = f"{self.broker.base_url}/{path}"
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": self.broker.access_token,
            "appKey": self.broker.api_key,
            "appSecret": self.broker.api_secret,
            "tr_id": tr_id,
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            rt_cd = data.get("rt_cd", "1")
            if rt_cd != "0":
                msg = data.get("msg1", data.get("msg_cd", ""))
                logger.warning("[KIS API] %s 실패: %s", tr_id, msg)
            return data
        except Exception as e:
            logger.error("[KIS API] %s 호출 오류: %s", tr_id, e)
            return {}

    # ──────────────────────────────────────────
    # IntradayDataPort 구현
    # ──────────────────────────────────────────

    def fetch_tick(self, ticker: str) -> dict:
        """1분 단위 현재가/체결 데이터 조회"""
        _rate_limit()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:00")
        try:
            data = self.broker.fetch_price(ticker)
            output = data.get("output", {})
            return {
                "ticker": ticker,
                "timestamp": now_str,
                "current_price": int(output.get("stck_prpr", 0)),
                "open_price": int(output.get("stck_oprc", 0)),
                "high_price": int(output.get("stck_hgpr", 0)),
                "low_price": int(output.get("stck_lwpr", 0)),
                "volume": int(output.get("acml_vol", 0)),
                "cum_volume": int(output.get("acml_vol", 0)),
                "change_pct": float(output.get("prdy_ctrt", 0)),
                "bid_price": int(output.get("bidp1", output.get("stck_prpr", 0))),
                "ask_price": int(output.get("askp1", output.get("stck_prpr", 0))),
                "strength": float(output.get("seln_cntg_smtn", 0)),
            }
        except Exception as e:
            logger.error("[KIS장중] %s 틱 조회 실패: %s", ticker, e)
            return {"ticker": ticker, "timestamp": now_str, "current_price": 0}

    def fetch_full_day_1m_candles(self, ticker: str, date_str: str | None = None) -> list[dict]:
        """
        당일 전체 1분봉 수집 (페이지네이션).

        KIS API는 1회 호출당 ~30건만 반환하므로,
        to_time을 역방향으로 이동하며 전체 장중 데이터를 수집한다.

        Args:
            ticker: 종목코드 (예: "005930")
            date_str: 날짜 (YYYY-MM-DD). None이면 오늘.

        Returns:
            시간순 정렬된 1분봉 리스트
            [{"timestamp": "2026-02-19 09:01:00", "open": ..., "high": ..., ...}, ...]
        """
        date_prefix = date_str or datetime.now().strftime("%Y-%m-%d")
        to_time = "153000"
        all_candles = {}  # timestamp → candle (중복 방지)
        max_pages = 15  # 안전 장치 (390분 / 30건 ≈ 13페이지)

        for _ in range(max_pages):
            _rate_limit()
            try:
                data = self.broker._fetch_today_1m_ohlcv(ticker, to_time)
                raw = data.get("output2", [])
                if not raw:
                    break

                earliest_hour = None
                for c in raw:
                    hour = c.get("stck_cntg_hour", "")
                    if len(hour) < 6:
                        continue
                    ts = f"{date_prefix} {hour[:2]}:{hour[2:4]}:00"
                    if ts not in all_candles:
                        all_candles[ts] = {
                            "timestamp": ts,
                            "open": int(c.get("stck_oprc", 0)),
                            "high": int(c.get("stck_hgpr", 0)),
                            "low": int(c.get("stck_lwpr", 0)),
                            "close": int(c.get("stck_prpr", 0)),
                            "volume": int(c.get("cntg_vol", 0)),
                        }
                    if earliest_hour is None or hour < earliest_hour:
                        earliest_hour = hour

                # 09:00 이전이면 전체 수집 완료
                if earliest_hour is None or earliest_hour <= "090000":
                    break

                # 다음 페이지: 가장 이른 시각에서 1분 빼기
                h, m = int(earliest_hour[:2]), int(earliest_hour[2:4])
                m -= 1
                if m < 0:
                    m = 59
                    h -= 1
                if h < 9:
                    break
                to_time = f"{h:02d}{m:02d}00"

            except Exception as e:
                logger.error("[KIS장중] %s 전체1분봉 수집 오류 (page to=%s): %s", ticker, to_time, e)
                break

        result = sorted(all_candles.values(), key=lambda x: x["timestamp"])
        return result

    def fetch_minute_candles(self, ticker: str, period: int = 5) -> list[dict]:
        """
        최근 N분봉 데이터 조회.
        KIS API는 1분봉만 제공하므로, 1분봉을 가져온 후 N분 단위로 집계한다.
        """
        _rate_limit()
        now = datetime.now()
        to_time = now.strftime("%H%M%S")
        if to_time > "153000":
            to_time = "153000"

        try:
            data = self.broker._fetch_today_1m_ohlcv(ticker, to_time)
            raw_candles = data.get("output2", [])
            if not raw_candles:
                return []

            # 1분봉 → N분봉 집계
            one_min = []
            for c in raw_candles:
                hour = c.get("stck_cntg_hour", "")
                if len(hour) < 6:
                    continue
                ts = f"{now.strftime('%Y-%m-%d')} {hour[:2]}:{hour[2:4]}:00"
                one_min.append({
                    "timestamp": ts,
                    "open": int(c.get("stck_oprc", 0)),
                    "high": int(c.get("stck_hgpr", 0)),
                    "low": int(c.get("stck_lwpr", 0)),
                    "close": int(c.get("stck_prpr", 0)),
                    "volume": int(c.get("cntg_vol", 0)),
                })

            if period == 1:
                return [{"ticker": ticker, **c} for c in one_min]

            return self._aggregate_candles(ticker, one_min, period)

        except Exception as e:
            logger.error("[KIS장중] %s %d분봉 조회 실패: %s", ticker, period, e)
            return []

    def _aggregate_candles(
        self, ticker: str, one_min: list[dict], period: int,
    ) -> list[dict]:
        """1분봉을 N분봉으로 집계"""
        if not one_min:
            return []

        # 시간순 정렬
        one_min.sort(key=lambda x: x["timestamp"])

        candles = []
        bucket = []

        for c in one_min:
            # 분 단위 추출
            ts = c["timestamp"]
            minute = int(ts[14:16])
            bucket_start = (minute // period) * period

            if bucket and int(bucket[0]["timestamp"][14:16]) // period * period != bucket_start:
                candles.append(self._merge_bucket(ticker, bucket, period))
                bucket = []

            bucket.append(c)

        if bucket:
            candles.append(self._merge_bucket(ticker, bucket, period))

        return candles

    @staticmethod
    def _merge_bucket(ticker: str, bucket: list[dict], period: int) -> dict:
        """1분봉 버킷을 하나의 N분봉으로 합산"""
        first = bucket[0]
        total_vol = sum(c["volume"] for c in bucket)
        total_value = sum(c["close"] * c["volume"] for c in bucket)
        vwap = total_value / total_vol if total_vol > 0 else first["close"]

        return {
            "ticker": ticker,
            "timestamp": first["timestamp"],
            "open": first["open"],
            "high": max(c["high"] for c in bucket),
            "low": min(c["low"] for c in bucket),
            "close": bucket[-1]["close"],
            "volume": total_vol,
            "vwap": round(vwap, 2),
        }

    def fetch_investor_flow(self, ticker: str) -> dict:
        """
        투자자별 매매동향 조회 (당일 누적).
        KIS API: FHKST01010900 (주식현재가 투자자)
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:00")
        data = self._api_get(
            path="uapi/domestic-stock/v1/quotations/inquire-investor",
            tr_id="FHKST01010900",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": ticker,
            },
        )

        # output은 dict 또는 list[dict]일 수 있음
        raw_output = data.get("output", {})
        if isinstance(raw_output, list):
            output = raw_output[0] if raw_output else {}
        else:
            output = raw_output

        if not output:
            return {
                "ticker": ticker,
                "timestamp": now_str,
                "foreign_net_buy": 0,
                "inst_net_buy": 0,
                "individual_net_buy": 0,
                "foreign_cum_net": 0,
                "inst_cum_net": 0,
                "program_net_buy": 0,
            }

        # KIS 응답 파싱 — 키가 다를 수 있으므로 안전하게 처리
        foreign_buy = int(output.get("frgn_ntby_qty", 0) or 0)
        inst_buy = int(output.get("orgn_ntby_qty", 0) or 0)
        individual_buy = int(output.get("prsn_ntby_qty", 0) or 0)

        return {
            "ticker": ticker,
            "timestamp": now_str,
            "foreign_net_buy": foreign_buy,
            "inst_net_buy": inst_buy,
            "individual_net_buy": individual_buy,
            "foreign_cum_net": foreign_buy,
            "inst_cum_net": inst_buy,
            "program_net_buy": 0,
        }

    def fetch_market_index(self) -> dict:
        """
        시장 지수 조회 (KOSPI, KOSDAQ).
        KIS API: FHPUP02100000 (업종/지수 현재가)
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:00")

        result = {
            "timestamp": now_str,
            "kospi": 0.0,
            "kospi_change_pct": 0.0,
            "kosdaq": 0.0,
            "kosdaq_change_pct": 0.0,
            "usd_krw": 0.0,
            "us_futures": 0.0,
            "vix": 0.0,
            "bond_yield_kr_3y": 0.0,
            "kospi_volume": 0,
            "kosdaq_volume": 0,
        }

        # KOSPI (0001)
        kospi_data = self._api_get(
            path="uapi/domestic-stock/v1/quotations/inquire-index-price",
            tr_id="FHPUP02100000",
            params={
                "FID_COND_MRKT_DIV_CODE": "U",
                "FID_INPUT_ISCD": "0001",
            },
        )
        kospi_out = kospi_data.get("output", {})
        if kospi_out:
            result["kospi"] = float(kospi_out.get("bstp_nmix_prpr", 0))
            result["kospi_change_pct"] = float(kospi_out.get("bstp_nmix_prdy_ctrt", 0))
            result["kospi_volume"] = int(
                float(kospi_out.get("acml_tr_pbmn", 0)) / 100_000_000
            )  # 원 → 억

        # KOSDAQ (1001)
        kosdaq_data = self._api_get(
            path="uapi/domestic-stock/v1/quotations/inquire-index-price",
            tr_id="FHPUP02100000",
            params={
                "FID_COND_MRKT_DIV_CODE": "U",
                "FID_INPUT_ISCD": "1001",
            },
        )
        kosdaq_out = kosdaq_data.get("output", {})
        if kosdaq_out:
            result["kosdaq"] = float(kosdaq_out.get("bstp_nmix_prpr", 0))
            result["kosdaq_change_pct"] = float(kosdaq_out.get("bstp_nmix_prdy_ctrt", 0))
            result["kosdaq_volume"] = int(
                float(kosdaq_out.get("acml_tr_pbmn", 0)) / 100_000_000
            )

        return result

    def fetch_sector_prices(self) -> list[dict]:
        """
        주요 업종 지수 조회.
        KIS 업종 코드 매핑을 사용하여 주요 업종만 조회.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:00")

        # 주요 업종 코드 (KIS 기준)
        sector_codes = {
            "0003": "반도체",
            "0007": "자동차",
            "0010": "철강",
            "0011": "화학",
            "0012": "의약품",
            "0015": "운수장비",
            "0017": "전기전자",
            "0019": "유통업",
            "0022": "은행",
            "0024": "증권",
            "0025": "보험",
            "0028": "건설업",
        }

        sectors = []
        for code, name in sector_codes.items():
            data = self._api_get(
                path="uapi/domestic-stock/v1/quotations/inquire-index-price",
                tr_id="FHPUP02100000",
                params={
                    "FID_COND_MRKT_DIV_CODE": "U",
                    "FID_INPUT_ISCD": code,
                },
            )
            output = data.get("output", {})
            if output:
                sectors.append({
                    "timestamp": now_str,
                    "sector_code": code,
                    "sector_name": name,
                    "index_value": float(output.get("bstp_nmix_prpr", 0)),
                    "change_pct": float(output.get("bstp_nmix_prdy_ctrt", 0)),
                    "volume": int(
                        float(output.get("acml_tr_pbmn", 0)) / 100_000_000
                    ),
                    "advance_count": int(output.get("stck_prpr_updn", 0)),
                    "decline_count": 0,
                })

        return sectors

    # ──────────────────────────────────────────
    # 호가창 (10호가)
    # ──────────────────────────────────────────

    def fetch_orderbook(self, ticker: str) -> dict:
        """
        10호가 (매수/매도 잔량) 조회.
        KIS API: FHKST01010200 (주식현재가 호가/예상체결)
        """
        data = self._api_get(
            path="uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
            tr_id="FHKST01010200",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": ticker,
            },
        )
        output1 = data.get("output1", {})
        output2 = data.get("output2", {})

        if not output1 and not output2:
            return {"ticker": ticker, "asks": [], "bids": [],
                    "total_ask_vol": 0, "total_bid_vol": 0, "bid_ask_ratio": 0}

        out = output1 or output2

        asks = []  # 매도호가 (낮은→높은)
        bids = []  # 매수호가 (높은→낮은)

        for i in range(1, 11):
            ask_p = int(out.get(f"askp{i}", 0) or 0)
            ask_v = int(out.get(f"askp_rsqn{i}", 0) or 0)
            bid_p = int(out.get(f"bidp{i}", 0) or 0)
            bid_v = int(out.get(f"bidp_rsqn{i}", 0) or 0)
            if ask_p > 0:
                asks.append({"price": ask_p, "volume": ask_v})
            if bid_p > 0:
                bids.append({"price": bid_p, "volume": bid_v})

        total_ask = int(out.get("total_askp_rsqn", 0) or 0)
        total_bid = int(out.get("total_bidp_rsqn", 0) or 0)
        ratio = round(total_bid / total_ask, 2) if total_ask > 0 else 0

        return {
            "ticker": ticker,
            "asks": asks,
            "bids": bids,
            "total_ask_vol": total_ask,
            "total_bid_vol": total_bid,
            "bid_ask_ratio": ratio,
        }

    # ──────────────────────────────────────────
    # 편의 메서드
    # ──────────────────────────────────────────

    def fetch_ticks_batch(self, tickers: list[str]) -> list[dict]:
        """여러 종목 틱 데이터 일괄 조회"""
        results = []
        for ticker in tickers:
            tick = self.fetch_tick(ticker)
            if tick.get("current_price", 0) > 0:
                results.append(tick)
        return results

    def is_market_open(self) -> bool:
        """장 운영 시간 여부 (09:00~15:30 KST)"""
        now = datetime.now()
        hour_min = now.hour * 100 + now.minute
        # 장 시작 08:30(동시호가) ~ 15:30(정규 마감)
        return 830 <= hour_min <= 1530

    def get_api_call_count(self) -> int:
        """(디버깅용) 세션 중 API 호출 횟수 — 미구현, 추후 추가"""
        return 0
