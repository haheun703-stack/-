# -*- coding: utf-8 -*-
"""KIS(한국투자증권) OpenAPI — 투자자별 매매동향 + 현재가.

키가 없거나 호출 실패 시 조용히 None을 반환해 전체 파이프라인은 계속 돈다.
토큰은 파일 캐시(약 24시간 유효, 재발급 제한이 있으므로 반드시 캐시 사용).

주의: 시세/수급 조회는 실전 도메인 키 기준. 모의(paper) 키만 있으면
config에서 base_url을 모의 도메인으로 바꿔야 하며 일부 TR은 미지원일 수 있음.
"""
from __future__ import annotations

import json
import logging
import os
import time

import requests

from insight_signals.entities import FlowSnapshot

log = logging.getLogger("insight_signals.kis")

REAL_BASE = "https://openapi.koreainvestment.com:9443"

TR_INVESTOR = "FHKST01010900"   # 주식현재가 투자자
TR_PRICE = "FHKST01010100"      # 주식현재가 시세


def _to_int(v) -> int:
    try:
        return int(str(v).replace(",", "").strip() or 0)
    except ValueError:
        return 0


class KisFlowClient:
    def __init__(self, app_key: str, app_secret: str, cache_dir: str,
                 base_url: str = REAL_BASE, timeout: int = 10,
                 token_cache_path: str = ""):
        """token_cache_path: 기존 봇의 토큰 캐시 파일 경로를 지정하면 공유한다.

        KIS 토큰은 발급 횟수 제한이 있어(기존 봇과 각자 발급하면 rate limit 충돌)
        반드시 하나의 캐시를 같이 쓰는 것이 안전하다.
        - 우리 포맷({"access_token", "issued_at"})과
          KIS 공식 예제 포맷({"token", "valid-date"}) 모두 읽는다.
        - 남의 포맷 파일은 절대 덮어쓰지 않는다 (기존 봇 캐시 보호).
        """
        self.app_key = app_key
        self.app_secret = app_secret
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._token_file = token_cache_path or os.path.join(cache_dir, "kis_token.json")
        self._foreign_format = False  # 남의 포맷이면 쓰기 금지
        self._session = requests.Session()

    @property
    def available(self) -> bool:
        return bool(self.app_key and self.app_secret)

    # ------------------------------------------------------------------
    def _read_cached_token(self):
        """우리 포맷 + KIS 공식 예제 포맷 모두 지원해서 캐시 토큰을 읽는다."""
        try:
            with open(self._token_file, encoding="utf-8") as f:
                cached = json.load(f)
        except (OSError, ValueError):
            return None
        # 우리 포맷: {"access_token": ..., "issued_at": epoch}
        tok = cached.get("access_token")
        if tok and time.time() - cached.get("issued_at", 0) < 23 * 3600:
            return tok
        # KIS 공식 예제 포맷: {"token": ..., "valid-date": "YYYY-MM-DD HH:MM:SS"}
        tok = cached.get("token")
        valid = cached.get("valid-date") or cached.get("valid_date")
        if tok and valid:
            self._foreign_format = True
            try:
                exp = time.strptime(str(valid)[:19], "%Y-%m-%d %H:%M:%S")
                if time.mktime(exp) - time.time() > 600:  # 10분 여유
                    return tok
            except ValueError:
                pass
        elif cached and "access_token" not in cached:
            self._foreign_format = True
        return None

    def _token(self):
        if not self.available:
            return None
        tok = self._read_cached_token()
        if tok:
            return tok
        try:
            r = self._session.post(
                f"{self.base}/oauth2/tokenP",
                json={
                    "grant_type": "client_credentials",
                    "appkey": self.app_key,
                    "appsecret": self.app_secret,
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            tok = r.json().get("access_token")
            if tok and not self._foreign_format:
                # 남의 포맷(기존 봇 캐시)은 덮어쓰지 않는다
                with open(self._token_file, "w", encoding="utf-8") as f:
                    json.dump({"access_token": tok, "issued_at": time.time()}, f)
            return tok
        except Exception as e:  # noqa: BLE001
            log.warning("KIS 토큰 발급 실패: %s", e)
            return None

    def _headers(self, tr_id: str):
        tok = self._token()
        if not tok:
            return None
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {tok}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    # ------------------------------------------------------------------
    def investor_flow(self, stock_code: str, days: int = 3):
        """최근 N일 투자자별 순매수 합계. 실패 시 None."""
        headers = self._headers(TR_INVESTOR)
        if headers is None:
            return None
        try:
            r = self._session.get(
                f"{self.base}/uapi/domestic-stock/v1/quotations/inquire-investor",
                headers=headers,
                params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": stock_code},
                timeout=self.timeout,
            )
            r.raise_for_status()
            rows = (r.json().get("output") or [])[:days]
            if not rows:
                return None
            return FlowSnapshot(
                stock_code=stock_code,
                days=len(rows),
                person_net=sum(_to_int(x.get("prsn_ntby_qty")) for x in rows),
                foreign_net=sum(_to_int(x.get("frgn_ntby_qty")) for x in rows),
                org_net=sum(_to_int(x.get("orgn_ntby_qty")) for x in rows),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("KIS 수급 조회 실패 [%s]: %s", stock_code, e)
            return None

    def current_price(self, stock_code: str):
        """현재가. 실패 시 None."""
        headers = self._headers(TR_PRICE)
        if headers is None:
            return None
        try:
            r = self._session.get(
                f"{self.base}/uapi/domestic-stock/v1/quotations/inquire-price",
                headers=headers,
                params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": stock_code},
                timeout=self.timeout,
            )
            r.raise_for_status()
            out = r.json().get("output") or {}
            price = _to_int(out.get("stck_prpr"))
            return float(price) if price else None
        except Exception as e:  # noqa: BLE001
            log.warning("KIS 현재가 조회 실패 [%s]: %s", stock_code, e)
            return None
