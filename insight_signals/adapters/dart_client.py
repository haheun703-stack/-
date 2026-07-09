# -*- coding: utf-8 -*-
"""DART 전자공시 OpenAPI 클라이언트.

사용 API (모두 무료, 인증키 필요 — https://opendart.fss.or.kr):
  - corpCode.xml : 전체 회사 고유번호 <-> 종목코드 매핑 (zip)
  - list.json    : 공시 검색. pblntf_detail_ty=D002
                   (임원ㆍ주요주주 특정증권등 소유상황보고서)
  - elestock.json: 회사별 임원·주요주주 소유 보고 상세 (증감 수량 포함)
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
import zipfile
import xml.etree.ElementTree as ET

import requests

from insight_signals.entities import InsiderFiling

log = logging.getLogger("insight_signals.dart")

BASE = "https://opendart.fss.or.kr/api"
CORP_CACHE_TTL_SEC = 7 * 24 * 3600  # 회사 목록은 주 1회 갱신이면 충분


def _to_int(v) -> int:
    """DART 숫자 필드는 '1,234' / '-' / '' 형태가 섞여 있다."""
    if v is None:
        return 0
    s = str(v).strip().replace(",", "")
    if s in ("", "-"):
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


class DartClient:
    def __init__(self, api_key: str, cache_dir: str, timeout: int = 20):
        if not api_key:
            raise ValueError("DART API 키가 없습니다 (.env의 DART_API_KEY 확인)")
        self.key = api_key
        self.cache_dir = cache_dir
        self.timeout = timeout
        os.makedirs(cache_dir, exist_ok=True)
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # 회사 고유번호 <-> 종목코드 매핑
    # ------------------------------------------------------------------
    def corp_map(self) -> dict:
        """{corp_code: {"name": 회사명, "stock_code": 6자리 or ""}} (상장사 위주 사용)"""
        cache = os.path.join(self.cache_dir, "corp_map.json")
        if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < CORP_CACHE_TTL_SEC:
            with open(cache, encoding="utf-8") as f:
                return json.load(f)

        log.info("DART corpCode.xml 다운로드 (회사 목록 갱신)")
        r = self._session.get(
            f"{BASE}/corpCode.xml", params={"crtfc_key": self.key}, timeout=60
        )
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        xml_bytes = zf.read(zf.namelist()[0])
        root = ET.fromstring(xml_bytes)

        out = {}
        for el in root.iter("list"):
            corp_code = (el.findtext("corp_code") or "").strip()
            name = (el.findtext("corp_name") or "").strip()
            stock = (el.findtext("stock_code") or "").strip()
            if corp_code:
                out[corp_code] = {"name": name, "stock_code": stock}

        with open(cache, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        log.info("회사 목록 %d건 캐시 저장", len(out))
        return out

    def listed_name_map(self) -> dict:
        """상장사만: {회사명: 종목코드}. 뉴스 제목 -> 종목 매핑에 사용."""
        return {
            v["name"]: v["stock_code"]
            for v in self.corp_map().values()
            if v["stock_code"]
        }

    # ------------------------------------------------------------------
    # 임원·주요주주 소유상황보고 (D002)
    # ------------------------------------------------------------------
    def insider_filing_corps(self, bgn_de: str, end_de: str) -> list:
        """기간 내 D002 공시를 낸 회사 목록.

        Returns: [{"corp_code":..., "corp_name":..., "stock_code":...,
                   "rcept_no":..., "rcept_dt":..., "flr_nm": 제출인}, ...]
        """
        out, page = [], 1
        while True:
            r = self._session.get(
                f"{BASE}/list.json",
                params={
                    "crtfc_key": self.key,
                    "bgn_de": bgn_de,
                    "end_de": end_de,
                    "pblntf_detail_ty": "D002",
                    "page_no": page,
                    "page_count": 100,
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            status = data.get("status")
            if status == "013":  # 조회 결과 없음
                break
            if status != "000":
                log.warning("DART list.json 오류 status=%s msg=%s", status, data.get("message"))
                break
            for item in data.get("list", []):
                out.append(
                    {
                        "corp_code": item.get("corp_code", ""),
                        "corp_name": item.get("corp_name", ""),
                        "stock_code": item.get("stock_code", "") or "",
                        "rcept_no": item.get("rcept_no", ""),
                        "rcept_dt": item.get("rcept_dt", ""),
                        "flr_nm": item.get("flr_nm", ""),
                    }
                )
            if page >= int(data.get("total_page", 1)):
                break
            page += 1
            time.sleep(0.15)  # rate limit 예방 (분당 1000회 제한이지만 여유 있게)
        log.info("D002 공시 %d건 (%s ~ %s)", len(out), bgn_de, end_de)
        return out

    def insider_details(self, corp_code: str, since_yyyymmdd: str) -> list:
        """회사별 elestock.json에서 since 이후 보고 건들의 증감 내역을 InsiderFiling으로.

        elestock 응답 주요 필드:
          rcept_no, rcept_dt(YYYY.MM.DD 또는 YYYYMMDD), repror(보고자),
          isu_exctv_rgist_at(등기임원 여부), isu_exctv_ofcps(직위),
          sp_stock_lmp_irds_cnt(증감 수량), sp_stock_lmp_cnt(소유 수량)
        """
        r = self._session.get(
            f"{BASE}/elestock.json",
            params={"crtfc_key": self.key, "corp_code": corp_code},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "000":
            return []

        cmap = self.corp_map()
        info = cmap.get(corp_code, {"name": "", "stock_code": ""})
        out = []
        for row in data.get("list", []):
            dt = (row.get("rcept_dt") or "").replace(".", "").replace("-", "")[:8]
            if dt < since_yyyymmdd:
                continue
            qty = _to_int(row.get("sp_stock_lmp_irds_cnt"))
            if qty == 0:
                continue
            out.append(
                InsiderFiling(
                    rcept_no=row.get("rcept_no", ""),
                    rcept_dt=dt,
                    corp_code=corp_code,
                    corp_name=row.get("corp_name", "") or info["name"],
                    stock_code=info["stock_code"],
                    reporter=row.get("repror", ""),
                    position=(row.get("isu_exctv_ofcps") or row.get("isu_main_shrholdr") or "").strip(),
                    change_qty=qty,
                    change_reason=(row.get("sp_stock_lmp_irds_rate") or ""),
                )
            )
        return out
