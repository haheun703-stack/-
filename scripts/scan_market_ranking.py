#!/usr/bin/env python
"""KIS 국내종목순위 수집기 — 5개 API 장중 순위 데이터

수집 대상:
  1) 거래량순위 (FHPST01710000) — 폭발 거래량 감지
  2) 등락률순위 (FHPST01700000) — 상한가/급등 포착
  3) 체결강도상위 (FHPST01680000) — 매수 세력 감지
  4) 외국인/기관 가집계 (FHPTJ04400000) — 실시간 수급
  5) 상하한가 포착 (FHKST130000C0) — 상한가 직접 감지

스케줄: BAT-D 장후 (장중 마지막 스냅샷)
출력: data/market_ranking.json → quant_market_ranking 테이블

Usage:
    python -u -X utf8 scripts/scan_market_ranking.py
    python -u -X utf8 scripts/scan_market_ranking.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import mojito
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "market_ranking.json"

# Rate limit 보호 (초당 최대 20건)
_last_call_ts = 0.0
MIN_INTERVAL = 0.08  # ~12 req/sec (여유)


# ═══════════════════════════════════════════════════
# KIS API 호출
# ═══════════════════════════════════════════════════

def _get_broker() -> mojito.KoreaInvestment:
    """mojito 브로커 생성."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    return mojito.KoreaInvestment(
        api_key=os.getenv("KIS_APP_KEY", ""),
        api_secret=os.getenv("KIS_APP_SECRET", ""),
        acc_no=os.getenv("KIS_ACC_NO", ""),
        mock=os.getenv("MODEL") != "REAL",
    )


def _api_get(broker: mojito.KoreaInvestment, path: str,
             tr_id: str, params: dict) -> list[dict]:
    """KIS REST API GET → output 리스트 반환."""
    global _last_call_ts
    elapsed = time.time() - _last_call_ts
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_call_ts = time.time()

    url = f"{broker.base_url}/{path}"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": broker.access_token,
        "appKey": broker.api_key,
        "appSecret": broker.api_secret,
        "tr_id": tr_id,
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            msg = data.get("msg1", data.get("msg_cd", ""))
            logger.warning("[%s] API 응답 오류: %s", tr_id, msg)
            return []

        return data.get("output", [])
    except Exception as e:
        logger.error("[%s] API 호출 실패: %s", tr_id, e)
        return []


# ═══════════════════════════════════════════════════
# 1. 거래량순위 (FHPST01710000)
# ═══════════════════════════════════════════════════

def fetch_volume_rank(broker: mojito.KoreaInvestment,
                      top_n: int = 30) -> list[dict]:
    """거래량순위 — 폭발 거래량 상위 종목."""
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_COND_SCR_DIV_CODE": "20171",
        "FID_INPUT_ISCD": "0000",
        "FID_DIV_CLS_CODE": "0",         # 전체
        "FID_BLNG_CLS_CODE": "0",        # 평균거래량순
        "FID_TRGT_CLS_CODE": "111111111",
        "FID_TRGT_EXLS_CLS_CODE": "0000000000",
        "FID_INPUT_PRICE_1": "",
        "FID_INPUT_PRICE_2": "",
        "FID_VOL_CNT": "",
        "FID_INPUT_DATE_1": "",
    }

    items = _api_get(broker, "uapi/domestic-stock/v1/quotations/volume-rank",
                     "FHPST01710000", params)

    results = []
    for item in items[:top_n]:
        code = (item.get("mksc_shrn_iscd") or "").strip()
        name = (item.get("hts_kor_isnm") or "").strip()
        if not code or not name:
            continue
        results.append({
            "rank": int(item.get("data_rank", 0) or 0),
            "code": code,
            "name": name,
            "price": int(item.get("stck_prpr", 0) or 0),
            "change_pct": float(item.get("prdy_ctrt", 0) or 0),
            "volume": int(item.get("acml_vol", 0) or 0),
            "volume_rate": float(item.get("vol_inrt", 0) or 0),
            "turnover": float(item.get("vol_tnrt", 0) or 0),
            "amount_억": round(int(item.get("acml_tr_pbmn", 0) or 0) / 1_0000_0000, 1),
        })

    logger.info("거래량순위: %d종목", len(results))
    return results


# ═══════════════════════════════════════════════════
# 2. 등락률순위 (FHPST01700000)
# ═══════════════════════════════════════════════════

def fetch_fluctuation_rank(broker: mojito.KoreaInvestment,
                           top_n: int = 30) -> list[dict]:
    """등락률순위 — 상한가/급등 포착."""
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_COND_SCR_DIV_CODE": "20170",
        "FID_INPUT_ISCD": "0000",
        "FID_RANK_SORT_CLS_CODE": "0",   # 등락률순
        "FID_INPUT_CNT_1": "0",
        "FID_PRC_CLS_CODE": "0",
        "FID_INPUT_PRICE_1": "",
        "FID_INPUT_PRICE_2": "",
        "FID_VOL_CNT": "",
        "FID_TRGT_CLS_CODE": "111111111",
        "FID_TRGT_EXLS_CLS_CODE": "0000000000",
        "FID_DIV_CLS_CODE": "0",
        "FID_RSFL_RATE1": "",
        "FID_RSFL_RATE2": "",
    }

    items = _api_get(broker, "uapi/domestic-stock/v1/ranking/fluctuation",
                     "FHPST01700000", params)

    results = []
    for item in items[:top_n]:
        code = (item.get("stck_shrn_iscd") or "").strip()
        name = (item.get("hts_kor_isnm") or "").strip()
        if not code or not name:
            continue
        sign = item.get("prdy_vrss_sign", "3")
        change = int(item.get("prdy_vrss", 0) or 0)
        if sign in ("4", "5"):
            change = -abs(change)
        results.append({
            "rank": int(item.get("data_rank", 0) or 0),
            "code": code,
            "name": name,
            "price": int(item.get("stck_prpr", 0) or 0),
            "change": change,
            "change_pct": float(item.get("prdy_ctrt", 0) or 0),
            "volume": int(item.get("acml_vol", 0) or 0),
        })

    logger.info("등락률순위: %d종목", len(results))
    return results


# ═══════════════════════════════════════════════════
# 3. 체결강도상위 (FHPST01680000)
# ═══════════════════════════════════════════════════

def fetch_volume_power(broker: mojito.KoreaInvestment,
                       top_n: int = 30) -> list[dict]:
    """체결강도상위 — 매수 세력 감지."""
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_COND_SCR_DIV_CODE": "20168",
        "FID_INPUT_ISCD": "0000",
        "FID_DIV_CLS_CODE": "0",
        "FID_INPUT_PRICE_1": "",
        "FID_INPUT_PRICE_2": "",
        "FID_VOL_CNT": "",
        "FID_TRGT_CLS_CODE": "111111111",
        "FID_TRGT_EXLS_CLS_CODE": "0000000000",
    }

    items = _api_get(broker, "uapi/domestic-stock/v1/ranking/volume-power",
                     "FHPST01680000", params)

    results = []
    for item in items[:top_n]:
        code = (item.get("stck_shrn_iscd") or item.get("mksc_shrn_iscd") or "").strip()
        name = (item.get("hts_kor_isnm") or "").strip()
        if not code or not name:
            continue
        results.append({
            "rank": int(item.get("data_rank", 0) or 0),
            "code": code,
            "name": name,
            "price": int(item.get("stck_prpr", 0) or 0),
            "change_pct": float(item.get("prdy_ctrt", 0) or 0),
            "volume": int(item.get("acml_vol", 0) or 0),
            "strength": float(item.get("tday_rltv") or item.get("seln_cntg_smtn") or 0),
        })

    logger.info("체결강도상위: %d종목", len(results))
    return results


# ═══════════════════════════════════════════════════
# 4. 외국인/기관 가집계 (FHPTJ04400000)
# ═══════════════════════════════════════════════════

def fetch_foreign_institution(broker: mojito.KoreaInvestment,
                              top_n: int = 20) -> dict:
    """외국인/기관 순매수/순매도 상위.

    Returns:
        {"foreign_buy": [...], "foreign_sell": [...],
         "inst_buy": [...], "inst_sell": [...]}
    """
    result = {}

    configs = [
        ("foreign_buy",  "1", "0"),   # 외국인 순매수
        ("foreign_sell", "1", "1"),   # 외국인 순매도
        ("inst_buy",     "2", "0"),   # 기관 순매수
        ("inst_sell",    "2", "1"),   # 기관 순매도
    ]

    for key, etc_cls, sort_cls in configs:
        params = {
            "FID_COND_MRKT_DIV_CODE": "V",
            "FID_COND_SCR_DIV_CODE": "16449",
            "FID_INPUT_ISCD": "0000",
            "FID_DIV_CLS_CODE": "0",        # 수량정렬
            "FID_RANK_SORT_CLS_CODE": sort_cls,
            "FID_ETC_CLS_CODE": etc_cls,
        }

        items = _api_get(
            broker,
            "uapi/domestic-stock/v1/quotations/foreign-institution-total",
            "FHPTJ04400000", params,
        )

        entries = []
        for item in items[:top_n]:
            code = (item.get("mksc_shrn_iscd") or "").strip()
            name = (item.get("hts_kor_isnm") or "").strip()
            if not code or not name:
                continue
            entries.append({
                "code": code,
                "name": name,
                "price": int(item.get("stck_prpr", 0) or 0),
                "change_pct": float(item.get("prdy_ctrt", 0) or 0),
                "frgn_qty": int(item.get("frgn_ntby_qty", 0) or 0),
                "inst_qty": int(item.get("orgn_ntby_qty", 0) or 0),
                "frgn_amt_억": round(int(item.get("frgn_ntby_tr_pbmn", 0) or 0) / 100, 1),  # 백만원→억원
                "inst_amt_억": round(int(item.get("orgn_ntby_tr_pbmn", 0) or 0) / 100, 1),
            })

        result[key] = entries
        logger.info("외인/기관 %s: %d종목", key, len(entries))

    return result


# ═══════════════════════════════════════════════════
# 5. 상하한가 포착 (FHKST130000C0)
# ═══════════════════════════════════════════════════

def fetch_limit_price(broker: mojito.KoreaInvestment) -> dict:
    """상한가/하한가 종목 포착.

    Returns:
        {"upper_limit": [...], "lower_limit": [...]}
    """
    result = {}

    for label, prc_cls in [("upper_limit", "0"), ("lower_limit", "1")]:
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_COND_SCR_DIV_CODE": "11300",
            "FID_PRC_CLS_CODE": prc_cls,
            "FID_DIV_CLS_CODE": "0",        # 상하한가 종목
            "FID_INPUT_ISCD": "0000",        # 전체
            "FID_TRGT_CLS_CODE": "",
            "FID_TRGT_EXLS_CLS_CODE": "",
            "FID_INPUT_PRICE_1": "",
            "FID_INPUT_PRICE_2": "",
            "FID_VOL_CNT": "",
        }

        items = _api_get(
            broker,
            "uapi/domestic-stock/v1/quotations/capture-uplowprice",
            "FHKST130000C0", params,
        )

        entries = []
        for item in items:
            code = (item.get("stck_shrn_iscd") or item.get("mksc_shrn_iscd") or "").strip()
            name = (item.get("hts_kor_isnm") or "").strip()
            if not code or not name:
                continue
            entries.append({
                "code": code,
                "name": name,
                "price": int(item.get("stck_prpr", 0) or 0),
                "change_pct": float(item.get("prdy_ctrt", 0) or 0),
                "volume": int(item.get("acml_vol", 0) or 0),
            })

        result[label] = entries
        logger.info("상하한가 %s: %d종목", label, len(entries))

    return result


# ═══════════════════════════════════════════════════
# 업로드 + 출력
# ═══════════════════════════════════════════════════

def upload_market_ranking(payload: dict, date_str: str) -> bool:
    """quant_market_ranking 테이블에 업로드."""
    try:
        from src.adapters.flowx_uploader import FlowxUploader
        uploader = FlowxUploader()
        row = {"date": date_str, "data": payload}
        uploader.client.table("quant_market_ranking").upsert(
            row, on_conflict="date"
        ).execute()
        logger.info("[순위] 업로드 완료: %s", date_str)
        return True
    except Exception as e:
        logger.error("[순위] 업로드 오류: %s", e)
        return False


def print_report(vol_rank, fluct_rank, power_rank, fi_data, limit_data):
    """콘솔 리포트."""
    print(f"\n{'='*65}")
    print(f"  KIS 국내종목순위 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}")

    # 거래량 TOP 10
    print(f"\n  [거래량 TOP] {len(vol_rank)}종목")
    for s in vol_rank[:10]:
        print(
            f"    {s['rank']:>3}위 {s['name']:12s} ({s['code']}) "
            f"{s['change_pct']:+.1f}% | 거래량 {s['volume']:>12,} | 거래대금 {s['amount_억']:,.0f}억"
        )

    # 등락률 TOP 10
    print(f"\n  [급등 TOP] {len(fluct_rank)}종목")
    for s in fluct_rank[:10]:
        print(
            f"    {s['rank']:>3}위 {s['name']:12s} ({s['code']}) "
            f"{s['change_pct']:+.1f}% | {s['price']:>8,}원"
        )

    # 체결강도 TOP 10
    print(f"\n  [체결강도 TOP] {len(power_rank)}종목")
    for s in power_rank[:10]:
        print(
            f"    {s['rank']:>3}위 {s['name']:12s} ({s['code']}) "
            f"강도 {s['strength']:.1f} | {s['change_pct']:+.1f}%"
        )

    # 외인/기관 순매수 TOP 5
    for label, display in [("foreign_buy", "외국인 순매수"), ("inst_buy", "기관 순매수")]:
        items = fi_data.get(label, [])
        print(f"\n  [{display}] {len(items)}종목")
        for s in items[:5]:
            amt = s.get("frgn_amt_억", 0) if "foreign" in label else s.get("inst_amt_억", 0)
            print(
                f"    {s['name']:12s} ({s['code']}) "
                f"{s['change_pct']:+.1f}% | {amt:+,.0f}억원"
            )

    # 상한가
    upper = limit_data.get("upper_limit", [])
    lower = limit_data.get("lower_limit", [])
    if upper:
        names = ", ".join(f"{s['name']}" for s in upper[:10])
        print(f"\n  [상한가] {len(upper)}종목 — {names}")
    if lower:
        names = ", ".join(f"{s['name']}" for s in lower[:10])
        print(f"\n  [하한가] {len(lower)}종목 — {names}")

    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="KIS 국내종목순위 수집기")
    parser.add_argument("--dry-run", action="store_true", help="업로드 없이 출력만")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n[순위] 수집 시작 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    broker = _get_broker()

    # 5개 API 호출
    vol_rank = fetch_volume_rank(broker)
    fluct_rank = fetch_fluctuation_rank(broker)
    power_rank = fetch_volume_power(broker)
    fi_data = fetch_foreign_institution(broker)
    limit_data = fetch_limit_price(broker)

    # 콘솔 출력
    print_report(vol_rank, fluct_rank, power_rank, fi_data, limit_data)

    # JSON 저장
    date_str = datetime.now().strftime("%Y-%m-%d")
    payload = {
        "generated_at": datetime.now().isoformat(),
        "date": date_str,
        "volume_rank": vol_rank,
        "fluctuation_rank": fluct_rank,
        "volume_power": power_rank,
        "foreign_institution": fi_data,
        "limit_price": limit_data,
        "summary": {
            "volume_top": len(vol_rank),
            "fluct_top": len(fluct_rank),
            "power_top": len(power_rank),
            "frgn_buy": len(fi_data.get("foreign_buy", [])),
            "inst_buy": len(fi_data.get("inst_buy", [])),
            "upper_limit": len(limit_data.get("upper_limit", [])),
            "lower_limit": len(limit_data.get("lower_limit", [])),
        },
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {OUTPUT_PATH}")

    # 업로드
    if not args.dry_run:
        upload_market_ranking(payload, date_str)

    total = (len(vol_rank) + len(fluct_rank) + len(power_rank)
             + sum(len(v) for v in fi_data.values())
             + sum(len(v) for v in limit_data.values()))
    print(f"\n[순위] 완료 — 총 {total}건 (5개 API)")


if __name__ == "__main__":
    main()
