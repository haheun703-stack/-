"""한국은행 ECOS API 매크로 지표 수집 + 히스토리

기준금리, 원/달러, CPI, 국고채3년, 국고채10년 수집.
KOSPI PER은 pykrx에서 가져와 ERP(주식위험프리미엄) 계산에 사용.
ECOS 무료: 제한 없음 (일 10만건).

출력:
  data/ecos_macro.json       — 최신 스냅샷
  data/macro/macro_history.csv — 일별 히스토리 (추세 분석용)

사용: python -u -X utf8 scripts/fetch_ecos_macro.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MACRO_DIR = DATA_DIR / "macro"
OUTPUT_PATH = DATA_DIR / "ecos_macro.json"
HISTORY_PATH = MACRO_DIR / "macro_history.csv"


# ECOS 통계코드 (https://ecos.bok.or.kr/api)
INDICATORS = {
    "base_rate": {
        "stat_code": "722Y001",
        "item_code": "0101000",
        "freq": "M",
        "label": "기준금리(%)",
    },
    "usd_krw": {
        "stat_code": "731Y001",
        "item_code": "0000001",
        "freq": "D",
        "label": "원/달러 환율",
    },
    "cpi": {
        "stat_code": "901Y009",
        "item_code": "0",
        "freq": "M",
        "label": "소비자물가지수(2020=100)",
    },
    "bond_3y": {
        "stat_code": "817Y002",
        "item_code": "010200000",
        "freq": "D",
        "label": "국고채3년(%)",
    },
    "bond_10y": {
        "stat_code": "817Y002",
        "item_code": "010200001",
        "freq": "D",
        "label": "국고채10년(%)",
    },
}


def _fetch_ecos(api_key: str, name: str, cfg: dict) -> dict | None:
    """ECOS API에서 지표 시계열 조회 (최신값 + 이전 비교 기준).

    일별(D): 최근 90일 (약 60거래일)
    월별(M): 최근 12개월
    → 최신값 + 1개월전 + 3개월전 + 전체 시계열 반환
    """
    import requests

    freq = cfg["freq"]
    now = datetime.now()

    if freq == "D":
        start = (now - timedelta(days=100)).strftime("%Y%m%d")
        end = now.strftime("%Y%m%d")
        max_rows = 70
    else:
        start = (now - timedelta(days=400)).strftime("%Y%m")
        end = now.strftime("%Y%m")
        max_rows = 13

    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch"
        f"/{api_key}/json/kr/1/{max_rows}"
        f"/{cfg['stat_code']}/{freq}/{start}/{end}/{cfg['item_code']}"
    )

    try:
        r = requests.get(url, timeout=15)
        data = r.json()

        if "StatisticSearch" in data:
            rows = data["StatisticSearch"].get("row", [])
            if rows:
                # 시계열 파싱
                series = []
                for row in rows:
                    try:
                        val = float(row["DATA_VALUE"].replace(",", ""))
                        series.append({"period": row["TIME"], "value": val})
                    except (ValueError, KeyError):
                        continue

                if not series:
                    return None

                latest = series[-1]

                # 비교 기준 계산
                comparisons = _calc_comparisons(series, freq)

                return {
                    "value": latest["value"],
                    "period": latest["period"],
                    "unit": rows[-1].get("UNIT_NAME", ""),
                    "label": cfg["label"],
                    **comparisons,
                    "series": series,  # 전체 시계열
                }

        err = data.get("RESULT", {})
        if err:
            logger.warning(f"  {name}: {err.get('MESSAGE', 'unknown error')}")
        return None

    except Exception as e:
        logger.warning(f"  {name}: {e}")
        return None


def _calc_comparisons(series: list[dict], freq: str) -> dict:
    """시계열에서 1개월전, 3개월전 값을 찾아 변화량 계산."""
    if len(series) < 2:
        return {}

    latest_val = series[-1]["value"]
    result = {}

    if freq == "D":
        # 일별: ~20거래일=1개월, ~60거래일=3개월
        offsets = {"1m": 20, "3m": 60}
    else:
        # 월별: 1=1개월, 3=3개월
        offsets = {"1m": 1, "3m": 3}

    for label, offset in offsets.items():
        idx = len(series) - 1 - offset
        if idx >= 0:
            prev_val = series[idx]["value"]
            prev_period = series[idx]["period"]
            change = round(latest_val - prev_val, 3)
            change_pct = round((latest_val / prev_val - 1) * 100, 2) if prev_val != 0 else 0
            result[f"prev_{label}"] = prev_val
            result[f"prev_{label}_period"] = prev_period
            result[f"change_{label}"] = change
            result[f"change_{label}_pct"] = change_pct

    return result


def _fetch_kospi_per() -> dict | None:
    """pykrx로 KOSPI PER/PBR/배당수익률 조회 (ERP 계산용)."""
    try:
        from pykrx import stock as krx
    except ImportError:
        logger.warning("pykrx 미설치 — KOSPI PER 생략")
        return None

    # pykrx logging.info 버그 우회: util.py의 logging.info(args, kwargs) TypeError 방지
    try:
        import pykrx.website.comm.util as _pykrx_util
        import functools as _ft

        _orig_wrapper = None
        for _attr in dir(_pykrx_util):
            _obj = getattr(_pykrx_util, _attr)
            if callable(_obj) and hasattr(_obj, "__wrapped__"):
                break
        # 직접 monkey-patch: wrapper 내부 logging.info 호출을 안전하게
        _root = logging.getLogger()
        _saved_level = _root.level
    except Exception:
        pass

    # 오늘부터 최대 7일 전까지 시도 (주말/공휴일 대응)
    for delta in range(8):
        dt = datetime.now() - timedelta(days=delta)
        dt_str = dt.strftime("%Y%m%d")
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # pykrx 내부 로그 억제 (root 포함 — TypeError 방지)
                pykrx_logger = logging.getLogger("pykrx")
                old_level = pykrx_logger.level
                pykrx_logger.setLevel(logging.CRITICAL)
                logging.disable(logging.CRITICAL)
                try:
                    df = krx.get_index_fundamental(dt_str, dt_str, "1001")
                finally:
                    pykrx_logger.setLevel(old_level)
                    logging.disable(logging.NOTSET)

            if df is not None and not df.empty:
                row = df.iloc[-1]
                per = float(row.get("PER", 0))
                pbr = float(row.get("PBR", 0))
                div_yield = float(row.get("DIV", 0))
                if per > 0:
                    logger.info(f"  KOSPI PER: {per:.1f} (PBR {pbr:.2f}, 배당 {div_yield:.1f}%) [{dt_str}]")
                    return {
                        "per": per,
                        "pbr": pbr,
                        "div_yield": div_yield,
                        "date": dt_str,
                    }
        except Exception:
            continue
    logger.warning("  KOSPI PER 조회 실패 (최근 7일 데이터 없음)")
    return None


def _save_history(result: dict) -> None:
    """일별 히스토리 CSV에 추가."""
    import csv

    MACRO_DIR.mkdir(parents=True, exist_ok=True)

    ind = result.get("indicators", {})
    row = {
        "date": result["date"],
        "base_rate": ind.get("base_rate", {}).get("value", ""),
        "usd_krw": ind.get("usd_krw", {}).get("value", ""),
        "cpi": ind.get("cpi", {}).get("value", ""),
        "bond_3y": ind.get("bond_3y", {}).get("value", ""),
        "bond_10y": ind.get("bond_10y", {}).get("value", ""),
        "kospi_per": result.get("kospi", {}).get("per", ""),
        "kospi_pbr": result.get("kospi", {}).get("pbr", ""),
        "erp": result.get("erp", ""),
    }

    fieldnames = list(row.keys())

    # 기존 파일 읽기 (중복 날짜 방지)
    existing_rows = []
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r.get("date") != row["date"]]

    existing_rows.append(row)
    existing_rows.sort(key=lambda r: r.get("date", ""))

    with open(HISTORY_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

    logger.info(f"히스토리: {HISTORY_PATH} ({len(existing_rows)}일)")


def main():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("ECOS_API_KEY", "").strip()
    if not api_key:
        logger.error("ECOS_API_KEY 없음. .env 확인 필요.")
        return

    logger.info("=== 한국은행 ECOS 매크로 지표 수집 ===")

    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(),
        "source": "ecos_bok",
        "indicators": {},
    }

    for name, cfg in INDICATORS.items():
        logger.info(f"  {cfg['label']}...")
        data = _fetch_ecos(api_key, name, cfg)
        if data:
            result["indicators"][name] = data
            # 비교 기준 출력
            parts = [f"{data['value']} {data.get('unit', '')} ({data['period']})"]
            if "prev_1m" in data:
                parts.append(f"1개월전 {data['prev_1m']} ({data['change_1m']:+.3f})")
            if "prev_3m" in data:
                parts.append(f"3개월전 {data['prev_3m']} ({data['change_3m']:+.3f})")
            logger.info(f"    → {' | '.join(parts)}")
        else:
            logger.warning(f"    → 실패")
        time.sleep(0.5)

    # KOSPI PER/PBR (pykrx)
    logger.info("  KOSPI PER/PBR...")
    kospi_data = _fetch_kospi_per()
    if kospi_data:
        result["kospi"] = kospi_data

        # ERP (Equity Risk Premium) = 주식기대수익률 - 무위험수익률
        # 주식기대수익률 = 1/PER (= 이익수익률)
        # 무위험수익률 = 국고채10년
        bond_10y = result["indicators"].get("bond_10y", {}).get("value")
        if bond_10y and kospi_data["per"] > 0:
            earnings_yield = (1 / kospi_data["per"]) * 100
            erp = round(earnings_yield - bond_10y, 2)
            result["erp"] = erp
            result["earnings_yield"] = round(earnings_yield, 2)
            logger.info(f"  ERP: {erp:.2f}%p (이익수익률 {earnings_yield:.1f}% - 국고채10년 {bond_10y:.3f}%)")
    else:
        result["kospi"] = {}

    # 저장
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"저장: {OUTPUT_PATH} ({len(result['indicators'])}개 ECOS 지표)")

    # 히스토리 CSV 저장
    _save_history(result)

    return result


if __name__ == "__main__":
    main()
