"""원자재 가격 수집 + 원가 갭(Cost Gap) 분석 — BAT-A Level 0

Alpha Vantage + EIA API로 원자재 시세를 수집하고,
config/cost_floors.yaml의 생산원가와 비교하여 원가 갭을 산출한다.

원가 갭 = (시장가 - 생산원가) / 생산원가 * 100%
갭이 낮을수록 하방 제한 → 매수 구간.

출력:
  data/commodity_prices.json          — 최신 가격 + 원가 갭
  data/commodities/history/YYYY-MM-DD.json — 일별 스냅샷

사용:
  python -u -X utf8 scripts/fetch_commodity_prices.py
  python -u -X utf8 scripts/fetch_commodity_prices.py --send   # 텔레그램 포함
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HISTORY_DIR = DATA_DIR / "commodities" / "history"
OUTPUT_PATH = DATA_DIR / "commodity_prices.json"
COST_FLOORS_PATH = PROJECT_ROOT / "config" / "cost_floors.yaml"


def load_cost_floors() -> dict:
    """config/cost_floors.yaml 로드."""
    import yaml
    if not COST_FLOORS_PATH.exists():
        logger.warning("cost_floors.yaml 없음 — 원가 갭 계산 불가")
        return {}
    with open(COST_FLOORS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("commodities", {})


def calc_cost_gap(price: float, cost_floor: float) -> dict:
    """원가 갭 계산 + 구간 판별."""
    if cost_floor <= 0 or price <= 0:
        return {"gap_pct": None, "zone": "unknown"}
    gap_pct = round((price - cost_floor) / cost_floor * 100, 1)
    if gap_pct <= 15:
        zone = "buy"
        emoji = "\U0001f7e2"  # 🟢
    elif gap_pct <= 40:
        zone = "watch"
        emoji = "\U0001f7e1"  # 🟡
    elif gap_pct <= 80:
        zone = "hold"
        emoji = "\U0001f7e0"  # 🟠
    else:
        zone = "overheated"
        emoji = "\U0001f534"  # 🔴
    return {
        "gap_pct": gap_pct,
        "zone": zone,
        "emoji": emoji,
        "production_cost": cost_floor,
    }


# ────────────────────────────────────────────
# API 수집 함수들
# ────────────────────────────────────────────

def fetch_eia_wti(api_key: str) -> dict | None:
    """EIA API: WTI 일별 가격 (무제한, 가장 신뢰)."""
    import requests
    try:
        params = {
            "api_key": api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[product][]": "EPCWTI",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 5,
        }
        r = requests.get(
            "https://api.eia.gov/v2/petroleum/pri/spt/data/",
            params=params, timeout=15,
        )
        data = r.json()
        rows = data.get("response", {}).get("data", [])
        if rows:
            latest = rows[0]
            return {
                "price": float(latest["value"]),
                "date": latest["period"],
                "source": "eia",
            }
    except Exception as e:
        logger.warning(f"EIA WTI 실패: {e}")
    return None


def fetch_eia_natural_gas(api_key: str) -> dict | None:
    """EIA API: Henry Hub 천연가스 일별 가격."""
    import requests
    try:
        params = {
            "api_key": api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[product][]": "EPG0",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 5,
        }
        r = requests.get(
            "https://api.eia.gov/v2/natural-gas/pri/fut/data/",
            params=params, timeout=15,
        )
        data = r.json()
        rows = data.get("response", {}).get("data", [])
        if rows:
            latest = rows[0]
            return {
                "price": float(latest["value"]),
                "date": latest["period"],
                "source": "eia",
            }
    except Exception as e:
        logger.warning(f"EIA NatGas 실패: {e}")
    return None


def fetch_alpha_vantage_commodity(api_key: str, function: str) -> dict | None:
    """Alpha Vantage 원자재 endpoint (COPPER, NATURAL_GAS, ALUMINUM, WTI)."""
    import requests
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": function, "interval": "daily", "apikey": api_key},
            timeout=15,
        )
        data = r.json()
        series = data.get("data", [])
        if series:
            latest = series[0]
            val = latest.get("value", ".")
            if val and val != ".":
                return {
                    "price": float(val),
                    "date": latest.get("date", ""),
                    "source": "alpha_vantage",
                }
        if "Note" in data or "Information" in data:
            logger.warning(f"  AV {function}: Rate limit")
    except Exception as e:
        logger.warning(f"AV {function} 실패: {e}")
    return None


def fetch_alpha_vantage_quote(api_key: str, symbol: str) -> dict | None:
    """Alpha Vantage ETF 실시간 시세 (GLD, SLV, UUP 등)."""
    import requests
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
            timeout=15,
        )
        data = r.json()
        quote = data.get("Global Quote", {})
        if quote and quote.get("05. price"):
            return {
                "price": float(quote["05. price"]),
                "change_pct": float(quote.get("10. change percent", "0%").replace("%", "")),
                "date": quote.get("07. latest trading day", ""),
                "symbol": symbol,
                "source": "alpha_vantage",
            }
        if "Note" in data:
            logger.warning(f"  AV {symbol}: Rate limit")
    except Exception as e:
        logger.warning(f"AV {symbol} 실패: {e}")
    return None


def fetch_finnhub_quote(api_key: str, symbol: str) -> dict | None:
    """Finnhub 실시간 시세 (VIX 등)."""
    import requests
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": api_key},
            timeout=10,
        )
        data = r.json()
        if data.get("c"):
            return {
                "price": float(data["c"]),
                "change_pct": round(float(data.get("dp", 0)), 2),
                "high": float(data.get("h", 0)),
                "low": float(data.get("l", 0)),
                "source": "finnhub",
            }
    except Exception as e:
        logger.warning(f"Finnhub {symbol} 실패: {e}")
    return None


# ────────────────────────────────────────────
# 메인 수집 로직
# ────────────────────────────────────────────

def collect_all(av_key: str, eia_key: str, fh_key: str) -> dict:
    """모든 원자재 가격 수집 + 원가 갭 계산."""
    cost_floors = load_cost_floors()
    results = {}
    av_calls = 0
    MAX_AV_CALLS = 15  # 25회/일 중 15회만 사용 (뉴스 감성분석 10회 예약)

    # ── 1. EIA (무제한, 우선) ──
    logger.info("[EIA] WTI 원유...")
    wti_data = fetch_eia_wti(eia_key)
    if wti_data:
        floor = cost_floors.get("wti", {})
        wti_data["cost_gap"] = calc_cost_gap(wti_data["price"], floor.get("production_cost", 0))
        wti_data["name"] = "WTI 원유"
        wti_data["unit"] = "USD/barrel"
        results["wti"] = wti_data
        logger.info(f"  WTI: ${wti_data['price']} | 갭: {wti_data['cost_gap']['gap_pct']}%")

    logger.info("[EIA] 천연가스...")
    ng_data = fetch_eia_natural_gas(eia_key)
    if ng_data:
        floor = cost_floors.get("natural_gas", {})
        ng_data["cost_gap"] = calc_cost_gap(ng_data["price"], floor.get("production_cost", 0))
        ng_data["name"] = "천연가스"
        ng_data["unit"] = "USD/MMBtu"
        results["natural_gas"] = ng_data
        logger.info(f"  NatGas: ${ng_data['price']} | 갭: {ng_data['cost_gap']['gap_pct']}%")

    # ── 2. Alpha Vantage 원자재 ──
    av_commodities = [
        ("copper", "COPPER", "구리", "USD/metric ton"),
    ]
    for name, func, label, unit in av_commodities:
        if av_calls >= MAX_AV_CALLS:
            logger.warning(f"AV 호출 제한 도달 ({MAX_AV_CALLS}회)")
            break
        logger.info(f"[AV] {label}...")
        data = fetch_alpha_vantage_commodity(av_key, func)
        av_calls += 1
        if data:
            floor = cost_floors.get(name, {})
            data["cost_gap"] = calc_cost_gap(data["price"], floor.get("production_cost", 0))
            data["name"] = label
            data["unit"] = unit
            results[name] = data
            logger.info(f"  {label}: ${data['price']} | 갭: {data['cost_gap']['gap_pct']}%")
        time.sleep(13)

    # ── 3. Alpha Vantage ETF proxy ──
    etf_proxies = [
        ("gold", "GLD", "금 (GLD)", "USD/oz", 10),
        ("silver", "SLV", "은 (SLV)", "USD/oz", 1),
        ("dxy", "UUP", "달러 (UUP)", "index", None),
    ]
    for name, symbol, label, unit, multiplier in etf_proxies:
        if av_calls >= MAX_AV_CALLS:
            break
        logger.info(f"[AV] {label}...")
        data = fetch_alpha_vantage_quote(av_key, symbol)
        av_calls += 1
        if data:
            actual_price = data["price"] * multiplier if multiplier else data["price"]
            floor = cost_floors.get(name, {})
            cost = floor.get("production_cost", 0)
            if cost > 0 and multiplier:
                data["estimated_price"] = round(actual_price, 2)
                data["cost_gap"] = calc_cost_gap(actual_price, cost)
            data["name"] = label
            data["unit"] = unit
            results[name] = data
            gap_str = f"갭: {data.get('cost_gap', {}).get('gap_pct', 'N/A')}%"
            logger.info(f"  {label}: ${data['price']} | {gap_str}")
        time.sleep(13)

    # ── 4. Finnhub (무제한) — VIX proxy (VIXY ETF) ──
    logger.info("[Finnhub] VIX (VIXY)...")
    vix = fetch_finnhub_quote(fh_key, "VIXY")
    if vix:
        vix["name"] = "VIX"
        vix["unit"] = "index"
        results["vix"] = vix
        logger.info(f"  VIX: {vix['price']}")

    # ── 5. 수동 업데이트 원자재 (cost_floors.yaml에서 직접 로드) ──
    for name in ["tio2", "uranium", "naphtha"]:
        floor = cost_floors.get(name, {})
        if floor.get("current_price"):
            manual_data = {
                "price": floor["current_price"],
                "date": floor.get("price_date", ""),
                "source": "manual",
                "name": floor.get("name", name),
                "unit": floor.get("unit", ""),
                "cost_gap": calc_cost_gap(floor["current_price"], floor.get("production_cost", 0)),
                "note": "수동 업데이트 (API 없음)",
            }
            results[name] = manual_data
            logger.info(f"  {manual_data['name']}: ${manual_data['price']} | 갭: {manual_data['cost_gap']['gap_pct']}%")

    logger.info(f"AV API 사용: {av_calls}/{MAX_AV_CALLS}회")
    return results


def build_cost_gap_summary(results: dict) -> str:
    """텔레그램 보고용 원가 갭 요약 생성."""
    lines = ["\U0001f4ca [원가 갭 대시보드]", ""]

    # 갭 낮은 순 정렬 (매수 구간 먼저)
    items = []
    for name, data in results.items():
        gap = data.get("cost_gap", {})
        if gap.get("gap_pct") is not None:
            items.append((name, data, gap))

    items.sort(key=lambda x: x[2]["gap_pct"])

    for name, data, gap in items:
        emoji = gap.get("emoji", "")
        label = data.get("name", name)
        price = data.get("price", 0)
        cost = gap.get("production_cost", 0)
        gap_pct = gap["gap_pct"]
        zone = gap["zone"]
        zone_label = {"buy": "매수구간", "watch": "관찰", "hold": "보류", "overheated": "과열"}.get(zone, zone)
        lines.append(f"{emoji} {label:8s} ${price:>8} (원가${cost:>7}) 갭{gap_pct:>5.1f}% {zone_label}")

    # VIX (갭 없음)
    if "vix" in results:
        vix = results["vix"]
        vix_val = vix.get("price", 0)
        vix_emoji = "\U0001f534" if vix_val >= 30 else "\U0001f7e1" if vix_val >= 20 else "\U0001f7e2"
        lines.append(f"\n{vix_emoji} VIX: {vix_val}")

    return "\n".join(lines)


def main():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", help="텔레그램 전송")
    args = parser.parse_args()

    av_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip().rstrip(".")
    eia_key = os.environ.get("EIA_API_KEY", "")
    fh_key = os.environ.get("FINNHUB_API_KEY", "")

    if not av_key and not eia_key:
        logger.error("ALPHA_VANTAGE_API_KEY 또는 EIA_API_KEY 필요")
        return

    logger.info("=== 원자재 가격 수집 + 원가 갭 분석 ===")

    results = collect_all(av_key, eia_key, fh_key)

    # 결과 저장
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(),
        "commodity_count": len(results),
        "commodities": results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"저장: {OUTPUT_PATH} ({len(results)}종)")

    # 히스토리 저장
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_path = HISTORY_DIR / f"{output['date']}.json"
    history_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"히스토리: {history_path}")

    # 원가 갭 요약 출력
    summary = build_cost_gap_summary(results)
    print("\n" + summary)

    # 텔레그램 전송
    if args.send:
        try:
            from src.telegram_sender import send_message
            send_message(summary)
            logger.info("텔레그램 전송 완료")
        except Exception as e:
            logger.error(f"텔레그램 전송 실패: {e}")

    return output


if __name__ == "__main__":
    main()
