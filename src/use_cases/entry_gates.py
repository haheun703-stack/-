"""진입 게이트 — 5/21+22 워밍업 / 5/26~ 실전 대비 (2026-05-19 신규)

배경 (사장님 5/19 결단):
  기존 안전선 9건은 "어제 BAT-D 결과 + VWAP + EYE + 기본 가드"로
  장중 모멘텀 검증이 약함. 단타봇은 호가창/체결강도/5분봉을 보지만 퀀트봇은 못 봄.
  → 옵션 A~C 타입 (5분봉 + 호가창 + 체결강도) 3종 진입 게이트로 동급 정밀도 달성.

  ★ 사장님 5/19 정정: "분봉 아니고 5분봉" — 1분봉 X, 5분봉 단위 정확 사용.
  이유: KIS API rate limit 안전 + 자체 5분봉 parquet 30거래일 보유 + 단타봇과 동일 단위.

게이트 3종 (auto_buy_executor의 안전선 9건과 직교):
  A. 5분봉 진입 게이트 (intraday)        — 직전 30분 (5분봉 × 6봉) 추세 + VWAP 안착
  B. 호가창 게이트 (orderbook)           — 매수/매도 잔량 우위
  C. 체결강도 게이트 (volume_power)      — 매수 체결 우위 (단타봇 동급)

데이터 소스:
  A. VPS 자체 5분봉 parquet (~/quantum-master/data/intraday/5min/{date}/{ticker}.parquet)
     없으면 KIS API fetch (KisIntradayAdapter.fetch_minute_candles)
  B. KIS API: FHKST01010200 (KisIntradayAdapter.fetch_orderbook 재사용)
  C. KIS API: 주식현재가 (fetch_price output)의 tday_rltv 필드
     fallback: shnu_cntg_smtn / seln_cntg_smtn 계산

환경변수 (선택):
  ENTRY_GATE_BULL_RATIO_MIN=0.6
  ENTRY_GATE_BID_ASK_MIN=1.0
  ENTRY_GATE_DEPTH_RATIO_MIN=0.8
  ENTRY_GATE_VOLUME_POWER_MIN=150.0
  ENTRY_GATE_VWAP_FLOOR=1.0           # 현재가 ≥ VWAP × 1.00
  ENTRY_GATE_VOL_SURGE_MIN=1.5        # 마지막 봉 거래량 ≥ 평균 × 1.5
  ENTRY_GATE_REQUIRE_ALL=true         # all_passed 산정 (false 시 2/3로 완화)

사용:
  from src.use_cases.entry_gates import check_all_entry_gates
  gates = check_all_entry_gates(broker, ticker)
  if not gates["all_passed"]:
      skip_reason = f"진입 게이트 {gates['passed_count']}/3: {gates['summary']}"
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INTRADAY_5MIN_DIR = PROJECT_ROOT / "data" / "intraday" / "5min"

# ─────────────────────────────────────────────
# 임계값 (환경변수로 조정 가능)
# ─────────────────────────────────────────────

INTRADAY_BULL_RATIO_MIN = float(os.environ.get("ENTRY_GATE_BULL_RATIO_MIN", "0.6"))
ORDERBOOK_BID_ASK_MIN = float(os.environ.get("ENTRY_GATE_BID_ASK_MIN", "1.0"))
ORDERBOOK_DEPTH_RATIO_MIN = float(os.environ.get("ENTRY_GATE_DEPTH_RATIO_MIN", "0.8"))
VOLUME_POWER_MIN = float(os.environ.get("ENTRY_GATE_VOLUME_POWER_MIN", "150.0"))
INTRADAY_VWAP_FLOOR = float(os.environ.get("ENTRY_GATE_VWAP_FLOOR", "1.0"))
INTRADAY_VOL_SURGE_MIN = float(os.environ.get("ENTRY_GATE_VOL_SURGE_MIN", "1.5"))
REQUIRE_ALL = os.environ.get("ENTRY_GATE_REQUIRE_ALL", "true").lower() == "true"

# 직전 30분 = 6 × 5분봉
LOOKBACK_5MIN_BARS = 6


# ─────────────────────────────────────────────
# A. 5분봉 진입 게이트
# ─────────────────────────────────────────────


def _load_5min_from_parquet(ticker: str, date_str: str | None = None) -> list[dict] | None:
    """VPS 자체 5분봉 parquet 로드. 없으면 None.

    경로: data/intraday/5min/{YYYY-MM-DD}/{ticker}.parquet
    스키마: timestamp, open, high, low, close, volume, vwap
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    path = INTRADAY_5MIN_DIR / date_str / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        if df.empty:
            return None
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df.to_dict("records")
    except Exception as e:
        logger.warning("[entry_gate A] parquet 로드 실패 %s: %s", path, e)
        return None


def _load_5min_from_kis(broker, ticker: str) -> list[dict]:
    """KIS API에서 5분봉 fetch (KisIntradayAdapter 재사용).

    Returns: [{timestamp, open, high, low, close, volume, vwap}, ...]
    """
    try:
        from src.adapters.kis_intraday_adapter import KisIntradayAdapter

        # CachedBroker wrapper일 수도 있으니 원본 broker로 진입
        underlying = getattr(broker, "_broker", broker)
        intra = KisIntradayAdapter(broker=underlying)
        candles = intra.fetch_minute_candles(ticker, period=5)
        # vwap이 누락된 봉은 close로 대체
        for c in candles:
            if "vwap" not in c or not c["vwap"]:
                c["vwap"] = c["close"]
        return candles
    except Exception as e:
        logger.warning("[entry_gate A] KIS 5분봉 fetch 실패 %s: %s", ticker, e)
        return []


def check_intraday_gate(
    broker,
    ticker: str,
    lookback_bars: int = LOOKBACK_5MIN_BARS,
    current_price: int | None = None,
) -> dict[str, Any]:
    """직전 N개 5분봉 추세 검증.

    통과 조건:
      1. 직전 N봉(기본 30분/6봉) 중 양봉 비율 ≥ INTRADAY_BULL_RATIO_MIN (기본 0.6)
      2. 마지막 5분봉이 양봉 (close > open)
      3. 마지막 5분봉 거래량 ≥ 직전 N봉 평균 × INTRADAY_VOL_SURGE_MIN (기본 1.5)
      4. 현재가 ≥ 최근 VWAP × INTRADAY_VWAP_FLOOR (기본 1.0)

    Args:
        broker: KIS broker (CachedBroker 가능)
        ticker: 종목 코드
        lookback_bars: 직전 N봉 (기본 6 = 30분)
        current_price: 평가 시점 현재가. None이면 broker.fetch_price 호출

    Returns:
        {
          "passed": bool,
          "bull_ratio": float,
          "last_bull": bool,
          "vol_surge": bool,
          "above_vwap": bool,
          "n_bars": int,
          "vwap_last": float,
          "current_price": int,
          "reason": str,
          "source": "parquet" | "kis" | "none",
        }
    """
    result: dict[str, Any] = {
        "passed": False,
        "bull_ratio": 0.0,
        "last_bull": False,
        "vol_surge": False,
        "above_vwap": False,
        "n_bars": 0,
        "vwap_last": 0.0,
        "current_price": current_price or 0,
        "reason": "",
        "source": "none",
    }

    # 1) parquet 우선
    candles = _load_5min_from_parquet(ticker)
    if candles:
        result["source"] = "parquet"
    else:
        # 2) KIS API fallback
        candles = _load_5min_from_kis(broker, ticker)
        if candles:
            result["source"] = "kis"

    if not candles:
        result["reason"] = "5분봉 데이터 없음 (parquet + KIS 모두 실패)"
        return result

    # 마지막 N봉만 사용
    bars = candles[-lookback_bars:]
    n = len(bars)
    result["n_bars"] = n

    if n == 0:
        result["reason"] = "5분봉 0건"
        return result

    # 조건 1: 양봉 비율
    bull_count = sum(1 for b in bars if (b.get("close") or 0) > (b.get("open") or 0))
    bull_ratio = bull_count / n if n > 0 else 0.0
    result["bull_ratio"] = round(bull_ratio, 3)
    cond1 = bull_ratio >= INTRADAY_BULL_RATIO_MIN

    # 조건 2: 마지막 봉 양봉
    last = bars[-1]
    last_bull = (last.get("close") or 0) > (last.get("open") or 0)
    result["last_bull"] = last_bull
    cond2 = last_bull

    # 조건 3: 거래량 폭증
    last_vol = float(last.get("volume", 0) or 0)
    if n >= 2:
        avg_vol = sum(float(b.get("volume", 0) or 0) for b in bars[:-1]) / (n - 1)
    else:
        avg_vol = last_vol
    vol_surge = last_vol >= avg_vol * INTRADAY_VOL_SURGE_MIN if avg_vol > 0 else True
    result["vol_surge"] = vol_surge
    cond3 = vol_surge

    # 조건 4: VWAP 안착
    vwap_last = float(last.get("vwap") or last.get("close") or 0)
    result["vwap_last"] = round(vwap_last, 2)
    if current_price is None or current_price <= 0:
        try:
            px = broker.fetch_price(ticker).get("output", {})
            current_price = int(px.get("stck_prpr", 0) or 0)
        except Exception:
            current_price = 0
    result["current_price"] = current_price
    above_vwap = (current_price >= vwap_last * INTRADAY_VWAP_FLOOR) if vwap_last > 0 else False
    result["above_vwap"] = above_vwap
    cond4 = above_vwap

    passed = cond1 and cond2 and cond3 and cond4
    result["passed"] = passed

    vol_ratio = (last_vol / avg_vol) if avg_vol > 0 else 0.0
    result["reason"] = (
        f"양봉 {bull_count}/{n}({bull_ratio*100:.0f}%) "
        f"+ 마지막 {'양봉' if last_bull else '음봉'} "
        f"+ 거래량 {vol_ratio:.1f}x "
        f"+ 현재가 {current_price:,} vs VWAP {vwap_last:,.0f}({(current_price/vwap_last*100 - 100) if vwap_last > 0 else 0:+.2f}%)"
    )
    return result


# ─────────────────────────────────────────────
# B. 호가창 게이트
# ─────────────────────────────────────────────


def check_orderbook_gate(broker, ticker: str) -> dict[str, Any]:
    """호가창 매수/매도 잔량 비율 검증.

    통과 조건:
      1. 매수1호가 잔량 / 매도1호가 잔량 ≥ ORDERBOOK_BID_ASK_MIN (기본 1.0)
      2. 매수 10호가 합계 / 매도 10호가 합계 ≥ ORDERBOOK_DEPTH_RATIO_MIN (기본 0.8)

    Returns:
        {
          "passed": bool,
          "bid_ask_ratio": float,  # 매수1/매도1
          "depth_ratio": float,    # 매수합/매도합
          "total_bid": int,
          "total_ask": int,
          "reason": str,
        }
    """
    result: dict[str, Any] = {
        "passed": False,
        "bid_ask_ratio": 0.0,
        "depth_ratio": 0.0,
        "total_bid": 0,
        "total_ask": 0,
        "reason": "",
    }

    try:
        from src.adapters.kis_intraday_adapter import KisIntradayAdapter

        underlying = getattr(broker, "_broker", broker)
        intra = KisIntradayAdapter(broker=underlying)
        ob = intra.fetch_orderbook(ticker)
    except Exception as e:
        result["reason"] = f"호가창 fetch 실패: {e}"
        return result

    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    total_bid = int(ob.get("total_bid_vol", 0) or 0)
    total_ask = int(ob.get("total_ask_vol", 0) or 0)
    result["total_bid"] = total_bid
    result["total_ask"] = total_ask

    if not bids or not asks:
        result["reason"] = "호가창 비어있음 (장 마감 / 거래정지 추정)"
        return result

    # 조건 1: 1호가 비율
    bid1 = float(bids[0].get("volume", 0) or 0)
    ask1 = float(asks[0].get("volume", 0) or 0)
    bid_ask_ratio = (bid1 / ask1) if ask1 > 0 else 0.0
    result["bid_ask_ratio"] = round(bid_ask_ratio, 2)
    cond1 = bid_ask_ratio >= ORDERBOOK_BID_ASK_MIN

    # 조건 2: 호가 합계 비율
    depth_ratio = (total_bid / total_ask) if total_ask > 0 else 0.0
    result["depth_ratio"] = round(depth_ratio, 2)
    cond2 = depth_ratio >= ORDERBOOK_DEPTH_RATIO_MIN

    passed = cond1 and cond2
    result["passed"] = passed
    result["reason"] = (
        f"매수1/매도1 {bid_ask_ratio:.2f}x "
        f"+ 호가합 {depth_ratio:.2f}x "
        f"(매수합 {total_bid:,} / 매도합 {total_ask:,})"
    )
    return result


# ─────────────────────────────────────────────
# C. 체결강도 게이트
# ─────────────────────────────────────────────


def _fetch_volume_power(broker, ticker: str) -> tuple[float, str]:
    """체결강도 (volume power) 계산.

    2026-05-21 수정: 기존 mojito.fetch_price()는 inquire-price 엔드포인트라
    체결강도 필드(tday_rltv) 미포함 → 5/21 자비스 9건 전부 체결강도=0 차단 사고.
    KIS inquire-ccnl (TR_ID FHKST01010300) 직접 호출로 변경.

    Returns:
        (volume_power, source) — source ∈ {"ccnl_tday_rltv", "fetch_price_fallback", "none"}
    """
    import requests as _rq

    # 1) KIS inquire-ccnl 직접 호출 (체결강도 정확 필드)
    try:
        url = "https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/inquire-ccnl"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": broker.access_token,
            "appkey": broker.api_key,
            "appsecret": broker.api_secret,
            "tr_id": "FHKST01010300",
        }
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}
        r = _rq.get(url, headers=headers, params=params, timeout=5)
        data = r.json()
        if data.get("rt_cd") == "0":
            output = data.get("output", [])
            first = output[0] if isinstance(output, list) and output else (output if isinstance(output, dict) else None)
            if first:
                raw = first.get("tday_rltv")
                if raw not in (None, "", 0, "0"):
                    vp = float(raw)
                    if vp > 0:
                        return vp, "ccnl_tday_rltv"
        else:
            logger.warning("[entry_gate C] inquire-ccnl rt_cd=%s msg=%s", data.get("rt_cd"), data.get("msg1"))
    except Exception as e:
        logger.warning("[entry_gate C] inquire-ccnl 예외 %s: %s", ticker, e)

    # 2) fetch_price fallback (이전 코드 유지 — 만약 일부 종목 응답 다를 경우)
    try:
        out = broker.fetch_price(ticker).get("output", {})
        raw = out.get("tday_rltv")
        if raw not in (None, "", 0, "0"):
            vp = float(raw)
            if vp > 0:
                return vp, "fetch_price_fallback"
        buy_vol = int(out.get("shnu_cntg_smtn", 0) or 0)
        sell_vol = int(out.get("seln_cntg_smtn", 0) or 0)
        if sell_vol > 0:
            return round((buy_vol / sell_vol) * 100.0, 2), "fetch_price_fallback"
        if buy_vol > 0:
            return 999.0, "fetch_price_fallback"
    except Exception:
        pass

    return 0.0, "none"


def check_volume_power_gate(
    broker,
    ticker: str,
    threshold: float = VOLUME_POWER_MIN,
) -> dict[str, Any]:
    """체결강도 검증 (단타봇 동일 임계).

    체결강도 = (매수 체결량 / 매도 체결량) × 100
      - 150+ = 강한 매수세 (단타봇 동급)
      - 100  = 중립
      - 50-  = 강한 매도세

    Returns:
        {
          "passed": bool,
          "volume_power": float,
          "source": "tday_rltv" | "calc" | "none",
          "threshold": float,
          "reason": str,
        }
    """
    vp, source = _fetch_volume_power(broker, ticker)
    passed = vp >= threshold

    if source == "none":
        reason = f"체결강도 fetch 실패 (장 마감/거래정지 추정) — 임계 {threshold:.0f}"
        passed = False
    elif passed:
        reason = f"체결강도 {vp:.1f} ≥ {threshold:.0f} (강한 매수세, src={source})"
    else:
        reason = f"체결강도 {vp:.1f} < {threshold:.0f} (매수세 부족, src={source})"

    return {
        "passed": passed,
        "volume_power": round(vp, 2),
        "source": source,
        "threshold": threshold,
        "reason": reason,
    }


# ─────────────────────────────────────────────
# 통합 — 3종 게이트 평가
# ─────────────────────────────────────────────


def check_all_entry_gates(
    broker,
    ticker: str,
    current_price: int | None = None,
    require_all: bool | None = None,
) -> dict[str, Any]:
    """3종 진입 게이트 통합 평가.

    Args:
        broker: KIS broker (CachedBroker 가능)
        ticker: 종목 코드
        current_price: A 게이트에서 사용할 현재가 (None이면 fetch_price)
        require_all: True 시 3/3 모두 통과해야 all_passed=True
                     False 시 2/3 이상 통과 시 all_passed=True (완화 모드)
                     None 시 환경변수 ENTRY_GATE_REQUIRE_ALL 사용

    Returns:
        {
          "all_passed": bool,
          "passed_count": int,         # 0~3
          "intraday": {...},
          "orderbook": {...},
          "volume_power": {...},
          "summary": str,
          "require_all": bool,
        }
    """
    if require_all is None:
        require_all = REQUIRE_ALL

    a = check_intraday_gate(broker, ticker, current_price=current_price)
    b = check_orderbook_gate(broker, ticker)
    c = check_volume_power_gate(broker, ticker)

    passed_count = sum(1 for r in (a, b, c) if r.get("passed"))

    if require_all:
        all_passed = passed_count == 3
    else:
        all_passed = passed_count >= 2

    # 간결 요약
    parts = []
    parts.append(
        f"A[{'✓' if a['passed'] else '✗'}] 양봉 {a['bull_ratio']*100:.0f}%/"
        f"vol {(a.get('current_price', 0)):,}원"
    )
    parts.append(
        f"B[{'✓' if b['passed'] else '✗'}] 매수1/매도1 {b['bid_ask_ratio']:.2f}x"
    )
    parts.append(
        f"C[{'✓' if c['passed'] else '✗'}] 체결강도 {c['volume_power']:.0f}"
    )
    summary = f"{passed_count}/3 통과 — " + " | ".join(parts)

    return {
        "all_passed": all_passed,
        "passed_count": passed_count,
        "intraday": a,
        "orderbook": b,
        "volume_power": c,
        "summary": summary,
        "require_all": require_all,
    }


# ─────────────────────────────────────────────
# CLI dry-run
# ─────────────────────────────────────────────


def _format_pretty(result: dict[str, Any]) -> str:
    """게이트 결과를 사람용 멀티라인 텍스트로."""
    lines = [
        f"=== 진입 게이트 평가 (require_all={result['require_all']}) ===",
        f"  전체: {'✅ 통과' if result['all_passed'] else '❌ 차단'}"
        f" ({result['passed_count']}/3)",
        "",
        f"  A. 5분봉 진입 게이트   [{'✓' if result['intraday']['passed'] else '✗'}]",
        f"     source: {result['intraday']['source']}, n_bars: {result['intraday']['n_bars']}",
        f"     양봉비율: {result['intraday']['bull_ratio']*100:.0f}% "
        f"(요건 ≥{INTRADAY_BULL_RATIO_MIN*100:.0f}%)",
        f"     마지막봉 양봉: {result['intraday']['last_bull']}",
        f"     거래량 폭증: {result['intraday']['vol_surge']} "
        f"(요건 ≥{INTRADAY_VOL_SURGE_MIN}x)",
        f"     VWAP 안착: {result['intraday']['above_vwap']} "
        f"(현재가 {result['intraday']['current_price']:,} / VWAP {result['intraday']['vwap_last']:,.0f})",
        f"     {result['intraday']['reason']}",
        "",
        f"  B. 호가창 게이트       [{'✓' if result['orderbook']['passed'] else '✗'}]",
        f"     매수1/매도1: {result['orderbook']['bid_ask_ratio']:.2f}x "
        f"(요건 ≥{ORDERBOOK_BID_ASK_MIN})",
        f"     호가합 비율: {result['orderbook']['depth_ratio']:.2f}x "
        f"(요건 ≥{ORDERBOOK_DEPTH_RATIO_MIN})",
        f"     {result['orderbook']['reason']}",
        "",
        f"  C. 체결강도 게이트     [{'✓' if result['volume_power']['passed'] else '✗'}]",
        f"     체결강도: {result['volume_power']['volume_power']:.1f} "
        f"(요건 ≥{result['volume_power']['threshold']:.0f}, src={result['volume_power']['source']})",
        f"     {result['volume_power']['reason']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="진입 게이트 단독 평가 (dry-run)")
    parser.add_argument("ticker", nargs="?", default="005930", help="종목 코드 (기본: 005930 삼성전자)")
    parser.add_argument("--relax", action="store_true", help="2/3 통과도 허용 (require_all=False)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sys.path.insert(0, str(PROJECT_ROOT))
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv(PROJECT_ROOT / ".env")

    from src.adapters.kis_stock_data_adapter import KisStockDataAdapter  # noqa: E402

    broker = KisStockDataAdapter().broker
    result = check_all_entry_gates(broker, args.ticker, require_all=not args.relax)
    print(_format_pretty(result))
    sys.exit(0 if result["all_passed"] else 1)
