"""MarketScanner — 장중 15분 자동 시장 스캔 (5/19 사장님 지시 강화).

목적:
- 메인 AI가 응답 시작 시 시장 흐름 즉시 파악
- 사장님이 "시장 봤나" 묻기 전에 선제 정보 준비
- KILL_SWITCH 활성화 안 함 (정보만)

차이점: MarketRegimeGate는 매수 차단 결정, MarketScanner는 종일 흐름 추적.

스캔 항목:
1. 시장 흐름: KODEX 200(069500) + KODEX 인버스(114800) + KODEX 200선물인버스2X(252670)
2. 코스닥 인버스: KODEX 코스닥150선물인버스(251340)
3. 안전자산: KODEX 골드선물(132030) + KODEX 미국달러선물(261240)
4. tomorrow_picks 강력포착 9건 현재가
5. 09:30 SNAPSHOT 대비 변동률 (snapshot json 있으면)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TOMORROW_PICKS = PROJECT_ROOT / "data" / "tomorrow_picks.json"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"

logger = logging.getLogger(__name__)


MARKET_TICKERS = {
    "069500": "KODEX 200",
    "114800": "KODEX 인버스",
    "252670": "KODEX 200선물인버스2X",
    "251340": "KODEX 코스닥150선물인버스",
    "132030": "KODEX 골드선물(H)",
    "261240": "KODEX 미국달러선물",
}


def is_market_open() -> bool:
    """현재 정규장 시간인지 (평일 09:00~15:30 KST)."""
    now = datetime.now()
    if now.weekday() >= 5:  # 토일
        return False
    return time(9, 0) <= now.time() <= time(15, 30)


class MarketScanner:
    """장중 시장 흐름 자동 스캔."""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def scan(self) -> dict:
        """전체 스캔.

        Returns:
            {
                "agent": "market_scanner",
                "status": "OK" | "FAIL",
                "timestamp": "...",
                "market_open": True | False,
                "market": {"069500": {...}, ...},
                "picks": [{"ticker": "...", "name": "...", "current": ..., "change_pct": ...}, ...],
                "summary": "KODEX 200 -3.42%, 인버스 +3.28% (5/5 fetched)",
            }
        """
        result = {
            "agent": "market_scanner",
            "status": "OK",
            "timestamp": self.timestamp,
            "market_open": is_market_open(),
        }

        # KIS adapter 초기화
        try:
            from src.adapters.kis_stock_data_adapter import KisStockDataAdapter
            adp = KisStockDataAdapter()
            broker = adp.broker
        except Exception as e:
            logger.warning("[MarketScanner] KIS adapter 초기화 실패: %s", e)
            result["status"] = "FAIL"
            result["error"] = str(e)
            # 실패도 latest.json에 기록 (메인 AI가 상태 파악)
            try:
                from src.agents.kill_switch_manager import save_worker_report
                save_worker_report("market_scanner", result)
            except Exception:
                pass
            return result

        # 1. 시장 흐름 fetch
        market_data = {}
        fetch_errors = 0
        for tk, nm in MARKET_TICKERS.items():
            try:
                px = broker.fetch_price(tk).get("output", {})
                market_data[tk] = {
                    "name": nm,
                    "current": int(px.get("stck_prpr", 0)),
                    "change_pct": float(px.get("prdy_ctrt", 0) or 0),
                }
            except Exception as e:
                market_data[tk] = {"name": nm, "error": str(e)[:80]}
                fetch_errors += 1
        result["market"] = market_data

        # 2. tomorrow_picks 9건 (있으면)
        picks_data = []
        if TOMORROW_PICKS.exists():
            try:
                picks_json = json.loads(TOMORROW_PICKS.read_text(encoding="utf-8"))
                strong = [p for p in picks_json.get("picks", []) if p.get("grade") == "강력 포착"][:9]
                for p in strong:
                    tk = p.get("ticker", "")
                    nm = p.get("name", tk)
                    try:
                        px = broker.fetch_price(tk).get("output", {})
                        picks_data.append({
                            "ticker": tk,
                            "name": nm,
                            "current": int(px.get("stck_prpr", 0)),
                            "change_pct": float(px.get("prdy_ctrt", 0) or 0),
                        })
                    except Exception as e:
                        picks_data.append({"ticker": tk, "name": nm, "error": str(e)[:80]})
            except Exception as e:
                logger.warning("[MarketScanner] picks 로드 실패: %s", e)
        result["picks"] = picks_data

        # 3. summary 생성 (1줄)
        k200 = market_data.get("069500", {})
        inverse = market_data.get("114800", {})
        if "change_pct" in k200 and "change_pct" in inverse:
            k200_chg = k200["change_pct"]
            inv_chg = inverse["change_pct"]
            # picks 평균
            valid_picks = [p for p in picks_data if "change_pct" in p]
            picks_avg = sum(p["change_pct"] for p in valid_picks) / len(valid_picks) if valid_picks else 0
            result["summary"] = (
                f"KODEX 200 {k200_chg:+.2f}%, 인버스 {inv_chg:+.2f}% | "
                f"강력포착 {len(valid_picks)}/9 평균 {picks_avg:+.2f}%"
            )
        else:
            result["summary"] = f"시장 fetch 일부 실패 ({fetch_errors}건)"

        # latest.json 저장 (메인 AI가 응답 시작 시 읽음)
        try:
            from src.agents.kill_switch_manager import save_worker_report
            save_worker_report("market_scanner", result)
        except Exception as e:
            logger.warning("[MarketScanner] save_worker_report 실패: %s", e)

        logger.info("[MarketScanner] %s", result["summary"])
        return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not is_market_open():
        logger.info("[MarketScanner] 장 마감 시간 — 스캔 SKIP")
        return 0

    scanner = MarketScanner()
    result = scanner.scan()

    print(f"\n=== MarketScanner {result['timestamp']} ===")
    print(f"Status: {result['status']}")
    print(f"Summary: {result.get('summary', '?')}")
    print(f"\n시장 흐름:")
    for tk, info in result.get("market", {}).items():
        if "error" in info:
            print(f"  {info['name']}: ERROR")
        else:
            emoji = "🔴" if info["change_pct"] < -1 else ("🟢" if info["change_pct"] > 1 else "⚪")
            print(f"  {emoji} {info['name']:>25}: {info['current']:>7,}원 ({info['change_pct']:+.2f}%)")

    print(f"\n강력포착 픽 {len(result.get('picks', []))}건:")
    for p in result.get("picks", []):
        if "error" in p:
            print(f"  {p['name']}: ERROR")
        else:
            emoji = "🔴" if p["change_pct"] < -2 else ("🟢" if p["change_pct"] > 2 else "⚪")
            print(f"  {emoji} {p['name']:>20}: {p['current']:>7,}원 ({p['change_pct']:+.2f}%)")

    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
