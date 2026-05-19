"""MarketScanner — 장중 5분 자동 시장 스캔 (5/19 사장님 지시 강화 v2).

목적:
- 메인 AI가 응답 시작 시 시장 흐름 즉시 파악
- 사장님이 "시장 봤나" 묻기 전에 선제 정보 준비
- KILL_SWITCH 활성화 안 함 (정보만)

차이점: MarketRegimeGate는 매수 차단 결정, MarketScanner는 종일 흐름 추적.

스캔 항목 (5/19 13:40 사장님 지시 v2):
1. 시장 흐름 ETF: KODEX 200(069500) + KODEX 인버스(114800) + KODEX 200선물인버스2X(252670)
2. 코스닥 인버스: KODEX 코스닥150선물인버스(251340)
3. 안전자산: KODEX 골드선물(132030) + KODEX 미국달러선물(261240)
4. ★대장주 추가: 삼성전자(005930) + SK하이닉스(000660)
5. tomorrow_picks 강력포착 9건 현재가
6. ★분봉 흐름: 직전 5분 스캔 대비 변동률 (change_5min_pct) 자동 계산

cron: 매 5분 (기존 15분 → 단축, 사장님 "15분마다는 부족")
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TOMORROW_PICKS = PROJECT_ROOT / "data" / "tomorrow_picks.json"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"
LATEST_REPORT = PROJECT_ROOT / "data" / "agent_reports" / "market_scanner_latest.json"

logger = logging.getLogger(__name__)


MARKET_TICKERS = {
    "069500": "KODEX 200",
    "114800": "KODEX 인버스",
    "252670": "KODEX 200선물인버스2X",
    "251340": "KODEX 코스닥150선물인버스",
    "132030": "KODEX 골드선물(H)",
    "261240": "KODEX 미국달러선물",
}

# ★ 5/19 사장님 지시: 대장주 추가 (분봉 흐름 핵심 종목)
LEADER_TICKERS = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
}

# 통합 (스캔 대상 8종)
ALL_MARKET_TICKERS = {**MARKET_TICKERS, **LEADER_TICKERS}


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

    def _load_previous_scan(self) -> dict | None:
        """직전 5분 스캔 결과 로드 (분봉 흐름 계산용).

        5~7분 이내 데이터만 유효. cron 5분 간격이지만 약간 여유(7분 컷오프).
        """
        if not LATEST_REPORT.exists():
            return None
        try:
            prev = json.loads(LATEST_REPORT.read_text(encoding="utf-8"))
            prev_ts_str = prev.get("timestamp", "")
            if not prev_ts_str:
                return None
            prev_ts = datetime.strptime(prev_ts_str, "%Y-%m-%d %H:%M:%S")
            if datetime.now() - prev_ts > timedelta(minutes=7):
                logger.debug("[MarketScanner] 이전 스캔이 7분 초과 (stale) — 분봉 흐름 SKIP")
                return None
            return prev
        except Exception as e:
            logger.debug("[MarketScanner] 이전 스캔 로드 실패: %s", e)
            return None

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

        # ★ 직전 5분 스캔 로드 (분봉 흐름 계산용)
        prev_scan = self._load_previous_scan()
        prev_market = prev_scan.get("market", {}) if prev_scan else {}
        prev_picks = {p.get("ticker"): p for p in (prev_scan.get("picks", []) if prev_scan else []) if p.get("ticker")}

        # 1. 시장 흐름 fetch (8종: 기존 6 + 대장주 2)
        market_data = {}
        fetch_errors = 0
        for tk, nm in ALL_MARKET_TICKERS.items():
            try:
                px = broker.fetch_price(tk).get("output", {})
                current = int(px.get("stck_prpr", 0))
                chg_prev_day = float(px.get("prdy_ctrt", 0) or 0)

                # ★ 5분 전 대비 변동률
                change_5min_pct = None
                prev_info = prev_market.get(tk, {})
                prev_price = prev_info.get("current")
                if prev_price and prev_price > 0 and current > 0:
                    change_5min_pct = round((current - prev_price) / prev_price * 100, 3)

                market_data[tk] = {
                    "name": nm,
                    "current": current,
                    "change_pct": chg_prev_day,
                    "change_5min_pct": change_5min_pct,
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
                        current = int(px.get("stck_prpr", 0))
                        chg_prev_day = float(px.get("prdy_ctrt", 0) or 0)

                        # ★ 5분 전 대비 변동률 (picks)
                        change_5min_pct = None
                        prev_pick = prev_picks.get(tk, {})
                        prev_price = prev_pick.get("current")
                        if prev_price and prev_price > 0 and current > 0:
                            change_5min_pct = round((current - prev_price) / prev_price * 100, 3)

                        picks_data.append({
                            "ticker": tk,
                            "name": nm,
                            "current": current,
                            "change_pct": chg_prev_day,
                            "change_5min_pct": change_5min_pct,
                        })
                    except Exception as e:
                        picks_data.append({"ticker": tk, "name": nm, "error": str(e)[:80]})
            except Exception as e:
                logger.warning("[MarketScanner] picks 로드 실패: %s", e)
        result["picks"] = picks_data

        # 3. summary 생성 (대장주 + 분봉 흐름 포함, 5/19 v2)
        k200 = market_data.get("069500", {})
        samsung = market_data.get("005930", {})
        sk_hynix = market_data.get("000660", {})
        inverse2x = market_data.get("252670", {})

        parts = []
        if "change_pct" in k200:
            p5 = k200.get("change_5min_pct")
            p5_str = f" (5분 {p5:+.2f}%)" if p5 is not None else ""
            parts.append(f"KODEX 200 {k200['change_pct']:+.2f}%{p5_str}")
        if "change_pct" in samsung:
            p5 = samsung.get("change_5min_pct")
            p5_str = f" (5분 {p5:+.2f}%)" if p5 is not None else ""
            parts.append(f"삼성전자 {samsung['change_pct']:+.2f}%{p5_str}")
        if "change_pct" in sk_hynix:
            p5 = sk_hynix.get("change_5min_pct")
            p5_str = f" (5분 {p5:+.2f}%)" if p5 is not None else ""
            parts.append(f"SK하이닉스 {sk_hynix['change_pct']:+.2f}%{p5_str}")
        if "change_pct" in inverse2x:
            parts.append(f"인버스2X {inverse2x['change_pct']:+.2f}%")

        # 강력포착 평균
        valid_picks = [p for p in picks_data if "change_pct" in p]
        if valid_picks:
            picks_avg = sum(p["change_pct"] for p in valid_picks) / len(valid_picks)
            parts.append(f"강력포착 {len(valid_picks)}/9 평균 {picks_avg:+.2f}%")

        if parts:
            result["summary"] = " | ".join(parts)
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
    print(f"\n시장 흐름 (대장주 포함 {len(result.get('market', {}))}종):")
    for tk, info in result.get("market", {}).items():
        if "error" in info:
            print(f"  {info['name']}: ERROR")
        else:
            emoji = "🔴" if info["change_pct"] < -1 else ("🟢" if info["change_pct"] > 1 else "⚪")
            # 5분 흐름 표시
            chg_5min = info.get("change_5min_pct")
            chg_5min_str = ""
            if chg_5min is not None:
                chg_5min_emoji = "📈" if chg_5min > 0.5 else ("📉" if chg_5min < -0.5 else "➖")
                chg_5min_str = f"  {chg_5min_emoji} 5분 {chg_5min:+.2f}%"
            print(f"  {emoji} {info['name']:>25}: {info['current']:>7,}원 ({info['change_pct']:+.2f}%){chg_5min_str}")

    print(f"\n강력포착 픽 {len(result.get('picks', []))}건:")
    for p in result.get("picks", []):
        if "error" in p:
            print(f"  {p['name']}: ERROR")
        else:
            emoji = "🔴" if p["change_pct"] < -2 else ("🟢" if p["change_pct"] > 2 else "⚪")
            chg_5min = p.get("change_5min_pct")
            chg_5min_str = ""
            if chg_5min is not None:
                chg_5min_emoji = "📈" if chg_5min > 0.5 else ("📉" if chg_5min < -0.5 else "➖")
                chg_5min_str = f"  {chg_5min_emoji} 5분 {chg_5min:+.2f}%"
            print(f"  {emoji} {p['name']:>20}: {p['current']:>7,}원 ({p['change_pct']:+.2f}%){chg_5min_str}")

    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
