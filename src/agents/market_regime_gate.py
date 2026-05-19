"""MarketRegimeGate — 시장 폭락장 검출 워커 (6번째 검수팀, 2026-05-19 신규)

배경 (5/19 11:14 KST 실측):
  - KODEX 200(069500) -5.05% (대형주 폭락)
  - KODEX 인버스(114800) +5.05%
  - KODEX 200선물인버스2X(252670) +9.48% (2배 인버스 폭등 = 시장 -5%)
  - 퀀트봇 강력포착 9건 모두 09:30 대비 -4~-8% 하락
  - 만약 오늘이 5/20이었다면 매수 → 즉시 -3% 손절 발동

기존 5명 검수팀(EnvChecker/CodeAuditor/FlowMonitor/DataIntegrity/Reporter)은
환경변수/코드/로그/데이터/리포트는 검증하지만 **시장 약세 자체는 검출 못 함**.
이 6번째 워커가 14:00 자동매매 가동 직전(13:55) 호출되어 폭락장이면
KILL_SWITCH 자동 활성화 → 매수 차단.

사용:
  python scripts/run_market_regime_gate.py            # 텔레그램 + 출력
  python scripts/run_market_regime_gate.py --no-tg    # 출력만
  FAIL 시 exit 1

cron 등록 (5/20 가동 직전):
  55 13 20 5 * cd ~/quantum-master && ./venv/bin/python3.11 \
      scripts/run_market_regime_gate.py >> /tmp/market_regime.log 2>&1

판정 로직:
  - KODEX 인버스(114800) +3% 이상 → BEARISH 트리거 1건
  - KODEX 200선물인버스2X(252670) +5% 이상 → BEARISH 트리거 1건
  - KODEX 200(069500) -2% 이하 → BEARISH 트리거 1건
  - 1건 이상 트리거 = BEARISH = KILL_SWITCH 자동 활성화 (보수적)
  - 2건 이상 트리거 = STRONG_BEARISH (라벨링용)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class MarketRegimeGate:
    """시장 폭락장 검출 워커 (6번째 검수팀, 5/19 실측 후 신규).

    14:00 자동매매 가동 직전 (13:55) 호출.
    폭락장 검출 시 KILL_SWITCH 자동 활성화 → 매수 차단.
    """

    # 임계값 (보수적, 폭락장 명백히 검출)
    THRESHOLDS = {
        "kodex_inverse_1x": 3.0,      # KODEX 인버스(114800) +3%+ → 약세
        "kodex_inverse_2x": 5.0,      # KODEX 200선물인버스2X(252670) +5%+ → 강한 약세
        "kodex_200_drop": -2.0,        # KODEX 200(069500) -2%- → 약세
    }

    TICKERS = {
        "114800": "KODEX 인버스",
        "252670": "KODEX 200선물인버스2X",
        "069500": "KODEX 200",
    }

    def __init__(self):
        self.timestamp = datetime.now().isoformat()

    def check_market_regime(self) -> dict:
        """현재 시장 약세 여부 판정.

        Returns:
            {
                "agent": "market_regime_gate",
                "status": "OK" | "FAIL",
                "regime": "NORMAL" | "BEARISH" | "STRONG_BEARISH",
                "indicators": [{"ticker": ..., "name": ..., "current": ...,
                                "change_pct": ..., "triggered": bool, "reason": ...}, ...],
                "triggered_count": N,
                "summary": "...",
                "timestamp": "...",
            }
        """
        indicators: list[dict] = []
        triggered: list[str] = []

        # KIS adapter (지연 import — 로컬 dry-run에서 mojito 미설치/키 누락 대비)
        broker = None
        adapter_err: str | None = None
        try:
            from src.adapters.kis_stock_data_adapter import KisStockDataAdapter
            adp = KisStockDataAdapter()
            broker = adp.broker
        except Exception as e:
            adapter_err = f"KIS adapter 초기화 실패: {e}"
            logger.warning("[MarketRegimeGate] %s", adapter_err)

        for ticker, name in self.TICKERS.items():
            if broker is None:
                indicators.append({
                    "ticker": ticker,
                    "name": name,
                    "error": adapter_err or "broker is None",
                    "triggered": False,
                })
                continue
            try:
                px = broker.fetch_price(ticker).get("output", {})
                chg = float(px.get("prdy_ctrt", 0) or 0)
                current = int(px.get("stck_prpr", 0) or 0)

                # 임계 판정
                is_triggered = False
                reason = ""
                if ticker == "114800" and chg >= self.THRESHOLDS["kodex_inverse_1x"]:
                    is_triggered = True
                    reason = (
                        f"KODEX 인버스 +{chg:.2f}% "
                        f">= +{self.THRESHOLDS['kodex_inverse_1x']}%"
                    )
                elif ticker == "252670" and chg >= self.THRESHOLDS["kodex_inverse_2x"]:
                    is_triggered = True
                    reason = (
                        f"KODEX 200선물인버스2X +{chg:.2f}% "
                        f">= +{self.THRESHOLDS['kodex_inverse_2x']}%"
                    )
                elif ticker == "069500" and chg <= self.THRESHOLDS["kodex_200_drop"]:
                    is_triggered = True
                    reason = (
                        f"KODEX 200 {chg:+.2f}% "
                        f"<= {self.THRESHOLDS['kodex_200_drop']}%"
                    )

                indicators.append({
                    "ticker": ticker,
                    "name": name,
                    "current": current,
                    "change_pct": chg,
                    "triggered": is_triggered,
                    "reason": reason,
                })
                if is_triggered:
                    triggered.append(reason)
            except Exception as e:
                indicators.append({
                    "ticker": ticker,
                    "name": name,
                    "error": str(e),
                    "triggered": False,
                })

        # regime 판정 (1건 이상 = BEARISH, 2건 이상 = STRONG_BEARISH)
        if len(triggered) >= 2:
            regime = "STRONG_BEARISH"
            status = "FAIL"
        elif len(triggered) == 1:
            regime = "BEARISH"
            status = "FAIL"
        else:
            regime = "NORMAL"
            status = "OK"

        summary = f"{regime} ({len(triggered)}/3 임계 초과)"
        result = {
            "agent": "market_regime_gate",
            "status": status,
            "regime": regime,
            "indicators": indicators,
            "triggered_count": len(triggered),
            "summary": summary,
            "timestamp": self.timestamp,
        }

        # Layer 7 — 약세 검출 시 KILL_SWITCH 자동 활성화 (5/19 결단 C)
        if status == "FAIL":
            try:
                from src.agents.kill_switch_manager import activate_kill_switch
                activate_kill_switch(
                    reason=f"시장 약세 검출: {', '.join(triggered)}",
                    source="MarketRegimeGate",
                    send_tg=True,  # RED 단일 채널
                )
            except Exception as e:
                logger.warning(
                    "[MarketRegimeGate] KILL_SWITCH 활성화 실패: %s", e
                )

        # latest.json 저장 (Reporter가 읽음)
        try:
            from src.agents.kill_switch_manager import save_worker_report
            save_worker_report("market_regime_gate", result)
        except Exception as e:
            logger.warning("[MarketRegimeGate] latest.json 저장 실패: %s", e)

        return result

    def report_to_telegram(self, result: dict) -> None:
        """AGENT_TELEGRAM_ENABLED=true 시만 카톡. 디폴트 OFF.

        KILL_SWITCH RED는 kill_switch_manager가 별도 발송 (단일 채널).
        이 메서드는 평상시 NORMAL 상태 로그용.
        """
        if os.environ.get("AGENT_TELEGRAM_ENABLED", "false").lower() != "true":
            logger.info(
                "[MarketRegimeGate] 결과 logger.info만 (AGENT_TELEGRAM_ENABLED=false): "
                "%s — regime=%s triggered=%d/3",
                result.get("status", "?"),
                result.get("regime", "?"),
                result.get("triggered_count", 0),
            )
            return

        try:
            from src.telegram_sender import send_message
        except Exception as e:
            logger.warning("telegram_sender import 실패: %s", e)
            return

        now_hhmm = datetime.now().strftime("%H:%M")
        if result["status"] == "OK":
            msg = (
                f"[MarketRegimeGate] NORMAL ({now_hhmm}) "
                f"— {result['triggered_count']}/3 임계 초과"
            )
        else:
            lines: list[str] = []
            for ind in result["indicators"]:
                if "error" in ind:
                    lines.append(f"  ! {ind['name']}: {ind['error']}")
                    continue
                emoji = "[!]" if ind["triggered"] else "[ ]"
                lines.append(
                    f"  {emoji} {ind['name']} {ind['change_pct']:+.2f}% "
                    f"({ind['current']:,}원)"
                )
            body = "\n".join(lines)
            msg = (
                f"[MarketRegimeGate] {result['regime']} ({now_hhmm})\n"
                f"{body}\n"
                f"  -> KILL_SWITCH 자동 활성화됨"
            )
        try:
            send_message(msg)
        except Exception as e:
            logger.error("텔레그램 발송 실패: %s", e)


def main() -> int:
    """CLI 진입점 (run_market_regime_gate.py에서 호출)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MarketRegimeGate — 시장 폭락장 검출 워커"
    )
    parser.add_argument("--no-tg", action="store_true", help="텔레그램 OFF, 출력만")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    gate = MarketRegimeGate()
    result = gate.check_market_regime()

    # 콘솔 출력
    print("=" * 60)
    print(f"  MarketRegimeGate {result['timestamp']}")
    print(f"  Status: {result['status']} | Regime: {result['regime']}")
    print(f"  Triggered: {result['triggered_count']}/3")
    print("=" * 60)
    for ind in result["indicators"]:
        if "error" in ind:
            print(f"  [ERR] {ind['name']}: {ind['error']}")
            continue
        mark = "[!]" if ind["triggered"] else "[ ]"
        print(
            f"  {mark} {ind['name']} {ind['change_pct']:+.2f}% "
            f"({ind['current']:,}원)"
        )
        if ind["triggered"]:
            print(f"      -> {ind['reason']}")

    # 텔레그램 (AGENT_TELEGRAM_ENABLED=true 시만)
    if not args.no_tg:
        gate.report_to_telegram(result)

    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
