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

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ───────────────────────────────────────────────────────────────
# 5/22 09:00 사고 보강: fail-safe retry + 임계 완화
# 배경: KIS access_token 1회 실패 → KILL_SWITCH 영구 활성화 → 1주차 워밍업 둘째날
#       매수 0건. 일시적 네트워크/토큰 장애를 "시장 붕괴"로 오판한 결함.
# 조치: ① fetch retry 3회 + 5초 대기, ② adapter 초기화 retry (25초 대기, KIS 토큰
#       1분 제한 회피), ③ fail-safe 임계 완화 — 1차 실패는 warning만, 연속 2회 cron
#       실패에만 KILL_SWITCH 활성화, ④ 성공 시 카운터 자동 리셋.
# ───────────────────────────────────────────────────────────────

FAIL_COUNTER_PATH = PROJECT_ROOT / "data" / "market_regime_fail_count.json"
FAIL_COUNTER_THRESHOLD = 2  # 연속 N회 cron 실패 시에만 KILL_SWITCH
FAIL_COUNTER_TTL_MIN = 90   # 카운터 자동 만료 (다음 cron까지 ~30분 + 여유)
KIS_INIT_RETRY_WAIT_SEC = 25  # KIS 토큰 1분 제한 회피 (5초 + 25초 = 30초)
FETCH_RETRY_COUNT = 3
FETCH_RETRY_WAIT_SEC = 5


def _read_fail_counter() -> dict:
    """연속 실패 카운터 읽기 (TTL 만료 시 0 반환)."""
    if not FAIL_COUNTER_PATH.exists():
        return {"count": 0, "last_reason": None, "updated_at": None}
    try:
        data = json.loads(FAIL_COUNTER_PATH.read_text())
        updated = data.get("updated_at")
        if updated:
            updated_dt = datetime.fromisoformat(updated)
            elapsed_min = (datetime.now() - updated_dt).total_seconds() / 60
            if elapsed_min > FAIL_COUNTER_TTL_MIN:
                return {"count": 0, "last_reason": None, "updated_at": None}
        return data
    except Exception:
        return {"count": 0, "last_reason": None, "updated_at": None}


def _write_fail_counter(count: int, reason: str) -> None:
    """연속 실패 카운터 저장."""
    FAIL_COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAIL_COUNTER_PATH.write_text(json.dumps({
        "count": count,
        "last_reason": reason,
        "updated_at": datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2))


def _reset_fail_counter() -> None:
    """성공 시 카운터 리셋."""
    if FAIL_COUNTER_PATH.exists():
        try:
            FAIL_COUNTER_PATH.unlink()
        except Exception:
            pass


def _send_warning_tg(msg: str) -> None:
    """Warning 텔레그램 (KILL_SWITCH 활성화 X, 일시 장애 통지용)."""
    try:
        import requests
        bot = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        if bot and chat:
            requests.post(
                f"https://api.telegram.org/bot{bot}/sendMessage",
                json={"chat_id": chat, "text": msg},
                timeout=5,
            )
    except Exception as e:
        logger.warning("[MarketRegimeGate] warning 텔레그램 실패: %s", e)


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
        # C2: timestamp 포맷을 다른 4명 워커와 통일 (YYYY-MM-DD HH:MM:SS)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def check_market_regime(self) -> dict:
        """현재 시장 약세 여부 판정.

        Returns:
            {
                "agent": "market_regime_gate",
                "status": "OK" | "FAIL",
                "regime": "NORMAL" | "BEARISH" | "STRONG_BEARISH" | "UNKNOWN",
                "indicators": [{"ticker": ..., "name": ..., "current": ...,
                                "change_pct": ..., "triggered": bool, "reason": ...}, ...],
                "triggered_count": N,
                "summary": "...",
                "timestamp": "...",
            }
        """
        # ──────────────────────────────────────────────────────────────
        # C3-A: MODEL=REAL 검증 (fail-safe 디폴트)
        # 모의투자 모드(MOCK)에서 시세 호출하면 모의서버 데이터로 잘못 판정.
        # MODEL!=REAL이면 즉시 FAIL → KILL_SWITCH 자동 활성화.
        # ──────────────────────────────────────────────────────────────
        model = os.environ.get("MODEL", "MOCK")
        if model != "REAL":
            result = {
                "agent": "market_regime_gate",
                "status": "FAIL",
                "regime": "UNKNOWN",
                "indicators": [],
                "triggered_count": 0,
                "summary": f"MODEL={model} != REAL — 모의 모드 차단",
                "timestamp": self.timestamp,
            }
            try:
                from src.agents.kill_switch_manager import (
                    activate_kill_switch,
                    save_worker_report,
                )
                activate_kill_switch(
                    reason=f"MarketRegimeGate fail-safe: MODEL={model} (실전 모드 아님)",
                    source="MarketRegimeGate",
                    send_tg=True,
                )
                save_worker_report("market_regime_gate", result)
            except Exception as e:
                logger.error(
                    "[MarketRegimeGate] MODEL 검증 후 활성화 실패: %s", e
                )
            return result

        indicators: list[dict] = []
        triggered: list[str] = []

        # KIS adapter (5/22 보강: 2회 retry — KIS 토큰 1분 제한 회피)
        broker = None
        adapter_err: str | None = None
        for attempt in range(2):
            try:
                from src.adapters.kis_stock_data_adapter import KisStockDataAdapter
                adp = KisStockDataAdapter()
                broker = adp.broker
                if broker is not None:
                    break
            except Exception as e:
                adapter_err = f"KIS adapter 초기화 실패(attempt={attempt+1}/2): {e}"
                logger.warning("[MarketRegimeGate] %s", adapter_err)
                if attempt == 0:
                    logger.info(
                        "[MarketRegimeGate] %d초 대기 후 재시도 (KIS 토큰 1분 제한 회피)",
                        KIS_INIT_RETRY_WAIT_SEC,
                    )
                    time.sleep(KIS_INIT_RETRY_WAIT_SEC)

        for ticker, name in self.TICKERS.items():
            if broker is None:
                indicators.append({
                    "ticker": ticker,
                    "name": name,
                    "error": adapter_err or "broker is None",
                    "triggered": False,
                })
                continue
            # 5/22 보강: fetch_price 3회 retry + 5초 대기 (일시 네트워크 장애 회복)
            px = None
            fetch_err: str | None = None
            for attempt in range(FETCH_RETRY_COUNT):
                try:
                    px = broker.fetch_price(ticker).get("output", {})
                    if px:
                        break
                except Exception as e:
                    fetch_err = str(e)
                    logger.warning(
                        "[MarketRegimeGate] fetch_price %s 실패(attempt=%d/%d): %s",
                        ticker, attempt + 1, FETCH_RETRY_COUNT, fetch_err,
                    )
                if attempt < FETCH_RETRY_COUNT - 1:
                    time.sleep(FETCH_RETRY_WAIT_SEC)

            if not px:
                indicators.append({
                    "ticker": ticker,
                    "name": name,
                    "error": fetch_err or f"empty output after {FETCH_RETRY_COUNT} retries",
                    "triggered": False,
                })
                continue

            try:
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

        # ──────────────────────────────────────────────────────────────
        # C3-B: fetch_price 다수 실패 시 fail-safe
        # 3종목 중 2건 이상 fetch 실패 = 시장 데이터 수신 불가 → FAIL.
        # 0건만 triggered인데 정상 판정하면 "데이터 없음 = NORMAL" 오판 위험.
        # ──────────────────────────────────────────────────────────────
        # 5/22 09:00 사고 보강: 1차 실패는 warning만, 연속 N회 실패에만 KILL_SWITCH
        error_count = sum(1 for ind in indicators if "error" in ind)
        if error_count >= 2:
            reason_text = f"{error_count}/3 fetch 실패"
            counter = _read_fail_counter()
            new_count = counter.get("count", 0) + 1

            if new_count < FAIL_COUNTER_THRESHOLD:
                # 1차 실패 — warning 텔레그램만, KILL_SWITCH 활성화 X
                _write_fail_counter(new_count, reason_text)
                warning_msg = (
                    f"⚠️ [MarketRegimeGate] WARNING ({new_count}/{FAIL_COUNTER_THRESHOLD})\n"
                    f"시장 데이터 일시 수신 실패: {reason_text}\n"
                    f"다음 cron(약 30분 후) 재시도 — KILL_SWITCH 미활성화\n"
                    f"연속 {FAIL_COUNTER_THRESHOLD}회 실패 시 자동 차단"
                )
                _send_warning_tg(warning_msg)
                logger.warning(
                    "[MarketRegimeGate] 1차 fetch 실패 — warning만 (count=%d/%d)",
                    new_count, FAIL_COUNTER_THRESHOLD,
                )
                return {
                    "agent": "market_regime_gate",
                    "status": "FAIL_TRANSIENT",
                    "regime": "UNKNOWN",
                    "indicators": indicators,
                    "triggered_count": 0,
                    "error_count": error_count,
                    "fail_counter": new_count,
                    "summary": (
                        f"시장 데이터 일시 수신 실패 ({reason_text}, "
                        f"{new_count}/{FAIL_COUNTER_THRESHOLD}회) — warning만"
                    ),
                    "timestamp": self.timestamp,
                }

            # 연속 FAIL_COUNTER_THRESHOLD 회 이상 — KILL_SWITCH 활성화
            _write_fail_counter(new_count, reason_text)
            result = {
                "agent": "market_regime_gate",
                "status": "FAIL",
                "regime": "UNKNOWN",
                "indicators": indicators,
                "triggered_count": 0,
                "error_count": error_count,
                "fail_counter": new_count,
                "summary": (
                    f"시장 데이터 수신 불가 ({reason_text}, "
                    f"연속 {new_count}회) — KILL_SWITCH 활성화"
                ),
                "timestamp": self.timestamp,
            }
            try:
                from src.agents.kill_switch_manager import (
                    activate_kill_switch,
                    save_worker_report,
                )
                activate_kill_switch(
                    reason=(
                        f"MarketRegimeGate fail-safe: 연속 {new_count}회 cron "
                        f"fetch 실패 ({reason_text})"
                    ),
                    source="MarketRegimeGate",
                    send_tg=True,
                )
                save_worker_report("market_regime_gate", result)
            except Exception as e:
                logger.error(
                    "[MarketRegimeGate] 연속 실패 fail-safe 처리 중 활성화 실패: %s",
                    e,
                )
            return result

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
            # 5/22 보강: 성공 시 fail 카운터 자동 리셋
            _reset_fail_counter()

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
