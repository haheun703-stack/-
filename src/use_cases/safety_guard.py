"""
v4.0 안전장치 — STOP.signal, 손실 한도, 긴급 전량 청산, 공휴일 감지

안전장치 우선순위:
1. STOP.signal 파일 → 즉시 매매 중단
2. 총 손실 -10% → 전량 시장가 청산 + STOP.signal 생성
3. 일일 손실 -3% → 당일 매매 중단 (보유 유지)
4. 공휴일/주말 → 매매 스킵
5. API 연결 실패 3회 → 매매 중단 + 알림
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

from src.entities.trading_models import SafetyState

logger = logging.getLogger(__name__)


class SafetyGuard:
    """자동매매 안전장치"""

    def __init__(self, config: dict | None = None):
        safety = (config or {}).get("live_trading", {}).get("safety", {})
        self.max_daily_loss_pct = safety.get("max_daily_loss_pct", -0.03)
        self.max_total_loss_pct = safety.get("max_total_loss_pct", -0.10)
        self.stop_signal_file = Path(safety.get("stop_signal_file", "STOP.signal"))
        self.reboot_trigger_file = Path(safety.get("reboot_trigger_file", "reboot.trigger"))
        self.emergency_sell_type = safety.get("emergency_sell_type", "market")
        self.api_fail_count = 0
        self.max_api_fails = 3

    # ──────────────────────────────────────────
    # 종합 상태 체크
    # ──────────────────────────────────────────

    def check_all(
        self,
        daily_loss_pct: float = 0.0,
        total_loss_pct: float = 0.0,
    ) -> SafetyState:
        """안전장치 전체 상태 체크"""
        state = SafetyState(
            stop_signal_active=self.check_stop_signal(),
            daily_loss_pct=daily_loss_pct,
            max_daily_loss_pct=self.max_daily_loss_pct,
            total_loss_pct=total_loss_pct,
            max_total_loss_pct=self.max_total_loss_pct,
            emergency_triggered=total_loss_pct <= self.max_total_loss_pct,
            is_holiday=self.check_holiday(),
        )

        # 로깅
        if state.stop_signal_active:
            logger.warning("[안전] STOP.signal 활성 — 매매 중단")
        if state.emergency_triggered:
            logger.critical("[안전] 총 손실 %.1f%% — 긴급 전량 청산 필요!", total_loss_pct * 100)
        if not state.can_trade and not state.stop_signal_active and not state.is_holiday:
            if daily_loss_pct <= self.max_daily_loss_pct:
                logger.warning("[안전] 일일 손실 %.1f%% — 당일 매매 중단", daily_loss_pct * 100)

        return state

    # ──────────────────────────────────────────
    # STOP.signal
    # ──────────────────────────────────────────

    def check_stop_signal(self) -> bool:
        """STOP.signal 파일 존재 확인"""
        return self.stop_signal_file.exists()

    def create_stop_signal(self, reason: str = "") -> None:
        """STOP.signal 생성"""
        content = f"STOP at {datetime.now().isoformat()}\nReason: {reason}\n"
        self.stop_signal_file.write_text(content, encoding="utf-8")
        logger.critical("[안전] STOP.signal 생성: %s", reason)

    def clear_stop_signal(self) -> None:
        """STOP.signal 삭제"""
        if self.stop_signal_file.exists():
            self.stop_signal_file.unlink()
            logger.info("[안전] STOP.signal 삭제 완료")

    # ──────────────────────────────────────────
    # reboot.trigger
    # ──────────────────────────────────────────

    def check_reboot_trigger(self) -> bool:
        """reboot.trigger 파일 감지"""
        if self.reboot_trigger_file.exists():
            self.reboot_trigger_file.unlink()
            logger.info("[안전] reboot.trigger 감지 — 재시작 요청")
            return True
        return False

    # ──────────────────────────────────────────
    # 손실 한도
    # ──────────────────────────────────────────

    def check_daily_loss(self, daily_loss_pct: float) -> bool:
        """일일 손실 한도 초과 여부 (True=초과)"""
        return daily_loss_pct <= self.max_daily_loss_pct

    def check_total_loss(self, total_loss_pct: float) -> bool:
        """총 손실 한도 초과 여부 (True=초과, 긴급청산 필요)"""
        return total_loss_pct <= self.max_total_loss_pct

    # ──────────────────────────────────────────
    # 긴급 전량 청산
    # ──────────────────────────────────────────

    def emergency_liquidate(self, tracker, order_port) -> list:
        """전종목 시장가 청산"""
        from src.entities.trading_models import ExitReason

        logger.critical("[안전] 긴급 전량 청산 시작!")
        results = []

        for pos in list(tracker.positions):
            try:
                order = order_port.sell_market(pos.ticker, pos.shares)
                results.append({
                    "ticker": pos.ticker,
                    "shares": pos.shares,
                    "order": order,
                })
                tracker.apply_partial_exit(pos, pos.shares, ExitReason.EMERGENCY)
                logger.info(
                    "[긴급청산] %s %d주 시장가 매도 접수", pos.ticker, pos.shares
                )
            except Exception as e:
                logger.error("[긴급청산] %s 매도 실패: %s", pos.ticker, e)
                results.append({"ticker": pos.ticker, "error": str(e)})

        # STOP.signal 생성
        self.create_stop_signal("긴급 전량 청산 (총 손실 한도 초과)")
        logger.critical("[안전] 긴급 전량 청산 완료: %d건 처리", len(results))
        return results

    # ──────────────────────────────────────────
    # 공휴일
    # ──────────────────────────────────────────

    def check_holiday(self) -> bool:
        """오늘이 공휴일/주말인지 확인"""
        today = date.today()
        # 주말 체크
        if today.weekday() >= 5:  # 토(5), 일(6)
            return True

        # holidays 라이브러리 사용 (설치된 경우)
        try:
            import holidays
            kr_holidays = holidays.KR(years=today.year)
            if today in kr_holidays:
                logger.info("[안전] 오늘은 공휴일: %s", kr_holidays.get(today))
                return True
        except ImportError:
            # holidays 미설치 시 주말만 체크
            pass

        return False

    # ──────────────────────────────────────────
    # API 실패 추적
    # ──────────────────────────────────────────

    def record_api_failure(self) -> bool:
        """API 실패 기록. 3회 연속 시 True 반환"""
        self.api_fail_count += 1
        if self.api_fail_count >= self.max_api_fails:
            logger.critical(
                "[안전] API 연결 실패 %d회 연속 — 매매 중단",
                self.api_fail_count,
            )
            return True
        return False

    def reset_api_failure(self) -> None:
        """API 성공 시 실패 카운터 초기화"""
        self.api_fail_count = 0
