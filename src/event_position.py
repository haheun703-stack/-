"""
v3.1 이벤트 드리븐 포지션 관리

- 시총 기반 포지션 사이징 (소형 2% / 대형 3%)
- 진입 제약 검증 (갭업 15% / RSI 65 / R:R 1.2)
- 청산 조건 체크 (TARGET / STOP_LOSS / RSI_70 / 체결강도 / 3일경과)
- 동시 이벤트 포지션 2개 이하

엔티티(news_models.EventPosition)에만 의존.
"""

from __future__ import annotations

import yaml
from pathlib import Path

from src.entities.news_models import EventPosition, NewsGrade

_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def _load_news_gate_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path) if config_path else _CFG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("news_gate", {})


class EventPositionManager:
    """이벤트 드리븐 포지션 관리"""

    def __init__(self, config_path: Path | str | None = None):
        cfg = _load_news_gate_config(config_path)
        ga = cfg.get("grade_a", {})
        self.max_gap_up_pct = ga.get("max_gap_up_pct", 15)
        self.rsi_max = ga.get("rsi_entry_max", 65)
        self.rr_min = ga.get("rr_min", 1.2)
        self.pos_pct = ga.get("position_size_pct", 3)
        self.pos_pct_small = ga.get("position_size_pct_small_cap", 2)
        self.max_hold = ga.get("max_hold_days", 3)
        self.settle_floor = ga.get("settlement_strength_floor", 95)
        self.max_concurrent = ga.get("max_concurrent_positions", 2)

        tgt = cfg.get("target", {})
        self.tgt1_rr = tgt.get("target_1_rr", 1.0)
        self.tgt2_rr = tgt.get("target_2_rr", 2.0)
        self.stop_atr_mult = tgt.get("stop_loss_atr_mult", 1.5)

    # ──────────────────────────────────────────
    # 진입 가능 여부 판단
    # ──────────────────────────────────────────

    def check_entry(
        self,
        news_grade: NewsGrade,
        current_price: float,
        prev_close: float,
        rsi: float,
        rr_ratio: float,
        active_event_count: int = 0,
    ) -> tuple[bool, str]:
        """
        이벤트 드리븐 진입 가능 여부 판단.

        Returns:
            (can_enter, reason)
        """
        if news_grade != NewsGrade.A:
            return False, "A급만 즉시 진입 허용"

        # 갭업 제한
        if prev_close > 0:
            gap_pct = (current_price - prev_close) / prev_close * 100
            if gap_pct > self.max_gap_up_pct:
                return False, f"갭업 {gap_pct:.1f}% > {self.max_gap_up_pct}% — 추격 금지"

        # RSI 제한
        if rsi > self.rsi_max:
            return False, f"RSI {rsi:.1f} > {self.rsi_max} — 되돌림 대기"

        # R:R 제한
        if rr_ratio < self.rr_min:
            return False, f"R:R {rr_ratio:.2f} < {self.rr_min} — 기준 미달"

        # 동시 포지션 제한
        if active_event_count >= self.max_concurrent:
            return False, f"동시 이벤트 포지션 {active_event_count}개 ≥ {self.max_concurrent} 제한"

        return True, "진입 가능"

    # ──────────────────────────────────────────
    # 이벤트 포지션 생성
    # ──────────────────────────────────────────

    def create_event_position(
        self,
        ticker: str,
        entry_price: float,
        atr: float,
        market_cap: float = 0,
        prev_close: float = 0,
    ) -> EventPosition:
        """
        이벤트 드리븐 포지션 생성.

        Args:
            ticker: 종목코드
            entry_price: 진입가
            atr: ATR 값
            market_cap: 시가총액 (원) — 1조 미만이면 소형주
            prev_close: 전일 종가 (갭업 계산용)
        """
        # 포지션 사이즈
        is_small_cap = market_cap < 1_000_000_000_000 if market_cap > 0 else True
        pos_pct = self.pos_pct_small if is_small_cap else self.pos_pct

        # 손절가 (ATR 기반)
        stop_loss = entry_price - (atr * self.stop_atr_mult)

        # 목표가 (R:R 기반)
        risk = entry_price - stop_loss
        target_1 = entry_price + risk * self.tgt1_rr
        target_2 = entry_price + risk * self.tgt2_rr

        # 갭업 비율
        gap_up_pct = 0.0
        if prev_close > 0:
            gap_up_pct = (entry_price - prev_close) / prev_close * 100

        from datetime import date
        return EventPosition(
            ticker=ticker,
            news_grade="A",
            entry_price=entry_price,
            target_1=round(target_1),
            target_2=round(target_2),
            stop_loss=round(stop_loss),
            position_pct=pos_pct,
            max_hold_days=self.max_hold,
            entry_date=date.today().isoformat(),
            gap_up_pct=round(gap_up_pct, 1),
        )

    # ──────────────────────────────────────────
    # 청산 조건 체크
    # ──────────────────────────────────────────

    def check_exit_conditions(
        self,
        position: EventPosition,
        current_price: float,
        rsi: float = 0,
        settlement_strength: float = 100,
        days_held: int = 0,
    ) -> list[tuple[str, str, float]]:
        """
        청산 조건 목록 반환.

        Returns:
            list of (exit_type, description, exit_ratio)
            exit_ratio: 0.5=반익, 1.0=전량 청산
        """
        exits = []

        # 1. 목표가 도달
        if current_price >= position.target_2:
            exits.append(("TARGET_2", "전량 청산 목표 도달", 1.0))
        elif current_price >= position.target_1:
            exits.append(("TARGET_1", "반익 목표 도달", 0.5))

        # 2. 손절
        if current_price <= position.stop_loss:
            exits.append(("STOP_LOSS", "손절가 이탈 — 전량 매도", 1.0))

        # 3. RSI 과매수
        if rsi >= 70:
            exits.append(("RSI_70", f"RSI {rsi:.1f} ≥ 70 — 반익 검토", 0.5))

        # 4. 체결강도 하락
        if settlement_strength < self.settle_floor:
            exits.append((
                "SETTLEMENT_WEAK",
                f"체결강도 {settlement_strength:.1f} < {self.settle_floor} — 반익",
                0.5,
            ))

        # 5. 시간 경과
        if days_held >= position.max_hold_days and current_price < position.target_1:
            exits.append((
                "TIME_DECAY",
                f"{days_held}일 경과 + 목표 미도달 — 청산",
                1.0,
            ))

        return exits
