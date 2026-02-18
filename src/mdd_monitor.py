"""
MDD 모니터 — 실시간 낙폭 추적 + 백테스트 비교 알림.

잼블랙 인사이트: "백테스팅 코드 다시 돌려봤는데 아무것도 틀린 게 없어서 믿고 갔다."
→ 실전 MDD 발생 시 백테스트 예상 범위와 자동 비교하여 멘탈 관리 지원.

사용법:
    monitor = MDDMonitor()
    monitor.update(current_equity=98_500_000)  # 매일 장마감 후 호출
    alert = monitor.get_alert()  # 텔레그램 알림용
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

EQUITY_FILE = Path("data/equity_tracker.json")

# 백테스트 기준값 (v10.3 C_new, 84종목, 0.5% slippage)
BACKTEST_BENCHMARKS = {
    "expected_mdd_pct": -4.5,       # 백테스트 MDD
    "expected_cagr_pct": 21.7,      # 백테스트 연수익률
    "warn_mdd_pct": -3.0,           # 주의 알림 (백테스트 MDD의 67%)
    "danger_mdd_pct": -6.0,         # 위험 알림 (백테스트 MDD의 133%)
    "critical_mdd_pct": -10.0,      # 긴급 알림 (백테스트 MDD의 222%)
}


class MDDMonitor:
    """일일 자산 추적 + MDD 계산 + 백테스트 비교."""

    def __init__(self, initial_capital: int = 100_000_000):
        self.initial_capital = initial_capital
        self.data = self._load()

    def _load(self) -> dict:
        """equity_tracker.json 로드."""
        default = {
            "initial_capital": self.initial_capital,
            "peak_equity": self.initial_capital,
            "peak_date": "",
            "current_equity": self.initial_capital,
            "current_mdd_pct": 0.0,
            "max_mdd_pct": 0.0,
            "max_mdd_date": "",
            "daily_log": [],       # [{date, equity, mdd_pct}, ...]
            "alert_history": [],   # 알림 발송 이력 (중복 방지)
        }
        if EQUITY_FILE.exists():
            try:
                data = json.loads(EQUITY_FILE.read_text(encoding="utf-8"))
                # 기존 데이터에 누락 키 보충
                for k, v in default.items():
                    if k not in data:
                        data[k] = v
                return data
            except Exception as e:
                logger.error("[MDD모니터] 로드 실패: %s", e)
        return default

    def _save(self) -> None:
        """equity_tracker.json 저장."""
        EQUITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        EQUITY_FILE.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def update(self, current_equity: float) -> dict:
        """일일 자산 업데이트 + MDD 재계산.

        Args:
            current_equity: 현재 총 자산 (현금 + 평가금액)

        Returns:
            {"mdd_pct": float, "level": str, "message": str}
        """
        today_str = str(date.today())
        self.data["current_equity"] = current_equity

        # 고점 갱신
        if current_equity > self.data["peak_equity"]:
            self.data["peak_equity"] = current_equity
            self.data["peak_date"] = today_str

        # MDD 계산
        peak = self.data["peak_equity"]
        if peak > 0:
            mdd_pct = round((current_equity - peak) / peak * 100, 2)
        else:
            mdd_pct = 0.0

        self.data["current_mdd_pct"] = mdd_pct

        # 최대 MDD 기록
        if mdd_pct < self.data["max_mdd_pct"]:
            self.data["max_mdd_pct"] = mdd_pct
            self.data["max_mdd_date"] = today_str

        # 일일 로그 (같은 날 중복 방지)
        log = self.data["daily_log"]
        if log and log[-1].get("date") == today_str:
            log[-1]["equity"] = current_equity
            log[-1]["mdd_pct"] = mdd_pct
        else:
            log.append({
                "date": today_str,
                "equity": current_equity,
                "mdd_pct": mdd_pct,
            })
        # 최근 90일만 유지
        if len(log) > 90:
            self.data["daily_log"] = log[-90:]

        self._save()

        # 알림 레벨 판정
        level = self._classify_mdd(mdd_pct)
        return {
            "mdd_pct": mdd_pct,
            "level": level,
            "message": self._build_message(current_equity, mdd_pct, level),
        }

    def _classify_mdd(self, mdd_pct: float) -> str:
        """MDD 수준 분류."""
        b = BACKTEST_BENCHMARKS
        if mdd_pct >= 0:
            return "normal"
        elif mdd_pct > b["warn_mdd_pct"]:
            return "normal"
        elif mdd_pct > b["expected_mdd_pct"]:
            return "warn"       # 주의: 백테스트 MDD 접근 중
        elif mdd_pct > b["danger_mdd_pct"]:
            return "expected"   # 백테스트 예상 범위 내
        elif mdd_pct > b["critical_mdd_pct"]:
            return "danger"     # 백테스트 초과
        else:
            return "critical"   # 긴급

    def _build_message(self, equity: float, mdd_pct: float, level: str) -> str:
        """텔레그램 알림 메시지 생성."""
        b = BACKTEST_BENCHMARKS
        peak = self.data["peak_equity"]
        loss_amount = equity - peak

        level_info = {
            "normal": ("", "정상 운영 중"),
            "warn": ("\u26a0\ufe0f", "MDD 주의 — 백테스트 MDD 접근 중"),
            "expected": ("\u2705", "예상 범위 내 — 시스템 정상 동작"),
            "danger": ("\u274c", "백테스트 MDD 초과 — 주시 필요"),
            "critical": ("\ud83d\udea8", "긴급 — 시스템 점검 필요"),
        }
        icon, desc = level_info.get(level, ("", ""))

        if level == "normal":
            return ""  # 정상이면 알림 안 보냄

        lines = [
            f"{icon} [MDD 모니터] {desc}",
            "\u2500" * 26,
            f"  현재 자산: {equity:,.0f}원",
            f"  고점 자산: {peak:,.0f}원",
            f"  손실금액: {loss_amount:+,.0f}원",
            f"  현재 MDD: {mdd_pct:.1f}%",
            "\u2500" * 26,
            f"  \U0001f4ca 백테스트 기준 비교",
            f"  예상 MDD: {b['expected_mdd_pct']}%",
            f"  위험 MDD: {b['danger_mdd_pct']}%",
        ]

        # 멘탈 관리 메시지
        if level == "expected":
            lines.append(f"\n  \u2705 현재 낙폭은 백테스트에서 예상된 범위입니다.")
            lines.append(f"  \u2705 시스템을 믿고 유지하세요.")
        elif level == "danger":
            lines.append(f"\n  \u26a0\ufe0f 백테스트 MDD({b['expected_mdd_pct']}%)를 초과했습니다.")
            lines.append(f"  \u26a0\ufe0f 추가 진입을 보류하고 관망하세요.")
        elif level == "critical":
            lines.append(f"\n  \ud83d\udea8 심각한 낙폭입니다. 시스템 점검이 필요합니다.")
            lines.append(f"  \ud83d\udea8 신규 진입을 중단하세요.")

        lines.append("\u2500" * 26)
        return "\n".join(lines)

    def get_alert(self) -> str | None:
        """현재 MDD 상태 알림 반환 (중복 방지 포함).

        Returns:
            알림 메시지 or None (정상이거나 오늘 이미 발송)
        """
        mdd_pct = self.data.get("current_mdd_pct", 0.0)
        level = self._classify_mdd(mdd_pct)

        if level == "normal":
            return None

        # 오늘 같은 레벨 알림 이미 발송했으면 스킵
        today_str = str(date.today())
        history = self.data.get("alert_history", [])
        for h in history:
            if h.get("date") == today_str and h.get("level") == level:
                return None

        # 알림 발송 기록
        history.append({"date": today_str, "level": level})
        # 최근 30일만 유지
        if len(history) > 30:
            self.data["alert_history"] = history[-30:]
        self._save()

        return self._build_message(
            self.data["current_equity"], mdd_pct, level,
        )

    def get_status(self) -> dict:
        """현재 MDD 상태 요약 (텔레그램 /상태 명령용)."""
        return {
            "current_equity": self.data.get("current_equity", 0),
            "peak_equity": self.data.get("peak_equity", 0),
            "current_mdd_pct": self.data.get("current_mdd_pct", 0.0),
            "max_mdd_pct": self.data.get("max_mdd_pct", 0.0),
            "max_mdd_date": self.data.get("max_mdd_date", ""),
            "peak_date": self.data.get("peak_date", ""),
            "backtest_mdd": BACKTEST_BENCHMARKS["expected_mdd_pct"],
            "level": self._classify_mdd(self.data.get("current_mdd_pct", 0.0)),
            "days_tracked": len(self.data.get("daily_log", [])),
        }

    def format_status_line(self) -> str:
        """한 줄 요약 (다른 리포트에 삽입용)."""
        s = self.get_status()
        level_emoji = {
            "normal": "\U0001f7e2",
            "warn": "\U0001f7e1",
            "expected": "\U0001f7e0",
            "danger": "\U0001f534",
            "critical": "\U0001f6a8",
        }
        emoji = level_emoji.get(s["level"], "\u26aa")
        return f"{emoji} MDD: {s['current_mdd_pct']:.1f}% (백테스트: {s['backtest_mdd']}%)"
