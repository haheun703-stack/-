"""
6-Layer Signal Pipeline 진단 시스템

v3.0 신규:
  각 레이어별 통과/차단 추적 → 파이프라인 병목 진단
  grade_blocked 추적 (Claude 채팅에서 발견한 버그 수정)

레이어 구성:
  L0_pre_gate   → Pre-screening (매출/거래대금/수익성)
  L0_grade      → Zone Score + Grade (A/B/C/F)
  L1_regime     → HMM 레짐 (Accumulation만 통과)
  L2_ou         → OU 필터 (z-score, half-life, SNR)
  L3_momentum   → 모멘텀 (거래량 서지 + MA60 slope)
  L4_smart_money→ Smart Money Z-score
  L5_risk       → 손익비 + 포트폴리오 한도
  L6_trigger    → Impulse/Confirm/Breakout
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


LAYER_NAMES = [
    "L-1_news_gate",   # v3.1 이벤트 드리븐 레이어
    "L0_pre_gate",
    "L0_grade",
    "L1_regime",
    "L2_ou",
    "L3_momentum",
    "L4_smart_money",
    "L5_risk",
    "L6_trigger",
]


@dataclass
class LayerResult:
    """단일 레이어 판정 결과"""
    name: str
    passed: bool
    block_reason: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class DiagnosticRecord:
    """하루 1종목의 전체 레이어 진단"""
    date: str
    ticker: str
    layers: list[LayerResult] = field(default_factory=list)
    final_signal: bool = False
    blocked_at: str = ""  # 첫 차단 레이어

    def add_layer(self, result: LayerResult):
        self.layers.append(result)
        if not result.passed and not self.blocked_at:
            self.blocked_at = result.name
            self.final_signal = False

    @property
    def passed_all(self) -> bool:
        return all(lr.passed for lr in self.layers)


class SignalDiagnostic:
    """6-Layer 파이프라인 진단기"""

    def __init__(self):
        self.records: list[DiagnosticRecord] = []

    def new_record(self, date: str, ticker: str) -> DiagnosticRecord:
        rec = DiagnosticRecord(date=date, ticker=ticker)
        self.records.append(rec)
        return rec

    def summarize(self) -> dict:
        """레이어별 통과율 집계"""
        if not self.records:
            return {}

        total = len(self.records)
        layer_stats = {}

        for name in LAYER_NAMES:
            reached = 0
            passed = 0
            block_reasons = {}

            for rec in self.records:
                for lr in rec.layers:
                    if lr.name == name:
                        reached += 1
                        if lr.passed:
                            passed += 1
                        elif lr.block_reason:
                            block_reasons[lr.block_reason] = (
                                block_reasons.get(lr.block_reason, 0) + 1
                            )
                        break

            layer_stats[name] = {
                "reached": reached,
                "passed": passed,
                "pass_rate": round(passed / reached * 100, 1) if reached > 0 else 0,
                "block_reasons": block_reasons,
            }

        # 최종 시그널
        final_signals = sum(1 for r in self.records if r.final_signal)

        return {
            "total_evaluations": total,
            "final_signals": final_signals,
            "signal_rate": round(final_signals / total * 100, 2) if total > 0 else 0,
            "layers": layer_stats,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """진단 레코드를 DataFrame으로 변환"""
        rows = []
        for rec in self.records:
            row = {
                "date": rec.date,
                "ticker": rec.ticker,
                "final_signal": rec.final_signal,
                "blocked_at": rec.blocked_at,
            }
            for lr in rec.layers:
                row[f"{lr.name}_passed"] = lr.passed
                row[f"{lr.name}_reason"] = lr.block_reason
            rows.append(row)
        return pd.DataFrame(rows)

    def print_summary(self):
        """콘솔에 진단 요약 출력"""
        summary = self.summarize()
        if not summary:
            logger.info("진단 데이터 없음")
            return

        logger.info(f"\n{'='*60}")
        logger.info("6-Layer Pipeline 진단 리포트")
        logger.info(f"{'='*60}")
        logger.info(f"총 평가: {summary['total_evaluations']}건")
        logger.info(f"최종 시그널: {summary['final_signals']}건 "
                     f"({summary['signal_rate']}%)")
        logger.info(f"{'-'*60}")

        for name in LAYER_NAMES:
            stats = summary["layers"].get(name, {})
            reached = stats.get("reached", 0)
            passed = stats.get("passed", 0)
            rate = stats.get("pass_rate", 0)
            reasons = stats.get("block_reasons", {})

            bar = "#" * int(rate / 5) + "." * (20 - int(rate / 5))
            logger.info(f"  {name:16s} [{bar}] {rate:5.1f}%  "
                         f"({passed}/{reached})")
            if reasons:
                top_reasons = sorted(reasons.items(), key=lambda x: -x[1])[:3]
                reason_str = ", ".join(f"{r}({c})" for r, c in top_reasons)
                logger.info(f"    차단: {reason_str}")

        logger.info(f"{'='*60}")
