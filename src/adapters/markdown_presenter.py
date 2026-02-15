"""마크다운 프레젠터 - AnalysisReport를 마크다운 문자열로 변환"""

from __future__ import annotations

from src.entities.models import AnalysisReport, ConditionType
from src.use_cases.ports import ReportPresenterPort


class MarkdownPresenter(ReportPresenterPort):
    """📋 마크다운 리포트 프레젠터 - ReportPresenterPort 구현"""

    def present(self, report: AnalysisReport) -> str:
        sections = [
            self._header(report),
            self._stock_info(report),
            self._technical_analysis(report),
            self._volume_analysis(report),
            self._investor_flow(report),
            self._key_points(report),
            self._supply_demand(report),
            self._score_section(report),
            self._flow_prediction(report),
            self._conditions(report),
            self._footer(report),
        ]
        return "\n\n".join(sections)

    @staticmethod
    def _header(r: AnalysisReport) -> str:
        return f"# 📊 {r.stock.name} ({r.stock.ticker}) 종합 분석 리포트"

    @staticmethod
    def _stock_info(r: AnalysisReport) -> str:
        latest = r.chart_data.latest
        price = f"{latest.close:,.0f}원" if latest else "N/A"
        return (
            "## 1️⃣ 업체 정보\n"
            f"| 항목 | 내용 |\n|------|------|\n"
            f"| 종목코드 | {r.stock.ticker} |\n"
            f"| 종목명 | {r.stock.name} |\n"
            f"| 시장 | {r.stock.market.value} |\n"
            f"| 업종 | {r.stock.sector} |\n"
            f"| 현재가 | {price} |"
        )

    @staticmethod
    def _technical_analysis(r: AnalysisReport) -> str:
        p = r.technical_pattern
        ind = r.chart_data.indicators
        return (
            "## 2️⃣ 기술적 형태/지표\n"
            f"| 지표 | 값 | 해석 |\n|------|-----|------|\n"
            f"| 캔들 패턴 | - | {p.candle_pattern} |\n"
            f"| 이평선 배열 | {p.ma_alignment.value} | 5/20/60/120일 |\n"
            f"| RSI | {f'{ind.rsi:.1f}' if ind.rsi else 'N/A'} | {p.rsi_signal} |\n"
            f"| MACD | {ind.macd or 'N/A'} | {p.macd_signal} |\n"
            f"| 볼린저밴드 | - | {p.bollinger_signal} |\n"
            f"| 스토캐스틱 | - | {p.stochastic_signal} |\n"
            f"| **종합 추세** | **{p.overall_trend.value}** | **강도: {p.strength.value}** |"
        )

    @staticmethod
    def _volume_analysis(r: AnalysisReport) -> str:
        v = r.volume_analysis
        return (
            "## 3️⃣ 거래량 분석\n"
            f"| 항목 | 내용 |\n|------|------|\n"
            f"| 평균 거래량 대비 | {v.avg_volume_ratio:.1f}배 |\n"
            f"| 거래량 추세 | {v.volume_trend.value} |\n"
            f"| 매집/분산 | {v.accumulation_signal} |"
        )

    @staticmethod
    def _key_points(r: AnalysisReport) -> str:
        points = r.technical_pattern.key_points + r.volume_analysis.key_points
        if not points:
            return "## 4️⃣ 핵심 분석 포인트\n- 분석 포인트 없음"
        items = "\n".join(f"- {p}" for p in points)
        return f"## 4️⃣ 핵심 분석 포인트\n{items}"

    @staticmethod
    def _supply_demand(r: AnalysisReport) -> str:
        zones = r.volume_analysis.zones
        if not zones:
            return "## 5️⃣ 매물대 분석\n- 주요 매물대 없음"
        rows = "\n".join(
            f"| {z.zone_type} | {z.price_low:,.0f} ~ {z.price_high:,.0f} | {z.strength.value} | {z.description} |"
            for z in zones
        )
        return (
            "## 5️⃣ 매물대 분석\n"
            f"| 구분 | 가격 구간 | 강도 | 설명 |\n|------|----------|------|------|\n{rows}"
        )

    @staticmethod
    def _investor_flow(r: AnalysisReport) -> str:
        flow = r.investor_flow
        if not flow:
            return "## 6️⃣ 수급 동향\n- 수급 데이터 없음"
        return (
            "## 6️⃣ 수급 동향\n"
            f"| 투자자 | 순매수 |\n|--------|--------|\n"
            f"| 외국인 | {flow.foreign_net:,}주 |\n"
            f"| 기관 | {flow.inst_net:,}주 |\n"
            f"| 개인 | {flow.individual_net:,}주 |"
        )

    @staticmethod
    def _score_section(r: AnalysisReport) -> str:
        score = r.score
        if not score:
            return ""
        lines = [
            f"## AI 종합 스코어: {score.total_score:.0f}/{score.max_score:.0f}점 ({score.grade})\n",
        ]
        for cat in score.categories:
            lines.append(f"### {cat.name} ({cat.score:.0f}/{cat.max_score:.0f}점)")
            for d in cat.details:
                bar_len = int(d.score / d.max_score * 10) if d.max_score > 0 else 0
                bar = "=" * bar_len + "-" * (10 - bar_len)
                lines.append(f"- {d.name}: [{bar}] {d.score:.0f}/{d.max_score:.0f} — {d.comment}")
            lines.append("")

        if score.summary:
            lines.append(f"> **요약**: {score.summary}")
        if score.recommendation:
            lines.append(f"> **권고**: {score.recommendation}")

        return "\n".join(lines)

    @staticmethod
    def _flow_prediction(r: AnalysisReport) -> str:
        f = r.flow_prediction
        factors = "\n".join(f"- {kf}" for kf in f.key_factors)
        return (
            "## 7️⃣ 🔮 내일 흐름 예측\n"
            f"| 항목 | 내용 |\n|------|------|\n"
            f"| 예상 방향 | **{f.direction.value}** |\n"
            f"| 확신도 | {f.confidence:.0%} |\n"
            f"| 예상 가격 범위 | {f.price_low:,.0f} ~ {f.price_high:,.0f} |\n\n"
            f"**핵심 요인:**\n{factors}\n\n"
            f"> {f.summary}"
        )

    @staticmethod
    def _conditions(r: AnalysisReport) -> str:
        hold = [c for c in r.conditions if c.condition_type == ConditionType.HOLD]
        action = [c for c in r.conditions if c.condition_type == ConditionType.ACTION]

        hold_items = "\n".join(
            f"- **{c.title}** (확신도 {c.confidence:.0%}): {c.description}"
            for c in sorted(hold, key=lambda x: x.priority)
        ) or "- 없음"

        action_items = "\n".join(
            f"- **{c.title}** (확신도 {c.confidence:.0%})"
            f"{f' → 트리거: {c.trigger_price:,.0f}원' if c.trigger_price else ''}: {c.description}"
            for c in sorted(action, key=lambda x: x.priority)
        ) or "- 없음"

        return (
            "## 8️⃣ 📋 유지/대응 조건\n\n"
            f"### ✅ 유지해도 되는 조건\n{hold_items}\n\n"
            f"### 🚨 반드시 대응해야 할 조건\n{action_items}"
        )

    @staticmethod
    def _footer(r: AnalysisReport) -> str:
        return (
            "---\n"
            f"*분석 시각: {r.analyzed_at.strftime('%Y-%m-%d %H:%M')}*\n"
            "*본 분석은 AI 기반 참고 자료이며, 투자 판단의 책임은 투자자 본인에게 있습니다.*"
        )
