"""v3 Agent 2B — 섹터 집중 + 릴레이 경로 매핑 (Sector Strategist)

Phase 1 StrategicBrain의 산업 thesis를 받아서:
  - 실제 ETF 모멘텀 데이터와 교차 검증
  - 스크리닝 boost/suppress 목록 생성
  - 릴레이 타이밍 판단

입력:
  - ai_strategic_analysis.json (Phase 1 결과)
  - etf_trading_signal.json (20개 섹터 ETF 모멘텀)
  - relay_pattern_db.json

출력:
  - ai_sector_focus.json
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_SECTOR_FOCUS = """\
당신은 한국 주식시장 섹터 전략가입니다.
상위 전략가(Agent 2A)의 산업 구조 분석과 실제 ETF 모멘텀 데이터를 교차 검증하여,
오늘 집중할 섹터와 스크리닝 부스트/억제 목록을 생성합니다.

## 역할

1. **교차 검증**: AI thesis와 실제 모멘텀 데이터가 일치하는지 확인
   - thesis "공격" + 모멘텀 상위 = 강한 확신 → boost 대상
   - thesis "공격" + 모멘텀 하위 = 아직 초기 or 빗나감 → watch
   - thesis "회피" + 모멘텀 상위 = 과열 경고 → suppress 대상
   - thesis 없음 + 모멘텀 상위 = 구조적 근거 부족 → watch

2. **릴레이 타이밍**: 소스 섹터의 현재 상태를 보고 타겟 진입 타이밍 판단
   - 소스 급등 중 → 타겟 "매수 준비" (1~2일 후 진입)
   - 소스 이미 고점 → 타겟 "즉시 진입" (릴레이 시작)
   - 소스 하락 전환 → 타겟 "보류" (릴레이 무효)

3. **진입 타이밍 판단**: 각 섹터의 기술적 상태로 진입 시점 결정
   - RSI < 70 + ADX 상승 = "즉시 진입"
   - RSI > 75 + BB > 95% = "과열, 눌림 대기"
   - ADX < 20 + 모멘텀 하위 = "관망"

## 출력 형식
반드시 아래 JSON 형식으로만 응답하세요.

```json
{
  "analysis_date": "YYYY-MM-DD",
  "focus_sectors": [
    {
      "sector": "섹터명",
      "entry_timing": "즉시진입|눌림대기|관망",
      "size_weight": 1.0,
      "thesis_alignment": "강한일치|약한일치|불일치|thesis없음",
      "momentum_rank": 1,
      "momentum_score": 91.3,
      "rsi": 80.5,
      "smart_money": true,
      "relay_source": "소스 섹터 (해당 시)",
      "reasoning": "판단 근거 1문장"
    }
  ],
  "screening_boost": ["부스트할 섹터명"],
  "screening_suppress": ["억제할 섹터명"],
  "sector_warnings": [
    {
      "sector": "섹터명",
      "warning": "경고 내용",
      "severity": "high|medium|low"
    }
  ]
}
```
"""


class SectorStrategistAgent(BaseAgent):
    """v3 Agent 2B — 섹터 집중 + 릴레이 경로 매핑"""

    def __init__(self, model: str | None = None):
        if model is None:
            model = self._load_model_from_settings()
        super().__init__(model=model)

    @staticmethod
    def _load_model_from_settings() -> str:
        """settings.yaml에서 tactical_model 로드"""
        try:
            import yaml
            settings_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
            with open(settings_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("ai_brain_v3", {}).get("tactical_model", "claude-sonnet-4-5-20250929")
        except Exception:
            return "claude-sonnet-4-5-20250929"

    async def focus(
        self,
        strategic_analysis: dict,
        sector_flow: dict,
        relay_db: dict,
    ) -> dict:
        """섹터 집중 분석 — thesis × 모멘텀 교차 검증.

        Args:
            strategic_analysis: ai_strategic_analysis.json (Phase 1 결과)
            sector_flow: etf_trading_signal.json (20개 섹터 ETF)
            relay_db: relay_pattern_db.json

        Returns:
            ai_sector_focus.json 형식의 dict
        """
        # 데이터 포매팅
        thesis_text = self._format_thesis(strategic_analysis)
        etf_text = self._format_etf_data(sector_flow)
        relay_text = self._format_relay(relay_db)

        today = datetime.now().strftime("%Y-%m-%d")
        user_prompt = f"""\
## 분석 날짜: {today}

{thesis_text}

{etf_text}

{relay_text}

위 데이터를 교차 검증하여 오늘 집중할 섹터와 스크리닝 부스트/억제 목록을 JSON으로 응답하세요.
thesis와 실제 모멘텀이 일치하는 섹터를 우선 배치하세요.
과열(RSI>75, BB>95%) 섹터는 suppress에 넣으세요.
"""

        logger.info("v3 Sector Strategist 분석 시작")

        try:
            result = await self._ask_claude_json(SYSTEM_SECTOR_FOCUS, user_prompt)
        except Exception as e:
            logger.error("v3 Sector Strategist 분석 실패: %s", e)
            return self._fallback_result(str(e))

        result.setdefault("analysis_date", today)
        result = self._validate_result(result)

        logger.info(
            "v3 Sector Strategist 완료: focus=%d, boost=%s, suppress=%s",
            len(result.get("focus_sectors", [])),
            result.get("screening_boost", []),
            result.get("screening_suppress", []),
        )
        return result

    @staticmethod
    def _format_thesis(data: dict) -> str:
        """Phase 1 결과 → 텍스트"""
        lines = ["[Phase 1 전략 두뇌 판단]"]
        lines.append(f"레짐: {data.get('regime', '?')} (신뢰도: {data.get('regime_confidence', 0):.0%})")
        lines.append(f"최대 매수: {data.get('max_new_buys', 0)}종목")

        priority = data.get("sector_priority", {})
        lines.append(f"공격 섹터: {', '.join(priority.get('attack', []))}")
        lines.append(f"관찰 섹터: {', '.join(priority.get('watch', []))}")
        lines.append(f"회피 섹터: {', '.join(priority.get('avoid', []))}")

        theses = data.get("industry_thesis", [])
        if theses:
            lines.append(f"\n[산업 Thesis ({len(theses)}개)]")
            for t in theses:
                lines.append(
                    f"  {t.get('sector', '?')} (확신 {t.get('confidence', 0)}/10): "
                    f"{t.get('thesis', '?')}"
                )
                lines.append(
                    f"    수급={t.get('demand_supply', '?')}, ASP={t.get('asp_trend', '?')}, "
                    f"글로벌={t.get('global_alignment', '?')}"
                )

        alerts = data.get("relay_alerts", [])
        if alerts:
            lines.append(f"\n[릴레이 알림]")
            for a in alerts:
                lines.append(f"  {a.get('pattern', '?')}: {a.get('status', '?')} → {a.get('action', '?')}")

        return "\n".join(lines)

    @staticmethod
    def _format_etf_data(data: dict) -> str:
        """ETF 모멘텀 → 텍스트 (all_etf 기준 상위 10개)"""
        lines = ["[ETF 모멘텀 데이터 (20개 섹터)]"]

        all_etf = data.get("all_etf", [])
        if not all_etf:
            return "[ETF 모멘텀] 데이터 없음"

        # 모멘텀 순위별 정렬
        sorted_etf = sorted(all_etf, key=lambda x: x.get("momentum_rank", 99))

        lines.append(
            f"{'순위':>4} {'섹터':<12} {'점수':>6} {'RSI':>6} {'BB%':>6} "
            f"{'ADX':>5} {'ADX↑':>4} {'외인5d':>10} {'기관5d':>10} {'SM':>3}"
        )
        lines.append("-" * 80)

        for etf in sorted_etf:
            rank = etf.get("momentum_rank", "?")
            sector = etf.get("sector", "?")
            score = etf.get("momentum_score", 0)
            rsi = etf.get("rsi", 0)
            bb = etf.get("bb_pct", 0)
            adx = etf.get("adx", 0)
            adx_up = "Y" if etf.get("adx_rising") else "N"
            foreign = etf.get("foreign_5d_bil", 0)
            inst = etf.get("inst_5d_bil", 0)
            smart = "O" if etf.get("is_smart") else "X"

            lines.append(
                f"{rank:>4} {sector:<12} {score:>6.1f} {rsi:>6.1f} {bb:>6.1f} "
                f"{adx:>5.1f} {adx_up:>4} {foreign:>+10.0f} {inst:>+10.0f} {smart:>3}"
            )

        # 시그널 요약
        smart_buy = data.get("smart_money_etf", [])
        theme_buy = data.get("theme_money_etf", [])
        watch = data.get("watch_list", [])

        if smart_buy:
            lines.append(f"\n[Smart Money BUY] {', '.join(e.get('sector', '?') for e in smart_buy)}")
        if theme_buy:
            lines.append(f"[Theme BUY] {', '.join(e.get('sector', '?') for e in theme_buy)}")
        if watch:
            lines.append(f"[Watch] {', '.join(e.get('sector', '?') for e in watch)}")

        return "\n".join(lines)

    @staticmethod
    def _format_relay(data: dict) -> str:
        """릴레이 DB → 텍스트 (active만)"""
        lines = ["[릴레이 패턴 DB]"]
        patterns = data.get("patterns", [])
        active = [p for p in patterns if p.get("status") == "active"]
        for p in active:
            lines.append(
                f"  {p.get('source_sector')} → {p.get('target_sector')}: "
                f"승률 {p.get('win_rate', 0):.1f}%, 지연 {p.get('avg_lag_days', 0)}일"
            )
        return "\n".join(lines)

    @staticmethod
    def _validate_result(result: dict) -> dict:
        """결과 검증"""
        result.setdefault("focus_sectors", [])
        result.setdefault("screening_boost", [])
        result.setdefault("screening_suppress", [])
        result.setdefault("sector_warnings", [])

        # size_weight 클램핑
        for s in result.get("focus_sectors", []):
            w = s.get("size_weight", 1.0)
            s["size_weight"] = max(0.3, min(1.5, w))

        return result

    @staticmethod
    def _fallback_result(error_msg: str) -> dict:
        """AI 장애 시 빈 결과"""
        return {
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "focus_sectors": [],
            "screening_boost": [],
            "screening_suppress": [],
            "sector_warnings": [],
            "error": error_msg,
        }
