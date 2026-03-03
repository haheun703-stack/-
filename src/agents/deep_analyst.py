"""v3 Agent 2D — 개별 종목 정밀 분석 (Deep Analyst)

Phase 2에서 선별된 20~30개 후보 종목을 개별 심층 분석.
기존 chart_analysis + volume_analysis + game_analyst의 핵심을 통합.

입력:
  - 후보 종목 리스트 (scan_cache.json에서 추출)
  - ai_strategic_analysis.json의 industry_thesis (산업 맥락)
  - ai_sector_focus.json의 focus_sectors (섹터 맥락)

출력:
  - 종목별 {buy: bool, conviction: 1~10, strategy, thesis_alignment, reason, risk}
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_DEEP_ANALYSIS = """\
당신은 한국 주식시장 종목 정밀 분석 전문가입니다.
상위 전략가의 산업 구조 분석(industry thesis)과 개별 종목의 기술적 데이터를 결합하여
매수 여부와 확신도를 판단합니다.

## 판단 프레임워크

1. **Thesis 정합성** (40%)
   - 이 종목이 상위 전략가의 산업 thesis에 부합하는가?
   - 해당 섹터가 attack 목록에 있는가?
   - 릴레이 경로의 수혜 종목인가?

2. **기술적 타이밍** (30%)
   - 추세: MA60 > MA120 (중기 정배열), ADX > 14
   - 에너지: RSI 38~55 구간 (과매수 아닌 초기 반등)
   - 수급: 외국인/기관 순매수 방향
   - SAR: 상승 반전(sar_trend=1) 확인

3. **수급/세력** (20%)
   - 거래량 변화 (vol_surge): 0.5 이상이면 관심
   - OBV 추세: up이면 매집 가능성
   - 외인/기관 5일 누적: 양전환이면 유리

4. **리스크** (10%)
   - 52주 고점 대비 위치 (0.95 이상이면 과열)
   - 변동성 (drawdown_from_high)
   - 섹터 avoid 여부

## 확신도 기준 (conviction 1~10)
- 9~10: 산업 thesis 강한 부합 + 기술적 완벽 + 수급 양호 → 풀 사이즈
- 7~8: thesis 부합 + 기술적 양호 → 표준 사이즈
- 5~6: 일부 부합 → 축소 사이즈
- 1~4: 부합 미흡 → 매수 불가

## 출력 형식
반드시 아래 JSON 형식으로만 응답하세요.

```json
{
  "ticker": "종목코드",
  "name": "종목명",
  "buy": true,
  "conviction": 1~10,
  "strategy": "momentum|pullback|relay|event|turnaround",
  "thesis_alignment": "strong|moderate|weak|none",
  "entry_price": 현재가,
  "target_pct": 예상 상승률,
  "stop_loss_pct": 손절 기준,
  "reasoning": "판단 근거 2~3문장",
  "risks": ["리스크1", "리스크2"],
  "catalysts": ["촉매1"]
}
```
"""


class DeepAnalystAgent(BaseAgent):
    """v3 Agent 2D — 개별 종목 정밀 분석 (Sonnet 배치)"""

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

    async def analyze_single(
        self,
        stock_context: dict,
        industry_thesis: list[dict],
        sector_focus: dict,
        chart_image: str | None = None,
    ) -> dict:
        """단일 종목 정밀 분석.

        Args:
            stock_context: scan_cache의 개별 종목 데이터
            industry_thesis: Phase 1의 industry_thesis 리스트
            sector_focus: Phase 2의 ai_sector_focus.json
            chart_image: base64 PNG 차트 이미지 (Vision 사용 시)

        Returns:
            {buy, conviction, strategy, thesis_alignment, reasoning, ...}
        """
        stock_text = self._format_stock(stock_context)
        thesis_text = self._format_thesis_context(industry_thesis, sector_focus)

        user_prompt = f"""\
{thesis_text}

{stock_text}

위 종목을 산업 thesis 관점에서 정밀 분석하여 매수 판단을 JSON으로 응답하세요.
"""
        if chart_image:
            user_prompt += "\n첨부된 차트 이미지도 참고하여 기술적 패턴을 분석하세요."

        try:
            if chart_image:
                result = await self._ask_claude_vision_json(
                    SYSTEM_DEEP_ANALYSIS, user_prompt, [chart_image]
                )
            else:
                result = await self._ask_claude_json(SYSTEM_DEEP_ANALYSIS, user_prompt)
        except Exception as e:
            logger.warning(
                "종목 분석 실패 %s(%s): %s",
                stock_context.get("name", "?"),
                stock_context.get("ticker", "?"),
                e,
            )
            return {
                "ticker": stock_context.get("ticker", ""),
                "name": stock_context.get("name", ""),
                "buy": False,
                "conviction": 0,
                "error": str(e),
            }

        # 필수 필드 보정
        result.setdefault("ticker", stock_context.get("ticker", ""))
        result.setdefault("name", stock_context.get("name", ""))
        result.setdefault("buy", False)
        result.setdefault("conviction", 0)

        # conviction 클램핑
        result["conviction"] = max(0, min(10, result.get("conviction", 0)))

        return result

    async def analyze_batch(
        self,
        candidates: list[dict],
        industry_thesis: list[dict],
        sector_focus: dict,
        min_conviction: int = 5,
        max_concurrent: int = 3,
        enable_vision: bool = False,
    ) -> list[dict]:
        """후보 종목 배치 분석 — conviction >= min_conviction만 통과.

        Args:
            candidates: 후보 종목 리스트
            industry_thesis: Phase 1 산업 thesis
            sector_focus: Phase 2 섹터 포커스
            min_conviction: 최소 conviction 기준
            max_concurrent: 동시 분석 수 (API rate limit 고려)
            enable_vision: True이면 차트 이미지 첨부 (Vision API)

        Returns:
            conviction >= min_conviction인 종목만 필터링된 리스트
        """
        logger.info(
            "Deep Analyst 배치 분석 시작: %d종목, min_conviction=%d, vision=%s",
            len(candidates), min_conviction, enable_vision,
        )

        # Vision 활성화 시 차트 렌더링
        chart_images: dict[str, str | None] = {}
        if enable_vision:
            try:
                from src.chart_renderer import render_batch
                tickers = [c.get("ticker", "") for c in candidates if c.get("ticker")]
                chart_images = render_batch(tickers)
                rendered = sum(1 for v in chart_images.values() if v)
                logger.info("차트 렌더링: %d/%d 성공", rendered, len(tickers))
            except Exception as e:
                logger.warning("차트 렌더링 실패 (텍스트만 분석): %s", e)

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _analyze_with_limit(stock):
            async with semaphore:
                img = chart_images.get(stock.get("ticker", "")) if enable_vision else None
                return await self.analyze_single(
                    stock, industry_thesis, sector_focus, chart_image=img
                )

        tasks = [_analyze_with_limit(c) for c in candidates]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in raw_results:
            if isinstance(r, Exception):
                logger.warning("배치 분석 예외: %s", r)
                continue
            if isinstance(r, dict):
                results.append(r)

        # conviction 필터
        passed = [r for r in results if r.get("conviction", 0) >= min_conviction and r.get("buy")]
        failed = len(results) - len(passed)

        logger.info(
            "Deep Analyst 완료: %d분석 → %d통과 (conviction>=%d), %d탈락",
            len(results), len(passed), min_conviction, failed,
        )

        # conviction 내림차순 정렬
        passed.sort(key=lambda x: x.get("conviction", 0), reverse=True)

        return passed

    @staticmethod
    def _format_stock(data: dict) -> str:
        """개별 종목 데이터 → 텍스트"""
        lines = [f"[종목 분석 대상]"]
        ticker = data.get("ticker", "?")
        name = data.get("name", "?")
        grade = data.get("grade", "?")
        zone_score = data.get("zone_score", 0)

        lines.append(f"종목: {name} ({ticker})")
        lines.append(f"등급: {grade}, Zone Score: {zone_score:.4f}")

        # 기술적 지표
        lines.append(f"\n[기술적 지표]")
        lines.append(f"  RSI: {data.get('rsi', 0):.1f}")
        lines.append(f"  ADX: {data.get('adx', 0):.1f} (+DI: {data.get('plus_di', 0):.1f}, -DI: {data.get('minus_di', 0):.1f})")
        lines.append(f"  TRIX: {data.get('trix', 0):.4f} (Signal: {data.get('trix_signal', 0):.4f})")
        lines.append(f"  MA60 위: {'O' if data.get('above_ma60') else 'X'}, MA120 위: {'O' if data.get('above_ma120') else 'X'}")
        lines.append(f"  MA60 기울기: {data.get('slope_ma60', 0):.2f}%")
        lines.append(f"  52주 고점 대비: {data.get('pct_of_52w_high', 0):.1%}")
        lines.append(f"  고점 대비 하락: {data.get('drawdown_from_high', 0):.1f}%")

        # 수급
        lines.append(f"\n[수급]")
        lines.append(f"  거래량 서지: {data.get('vol_surge', 0):.2f}x")
        lines.append(f"  거래량 수축: {data.get('vol_contraction', 0):.2f}")
        lines.append(f"  OBV 추세: {data.get('obv_trend', '?')}")
        lines.append(f"  외국인 연속: {data.get('foreign_streak', 0)}일, 기관 연속: {data.get('inst_streak', 0)}일")
        lines.append(f"  외국인 5일 순매수: {data.get('foreign_net_5d', 0):,.0f}")

        # 가격
        lines.append(f"\n[가격]")
        lines.append(f"  진입가: {data.get('entry_price', 0):,}원")
        lines.append(f"  손절가: {data.get('stop_loss', 0):,}원")
        lines.append(f"  목표가: {data.get('target_price', 0):,}원")
        lines.append(f"  손익비: {data.get('risk_reward', 0):.2f}")
        lines.append(f"  트리거: {data.get('trigger_type', '?')}")

        # 밸류에이션
        per = data.get("per", 0)
        pbr = data.get("pbr", 0)
        if per or pbr:
            lines.append(f"\n[밸류에이션]")
            if per:
                lines.append(f"  PER: {per:.1f}x")
            if pbr:
                lines.append(f"  PBR: {pbr:.2f}x")

        # 점수 상세
        scores = data.get("scores", {})
        if scores:
            lines.append(f"\n[점수 상세]")
            for key, val in scores.items():
                if isinstance(val, (int, float)):
                    lines.append(f"  {key}: {val}")

        return "\n".join(lines)

    @staticmethod
    def _format_thesis_context(theses: list[dict], sector_focus: dict) -> str:
        """산업 thesis + 섹터 포커스 → 맥락 텍스트"""
        lines = ["[산업 구조 맥락]"]

        if theses:
            for t in theses[:5]:
                lines.append(
                    f"  {t.get('sector', '?')} (확신 {t.get('confidence', 0)}/10): "
                    f"{t.get('thesis', '?')}"
                )

        boost = sector_focus.get("screening_boost", [])
        suppress = sector_focus.get("screening_suppress", [])
        if boost:
            lines.append(f"\n부스트 섹터: {', '.join(boost)}")
        if suppress:
            lines.append(f"억제 섹터: {', '.join(suppress)}")

        focus = sector_focus.get("focus_sectors", [])
        if focus:
            lines.append(f"\n[집중 섹터]")
            for s in focus[:5]:
                lines.append(
                    f"  {s.get('sector', '?')}: "
                    f"타이밍={s.get('entry_timing', '?')}, "
                    f"정합={s.get('thesis_alignment', '?')}, "
                    f"비중={s.get('size_weight', 1.0):.1f}x"
                )

        return "\n".join(lines)
