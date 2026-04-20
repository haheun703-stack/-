"""Master Brain — 전체 시스템 통합 두뇌

30개+ 데이터 소스를 종합하여 추론 체인(cause→impact→action)을 생성하고
개별종목 + ETF + 레버리지/인버스 + 원자재 + 현금 비중까지 통합 추천.

입력: MacroAggregator가 수집한 전체 시그널 데이터
출력: data/master_brain_judgment.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.agents.base import BaseAgent
from src.macro.macro_aggregator import MacroAggregator

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = _PROJECT_ROOT / "data" / "master_brain_judgment.json"

# ─── 시스템 프롬프트 ───────────────────────────────────────────────

SYSTEM_MASTER_BRAIN = """\
당신은 Quantum Master 시스템의 최상위 두뇌입니다.
30개+ 데이터 소스를 종합하여 "오늘의 시장 판단"과 "구체적 행동 추천"을 합니다.

## 추론 체인 규칙 (CRITICAL)
반드시 macro_cause → sector_impact → specific_action 순서로 논리를 전개하세요.
예시: "미국 CPI 4.5% 상승 → 스태그플레이션 우려 확대 → 방어주/금 ETF 비중 확대 → KODEX 골드선물 매수"

각 추론 체인에는 반드시:
1. 원인 (cause): 매크로/거시 이벤트
2. 영향 (sector_impact): 어떤 섹터/자산에 어떻게 영향
3. 행동 (action): 구체적 매매 행동 (종목명/ETF명 포함)
4. 신뢰도 (confidence): 0.0~1.0
5. 시간프레임 (timeframe): "단기(1주)", "중기(1~3개월)", "장기(3~6개월)"

## 출력 자산 클래스 (모두 포함 필수)
1. 개별종목 (individual_stocks): 대장주 + 소부장 — 시스템 추천 종목 중 최종 선별
2. 섹터 ETF (sector_etf): KODEX/TIGER 등 20개 섹터 ETF 중 매매 시그널 있는 것
3. 레버리지/인버스 ETF (leverage_inverse): 레짐에 따른 공격/방어 도구
4. 원자재 ETF (commodity_etf): 금/은/원유/천연가스/구리/우라늄 — US ETF 기준
5. 인덱스 ETF (index_etf): KOSPI200/KOSDAQ150 등 시장 전체 ETF
6. 현금 비중 (cash_strategy): 25% 최소 유지 원칙

## 판단 기준 (우선순위)
1순위: 거시 레짐 (KOSPI 레짐 + US Overnight 등급 + CPI/스태그플레이션)
2순위: 산업 구조 (수요/공급/CAPA → 섹터 thesis)
3순위: 수급 (기관/외국인 동향, 원자재 흐름, 고래 감지)
4순위: 기술적 확인 (차트/시그널은 마지막 보조 수단)

## 현금 관리 원칙 (서보성 원칙)
- 최소 25% 현금 항상 유지 (3~4년 주기 폭락 대비)
- CRISIS 레짐: 65% 현금
- BEAR 레짐: 40% 현금
- CAUTION 레짐: 30% 현금
- BULL 레짐: 25% 현금 (최저선)

## 특별 지시
- 천연가스(UNG): 데이터센터 전력 수요 급증 → 채굴원가 $2.8~2.9 수준이면 바닥 가능
- 원유: 호르무즈 해협 리스크 시 유가 $130+ 시나리오 고려
- 금/은/구리: 인플레이션 헤지 + 경기 사이클 지표로 활용
- 방산: 수주잔고 역대급이나, "실탄 부족" → 실적 확인 필요
- 현대차: 유가 상승 → 운송비 증가 → 소비 위축 → 부정적

## 출력 형식 (JSON)
```json
{
    "date": "YYYY-MM-DD",
    "market_regime": {
        "overall": "BULL|CAUTION|BEAR|CRISIS",
        "kospi_regime": "...",
        "us_overnight": "STRONG_BULL|MILD_BULL|NEUTRAL|MILD_BEAR|STRONG_BEAR",
        "stagflation": "NONE|WARNING|ALERT",
        "summary": "시장 상황 한줄 요약"
    },
    "reasoning_chains": [
        {
            "chain_id": 1,
            "cause": "원인",
            "sector_impact": "섹터 영향",
            "action": "구체적 행동",
            "confidence": 0.75,
            "timeframe": "1~3개월"
        }
    ],
    "unified_picks": {
        "individual_stocks": [
            {"ticker": "006코드", "name": "종목명", "action": "BUY|HOLD|SELL|WATCH",
             "reason": "이유", "chain_ref": 1}
        ],
        "sector_etf": [...],
        "leverage_inverse": [...],
        "commodity_etf": [...],
        "index_etf": [...]
    },
    "cash_strategy": {
        "target_cash_pct": 30,
        "current_cash_pct": 25,
        "action": "현금 비중 조정 액션",
        "reason": "이유"
    },
    "risk_alerts": ["리스크 경고 1", "리스크 경고 2"],
    "daily_summary": "시장 한줄평 (텔레그램용)"
}
```

반드시 위 JSON 형식으로만 응답하세요. 추가 설명 없이 JSON만 출력하세요.
"""


class MasterBrainAgent(BaseAgent):
    """전체 시스템 통합 두뇌 — Sonnet + Opus Advisor 패턴."""

    ADVISOR_INSTRUCTION = (
        "전략 판단을 내리기 전에 advisor에게 논리적 허점과 "
        "리스크 시나리오를 100단어 이내로 검증받으세요."
    )

    def __init__(self, model: str | None = None):
        from src.agents.base import MODEL_SONNET
        super().__init__(model=model or MODEL_SONNET)
        self.aggregator = MacroAggregator()

    async def think(self) -> dict:
        """전체 데이터 수집 → Sonnet+Opus Advisor 추론 → 통합 판단."""
        logger.info("Master Brain: 데이터 수집 시작...")
        data = self.aggregator.aggregate()
        prompt_text = self.aggregator.summarize_for_prompt(data)

        logger.info(f"Master Brain: 프롬프트 {len(prompt_text)}자 → Sonnet+Opus Advisor...")
        text = await self._ask_claude_with_advisor(
            SYSTEM_MASTER_BRAIN, prompt_text,
            advisor_instruction=self.ADVISOR_INSTRUCTION,
        )
        result = self._parse_json_response(text)

        # 에러 체크
        if result.get("error"):
            logger.error(f"Master Brain JSON 파싱 실패: {result['error']}")
            return self._fallback_result(data, result.get("raw_text", ""))
        # 날짜는 항상 오늘로 강제 (Claude 응답의 date를 신뢰하지 않음)
        result["date"] = datetime.now().strftime("%Y-%m-%d")

        result["model"] = "sonnet+opus_advisor"
        result["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 저장
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Master Brain: 판단 저장 → {OUTPUT_PATH}")

        return result

    def _fallback_result(self, data: dict, raw_text: str) -> dict:
        """Claude 응답 파싱 실패 시 최소한의 결과 반환."""
        overnight = data.get("overnight", {}) or {}
        regime = data.get("regime_macro", {}) or {}

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model": self.model,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "market_regime": {
                "overall": regime.get("regime", "UNKNOWN"),
                "kospi_regime": regime.get("regime", "UNKNOWN"),
                "us_overnight": overnight.get("grade", "UNKNOWN"),
                "stagflation": "UNKNOWN",
                "summary": "Master Brain 분석 실패 — 데이터 직접 확인 필요",
            },
            "reasoning_chains": [],
            "unified_picks": {
                "individual_stocks": [],
                "sector_etf": [],
                "leverage_inverse": [],
                "commodity_etf": [],
                "index_etf": [],
            },
            "cash_strategy": {
                "target_cash_pct": 30,
                "current_cash_pct": 0,
                "action": "분석 실패 — 기존 포지션 유지",
                "reason": "Master Brain 응답 파싱 실패",
            },
            "risk_alerts": ["Master Brain 분석 실패 — 수동 판단 필요"],
            "daily_summary": "분석 실패",
            "parse_error": raw_text[:500] if raw_text else "empty",
        }
