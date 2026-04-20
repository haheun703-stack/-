"""서브에이전트 공통 베이스 - Claude API 호출 공통 로직

비용 최적화 (2026-04 정보봇 지시):
- Opus 직접 호출 → Sonnet + Opus Advisor 전환
- 3단계 폴백: Sonnet+Opus Advisor → Sonnet 단독 → Haiku
- 일별 모델별 토큰 사용량 추적
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

import anthropic

logger = logging.getLogger(__name__)

# ─── 비용 추적 (일별 모델별 토큰 사용량) ───
_usage_tracker: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "calls": 0})


def get_usage_report() -> dict:
    """현재 세션 토큰 사용량 리포트."""
    return dict(_usage_tracker)


def _track_usage(model: str, response):
    """응답에서 토큰 사용량 추적."""
    if hasattr(response, "usage"):
        _usage_tracker[model]["input"] += getattr(response.usage, "input_tokens", 0)
        _usage_tracker[model]["output"] += getattr(response.usage, "output_tokens", 0)
        _usage_tracker[model]["calls"] += 1


# ─── 모델 상수 ───
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_OPUS = "claude-opus-4-6"
MODEL_HAIKU = "claude-haiku-4-5-20251001"


class BaseAgent:
    """Claude API를 호출하는 서브에이전트 베이스 클래스"""

    def __init__(self, model: str = MODEL_SONNET):
        self.client = anthropic.AsyncAnthropic()
        self.model = model

    async def _ask_claude(self, system_prompt: str, user_prompt: str, max_tokens: int = 16000) -> str:
        """Claude API에 질문하고 텍스트 응답을 받는다"""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        _track_usage(self.model, response)
        return response.content[0].text

    async def _ask_claude_with_advisor(
        self,
        system_prompt: str,
        user_prompt: str,
        advisor_instruction: str = "",
        max_tokens: int = 16000,
    ) -> str:
        """Sonnet executor + Opus advisor 패턴.

        3단계 폴백: Sonnet+Opus Advisor → Sonnet 단독 → Haiku
        지시서: 정보봇 → 퀀트봇 AI 모델 비용 최적화 [최적화 1]
        """
        full_prompt = user_prompt
        if advisor_instruction:
            full_prompt = f"{user_prompt}\n\n[ADVISOR 검증 지시]\n{advisor_instruction}"

        # 1차: Sonnet + Opus Advisor
        try:
            response = await self.client.messages.create(
                model=MODEL_SONNET,
                max_tokens=max_tokens,
                extra_headers={"anthropic-beta": "advisor-tool-2026-03-01"},
                system=system_prompt,
                messages=[{"role": "user", "content": full_prompt}],
                tools=[{
                    "type": "advisor_20260301",
                    "name": "advisor",
                    "model": MODEL_OPUS,
                    "max_uses": 1,
                }],
            )
            _track_usage(f"{MODEL_SONNET}+advisor({MODEL_OPUS})", response)
            # content 배열에서 text 블록만 추출
            text = "".join(
                b.text for b in response.content
                if hasattr(b, "text") and getattr(b, "type", "") == "text"
            )
            if text:
                logger.info("Advisor 패턴 성공 (Sonnet+Opus)")
                return text
        except Exception as e:
            logger.warning("Advisor 패턴 실패: %s → Sonnet 단독 폴백", e)

        # 2차: Sonnet 단독 (스트리밍)
        try:
            logger.info("Sonnet 단독 폴백...")
            async with self.client.messages.stream(
                model=MODEL_SONNET,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                resp = await stream.get_final_message()
                _track_usage(MODEL_SONNET, resp)
                return await stream.get_final_text()
        except Exception as e:
            logger.warning("Sonnet 단독 실패: %s → Haiku 폴백", e)

        # 3차: Haiku 폴백
        logger.warning("Haiku 최종 폴백!")
        response = await self.client.messages.create(
            model=MODEL_HAIKU,
            max_tokens=min(max_tokens, 8000),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        _track_usage(MODEL_HAIKU, response)
        return response.content[0].text

    async def _ask_claude_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Claude API에 질문하고 JSON 응답을 파싱한다"""
        text = await self._ask_claude(system_prompt, user_prompt)
        return self._parse_json_response(text)

    async def _ask_claude_vision(
        self,
        system_prompt: str,
        text_prompt: str,
        images: list[str],
        max_tokens: int = 16000,
    ) -> str:
        """Claude Vision API — 텍스트 + base64 이미지 분석.

        Args:
            system_prompt: 시스템 프롬프트
            text_prompt: 유저 텍스트
            images: base64 PNG 문자열 리스트
            max_tokens: 최대 토큰

        Returns:
            Claude 텍스트 응답
        """
        content = []
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img,
                },
            })
        content.append({"type": "text", "text": text_prompt})

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    async def _ask_claude_vision_json(
        self,
        system_prompt: str,
        text_prompt: str,
        images: list[str],
    ) -> dict:
        """Claude Vision API + JSON 파싱."""
        text = await self._ask_claude_vision(system_prompt, text_prompt, images)
        return self._parse_json_response(text)

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """응답 텍스트에서 JSON 추출 및 파싱."""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning("JSON 파싱 실패: %s — 원문: %s", e, text[:200])
            return {"error": f"JSON 파싱 실패: {e}", "raw_text": text[:500]}
