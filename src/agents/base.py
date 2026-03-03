"""서브에이전트 공통 베이스 - Claude API 호출 공통 로직"""

from __future__ import annotations

import json

import anthropic


class BaseAgent:
    """Claude API를 호출하는 서브에이전트 베이스 클래스"""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
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
        return json.loads(text.strip())
