"""서브에이전트 공통 베이스 - Claude API 호출 공통 로직"""

from __future__ import annotations

import json

import anthropic


class BaseAgent:
    """Claude API를 호출하는 서브에이전트 베이스 클래스"""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        self.client = anthropic.AsyncAnthropic()
        self.model = model

    async def _ask_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Claude API에 질문하고 텍스트 응답을 받는다"""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    async def _ask_claude_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Claude API에 질문하고 JSON 응답을 파싱한다"""
        text = await self._ask_claude(system_prompt, user_prompt)

        # JSON 블록 추출
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())
