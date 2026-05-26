"""주간 ChatGPT 외부 검수 — 옵션 B (5/27 신규).

배경: 매주 토요일 18:00 cron 자동 실행.
1주간 commit 변경 + 현재 시스템 상태를 ChatGPT GPT-4o에 외부 검증.
P0/P1 발견 시 텔레그램 알림 + GitHub Issue 자동 생성 (선택).

실행:
  PYTHONPATH=. ./venv/bin/python3.11 scripts/weekly_chatgpt_review.py

cron: 0 18 * * 6  (매주 토 18:00)
비용: 약 $0.015/회 (1주 1회 = 월 $0.06)
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_weekly_diff() -> str:
    """1주간 commit log + diff stat 추출."""
    try:
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        log = subprocess.check_output(
            ["git", "log", f"--since={week_ago}", "--oneline", "--stat"],
            cwd=PROJECT_ROOT, encoding="utf-8", timeout=10,
        )
        # 너무 길면 잘라냄
        return log[:8000]
    except Exception as e:
        return f"git log 실패: {e}"


def get_system_summary() -> str:
    """현재 시스템 핵심 모듈 요약."""
    use_cases = list((PROJECT_ROOT / "src" / "use_cases").glob("*.py"))
    return (
        f"신규/통합 모듈 ({len(use_cases)}개):\n"
        + "\n".join(f"  - {p.name}" for p in sorted(use_cases) if not p.name.startswith("_"))
    )


def send_to_chatgpt(weekly_log: str, system_summary: str) -> str:
    """ChatGPT GPT-4o 외부 검수 요청."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OPENAI_API_KEY 없음"

        client = OpenAI(api_key=api_key)

        system_prompt = (
            "당신은 한국 주식 자동매매 시스템 검증 전문가입니다. "
            "Claude Code AI가 1주간 작성한 변경 사항을 외부 검증합니다. "
            "P0 (즉시 fix)/P1 (개선)/P2 (코드 품질) 분류 보고."
        )

        user_prompt = f"""
## 1주간 commit + 변경 stat
{weekly_log}

## 현재 시스템 모듈
{system_summary}

## 검증 요청
1. 새로 도입된 룰의 논리적 충돌 가능성
2. 누락된 위험 시나리오 (변동성/갭/외인/이벤트)
3. 백테스트 X 잠정치의 위험 평가
4. 5/27~6/2 한 주 가동 후 예상 이슈

답변: 한국어 600~1000단어, P0/P1/P2 분류.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ChatGPT 호출 실패: {e}"


def send_telegram_report(review: str):
    """검수 결과 텔레그램."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.telegram_sender import send_message

        # P0 항목 자동 추출
        has_p0 = "P0" in review and "즉시" in review
        emoji = "🚨" if has_p0 else "📋"
        title = f"{emoji} [주간 ChatGPT 외부 검수] {datetime.now():%Y-%m-%d}"

        # 너무 길면 잘라내기 (텔레그램 4096자 한도)
        review_short = review[:3500] + "\n...\n(중략)" if len(review) > 3700 else review
        msg = f"{title}\n\n{review_short}"
        send_message(msg)

        # 전체 보고는 파일로 저장
        report_dir = PROJECT_ROOT / "logs" / "weekly_reviews"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"chatgpt_review_{datetime.now():%Y%m%d_%H%M}.md"
        report_file.write_text(f"# {title}\n\n{review}", encoding="utf-8")
        print(f"보고서 저장: {report_file}")
    except Exception as e:
        print(f"텔레그램 발송 실패: {e}")


def main():
    print(f"[ChatGPT 주간 검수] {datetime.now():%Y-%m-%d %H:%M:%S} 시작")
    weekly_log = get_weekly_diff()
    system_summary = get_system_summary()
    print(f"1주 commit: {len(weekly_log)} bytes")
    print(f"시스템: {system_summary[:200]}...")

    review = send_to_chatgpt(weekly_log, system_summary)
    print()
    print("=" * 70)
    print(review)
    print("=" * 70)

    send_telegram_report(review)


if __name__ == "__main__":
    main()
