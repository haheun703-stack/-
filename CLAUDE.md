# Sub Agent Project

## 언어
- 모든 응답, 질문, 선택지는 **한국어**로 작성할 것

## 환경
- Python 가상환경: `venv/`
- 가상환경 활성화: `source venv/Scripts/activate`
- 주요 라이브러리: openai, anthropic, python-dotenv

## 규칙
- Bash 명령 실행 시 항상 가상환경을 활성화한 후 실행할 것
- 환경변수는 `.env` 파일로 관리하며 절대 커밋하지 않을 것

## 프로젝트 아키텍처: 클린 아키텍처
이 프로젝트는 클린 아키텍처를 따릅니다.

## 계층 규칙 (필수 준수)
- `entities/`: 핵심 비즈니스 객체 (외부 의존 금지)
- `use_cases/`: 비즈니스 로직 (ports 인터페이스만 사용)
- `adapters/`: Port 구현체, API 변환기
- `agents/`: AI 서브에이전트 구현

## 의존성 규칙
- 안쪽 계층은 바깥 계층을 절대 import하지 않는다
- 바깥 계층은 안쪽의 Port(인터페이스)를 구현한다

## 기술 스택
- Python 3.13 + asyncio
- Anthropic Claude API
- 한국투자증권 API (mojito2)
