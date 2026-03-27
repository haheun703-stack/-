# Quantum Master — 한국 주식 자동매매 시스템

## 언어
- 모든 응답은 **한국어**

## 환경
- `source venv/Scripts/activate` 항상 실행
- `python -u -X utf8` 한글 출력 시 사용
- `.env` 절대 커밋 금지

## 필수 규칙
- **BAT 파일**: `set PYTHONPATH=D:\sub-agent-project_퀀트봇` 필수 + `sys.path.insert(0, ...)` 안전장치
- **버전 업그레이드**: 즉시 git add → commit → tag → push
- **클린 아키텍처**: entities/ → use_cases/ → adapters/ → agents/ (안쪽→바깥 import 금지)

## 시스템 상세 참조
- **전체 시스템 맵**: `docs/SYSTEM_MAP.md` (8개 핵심 시스템, BAT 스케줄, 데이터 경로)
- **설정**: `config/settings.yaml`, `config/relay_sectors.yaml`
