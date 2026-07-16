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

## AWS Lightsail 배포
- **인스턴스**: Bodyhunter-60GB (Seoul), RAM 2GB (서비스 3개 공유)
- **고정 IP**: 13.209.153.221 (KIS API 화이트리스트 등록됨)
- **SSH 키**: `D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem` (단타봇 프로젝트에 위치, 복사 불필요)
- **SSH 접속**: `ssh -i "D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem" -o ConnectTimeout=10 ubuntu@13.209.153.221`
- **서버 경로**: `/home/ubuntu/quantum-master/`
- **자동화 정본 = cron** (`scripts/cron/run_bat.sh`, ubuntu crontab). 평일 06:10~18:45 단계별(A~HEALTH) 직접 호출. ★`quantum-scheduler.service`는 5/27부터 **의도적 비활성(inactive)** — 살리지 말 것(cron 이중실행·freeze 위반).
- **배포**: `ssh -i "D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem" -o ConnectTimeout=10 ubuntu@13.209.153.221 "cd ~/quantum-master && git pull origin main"` — **`git pull`만**. 다음 cron 실행부터 새 코드 적용. ★`systemctl restart` 치지 말 것(무의미+위험). ⚠️장중(09:00~15:30) 금지, 페이퍼 cron(16:30 D·17:00 J) 전 권장.
- **로그**: cron 로그 `~/quantum-master/logs/cron_*.log` (scheduler journalctl 아님)

## 금지 경로 (LOCK)
- **`scripts/archive/`** — 폐기/백테스트 스크립트 보관소. **절대 참조·실행·import 금지**
- **`_etf_ref/`** — 구현 완료 후 보존용 참조 코드. 참조 금지
- 루트의 `.docx`, `.html`, 한글 지시서 파일 — 사람용 문서이며 코드가 아님. 무시할 것

## 시스템 상세 참조
- **전체 시스템 맵**: `docs/SYSTEM_MAP.md` (8개 핵심 시스템, BAT 스케줄, 데이터 경로)
- **설정**: `config/settings.yaml`, `config/relay_sectors.yaml`

## 데일리 운영 루프 (매 세션 필수)
- **정본**: `docs/DAILY_CHECKLIST.md` — 세션 시작 시 §1 아침 루틴(날짜·전일 cron·헬스체크·기초데이터·배포분 첫 실동작) 수행 후 §4에 당일 기록, 마감 시 §2 자기성찰 3질문. 개선점은 각주 달아 §3 백로그 등재, 매일 1건 이상 소화.
