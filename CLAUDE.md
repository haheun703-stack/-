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
- **서비스**: `quantum-scheduler.service` (systemd)
- **배포**: `ssh -i "D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem" -o ConnectTimeout=10 ubuntu@13.209.153.221 "cd ~/quantum-master && git pull && sudo systemctl restart quantum-scheduler"`
- **로그**: `sudo journalctl -u quantum-scheduler --no-pager -n 50`

## 시스템 상세 참조
- **전체 시스템 맵**: `docs/SYSTEM_MAP.md` (8개 핵심 시스템, BAT 스케줄, 데이터 경로)
- **설정**: `config/settings.yaml`, `config/relay_sectors.yaml`
