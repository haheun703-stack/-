@echo off
REM ============================================================
REM 상한가 풀림 실시간 감지 스캐너 (BAT-LU)
REM 실행 시간: 08:55 (장 시작 5분 전)
REM 종료: 15:20 자동 종료 (엔진 내부)
REM ============================================================

set PYTHONPATH=D:\sub-agent-project_퀀트봇
cd /d D:\sub-agent-project_퀀트봇

echo ============================================================
echo  [BAT-LU] 상한가 풀림 실시간 감지 시작
echo  시간: %date% %time%
echo ============================================================

call venv\Scripts\activate

REM Phase 1: 후보 생성 (최신 데이터 기반)
echo [1/2] 후보 종목 갱신...
python -u -X utf8 scripts/run_limit_up_scanner.py --generate

REM Phase 2: 실시간 스캔 (dry-run)
echo [2/2] 실시간 스캔 시작 (DRY-RUN)...
python -u -X utf8 scripts/run_limit_up_scanner.py --scan --dry-run

echo ============================================================
echo  [BAT-LU] 종료: %time%
echo ============================================================
