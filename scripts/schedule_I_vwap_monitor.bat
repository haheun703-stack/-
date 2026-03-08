@echo off
REM ============================================================
REM  Quantum Master - BAT-I: 장중 VWAP 모니터
REM  스케줄: 매일 08:55 (월~금, 장 시작 5분 전)
REM  등록: schtasks /create /tn "QM_I_VWAP" /tr "D:\sub-agent-project\scripts\schedule_I_vwap_monitor.bat" /sc daily /st 08:55
REM
REM  09:00 개장 갭 → 09:30 VWAP 기준선 → 14:00까지 모니터링
REM  VWAP 눌림/회복 알림 + 11:30 AI 분석 통합
REM  장중 약 5시간 실행 (long-running)
REM ============================================================

echo [%date% %time%] BAT-I 시작: VWAP 모니터 >> D:\sub-agent-project\logs\schedule.log

call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

REM ── 주말 가드 ──
for /f %%a in ('python -c "from datetime import date; print(date.today().weekday())"') do set DOW=%%a
if "%DOW%"=="5" (
    echo [%date% %time%] BAT-I 스킵: 토요일 >> logs\schedule.log
    goto :eof
)
if "%DOW%"=="6" (
    echo [%date% %time%] BAT-I 스킵: 일요일 >> logs\schedule.log
    goto :eof
)

REM 공휴일 체크
for /f %%a in ('python -c "from datetime import date; exec(\"try:\n import holidays\n print(1 if date.today() in holidays.KR(years=date.today().year) else 0)\nexcept ImportError:\n print(0)\")"') do set IS_HOLIDAY=%%a
if "%IS_HOLIDAY%"=="1" (
    echo [%date% %time%] BAT-I 스킵: 공휴일 >> logs\schedule.log
    goto :eof
)

echo ========================================
echo [QM-I] VWAP 모니터 (08:55~14:00)
echo   09:00 개장 갭 분석
echo   09:30 VWAP 기준선 설정
echo   11:30 AI 분석 통합
echo   14:00 종료
echo ========================================

python -u -X utf8 scripts/run_vwap_monitor.py >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-I 완료 >> logs\schedule.log
