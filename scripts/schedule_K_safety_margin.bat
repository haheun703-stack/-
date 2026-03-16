@echo off
chcp 65001 >nul
REM ============================================================
REM  Quantum Master - BAT-K: 안전마진 일일 스캔
REM  스케줄: 매일 07:30 (월~금, BAT-B 다음)
REM  등록: schtasks /create /tn "QM_K_SafetyMargin" /tr "D:\sub-agent-project\scripts\schedule_K_safety_margin.bat" /sc daily /st 07:30
REM
REM  컨센서스 기반 GREEN/YELLOW 판별 → 텔레그램 알림
REM  DART 폴백 OFF, wisereport fetch_one 활성
REM ============================================================

call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

REM ── 주말 가드 ──
for /f %%a in ('python -c "from datetime import date; print(date.today().weekday())"') do set DOW=%%a
if "%DOW%"=="5" (
    echo [%date% %time%] BAT-K 스킵: 토요일 >> D:\sub-agent-project\logs\schedule.log
    goto :eof
)
if "%DOW%"=="6" (
    echo [%date% %time%] BAT-K 스킵: 일요일 >> D:\sub-agent-project\logs\schedule.log
    goto :eof
)

REM ── 공휴일 가드 ──
for /f %%a in ('python -c "from datetime import date; import holidays; print(1 if date.today() in holidays.KR(years=date.today().year) else 0)"') do set IS_HOLIDAY=%%a
if "%IS_HOLIDAY%"=="1" (
    echo [%date% %time%] BAT-K 스킵: 공휴일 >> D:\sub-agent-project\logs\schedule.log
    goto :eof
)

echo [%date% %time%] BAT-K 시작: 안전마진 스캔 >> D:\sub-agent-project\logs\schedule.log

python -u -X utf8 scripts\scan_safety_margin.py >> D:\sub-agent-project\logs\schedule.log 2>&1

echo [%date% %time%] BAT-K 완료 >> D:\sub-agent-project\logs\schedule.log
