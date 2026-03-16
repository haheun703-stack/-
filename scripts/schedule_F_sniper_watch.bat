@echo off
chcp 65001 >nul
REM ============================================
REM BAT-F: 스나이퍼 워치 일일 스캔
REM 매일 17:30 실행 (장마감 후)
REM 등록: schtasks /create /tn "QM_F_SniperWatch" /tr "D:\sub-agent-project\scripts\schedule_F_sniper_watch.bat" /sc daily /st 17:30
REM ============================================

call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

REM ── 주말 가드 ──
for /f %%a in ('python -c "from datetime import date; print(date.today().weekday())"') do set DOW=%%a
if "%DOW%"=="5" (
    echo [%date% %time%] BAT-F 스킵: 토요일 >> D:\sub-agent-project\logs\schedule.log
    goto :eof
)
if "%DOW%"=="6" (
    echo [%date% %time%] BAT-F 스킵: 일요일 >> D:\sub-agent-project\logs\schedule.log
    goto :eof
)

REM ── 공휴일 가드 ──
for /f %%a in ('python -c "from datetime import date; import holidays; print(1 if date.today() in holidays.KR(years=date.today().year) else 0)"') do set IS_HOLIDAY=%%a
if "%IS_HOLIDAY%"=="1" (
    echo [%date% %time%] BAT-F 스킵: 공휴일 >> D:\sub-agent-project\logs\schedule.log
    goto :eof
)

echo [%date% %time%] BAT-F 스나이퍼 워치 시작 >> D:\sub-agent-project\logs\schedule.log

REM v13: --telegram 제거 → 저녁 통합 리포트에 흡수 (JSON만 저장)
python -u -X utf8 scripts/scan_value_trap.py >> D:\sub-agent-project\logs\schedule.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] BAT-F 스나이퍼 워치 실패 >> D:\sub-agent-project\logs\schedule.log
) else (
    echo [%date% %time%] BAT-F 스나이퍼 워치 완료 >> D:\sub-agent-project\logs\schedule.log
)
