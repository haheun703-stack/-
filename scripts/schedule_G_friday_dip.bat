@echo off
REM ============================================================
REM BAT-G: 금요일 투매 역매수 스캐너
REM 실행: 매주 금요일 14:00 (Windows 작업 스케줄러)
REM ============================================================

chcp 65001 >nul
call D:\sub-agent-project_퀀트봇\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_퀀트봇
set PYTHONPATH=D:\sub-agent-project_퀀트봇

REM 금요일 확인 (schtasks에서 제어하지만 이중 안전장치)
for /f %%a in ('python -c "from datetime import date; print(date.today().weekday())"') do set DOW=%%a
if NOT "%DOW%"=="4" (
    echo [%date% %time%] BAT-G 스킵: 금요일 아님 >> D:\sub-agent-project_퀀트봇\logs\schedule.log
    goto :eof
)

echo [%date% %time%] BAT-G 시작: 금요일 투매 스캐너 >> D:\sub-agent-project_퀀트봇\logs\schedule.log

python -u -X utf8 scripts/friday_dip_scanner.py --scan --telegram >> D:\sub-agent-project_퀀트봇\logs\schedule.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] [FAIL] BAT-G 금요일 투매 스캐너 실패 >> D:\sub-agent-project_퀀트봇\logs\schedule.log
)

echo [%date% %time%] BAT-G 완료 >> D:\sub-agent-project_퀀트봇\logs\schedule.log
