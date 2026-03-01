@echo off
REM ============================================
REM BAT-F: 스나이퍼 워치 일일 스캔
REM 매일 17:30 실행 (장마감 후)
REM ============================================

set PYTHONPATH=D:\sub-agent-project
cd /d D:\sub-agent-project
call venv\Scripts\activate.bat

echo [%date% %time%] BAT-F 스나이퍼 워치 시작

python -u -X utf8 scripts/scan_value_trap.py --telegram >> logs\sniper_watch.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] BAT-F 스나이퍼 워치 실패 >> logs\sniper_watch.log
) else (
    echo [%date% %time%] BAT-F 스나이퍼 워치 완료 >> logs\sniper_watch.log
)
