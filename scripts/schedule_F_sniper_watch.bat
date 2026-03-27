@echo off
chcp 65001 >nul
REM ============================================
REM BAT-F: 스나이퍼 워치 일일 스캔
REM 매일 17:30 실행 (장마감 후)
REM 등록: schtasks /create /tn "QM_F_SniperWatch" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_F_sniper_watch.bat" /sc daily /st 17:30
REM ============================================

call D:\sub-agent-project_퀀트봇\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_퀀트봇
set PYTHONPATH=D:\sub-agent-project_퀀트봇

REM ── 거래일 가드 (trading_calendar 사용) ──
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-F 스킵: 비거래일 >> D:\sub-agent-project_퀀트봇\logs\schedule.log
    goto :eof
)

echo [%date% %time%] BAT-F 스나이퍼 워치 시작 >> D:\sub-agent-project_퀀트봇\logs\schedule.log

REM v13: --telegram 제거 → 저녁 통합 리포트에 흡수 (JSON만 저장)
python -u -X utf8 scripts/scan_value_trap.py >> D:\sub-agent-project_퀀트봇\logs\schedule.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] BAT-F 스나이퍼 워치 실패 >> D:\sub-agent-project_퀀트봇\logs\schedule.log
) else (
    echo [%date% %time%] BAT-F 스나이퍼 워치 완료 >> D:\sub-agent-project_퀀트봇\logs\schedule.log
)
