@echo off
REM ═══════════════════════════════════════════════
REM  BAT-P: FLOWX 단타봇 (시그널 기록 + 일괄 청산)
REM  스케줄:
REM    1차: 09:05 KST (QM_P1_DaySignal) — 단타 시그널 기록
REM    2차: 15:20 KST (QM_P2_DayClose)  — 일괄 청산 + 성적표
REM  사용법: schedule_P_daytrading.bat [log|close]
REM ═══════════════════════════════════════════════
set PYTHONPATH=D:\sub-agent-project
cd /d D:\sub-agent-project

call venv\Scripts\activate.bat

set MODE=%1
if "%MODE%"=="" set MODE=log

if "%MODE%"=="log" (
    echo [%date% %time%] BAT-P 단타 시그널 기록 시작
    python -u -X utf8 scripts/signal_logger_daytrading.py >> logs\daytrading.log 2>&1
    echo [%date% %time%] BAT-P 단타 시그널 기록 완료
) else if "%MODE%"=="close" (
    echo [%date% %time%] BAT-P 단타 일괄 청산 시작
    python -u -X utf8 scripts/daily_close_daytrading.py >> logs\daytrading.log 2>&1
    echo [%date% %time%] BAT-P 단타 일괄 청산 완료
) else (
    echo 사용법: schedule_P_daytrading.bat [log^|close]
)

deactivate
