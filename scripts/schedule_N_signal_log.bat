@echo off
REM ═══════════════════════════════════════════════
REM  BAT-N: FLOWX 시그널 기록
REM  스케줄: 매일 08:20 KST (QM_N_SignalLog)
REM  BAT-D(스캔) 이후 실행 — tomorrow_picks.json 기준
REM ═══════════════════════════════════════════════
set PYTHONPATH=D:\sub-agent-project_퀀트봇
cd /d D:\sub-agent-project_퀀트봇

call venv\Scripts\activate.bat

echo [%date% %time%] BAT-N 시그널 기록 시작
python -u -X utf8 scripts/cron_signal_tracker.py --mode log >> logs\signal_tracker.log 2>&1
echo [%date% %time%] BAT-N 시그널 기록 완료

deactivate
