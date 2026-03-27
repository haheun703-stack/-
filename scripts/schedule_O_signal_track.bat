@echo off
REM ═══════════════════════════════════════════════
REM  BAT-O: FLOWX 시그널 성과추적 + 종료 + 성적표
REM  스케줄: 매일 16:10 KST (QM_O_SignalTrack)
REM  장마감 후 실행 — 현재가 업데이트 + 종료 판정 + 집계
REM ═══════════════════════════════════════════════
set PYTHONPATH=D:\sub-agent-project_퀀트봇
cd /d D:\sub-agent-project_퀀트봇

call venv\Scripts\activate.bat

echo [%date% %time%] BAT-O 시그널 성과추적 시작
python -u -X utf8 scripts/cron_signal_tracker.py --mode track >> logs\signal_tracker.log 2>&1
echo [%date% %time%] BAT-O 시그널 성과추적 완료

deactivate
