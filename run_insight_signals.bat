@echo off
REM 인사이트 시그널 일일 수집 (관찰 모드) — 장 마감 후 실행 권장 (예: 16:20)
set PYTHONPATH=D:\sub-agent-project_퀀트봇
cd /d D:\sub-agent-project_퀀트봇
call venv\Scripts\activate
python -u -X utf8 -m insight_signals.agent.run_daily %*
