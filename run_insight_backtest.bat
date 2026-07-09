@echo off
REM D002 임원 매수 과거 백테스트 — 승격 판단 전 필수 실행 (기본 6개월)
set PYTHONPATH=D:\sub-agent-project_퀀트봇
cd /d D:\sub-agent-project_퀀트봇
call venv\Scripts\activate
python -u -X utf8 -m insight_signals.agent.backtest_dart %*
