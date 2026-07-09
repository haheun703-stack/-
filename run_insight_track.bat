@echo off
REM 인사이트 시그널 누적 성과 평가 (단독 실행용 — 일일 리포트에도 요약 포함됨)
set PYTHONPATH=D:\sub-agent-project_퀀트봇
cd /d D:\sub-agent-project_퀀트봇
call venv\Scripts\activate
python -u -X utf8 -m insight_signals.agent.track_performance %*
