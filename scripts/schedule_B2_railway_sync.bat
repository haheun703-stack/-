@echo off
REM ============================================================
REM  Quantum Master - BAT-B2: Railway 앱 데이터 동기화 (장전)
REM  스케줄: 매일 07:20 (월~금, BAT-A/B 완료 후)
REM  등록: schtasks /create /tn "QM_B2_RailwaySync" /tr "D:\sub-agent-project\scripts\schedule_B2_railway_sync.bat" /sc daily /st 07:20
REM ============================================================

echo [%date% %time%] BAT-B2 시작: Railway 동기화 (장전) >> D:\sub-agent-project\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

echo [%date% %time%] Railway 동기화 실행 >> logs\schedule.log
python -u -X utf8 scripts\sync_to_railway.py >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-B2 완료 >> logs\schedule.log
