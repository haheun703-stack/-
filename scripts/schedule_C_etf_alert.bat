@echo off
REM ============================================================
REM  Quantum Master - BAT-C: ETF 매매 시그널 텔레그램 발송
REM  스케줄: 매일 08:00 (월~금, 장 시작 1시간 전)
REM  등록: schtasks /create /tn "QM_C_ETFAlert" /tr "D:\sub-agent-project\scripts\schedule_C_etf_alert.bat" /sc daily /st 08:00
REM ============================================================

echo [%date% %time%] BAT-C 시작: ETF 시그널 발송 >> D:\sub-agent-project\logs\schedule.log

call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

REM ETF 매매 시그널 텔레그램 발송 (전일 생성된 JSON 기반)
echo [%date% %time%] ETF 시그널 텔레그램 >> logs\schedule.log
python -u -X utf8 scripts\run_etf_telegram.py >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-C 완료 >> logs\schedule.log
