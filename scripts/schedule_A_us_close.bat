@echo off
REM ============================================================
REM  Quantum Master - BAT-A: 미장 마감 + US Overnight Signal
REM  스케줄: 매일 06:10 (월~토, 미장 마감 직후)
REM  등록: schtasks /create /tn "QM_A_USClose" /tr "D:\sub-agent-project\scripts\schedule_A_us_close.bat" /sc daily /st 06:10
REM ============================================================

echo [%date% %time%] BAT-A 시작: 미장 마감 데이터 >> D:\sub-agent-project\logs\schedule.log

call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project

REM 1) US 시장 데이터 업데이트 + Overnight Signal
echo [%date% %time%] US Overnight Signal 업데이트 >> logs\schedule.log
python -u -X utf8 scripts\us_overnight_signal.py --update >> logs\schedule.log 2>&1

REM 2) US-KR 패턴DB 일일 누적
echo [%date% %time%] US-KR 패턴DB 업데이트 >> logs\schedule.log
python -u -X utf8 scripts\update_us_kr_daily.py >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-A 완료 >> logs\schedule.log
