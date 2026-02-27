@echo off
REM ============================================================
REM  Quantum Master - BAT-B: 뉴스 스캔 + 장전 마켓 브리핑
REM  스케줄: 매일 07:00 (월~금, 장 시작 전)
REM  등록: schtasks /create /tn "QM_B_Morning" /tr "D:\sub-agent-project\scripts\schedule_B_morning.bat" /sc daily /st 07:00
REM ============================================================

echo [%date% %time%] BAT-B 시작: 장전 브리핑 >> D:\sub-agent-project\logs\schedule.log

call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

REM 1단계: 장전 리포트 스캔 (증권사 리포트 + Perplexity 테마 + 후보풀 뉴스)
echo [%date% %time%] [1/2] 장전 리포트 스캔 >> logs\schedule.log
python -u -X utf8 scripts\crawl_morning_reports.py --send >> logs\schedule.log 2>&1

REM 2단계: 테마스캔 + 뉴스스캔 + 장전 브리핑 (텔레그램 발송 포함)
echo [%date% %time%] [2/2] 장전 브리핑 실행 >> logs\schedule.log
python -u -X utf8 scripts\run_morning_briefing.py >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-B 완료 >> logs\schedule.log
