@echo off
REM ============================================================
REM  Quantum Master - BAT-B: 뉴스 스캔 + 장전 마켓 브리핑
REM  스케줄: 매일 07:00 (월~금, 장 시작 전)
REM  등록: schtasks /create /tn "QM_B_Morning" /tr "D:\sub-agent-project\scripts\schedule_B_morning.bat" /sc daily /st 07:00
REM ============================================================

echo [%date% %time%] BAT-B 시작: 장전 브리핑 >> D:\sub-agent-project\logs\schedule.log

call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project

REM 1) RSS 테마 스캔 + Grok 확장 (텔레그램 발송 포함)
echo [%date% %time%] 테마 스캔 >> logs\schedule.log
python -u -X utf8 -c "from scripts.theme_scan_runner import run_theme_scan; run_theme_scan(use_grok=True, send_telegram=True)" >> logs\schedule.log 2>&1

REM 2) 뉴스 스캔
echo [%date% %time%] 뉴스 스캔 >> logs\schedule.log
python -u -X utf8 -c "from main import step_news_scan; step_news_scan(send_telegram=False)" >> logs\schedule.log 2>&1

REM 3) 장전 마켓 브리핑 (텔레그램 발송)
echo [%date% %time%] 장전 브리핑 텔레그램 발송 >> logs\schedule.log
python -u -X utf8 -c "from scripts.send_market_briefing import build_briefing_message; from src.telegram_sender import send_message; msg=build_briefing_message(); send_message(msg)" >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-B 완료 >> logs\schedule.log
