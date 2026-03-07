@echo off
REM ============================================================
REM BAT-G: 금요일 투매 역매수 스캐너
REM 실행: 매주 금요일 14:00 (Windows 작업 스케줄러)
REM ============================================================

set PYTHONPATH=D:\sub-agent-project
cd /d D:\sub-agent-project
call venv\Scripts\activate

echo [%date% %time%] BAT-G 금요일 투매 스캐너 시작

python -u -X utf8 scripts/friday_dip_scanner.py --scan --telegram

echo [%date% %time%] BAT-G 완료
