@echo off
chcp 65001 >nul
REM ============================================================
REM  Quantum Master - BAT-H: 11:30 장중 AI 분석
REM  스케줄: 매일 11:30 (월~금)
REM  등록: schtasks /create /tn "QM_H_Midday" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_H_midday_analysis.bat" /sc daily /st 11:30
REM
REM  5종목 TradeAdvisor 분석 → 텔레그램 전송
REM  12:00 매수 판단용 데이터 제공
REM ============================================================

echo [%date% %time%] BAT-H 시작: 장중 AI 분석 >> D:\sub-agent-project_퀀트봇\logs\schedule.log

call D:\sub-agent-project_퀀트봇\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_퀀트봇
set PYTHONPATH=D:\sub-agent-project_퀀트봇

REM ── 거래일 가드 (trading_calendar 사용) ──
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-H 스킵: 비거래일 >> logs\schedule.log
    goto :eof
)

echo ========================================
echo [QM-H] 11:30 장중 AI 분석 (5종목)
echo   TradeAdvisor 분석 → 텔레그램 전송
echo   12:00 매수 판단용 데이터
echo ========================================

python -u -X utf8 scripts/run_midday_analysis.py >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-H Market Pulse 시작 >> logs\schedule.log
python -u -X utf8 scripts/market_pulse.py --no-send >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-H 완료 >> logs\schedule.log
