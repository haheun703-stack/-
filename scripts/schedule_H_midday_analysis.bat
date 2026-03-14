@echo off
chcp 65001 >nul
REM ============================================================
REM  Quantum Master - BAT-H: 11:30 장중 AI 분석
REM  스케줄: 매일 11:30 (월~금)
REM  등록: schtasks /create /tn "QM_H_Midday" /tr "D:\sub-agent-project\scripts\schedule_H_midday_analysis.bat" /sc daily /st 11:30
REM
REM  5종목 TradeAdvisor 분석 → 텔레그램 전송
REM  12:00 매수 판단용 데이터 제공
REM ============================================================

echo [%date% %time%] BAT-H 시작: 장중 AI 분석 >> D:\sub-agent-project\logs\schedule.log

call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

REM ── 주말 가드 ──
for /f %%a in ('python -c "from datetime import date; print(date.today().weekday())"') do set DOW=%%a
if "%DOW%"=="5" (
    echo [%date% %time%] BAT-H 스킵: 토요일 >> logs\schedule.log
    goto :eof
)
if "%DOW%"=="6" (
    echo [%date% %time%] BAT-H 스킵: 일요일 >> logs\schedule.log
    goto :eof
)

REM 공휴일 체크
for /f %%a in ('python -c "from datetime import date; exec(\"try:\\n import holidays\\n print(1 if date.today() in holidays.KR(years=date.today().year) else 0)\\nexcept ImportError:\\n print(0)\")"') do set IS_HOLIDAY=%%a
if "%IS_HOLIDAY%"=="1" (
    echo [%date% %time%] BAT-H 스킵: 공휴일 >> logs\schedule.log
    goto :eof
)

echo ========================================
echo [QM-H] 11:30 장중 AI 분석 (5종목)
echo   TradeAdvisor 분석 → 텔레그램 전송
echo   12:00 매수 판단용 데이터
echo ========================================

python -u -X utf8 scripts/run_midday_analysis.py >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-H 완료 >> logs\schedule.log
