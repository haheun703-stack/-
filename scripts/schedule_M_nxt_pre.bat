@echo off
REM ============================================================
REM  Quantum Master - BAT-M: NXT 프리마켓 데이터 수집
REM  스케줄: 매일 07:55 (월~금, 프리마켓 시작 직전)
REM  등록: schtasks /create /tn "QM_M_NXT_Pre" /tr "D:\sub-agent-project\scripts\schedule_M_nxt_pre.bat" /sc daily /st 07:55
REM
REM  NXT 프리마켓 (08:00~08:55) 체결/수급 데이터 수집
REM  수집 완료 → SmartEntry (BAT-E, 08:50)에서 자동 활용
REM ============================================================

echo [%date% %time%] BAT-M 시작: NXT 프리마켓 수집 >> D:\sub-agent-project\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

REM ── 주말 가드 ──
for /f %%a in ('python -c "from datetime import date; print(date.today().weekday())"') do set DOW=%%a
if "%DOW%"=="5" (
    echo [%date% %time%] BAT-M 스킵: 토요일 >> logs\schedule.log
    goto :eof
)
if "%DOW%"=="6" (
    echo [%date% %time%] BAT-M 스킵: 일요일 >> logs\schedule.log
    goto :eof
)
REM 공휴일 체크
for /f %%a in ('python -c "from datetime import date; exec(\"try:\\n import holidays\\n print(1 if date.today() in holidays.KR(years=date.today().year) else 0)\\nexcept ImportError:\\n print(0)\")"') do set IS_HOLIDAY=%%a
if "%IS_HOLIDAY%"=="1" (
    echo [%date% %time%] BAT-M 스킵: 공휴일 >> logs\schedule.log
    goto :eof
)

echo ========================================
echo [QM-M] NXT 프리마켓 수집 (08:00~08:55)
echo   수집 간격: 5분
echo   BAT-E(08:50)에서 자동 활용
echo ========================================

REM ── 프리마켓 데이터 수집 ──
echo [%date% %time%] NXT 프리마켓 수집 시작 >> logs\schedule.log
python -u -X utf8 scripts/nxt_market_collector.py --session pre >> logs\schedule.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] [FAIL] nxt_market_collector (pre) 실패 (code=%ERRORLEVEL%) >> logs\schedule.log
)

echo [%date% %time%] BAT-M 완료 (프리마켓 데이터 → BAT-E에서 활용) >> logs\schedule.log
echo ========================================
echo [QM-M] 완료
echo ========================================
