@echo off
REM ============================================================
REM  Quantum Master - BAT-L: NXT 애프터마켓 데이터 수집
REM  스케줄: 매일 15:35 (월~금, 장 마감 직후)
REM  등록: schtasks /create /tn "QM_L_NXT_After" /tr "D:\sub-agent-project\scripts\schedule_L_nxt_after.bat" /sc daily /st 15:35
REM
REM  NXT 애프터마켓 (15:30~20:00) 체결/수급 데이터 수집
REM  종료 후 NXT 시그널 자동 분석
REM ============================================================

echo [%date% %time%] BAT-L 시작: NXT 애프터마켓 수집 >> D:\sub-agent-project\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

REM ── 주말 가드 ──
for /f %%a in ('python -c "from datetime import date; print(date.today().weekday())"') do set DOW=%%a
if "%DOW%"=="5" (
    echo [%date% %time%] BAT-L 스킵: 토요일 >> logs\schedule.log
    goto :eof
)
if "%DOW%"=="6" (
    echo [%date% %time%] BAT-L 스킵: 일요일 >> logs\schedule.log
    goto :eof
)
REM 공휴일 체크
for /f %%a in ('python -c "from datetime import date; exec(\"try:\\n import holidays\\n print(1 if date.today() in holidays.KR(years=date.today().year) else 0)\\nexcept ImportError:\\n print(0)\")"') do set IS_HOLIDAY=%%a
if "%IS_HOLIDAY%"=="1" (
    echo [%date% %time%] BAT-L 스킵: 공휴일 >> logs\schedule.log
    goto :eof
)

echo ========================================
echo [QM-L] NXT 애프터마켓 수집 (15:35~20:00)
echo   수집 간격: 10분
echo   종료 후: NXT 시그널 분석
echo ========================================

REM ── 1단계: 애프터마켓 데이터 수집 ──
echo [%date% %time%] NXT 애프터마켓 수집 시작 >> logs\schedule.log
python -u -X utf8 scripts/nxt_market_collector.py --session after >> logs\schedule.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] [FAIL] nxt_market_collector 실패 (code=%ERRORLEVEL%) >> logs\schedule.log
)

REM ── 2단계: NXT 시그널 분석 ──
echo [%date% %time%] NXT 시그널 분석 시작 >> logs\schedule.log
python -u -X utf8 -c "import sys; sys.path.insert(0, 'D:\\sub-agent-project'); from src.use_cases.nxt_signal import NxtSignalAnalyzer; a = NxtSignalAnalyzer(); r = a.generate_signal(); print(f'NXT 시그널: STRONG_BUY={r[\"summary\"][\"after_strong_buy\"]}, BUY={r[\"summary\"][\"after_buy\"]}')" >> logs\schedule.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] [FAIL] nxt_signal 분석 실패 (code=%ERRORLEVEL%) >> logs\schedule.log
)

echo [%date% %time%] BAT-L 완료 >> logs\schedule.log
echo ========================================
echo [QM-L] 완료
echo ========================================
