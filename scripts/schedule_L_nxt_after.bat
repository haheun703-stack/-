@echo off
REM ============================================================
REM  Quantum Master - BAT-L: NXT 애프터마켓 데이터 수집
REM  스케줄: 매일 15:35 (월~금, 장 마감 직후)
REM  등록: schtasks /create /tn "QM_L_NXT_After" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_L_nxt_after.bat" /sc daily /st 15:35
REM
REM  NXT 애프터마켓 (15:30~20:00) 체결/수급 데이터 수집
REM  종료 후 NXT 시그널 분석 + NXT 추천 엔진
REM ============================================================

echo [%date% %time%] BAT-L 시작: NXT 애프터마켓 수집 >> D:\sub-agent-project_퀀트봇\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project_퀀트봇\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_퀀트봇
set PYTHONPATH=D:\sub-agent-project_퀀트봇

REM ── 거래일 가드 (trading_calendar 사용) ──
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-L 스킵: 비거래일 >> logs\schedule.log
    goto :eof
)

echo ========================================
echo [QM-L] NXT 애프터마켓 수집 (15:35~20:00)
echo   수집 간격: 10분
echo   종료 후: NXT 시그널 분석 + 추천 엔진
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

REM ── 3단계: NXT 추천 엔진 (tomorrow_picks × 애프터 수급 교차) ──
echo [%date% %time%] NXT 추천 엔진 시작 >> logs\schedule.log
python -u -X utf8 scripts/nxt_recommend.py >> logs\schedule.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] [FAIL] nxt_recommend 실패 (code=%ERRORLEVEL%) >> logs\schedule.log
)

echo [%date% %time%] BAT-L 완료 >> logs\schedule.log
echo ========================================
echo [QM-L] 완료
echo ========================================
