@echo off
REM ============================================================
REM  Quantum Master - BAT-E: AI 스마트 진입 (LIVE)
REM  스케줄: 매일 08:50 (월~금, 장 시작 10분 전)
REM  등록: schtasks /create /tn "QM_E_SmartEntry" /tr "D:\sub-agent-project\scripts\schedule_E_smart_entry.bat" /sc daily /st 08:50
REM
REM  매수 소스: v3 AI Brain picks (우선) + 기존 TOP 7 (보조)
REM  안전장치:
REM    - max_stocks=5 (v3 2종목 + TOP 3종목)
REM    - v3 종목: 예수금 × size_pct 동적 사이징
REM    - 비-v3 종목: 종목당 max 500만원
REM    - 킬스위치: data/KILL_SWITCH 파일 생성 시 즉시 중단
REM    - 당일 중복 실행 방지 (order_audit.db)
REM    - 모든 주문 감사 로그 기록
REM ============================================================

echo [%date% %time%] BAT-E 시작: AI 스마트 진입 (LIVE) >> D:\sub-agent-project\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

REM ── 거래일 가드 (trading_calendar 사용) ──
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-E 스킵: 비거래일 >> logs\schedule.log
    goto :eof
)

echo ========================================
echo [QM-E] AI 스마트 진입 LIVE (08:50~10:30)
echo   v3 + TOP7 병행, max_stocks=5
echo   v3: 예수금 x size_pct 동적 사이징
echo   킬스위치: data/KILL_SWITCH
echo ========================================

echo [%date% %time%] SmartEntry LIVE 실행 >> logs\schedule.log
python -u -X utf8 scripts/smart_entry_runner.py --live --force >> logs\schedule.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] [FAIL] smart_entry_runner 실패 (code=%ERRORLEVEL%) >> logs\schedule.log
)

REM ── 듀얼 AI 매도 모니터: Claude+GPT (DRY-RUN, 2026-03-03 재설계) ──
echo [%date% %time%] Dual AI Sell Monitor (DRY-RUN) 시작 >> logs\schedule.log
echo ========================================
echo [QM-E] Dual AI Sell Monitor (DRY-RUN)
echo   Claude: 기술적 분석 + 포트폴리오
echo   GPT-4o: 뉴스 촉매 + 내일 전망
echo   합의규칙: 촉매 우선 + 수동매수 보호
echo ========================================
python -u -X utf8 scripts/sell_monitor.py --dry-run >> logs\schedule.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] [FAIL] sell_monitor 실패 (code=%ERRORLEVEL%) >> logs\schedule.log
)

echo [%date% %time%] BAT-E 완료 (매수+매도) >> logs\schedule.log
echo ========================================
echo [QM-E] 완료
echo ========================================
