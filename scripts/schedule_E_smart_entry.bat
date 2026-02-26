@echo off
REM ============================================================
REM  Quantum Master - BAT-E: AI 스마트 진입 (LIVE)
REM  스케줄: 매일 08:50 (월~금, 장 시작 10분 전)
REM  등록: schtasks /create /tn "QM_E_SmartEntry" /tr "D:\sub-agent-project\scripts\schedule_E_smart_entry.bat" /sc daily /st 08:50
REM
REM  안전장치:
REM    - max_stocks=1 (TOP1만), max_amount=100만원
REM    - 킬스위치: data/KILL_SWITCH 파일 생성 시 즉시 중단
REM    - 당일 중복 실행 방지 (order_audit.db)
REM    - 모든 주문 감사 로그 기록
REM ============================================================

echo [%date% %time%] BAT-E 시작: AI 스마트 진입 (LIVE) >> D:\sub-agent-project\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

echo ========================================
echo [QM-E] AI 스마트 진입 LIVE (08:50~10:30)
echo   max_stocks=1, max_amount=100만원
echo   킬스위치: data/KILL_SWITCH
echo ========================================

echo [%date% %time%] SmartEntry LIVE 실행 >> logs\schedule.log
python -u -X utf8 scripts/smart_entry_runner.py --live --force >> logs\schedule.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] [FAIL] smart_entry_runner 실패 (code=%ERRORLEVEL%) >> logs\schedule.log
)

echo [%date% %time%] BAT-E 완료 >> logs\schedule.log
echo ========================================
echo [QM-E] 완료
echo ========================================
