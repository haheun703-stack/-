@echo off
chcp 65001 >nul
REM ============================================================
REM  Quantum Master - BAT-J: 장마감 포트폴리오 방향성 판단
REM  스케줄: 매일 17:00 (월~금, 장마감 후 데이터 안정화)
REM  등록: schtasks /create /tn "QM_J_Outlook" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_J_portfolio_outlook.bat" /sc daily /st 17:00
REM
REM  보유 종목별 내일 방향성 예측 (↑→↓) + 행동 추천 (HOLD/ADD/TRIM/SELL)
REM  결과: 텔레그램 전송 + data/portfolio_outlook.json 저장
REM ============================================================

echo [%date% %time%] BAT-J 시작: 포트폴리오 방향성 판단 >> D:\sub-agent-project_퀀트봇\logs\schedule.log

call D:\sub-agent-project_퀀트봇\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_퀀트봇
set PYTHONPATH=D:\sub-agent-project_퀀트봇

REM ── 거래일 가드 (trading_calendar 사용) ──
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-J 스킵: 비거래일 >> logs\schedule.log
    goto :eof
)

echo ========================================
echo [QM-J] 포트폴리오 방향성 판단 (17:00)
echo   보유 종목별 내일 방향성 예측
echo   AI 종합 분석 + 텔레그램 전송
echo ========================================

python -u -X utf8 scripts/run_portfolio_outlook.py >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-J 완료 >> logs\schedule.log
