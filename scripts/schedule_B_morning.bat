@echo off
chcp 65001 >nul
REM ============================================================
REM  Quantum Master - BAT-B: 장전 통합 브리핑 (텔레그램 1건)
REM  스케줄: 매일 07:00 (월~금, 장 시작 전)
REM  등록: schtasks /create /tn "QM_B_Morning" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_B_morning.bat" /sc daily /st 07:00
REM
REM  변경 이력: 기존 3~4건 개별 발송 → 1건 통합 (2026-02-28)
REM  BAT-C (ETF 시그널) 기능을 흡수하여 BAT-C 폐지
REM ============================================================

echo [%date% %time%] BAT-B 시작: 장전 통합 브리핑 >> D:\sub-agent-project_퀀트봇\logs\schedule.log

call D:\sub-agent-project_퀀트봇\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_퀀트봇
set PYTHONPATH=D:\sub-agent-project_퀀트봇

REM ── 거래일 가드 (trading_calendar 사용) ──
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-B 스킵: 비거래일 >> logs\schedule.log
    goto :eof
)

REM 1단계: 증권사 리포트 스캔 (텔레그램 미발송, JSON만 저장)
echo [%date% %time%] [1/2] 증권사 리포트 스캔 >> logs\schedule.log
python -u -X utf8 scripts\crawl_morning_reports.py >> logs\schedule.log 2>&1

REM 2단계: 통합 브리핑 (테마+뉴스+브리핑+ETF → 텔레그램 1건)
echo [%date% %time%] [2/2] 통합 브리핑 실행 >> logs\schedule.log
python -u -X utf8 scripts\run_morning_briefing.py >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-B 완료 >> logs\schedule.log
