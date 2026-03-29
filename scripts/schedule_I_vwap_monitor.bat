@echo off
chcp 65001 >nul
REM ============================================================
REM  Quantum Master - BAT-I: 장중 VWAP 모니터 + AI EYE
REM  스케줄: 평일 08:55 (월~금, 비 거래일 자동 스킵)
REM  등록: schtasks /create /tn "QM_I_VWAP" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_I_vwap_monitor.bat" /sc daily /st 08:55
REM
REM  08:55 시작 → 09:00 장 시작 → 09:30 VWAP 기준선 → 14:00 종료
REM  VWAP 눌림/회복 알림 + 11:30 AI 분석 통합
REM  --killer-picks: 킬러픽(교차검증 TOP + ETF) 자동 로딩
REM  장중 약 5시간 실행 (long-running)
REM ============================================================

echo [%date% %time%] BAT-I 시작: VWAP+EYE 킬러픽 모니터 >> D:\sub-agent-project_퀀트봇\logs\schedule.log

call D:\sub-agent-project_퀀트봇\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_퀀트봇
set PYTHONPATH=D:\sub-agent-project_퀀트봇

REM 비거래일 체크 (trading_calendar 기반) 스킵
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-I 스킵: 비거래일 >> logs\schedule.log
    goto :eof
)

echo ========================================
echo [QM-I] VWAP + AI EYE 킬러픽 모니터 (08:55~14:00)
echo   09:00 개장 갭 분석
echo   09:30 VWAP 기준선 확정
echo   11:30 AI 분석 통합
echo   14:00 종료
echo ========================================

REM Intraday EYE 백그라운드 실행 (킬러픽 워치리스트 포함)
start /b python -u -X utf8 scripts/intraday_eye.py --killer-picks >> logs\schedule.log 2>&1

REM VWAP 모니터 포그라운드 실행 (킬러픽 종목)
python -u -X utf8 scripts/run_vwap_monitor.py --killer-picks >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-I 완료 >> logs\schedule.log
