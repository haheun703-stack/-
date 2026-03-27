@echo off
chcp 65001 >nul
REM ============================================================
REM  Quantum Master - BAT-K: INTRADAY EYE 장중 실시간 감시
REM  스케줄: 매일 08:55 (월~금)
REM  등록: schtasks /create /tn "QM_K_IntradayEye" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_K_intraday_eye.bat" /sc daily /st 08:55
REM
REM  09:05~15:20 장중 5분 간격 모니터링 (long-running)
REM  EYE-01~07 감지기 → 이벤트 발생 시 텔레그램 알림 (0~3건/일)
REM  거래일 체크: trading_calendar.py 사용
REM ============================================================

echo [%date% %time%] BAT-K 시작: INTRADAY EYE >> D:\sub-agent-project_퀀트봇\logs\schedule.log

call D:\sub-agent-project_퀀트봇\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_퀀트봇
set PYTHONPATH=D:\sub-agent-project_퀀트봇

REM ── 거래일 가드 (trading_calendar 사용) ──
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-K 스킵: 비거래일 >> logs\schedule.log
    goto :eof
)

echo ========================================
echo [QM-K] INTRADAY EYE (08:55~15:20)
echo   7개 감지기 장중 모니터링
echo   EYE-02 급락 / EYE-05 시장급변 우선
echo   이벤트 기반 텔레그램 알림
echo ========================================

python -u -X utf8 scripts/intraday_eye.py >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-K 완료 >> logs\schedule.log
