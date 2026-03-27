@echo off
chcp 65001 >nul
REM ============================================================
REM  Quantum Master - BAT-I: 장중 VWAP 모니터
REM  스케줄: 매일 08:55 (월~금, 장 시작 5분 전)
REM  등록: schtasks /create /tn "QM_I_VWAP" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_I_vwap_monitor.bat" /sc daily /st 08:55
REM
REM  09:00 개장 갭 → 09:30 VWAP 기준선 → 14:00까지 모니터링
REM  VWAP 눌림/회복 알림 + 11:30 AI 분석 통합
REM  장중 약 5시간 실행 (long-running)
REM ============================================================

echo [%date% %time%] BAT-I 시작: VWAP 모니터 >> D:\sub-agent-project_퀀트봇\logs\schedule.log

call D:\sub-agent-project_퀀트봇\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_퀀트봇
set PYTHONPATH=D:\sub-agent-project_퀀트봇

REM ── 거래일 가드 (trading_calendar 사용) ──
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-I 스킵: 비거래일 >> logs\schedule.log
    goto :eof
)

REM ── 자동매매 OFF 기간 VWAP 비활성화 (2026-03-20) ──
REM 자동매매 재개 시 아래 주석 해제
REM 이유: dry_run=True 상태에서 5~15건/일 텔레그램 노이즈
echo [%date% %time%] BAT-I 스킵: 자동매매 OFF 기간 VWAP 비활성화 >> logs\schedule.log
goto :eof

echo ========================================
echo [QM-I] VWAP 모니터 (08:55~14:00)
echo   09:00 개장 갭 분석
echo   09:30 VWAP 기준선 설정
echo   11:30 AI 분석 통합
echo   14:00 종료
echo ========================================

python -u -X utf8 scripts/run_vwap_monitor.py >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-I 완료 >> logs\schedule.log
