@echo off
REM ============================================================
REM  Quantum Master - BAT-D: COO Orchestrator 래핑 버전
REM  스케줄: 매일 16:30 (월~금, 종가 확정 후)
REM  등록: schtasks /create /tn "QM_D_AfterClose" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_D_after_close.bat" /sc daily /st 16:30
REM
REM  [v2] 기존 31단계 순차 실행 → COO Orchestrator 위임
REM       원본 백업: scripts\schedule_D_original.bat
REM       COO: 7그룹 66단계, 폴백 핸들러, FLOWX 보장
REM ============================================================

echo [%date% %time%] ================================================== >> D:\sub-agent-project_퀀트봇\logs\schedule.log
echo [%date% %time%] BAT-D 시작: COO Orchestrator >> D:\sub-agent-project_퀀트봇\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project_퀀트봇\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_퀀트봇
set PYTHONPATH=D:\sub-agent-project_퀀트봇

if not exist logs mkdir logs

REM ── 로그 로테이션 (10MB 초과 시 백업) ──
for %%F in (logs\schedule.log) do (
    if %%~zF GTR 10000000 (
        echo [%date% %time%] 로그 로테이션: %%~zF bytes >> logs\schedule.log
        copy /Y logs\schedule.log logs\schedule.log.old >nul 2>&1
        echo [%date% %time%] BAT-D 시작 (로테이션 후) > logs\schedule.log
    )
)

REM ── 거래일 가드 (trading_calendar 사용 — 주말+공휴일 체크) ──
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-D 스킵: 비거래일 >> logs\schedule.log
    goto :eof
)

REM ── COO Orchestrator 실행 (7그룹 66단계) ──
echo [%date% %time%] COO Orchestrator 시작 >> logs\schedule.log
python -u -X utf8 coo_orchestrator.py >> logs\coo_bat.log 2>&1
if errorlevel 1 (
    echo [%date% %time%] COO 실패 — 원본 BAT-D 폴백 실행 >> logs\schedule.log
    call scripts\schedule_D_original.bat
    goto :eof
)

REM ── FLOWX 공개추천 스캔 (COO 완료 후) ──
echo [%date% %time%] FLOWX 공개추천 스캔 시작 >> logs\schedule.log
python -u -X utf8 scripts\scan_tomorrow_picks.py --flowx >> logs\schedule.log 2>&1
echo [%date% %time%] FLOWX 업로드 >> logs\schedule.log
python -u -X utf8 scripts\upload_flowx.py >> logs\schedule.log 2>&1

REM ── 금요일 주간 보고서 (COO에 미포함) ──
for /f "tokens=1" %%a in ('python -c "from datetime import datetime; print(datetime.now().weekday())"') do set DOW=%%a
if "%DOW%"=="4" (
    echo [%date% %time%] 금요일 주간 보고서 >> logs\schedule.log
    python -u -X utf8 src\daily_archive.py --weekly >> logs\schedule.log 2>&1
    python -u -X utf8 scripts\run_v3_brain.py --weekly-review >> logs\schedule.log 2>&1
    python -u -X utf8 scripts\paper_trading_unified.py --weekly >> logs\schedule.log 2>&1
)

echo [%date% %time%] BAT-D 완료 (COO Orchestrator) >> logs\schedule.log
echo [%date% %time%] ================================================== >> logs\schedule.log
