@echo off
REM ============================================================
REM  Quantum Master - BAT-D PARALLEL: 병렬 파이프라인 실행
REM  기존 schedule_D_after_close.bat의 병렬 버전
REM
REM  효과: 50분 → 25~30분 (의존성 DAG 기반 병렬 실행)
REM
REM  등록: schtasks /create /tn "QM_D_Parallel" /tr "D:\sub-agent-project\scripts\schedule_D_parallel.bat" /sc daily /st 16:30
REM  (기존 QM_D_AfterClose와 동시 등록 X — 하나만 선택)
REM ============================================================

echo [%date% %time%] ================================================== >> D:\sub-agent-project\logs\schedule.log
echo [%date% %time%] BAT-D PARALLEL 시작 >> D:\sub-agent-project\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

if not exist logs mkdir logs

REM ── 거래일 가드 (trading_calendar 사용 — 주말+공휴일 체크) ──
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-D 스킵: 비거래일 >> logs\schedule.log
    goto :eof
)

REM ── 병렬 파이프라인 실행 ──
echo [%date% %time%] 병렬 파이프라인 시작 (max-workers=6) >> logs\schedule.log
python -u -X utf8 scripts\parallel_pipeline.py --max-workers 6 >> logs\schedule.log 2>&1
if errorlevel 1 echo [%date% %time%] 병렬 파이프라인 일부 실패 >> logs\schedule.log

REM ── 데이터 무결성 체크 ──
echo [%date% %time%] 데이터 건강검진 >> logs\schedule.log
python -u -X utf8 scripts\data_health_check.py >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-D PARALLEL 완료 >> logs\schedule.log
echo [%date% %time%] ================================================== >> logs\schedule.log
