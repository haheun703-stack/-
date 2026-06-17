@echo off
REM ============================================================================
REM  Quantum Master — 로컬 일일 기초데이터 자동 갱신
REM  서버 BAT-D 보조용(서버 6/10 stale + KRX 차단 대비). 장 마감 후 실행.
REM  Windows Task Scheduler 등록: 평일 16:10 KST (15:30 종가 + 버퍼).
REM  ★ collect_investor_flow는 --full 금지(이력 잘림). 증분만(장마감후 1회=종가).
REM ============================================================================
setlocal
set ROOT=D:\sub-agent-project_퀀트봇
set PYTHONPATH=%ROOT%
set PY=%ROOT%\venv\Scripts\python.exe
set LOG=%ROOT%\logs\daily_local_update.log

cd /d %ROOT%
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"

echo. >> "%LOG%"
echo ========================================= >> "%LOG%"
echo [START] %DATE% %TIME% >> "%LOG%"
echo ========================================= >> "%LOG%"

REM --- 1) 종가 parquet 증분(raw, fdr+KIS·KRX독립) ---
echo [1/6] extend_parquet_data >> "%LOG%"
"%PY%" -u -X utf8 scripts\extend_parquet_data.py >> "%LOG%" 2>&1

REM --- 2) 지표 재계산(raw -> processed) ---
echo [2/6] rebuild_indicators >> "%LOG%"
"%PY%" -u -X utf8 scripts\rebuild_indicators.py >> "%LOG%" 2>&1

REM --- 3) 수급 db 3주체(KIS FHKST01010900·KRX독립) 최근 2거래일 ---
echo [3/6] collect_investor_kis --days 2 >> "%LOG%"
"%PY%" -u -X utf8 scripts\collect_investor_kis.py --days 2 >> "%LOG%" 2>&1

REM --- 4) kospi 시장수급 csv(네이버·증분, --full 금지) ---
echo [4/6] collect_investor_flow >> "%LOG%"
"%PY%" -u -X utf8 scripts\collect_investor_flow.py >> "%LOG%" 2>&1

REM --- 5) 종목별 차트 csv 동기화(db -> 종목별 CSV) ---
echo [5/6] sync_investor_to_csv >> "%LOG%"
"%PY%" -u -X utf8 scripts\sync_investor_to_csv.py >> "%LOG%" 2>&1

REM --- 6) 밸류밴드 KR+US Supabase 적재(관측·UPSERT 멱등) ---
echo [6/6] upload_valuation_band --market ALL --write >> "%LOG%"
"%PY%" -u -X utf8 scripts\upload_valuation_band.py --market ALL --write >> "%LOG%" 2>&1

echo [DONE] %DATE% %TIME% >> "%LOG%"
endlocal
