@echo off
REM ============================================================
REM  Quantum Master - BAT-D: 장마감 후 전체 데이터 수집 + 스캔
REM  스케줄: 매일 16:30 (월~금, 종가 확정 후)
REM  등록: schtasks /create /tn "QM_D_AfterClose" /tr "D:\sub-agent-project\scripts\schedule_D_after_close.bat" /sc daily /st 16:30
REM
REM  예상 소요: ~50분 (16:30 → 17:20)
REM  순서: 기초데이터 → 지표 → 분석 → 뉴스+스캔 → 추천 (22단계)
REM ============================================================

echo [%date% %time%] ================================================== >> D:\sub-agent-project\logs\schedule.log
echo [%date% %time%] BAT-D 시작: 장마감 전체 데이터 수집 >> D:\sub-agent-project\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

if not exist logs mkdir logs

REM ══════════════════════════════════════════════
REM  PHASE 1: 기초 데이터 수집 (~15분)
REM ══════════════════════════════════════════════

REM 1단계: CSV 전종목 종가 업데이트 (FinanceDataReader)
echo [%date% %time%] [1/20] CSV 전종목 업데이트 >> logs\schedule.log
python -u -X utf8 scripts\update_daily_data.py >> logs\schedule.log 2>&1

REM 2단계: Parquet 유니버스 증분 업데이트 (pykrx)
echo [%date% %time%] [2/20] Parquet 증분 업데이트 >> logs\schedule.log
python -u -X utf8 scripts\extend_parquet_data.py >> logs\schedule.log 2>&1

REM 3단계: 수급 데이터 수집 (외인/기관 매매동향)
echo [%date% %time%] [3/20] 수급 데이터 수집 >> logs\schedule.log
python -u -X utf8 scripts\collect_supply_data.py >> logs\schedule.log 2>&1

REM 4단계: KOSPI 인덱스 업데이트 (레짐 판별용)
echo [%date% %time%] [4/20] KOSPI 인덱스 업데이트 >> logs\schedule.log
python -u -X utf8 scripts\update_kospi_index.py >> logs\schedule.log 2>&1

REM ══════════════════════════════════════════════
REM  PHASE 2: 지표 계산 (~10분)
REM ══════════════════════════════════════════════

REM 5단계: 기술지표 재계산 (raw -> processed parquet, 35개 지표)
echo [%date% %time%] [5/20] 기술지표 재계산 >> logs\schedule.log
python -u -X utf8 scripts\rebuild_indicators.py >> logs\schedule.log 2>&1

REM 6단계: US 시장 데이터 + Overnight Signal 갱신
echo [%date% %time%] [6/20] US Overnight Signal >> logs\schedule.log
python -u -X utf8 scripts\us_overnight_signal.py --update >> logs\schedule.log 2>&1

REM 7단계: US-KR 패턴매칭 DB 일일 누적
echo [%date% %time%] [7/20] US-KR 패턴DB >> logs\schedule.log
python -u -X utf8 scripts\update_us_kr_daily.py >> logs\schedule.log 2>&1

REM ══════════════════════════════════════════════
REM  PHASE 3: 섹터 + ETF 분석 (~10분)
REM ══════════════════════════════════════════════

REM 8단계: 섹터 ETF 시세 업데이트
echo [%date% %time%] [8/20] 섹터 ETF 시세 >> logs\schedule.log
python -u -X utf8 scripts\sector_etf_builder.py --daily >> logs\schedule.log 2>&1

REM 9단계: 섹터 모멘텀 + z-score + 수급 + 통합 리포트
echo [%date% %time%] [9/20] 섹터 순환매 분석 >> logs\schedule.log
python -u -X utf8 scripts\sector_momentum.py --history >> logs\schedule.log 2>&1
python -u -X utf8 scripts\sector_zscore.py --top 5 >> logs\schedule.log 2>&1
python -u -X utf8 scripts\sector_investor_flow.py --days 5 >> logs\schedule.log 2>&1
python -u -X utf8 scripts\sector_daily_report.py >> logs\schedule.log 2>&1

REM 10단계: ETF 마스터 데이터 빌드 (수급 + 지표 + 추천점수)
echo [%date% %time%] [10/20] ETF 마스터 빌드 >> logs\schedule.log
python -u -X utf8 scripts\update_etf_master.py >> logs\schedule.log 2>&1

REM 11단계: ETF 매매 시그널 생성 (다음날 08:00 텔레그램용)
echo [%date% %time%] [11/20] ETF 매매 시그널 >> logs\schedule.log
python -u -X utf8 scripts\etf_trading_signal.py >> logs\schedule.log 2>&1

REM ══════════════════════════════════════════════
REM  PHASE 4: 종목 스캔 (~10분)
REM ══════════════════════════════════════════════

REM 12단계: 눌림목 스캔 (건강한 조정 매수 후보)
echo [%date% %time%] [12/20] 눌림목 스캔 >> logs\schedule.log
python -u -X utf8 scripts\scan_pullback.py >> logs\schedule.log 2>&1

REM 13단계: 시장 뉴스 크롤링 (이벤트 레이더용)
echo [%date% %time%] [13/22] 시장 뉴스 크롤링 >> logs\schedule.log
python -u -X utf8 scripts\crawl_market_news.py >> logs\schedule.log 2>&1

REM 14단계: 세력감지 스캔 (기존 whale_detect)
echo [%date% %time%] [14/22] 세력감지 스캔 >> logs\schedule.log
python -u -X utf8 scripts\scan_whale_detect.py >> logs\schedule.log 2>&1

REM 15단계: 세력감지 하이브리드 (3-Layer 통합)
echo [%date% %time%] [15/22] 세력감지 하이브리드 >> logs\schedule.log
python -u -X utf8 scripts\scan_force_hybrid.py >> logs\schedule.log 2>&1

REM 16단계: 동반매수 스캔 (외인+기관 동시매수)
echo [%date% %time%] [16/22] 동반매수 스캔 >> logs\schedule.log
python -u -X utf8 scripts\scan_dual_buying.py >> logs\schedule.log 2>&1

REM 17단계: 그룹 릴레이 감지 (재벌 계열사 발화)
echo [%date% %time%] [17/22] 그룹 릴레이 감지 >> logs\schedule.log
python -u -X utf8 scripts\group_relay_detector.py >> logs\schedule.log 2>&1

REM ══════════════════════════════════════════════
REM  PHASE 5: 성과추적 + 내일 추천 (~5분)
REM ══════════════════════════════════════════════

REM 18단계: 추천 성과 추적 (이전 추천 결과 판정)
echo [%date% %time%] [18/22] 추천 성과 추적 >> logs\schedule.log
python -u -X utf8 scripts\track_pick_results.py >> logs\schedule.log 2>&1

REM 19단계: 내일 추천 종목 통합 스캔 (5개 시그널 교차검증) *** 최종 ***
echo [%date% %time%] [19/22] 내일 추천 종목 스캔 >> logs\schedule.log
python -u -X utf8 scripts\scan_tomorrow_picks.py >> logs\schedule.log 2>&1

REM ══════════════════════════════════════════════
REM  PHASE 6: 아카이브 + 보고서 (~1분)
REM ══════════════════════════════════════════════

REM 20단계: 일일 아카이브 (JSON → SQLite 영구 저장)
echo [%date% %time%] [20/22] 일일 아카이브 >> logs\schedule.log
python -u -X utf8 src\daily_archive.py >> logs\schedule.log 2>&1

REM 21단계: 주간 보고서 자동 생성 (금요일만)
for /f "tokens=1" %%a in ('python -c "from datetime import datetime; print(datetime.now().weekday())"') do set DOW=%%a
if "%DOW%"=="4" (
    echo [%date% %time%] [21/22] 주간 보고서 생성 (금요일) >> logs\schedule.log
    python -u -X utf8 src\daily_archive.py --weekly >> logs\schedule.log 2>&1
) else (
    echo [%date% %time%] [21/22] 주간 보고서 스킵 (금요일 아님) >> logs\schedule.log
)

REM 22단계: Railway 원격 동기화 (RAILWAY_URL 설정 시에만 실행)
if defined RAILWAY_URL (
    echo [%date% %time%] [22/22] Railway 동기화 >> logs\schedule.log
    python -u -X utf8 scripts\sync_to_railway.py >> logs\schedule.log 2>&1
) else (
    echo [%date% %time%] [22/22] Railway 동기화 스킵 (RAILWAY_URL 미설정) >> logs\schedule.log
)

REM ══════════════════════════════════════════════
echo [%date% %time%] BAT-D 완료 (22단계 순차 실행) >> logs\schedule.log
echo [%date% %time%] ================================================== >> logs\schedule.log
