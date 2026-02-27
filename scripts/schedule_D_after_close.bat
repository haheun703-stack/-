@echo off
REM ============================================================
REM  Quantum Master - BAT-D: 장마감 후 전체 데이터 수집 + 스캔
REM  스케줄: 매일 16:30 (월~금, 종가 확정 후)
REM  등록: schtasks /create /tn "QM_D_AfterClose" /tr "D:\sub-agent-project\scripts\schedule_D_after_close.bat" /sc daily /st 16:30
REM
REM  예상 소요: ~50분 (16:30 → 17:20)
REM  순서: 기초데이터 → 지표 → 분석 → DART+뉴스+스캔 → 추천 (23단계)
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

REM 11.5단계: 레버리지/인버스 ETF 로테이션 스캔
echo [%date% %time%] [11.5/24] 레버리지 ETF 스캔 >> logs\schedule.log
python -u -X utf8 scripts\leverage_etf_scanner.py >> logs\schedule.log 2>&1

REM 12단계: 눌림목 스캔 (건강한 조정 매수 후보)
echo [%date% %time%] [12/24] 눌림목 스캔 >> logs\schedule.log
python -u -X utf8 scripts\scan_pullback.py >> logs\schedule.log 2>&1

REM 12.5단계: 수급 폭발 → 조정 매수 스캐너 (전략 A)
echo [%date% %time%] [12.5/26] 수급 폭발 스캐너 >> logs\schedule.log
python -u -X utf8 scripts\scan_volume_spike.py >> logs\schedule.log 2>&1

REM 12.7단계: 밸류체인 릴레이 스캔 (대장주→소부장)
echo [%date% %time%] [12.7/28] 밸류체인 스캔 >> logs\schedule.log
python -u -X utf8 scripts\scan_value_chain.py >> logs\schedule.log 2>&1

REM 13단계: DART 전자공시 크롤링 (뉴스 선행 감지)
echo [%date% %time%] [13/23] DART 공시 크롤링 >> logs\schedule.log
python -u -X utf8 scripts\crawl_dart_disclosure.py >> logs\schedule.log 2>&1

REM 13.5단계: 레짐 전환 매크로 시그널
echo [%date% %time%] [13.5/26] 레짐 매크로 시그널 >> logs\schedule.log
python -u -X utf8 scripts\regime_macro_signal.py >> logs\schedule.log 2>&1

REM 14단계: 시장 뉴스 크롤링 (이벤트 레이더용)
echo [%date% %time%] [14/26] 시장 뉴스 크롤링 >> logs\schedule.log
python -u -X utf8 scripts\crawl_market_news.py >> logs\schedule.log 2>&1

REM 15단계: 세력감지 스캔 (기존 whale_detect)
echo [%date% %time%] [15/23] 세력감지 스캔 >> logs\schedule.log
python -u -X utf8 scripts\scan_whale_detect.py >> logs\schedule.log 2>&1

REM 16단계: 세력감지 하이브리드 (3-Layer 통합 + DART 연동)
echo [%date% %time%] [16/23] 세력감지 하이브리드 >> logs\schedule.log
python -u -X utf8 scripts\scan_force_hybrid.py >> logs\schedule.log 2>&1

REM 17단계: 동반매수 스캔 (외인+기관 동시매수)
echo [%date% %time%] [17/23] 동반매수 스캔 >> logs\schedule.log
python -u -X utf8 scripts\scan_dual_buying.py >> logs\schedule.log 2>&1

REM 17.5단계: 섹터 릴레이 트레이딩 시그널 (발화 섹터 → 종목 선정)
echo [%date% %time%] [17.5/24] 섹터 릴레이 시그널 >> logs\schedule.log
python -u -X utf8 scripts\relay_report.py >> logs\schedule.log 2>&1

REM 18단계: 그룹 릴레이 감지 (재벌 계열사 발화)
echo [%date% %time%] [18/28] 그룹 릴레이 감지 >> logs\schedule.log
python -u -X utf8 scripts\group_relay_detector.py >> logs\schedule.log 2>&1

REM 18.5단계: 매집 추적 스캔 (전략D — 거래량폭발 이후 매집 진행 종목)
echo [%date% %time%] [18.5/28] 매집 추적 스캔 >> logs\schedule.log
python -u -X utf8 scripts\scan_accumulation_tracker.py >> logs\schedule.log 2>&1

REM ══════════════════════════════════════════════
REM  PHASE 5: 성과추적 + 내일 추천 (~5분)
REM ══════════════════════════════════════════════

REM 19단계: 추천 성과 추적 (이전 추천 결과 판정)
echo [%date% %time%] [19/28] 추천 성과 추적 >> logs\schedule.log
python -u -X utf8 scripts\track_pick_results.py >> logs\schedule.log 2>&1

REM 19.3단계: DART 이벤트 드리븐 시그널 (추천 스캔 전에 생성해야 함!)
echo [%date% %time%] [19.3/28] DART 이벤트 시그널 >> logs\schedule.log
python -u -X utf8 scripts\dart_event_signal.py >> logs\schedule.log 2>&1

REM 19.5단계: 기관 추정 목표가 계산 (VPOC + 외인VWAP + 피보나치 + MA120)
echo [%date% %time%] [19.5/28] 기관 목표가 계산 >> logs\schedule.log
python -u -X utf8 scripts\calc_institutional_targets.py >> logs\schedule.log 2>&1

REM 19.6단계: 보유종목 동적 목표가 재판정
echo [%date% %time%] [19.6/28] 보유종목 재판정 >> logs\schedule.log
python -u -X utf8 scripts\position_monitor.py >> logs\schedule.log 2>&1

REM 19.7단계: Perplexity 시장 인텔리전스 (전략E — US 이벤트→KR 파급 분석)
echo [%date% %time%] [19.7/28] Perplexity 인텔리전스 >> logs\schedule.log
python -u -X utf8 scripts\perplexity_market_intel.py >> logs\schedule.log 2>&1

REM 19.8단계: AI 두뇌 뉴스 분석 (Claude API → 정성적 종목 판단)
echo [%date% %time%] [19.8/30] AI 두뇌 뉴스 분석 >> logs\schedule.log
python -u -X utf8 scripts\ai_news_brain.py >> logs\schedule.log 2>&1

REM 20단계: 내일 추천 종목 통합 스캔 (10개 시그널 교차검증) *** 최종 ***
echo [%date% %time%] [20/30] 내일 추천 종목 스캔 >> logs\schedule.log
python -u -X utf8 scripts\scan_tomorrow_picks.py >> logs\schedule.log 2>&1

REM 20.7단계: 멀티전략 포트폴리오 배분
echo [%date% %time%] [20.7/28] 포트폴리오 배분 >> logs\schedule.log
python -u -X utf8 scripts\portfolio_allocator.py >> logs\schedule.log 2>&1

REM ══════════════════════════════════════════════
REM  PHASE 6: 아카이브 + 보고서 (~1분)
REM ══════════════════════════════════════════════

REM 21단계: 일일 아카이브 (JSON → SQLite 영구 저장)
echo [%date% %time%] [21/28] 일일 아카이브 >> logs\schedule.log
python -u -X utf8 src\daily_archive.py >> logs\schedule.log 2>&1

REM 22단계: 주간 보고서 자동 생성 (금요일만)
for /f "tokens=1" %%a in ('python -c "from datetime import datetime; print(datetime.now().weekday())"') do set DOW=%%a
if "%DOW%"=="4" (
    echo [%date% %time%] [22/28] 주간 보고서 생성 (금요일) >> logs\schedule.log
    python -u -X utf8 src\daily_archive.py --weekly >> logs\schedule.log 2>&1
) else (
    echo [%date% %time%] [22/28] 주간 보고서 스킵 (금요일 아님) >> logs\schedule.log
)

REM 23단계: Railway 원격 동기화 (RAILWAY_URL 설정 시에만 실행)
if defined RAILWAY_URL (
    echo [%date% %time%] [23/28] Railway 동기화 >> logs\schedule.log
    python -u -X utf8 scripts\sync_to_railway.py >> logs\schedule.log 2>&1
) else (
    echo [%date% %time%] [23/28] Railway 동기화 스킵 (RAILWAY_URL 미설정) >> logs\schedule.log
)

REM 24단계: 저녁 통합 텔레그램 (보유종목+DART+추천+밸류체인 → 1건)
echo [%date% %time%] [24/29] 저녁 통합 리포트 >> logs\schedule.log
python -u -X utf8 scripts\send_evening_summary.py --send >> logs\schedule.log 2>&1

REM ══════════════════════════════════════════════
echo [%date% %time%] BAT-D 완료 (29단계 순차 실행) >> logs\schedule.log
echo [%date% %time%] ================================================== >> logs\schedule.log
