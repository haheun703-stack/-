@echo off
REM Quantum Master - 매일 17:00 자동 데이터 업데이트
REM Windows 작업 스케줄러에 등록:
REM   schtasks /create /tn "QuantumMaster_DailyUpdate" /tr "D:\sub-agent-project\scripts\daily_auto_update.bat" /sc daily /st 17:00

echo [%date% %time%] 일일 데이터 업데이트 시작 >> D:\sub-agent-project\logs\auto_update.log

call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project

REM 1단계: CSV 2,859종목 업데이트 (FDR)
echo [%date% %time%] CSV 업데이트 시작 >> logs\auto_update.log
python scripts\update_daily_data.py >> logs\auto_update.log 2>&1

REM 2단계: Parquet 101종목 증분 업데이트 (pykrx)
echo [%date% %time%] Parquet 업데이트 시작 >> logs\auto_update.log
python scripts\extend_parquet_data.py >> logs\auto_update.log 2>&1

REM 3단계: 수급 데이터 수집
echo [%date% %time%] 수급 데이터 수집 >> logs\auto_update.log
python scripts\collect_supply_data.py >> logs\auto_update.log 2>&1

REM 4단계: 기술지표 재계산 (raw -> processed parquet) *** 필수 ***
REM   2단계에서 raw가 업데이트됐으므로 processed에 지표를 재생성해야
REM   다음날 스캔에서 최신 종가 기반 지표를 사용할 수 있음
echo [%date% %time%] 지표 재계산 시작 (raw -> processed) >> logs\auto_update.log
python -c "from src.indicators import IndicatorEngine; e = IndicatorEngine(); e.process_all()" >> logs\auto_update.log 2>&1

REM 5단계: US 시장 데이터 업데이트 + Overnight Signal 생성
echo [%date% %time%] US Overnight Signal 업데이트 >> logs\auto_update.log
python scripts\us_overnight_signal.py --update >> logs\auto_update.log 2>&1

REM 6단계: US-KR 학습 루프 (패턴매칭 DB 일일 누적)
echo [%date% %time%] US-KR 패턴DB 업데이트 >> logs\auto_update.log
python scripts\update_us_kr_daily.py >> logs\auto_update.log 2>&1

REM 7단계: 섹터 ETF 시세 업데이트
echo [%date% %time%] 섹터 ETF 시세 업데이트 >> logs\auto_update.log
python scripts\sector_etf_builder.py --daily >> logs\auto_update.log 2>&1

REM 8단계: 섹터 모멘텀 + z-score + 수급 + 통합 리포트
echo [%date% %time%] 섹터 순환매 분석 >> logs\auto_update.log
python scripts\sector_momentum.py --history >> logs\auto_update.log 2>&1
python scripts\sector_zscore.py --top 5 >> logs\auto_update.log 2>&1
python scripts\sector_investor_flow.py --days 5 >> logs\auto_update.log 2>&1
python scripts\sector_daily_report.py >> logs\auto_update.log 2>&1

REM 9단계: ETF 마스터 데이터 빌드 (수급 + 지표 + 추천점수)
echo [%date% %time%] ETF 마스터 데이터 빌드 >> logs\auto_update.log
python scripts\update_etf_master.py >> logs\auto_update.log 2>&1

REM 10단계: 눌림목 스캔 (전종목 건강한 조정 매수)
echo [%date% %time%] 눌림목 스캔 >> logs\auto_update.log
python scripts\scan_pullback.py >> logs\auto_update.log 2>&1

REM 11단계: 세력감지 스캔 (전종목 이상패턴 탐지)
echo [%date% %time%] 세력감지 스캔 >> logs\auto_update.log
python scripts\scan_whale_detect.py >> logs\auto_update.log 2>&1

REM 12단계: ETF 매매 시그널 생성 + 텔레그램 발송
echo [%date% %time%] ETF 매매 시그널 생성 >> logs\auto_update.log
python scripts\etf_trading_signal.py >> logs\auto_update.log 2>&1

REM 13단계: 추천 성과 추적 (이전 추천 결과 판정 + 보유중 업데이트)
echo [%date% %time%] 추천 성과 추적 >> logs\auto_update.log
python scripts\track_pick_results.py >> logs\auto_update.log 2>&1

REM 14단계: 내일 추천 종목 통합 스캔 (5개 시그널 교차검증)
echo [%date% %time%] 내일 추천 종목 스캔 >> logs\auto_update.log
python scripts\scan_tomorrow_picks.py >> logs\auto_update.log 2>&1

echo [%date% %time%] 일일 데이터 업데이트 완료 >> logs\auto_update.log
