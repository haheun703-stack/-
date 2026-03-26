@echo off
REM ============================================================
REM  Quantum Master - BAT-A: 미장 마감 + 아침 재스캔 + 텔레그램
REM  스케줄: 매일 06:10 (월~토, 미장 마감 직후)
REM  등록: schtasks /create /tn "QM_A_USClose" /tr "D:\sub-agent-project\scripts\schedule_A_us_close.bat" /sc daily /st 06:10
REM
REM  [v3] 미장 데이터 반영 + 릴레이 경보 + 추천종목 재스캔 + 텔레그램
REM       → BAT-E(08:50)가 최신 추천으로 자동매수
REM ============================================================

echo [%date% %time%] BAT-A 시작: 미장 마감 + 릴레이 + 아침 재스캔 >> D:\sub-agent-project\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

REM ── 거래일 가드 (trading_calendar 사용) ──
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-A 스킵: 비거래일 >> logs\schedule.log
    goto :eof
)

REM ── PHASE 1: 미국장 데이터 업데이트 ──

REM 1) US 시장 데이터 업데이트 + Overnight Signal (원자재 포함)
echo [%date% %time%] [1/7] US Overnight Signal 업데이트 >> logs\schedule.log
python -u -X utf8 scripts\us_overnight_signal.py --update >> logs\schedule.log 2>&1

REM 2) US-KR 패턴DB 일일 누적
echo [%date% %time%] [2/7] US-KR 패턴DB 업데이트 >> logs\schedule.log
python -u -X utf8 scripts\update_us_kr_daily.py >> logs\schedule.log 2>&1

REM 2.5) COT 주간 업데이트 (매일 체크, 금요일에만 실질 변경)
echo [%date% %time%] [2.5/7] COT 주간 업데이트 >> logs\schedule.log
python -u -X utf8 scripts\fetch_cot_weekly.py >> logs\schedule.log 2>&1

REM 2.7) 유동성 사이클 데이터 업데이트 (FRED 5대 지표)
echo [%date% %time%] [2.7/7] 유동성 데이터 업데이트 >> logs\schedule.log
python -u -X utf8 scripts\fetch_liquidity_data.py >> logs\schedule.log 2>&1

REM 2.8) 유동성 시그널 생성
echo [%date% %time%] [2.8/7] 유동성 시그널 생성 >> logs\schedule.log
python -u -X utf8 scripts\run_liquidity_signal.py >> logs\schedule.log 2>&1

REM 3) 섹터 릴레이 엔진 (US 대장주 업데이트 + 경보 판정, 텔레그램은 아침 통합에 흡수)
REM v13: --all → --update --signal (릴레이 별도 알림 제거, BAT-B 아침 브리핑에서 통합)
echo [%date% %time%] [3/7] 섹터 릴레이 데이터 업데이트 >> logs\schedule.log
python -u -X utf8 scripts\run_relay_engine.py --update --signal >> logs\schedule.log 2>&1

REM 2.9) 원자재 가격 수집 + 원가 갭 분석
echo [%date% %time%] [2.9/9] 원자재 가격 + 원가 갭 수집 >> logs\schedule.log
python -u -X utf8 scripts\fetch_commodity_prices.py >> logs\schedule.log 2>&1

REM 2.95) 시나리오 뉴스 엔진 (뉴스 수집 + 시나리오 평가 + 매수신호)
echo [%date% %time%] [2.95/9] 시나리오 뉴스 엔진 >> logs\schedule.log
python -u -X utf8 scripts\news_scenario_engine.py --no-sentiment >> logs\schedule.log 2>&1

REM ── PHASE 2: 아침 재스캔 (미국장 반영) ──

REM 4) v3 AI Brain 재실행 (미장 데이터 반영 → ai_v3_picks.json 갱신)
echo [%date% %time%] [4/7] v3 AI Brain 아침 재스캔 >> logs\schedule.log
python -u -X utf8 scripts\run_v3_brain.py --no-telegram >> logs\schedule.log 2>&1
if errorlevel 1 echo [%date% %time%] v3 Brain 실패 (기존 picks 유지) >> logs\schedule.log

REM 5) 추천종목 재스캔 (overnight_signal + v3 picks 반영)
echo [%date% %time%] [5/7] 추천종목 아침 재스캔 >> logs\schedule.log
python -u -X utf8 scripts\scan_tomorrow_picks.py >> logs\schedule.log 2>&1

REM 6) 아침 텔레그램 — BAT-B (07:00) 통합 브리핑에서 1건으로 발송
REM v13: BAT-A에서는 데이터만 준비, BAT-B에서 증권사+테마+릴레이 통합 1건 발송
echo [%date% %time%] [6/7] 아침 텔레그램은 BAT-B(07:00)에서 통합 발송 >> logs\schedule.log

echo [%date% %time%] BAT-A 완료 (9단계) >> logs\schedule.log
