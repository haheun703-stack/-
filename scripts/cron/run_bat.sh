#!/bin/bash
# Quantum Master BAT→Linux 크론 래퍼
# Usage: run_bat.sh <BAT_ID>
# KST 스케줄은 crontab에서 UTC로 변환하여 등록

set -uo pipefail
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

BAT="$1"
QM=/home/ubuntu/quantum-master
PY=$QM/venv/bin/python3.11
LOG=$QM/logs/cron_$(date +%Y%m%d).log
cd "$QM"
export PYTHONPATH="$QM"
set -a; source "$QM/.env" 2>/dev/null; set +a
mkdir -p logs

echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] === BAT-$BAT 시작 ===" >> "$LOG"

case "$BAT" in
  A) # 06:10 KST — 미장 마감
    $PY scripts/us_overnight_signal.py --update >> "$LOG" 2>&1 || true
    $PY scripts/update_us_kr_daily.py >> "$LOG" 2>&1 || true
    $PY scripts/fetch_cot_weekly.py >> "$LOG" 2>&1 || true
    $PY scripts/fetch_liquidity_data.py >> "$LOG" 2>&1 || true
    $PY scripts/run_liquidity_signal.py >> "$LOG" 2>&1 || true
    $PY scripts/run_relay_engine.py --update --signal >> "$LOG" 2>&1 || true
    $PY scripts/run_v3_brain.py --no-telegram >> "$LOG" 2>&1 || true
    $PY scripts/scan_tomorrow_picks.py >> "$LOG" 2>&1 || true
    ;;
  B) # 07:00 KST — 장전 브리핑
    $PY scripts/crawl_morning_reports.py >> "$LOG" 2>&1 || true
    $PY scripts/run_morning_briefing.py >> "$LOG" 2>&1 || true
    ;;
  K_safety) # 07:30 KST — 안전마진
    $PY scripts/scan_safety_margin.py >> "$LOG" 2>&1 || true
    ;;
  M_morning) # 08:00 KST — 모닝 브리핑
    $PY scripts/cron_morning_briefing.py >> "$LOG" 2>&1 || true
    ;;
  N) # 08:20 KST — 시그널 로그
    $PY scripts/cron_signal_tracker.py --mode log >> "$LOG" 2>&1 || true
    ;;
  E) # 08:50 KST — 스마트 진입
    $PY scripts/smart_entry_runner.py --live --force >> "$LOG" 2>&1 || true
    $PY scripts/sell_monitor.py --dry-run >> "$LOG" 2>&1 || true
    ;;
  I) # 08:55 KST — VWAP + EYE 킬러픽 모니터
    $PY scripts/intraday_eye.py --killer-picks >> "$LOG" 2>&1 &
    $PY scripts/run_vwap_monitor.py --killer-picks >> "$LOG" 2>&1 || true
    ;;
  H) # 11:30 KST — 장중 분석
    $PY scripts/run_midday_analysis.py >> "$LOG" 2>&1 || true
    ;;
  L) # 15:35 KST — NXT 장마감 후
    $PY scripts/nxt_market_collector.py --session after >> "$LOG" 2>&1 || true
    ;;
  O) # 16:10 KST — 시그널 트래킹
    $PY scripts/cron_signal_tracker.py --mode track >> "$LOG" 2>&1 || true
    ;;
  D) # 16:30 KST — 장마감 전체 파이프라인 (데이터 + 분석 + 추천)
    # --- G1: 데이터 수집 ---
    $PY scripts/update_daily_data.py >> "$LOG" 2>&1 || true
    $PY scripts/extend_parquet_data.py --workers 2 >> "$LOG" 2>&1 || true
    $PY scripts/update_kospi_index.py >> "$LOG" 2>&1 || true
    $PY scripts/collect_intraday_candles.py >> "$LOG" 2>&1 || true
    $PY scripts/us_overnight_signal.py --update >> "$LOG" 2>&1 || true
    $PY scripts/scan_nationality.py >> "$LOG" 2>&1 || true
    $PY scripts/collect_foreign_exhaustion.py >> "$LOG" 2>&1 || true
    $PY scripts/collect_short_selling.py >> "$LOG" 2>&1 || true
    # --- G2: 지표 + 릴레이 ---
    $PY scripts/rebuild_indicators.py >> "$LOG" 2>&1 || true
    $PY scripts/run_ict_levels.py >> "$LOG" 2>&1 || true
    $PY scripts/run_relay_engine.py --update --signal >> "$LOG" 2>&1 || true
    $PY scripts/relay_report.py >> "$LOG" 2>&1 || true
    # --- G3: 시그널 스캔 + BRAIN + SHIELD ---
    $PY scripts/scan_buy_candidates.py >> "$LOG" 2>&1 || true
    $PY scripts/run_shield.py --send >> "$LOG" 2>&1 || true
    $PY scripts/run_brain.py >> "$LOG" 2>&1 || true
    $PY scripts/run_v3_brain.py --no-telegram >> "$LOG" 2>&1 || true
    # --- G4: 추천 + FLOWX ---
    $PY scripts/scan_tomorrow_picks.py >> "$LOG" 2>&1 || true
    $PY scripts/scan_tomorrow_picks.py --flowx --no-send >> "$LOG" 2>&1 || true
    $PY scripts/upload_flowx.py >> "$LOG" 2>&1 || true
    $PY scripts/dashboard_data.py >> "$LOG" 2>&1 || true
    # --- G5: 기록 ---
    $PY scripts/market_journal.py >> "$LOG" 2>&1 || true
    ;;
  J) # 17:00 KST — 포트폴리오 전망
    $PY scripts/run_portfolio_outlook.py >> "$LOG" 2>&1 || true
    ;;
  *)
    echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] 알 수 없는 BAT: $BAT" >> "$LOG"
    ;;
esac

echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] === BAT-$BAT 완료 ===" >> "$LOG"
