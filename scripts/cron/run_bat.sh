#!/bin/bash
# Quantum Master BAT→Linux 크론 래퍼
# Usage: run_bat.sh <BAT_ID>
# 시스템 타임존: KST (Asia/Seoul) — crontab도 KST 기준

set -o pipefail  # -u 제거: cron 환경에서 미정의 변수로 인한 즉시종료 방지
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

BAT="${1:-UNKNOWN}"
QM=/home/ubuntu/quantum-master
PY=$QM/venv/bin/python3.11
LOG=$QM/logs/cron_$(date +%Y%m%d).log
FAIL_COUNT=0
cd "$QM" || exit 1
export PYTHONPATH="$QM"
set -a; source "$QM/.env" 2>/dev/null; set +a
mkdir -p logs stock_data_daily

# 실패 카운터 함수: 스크립트 실패 시 카운트 증가
run_py() {
  local script="$1"; shift
  $PY "$script" "$@" >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[$(date +%H:%M:%S)] [WARN] $script 실패 (exit=$rc)" >> "$LOG"
  fi
  return 0  # || true 불필요 — 항상 성공 반환
}

# 텔레그램 실패 알림 함수
send_fail_alert() {
  local msg="$1"
  local token="${TELEGRAM_BOT_TOKEN:-}"
  local chat="${TELEGRAM_CHAT_ID:-}"
  if [ -n "$token" ] && [ -n "$chat" ]; then
    curl -s -X POST "https://api.telegram.org/bot${token}/sendMessage" \
      -d chat_id="$chat" -d parse_mode=HTML \
      -d text="$msg" > /dev/null 2>&1 || true
  fi
}

echo "[$(date +%H:%M:%S)] === BAT-$BAT 시작 ===" >> "$LOG"

case "$BAT" in
  A) # 06:10 KST — 미장 마감
    run_py scripts/us_overnight_signal.py --update
    run_py scripts/update_us_kr_daily.py
    run_py scripts/fetch_cot_weekly.py
    run_py scripts/fetch_liquidity_data.py
    run_py scripts/run_liquidity_signal.py
    run_py scripts/run_relay_engine.py --update --signal
    run_py scripts/run_v3_brain.py --no-telegram
    run_py scripts/scan_tomorrow_picks.py
    ;;
  B) # 07:00 KST — 장전 브리핑
    run_py scripts/crawl_morning_reports.py
    run_py scripts/run_morning_briefing.py
    ;;
  K_safety) # 07:30 KST — 안전마진
    run_py scripts/scan_safety_margin.py
    ;;
  M_morning) # 08:00 KST — 모닝 브리핑
    run_py scripts/cron_morning_briefing.py
    ;;
  N) # 08:20 KST — 시그널 로그
    run_py scripts/cron_signal_tracker.py --mode log
    ;;
  E) # 08:50 KST — 스마트 진입
    run_py scripts/smart_entry_runner.py --live --force
    run_py scripts/sell_monitor.py --dry-run
    ;;
  I) # 08:55 KST — VWAP + EYE 킬러픽 모니터
    $PY scripts/intraday_eye.py --killer-picks >> "$LOG" 2>&1 &
    run_py scripts/run_vwap_monitor.py --killer-picks
    ;;
  H) # 11:30 KST — 장중 분석
    run_py scripts/run_midday_analysis.py
    ;;
  L) # 15:35 KST — NXT 장마감 후
    run_py scripts/nxt_market_collector.py --session after
    ;;
  O) # 16:10 KST — 시그널 트래킹
    run_py scripts/cron_signal_tracker.py --mode track
    ;;
  D) # 16:30 KST — 장마감 전체 파이프라인 (데이터 + 분석 + 추천)
    # --- G1: 데이터 수집 ---
    run_py scripts/update_daily_data.py
    run_py scripts/extend_parquet_data.py --workers 2
    run_py scripts/update_kospi_index.py
    run_py scripts/collect_intraday_candles.py
    run_py scripts/us_overnight_signal.py --update
    run_py scripts/scan_nationality.py
    run_py scripts/collect_foreign_exhaustion.py
    run_py scripts/collect_short_selling.py
    # --- G2: 지표 + 릴레이 ---
    run_py scripts/rebuild_indicators.py
    run_py scripts/run_ict_levels.py
    run_py scripts/run_relay_engine.py --update --signal
    run_py scripts/relay_report.py
    # --- G3: 시그널 스캔 + BRAIN + SHIELD ---
    run_py scripts/scan_buy_candidates.py
    run_py scripts/run_shield.py --send
    run_py scripts/run_brain.py
    run_py scripts/run_v3_brain.py --no-telegram
    # --- G4: 추천 + FLOWX ---
    run_py scripts/scan_earnings_acceleration.py
    run_py scripts/scan_turnaround.py
    run_py scripts/scan_tomorrow_picks.py
    run_py scripts/scan_tomorrow_picks.py --flowx --no-send
    run_py scripts/upload_flowx.py
    run_py scripts/dashboard_data.py
    # --- G5: 기록 ---
    run_py scripts/market_journal.py
    run_py scripts/daily_market_learner.py
    ;;
  J) # 17:00 KST — 포트폴리오 전망
    run_py scripts/run_portfolio_outlook.py
    ;;
  *)
    echo "[$(date +%H:%M:%S)] 알 수 없는 BAT: $BAT" >> "$LOG"
    ;;
esac

echo "[$(date +%H:%M:%S)] === BAT-$BAT 완료 (실패: ${FAIL_COUNT}건) ===" >> "$LOG"

# 실패 알림: BAT-D는 3건 이상, 나머지는 전체 실패 시
if [ "$BAT" = "D" ] && [ "$FAIL_COUNT" -ge 3 ]; then
  send_fail_alert "⚠️ <b>BAT-$BAT</b> ${FAIL_COUNT}건 실패 — 로그 확인 필요
<code>tail -50 $LOG</code>"
elif [ "$BAT" != "D" ] && [ "$FAIL_COUNT" -gt 0 ]; then
  send_fail_alert "⚠️ <b>BAT-$BAT</b> ${FAIL_COUNT}건 실패"
fi
