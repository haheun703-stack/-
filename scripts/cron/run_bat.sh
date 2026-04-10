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

# 실패 카운터 함수: 개별 타임아웃 기본 300초, 긴 작업은 run_py_long 사용
run_py() {
  local script="$1"; shift
  timeout 300 $PY "$script" "$@" >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 124 ]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[$(date +%H:%M:%S)] [WARN] $script 타임아웃 (300초 초과)" >> "$LOG"
  elif [ $rc -ne 0 ]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[$(date +%H:%M:%S)] [WARN] $script 실패 (exit=$rc)" >> "$LOG"
  fi
  return 0
}

# 긴 작업용 (타임아웃 900초 = 15분)
run_py_long() {
  local script="$1"; shift
  timeout 900 $PY "$script" "$@" >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 124 ]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[$(date +%H:%M:%S)] [WARN] $script 타임아웃 (900초 초과)" >> "$LOG"
  elif [ $rc -ne 0 ]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[$(date +%H:%M:%S)] [WARN] $script 실패 (exit=$rc)" >> "$LOG"
  fi
  return 0
}

# 초대형 작업용 (타임아웃 1800초 = 30분) — KIS API 대량호출, parquet 전체확장
run_py_xlong() {
  local script="$1"; shift
  timeout 1800 $PY "$script" "$@" >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 124 ]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[$(date +%H:%M:%S)] [WARN] $script 타임아웃 (1800초 초과)" >> "$LOG"
  elif [ $rc -ne 0 ]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[$(date +%H:%M:%S)] [WARN] $script 실패 (exit=$rc)" >> "$LOG"
  fi
  return 0
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
    run_py_long scripts/run_v3_brain.py --no-telegram
    run_py scripts/scan_tomorrow_picks.py
    run_py scripts/market_sense_engine.py --send
    # NXT 모닝 추천 — 8시 프리마켓 전 텔레그램 발송
    run_py src/use_cases/nxt_signal.py
    run_py scripts/nxt_recommend.py
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
  M_US) # 08:10 KST — 미국장 매크로 필터 (정보봇 07:55 후 실행)
    run_py scripts/us_quant_filter.py
    run_py scripts/upload_quant_us.py
    ;;
  N) # 08:20 KST — 시그널 로그
    run_py scripts/cron_signal_tracker.py --mode log
    ;;
  E) # 08:50 KST — 스마트 진입
    run_py scripts/smart_entry_runner.py --live --force
    run_py scripts/sell_monitor.py --dry-run
    ;;
  I) # 08:55 KST — VWAP + EYE 킬러픽 모니터 (둘 다 장중 장기 실행)
    $PY scripts/intraday_eye.py --killer-picks >> "$LOG" 2>&1 &
    $PY scripts/run_vwap_monitor.py --killer-picks >> "$LOG" 2>&1 &
    ;;
  H) # 11:30 KST — 장중 분석
    run_py scripts/run_midday_analysis.py
    # 월/금 장중 유니버스 전체 재구성 (pykrx는 장중에만 안정적)
    DOW_H=$(date +%u)
    if [ "$DOW_H" = "1" ] || [ "$DOW_H" = "5" ]; then
        echo "[$(date +%H:%M:%S)] [INFO] 유니버스 전체 재구성 시작 (DOW=$DOW_H)" >> "$LOG"
        run_py_long scripts/rebuild_universe.py --min-cap 0.2 --no-cleanup
    fi
    ;;
  L) # 15:35 KST — NXT 장마감 후 (16:25까지만 수집, BAT-D 16:30 충돌 방지)
    timeout 3000 $PY scripts/nxt_market_collector.py --session after >> "$LOG" 2>&1 &
    ;;
  O) # 16:10 KST — 시그널 트래킹
    run_py scripts/cron_signal_tracker.py --mode track
    ;;
  D) # 16:30 KST — 장마감 전체 파이프라인 (데이터 + 분석 + 추천)
    # 텔레그램 QUIET 모드: [EVENING],[PAPER],[NXT],[HEALTH] 태그만 발송
    export TELEGRAM_QUIET=1
    # BAT-L(NXT)과 BAT-I(EYE) 잔여 프로세스 정리 — 동시 실행 시 KIS API 충돌 + OOM 방지
    pkill -f "nxt_market_collector" >> "$LOG" 2>&1 || true
    pkill -f "intraday_eye" >> "$LOG" 2>&1 || true
    pkill -f "run_vwap_monitor" >> "$LOG" 2>&1 || true
    sleep 2
    echo "[$(date +%H:%M:%S)] [INFO] BAT-L/I 잔여 프로세스 정리 완료" >> "$LOG"
    # .pyc 캐시 삭제 — git pull 후 구버전 캐시 실행 방지
    find "$QM" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    # --- G1: 데이터 수집 ---
    run_py scripts/update_daily_data.py
    run_py_xlong scripts/extend_parquet_data.py --workers 2
    run_py scripts/rebuild_universe.py --incremental
    run_py scripts/update_kospi_index.py
    # collect_intraday_candles.py 제거 — 종목선정에 미사용, smart_entry는 실시간 조회
    run_py scripts/us_overnight_signal.py --update
    run_py_long scripts/scan_nationality.py
    run_py_xlong scripts/collect_foreign_exhaustion.py
    # collect_short_selling.py 제거 — KRX 공매도 데이터 제공 중단 (2026-04)
    run_py_long scripts/institutional_flow_collector.py
    run_py scripts/scan_volume_spike.py
    run_py scripts/sector_etf_builder.py --daily
    run_py scripts/collect_investor_flow.py
    run_py scripts/fetch_ecos_macro.py
    # COO 복원: ETF/섹터 수급 수집
    run_py scripts/collect_etf_volume.py
    run_py scripts/collect_etf_investor_flow.py
    # --- G2: 지표 + 릴레이 ---
    run_py_long scripts/rebuild_indicators.py
    run_py scripts/run_ict_levels.py
    run_py scripts/run_relay_engine.py --update --signal
    run_py scripts/relay_report.py
    # COO 복원: 섹터 분석
    run_py scripts/sector_momentum.py --history
    run_py scripts/sector_zscore.py --top 5
    run_py_long scripts/sector_investor_flow.py --days 5
    # --- G3: 레짐 + BRAIN + SHIELD ---
    run_py scripts/regime_macro_signal.py
    run_py scripts/scan_buy_candidates.py
    run_py scripts/run_shield.py --send
    run_py scripts/run_brain.py
    run_py scripts/run_master_brain.py
    run_py_long scripts/run_v3_brain.py --no-telegram
    run_py scripts/run_lens.py
    # --- G3.5: ETF 방향 트레이딩 + 눈치 엔진 ---
    run_py scripts/market_sense_engine.py --send
    run_py scripts/jarvis_direction_engine.py --send
    run_py scripts/leverage_etf_scanner.py
    run_py scripts/etf_trading_signal.py
    run_py scripts/run_etf_rotation.py
    run_py scripts/calc_asset_etf_performance.py
    # --- G3.9: 종목 스캔 (scan_tomorrow_picks 입력 데이터) ---
    run_py scripts/scan_pullback.py
    run_py scripts/scan_crash_bounce.py
    run_py scripts/scan_dual_buying.py
    run_py scripts/scan_accumulation_tracker.py
    run_py scripts/calc_institutional_targets.py
    # COO 복원: 세력/밸류체인/공시/뉴스 스캔
    run_py_long scripts/scan_whale_detect.py
    run_py_long scripts/scan_force_hybrid.py
    run_py scripts/scan_value_chain.py
    run_py scripts/dart_event_signal.py
    run_py scripts/crawl_market_news.py
    run_py_long scripts/perplexity_market_intel.py
    run_py_long scripts/ai_news_brain.py
    # COO 복원: 컨센서스 스크리너 (wisereport 목표가 + 기술적 분석)
    run_py_long scripts/scan_consensus.py
    # --- G4: 추천 + FLOWX ---
    # COO 복원: 성과 추적 (추천 전에 실행)
    run_py scripts/track_pick_results.py
    run_py scripts/scan_earnings_acceleration.py
    run_py scripts/scan_turnaround.py
    run_py scripts/scan_tomorrow_picks.py
    run_py scripts/scan_tomorrow_picks.py --flowx --no-send
    run_py scripts/build_killer_picks.py
    run_py scripts/run_cto.py
    run_py src/use_cases/portfolio_cfo.py
    run_py scripts/build_brain_upload.py
    run_py scripts/upload_flowx.py
    # dashboard_data.py 제거 — 파일 미존재
    run_py scripts/send_evening_summary.py --send
    # --- G4.5: NXT 추천 + 바이오 CDMO 감시 ---
    run_py src/use_cases/nxt_signal.py
    run_py scripts/nxt_recommend.py --no-telegram
    run_py scripts/nxt_track_results.py
    run_py scripts/scan_nugget.py
    run_py scripts/scan_fibonacci.py
    run_py scripts/scan_market_ranking.py
    run_py scripts/scan_bio_cdmo.py
    # --- G5: 기록 + Paper ---
    run_py scripts/market_journal.py
    run_py_long scripts/daily_market_learner.py  # v2 패턴학습: parquet 2회 풀스캔
    run_py scripts/paper_trading_unified.py
    run_py scripts/data_health_check.py
    # 유니버스 전체 재구성은 BAT-H(11:30 장중)로 이동 — pykrx 장후 불안정 해결
    ;;
  F) # 17:15 KST — FLOWX 업로드 보장 (BAT-D 실패 대비, upsert이라 중복 안전)
    run_py scripts/build_brain_upload.py
    run_py scripts/upload_flowx.py
    ;;
  J) # 17:00 KST — 포트폴리오 전망
    run_py scripts/run_portfolio_outlook.py
    ;;
  HEALTH) # 18:00 KST — 자동 복구: 데이터 신선도 확인 → 낡은 파일만 개별 스크립트 재실행
    # run_py_xlong(1800초): 선택적 복구 최악 케이스(5개 파일 stale = 2400초) 대비 마진 확보
    run_py_xlong scripts/health_check.py
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
