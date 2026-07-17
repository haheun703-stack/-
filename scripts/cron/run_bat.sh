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

# exit code 반환 버전 (분기용, FAIL_COUNT 카운팅 안 함)
# 3-봇 분업: sync 결과로 다음 단계를 결정해야 할 때만 사용
run_py_long_check() {
  local script="$1"; shift
  timeout 900 $PY "$script" "$@" >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 124 ]; then
    echo "[$(date +%H:%M:%S)] [INFO] $script 타임아웃 (900초 초과, fallback 분기)" >> "$LOG"
  elif [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] [INFO] $script 비정상 종료 (exit=$rc, fallback 분기)" >> "$LOG"
  fi
  return $rc
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
    # B-12 (7/17): 원자재 시세 v2 (yfinance) — 4/21 수집기 아카이브 후 유물 방치 → 부활.
    #   scan_tomorrow_picks(전략 M)가 같은 A단계 후속에서 소비하므로 이 위치(앞쪽) 고정.
    run_py scripts/collect_commodity_prices.py
    run_py scripts/fetch_cot_weekly.py
    run_py scripts/run_cot_tracker.py
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
  # M_morning) 폐지 (2026-05-17): morning_briefing_generator + theme_scan_runner는 archive/deprecated.
  # CLAUDE.md LOCK 규칙 위반 + sys.exit(1)로 BAT-M_morning 매일 실패 카운트 누적 → 옵션 C로 통째 제거.
  M_US) # 08:10 KST — 미국장 매크로 필터 (정보봇 07:55 후 실행)
    run_py scripts/us_quant_filter.py
    run_py scripts/upload_quant_us.py
    ;;
  N) # 08:20 KST — 시그널 로그
    run_py scripts/cron_signal_tracker.py --mode log
    ;;
  E) # 08:50 KST — 스마트 진입 (5/28 P0: --live --force 제거, .env AUTO_TRADING_ENABLED 단일 신뢰)
    # 5/27 09:55 owner_rule_monitor 사고 후속 — cron 인자가 .env 덮어쓰지 못하도록 인자 폐기
    # smart_entry 활성화는 .env AUTO_TRADING_ENABLED=1 일 때만 (KisOrderAdapter._guard 통과)
    run_py scripts/smart_entry_runner.py
    run_py scripts/sell_monitor.py --dry-run
    ;;
  I) # 08:55 KST — VWAP + EYE 킬러픽 모니터 (둘 다 장중 장기 실행)
    $PY scripts/intraday_eye.py --killer-picks >> "$LOG" 2>&1 &
    $PY scripts/run_vwap_monitor.py --killer-picks >> "$LOG" 2>&1 &
    ;;
  LU) # 08:55 KST — 상한가 풀림 실시간 감지 (장중 장기 실행, 15:20 자동 종료)
    $PY scripts/run_limit_up_scanner.py --scan --dry-run >> "$LOG" 2>&1 &
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
    # BAT-L(NXT)과 BAT-I(EYE/VWAP/LU) 잔여 프로세스 정리 — 동시 실행 시 KIS API 충돌 + OOM 방지
    pkill -f "nxt_market_collector" >> "$LOG" 2>&1 || true
    pkill -f "intraday_eye" >> "$LOG" 2>&1 || true
    pkill -f "run_vwap_monitor" >> "$LOG" 2>&1 || true
    pkill -f "run_limit_up_scanner" >> "$LOG" 2>&1 || true
    sleep 2
    echo "[$(date +%H:%M:%S)] [INFO] BAT-L/I 잔여 프로세스 정리 완료" >> "$LOG"
    # .pyc 캐시 삭제 — git pull 후 구버전 캐시 실행 방지
    find "$QM" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    # --- G1: 데이터 수집 ---
    run_py_xlong scripts/update_daily_data.py
    run_py_xlong scripts/extend_parquet_data.py --workers 2
    run_py_long scripts/update_raw_parquet.py
    run_py scripts/rebuild_universe.py --incremental
    run_py scripts/update_kospi_index.py
    run_py scripts/update_kosdaq_index.py     # v1-3번: KOSDAQ 지수(^KQ11) → 시장별 국면
    run_py scripts/update_market_map.py       # v1-3번: FDR 시장구분(KOSPI/KOSDAQ) 맵
    # 지수 Buy&Hold 벤치마크 17종(국내외 지수·ETF·레버리지 1x/2x/3x) yfinance 증분 수집 → data/benchmark/
    run_py_long scripts/update_benchmarks.py
    # 수급 동기화: 단타봇 flow CSV(16:00 수집완료) → stock_data_daily (DB 폴백 자동)
    # (구 SYNC_OK 분기는 collect_investor_bulk 폐기로 제거 — 단타봇 sync 자체는 유지)
    run_py_long_check scripts/sync_investor_to_csv.py || true
    run_py scripts/us_overnight_signal.py --update
    run_py scripts/update_us_kr_daily.py  # 장마감 후 2차 수집 (BAT-A 06:10 시점 KR 미반영분 보충)
    # [KRX잠금 6/22 사장님지시] scan_nationality.py 비활성 — KRXSession ID/PW 로그인이 매 실행(gap-fill 최대 7회) CD007 계정잠금 유발(6/12~ 결측). KRX 일절 미접근.
    #   재가동은 KRX 사이트에서 잠금해제+비번 재설정 후 별도 결정. (3주체 수급은 아래 collect_investor_kis=KIS 화이트리스트로 정상 수집됨)
    # run_py_xlong scripts/scan_nationality.py
    run_py_xlong scripts/collect_foreign_exhaustion.py
    # collect_short_selling.py 제거 — KRX 공매도 데이터 제공 중단 (2026-04)
    run_py_long scripts/institutional_flow_collector.py
    run_py scripts/scan_volume_spike.py
    run_py scripts/sector_etf_builder.py --daily
    run_py scripts/collect_investor_flow.py
    # KIS 종목별 투자자 '세분' 수급 (11주체: 외인/기관계/개인 + 금융투자/연기금/투신/사모/은행/보험/기타금융/기타법인)
    # → investor_daily.db. TR=FHPTJ04160001(investor-trade-by-stock-daily): 한 호출에 30거래일치, 고정IP 13.209.153.221 화이트리스트.
    # [2026-06-27 사장님지시] KRX 영구 포기 → KIS 자급. 구 collect_investor_bulk(pykrx/KRX 로그인=CD007 잠금 원인) 폐기.
    #   이 수집으로 signal_engine 금투/연기금 신호(PHASE5_STAGE3) + export_investor_for_scalper(단타봇 공유) 정상 복원.
    run_py_xlong scripts/collect_investor_kis.py --days 5
    run_py scripts/export_investor_for_scalper.py
    # sync_investor_to_csv → G1 상단으로 이동 (단타봇 데이터 선행 활용)
    run_py scripts/fetch_ecos_macro.py
    # COO 복원: ETF/섹터 수급 수집
    run_py scripts/collect_etf_volume.py
    run_py scripts/collect_etf_investor_flow.py
    run_py_long scripts/crawl_china_money.py --no-telegram
    # --- G2: 지표 + 릴레이 ---
    run_py_long scripts/rebuild_indicators.py
    run_py scripts/run_ict_levels.py
    run_py scripts/run_relay_engine.py --update --signal
    run_py scripts/relay_report.py
    run_py scripts/group_relay_detector.py
    # COO 복원: 섹터 분석
    run_py scripts/sector_momentum.py --history
    run_py scripts/sector_zscore.py --top 5
    run_py_long scripts/sector_investor_flow.py --days 5
    run_py src/sector_composite.py
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
    run_py scripts/scan_surge_pullback.py --telegram
    run_py scripts/scan_crash_bounce.py
    run_py scripts/scan_dual_buying.py
    run_py scripts/scan_accumulation_tracker.py
    run_py scripts/calc_institutional_targets.py
    # COO 복원: 세력/밸류체인/공시/뉴스 스캔
    run_py_long scripts/scan_whale_detect.py
    run_py_long scripts/scan_force_hybrid.py
    run_py scripts/scan_value_chain.py
    run_py scripts/crawl_dart_disclosure.py
    run_py scripts/dart_event_signal.py
    run_py scripts/crawl_market_news.py
    run_py_long scripts/perplexity_market_intel.py
    run_py_long scripts/ai_news_brain.py
    # COO 복원: 컨센서스 스크리너 (wisereport 목표가 + 기술적 분석)
    run_py_long scripts/scan_consensus.py
    # G3.9: 상한가 풀림 감시 후보 갱신 (일봉 수집 완료 후)
    run_py scripts/run_limit_up_scanner.py --generate
    # G3.9.5: 공매도 3종 8시그널 + 4팩터 (정보봇 KIS API)
    run_py scripts/scan_short_factor.py
    # G3.9.6: 투자자 수급 통합본 시그널 (정보봇 pykrx)
    run_py scripts/scan_investor_flow.py
    # --- G4: 추천 ---
    # COO 복원: 성과 추적 (추천 전에 실행)
    run_py scripts/track_pick_results.py
    # B2 최적화: [EA]+[TA] 병렬 실행 (독립적, 입력=read-only/출력=다른 파일)
    run_py scripts/scan_earnings_acceleration.py &
    EA_PID=$!
    run_py scripts/scan_turnaround.py &
    TA_PID=$!
    wait $EA_PID $TA_PID
    # RSS 뉴스 테마 스냅샷 (scan_tomorrow_picks 전략T 동적 가점 입력). graceful=실패해도 exit0
    run_py scripts/run_rss_theme_scan.py
    run_py_long scripts/scan_tomorrow_picks.py
    run_py_long scripts/scan_tomorrow_picks.py --flowx --no-send
    run_py scripts/build_killer_picks.py
    run_py scripts/run_cto.py
    run_py src/use_cases/portfolio_cfo.py
    # --- G4.2: 수급 급변 + 바톤터치 + 바닥반등 스캐너 (upload 전 실행 필수) ---
    run_py scripts/scan_supply_surge.py
    run_py scripts/detect_supply_chain.py      # G4.2: 수급 바톤터치 감지
    run_py scripts/scan_structure_score.py     # G4.2: Structure Score (연간돌파+주봉StochRSI+시장레짐)
    run_py scripts/scan_sector_fire.py         # G4.2: 섹터 발화(FIRE) + Structure 병합 → Composite
    run_py scripts/scan_valuation_gap.py            # G4.2: 실적 괴리(GAP) DART자동+캐시폴백 (업로드는 G4.9 upload_flowx.py에서 일괄)
    run_py scripts/bluechip_timing.py          # G4.3: 우량주 TOP100 매매타이밍
    run_py scripts/scan_type2_bottom.py
    # --- G4.5: NXT 추천 + 바이오 CDMO 감시 (upload 전 실행 필수) ---
    run_py src/use_cases/nxt_signal.py
    run_py scripts/nxt_recommend.py --no-telegram
    run_py scripts/nxt_track_results.py
    run_py scripts/scan_nugget.py
    run_py scripts/scan_fibonacci.py
    run_py scripts/scan_market_ranking.py
    run_py scripts/scan_bio_cdmo.py
    run_py scripts/scan_bluechip_checkup.py
    # --- G4.6: EWY(MSCI Korea ETF) 보유종목 비중변화 수집 ---
    run_py_long scripts/collect_ewy_holdings.py --monthly --upload --telegram
    # --- G4.9: FLOWX 업로드 (모든 스캔 완료 후 일괄 업로드) ---
    run_py scripts/fetch_theme_intel.py           # Supabase에서 테마 인텔 읽기 (단타봇이 올린 데이터)
    run_py scripts/build_brain_upload.py
    run_py scripts/signal_logger.py              # G4.9: tomorrow_picks → signals 테이블 기록
    run_py scripts/upload_flowx.py
    # dashboard_data.py 단독실행 제거 — upload_flowx.py 내부 import로 통합 (루트에 파일 존재)
    # G4.9.5: ETF 수급 자동 시그널 (Phase 8 백테스트 기반, theme 88.9% 적중률)
    # 정보봇 16:28 업데이트 → 퀀트봇 BAT-D 후반부에서 fetch + 텔레그램
    run_py scripts/fetch_etf_flow_daily.py --analyze --telegram
    run_py scripts/send_evening_summary.py --send
    # --- G5: 기록 + Paper ---
    run_py scripts/market_journal.py
    run_py_long scripts/daily_market_learner.py  # v2 패턴학습: parquet 2회 풀스캔
    # 상한가 사후학습 트랙 (6/24 신규) — 전종목 상한가 D-1 수급선행 태깅 → 학습DB(limit_up.db) 축적.
    #   FDR StockListing snapshot 사용(KRX 로그인 無 = CD007 잠금 무관). "수급선행했는데 놓친 상한가"=개선 금맥.
    run_py scripts/scan_limit_up_postmortem.py --date $(date +%Y-%m-%d)
    run_py scripts/paper_trading_unified.py
    # 확신모델 B안 병렬 페이퍼 (과매집 감점 — conviction-reversal §6 사장님 승인 6/22).
    #   + 매도경로 변형 (6/23 검증 승인 — 보유일 연장 제거 + 손실구간 쌍끌이 조기탈출).
    #   관측·격리: 실주문 무접촉 · paper_portfolio_b.json만 누적 · 텔레그램/FLOWX 미발신.
    export PAPER_CONVICTION_MODE=B
    export PAPER_SELL_REVISION=1
    run_py scripts/paper_trading_unified.py
    unset PAPER_CONVICTION_MODE
    unset PAPER_SELL_REVISION
    # 밸류-피보나치 페이퍼 (4번째 독립포트, 6/24 백테스트 확정: 승률77%·+33%/년·MDD-9%).
    #   진입=60일고점-10%+RSI<40+반등확인+수급 / 청산=전고점or+25%/-20% / 회전금지(보유).
    #   관측·격리: 실주문 무접촉 · data/paper_portfolio_vf.json만 누적.
    run_py scripts/paper_value_fib.py
    # 지주사 NAV 디스카운트 페이퍼 (5번째 독립포트, 6/25 백테스트+정련: z[1,2]밴드 승률81%·+23%/D+60·최악-37%→손절-15%).
    #   신호=할인축소z∈[1,2]+NAVmom5d>0·할인거래 / 청산=D+60보유 or -15%손절 / 검증4종목(㈜LG·SK·두산·삼성물산)·쿨다운20일.
    #   관측·격리: 실주문 무접촉 · data/paper_portfolio_holdnav.json만 누적. 사업지주(한화)·단일베타(SK스퀘어)·CJ버그 제외.
    run_py scripts/paper_holding_nav.py
    # 지수 Buy&Hold 페이퍼 (6번째 독립포트, 영구 벤치마크 기준선 — 7/1 배선).
    #   17종 국내외 지수·ETF·레버리지(1x/2x/3x)를 baseline 6/22 정규화 수익률 시계열 추적.
    #   "게임1(타이밍) 폐기→지수보유 기준선" 확정 반영. 5개 전략 페이퍼 공통 벤치마크.
    #   관측·매매무관 · data/paper_portfolio_indexbh.json. 데이터=update_benchmarks(G1).
    run_py scripts/paper_index_buyhold.py
    # 파도타기 페이퍼 (7번째 트랙, 7/7 퐝가님 비전 "오르면 레버리지") — V3b 레짐 방향성 KR+US.
    #   BULL=레버2x/CAUTION=지수/BEAR·CRISIS=현금. 인버스는 phase12 기각으로 미포함(지수BH 관측만).
    #   V3b 실전검증 겸용 · 관측 전용 · 실주문 0 · data/paper_portfolio_wave_{kr,us}.json
    run_py scripts/paper_wave_rider.py
    # G5.5: 주도주 사이클 진단 shadow 관측 (6/30, 한규범 절대법칙) — 매매 미반영·관측 JSON만.
    #   global_leaders.yaml(US 대장주30 + KR60) 사이클 진단 → data/shadow/leader_cycle.json.
    #   US 가격/재무 yfinance(고정IP 화이트리스트) 선행 갱신 + KR DART TTM-YoY 델타. freeze 무관.
    #   대시보드 노출은 정보봇(FLOWX UI)이 이 JSON 소비(global_leaders.yaml 규약).
    run_py_long scripts/fetch_us_leader_data.py
    run_py_long scripts/run_leader_cycle.py --quiet
    # G5.6: 미래가치 통합 엔진 shadow (7/4) — 6축(밸류갭·실적가속·사이클·수주·수급·테마) 결합.
    #   입력 전부 위 단계 산출물(consensus G3.9·valuation_gap G4.2·leader_cycle G5.5·수급DB).
    #   관측 전용 data/shadow/future_value.json — 매매 미배선·graceful exit0. 설계=docs/02-design/future-value-engine_2026-07-04.md
    #   수주이력: 공급계약 공시 본문(계약금액/매출대비) 파싱 누적 → O축 입력(50%+만 단기가점, 7/4 이벤트스터디 607건 근거)
    run_py scripts/collect_contract_history.py
    run_py scripts/run_future_value.py
    # G5.7: 미국판 미래가치 엔진 shadow (7/6, 퐝가님 배선결정) — 컨센서스(yfinance)+역사PER밴드+
    #   leader_cycle(G5.5 US) 사이클. ★④ 백테스트 밴드 무효(t=0.61·reliable -0.61%p 역전)→밴드·저PER
    #   무가점(관측태그), 실점수=사이클. 컨센서스 괴리축은 역사목표가 부재로 백테스트 불가 →
    #   러너가 매일 목표가 스냅샷 축적(consensus_us_history.jsonl) → 20거래일 forward 검증.
    #   관측 전용 data/shadow/future_value_us.json — 매매 미배선·shadow_unvalidated·graceful exit0. VPS venv.
    run_py_long scripts/run_future_value_us.py
    # G5.8: 시나리오 v1 러너 (7/7, 퐝가님 "우리만의 시나리오" 지시) — 레짐(V0+V3b 섀도)·모드·
    #   FV 워치리스트를 data/shadow/scenario_v1.json 기록. 행동표=docs/scenario_v1.md.
    #   ★약세장 수급 팔로우 기각(bear_accumulation -2.22%p t=-9.6)·수급=BULL 확인지표로 재정의.
    #   관측 전용·실주문 0·graceful exit0. V3b divergence 축적 → 레짐 교체 결정 근거(퐝가님).
    run_py scripts/run_scenario_v1.py
    # 포트 3테이블 대시보드 적재 재개 (7/1, unfreeze와 분리 — 정보봇 aec1043 회신·사장님 결정).
    #   valuation_band(밸류밴드 verdict 60행)·two_layer(82/18 골격)·drawdown_alert(level=normal).
    #   관측·매매무관. Q2 foreign_outflow=시장전체·Q3 키명(port_exposure·recommended_actions)은 alert(-15%) 시 실값.
    run_py_long scripts/upload_valuation_band.py --write
    run_py scripts/upload_two_layer.py --write
    run_py scripts/data_health_check.py
    # G7: 약세장 알파 학습 + 인버스 시그널 (5/16 추가, 5/12~15 약세장 검증 기반)
    # KOSPI MA20 -2%↓ 시 알파 종목 자동 추출 + 외인 5일 매도 -3조+ 시 인버스 알림
    run_py scripts/bear_market_alpha_runner.py
    # G8: 자비스 시그널 엔진 (5/16, Phase 1) — 검증된 8개 시그널 통합 점수 + 상위 5종목
    # 매일 BAT-D 후반부 자동 실행 → 텔레그램 알림 (자동매매 OFF, Phase 5 이후 활성화)
    run_py scripts/signal_engine.py --top 5 --notify
    # G8.5: 추세추종 스캐너 (6/22, 풀백 스캐너의 빈칸=강추세주 발굴) — 신호 생성/관측만, 자동매매 OFF
    run_py scripts/trend_follow_scanner.py
    # 유니버스 전체 재구성은 BAT-H(11:30 장중)로 이동 — pykrx 장후 불안정 해결
    # G6: BAT-D 자동 메트릭 수집 + 이상 감지 + 텔레그램 (5/16) — ★BAT-D 마지막 스텝으로 이동(7/14 검수).
    #   완료 echo는 esac 밖이라 이 지점에선 로그 완료마커를 못 읽음 → FAIL_COUNT를 env로 전달.
    #   소요시간: 당일이면 현재시각 폴백으로 계산 복원 + 절대임계 175분 재보정(7/16 실측 133~150분 기반).
    BAT_D_FAIL_COUNT="$FAIL_COUNT" run_py scripts/bat_d_health_check.py
    ;;
  F) # 18:35 KST — FLOWX 업로드 보장 (BAT-D 완료 후 재시도, upsert이라 중복 안전)
    run_py scripts/fetch_theme_intel.py
    run_py scripts/build_brain_upload.py
    run_py scripts/signal_logger.py              # signals 테이블 재시도
    run_py scripts/upload_flowx.py
    ;;
  J) # 17:00 KST — 포트폴리오 전망
    run_py scripts/run_portfolio_outlook.py
    ;;
  PICKV2) # 17:45 KST - daily_pick_v2 (Silent Bet + 메인 스코어링)
    run_py scripts/daily_pick_v2.py
    ;;
  HEALTH) # 18:45 KST — 자동 복구: BAT-D 완료(~18:30) 후 신선도 확인 → 낡은 파일만 재실행
    # run_py_xlong(1800초): 선택적 복구 최악 케이스(5개 파일 stale = 2400초) 대비 마진 확보
    run_py_xlong scripts/health_check.py
    ;;
  *)
    echo "[$(date +%H:%M:%S)] 알 수 없는 BAT: $BAT" >> "$LOG"
    ;;
esac

echo "[$(date +%H:%M:%S)] === BAT-$BAT 완료 (실패: ${FAIL_COUNT}건) ===" >> "$LOG"

# 실패 알림: BAT-D는 1건 이상이면 상세 스크립트 목록 발송, 나머지는 전체 실패 시
if [ "$BAT" = "D" ] && [ "$FAIL_COUNT" -gt 0 ]; then
  # WARN 라인에서 실패한 스크립트명 추출
  FAIL_SCRIPTS=$(grep "\[WARN\]" "$LOG" | grep -oP 'scripts/\S+\.py|src/\S+\.py' | sort -u | sed 's/^/• /' | head -10)
  send_fail_alert "[HEALTH] ⚠️ <b>BAT-D</b> ${FAIL_COUNT}건 실패

<b>실패 스크립트:</b>
<code>${FAIL_SCRIPTS}</code>"
elif [ "$BAT" != "D" ] && [ "$FAIL_COUNT" -gt 0 ]; then
  send_fail_alert "[HEALTH] ⚠️ <b>BAT-$BAT</b> ${FAIL_COUNT}건 실패"
fi
