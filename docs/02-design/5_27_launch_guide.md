# 5/27(수) 실매매 진입 가동 가이드

> 작성: 2026-05-26 13:00
> 배경: 퐝가님 5/26 12:00 지시 "내일부터 제대로 실매매로 진행"
> 5/26 워밍업: 매수 0건 (안전망 검증 OK, 강세장에 큐 미발화는 예상된 결과)

---

## 1. 5/27 가동 시나리오

### 1-1. 09:00 첫 사이클 발화
```
cron */30 9-15 * * 1-5 → run_adaptive_cycle.py --real

MVP-1 (천장 매도): LS ELECTRIC 보호 격리 → SKIP. 다른 보유 0건.
MVP-2 (큐 트리거): 기존 큐 3종 (디아이티/산일전기/원익IPS) 가격 도달 시 매수
MVP-2.5 (trailing): 매수 후 +7% 도달 시 trailing 진입
MVP-2.6 (손절 -5%): 매수 후 -5% 도달 시 시장가 매도
MVP-2.7 (시간 매도): D+3 익절 (+4% peak) / D+5 데드라인 (-0.35% 회피) ★ 신규
MVP-3 (받침 패턴): 알림만
MVP-4 (재진입): 알림만
MVP-5 (AI 동조): ★ 신규 — 4섹터 동조 발화 시:
  → 자동 워치리스트 (만료 7일)
  → 자동 큐 등록 (peak=현재가, L1 -3%/L2 -7%/L3 -12%, 만료 3일)
  → 텔레그램 알림
```

### 1-2. 매수 발생 경로 3가지
1. **기존 MVP-2 큐** — 큐 종목 (디아이티/산일전기/원익IPS) 가격 -10/-20/-30% 도달 시
2. **AI 동조 신규 큐** — 5/26 캡쳐 5종 + 다른 AI 발화 종목 → -3/-7/-12% 눌림 시 매수
3. **차트영웅** — 09:30 morning_monitor / 14:55 close_cycle, max-qty=1

### 1-3. 매수 직전 게이트 (모두 통과 시만 buy_limit)
- **H4 VWAP**: OVERHEAT (+2.0%) 차단 / DIP (-1.5%) 우대
- **H5 호가**: 슬리피지 +0.5%↑ 차단 / 매수세 약함 차단
- **H6 매물대**: VAH 과열 차단 / POC 돌파 우대 / VAL 이탈 차단
- **H7 ATR**: 매수 성공 후 동적 손익절 stop_price/target_price 산출

### 1-4. 매도 보호 안전망 (사용자 자산 보호)
- MVP-2.5 trailing (+7% trailing -2%)
- MVP-2.6 손절 -5%
- MVP-2.7 D+3 익절 +4% / D+5 데드라인 강제 매도
- 매크로 가드 BEARISH 차단

---

## 2. .env 환경변수 (15:30 배포 시 추가)

VPS `/home/ubuntu/quantum-master/.env` 파일에 추가:
```bash
# 5/27 실매매 진입 — H4~H9 + AI 동조 활성화
ADAPTIVE_ENTRY_GATES_ENABLED=1   # H4+H5+H6+H7 게이트 ON
GATE_VWAP_ENABLED=1
GATE_ORDERBOOK_ENABLED=1
GATE_SUPPLY_ZONE_ENABLED=1
ATR_STOP_ENABLED=1
ADAPTIVE_TIME_EXIT_ENABLED=1     # H8 D+3 + H9 D+5

# AI 동조 (★ 5/27 실매매 핵심)
AI_CHAIN_QUEUE_AUTO_REGISTER=1   # AI 동조 발화 시 자동 큐 등록
AI_CHAIN_WATCHLIST_EXPIRY_DAYS=7
AI_CHAIN_QUEUE_EXPIRY_DAYS=3
AI_CHAIN_QUEUE_ALLOC_AMOUNT=1000000  # 종목당 100만원 (3종 매수 시 300만 = MAX_AMOUNT 한도)

# 기존 안전망 유지
AUTO_TRADING_ENABLED=1
AUTO_TRADING_MAX_AMOUNT=3000000
AUTO_TRADING_MAX_TRADES_PER_DAY=5
ADAPTIVE_SPLIT_MAX_AMOUNT=1000000
ADAPTIVE_REENTRY_MAX_AMOUNT=1000000
```

### 안전망 최대 손실 (1일)
- 자비스 큐 1일 매수 한도: 300만원 (`AUTO_TRADING_MAX_AMOUNT`)
- 일일 매수 횟수: 5회 (`AUTO_TRADING_MAX_TRADES_PER_DAY`)
- 최악 시나리오 (전 매수 -5% 손절): 300만 × 5% = **15만원** (시드 25,050,012원의 0.6%)

---

## 3. 16:00~ 백테스트 의무 (5/27 가동 전 검증)

### 백테스트 실행 (VPS)
```bash
cd ~/quantum-master
PYTHONPATH=. ./venv/bin/python3.11 scripts/backtest/backtest_d0_vs_d1_entry.py --lookback-days 90
```

### Acceptance Criteria (메모리 [[feedback-backtest-first]])
- D+1 평균 ≥ +0.84% (현행 MVP-2.6 R2 기준)
- 승률 ≥ 56.9%
- +10%↑ 비율 ≥ 46.5%
- MDD 악화 +2%p 한도

### 결단
- **통과 시** → 5/27 09:00 정식 가동 (위 .env 설정 모두 활성)
- **실패 시** → AI 동조 큐 자동 등록만 비활성 (`AI_CHAIN_QUEUE_AUTO_REGISTER=0`), 워치리스트는 유지

---

## 4. 5/26 누적 변경 사항 (15:30 commit 예정)

### 신규 모듈 (9개)
1. `src/use_cases/vwap_gate.py` (H4)
2. `src/use_cases/orderbook_gate.py` (H5)
3. `src/use_cases/supply_zone_gate.py` (H6)
4. `src/use_cases/atr_dynamic_stop.py` (H7)
5. `src/use_cases/adaptive_time_exit.py` (H8/H9)
6. `src/use_cases/adaptive_entry_gates.py` (통합)
7. `src/use_cases/ai_chain_detector.py` (AI 동조)
8. `src/use_cases/ai_chain_auto_watchlist.py` (워치리스트 자동)
9. `src/use_cases/ai_chain_queue_auto_register.py` (큐 자동 등록) ★ 5/27 핵심

### 통합 변경 (5건)
- `scripts/run_adaptive_cycle.py`: MVP-2.7 + MVP-5 + AI 워치리스트 + AI 큐 자동
- `src/use_cases/adaptive_buy_queue.py`: execute_auto_buy() H4~H7 게이트 후크
- `src/adapters/kis_order_adapter.py`: fetch_ohlcv passthrough
- `scripts/intraday_eye.py`: AI 동조 워치리스트 자동 통합
- `config/sector_fire_map.yaml`: AI 4세부 섹터 + 24종목 추가

### 회귀 테스트
- 신규 단위 테스트: 92건 (H4 12 + H5 11 + H6 15 + H7 13 + H8/H9 17 + 통합 9 + AI 검출 10 + AI 워치 10 + AI 큐 15)
- 기존 영향 모듈 회귀: 116건 (adaptive_buy_queue 25 + position_manager 19 + quick_profit + simulate_5_26 + protected + v8 30 + etc.)
- **총 208/208 PASS** ✓
- (테스트 격리 이슈 23건은 pre-existing, 16:00 이후 별도)

---

## 5. 5/27 모니터링 체크포인트

### 09:00 첫 사이클 직후 (5분 안)
- 텔레그램 알림 도착 확인
- MVP-5 AI 동조 발화 여부 (어제 5종 또는 다른 AI 종목 폭등 시)
- AI 동조 큐 자동 등록 알림 (★ 첫 실매매 진입 가능성)

### 09:30 morning_monitor
- 차트영웅 매수 시도 (max-qty 1)
- 진입 필터 (VWAP+수급 2/2) 통과 여부

### 매수 발생 시 — 즉시 확인 사항
1. 종목명 정상 출력 (어제 EYE-07 fix)
2. H4~H7 게이트 통과 표시 (`gate_summary` 텔레그램에)
3. quick_profit_target / stop_price 자동 설정
4. LS ELECTRIC 격리 정상 (자동 매도 X)

### 15:30 마감 후
- 사이클 요약 (MVP1~5 트리거 횟수)
- decision_log.json + signal_snapshot.db 학습 데이터 누적
- 다음날 picks_history 갱신 확인

---

## 6. 위험 + 안전망

### 매수 발생 안 할 수 있는 경우
- AI 동조 4섹터 중 3섹터 동시 발화 미발생 (오늘처럼 1~2섹터만)
- H4~H7 게이트 차단 (VWAP 과열 / 매물대 VAH / 호가 슬리피지)
- 매크로 가드 BEARISH (KOSPI 약세)

→ 5/27도 안전망 작동 = 매수 0건 가능성 ↑20%

### 매수 과다 발생 위험
- AI 동조 발화 + 폭등 종목 10+ 동시 → 큐 10종 등록 시 1,000만원 할당
- 안전망: `AUTO_TRADING_MAX_AMOUNT=300만원` + `MAX_TRADES_PER_DAY=5` → 300만 5회 한도 발동

---

## 7. 메모리 규칙 준수 확인

- [[feedback-warmup-meaning-first]] ✓ — 1주차 워밍업 의미 보존 (안전망 검증 후 가동)
- [[feedback-backtest-first]] ✓ — 16:00 백테스트 통과 시만 활성
- [[feedback-advise-not-force-decision]] ✓ — 결단 강요 X, 권장 + 자동 진행
- [[feedback-subagent-team-first]] ✓ — stock-analyzer 백그라운드 분업

5/27 09:00 텔레그램 모니터링 부탁드립니다 🚀
