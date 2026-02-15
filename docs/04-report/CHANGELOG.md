# 변경 로그 (CHANGELOG)

모든 프로젝트의 주요 변경 사항을 문서화합니다.

---

## [v4.0] - 2026-02-13

### 라이브 트레이딩 시스템 구현 완료

**Status**: ✅ 완료 (APPROVED)

#### 주요 추가 기능

##### 신규 모듈 (6개)

1. **src/entities/trading_models.py** (199줄)
   - Order, LivePosition, SafetyState, DailyPerformance 엔티티
   - OrderSide, OrderType, OrderStatus, ExitReason 열거형
   - JSON 직렬화/역직렬화 지원 (to_dict, from_dict)
   - PnL 계산 속성 (unrealized_pnl, unrealized_pnl_pct)

2. **src/adapters/kis_order_adapter.py** (~280줄)
   - KisOrderAdapter: OrderPort + BalancePort + CurrentPricePort 구현
   - mojito2 라이브러리 래핑 (8개 메서드)
   - 모의/실전 모드 자동 전환 (MODEL 환경변수)
   - 응답 파싱 및 Order 엔티티 변환

3. **src/use_cases/position_tracker.py** (~320줄)
   - PositionTracker: 보유 포지션 관리 + 청산 조건 판정
   - 4단계 부분청산 (2R/4R/8R/10R, 각 25%)
   - 트레일링 스탑, 손절, 최대보유일 체크
   - data/positions.json 영속화 (로드/저장)

4. **src/use_cases/safety_guard.py** (~250줄)
   - SafetyGuard: 다층 안전장치
   - STOP.signal 파일 기반 매매 중단
   - 일일 손실 -3% / 총 손실 -10% 체크
   - 긴급 전량 청산 (emergency_liquidate)
   - 공휴일/주말 감지 (holidays 라이브러리)

5. **src/use_cases/live_trading.py** (~400줄)
   - LiveTradingEngine: 시그널→주문→모니터링 루프
   - execute_buy_signals(): 시그널 정렬 → 주문 → 포지션 등록
   - execute_sell_signals(): 청산 조건 판정 → 매도
   - monitor_loop(): 1분 간격 현재가 갱신 + 부분청산/손절 체크

6. **scripts/daily_scheduler.py** (~600줄)
   - DailyScheduler: 8 Phase 일일 자동 실행
   - Phase 1~8: 리셋 → 매크로 → 뉴스 → 준비 → 매수 → 모니터 → 매도 → 파이프라인
   - schedule 라이브러리 기반 시간 트리거
   - --dry-run, --run-now 옵션 지원

##### 기존 파일 강화 (6개)

1. **src/use_cases/ports.py** (+67줄)
   - OrderPort: 6개 메서드 (buy/sell limit/market, cancel, modify, get_status)
   - BalancePort: 3개 메서드 (fetch_balance, fetch_holdings, get_available_cash)
   - CurrentPricePort: 1개 메서드 (fetch_current_price)

2. **config/settings.yaml** (+45줄)
   - live_trading 섹션: 8단계 스케줄 (00:00~15:35)
   - 주문 설정 (지정가/시장가, 슬리피지, 재시도)
   - 포지션 제한 (최대 4개, 단일종목 40%, 초기자본 5천만원)
   - 안전장치 (일일 -3%, 총 -10%, STOP.signal)
   - 모니터링 (1분 간격, 부분청산/트레일링 체크)

3. **main.py** (+50줄)
   - stock_buy, stock_sell, monitor, scheduler, positions, emergency-stop, balance 커맨드
   - argparse subparser 구조

4. **src/telegram_formatter.py** (+60줄)
   - format_order_result(), format_position_summary(), format_daily_performance()
   - format_emergency_alert(), format_scheduler_status()

5. **src/telegram_sender.py** (+40줄)
   - send_order_result(), send_position_summary(), send_daily_performance()
   - send_emergency_alert(), send_scheduler_status()

6. **data/** (3개 JSON 파일)
   - positions.json: 보유 포지션 영속 저장
   - trades_history.json: 체결 이력
   - daily_performance.json: 일일 성과

#### 설계 원칙

1. **클린 아키텍처 완전 준수**: entities → use_cases/ports → adapters
2. **의존성 역전**: Port 인터페이스 기반 느슨한 결합
3. **기존 코드 재활용**: backtest_engine, signal_engine, position_sizer 통합
4. **안전 최우선**: STOP.signal, 일일/총 손실, 공휴일 감지

#### 통계

- 신규 파일: 6개 (~1,200 LOC)
- 수정 파일: 6개 (~260 LOC)
- 총 추가 LOC: ~1,460줄
- 클래스: 9개
- Port 인터페이스: 3개
- CLI 커맨드: 7개
- 설계 일치율: 100% (14/14 Step)

#### 문서

- **완료 리포트**: docs/04-report/v4.0-live-trading-system.report.md
- **설계 문서**: C:\Users\ASUS\.claude\plans\binary-weaving-shore.md

#### 다음 단계

1. 모의 투자 7일 테스트
2. 지수 백오프 재시도 로직 추가
3. 웹 모니터링 대시보드 (v4.1)
4. 포지션 동기화 검증 강화
5. 실전 매매 운영

---

## [v3.1] - 2026-02-13

### News Gate + Smart Money v2 + Grok 통합 업그레이드

**Status**: ✅ 완료 (APPROVED)

#### 주요 추가 기능

##### 신규 모듈 (7개)

1. **src/entities/news_models.py**
   - 뉴스 관련 엔티티 9개: NewsGrade, NewsCategory, NewsItem, NewsGateResult, EventPosition, AccumulationSignal, DivergenceSignal, EventDrivenAction
   - 의존성 없음 (클린 아키텍처 최내층)

2. **src/news_classifier.py**
   - A/B/C 등급 분류: 4/4 체크리스트 (A급), 2/4 체크리스트 (B급)
   - 파라미터 오버라이드: RR 1.2 (A급), RSI 65 (A급)

3. **src/divergence_scanner.py**
   - OBV 다이버전스 자동 감지
   - 불리시 (주가↓+OBV↑) / 베어리시 (주가↑+OBV↓)
   - 선형 회귀 기반 추세 판정

4. **src/accumulation_detector.py**
   - 매집 3단계 감지 시스템
   - Phase 1: OBV 다이버전스 (+5점)
   - Phase 2: 기관 5일 연속 순매수 (+10점)
   - Phase 3: 동시지표 10일 이상 (+15점)
   - 투매 감지: -20점 최우선 체크

5. **src/event_position.py**
   - 이벤트 드리븐 포지션 관리
   - 사이즈: 소형주 2%, 대형주 3%
   - 청산 조건 6가지: TARGET_1, TARGET_2, STOP_LOSS, RSI_70, 약한신호, 타임스탑

6. **src/adapters/grok_news_adapter.py**
   - Grok Responses API (grok-4-1-fast) 래퍼
   - web_search + x_search 통합
   - 호재/악재/이슈/변동 자동 분류

7. **src/use_cases/news_gate.py**
   - L-1 News Gate 유스케이스
   - 진입 제약 검사: 갭업 15%, RSI 70, RR 1.2
   - 동시 포지션 제약: 최대 2개
   - Watchlist 관리

##### 기존 파일 강화 (8개)

1. **config/settings.yaml**
   - news_gate 섹션: grade_a (rr_min: 1.2, rsi_max: 65), grade_b (watchlist), grade_c (ignore)
   - smart_money_v2 섹션: 매집 보너스 (5/10/15), 다이버전스 보너스 (5), 투매 페널티 (-20)

2. **src/signal_engine.py** (핵심 통합)
   - AccumulationDetector, DivergenceScanner 초기화
   - set_news_gate() 메서드 추가
   - L-1 News Gate 블록 추가 (맨 처음)
   - L4 Smart Money 강화: 매집 + 다이버전스 보너스 누적
   - L5 위험관리 조정: 이벤트 드리븐시 RR/RSI 완화

3. **src/signal_diagnostic.py**
   - LAYER_NAMES에 "L-1_news_gate" 추가
   - 6-Layer → 7-Layer 진단

4. **src/smart_money.py**
   - calc_enhanced_smart_z(): 기본 SmartZ + 매집 보너스 + 다이버전스 보너스
   - calc_institutional_streak(): 기관 순매수 연속 일수

5. **src/indicators.py** (지표 확충)
   - TRIX (12, 9): 모멘텀 지표
   - TRIX Golden Cross: TRIX > Signal Line
   - 볼린저밴드 (20, 2σ): bb_upper, bb_lower, bb_width, bb_position
   - MACD (12, 26, 9): macd, macd_signal, macd_histogram
   - inst_streak, foreign_streak: 수급 연속 일수
   - gap_up_pct: 갭업 비율

6. **src/telegram_formatter.py**
   - format_news_alert(): 등급/종목/가격/뉴스/필터결과/진입가/목표가/손절가
   - format_accumulation_alert(): 매집단계/신뢰도/수급패턴

7. **src/telegram_sender.py**
   - send_news_alert(): 뉴스 알림 전송
   - send_accumulation_alert(): 매집 알림 전송

8. **main.py** (CLI 확장)
   - --news-grade A:"뉴스내용": 수동 등급 지정
   - --news-scan: Grok API 뉴스 자동 스캔
   - --smart-scan: 전종목 매집 단계 스캔
   - --sample 호환성 유지 (뉴스 없이 v3.0 동일)

#### 설계 원칙

1. **뉴스 오버라이드 금지**: v3.0 기본 필터 유지, 파라미터 튜닝만 가능
2. **위험 제어**: 갭업 15% 이상 진입 금지, 동시 이벤트 포지션 최대 2개
3. **클린 아키텍처**: entities → use_cases → adapters → signal_engine 계층 분리
4. **호환성 보장**: --sample 테스트 통과 (기존 v3.0 동일 동작)

#### 파이프라인 구조 (v3.1)

```
L-1: News Gate (신규) — 뉴스 등급 → 파라미터 조정
L0: Pre-gate — 매출/거래대금/수익성 필터
L1: Grade — Zone Score + Grade + Gap Up %
L2: HMM — Accumulation 모드
L3: OU Filter — z-score, half-life, SNR
L4: Smart Money (강화) — 매집 3단계 + OBV 다이버전스
L5: Risk (조정) — RR, RSI, Position Size (뉴스 등급 반영)
L6: Trigger — Impulse, Confirm, Breakout
```

#### 테스트 결과

- ✅ --sample 테스트: 52건 거래, 50% 승률 (v3.0 호환성 100%)
- ✅ L-1_news_gate: A급 3건, B급 5건, C급 2건 (통과율 100%)
- ✅ 7-Layer 진단: 모든 레이어 정상 출력
- ✅ 설계 매치율: 95% (부분 분산: 기능적 완벽)

#### 문서

- **완료 리포트**: docs/04-report/v3.1-news-gate-smart-money-grok-upgrade.report.md
- **갭 분석**: docs/03-analysis/v3.1-upgrade-gap.analysis.md
- **계획 문서**: C:\Users\ASUS\.claude\plans\binary-weaving-shore.md

#### 다음 단계

1. Grok API 실제 통합 테스트
2. 매집 3단계 파라미터 최적화 (실제 데이터)
3. 이벤트 포지션 12개월 백테스트
4. 추가 뉴스 소스 통합 (네이버, 증권사)

---

## [v3.0] - 2025년 (이전)

### 기존 6-Layer Pipeline 유지

- L0: Pre-gate
- L1: Grade
- L2: HMM Regime
- L3: OU Filter
- L4: Smart Money
- L5: Risk Management
- L6: Trigger

---

## 버전 관리 정책

- **Major**: 파이프라인 구조 변경 (v3 → v4)
- **Minor**: 신규 기능 추가 (v3.0 → v3.1)
- **Patch**: 버그 수정, 성능 개선 (v3.1.0 → v3.1.1)

---

**마지막 업데이트**: 2026-02-13
**상태**: Active
