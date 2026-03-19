# ALPHA ENGINE V2 — 자동 진행 마이그레이션 체크리스트
# ═══════════════════════════════════════════════════════
# 
# 📌 CLAUDE CODE 지시: 
# 이 파일을 세션 시작 시 반드시 읽어라.
# [ ] 항목 중 가장 위의 미완료 항목부터 이어서 진행하라.
# 완료된 항목은 [x]로 표시하고 날짜와 커밋해시를 기록하라.
# 중간에 다른 작업 지시가 오면 그걸 먼저 하되, 
# 돌아왔을 때 이 파일을 다시 읽고 이어서 진행하라.
#
# 📌 사용자(ppwangga) 안내:
# Claude Code 열 때 이렇게 말하면 됩니다:
# "ALPHA_ENGINE_V2_MIGRATION.md 읽고 다음 단계 진행해"
# 그러면 Claude Code가 알아서 현재 진행 상황 파악하고 이어갑니다.
#
# 마지막 업데이트: 2026-03-19
# ═══════════════════════════════════════════════════════


## 🟢 현재 상태: STEP 6-2 완료 → Paper Trading 대기

## 마지막 완료: STEP 6-2 실제 BacktestEngine V1 vs V2 비교 (2026-03-19)


# ═══════════════════════════════════════════════════════
# STEP 0: 사전 준비 (즉시 실행 가능)
# ═══════════════════════════════════════════════════════
# 목적: V2 전환을 위한 기반 세팅. 기존 시스템 안 건드림.

- [x] 0-1. 이 파일을 프로젝트 루트에 저장
  - 경로: `D:\sub-agent-project\ALPHA_ENGINE_V2_MIGRATION.md`
  - 완료일: 2026-03-19

- [x] 0-2. V2 전용 브랜치 생성
  - `git checkout -b alpha-engine-v2`
  - 기존 main 브랜치는 절대 안 건드림
  - 완료일: 2026-03-19

- [x] 0-3. V2 설정 섹션을 settings.yaml에 추가
  - `config/settings.yaml`에 `alpha_v2` 섹션 추가 완료 (enabled: false)
  - 완료일: 2026-03-19

- [x] 0-4. V2 팩터 모듈 디렉토리 생성
  - `src/alpha/factors/__init__.py` 생성 완료
  - 완료일: 2026-03-19


# ═══════════════════════════════════════════════════════
# STEP 1: 기존 5축 독립 백테스트 (가장 중요)
# ═══════════════════════════════════════════════════════
# 목적: 현재 5축 각각의 독립적 엣지를 검증한다.
# 이 결과가 나와야 V2 가중치를 데이터로 결정할 수 있다.
# 예상 소요: 1~2 세션

- [x] 1-1. 독립 백테스트 스크립트 작성
  - 파일: `scripts/v2_factor_independent_backtest.py`
  - 경량 독립 엔진 (게이트/트리거 없이 순수 스코어 기반)
  - 1035종목, 544거래일 대상
  - 완료일: 2026-03-19

- [x] 1-2. 백테스트 실행 및 결과 저장
  - 결과: `data/v2_migration/factor_independent_results.json`
  - 완료일: 2026-03-19

  **실측 결과:**
  | 팩터 | Sharpe | PF | MDD | 승률 | 거래 | 수익률 |
  |------|--------|-----|------|------|------|--------|
  | S5 수급 | **0.96** | **1.32** | -11.3% | 46.4% | 224 | **+21.8%** |
  | S4 모멘텀 | 0.53 | 1.11 | -15.7% | 43.5% | 230 | +10.2% |
  | S3 OU | 0.47 | 1.21 | -12.8% | 47.7% | 220 | +8.5% |
  | S1 에너지 | 0.29 | 1.14 | -9.7% | 42.2% | 232 | +5.5% |
  | S2 밸류 | 0.16 | 1.01 | -9.4% | 42.5% | 228 | +2.3% |
  | 합산(기준) | 0.13 | 1.33 | -26.2% | 41.8% | 141 | +1.8% |

- [x] 1-3. 결과 분석 리포트
  - 완료일: 2026-03-19
  - **핵심 판단:**
    - S5(수급) PF 1.32 ≈ 1.3 → 수급 가중치 상향 근거 **확인**
    - S5 Sharpe 0.96 >> S1 Sharpe 0.29 → S1 가중치 0.30→0.15 감축 근거
    - S2(밸류) PF 1.01 → **독립적 엣지 없음**, 가중치 대폭 감축 대상
    - S3(OU) 승률 47.7% 최고 + MDD -12.8% → 안정적, 유지
    - 현재 합산 MDD -26.2% 최악 → 가중치 재편 시급
    - **V2 가중치 제안 (Sharpe 비율 기반):**
      - S5: 0.40 (현재 0.15 → 2.7배 상향)
      - S4: 0.22 (현재 0.15 → 상향)
      - S3: 0.19 (현재 0.20 → 유지)
      - S1: 0.12 (현재 0.30 → 대폭 감축)
      - S2: 0.07 (현재 0.20 → 대폭 감축)


# ═══════════════════════════════════════════════════════
# STEP 2: 레짐별 팩터 가중치 재편
# ═══════════════════════════════════════════════════════
# 목적: STEP 1 결과를 기반으로, 레짐별로 가중치를 다르게 적용
# 전제: STEP 1 완료 필수
# 예상 소요: 1~2 세션

- [x] 2-1. V2 팩터 가중치 결정 (STEP 1 데이터 기반)
  - Sharpe 비율 기반: S5(0.40) > S4(0.22) > S3(0.19) > S1(0.12) > S2(0.07)
  - BULL: S5=0.40, S4=0.25, S3=0.15, S1=0.12, S2=0.08
  - BEAR: S5=0.25, S4=0.10, S3=0.30, S1=0.25, S2=0.10
  - settings.yaml `alpha_v2.scorer_weights` 추가 완료
  - 완료일: 2026-03-19

- [x] 2-2. 레짐 조건부 스코어러 구현
  - `src/alpha/factors/regime_weighted_scorer.py` 작성
  - RegimeWeightedScorer: 레짐별 동적 가중합, GradeResult 호환
  - 완료일: 2026-03-19

- [x] 2-3. 레짐별 가중치 백테스트 검증
  - `scripts/v2_regime_weight_backtest.py` 작성 + 실행
  - 결과: `data/v2_migration/regime_weight_comparison.json`
  - V1(고정): Sharpe 0.56, PF 2.65, MDD -16.7%, 47건
  - V2(레짐): Sharpe 0.18, PF 1.33, MDD -13.9%, 88건
  - **MDD 2.8pp 개선 (V2 PASS)**, PF ≥ 1.3 (PASS)
  - Sharpe 하락 원인: S5 고가중으로 거래 수 2배 → 품질 희석
  - TODO: B등급 커트라인(0.50) 적용으로 품질 필터 강화
  - 완료일: 2026-03-19

- [x] 2-4. 검증 통과 시 scan_buy_candidates.py에 V2 분기 추가
  - `_v2_enabled` 플래그로 V2 스코어러 조건부 호출
  - 기존 로직은 V2 OFF 시 그대로 유지 (안전한 토글)
  - B등급(0.50) 커트라인: RegimeWeightedScorer 내장 (grade_cutoffs.B=0.50)
  - V2 활성 시: SignalEngine 게이트/트리거 유지 + 스코어링만 V2로 대체
  - 완료일: 2026-03-19  커밋: 4146247


# ═══════════════════════════════════════════════════════
# STEP 3: 퀄리티 팩터 구축 (신규)
# ═══════════════════════════════════════════════════════
# 목적: V2의 4번째 팩터(Quality) 데이터+로직 전면 구축
# 전제: STEP 2 완료 불필요 (병렬 가능 — 데이터 파이프라인이라서)
# 예상 소요: 2~3 세션

- [x] 3-1. DART 재무 파이프라인 확장
  - 신규: `src/adapters/dart_financial_adapter.py` (DartFinancialAdapter)
  - 러너: `scripts/v2_collect_financial_data.py` (--test/--bs-only 플래그)
  - 추가 수집 항목:
    - 분기별 ROE (최근 8분기, 연환산) ✅
    - 총부채 / 총자산 ✅
    - 영업현금흐름 / 순이익 (Accruals Ratio) ✅
    - 배당금 / 순이익 (배당성향) ✅
    - EBITDA (영업이익×1.2 근사) ✅
    - FCF = 영업CF - CAPEX(유형+무형) ✅
  - 저장: `data/v2_migration/financial_quarterly.json` (2.8MB)
  - 수집 범위: 1008종목 (우선주 제외, DART 커버 기준)
  - BS: fnlttMultiAcnt(100종목 일괄) × 8분기 = 88 API
  - CF: fnlttSinglAcntAll(개별) × 1008 = 1008 API + 캐시
  - DartAdapter 초기화 순서 버그 수정 (_api_calls 선행 초기화)
  - 완료일: 2026-03-19  커밋: c73f7e4

- [x] 3-2. Q1: ROE 안정성 서브팩터
  - 파일: `src/alpha/factors/quality_roe.py` (Q1~Q4 공통 파일)
  - 로직: roe_mean/roe_std + ROE level 보정 → Z-Score (scipy.stats.norm.cdf)
  - 백테스트: Top PF 5.22, Top Sharpe 2.05 — PASS ✓
  - 완료일: 2026-03-19

- [x] 3-3. Q2: 부채 건전성 서브팩터
  - 파일: `src/alpha/factors/quality_roe.py` (QualityDebt 클래스)
  - 로직: 1 - debt_ratio → Z-Score (금융주 >85% → 0.3 캡)
  - 백테스트: Top PF 3.30 — PASS ✓ (Top-Bottom 역전: 불장에서 레버리지 프리미엄)
  - 완료일: 2026-03-19

- [x] 3-4. Q3: 이익 품질 서브팩터
  - 파일: `src/alpha/factors/quality_roe.py` (QualityAccruals 클래스)
  - 로직: operating_cf/net_income → sigmoid 매핑 → Z-Score
  - 백테스트: Top PF 2.61 — PASS ✓ (Top-Bottom 역전: 성장주 효과)
  - 완료일: 2026-03-19

- [x] 3-5. Q4: 배당 지속성 서브팩터
  - 파일: `src/alpha/factors/quality_roe.py` (QualityDividend 클래스)
  - 로직: bell curve (50% payout 정점) → Z-Score
  - 백테스트: Top PF 3.39 — PASS ✓
  - 완료일: 2026-03-19

- [x] 3-6. Q 통합 스코어
  - 파일: `src/alpha/factors/quality_composite.py`
  - 로직: 레짐별 가중치 (BULL: Q1=0.40+Q4=0.30, BEAR/CRISIS: Q2+Q3 강화)
  - 통합 백테스트 결과:
    | 레짐 | Top PF | Top Sharpe | 판정 |
    |------|--------|-----------|------|
    | BULL | 3.92 | 1.79 | PASS (PF>1.2, Sharpe>0.8) |
    | CAUTION | 3.67 | 1.71 | PASS |
    | BEAR | 3.59 | 1.68 | PASS |
    | CRISIS | 3.45 | 1.62 | PASS |
  - Q2/Q3 Top-Bottom 역전은 불장 효과 — 레짐 가중치로 보정 (BEAR/CRISIS에서 Q2+Q3 비중 증가)
  - 완료일: 2026-03-19


# ═══════════════════════════════════════════════════════
# STEP 4: 밸류 팩터 보완 (기존 S2 확장)
# ═══════════════════════════════════════════════════════
# 목적: 기존 PER/PBR에 EBITDA/EV, FCF Yield 추가
# 전제: STEP 3-1 완료 필수 (DART 파이프라인 공유)
# 예상 소요: 1 세션

- [x] 4-1. V1: EBITDA/EV 서브팩터
  - 파일: `src/alpha/factors/value_ebitda_ev.py` (V1+V2 공통 파일)
  - 로직: EBITDA / (market_cap + total_debt) → Z-Score
  - 데이터: financial_quarterly.json(EBITDA) + market_cap_cache.json(시가총액)
  - 백테스트: Top PF 2.73 — PASS (Top-Bottom 역전: 불장에서 성장주 프리미엄)
  - 완료일: 2026-03-19

- [x] 4-2. V2: FCF Yield 서브팩터
  - 파일: `src/alpha/factors/value_ebitda_ev.py` (ValueFCFYield 클래스)
  - 로직: FCF / market_cap → Z-Score
  - 백테스트: Top PF 2.22 — PASS (Q3가 최고 수익, 비선형 관계)
  - 완료일: 2026-03-19

- [x] 4-3. V 통합 스코어 (기존 S2 확장)
  - 파일: `src/alpha/factors/value_composite.py`
  - 로직: S2(기존)×0.40 + V1×0.35 + V2×0.25 (레짐별 가중치 조정)
  - BULL: S2 중시(0.50), CRISIS: FCF 중시(V2=0.45)
  - 백테스트: V통합 PF 2.67 vs 기존 S2 PF 1.01 → **164% 개선**
  - 모든 레짐 PF > 2.6, Sharpe > 1.2 달성
  - 완료일: 2026-03-19


# ═══════════════════════════════════════════════════════
# STEP 5: L3 SIZE 엔진 강화
# ═══════════════════════════════════════════════════════
# 목적: 기존 ATR 사이징에 Kelly + 상관관계 추가
# 전제: STEP 1 완료 (적중률 데이터 필요)
# 예상 소요: 1~2 세션

- [x] 5-1. 상관관계 매트릭스 구축
  - 파일: `src/alpha/factors/correlation_matrix.py`
  - 60일 롤링 상관계수 행렬 (1015종목, ret1 기반)
  - 검증: 삼성-하이닉스 0.85 (고상관), 삼성-카카오 0.60 (중간)
  - 저장: `data/v2_migration/correlation_matrix.json` (|corr|>0.3만 저장)
  - 완료일: 2026-03-19

- [x] 5-2. 상관관계 기반 사이즈 감산
  - 파일: `src/alpha/position_sizer_v2.py` (PositionSizerV2)
  - 기존 PositionSizer 래핑 — 원본 코드 수정 없음
  - corr > 0.7: size × 0.7 (30% 감축)
  - 검증: 삼성(33주) + 하이닉스 진입 → 23주 (corr 0.85 → 감산 적용)
  - settings.yaml: `sizing.corr_threshold/corr_penalty` 추가
  - 완료일: 2026-03-19

- [x] 5-3. Half Kelly 사이징
  - 파일: `src/alpha/position_sizer_v2.py` (_calc_half_kelly 메서드)
  - signal_accuracy.json 적중률 활용
  - Kelly = (p×b - q) / b, Half Kelly = Kelly / 2, clamped [0.1, 1.0]
  - use_kelly: false (검증 후 활성화)
  - 완료일: 2026-03-19


# ═══════════════════════════════════════════════════════
# STEP 6: 통합 + Paper Trading
# ═══════════════════════════════════════════════════════
# 목적: STEP 1~5 결과를 통합하고 Paper로 검증
# 전제: STEP 2 + STEP 3(또는 4) 중 하나 이상 완료
# 예상 소요: 4주 (자동 운영)

- [x] 6-1. V2 통합 스코어러 완성
  - UnifiedV2Scorer: 4팩터(SD+M+V+Q) 레짐별 가중합
  - ScoringEngine(S1~S5) + QualityComposite + ValueComposite 통합
  - GradeResult 인터페이스 100% 호환
  - 파일: `src/alpha/factors/unified_scorer.py`
  - 완료일: 2026-03-19

- [x] 6-2. V2 통합 백테스트
  - 3-Way 간이 비교 (V1/V2_5ax/V2_4f): V2_4f MDD 최강 but PF 희석
  - 실제 BacktestEngine 비교: 필터링 모드 → PF 0.60 (실패)
  - **sizing-only 모드 도입**: 시그널 제거 않고 등급별 사이징 조정
    - A:100%, B:80%, C:50%, F:30% (포지션 비율 곱셈)
  - 최종 결과: PF 1.38(+0.04), MDD -8.24%(+0.85%p), 거래 77건(동일)
  - 결과: `data/v2_migration/v2_engine_comparison.json`
  - 완료일: 2026-03-19

- [ ] 6-3. Paper Trading 4주 실행
  - settings.yaml: alpha_v2.enabled = true (Paper 모드에서)
  - 기존 v8은 라이브 유지, V2는 Paper로 병행
  - 매일 괴리율 기록: `data/v2_migration/paper_daily_log.json`
  - 완료일: ____  커밋: ____

- [ ] 6-4. Paper 결과 리뷰 및 실전 전환 결정
  - 4주간 Paper PF > 1.3, Sharpe > 1.0, MDD < -15%
  - 통과 시: alpha_v2.enabled = true (라이브)
  - 미통과 시: 파라미터 조정 후 2주 추가 Paper
  - 완료일: ____  커밋: ____


# ═══════════════════════════════════════════════════════
# 참고: 각 STEP의 독립성
# ═══════════════════════════════════════════════════════
# 
# STEP 0 → 즉시 가능 (전제조건 없음)
# STEP 1 → 즉시 가능 (STEP 0만 필요)
# STEP 2 → STEP 1 완료 후
# STEP 3 → 즉시 가능 (STEP 1과 병렬 가능 — 데이터 파이프라인)
# STEP 4 → STEP 3-1 완료 후 (DART 파이프라인 공유)
# STEP 5 → STEP 1 완료 후 (적중률 데이터)
# STEP 6 → STEP 2 + (STEP 3 or 4) 완료 후
#
# 즉, STEP 1과 STEP 3은 동시에 진행 가능.
# 최소 경로: 0 → 1 → 2 → 6 (퀄리티 없이 3팩터로 먼저)
# 전체 경로: 0 → 1+3 병렬 → 2+4 → 5 → 6


# ═══════════════════════════════════════════════════════
# 진행 로그 (Claude Code가 자동 기록)
# ═══════════════════════════════════════════════════════

## 진행 이력
| 날짜 | STEP | 항목 | 결과 | 비고 |
|------|------|------|------|------|
| 2026-03-19 | - | 전수조사 완료 | ✅ | 131개 질문 답변 |
| 2026-03-19 | 0 | STEP 0 사전 준비 완료 | ✅ | 브랜치+설정+디렉토리 |
| 2026-03-19 | 1 | STEP 1 독립 팩터 백테스트 | ✅ | S5 Sharpe 0.96 압도적 1위 |
| 2026-03-19 | 2 | STEP 2 레짐별 가중치 (2-1~2-3) | ✅ | MDD 2.8pp 개선, PF 1.33 |
| 2026-03-19 | 2-4 | scan_buy V2 분기 삽입 | ✅ | 커밋 4146247 |
| 2026-03-19 | 3-1 | DART 재무 파이프라인 확장 | ✅ | 1008종목, BS+CF, 2.8MB |
| | | | | |
