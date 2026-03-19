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


## 🟢 현재 상태: STEP 0 완료 → STEP 1 진행 중
## 마지막 완료: STEP 0 사전 준비 (2026-03-19)


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

- [ ] 1-1. 독립 백테스트 스크립트 작성
  - 파일: `scripts/v2_factor_independent_backtest.py`
  - 로직: 
    - 기존 backtest_engine.py 활용
    - 5축(S1~S5) 각각을 단독 팩터로 사용하여 백테스트
    - S1(에너지소진) 단독: S1 스코어 상위 10% 매수 → X1~X5 매도
    - S2(밸류에이션) 단독: S2 스코어 상위 10% 매수 → X1~X5 매도
    - S3(OU평균회귀) 단독: 동일
    - S4(모멘텀감속) 단독: 동일
    - S5(수급/스마트머니) 단독: 동일
    - 현재 합산(5축 가중합) 그대로 백테스트 (기준선)
  - 출력: 각 축별 Sharpe, PF, MDD, 승률, 거래횟수
  - 기간: 전체 parquet 데이터 (3~5년)
  - 완료일: ____  커밋: ____

- [ ] 1-2. 백테스트 실행 및 결과 저장
  - 실행: `python scripts/v2_factor_independent_backtest.py`
  - 결과 저장: `data/v2_migration/factor_independent_results.json`
  - 결과 형식:
  ```json
  {
    "S1_energy_exhaustion": {"sharpe": ?, "pf": ?, "mdd": ?, "win_rate": ?, "trades": ?},
    "S2_valuation":         {"sharpe": ?, "pf": ?, "mdd": ?, "win_rate": ?, "trades": ?},
    "S3_ou_reversion":      {"sharpe": ?, "pf": ?, "mdd": ?, "win_rate": ?, "trades": ?},
    "S4_momentum_decel":    {"sharpe": ?, "pf": ?, "mdd": ?, "win_rate": ?, "trades": ?},
    "S5_supply_demand":     {"sharpe": ?, "pf": ?, "mdd": ?, "win_rate": ?, "trades": ?},
    "current_combined":     {"sharpe": ?, "pf": ?, "mdd": ?, "win_rate": ?, "trades": ?}
  }
  ```
  - 완료일: ____  커밋: ____

- [ ] 1-3. 결과 분석 리포트 생성
  - 텔레그램으로 결과 요약 발송
  - 핵심 판단:
    - S5(수급) 단독 PF가 1.3 이상인가? → 수급 가중치 상향 근거
    - S1(에너지소진) 단독 PF가 S5보다 낮은가? → 가중치 하향 근거
    - 어떤 축이 가장 높은 Sharpe를 보이나?
  - 완료일: ____  커밋: ____


# ═══════════════════════════════════════════════════════
# STEP 2: 레짐별 팩터 가중치 재편
# ═══════════════════════════════════════════════════════
# 목적: STEP 1 결과를 기반으로, 레짐별로 가중치를 다르게 적용
# 전제: STEP 1 완료 필수
# 예상 소요: 1~2 세션

- [ ] 2-1. V2 팩터 가중치 결정 (STEP 1 데이터 기반)
  - STEP 1 결과를 읽어서:
    - 각 축의 단독 Sharpe 비율로 기본 가중치 산출
    - BULL 레짐: 모멘텀(S4)+수급(S5) 비중 상향
    - BEAR 레짐: 에너지소진(S1)+밸류(S2)+OU(S3) 비중 상향
  - settings.yaml의 alpha_v2.factor_weights 업데이트
  - 완료일: ____  커밋: ____

- [ ] 2-2. 레짐 조건부 스코어러 구현
  - 파일: `src/alpha/factors/regime_weighted_scorer.py`
  - 로직:
    - 현재 레짐 판별 (`src/alpha/regime.py` 활용)
    - 레짐에 따라 settings.yaml에서 가중치 로드
    - 기존 v8_scorers.py의 5축 점수를 받아서 재가중합
    - 출력: regime_weighted_score (0.0~1.0)
  - 기존 v8_scorers.py는 수정하지 않음 (V2 토글 OFF면 기존대로)
  - 완료일: ____  커밋: ____

- [ ] 2-3. 레짐별 가중치 백테스트 검증
  - 기존 5축 가중합(기준선) vs 레짐별 동적 가중합(V2) A/B 비교
  - Walk-Forward 검증 (`src/walk_forward.py` 활용)
  - V2 Sharpe > 기존 Sharpe 확인
  - V2 MDD < 기존 MDD 확인
  - 결과 저장: `data/v2_migration/regime_weight_comparison.json`
  - 완료일: ____  커밋: ____

- [ ] 2-4. 검증 통과 시 scan_buy_candidates.py에 V2 분기 추가
  - `if settings.alpha_v2.enabled:` 분기로 V2 스코어러 호출
  - 기존 로직은 else에 유지 (안전한 토글)
  - 완료일: ____  커밋: ____


# ═══════════════════════════════════════════════════════
# STEP 3: 퀄리티 팩터 구축 (신규)
# ═══════════════════════════════════════════════════════
# 목적: V2의 4번째 팩터(Quality) 데이터+로직 전면 구축
# 전제: STEP 2 완료 불필요 (병렬 가능 — 데이터 파이프라인이라서)
# 예상 소요: 2~3 세션

- [ ] 3-1. DART 재무 파이프라인 확장
  - 파일: `src/adapters/dart_financial_adapter.py` (신규 또는 dart_adapter.py 확장)
  - 추가 수집 항목:
    - 분기별 ROE (최근 8분기)
    - 총부채 / 총자산
    - 영업현금흐름 / 순이익 (Accruals Ratio)
    - 연간 배당금 / 순이익 (배당성향)
    - EBITDA (V팩터용)
    - Free Cash Flow (V팩터용)
  - 저장: `data/v2_migration/financial_quarterly.json` 또는 SQLite 테이블
  - 유니버스 84종목 전체 대상
  - 완료일: ____  커밋: ____

- [ ] 3-2. Q1: ROE 안정성 서브팩터
  - 파일: `src/alpha/factors/quality_roe.py`
  - 로직: 최근 8분기 ROE의 (평균 / 표준편차) → Z-Score
  - 높으면서 안정적인 ROE = 높은 점수
  - 백테스트: 단독 PF > 1.1 확인
  - 완료일: ____  커밋: ____

- [ ] 3-3. Q2: 부채 건전성 서브팩터
  - 파일: `src/alpha/factors/quality_debt.py`
  - 로직: 1 - (총부채/총자산) → Z-Score
  - 낮은 레버리지 = 높은 점수
  - 백테스트: 단독 PF > 1.1 확인
  - 완료일: ____  커밋: ____

- [ ] 3-4. Q3: 이익 품질 서브팩터
  - 파일: `src/alpha/factors/quality_accruals.py`
  - 로직: 영업CF / 순이익 → Z-Score
  - >1이면 이익이 현금으로 뒷받침 = 높은 점수
  - 백테스트: 단독 PF > 1.1 확인
  - 완료일: ____  커밋: ____

- [ ] 3-5. Q4: 배당 지속성 서브팩터
  - 파일: `src/alpha/factors/quality_dividend.py`
  - 로직: 배당금/순이익 (0~80% 범위만 유효) → Z-Score
  - 백테스트: 단독 PF > 1.1 확인
  - 완료일: ____  커밋: ____

- [ ] 3-6. Q 통합 스코어
  - 파일: `src/alpha/factors/quality_composite.py`
  - 로직: Z(Q1)×0.35 + Z(Q2)×0.25 + Z(Q3)×0.25 + Z(Q4)×0.15
  - 통합 백테스트: PF > 1.2, Sharpe > 0.8
  - 완료일: ____  커밋: ____


# ═══════════════════════════════════════════════════════
# STEP 4: 밸류 팩터 보완 (기존 S2 확장)
# ═══════════════════════════════════════════════════════
# 목적: 기존 PER/PBR에 EBITDA/EV, FCF Yield 추가
# 전제: STEP 3-1 완료 필수 (DART 파이프라인 공유)
# 예상 소요: 1 세션

- [ ] 4-1. V1: EBITDA/EV 서브팩터
  - 파일: `src/alpha/factors/value_ebitda_ev.py`
  - 로직: EBITDA / (시가총액 + 순부채) → Z-Score
  - DART 데이터 + 시가총액(pykrx)
  - 완료일: ____  커밋: ____

- [ ] 4-2. V2: FCF Yield 서브팩터
  - 파일: `src/alpha/factors/value_fcf_yield.py`
  - 로직: Free Cash Flow / 시가총액 → Z-Score
  - 완료일: ____  커밋: ____

- [ ] 4-3. V 통합 스코어 (기존 S2 대체)
  - 파일: `src/alpha/factors/value_composite.py`
  - 기존 S2(PER할인+PBR백분위+Forward PER+턴어라운드) +
    신규(EBITDA/EV + FCF Yield) 통합
  - 백테스트: 기존 S2 단독 vs 확장 V 스코어 비교
  - 완료일: ____  커밋: ____


# ═══════════════════════════════════════════════════════
# STEP 5: L3 SIZE 엔진 강화
# ═══════════════════════════════════════════════════════
# 목적: 기존 ATR 사이징에 Kelly + 상관관계 추가
# 전제: STEP 1 완료 (적중률 데이터 필요)
# 예상 소요: 1~2 세션

- [ ] 5-1. 상관관계 매트릭스 구축
  - 파일: `src/alpha/factors/correlation_matrix.py`
  - 로직: 84종목 60일 롤링 상관계수 매트릭스
  - 매일 장 마감 후 업데이트 (BAT-D에 추가)
  - 저장: `data/v2_migration/correlation_matrix.json`
  - 완료일: ____  커밋: ____

- [ ] 5-2. 상관관계 기반 사이즈 감산
  - position_sizer.py 확장 (V2 분기)
  - if corr(A,B) > 0.7: size_A *= 0.7, size_B *= 0.7
  - 동일 섹터 합산 비중 30% 상한
  - 완료일: ____  커밋: ____

- [ ] 5-3. Half Kelly 사이징 (선택사항)
  - position_sizer.py에 Kelly 모드 추가
  - 전제: signal_accuracy.json의 적중률 데이터 활용
  - Kelly% = (win_rate × avg_win - loss_rate × avg_loss) / avg_win
  - Half_Kelly = Kelly% / 2
  - final_size = min(atr_size, half_kelly_size, regime_max)
  - alpha_v2.sizing.use_kelly = true로 활성화
  - 완료일: ____  커밋: ____


# ═══════════════════════════════════════════════════════
# STEP 6: 통합 + Paper Trading
# ═══════════════════════════════════════════════════════
# 목적: STEP 1~5 결과를 통합하고 Paper로 검증
# 전제: STEP 2 + STEP 3(또는 4) 중 하나 이상 완료
# 예상 소요: 4주 (자동 운영)

- [ ] 6-1. V2 통합 스코어러 완성
  - regime_weighted_scorer.py에 Q, V 확장 팩터 통합
  - 4-팩터 합성: SD + M + V + Q (레짐별 가중치)
  - 기존 5축은 SD/M/V의 서브팩터로 매핑
  - 완료일: ____  커밋: ____

- [ ] 6-2. V2 통합 백테스트
  - 기존 시스템(v8) vs V2 전체 A/B 비교
  - Walk-Forward + Out-of-Sample
  - 기준: V2 Sharpe > v8 Sharpe AND V2 MDD < v8 MDD
  - 결과 저장: `data/v2_migration/v2_vs_v8_comparison.json`
  - 완료일: ____  커밋: ____

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
| | | | | |
