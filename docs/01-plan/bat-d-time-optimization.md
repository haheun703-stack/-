# BAT-D 시간 최적화 — Plan

**작성**: 2026-05-14
**근거**: 5/14 BAT-D 실측 분석 (`cron_20260514.log`)
**목표**: BAT-D 106분 28초 → **70분 이내** (-36분, -34%)

---

## 1. 배경

`128d54b perf: BAT-D 3-봇 분업 적용` 후 5/14 첫 실전:
- 5/13: 119분 58초
- 5/14: 106분 28초
- 효과: **-13분 30초** (목표 -30~40분 대비 미흡)

`collect_investor_bulk` 부분 수집은 **6.5초**로 이미 한계. 진짜 단축 여지는 다른 단계에 있음.

## 2. 5/14 BAT-D 단계별 시간 분포 (실측)

| 구간 | 단계 | 시간 | 비율 |
|------|------|------|------|
| 16:30 ~ 16:49 | update_daily_data + 증분 동기화 | 19분 26초 | 18% |
| 16:49 ~ 17:01 | sync_investor_to_csv + BAT-J | 12분 | 11% |
| 17:01 ~ 17:20 | scan_nationality + 외인소진율 + 섹터수급 + ETF + 차이나머니 + ECOS | 19분 | 18% |
| **17:20 ~ 17:30** | **빈 구간 ⭐⭐⭐ (정체 미확인)** | 10분 40초 | 10% |
| **17:30 ~ 17:42** | **v3 AI Brain Phase 1-5 (LLM 순차)** ⭐⭐⭐ | 12분 | 11% |
| 17:42 ~ 17:51 | ETF 로테이션 + 매집 추적 + Perplexity + AI 두뇌 | 9분 | 8% |
| 17:51 ~ 17:56 | **wisereport 컨센서스 (418종목 × 0.5초)** ⭐⭐ | 5분 51초 | 5% |
| **17:56 ~ 18:07** | **[EA]/[TA] 재무 분석 (stdout 없음)** ⭐⭐⭐ | 11분 6초 | 10% |
| 18:07 ~ 18:16 | 킬러픽 + 밸류에이션 + NXT + 노다지 + 피보나치 + 학습 | 9분 | 9% |

⭐ 표시 = 단축 잠재력 큰 단계

## 3. 식별된 병목 (5개) — P0 완료 (2026-05-14)

### B1. v3 AI Brain Phase 1-5 (실측 7분 55초, 17:34:45 ~ 17:42:40)
- 스크립트: `scripts/run_v3_brain.py` (run_bat.sh L218)
- 5단계 LLM 순차 호출 (o1 Deep / Strategic / Sector / Deep Analyst / Portfolio)
- 외부 API 의존 (Anthropic/OpenAI)
- 코드 위치: `src/agents/` + v3_brain 러너

### B2. [EA]+[TA] 재무 분석 (실측 11분 6초, 17:56:39 ~ 18:07:45) ⭐⭐⭐
- 스크립트 1: `scripts/scan_earnings_acceleration.py` (L254) — [EA]
- 스크립트 2: `scripts/scan_turnaround.py` (L255) — [TA]
- 두 스크립트 순차 실행, 각 재무 1008종목 로드 후 stdout 출력 없음
- **병렬 실행 가능성 검증 필요** (독립적 분석이라면 -5분 가능)

### B3. rebuild_indicators (실측 10분 40초, 17:20:04 ~ 17:30:44) ⭐⭐⭐
- 스크립트: `scripts/rebuild_indicators.py` (L202)
- 단 8줄 코드: `IndicatorEngine().process_all()`
- 모든 종목 parquet 기술지표 재계산
- **단일 프로세스 가능성 매우 높음 → 병렬화 효과 클 듯** (-5분 가능)

### B4. wisereport 컨센서스 (실측 4분 53초, 17:51:31 ~ 17:56:24)
- 스크립트: `scripts/scan_consensus.py` (L244)
- 418종목 × delay 0.5초 = 약 209초 + HTML/PNG 생성
- 외부 wisereport API rate limit 의존
- 출력: 277/418 성공 (66%)

### B5. update_daily_data (실측 19분 18초, 16:30:09 ~ 16:49:27)
- 스크립트: `scripts/update_daily_data.py` (L165)
- 2859 CSV 병렬 처리 (workers=10)
- FinanceDataReader 외부 의존
- 8건 에러 (5/15부터 7건 fix 효과로 감소)

### 추가 발견
- **master_brain** (L217): 2분 14초
- **ai_news_brain** (L242): 2분 5초
- **perplexity_market_intel** (L241): 1분 5초
- **daily_market_learner** (L294, parquet 2회 풀스캔): 45초+

## 4. 단계별 개선 계획

### P0 — 정체 식별 (선행 필수, 30분)
- [ ] B2 17:56~18:07 [EA]/[TA] 정확한 스크립트 위치 식별
- [ ] B3 17:20~17:30 진행 스크립트 식별
- [ ] `run_bat.sh` BAT-D 함수 전체 흐름 도식화 (단계 순서 + 예상 단축 매핑)

### P1 — Low Risk Quick Win (총 -3~5분 예상)
- [ ] **wisereport delay 0.5 → 0.3 단축** (-1분 30초)
- [ ] **wisereport 종목 수 축소** (418 → TOP 200 등, 결과 품질 영향 검토)
- [ ] **AI 두뇌 배치 사이즈 증대** (1144 → 더 큰 배치)

### P2 — Medium Risk (총 -5~8분 예상)
- [ ] **v3 AI Brain Phase 2/3/4 병렬화** (-3~5분)
  - Strategic Brain + Sector Strategist 동시 실행
  - Deep Analyst를 Phase 2/3와 병렬
  - 위험: Portfolio Brain(Phase 5) 의존성 깨질 수 있음
- [ ] **차이나머니 KIS API 호출 병렬화** (이미 17:18:46~17:20:04 = 1.5분이라 효과 적음)

### P3 — High Risk / Effort (총 -10~15분 예상)
- [ ] **[EA]/[TA] 재무 분석 (B2) 최적화** — 정체 식별 후 결정
- [ ] **B3 빈 구간 스크립트 (rebuild_indicators 추정)** 병렬화/캐싱
- [ ] **update_daily_data workers 10 → 20** (CPU/IO 트레이드오프)

## 5. 위험 및 가드레일

| 위험 | 대응 |
|------|------|
| LLM 병렬 호출로 토큰 비용 증가 | 비용 모니터링 (Anthropic 대시보드) |
| Phase 의존성 깨짐 (v3 AI Brain) | 결과 비교 (기존 vs 병렬화) 1주 검증 |
| wisereport 종목 축소로 누락 종목 발생 | TOP 200 외 종목 별도 백업 수집 |
| 외부 API rate limit (wisereport, KIS) | 실패 시 fallback 로직 유지 |
| BAT-PICKV2 17:45 시작 시간 조정 시 의존 BAT 영향 | 단축된 시간만큼 17:30~17:40으로 앞당기기 검토 |

## 6. 검증 방법

- **A/B 비교**: 개선 전후 동일 평일 BAT-D 실행 시간 비교
- **품질 검증**: AI Brain 결과 (Strategic/Sector/Deep) 변경 전후 일치율 80% 이상
- **로그 모니터링**: `_update_log.txt`, `daily_audit_log.md` 매일 추적
- **롤백 기준**: BAT-D 실패 1건 발생 또는 결과 일치율 70% 미만 시 즉시 롤백

## 7. 진행 절차

1. **P0 (식별)**: 다음 세션 또는 즉시 (30분 작업)
2. **P1 (Quick Win)**: P0 완료 후 즉시 — 코드 변경 작음
3. **P2 (병렬화)**: 별개 design 문서 → 구현 → A/B 검증
4. **P3**: P0~P2 효과 측정 후 필요 시 진행

## 8. 의존 작업

- 효성_004800.csv fix (별개 PDCA)
- 퀀트봇 자동매매 사전 테스트 PDCA (별개)

---

**현재 단계**: Plan 완료, **다음 → P0 정체 식별** (식별 후 Design 단계)
