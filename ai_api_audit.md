# AI API 통합 현황 감사 보고서

**작성일**: 2026-03-07
**대상**: Quantum Master 전체 코드베이스
**BRAIN 버전**: v13.9 (5대 눈 완성)

---

## 1단계: AI API 사용처 전수조사

### Claude API (Anthropic)

| 파일 | 클래스/함수 | 모델 | 용도 | BRAIN 연동 |
|------|-----------|------|------|-----------|
| `src/agents/base.py` | `BaseAgent` | claude-sonnet-4-5 (기본) | 전체 Claude 에이전트 공통 베이스 | 간접 (하위 에이전트 통해) |
| `src/agents/strategic_brain.py` | `StrategicBrainAgent` | **claude-opus-4** | v3 Phase 1: 거시 전략 판단 | Y — ai_strategic_analysis.json |
| `src/agents/sector_strategist.py` | `SectorStrategistAgent` | claude-sonnet-4-5 | v3 Phase 2: 섹터 전략 | Y — ai_sector_focus.json |
| `src/agents/deep_analyst.py` | `DeepAnalystAgent` | claude-sonnet-4-5 + **Vision** | v3 Phase 4: 종목 정밀 분석 (차트 이미지 포함) | Y — conviction 필터 |
| `src/agents/portfolio_brain.py` | `PortfolioBrainAgent` | **claude-opus-4** | v3 Phase 5: 최종 포트폴리오 결정 | Y — ai_v3_picks.json |
| `src/agents/sell_brain.py` | `SellBrainAgent` | claude-sonnet-4-5 | 매도 판단 (기술적 + 맥락) | Y — sell_monitor.py 통해 |
| `src/agents/news_brain.py` | `NewsBrainAgent` | claude-sonnet-4-5 | 뉴스 정성적 판단 | Y — ai_brain_judgment.json |
| `src/agents/etf_brain.py` | `ETFBrainAgent` | claude-sonnet-4-5 | ETF 매수 필터 (KILL/HOLD/PASS) | Y — ETF 로테이션 |
| `src/adapters/dart_classifier.py` | `classify_disclosures()` | claude-sonnet-4-5 | DART 공시 촉매/소음 분류 | Y — 뉴스 소스로 공급 |
| `src/adapters/claude_scoring.py` | `ClaudeScoringAdapter` | claude-sonnet-4-5 | 100점 종합 스코어링 | **N — archive 전용** |
| `src/agents/macro_analyst.py` | `MacroAnalystAgent` | claude-sonnet-4-5 | 레짐/섹터/시장폭 분석 | **N — 독립 실행** |
| `src/agents/cfo.py` | `CFOAgent` | claude-sonnet-4-5 | 배분/건전성/낙폭 | **N — 독립 실행** |
| `src/agents/risk_sentinel.py` | `RiskSentinelAgent` | claude-sonnet-4-5 | 리스크 감시/경보 | **N — 독립 실행** |
| `src/agents/chart_analysis.py` | `ChartAnalysisAgent` | claude-sonnet-4-5 | 차트 패턴 분석 | **N — 독립 실행** |
| `src/agents/flow_prediction.py` | `FlowPredictionAgent` | claude-sonnet-4-5 | 수급 예측 | **N — 독립 실행** |
| `src/agents/condition_judge.py` | `ConditionJudgeAgent` | claude-sonnet-4-5 | 유지/대응 조건 판단 | **N — 독립 실행** |
| `src/agents/game_analyst.py` | `GameAnalystAgent` | claude-sonnet-4-5 | 6D 게임 분석 | **N — portfolio_reporter 전용** |
| `src/agents/volume_analysis.py` | `VolumeAnalysisAgent` | claude-sonnet-4-5 | 거래량 분석 | **N — 독립 실행** |
| `scripts/ai_news_brain.py` | (스크립트) | claude-sonnet-4-5 | 5개 뉴스소스 → AI 판단 | Y — BAT-D 실행 |
| `src/etf/ai_filter.py` | `ETFAIFilter` | claude-sonnet-4-5 | ETF AI 필터 래퍼 | Y — ETF 로테이션 |

### ChatGPT/OpenAI API

| 파일 | 클래스/함수 | 모델 | 용도 | BRAIN 연동 |
|------|-----------|------|------|-----------|
| `src/agents/gpt_catalyst.py` | `GPTCatalystAgent` | **gpt-4o** | 뉴스 촉매 분석 (매도용) | Y — sell_monitor 듀얼AI |
| `src/agents/o1_strategist.py` | `O1StrategistAgent` | **o1** | Deep Thinking 거시/미시 | Y — v3 Phase 0 → Phase 1 입력 |

### Perplexity API

| 파일 | 클래스/함수 | 모델 | 용도 | BRAIN 연동 |
|------|-----------|------|------|-----------|
| `scripts/perplexity_market_intel.py` | `query_perplexity()` | **sonar** | 미국장→한국 섹터 파급 분석 | Y — market_intelligence.json |
| `src/agents/perplexity_verifier.py` | `PerplexityVerifier` | **sonar** | v3 Phase 6 팩트체크 | Y — perplexity_verification.json |
| `scripts/crawl_morning_reports.py` | `query_perplexity_morning_theme()` | sonar | 장전 테마 예측 | Y — morning_reports.json |

### FRED API (비-AI, 참고)

| 파일 | 용도 |
|------|------|
| `scripts/fetch_liquidity_data.py` | FRED 5대 지표 수집 (유동성 사이클) |

---

## 2단계: 역할 분담 현황

### Claude API — "판단자 + 분석자" (주력)

```
┌─────────────────────────────────────────────────────┐
│ Claude Opus (전략)                                    │
│  ├─ StrategicBrain: Phase 1 거시 전략 판단              │
│  └─ PortfolioBrain: Phase 5 최종 포트폴리오 결정         │
│                                                       │
│ Claude Sonnet (전술)                                   │
│  ├─ SectorStrategist: Phase 2 섹터 전략                │
│  ├─ DeepAnalyst + Vision: Phase 4 종목 정밀 분석        │
│  ├─ SellBrain: 매도 기술적 판단                         │
│  ├─ NewsBrain: 뉴스 정성적 판단                         │
│  ├─ ETFBrain: ETF 매수 필터                            │
│  └─ DartClassifier: 공시 분류                          │
│                                                       │
│ Claude Sonnet (독립 — 미연동)                           │
│  ├─ MacroAnalyst: 레짐/섹터/시장폭                     │
│  ├─ CFOAgent: 배분/건전성/낙폭                         │
│  ├─ RiskSentinel: 리스크 감시                          │
│  ├─ ChartAnalysis: 차트 패턴                           │
│  ├─ FlowPrediction: 수급 예측                          │
│  ├─ ConditionJudge: 유지/대응 조건                     │
│  ├─ GameAnalyst: 6D 게임 분석                          │
│  ├─ VolumeAnalysis: 거래량 분석                        │
│  └─ ClaudeScoring: 100점 스코어링 (archive)            │
└─────────────────────────────────────────────────────┘
```

**핵심**: Claude가 전략(Opus) + 전술(Sonnet) + 매도(Sonnet) + ETF(Sonnet) 4개 영역의 "최종 판단"을 독점. 8개 에이전트는 코드만 있고 메인 파이프라인에 미연동.

### ChatGPT/OpenAI — "촉매 분석자 + 딥씽커"

```
┌─────────────────────────────────────────────────────┐
│ GPT-4o                                                │
│  └─ GPTCatalyst: 보유 종목 뉴스 촉매 생존 판단           │
│     → sell_monitor.py의 듀얼AI 합의에 사용               │
│     → Claude SellBrain과 GPT Catalyst가 합의 규칙 적용   │
│                                                       │
│ GPT o1                                                │
│  └─ O1Strategist: Deep Thinking 거시/미시 분석           │
│     → v3 Brain Phase 0에서 실행                         │
│     → 결과가 Phase 1 StrategicBrain의 컨텍스트로 주입     │
└─────────────────────────────────────────────────────┘
```

**핵심**: GPT는 2가지 역할만 — (1) 매도 시 촉매 살아있는지 판단 (2) 매수 전 거시 Deep Thinking.

### Perplexity — "실시간 검색 + 팩트체크"

```
┌─────────────────────────────────────────────────────┐
│ Perplexity sonar                                      │
│  ├─ market_intel: 미국장→한국 파급 실시간 분석            │
│  │  → market_intelligence.json → 다수 모듈이 참조        │
│  │                                                    │
│  ├─ verifier: v3 Phase 6 팩트체크                       │
│  │  → 종목 촉매/리스크/thesis 교차검증                    │
│  │  → HALLUCINATION_DETECTED → 해당 종목 스킵            │
│  │                                                    │
│  └─ morning_theme: 장전 테마 예측                        │
│     → morning_reports.json → scan_tomorrow_picks 보정    │
└─────────────────────────────────────────────────────┘
```

**핵심**: Perplexity는 "웹 검색"에 특화. Claude가 판단할 수 없는 실시간 사실관계 확인.

---

## 3단계: BRAIN과의 연동 상태 점검

### BRAIN 5대 눈 × AI API

| BRAIN 레이어 | AI API 직접 사용 | 데이터 소스 | 비고 |
|-------------|----------------|-----------|------|
| NightWatch (1D) | **없음** | overnight_signal.json (룰 기반) | VIX/금리/달러 — 순수 수치 계산 |
| 2D 구조적 선행 | **없음** | regime_macro_signal.json (룰 기반) | 레짐 전환 선행 — z-score 기반 |
| S4 상관 붕괴 | **없음** | 룰 기반 | 자산간 상관 변화 감지 |
| 유동성 5D | **없음** | liquidity_signal.json (FRED API) | Net Liquidity z-score |
| COT 4D | **없음** | cot_signal.json (CFTC 데이터) | 기관 포지셔닝 |
| 레짐 판정 | **없음** | kospi_regime.json (룰 기반) | KOSPI MA + 변동성 |
| 텔레그램 브리핑 | **없음** | 위 데이터 종합 텍스트 | 순수 포매팅 |

**결론: BRAIN 5대 눈은 AI API를 전혀 사용하지 않는다.** 전부 룰 기반(z-score, 임계값, 가중합산).

### v3 Brain 파이프라인 × AI API

| Phase | AI API | 에이전트 | 출력 |
|-------|--------|---------|------|
| Phase 0 | **GPT o1** | O1StrategistAgent | o1_deep_analysis.json |
| Phase 1 | **Claude Opus** | StrategicBrainAgent | ai_strategic_analysis.json |
| Phase 2 | **Claude Sonnet** | SectorStrategistAgent | ai_sector_focus.json |
| Phase 3 | 없음 (룰) | scan_cache 필터링 | — |
| Phase 4 | **Claude Sonnet + Vision** | DeepAnalystAgent | conviction 필터 |
| Phase 5 | **Claude Opus** | PortfolioBrainAgent | ai_v3_picks.json |
| Phase 6 | **Perplexity sonar** | PerplexityVerifier | perplexity_verification.json |

### 매도 파이프라인 × AI API

| Step | AI API | 에이전트 | 출력 |
|------|--------|---------|------|
| Step 1 | **GPT-4o** | GPTCatalystAgent | gpt_catalyst_analysis.json |
| Step 2 | **Claude Sonnet** | SellBrainAgent | ai_sell_cache.json |
| Step 3 | 없음 (룰) | 합의 규칙 적용 | ai_sell_consensus.json |

---

## 4단계: 호환성 진단

### Q1. 3개 AI API 중 실제로 코드에서 호출되고 있는 것은 몇 개인가?

**3개 전부 실제 호출됨.**

| API | 호출 경로 | 활성 상태 |
|-----|----------|----------|
| Claude (Anthropic) | `anthropic.AsyncAnthropic()` via BaseAgent | ✅ 활성 |
| OpenAI (GPT-4o/o1) | `openai.AsyncOpenAI()` via GPTCatalyst/O1Strategist | ✅ 활성 |
| Perplexity | `requests.post(PERPLEXITY_URL)` | ✅ 활성 |

### Q2. API 키가 설정만 되어 있고 호출되지 않는 "죽은 코드"가 있는가?

**있다. 8개 Claude 에이전트가 코드만 존재하고 메인 파이프라인에 미연동:**

| 에이전트 | 상태 | 설명 |
|---------|------|------|
| `MacroAnalystAgent` | 🔇 죽은 코드 | v3 Brain 이전 레거시. 독립 실행 가능하나 파이프라인 미연동 |
| `CFOAgent` | 🔇 죽은 코드 | Kelly Criterion 배분. BRAIN이 룰 기반으로 대체 |
| `RiskSentinelAgent` | 🔇 죽은 코드 | 리스크 감시. NightWatch+SHIELD가 대체 |
| `ChartAnalysisAgent` | 🔇 죽은 코드 | 차트 패턴. DeepAnalyst+Vision이 대체 |
| `FlowPredictionAgent` | 🔇 죽은 코드 | 수급 예측. 독립 실행 전용 |
| `ConditionJudgeAgent` | 🔇 죽은 코드 | 유지/대응 조건. 독립 실행 전용 |
| `GameAnalystAgent` | 🟡 반죽은 코드 | portfolio_reporter에서만 호출 (BAT 미연동) |
| `VolumeAnalysisAgent` | 🔇 죽은 코드 | 거래량 분석. 독립 실행 전용 |
| `ClaudeScoringAdapter` | 🔇 archive | scripts/archive에서만 참조 |

### Q3. AI API가 BRAIN 파이프라인과 연결되어 있는가?

**두 개의 분리된 세계가 존재한다:**

```
세계 1: BRAIN 5대 눈 (brain.py)
  → AI API 제로. 순수 룰 기반.
  → JSON 파일(overnight_signal, cot_signal, liquidity_signal 등) 읽기만.
  → 출력: 자산 배분 ARM + 레짐 + 브리핑

세계 2: v3 Brain 파이프라인 (run_v3_brain.py)
  → AI API 집중 사용 (Claude Opus/Sonnet + GPT o1 + Perplexity)
  → Phase 0~6 → 종목 추천 (ai_v3_picks.json)
  → 출력: 매수 추천 종목 리스트
```

**두 세계의 접점**:
- BRAIN(세계1)은 v3(세계2)의 결과를 **직접 참조하지 않음**
- v3 Brain은 BRAIN의 레짐 정보를 간접적으로 사용 (regime_macro_signal.json)
- 실질적으로 **독립 실행**

### Q4. 동일한 작업을 여러 AI가 중복으로 하고 있는 경우가 있는가?

**부분적 중복 2건 발견:**

| 중복 영역 | AI 1 | AI 2 | 차이점 |
|----------|------|------|--------|
| 거시 분석 | GPT o1 (Phase 0) | Claude Opus (Phase 1) | o1은 Deep Thinking, Opus는 전략 판단. o1 결과가 Opus의 입력으로 들어감 → **직렬 연결, 중복은 아님** |
| 뉴스 촉매 판단 | GPT-4o (매도용) | Claude Sonnet (매수용 NewsBrain) | GPT는 매도 촉매 생존, Claude는 매수 뉴스 판단 → **용도 분리, 중복 아님** |

**결론: 진정한 중복은 없다.** 역할이 명확히 분리되어 있음.

### Q5. AI API 호출이 실패했을 때 fallback 처리가 있는가?

| API | Fallback | 구현 |
|-----|----------|------|
| Claude (BaseAgent) | ✅ | try/except → 에러 로깅, 빈 dict 반환 |
| GPT-4o (Catalyst) | ✅ | `fallback_on_failure: true` → Claude 단독 판단으로 전환 |
| GPT o1 (Phase 0) | ✅ | 실패 시 Phase 1 독립 동작 (o1 컨텍스트 없이 진행) |
| Perplexity (Verifier) | ✅ | 실패 시 빈 dict → Phase 6 스킵 |
| Perplexity (market_intel) | ✅ | 실패 시 빈 dict → 보정 없음 |

**전 API fallback 완비.** 어느 하나가 죽어도 시스템 전체가 멈추지 않음.

---

## 5단계: 최적화 제안

### 5-A. 현재 3 AI 역할 분담 평가

| 평가 항목 | 상태 | 점수 |
|----------|------|------|
| 역할 분담 명확성 | 우수 | ★★★★☆ |
| 중복 제거 | 우수 | ★★★★★ |
| Fallback 처리 | 완비 | ★★★★★ |
| BRAIN 연동 | **미흡** | ★★☆☆☆ |
| 죽은 코드 정리 | **미흡** | ★★☆☆☆ |

### 5-B. 현재 역할 분담 (실제 동작 기준)

```
┌──────────────────────────────────────────────────────────┐
│  Claude (판단 + 전략 + 매도)                                │
│  ├─ Opus: 전략 판단(Phase1) + 포트폴리오 결정(Phase5)        │
│  ├─ Sonnet: 섹터전략(P2) + 종목분석(P4) + 매도판단 + ETF필터  │
│  └─ Sonnet+Vision: 차트 이미지 분석 (P4)                    │
│                                                            │
│  GPT (촉매 + Deep Thinking)                                 │
│  ├─ o1: 거시/미시 Deep Thinking (Phase 0 → Phase 1 입력)     │
│  └─ 4o: 매도 시 뉴스 촉매 생존 판단                          │
│                                                            │
│  Perplexity (실시간 검색 + 팩트체크)                          │
│  ├─ 시장 인텔리전스 (미국장→한국 파급)                        │
│  ├─ v3 Phase 6 촉매/thesis 교차검증                         │
│  └─ 장전 테마 예측                                          │
└──────────────────────────────────────────────────────────┘
```

### 5-C. 발견된 문제 + 제안

#### 문제 1: 죽은 에이전트 8개 (Claude API 비용 리스크)

코드에 존재하지만 파이프라인에 연결되지 않은 에이전트 8개. 누군가 실수로 호출하면 API 비용 발생.

**제안**: `scripts/archive/legacy_agents/`로 이동하거나, 현재 상태 유지 (코드 자체는 비용 없음, import 시에만 인스턴스화).

#### 문제 2: BRAIN 5대 눈 ↔ v3 Brain 파이프라인 단절

BRAIN(brain.py)은 "자산 배분"을 결정하지만, v3 Brain은 "종목 추천"을 결정. 두 시스템이 독립적으로 돌아감.

```
현재:
  BRAIN → "오늘 주식 30%, 금 20%, 채권 5%" (레짐 기반)
  v3 Brain → "삼성전자, 현대차, SK하이닉스 매수" (AI 분석 기반)
  → 서로의 결과를 모름
```

**제안**: 이것은 **의도된 설계**로 보임. BRAIN은 매크로 배분, v3는 종목 선정. 역할이 다르므로 단절이 아니라 계층 분리. 다만, BRAIN의 레짐(BEAR/CRISIS)이 v3의 max_new_buys를 제한하는 간접 연동은 이미 존재 (v3 Phase 5에서 regime 참조).

#### 문제 3: Perplexity가 3곳에서 개별 호출

market_intel, verifier, morning_theme 3곳에서 각각 독립적으로 Perplexity API를 호출.

**제안**: 현재로도 문제없음 (용도가 다르고, 각각 다른 시간대에 실행). 통합 클라이언트를 만들 수도 있으나 과도한 엔지니어링.

---

## 최종 요약

### 3 AI 역할 분담 — 현재 상태 진단

| AI | 역할 | 호출 빈도 | 비용 비중 (추정) |
|----|------|----------|---------------|
| **Claude** | 전략 판단 + 종목 분석 + 매도 + ETF | 가장 높음 (Opus 2회 + Sonnet N회) | **~70%** |
| **GPT** | 촉매 분석 + Deep Thinking | 중간 (4o 1회 + o1 1회) | **~20%** |
| **Perplexity** | 실시간 검색 + 팩트체크 | 낮음 (sonar 3~8회) | **~10%** |

### 핵심 결론

1. **역할 분담은 잘 되어 있다** — Claude(판단) / GPT(촉매+딥씽킹) / Perplexity(검색+팩트체크). 진정한 중복 없음.
2. **BRAIN 5대 눈은 AI를 안 쓴다** — 의도된 설계. 룰 기반이라 비용 0, 실시간성 보장.
3. **죽은 에이전트 8개**가 src/agents/에 존재 — 즉시 해악은 없지만 코드 복잡도 증가.
4. **두 세계(BRAIN ↔ v3)는 계층 분리** — BRAIN=매크로 배분, v3=종목 추천. 간접 연동 존재.
5. **Fallback 전부 완비** — 어느 AI가 죽어도 시스템은 돌아감.
