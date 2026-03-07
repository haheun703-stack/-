# BRAIN → v3 Brain 연동 설계 명세

**작성일**: 2026-03-07
**상태**: 설계 문서 (코드 미수정)
**목적**: BRAIN(매크로 배분)의 지시가 v3 Brain(종목 추천)에 반영되도록 연동

---

## 1. 현재 상태: 두 시스템의 독립 실행

### BRAIN (brain.py) — 매크로 배분

```
입력: overnight_signal + kospi_regime + macro + shield + COT + liquidity
처리: 5대 눈 → 레짐 판정 → ARM 배분
출력: brain_decision.json
  → effective_regime: BEAR
  → arms.swing.adjusted_pct: 12.5%  ← "개별 스윙에 12.5% 배분"
  → confidence: 0.45
```

### v3 Brain (run_v3_brain.py) — 종목 추천

```
입력: scan_cache + 뉴스 + 섹터모멘텀 + positions
처리: Phase 0(o1) → 1(Opus) → 2(Sonnet) → 3(필터) → 4(Deep) → 5(Opus) → 6(Perplexity)
출력: ai_v3_picks.json
  → buys: [{ticker, conviction, size_pct, strategy}]
  → max_new_buys: Claude Opus가 AI로 결정 (0~4)
```

### 현재 접점: 없음

- BRAIN이 "swing 12.5%"를 지시해도 v3 Brain은 이걸 안 읽음
- v3 Brain의 max_new_buys는 Claude Opus (Phase 1)가 자체 판단
- 두 시스템이 같은 시장을 보지만 **서로 다른 결론**에 도달 가능

### 현재 문제 시나리오

```
BRAIN: "CRISIS 레짐 → swing 0%, 현금 50%"  (주식 사지 마라)
v3 Brain: Phase 1 Opus가 "공격 레짐 → max_new_buys=4"  (4종목 사라)
→ 모순: 매크로는 방어인데 종목 추천은 공격
```

---

## 2. 연동 설계

### 핵심 원칙

1. **BRAIN이 상위 계층** — 매크로 배분 → v3가 그 범위 안에서 종목 선정
2. **단방향 참조** — v3 Brain이 brain_decision.json을 읽기만 (BRAIN은 v3를 모름)
3. **기존 로직 최소 변경** — v3의 Phase 1~5 로직은 그대로, 입력 제약만 추가

### 연동 지점: run_v3_brain.py 상단

```python
# ── BRAIN 지시서 로드 ──
brain_decision = _load_json(DATA_DIR / "brain_decision.json")
brain_regime = brain_decision.get("effective_regime", "NEUTRAL")
brain_swing_pct = 0
for arm in brain_decision.get("arms", []):
    if arm["name"] == "swing":
        brain_swing_pct = arm.get("adjusted_pct", 30)
brain_confidence = brain_decision.get("confidence", 0.5)
```

### 2-A. 슬롯 캡 연동

v3의 max_new_buys를 BRAIN 레짐으로 제한:

| BRAIN 레짐 | swing_pct 범위 | max_new_buys 상한 | 근거 |
|-----------|---------------|-------------------|------|
| BULL | 25%+ | 4 (기존 그대로) | 공격 허용 |
| CAUTION | 15~25% | 3 | 약간 보수적 |
| BEAR | 5~15% | 2 | 방어 모드 |
| CRISIS | 0~5% | 0 | 매수 금지 |

**적용 위치**: Phase 5 (PortfolioBrain) 진입 전

```python
# Phase 5 진입 전 — BRAIN 슬롯 캡 적용
regime_slot_cap = {
    "BULL": 4, "CAUTION": 3, "BEAR": 2, "CRISIS": 0
}.get(brain_regime, 3)

# Phase 1 Opus가 결정한 max_new_buys를 BRAIN 캡으로 클램핑
strategic_max = strategic_result.get("max_new_buys", 3)
effective_max = min(strategic_max, regime_slot_cap)
strategic_result["max_new_buys"] = effective_max
strategic_result["brain_regime_cap"] = regime_slot_cap
strategic_result["brain_regime"] = brain_regime
```

### 2-B. 종목당 배분 금액 연동

BRAIN의 swing_pct로 종목당 최대 배분 계산:

```python
# 총 자산 (예: 1억)
total_equity = 100_000_000

# BRAIN이 지시한 스윙 예산
swing_budget = total_equity * (brain_swing_pct / 100)
# 예: BEAR → 12.5% → 12,500,000원

# 종목당 최대 배분 = 스윙 예산 / max_new_buys
if effective_max > 0:
    max_per_stock = swing_budget / effective_max
else:
    max_per_stock = 0
# 예: 12,500,000 / 2 = 6,250,000원/종목
```

**적용 위치**: Phase 5 PortfolioBrain의 size_pct 계산 시

```python
# Phase 5에서 size_pct 상한 적용
for buy in buys:
    max_size_pct = (max_per_stock / total_equity) * 100
    buy["size_pct"] = min(buy["size_pct"], max_size_pct)
```

### 2-C. 확신도 필터 연동

BRAIN confidence가 낮으면 conviction threshold를 높임:

| BRAIN confidence | conviction 최소값 | 효과 |
|-----------------|------------------|------|
| ≥ 0.70 | 4 (기존) | 공격 |
| 0.50~0.70 | 5 | 약간 엄격 |
| 0.30~0.50 | 6 | 엄격 |
| < 0.30 | 8 | 매우 엄격 (거의 매수 안 함) |

**적용 위치**: Phase 4 (DeepAnalyst) conviction 필터 단계

```python
# Phase 4에서 conviction threshold 상향
base_min_conviction = settings.get("ai_brain_v3", {}).get("min_conviction", 4)
if brain_confidence < 0.30:
    effective_min_conviction = max(base_min_conviction, 8)
elif brain_confidence < 0.50:
    effective_min_conviction = max(base_min_conviction, 6)
elif brain_confidence < 0.70:
    effective_min_conviction = max(base_min_conviction, 5)
else:
    effective_min_conviction = base_min_conviction
```

---

## 3. 데이터 흐름 (연동 후)

```
BRAIN (brain.py)
  │
  ├─ brain_decision.json
  │   ├─ effective_regime: BEAR
  │   ├─ arms.swing.adjusted_pct: 12.5%
  │   └─ confidence: 0.45
  │
  ▼
v3 Brain (run_v3_brain.py)
  │
  ├─ [읽기] brain_decision.json
  │   → regime_slot_cap = 2 (BEAR)
  │   → swing_budget = 12,500,000원
  │   → effective_min_conviction = 6
  │
  ├─ Phase 0: o1 Deep Thinking (변경 없음)
  ├─ Phase 1: Strategic Brain (max_new_buys 자체 결정)
  │   → max_new_buys = 3 (Opus 판단)
  │   → 클램핑: min(3, 2) = 2 ← BRAIN 캡 적용
  │
  ├─ Phase 2: Sector Strategist (변경 없음)
  ├─ Phase 3: 후보 필터링 (변경 없음)
  ├─ Phase 4: Deep Analyst
  │   → conviction ≥ 6 (기존 4 → BRAIN confidence 0.45로 상향)
  │   → 통과 종목 수 감소
  │
  ├─ Phase 5: Portfolio Brain
  │   → max_new_buys = 2 (클램핑됨)
  │   → size_pct ≤ 6.25% (종목당 상한)
  │
  └─ Phase 6: Perplexity 검증 (변경 없음)
```

---

## 4. Graceful Degradation

brain_decision.json이 없거나 오래된 경우:

```python
brain_decision = _load_json(DATA_DIR / "brain_decision.json")
if not brain_decision or not brain_decision.get("arms"):
    logger.warning("BRAIN 지시서 없음 — 기존 로직으로 독립 실행")
    brain_regime = "NEUTRAL"
    brain_swing_pct = 30  # 기본값
    brain_confidence = 0.5
else:
    # brain_decision 타임스탬프 체크
    from datetime import datetime, timedelta
    ts = brain_decision.get("timestamp", "")
    try:
        decision_time = datetime.fromisoformat(ts)
        if datetime.now() - decision_time > timedelta(hours=24):
            logger.warning("BRAIN 지시서 24시간 경과 — 보수적 기본값 적용")
            brain_regime = "CAUTION"
            brain_swing_pct = 20
            brain_confidence = 0.5
    except:
        pass
```

---

## 5. settings.yaml 추가 설정 (안)

```yaml
brain_v3_integration:
  enabled: true                    # 연동 활성화
  regime_slot_cap:                 # 레짐별 max_new_buys 상한
    BULL: 4
    CAUTION: 3
    BEAR: 2
    CRISIS: 0
  confidence_conviction_map:       # confidence → min_conviction
    - {below: 0.30, min_conviction: 8}
    - {below: 0.50, min_conviction: 6}
    - {below: 0.70, min_conviction: 5}
    - {below: 1.01, min_conviction: 4}
  stale_hours: 24                  # 지시서 유효 시간
  stale_default_regime: "CAUTION"  # 만료 시 기본 레짐
```

---

## 6. 수정 파일 목록 (예상)

| 파일 | 변경 | 범위 |
|------|------|------|
| `scripts/run_v3_brain.py` | BRAIN 지시서 로드 + 슬롯캡 + conviction 조정 | ~30줄 추가 |
| `config/settings.yaml` | `brain_v3_integration` 섹션 추가 | ~15줄 추가 |

**brain.py 변경 없음** — BRAIN은 기존대로 brain_decision.json만 출력.

---

## 7. 리스크 + 안전장치

### 리스크 1: 이중 보수성

BRAIN이 BEAR로 슬롯 2개로 제한 + v3 Opus도 방어 판단 → max_new_buys = 0 가능.

**안전장치**: `regime_slot_cap`은 상한만 걸고, 하한은 안 건다. v3가 "공격"이라고 해도 BRAIN 캡(2)까지만 허용. v3가 "방어"(0)이면 그대로 0.

### 리스크 2: BRAIN-v3 레짐 모순

BRAIN이 BULL인데 v3 Opus가 "회피"로 판단하는 경우.

**안전장치**: 하위 레이어(v3)가 더 보수적인 것은 허용. 하위가 더 공격적인 것만 캡으로 차단. 즉, **BRAIN은 천장만 걸고 바닥은 안 건다**.

### 리스크 3: 타이밍 불일치

BRAIN은 06:10(BAT-A)에, v3 Brain은 17:00(BAT-D)에 실행. 10시간 시차.

**안전장치**: brain_decision.json의 timestamp 체크. 24시간 이내면 유효. 장중 급변(COMPOUND 충격 등)은 BRAIN이 재실행되므로 갱신됨.

---

## 8. 현재 실제 수치로 시뮬레이션

```
현재 BRAIN 출력 (2026-03-07):
  effective_regime: BEAR
  swing_pct: 12.5%
  confidence: 0.45

연동 적용 시:
  regime_slot_cap: 2 (BEAR)
  effective_min_conviction: 6 (confidence 0.45 → 0.30~0.50 구간)
  swing_budget: 12,500,000원 (1억 기준)
  max_per_stock: 6,250,000원

v3 Brain Phase 1이 max_new_buys=3 결정 시:
  → 클램핑: min(3, 2) = 2종목
  → conviction ≥ 6만 통과 (기존 4에서 상향)
  → 종목당 최대 6.25% (기존 제한 없음에서 추가)

효과:
  BRAIN이 "방어"라고 했는데 v3가 4종목 추천하는 모순 해소
  conviction이 낮은 "미지근한" 종목은 필터링
  종목당 배분도 BRAIN 예산 범위 내로 제한
```

---

## 결론

이 연동의 핵심은 **"BRAIN이 천장을 걸고, v3가 그 안에서 자유롭게 판단"**하는 구조. BRAIN은 "전체 자산의 몇 %를 주식에 넣을지"를 결정하고, v3는 "그 예산 안에서 어떤 종목을 살지"를 결정.

변경 범위가 작고(run_v3_brain.py ~30줄), brain.py는 건드리지 않으며, enabled 플래그로 즉시 비활성화 가능.
