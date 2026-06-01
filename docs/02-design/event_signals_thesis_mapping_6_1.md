# event_signals → 단타봇 명분게이트 매핑 (단타봇 주도, v1 — 6/1 정보봇 C-0/C-1 반영)

> 사장님 6/1 "추천대로 전체 진행" 승인. Phase C 분업 중 **매핑 합의** 단계 — 단타봇 주도.
> 목적: 정보봇이 생산할 `event_signals`(news/dart/edgar/policy/macro)가 단타봇 명분게이트의
> **어느 버킷에, 얼마(0~1 정규화)** 기여하는지 정의. 이게 있어야 C-3(소비배선) shadow 숫자가 의미.
> ★ 이 문서 = 단타봇 쪽 구조(확정) + 정보봇 쪽 필드(확인필요 마커). 양봇 합의 후 C-3 착수.

---

## 0. 전제 (단타봇 코드 검증 결과 — 확정)
명분게이트 `_aggregate_thesis_buckets(sig)` (data/morning_recommendation.py:93)는 자체 44신호를
4버킷으로 정규화하고, `nrm(v, cap)=min(1, max(0,v)/cap)`로 0~1 변환 후 버킷별 **max**를 취함:

| 버킷 | 현재 입력 신호(캡) | event_signals 수용? |
|---|---|---|
| **재료** | tv_direct(30)·relay_sc(45)·premove_sc(30)·surge_sc(10)·us_relay_sc(10)·rotation_bonus(12) | ★ **주 타깃** (명분=왜 오르나) |
| **수급** | nat_sc(50)·nat_power_sc(30)·doublebuy_sc(18)·largecap_sc(34)·fi_sc·bomb_sc·dual_buy_sc·invflow_sc·stflow_sc·etf_flow_sc | 일부(기관/외인 공시류) |
| 기술위치 | tech_sc(25)·trix_sc(25)·fib_adj(20) | ✗ (event 무관) |
| 재무 | pension_sc(15) | △ (실적공시 한정, v1 제외) |

통과판정: {재료·수급·기술위치} 중 정규화 ≥ `SAJANG.THESIS_GATE_STRONG_NORM` 인 게
`SAJANG.THESIS_GATE_MIN_STRONG`개 이상 → pass. (재무는 modifier, 자격박탈 아님)

→ **event_signals는 max() 항으로 1개 추가**되는 구조. 기존 신호를 안 건드리고 얹기만 함(fail-open).

---

## 1. source → 버킷 라우팅 (단타봇 제안)

| event source | → 버킷 | 근거 |
|---|---|---|
| `news` | 재료 | 재료/테마 명분의 핵심 |
| `dart`(국내공시) | 재료 (+수급 일부) | 계약·수주=재료 / 대량보유·자사주=수급 → **subtype 필요(정보봇 확인)** |
| `edgar`(미국공시) | 재료 | us_relay 성격 |
| `policy`(정책) | 재료 | 정책 테마 명분 |
| `macro`(매크로) | 재료 (약) | 시장 전체 → 종목 직접도 낮음, 가중 ↓ |

★ **정보봇 확인 필요**: dart subtype 구분 필드(계약/수주 vs 지분/수급) 존재 여부.
없으면 v1은 dart 전부 재료로 단순 라우팅, 수급 분기는 v2로 보류.

---

## 2. impact_score → 정규화 강도 (★ v1 교정 — 정보봇 구현 반영)
**[v0 폐기]** v0의 severity 4티어 테이블(critical 0.90…)은 폐기. 정보봇은 severity 등급 대신
**연속 `impact_score`(0~1)**를 공급(계약 §6.2: 임계는 소비자 자율 → 정보봇은 raw 강도, 단타봇이 임계).

→ **event_재료_raw = impact_score (그대로 0~1, 추가 변환 불필요)**.
- 단타봇 임계 = 기존 `SAJANG.THESIS_GATE_STRONG_NORM` 재사용(별도 EVENT 임계 안 만듦 — 단일진실).
- 즉 impact_score ≥ THESIS_GATE_STRONG_NORM 인 event 1건이면 재료 버킷 strong 도달 가능.
- 가상 캡(EVENT_재료_CAP) 불필요 — impact_score가 이미 정규화값.

---

## 3. 신선도 (★ v1 교정 — freshness_min 소비, 이중감쇠 금지)
정보봇이 `freshness_min`(event 경과 분) + `ts`(KST) 표준 제공.
**★ 미확정(정보봇 확인 1건)**: `impact_score`가 freshness를 **이미 반영**했는가?

- **case A (impact_score에 freshness 포함)**: 단타봇은 추가 감쇠 **안 함**(이중감쇠 방지).
  freshness_min은 **staleness 게이트로만** 사용 → `freshness_min > EVENT_STALE_MIN`이면 event 제외.
- **case B (impact_score=raw, freshness 별도)**: 단타봇이 감쇠 적용
  `event_재료 = impact_score × max(0, 1 - freshness_min/EVENT_DECAY_MIN)`.
- **기본 채택 = case A** (정보봇이 freshness 흡수했다 명시 → 단타봇은 게이트만). 확인 후 확정.
- `EVENT_STALE_MIN`(예: 당일물만 = 분 환산) SAJANG 상수. 단타 호흡상 짧게.

---

## 4. 종목별 다건 집계
한 종목에 event 여러 건 → 버킷 1값으로:
- **재료 = max(기존 6신호 정규화, max_over_events(event_재료))** — max 일관(기존 버킷 규칙과 동일).
- 합산(sum) 안 함: 명분 "중복 카운트"로 게이트 뚫는 것 방지(과적합·조작 내성).
- 종목 매칭: event_signals `ticker`(6자리 KR) ↔ 후보 종목코드.
  - **US 심볼**: 직접 종목조인 X → us_relay 성격으로 별도(v2) 또는 매핑 테이블 필요(보류).
  - **매크로 null ticker**: 특정 종목 조인 불가 → 시장레벨 신호로 분리(명분게이트 미투입, v1 제외).

---

## 5. C-3 통합 지점 (shadow-first)
`_aggregate_thesis_buckets` 내 재료 계산부에 항 1개 추가:
```python
재료 = max(
    nrm(sig.get("tv_direct"), 30), ..., nrm(sig.get("rotation_bonus"), 12),
    sig.get("event_재료_norm", 0.0),   # ← C-3 신규: event_signals 기여 (이미 0~1)
)
```
  여기서 `event_재료_norm` = 종목별 max(impact_score) (§2~3, case A면 staleness 게이트 통과분만).
- C-3은 **shadow**: `event_재료_norm`을 계산·기록·로그만, **picks 변경 0**(cutoff 미적용 — 5/31 브릭4와 동일 규율).
- recommendation.json에 `event_thesis_contrib` 필드로 관측 저장 → "event 켰으면 명분통과 몇 건 늘었나" 측정.
- 게이트 8/8 회귀 0 + 매도 무손상 확인 후에만 flip 논의(사장님 결정).

---

## 6. 안전·계약 준수
- **No action_hint 신뢰**: 정보봇 action_hint/tradable=null 유지 → 단타봇이 impact_score만 보고 자체 임계(소비자 자율 합의 준수).
- **raw_hash 중복방지**: 동일 event 중복 집계 차단(정보봇 canonical_hash 생성, 단타봇 dedup 키로 사용).
- **금지어 검증**: 정보봇 builder가 금지어 필터 → 단타봇은 통과분만 수신(이중 안전).
- **fail-open**: event_signals 없거나 파싱 실패 시 기존 44신호로 정상 가동(빈손 아님).
- **SAJANG 단일진실**: EVENT_* 상수 전부 `data/sajang_rules.py` 경유.

---

## 7. 정보봇 확인 — C-0/C-1 구현으로 4건 해소 (6/1)
1. ✅ `source` enum: news/dart/edgar/policy/macro (확정)
2. ⚠️ `dart` subtype(계약/수주 vs 지분/수급) — **여전히 미답**. v1은 dart 전부 재료, 수급 분기 v2 보류.
3. ✅ severity → **연속 `impact_score`(0~1) 공급**으로 대체(§2 v1 교정 반영). 4티어 테이블 폐기.
4. ✅ 타임스탬프 = `ts`(KST) + `freshness_min` 제공.
5. ✅ `ticker` = 6자리 KR강제 / US보존 / 매크로 null (§4 분기 반영).
6. ★ **신규 미답 1건**: `impact_score`가 freshness 이미 반영했는가? (case A/B, §3) — 이중감쇠 방지용.

## 8. 단타봇 다음 작업 (이 매핑 확정 후)
- SAJANG에 `EVENT_STALE_MIN`(+case B면 `EVENT_DECAY_MIN`) 신설 (default 활성, 5/26 룰). EVENT 임계는 기존 THESIS_GATE 재사용.
- C-3 shadow 배선: event_signals jsonl 로더 + 재료 max 항(impact_score) 추가 + recommendation.json 관측 필드(`event_thesis_contrib`)
- 테스트: event 로더 단위 + 명분게이트 회귀(event 유/무 동일픽 — shadow 불변 입증) + 게이트 8/8
- jsonl 며칠 관측 → "event 기여로 명분통과 분포 변화" 보고 → 사장님 flip 결정
