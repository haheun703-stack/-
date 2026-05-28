# cancel / modify 정책 (코덱스 4차 응답 5/28)

> **상태**: 정책 v1 (코덱스 검수 대기)
> **연결**: docs/01-plan/trading-factory-v1-architecture.md

## 1. 정책 결단

### 1-1. `cancel(order)` — intent 대상 X (운영 액션)
- **이유**: 기존 PENDING 주문 취소는 새 매매 발생 X. 기존 매매를 무효화하는 운영 액션.
- **가드 적용**:
  - L8 `assert_runtime_orders_allowed()` — KILL_SWITCH/PAPER_ONLY/AUTO_TRADE_DISABLED 차단 (5/27 추가)
  - L10 `order_intents_gate` — **호출 X** (정책)
- **정당성**:
  - cancel은 broker 측 기존 주문 ID 기반 — 새 ticker 신호 X
  - intent와 cancel은 별개 lifecycle (intent → buy → 미체결 시 cancel)
  - cancel을 intent로 강제하면 cancel intent 또 등록해야 하는 무한 루프

### 1-2. `modify(order, new_price, new_quantity)` — 향후 intent 대상 검토
- **현재**: L8만 통과 (5/27 추가). L10 미적용.
- **권장**: 가격/수량 변경은 사실상 매매 변경 → 향후 modify_intent 별도 등록 필수화
- **단계적 적용 일정**:
  - 5/28: 정책 결단만 (현재 L10 미적용 명시)
  - 6/2~: paper 리허설 중 modify 사용 빈도 모니터
  - 6/9+: live 가동 전 modify_intent 정식 도입 결단

### 1-3. order_intents_gate.register_intent — modify 지원 (옵션, v2)
- 향후 `intent_type: "modify"` 필드 추가 가능
- 기존 intent_id 참조 + 변경 사항 (new_price/new_quantity) 기록
- 현재 v1에서는 미지원

## 2. 코드 위치

| 파일 | 함수 | L8 | L10 | 정책 |
|------|------|----|----|------|
| `src/adapters/kis_order_adapter.py:321` | `cancel(order)` | ✅ | ❌ | 운영 액션 (intent 예외) |
| `src/adapters/kis_order_adapter.py:341` | `modify(order, ...)` | ✅ | ❌ | 향후 적용 (modify_intent v2) |

## 3. 회귀 테스트 권장 (TODO)

- `test_cancel_does_not_require_intent`: cancel 호출 시 order_intents 미등록 → 성공
- `test_modify_does_not_require_intent_yet`: modify 호출 시 미등록 → 성공 (현 정책)
- `test_modify_intent_required_after_v2`: v2 도입 후 modify intent 필수

## 4. 코덱스 검수 요청

1. cancel을 intent 예외로 두는 결단의 안전성
2. modify의 점진 적용 일정 (6/9+ live 전 의무)
3. modify intent 스키마 권장 (intent_type 필드 + parent_intent_id)
