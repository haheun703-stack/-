# 재가동 자료 4건 처리 계획 — 5/29(금)

> **상태**: 계획 단계 (사장님 결단 대기 — 코드/문서 작성 0건)
> **HEAD**: 35718cb (로컬, 변경 미커밋) / db5d731 (VPS)
> **계기**: 5/29 사장님 결단 "다음 보고에는 filelock / HMAC rotation / §9 dry-run / safety_guard P2 네 항목의 처리 계획"
> **고정 판정**: P1-A 호출자 마이그레이션은 PASS. 그러나 퀀트봇 운영 재가동은 filelock, HMAC rotation playbook, §9 dry-run 완료 전까지 HOLD.

---

## 0. 4건 우선순위 매트릭스 (권장)

| 순서 | 항목 | 등급 | 종류 | 예상 작업량 | 위험 |
|---|---|---|---|---|---|
| 1 | **safety_guard.emergency_liquidate** P2 마이그레이션 | P2 | 코드 (~10라인) | 30분 + 회귀 | 낮음 (P1-A 패턴 동일) |
| 2 | **filelock 정책 문서화** | P1-4 Med | 문서 + 정책 결단 | 1시간 | 낮음 |
| 3 | **HMAC rotation playbook 작성** | P1-4 Med | 문서 + 절차 + 운영 룰 | 1.5시간 | 낮음 |
| 4 | **§9 4단계 dry-run 정의 + 준비** | P1-2 High | 정의 + 스크립트 + 실행 | 2~3시간 | **중간** (실측 단계) |

**권장**: 1 → 2 → 3 → 4 순서. **이유**:
- 1 (P2)은 P1-A 패턴 그대로 적용 → 회귀 보장 안전
- 2 (filelock)은 정책 결단만 하면 즉시 마무리 가능 (Linux append atomicity 근거)
- 3 (HMAC rotation)은 .env + 키 회전 스크립트까지 묶어 한 번에
- 4 (§9 dry-run)은 1~3 완료 후 실측 단계로 가장 큰 작업

---

## 1. safety_guard.emergency_liquidate P2 (★ 신규 등록)

### 1-1. 현재 상태
- 위치: `src/use_cases/safety_guard.py:128`
- 코드: `order = order_port.sell_market(pos.ticker, pos.shares)` — mode/executor_bot 미전달
- 호출 경로: `LiveTradingEngine.guard.emergency_liquidate(tracker, order_port)` (총 손실 한도 초과 시 긴급 전량 청산)
- 영향: 긴급청산 시 L10 (`order_intents_gate`) 우회. _guard 9중은 유지 (KILL_SWITCH/AUTO_TRADING_ENABLED/거래시간/거래일/일일한도)
- 위험도: P2 (긴급청산 = 안전장치 발동 후 경로. 정상 매매 경로 X)
- 발견 일시: 5/29 10:00 (code-analyzer 검수 중 발견)

### 1-2. 목표
긴급청산 경로도 order_intents_gate 10중 가드 강제 → L10 우회 0건 달성.

### 1-3. 처리 방안

#### 옵션 A (권장): mode/executor_bot을 LiveTradingEngine에서 SafetyGuard로 전파
- `SafetyGuard.emergency_liquidate(tracker, order_port, *, mode, executor_bot)` 시그니처 변경
- `LiveTradingEngine._calc_total_loss_pct/check_all` 호출 위치에서 `self._mode/self._executor_bot` 전달
- L128 매도 호출에 명시 전달
- 장점: P1-A 패턴 동일 / backward compat 가능 (default None)
- 단점: SafetyGuard가 OrderPort 호출자가 되는 구조 그대로

#### 옵션 B: 긴급청산 자체를 LiveTradingEngine 내부로 이동
- SafetyGuard는 판정만, 실제 매도 호출은 LiveTradingEngine이 수행
- 장점: 책임 분리 깔끔 (SafetyGuard = 판정, Engine = 실행)
- 단점: 리팩토링 범위 큼 (시그니처 + 호출처 + 테스트)

### 1-4. 권장
**옵션 A** — 작업량 최소 + P1-A 패턴 일관.

### 1-5. 예상 작업량
- 코드 수정: ~10 라인 (safety_guard.py + live_trading.py 호출처 1곳)
- 회귀: `pytest tests/test_phase1_paper_trade.py` 107/107 유지 확인
- 분업 검수: code-analyzer 단독 (security-architect 불필요 — 동일 패턴 확장)
- Codex 검수: 미니 의뢰서 (P1-A 의뢰서 보강)
- **합계**: 30분 + 회귀 15분 + 분업 검수 10분 = **~1시간**

### 1-6. 위험/주의
- LiveTradingEngine.check_all → emergency_liquidate 호출 위치에서 mode/executor_bot 전파 누락 시 default None → backward compat 우회 위험. 명시 전달 필수.

### 1-7. 결단 사항 (사장님)
- 옵션 A vs B
- 본 처리 P1-A와 함께 commit할지 vs 별도 commit

---

## 2. filelock 정책 문서화 (P1-4 Med)

### 2-1. 현재 상태
- 위치: `src/use_cases/order_intents_gate.py:418-422`
- 코드: `with out_path.open("a", encoding="utf-8") as f: f.write(json.dumps(intent, ...) + "\n")`
- 보호: 없음 (append-only는 OK, multi-writer 동시성 보호 X)
- 위험: cron 여러 개 동시 실행 시 jsonl 손상 가능성
- 실제 위험도:
  - **Linux append** (POSIX `O_APPEND` flag): 단일 write() 시스템 콜이 4096 바이트(PIPE_BUF) 이하면 **사실상 atomic** (POSIX 보장)
  - intent JSON 1줄 ≈ 200~500B → 안전
  - VPS는 Linux → 실전 위험 낮음
  - 다만 Python `open("a")`는 buffered I/O라 단일 write() 호출 보장 안 됨 → 최악의 경우 partial write 가능
- 사장님 결정문 평가: Med (Linux 환경 + 단일 cron 가동 단계라 사고 가능성 낮음)

### 2-2. 목표
filelock 부재가 실전 안전에 영향 없는 근거를 문서화하거나, 안전을 위해 filelock 도입.

### 2-3. 처리 방안

#### 옵션 A (권장 — 단기): 정책 문서화 + filelock 미적용 결단
- 근거 문서화:
  - Linux POSIX `O_APPEND` atomic write (4096B 이하) 보장
  - intent JSON 줄당 평균 크기 (실측 grep으로 확인)
  - 단일 cron 가동 단계 (multi-writer race 가능성 미발생 단계)
  - 향후 다중 cron 동시 가동 진입 전 portalocker 도입 결단
- 산출물: `docs/02-design/filelock-policy-5_29.md` (~1페이지)
- 장점: 추가 라이브러리 의존 0 / 즉시 마무리
- 단점: 실측 검증 의무 (intent 줄 크기 + Python write 단일 시스템 콜 보장 여부)

#### 옵션 B (장기): portalocker 도입
- `register_intent` 호출 시 OS-level file lock (Linux flock + Windows msvcrt)
- 코드: ~5 라인 추가
- 의존: `portalocker` 라이브러리 (이미 P0-1 atomic write에 사용 중)
- 장점: 본질적 race 차단 / 향후 다중 cron 안전
- 단점: 추가 lock 비용 (수십 마이크로초)

### 2-4. 권장
**옵션 A + 옵션 B 조건부 도입**:
- 단기: 옵션 A 문서화 → §9 dry-run + paper 재가동 1차 진입 허용
- 중기: 다중 cron 동시 가동 진입 전 옵션 B 적용 (조건부 PDCA)

### 2-5. 예상 작업량
- 옵션 A 문서화: ~1시간 (intent 크기 실측 + Python write 동작 확인 + 정책 명시)
- 옵션 B 코드: ~30분 + 회귀 (별도 차수)

### 2-6. 결단 사항 (사장님)
- 옵션 A vs A+B 동시 vs B 단독
- portalocker import 위치 (register_intent만 vs 모든 jsonl 쓰기)

---

## 3. HMAC rotation playbook (P1-4 Med)

### 3-1. 현재 상태
- 위치: `src/use_cases/order_intents_gate.py:75-90` (`_get_hmac_key`)
- 환경변수: `ORDER_INTENTS_HMAC_KEY` (32+ 문자 강제)
- 현재 키 길이: 64자 (preflight 결과)
- 사용처 파일: 8개 (gate + tests + preflight + 문서)
- 회전 절차: **없음**
- 위험: `.env` 유출 / 백업 노출 시 위조 가능
- 사장님 결정문 평가: Med (오늘 사고 가능성 낮으나 운영 설계 구멍)

### 3-2. 목표
HMAC 키 회전 절차 명시 + 응급 대응 매뉴얼 작성 + 회전 스크립트 (선택).

### 3-3. 처리 방안

#### 옵션 A (권장): 절차 문서화 + 회전 스크립트 + 키 버전 관리
- 산출물: `docs/02-design/hmac-rotation-playbook-5_29.md` (~2페이지)
- 구성:
  1. **정상 회전 절차** (분기 1회 + 사고 시 즉시):
     - 신규 키 생성 (64+ 문자 무작위)
     - .env 키 교체 + 백업 (`.env.bak.20260529_1100`)
     - VPS .env 동기 (scp + 검증)
     - preflight 검증 (`tools/quant_preflight.py`)
     - 단일 cron smoke test (paper intent 등록 → 검증)
     - 기존 등록된 intent 만료 처리 (D+1 자동 만료 활용)
     - VPS 운영 재시작
  2. **응급 회전 절차** (키 유출 의심 시 즉시):
     - KILL_SWITCH 즉시 활성화
     - 신규 키 생성 + .env 교체
     - 등록된 intent 전체 무효화 (HMAC 재검증 실패 → 자동 거부)
     - VPS 동기 + preflight
     - 사고 보고서 작성
  3. **회전 스크립트** (선택): `tools/rotate_hmac_key.py`
     - 신규 키 생성 (`secrets.token_urlsafe(64)`)
     - .env 교체 + 백업
     - VPS scp + 검증
- 장점: 표준 운영 절차 확립 / 사고 시 대응 명확
- 단점: 키 버전 관리는 코드 변경 필요 (옵션 B로 분리)

#### 옵션 B (별도 차수): 다중 키 버전 관리 도입
- `ORDER_INTENTS_HMAC_KEY_V2` 환경변수 추가
- intent에 `key_version` 필드 추가 (P0-5 스키마 확장)
- 회전 기간 동안 V1 + V2 둘 다 검증 허용
- 회전 완료 후 V1 삭제
- 장점: zero-downtime 회전
- 단점: 코드 변경 큼 + intent 스키마 호환성 + HMAC 검증 로직 변경

### 3-4. 권장
**옵션 A 우선 (단기) + 옵션 B 별도 PDCA**:
- 단기: 옵션 A 문서화 → 응급 대응 가능 상태 도달
- 중기: 옵션 B 키 버전 관리 PDCA (별도 차수 — 다중 키 검증은 P1-B 등급 검수 필요)

### 3-5. 예상 작업량
- 옵션 A 문서: ~1시간
- 옵션 A 스크립트: ~30분 (`tools/rotate_hmac_key.py` Python ~50라인)
- 옵션 B (별도 차수): ~4시간 + 회귀 (별도 PDCA)
- **합계 (옵션 A만)**: ~1.5시간

### 3-6. 위험/주의
- 회전 스크립트는 .env에 접근 → 권한 검증 필수 (root only)
- VPS 동기 시 SSH 키 인증만 허용 (사장님 기존 룰)
- 신규 키 생성 후 기존 등록된 intent는 D+1 만료까지 유효 (서명 검증 실패 시 OrderIntentError 자동 거부)

### 3-7. 결단 사항 (사장님)
- 옵션 A 단독 vs A+B 동시 진행
- 회전 스크립트 작성 vs 수동 절차만
- 키 회전 주기 (분기 1회 권장)

---

## 4. §9 4단계 dry-run 정의 + 준비 (P1-2 High)

### 4-1. 현재 상태
- "§9 dry-run"의 정확한 명세 **부재** (메모리에 단편적 언급만)
- 메모리 5/28 19:11 기준 단계 요약:
  1. single-line crontab 초안
  2. `*_MODE=paper` env grep
  3. smoke test
  4. journalctl 5분 모니터링
- 추가 메모리: "Codex 재검수 + 사용자 승인" → 5단계로 확장 가능
- 사장님 결정문 평가: High (운영 재가동 직전 실측 검증 단계)

### 4-2. 목표
§9 4단계 (또는 5단계) dry-run 명세를 docs/02-design/에 명문화 + 실행 스크립트 + 실행 결과 템플릿.

### 4-3. 처리 방안 — 4단계 정의 제안 (사장님 결단)

#### 단계 1: One-line crontab 초안 작성
- 산출물: `ops/restart-cron-draft-5_29.txt` (실제 crontab 적용 X, 초안만)
- 내용: 활성화 후보 paper cron 1줄 + 기존 cron 주석 그대로 유지
- 검증: 정규식 grep으로 `*_MODE=paper` 명시 확인

#### 단계 2: `*_MODE=paper` env grep 검증
- 명령: VPS `grep -E "(LIVE_TRADING_MODE|SELL_MONITOR_MODE|TELEGRAM_TRADING_MODE)" .env`
- 기대: 미설정 (default "paper" fall-back) 또는 명시 `=paper`
- 차단: `=live` 라인 발견 시 즉시 중단

#### 단계 3: Smoke test (코드 미실행 dry-run)
- 명령: `tools/quant_preflight.py --simulate-paper` (옵션 추가 필요)
- 내용:
  - paper intent 등록 시뮬 (실제 KIS 미호출)
  - order_intents_gate 통과 시뮬 (paper mode + quant executor)
  - KILL_SWITCH 효과 검증 (현재 존재 → assert_runtime_orders_allowed raise 확인)
- 산출물: `tools/quant_preflight.py` 옵션 확장 (~20라인)

#### 단계 4: Journalctl 5분 모니터링
- 단계 3 통과 후 단일 cron 1회 실행
- 명령: `sudo journalctl -u quantum-scheduler --since '5 minutes ago' --no-pager`
- 기대: 실주문 0건 / paper intent 등록 N건 / KILL_SWITCH 차단 알림 X
- 차단: 실주문 트레이스 발견 시 즉시 중단

### 4-4. (선택) 단계 5: Codex 재검수 + 사장님 승인

### 4-5. 처리 방안 — 산출물

#### 옵션 A (권장): 명세 + 실행 스크립트 한 번에
- 문서: `docs/02-design/restart-dry-run-spec-5_29.md` (4~5단계 명세)
- 스크립트: `tools/quant_preflight.py` 옵션 확장 (`--simulate-paper`)
- 산출물 템플릿: `ops/restart-dry-run-result-template.md` (실행 후 채울 양식)

#### 옵션 B: 명세만 작성, 실행 스크립트는 별도 차수
- 단기: 명세 명문화 → §9 정의 고정
- 중기: 실행 스크립트 작성 후 1차 실행

### 4-6. 권장
**옵션 A** — 명세 + 스크립트 + 템플릿 한 번에. 단, **실행은 사장님 명시 승인 후만**.

### 4-7. 예상 작업량
- 옵션 A 명세: ~1시간
- 옵션 A 스크립트: ~1시간 (`tools/quant_preflight.py` 옵션 추가 + 테스트)
- 옵션 A 템플릿: ~30분
- **합계 (옵션 A)**: ~2.5시간
- **실측 실행** (사장님 승인 후): ~30분 (4단계 + journalctl 5분)

### 4-8. 위험/주의
- 단계 4 journalctl 모니터링 중 실제 매매 발생 가능성 0%여야 함 → 매매 cron 6개 정지 + scheduler masked 유지 필수
- 단일 cron 실행 시 paper intent만 등록되고 실주문 0 보장 (코드 단계에서 검증)
- 실측 단계라 사장님 입회 권장

### 4-9. 결단 사항 (사장님)
- 옵션 A vs B
- 4단계 vs 5단계 (Codex 재검수 포함 여부)
- 실측 실행 시점 (P1-A commit 전 vs 후)

---

## 5. 4건 종합 진행 시퀀스 (권장)

```
[현재] 5/29 P1-A 코드 변경 미커밋 + 분업 검수 PASS + Codex 의뢰 대기
   ↓
[권장 1] safety_guard.emergency_liquidate P2 마이그레이션 (~1시간)
   → 회귀 107/107 PASS 확인 + code-analyzer 검수
   ↓
[권장 2] filelock 정책 문서화 (옵션 A, ~1시간)
   → docs/02-design/filelock-policy-5_29.md
   ↓
[권장 3] HMAC rotation playbook (옵션 A, ~1.5시간)
   → docs/02-design/hmac-rotation-playbook-5_29.md + tools/rotate_hmac_key.py (선택)
   ↓
[권장 4] §9 4단계 dry-run 명세 + 스크립트 (옵션 A, ~2.5시간)
   → docs/02-design/restart-dry-run-spec-5_29.md + tools/quant_preflight.py 확장
   ↓
[필수] Codex 의뢰서 보강 (4건 모두 검수 요청)
   ↓
[필수] 사장님 최종 commit 승인
   ↓
[필수] 재가동 심사서 작성 (별도 산출물)
   ↓
[필수] 사장님 별도 paper cron 가동 결단
   ↓
[가능] 제한적 paper cron 1줄 가동 (전체 X)
```

**총 작업량 (1~4)**: ~6시간 (오늘 종일 진행 시 완료 가능)

---

## 6. 사장님 결단 요청 사항 (8건)

### 6-1. safety_guard P2
- 옵션 A (P1-A 패턴 확장) vs B (책임 분리 리팩토링)
- P1-A와 함께 commit vs 별도 commit

### 6-2. filelock
- 옵션 A (문서화만) vs A+B (portalocker 동시 도입) vs B 단독

### 6-3. HMAC rotation
- 옵션 A (문서 + 스크립트) vs A 문서만 vs A+B (다중 키 버전 동시)
- 회전 스크립트 작성 여부

### 6-4. §9 dry-run
- 옵션 A (명세+스크립트+템플릿) vs B (명세만)
- 4단계 vs 5단계 (Codex 재검수 포함)
- 실측 실행 시점

### 6-5. 진행 순서
- 권장 순서 (P2 → filelock → HMAC → §9) vs 다른 순서

### 6-6. commit 시점
- 4건 완료 후 일괄 commit (P1-A 4건 + 잔여 4건 = 8건 묶음)
- 권장 2건씩 단계 commit

### 6-7. 분업 검수 적용
- 4건 모두 security-architect + code-analyzer 분업 vs 핵심 항목만

### 6-8. 작업 시간
- 오늘 종일 진행 vs 분할 진행

---

## 7. 적용 금지 (5/29 종일 유지)
- ❌ cron 복원 / quantum-scheduler unmask
- ❌ 실거래 경로 실행
- ❌ 단독 commit (사장님 명시 승인 후만)
- ❌ paper 재가동 선언
- ❌ "운영 안전 완성" / "Phase 1 paper 가능" / "전체 L10 완료" 표현

---

## 8. 표현 룰

### 사용 가능
- "재가동 자료 4건 처리 계획 1차 산출"
- "P1-A 호출자 마이그레이션은 PASS"
- "운영 재가동은 filelock, HMAC rotation playbook, §9 dry-run 완료 전까지 HOLD"

### 사용 금지
- "재가동 임박" X
- "운영 안전 완성" X
- "Phase 1 paper 가능" X

---

## 9. 연결 문서
- `docs/02-design/p1-truth-pack-5-29.md` (P1 Truth Pack 1차)
- `docs/01-plan/trading-factory-v1-architecture.md` §9 (잔여 위험 — 4단계 dry-run 정의 부재 발견)
- `ops/codex_outbox/20260529T101341_..._review-requested.md` (P1-A Codex 의뢰서)
- `memory/decision_5_28_p1_blockers.md` (사장님 결정문)
