# 묶음 A/B/C exact diff 패키지 — 5/29(금)

> **상태**: commit 후보 정확 분리 패키지 (실제 commit 0건, Codex 회신 + 사장님 별도 승인 후만)
> **계기**: 5/29 사장님 결단 — "Codex 회신과 사장님 승인 즉시 판단할 수 있도록 commit 후보를 정확히 쪼갠다"
> **상위 문서**: `docs/02-design/deletion-quarantine-audit-5_29.md` + `docs/01-plan/v2-backlog-5_29.md`

---

## 0. 한 줄 결론

> **5/29 종일 작업을 3개 묶음으로 정확 분리. 묶음 A (코드 ~30라인) → 묶음 B (운영 안전 문서 5건) → 묶음 C (audit + diff + backlog 3건). 적용 순서: Codex 회신 + 사장님 별도 승인 → A → B → C → (별도) DELETE 2건 + Flask 비활성화.**

---

## 1. 묶음 A — 코드 게이트

### 1-1. 변경 파일 목록 (9 파일)
| # | 파일 | 변경 종류 |
|---|---|---|
| 1 | `src/use_cases/ports.py` | OrderPort abstract 시그니처 확장 (mode/executor_bot kwargs) |
| 2 | `src/telegram_command_handler.py` | TELEGRAM_TRADING_MODE default "live" → "paper" (3곳) |
| 3 | `scripts/sell_monitor.py` | SELL_MONITOR_MODE env + mode/executor_bot 인자 (2곳) + SmartSellExecutor 생성 인자 (1곳) |
| 4 | `src/use_cases/live_trading.py` | __init__ mode/executor_bot 추가 + 매매 호출 5곳 + create_live_engine LIVE_TRADING_MODE env + L411-418 Critical 봉합 |
| 5 | `src/use_cases/smart_sell.py` | __init__ mode/executor_bot + 매매 호출 3곳 |
| 6 | `src/use_cases/safety_guard.py` | emergency_liquidate keyword-only mode/executor_bot |
| 7 | `main.py` | cmd_emergency_stop LIVE_TRADING_MODE env + 명시 전달 |

(파일 수 7개 — 호출처 영향 받은 파일 + 위와 같음)

### 1-2. 변경 라인 수
- 약 30 라인 + Critical 봉합 8 라인 = **총 ~38 라인**

### 1-3. 목적
- P1-A 4건: order_intents_gate 10중 가드 강제 (sell_monitor / smart_sell / live_trading / telegram default)
- P2 1건: safety_guard.emergency_liquidate 동일 적용
- Critical 봉합: replace_all=true Edit 시 발생한 들여쓰기 결함 복구

### 1-4. 위험
- Low: 회귀 25/25 + preflight 10/10 PASS 검증 완료
- 잠재: 시간 의존 fixture 자연 만료 (별도 P3, B-10)
- 잠재: live_trading import 회귀 사각지대 (별도 P3, B-9)

### 1-5. rollback 방법
```bash
# commit hash X 알고 있다면
git revert <commit-hash>
# 또는 묶음별 분리 commit이라면 묶음 A만 revert
```
- 사전: VPS 변경 0건 (로컬만 commit) → rollback 영향 0

### 1-6. 회귀 명령
```bash
source venv/Scripts/activate
python -u -X utf8 -m pytest tests/test_phase1_paper_trade.py::TestCallerPassesModeExecutorBot tests/test_phase1_paper_trade.py::TestC2_NoSilentReturn tests/test_phase1_paper_trade.py::TestC3_HmacKeyFailFast -q --tb=short
# 기대: 25 passed

python -u -X utf8 tools/quant_preflight.py
# 기대: RESULT: PASS (10/10)

python -c "from src.use_cases.live_trading import LiveTradingEngine, create_live_engine; print('import OK')"
# 기대: import OK
```

### 1-7. commit 메시지 후보
```
feat(safety): P1-A 호출자 마이그레이션 + P2 safety_guard + Critical 봉합 (5/29 사장님 결단)

P1-A 4건 (5/29 오전):
- sell_monitor.py mode/executor_bot 인자 + SELL_MONITOR_MODE env
- smart_sell.py mode/executor_bot 인자 (__init__ + 매매 3곳)
- live_trading.py mode/executor_bot 인자 (__init__ + 매매 5곳) + create_live_engine
- telegram_command_handler.py default "live" → "paper" (3곳)
- ports.py OrderPort 시그니처 확장 (backward compat)

P2 1건 (5/29 오후):
- safety_guard.emergency_liquidate keyword-only mode/executor_bot
- live_trading 호출처 2곳 (L111, L411) self._mode 전파
- main.py cmd_emergency_stop LIVE_TRADING_MODE env

Critical 봉합 (5/29 오후 분업 검수 중 발견):
- live_trading.py L411-418 들여쓰기 결함 (replace_all=true 부작용)

회귀: 25/25 PASS + preflight 10/10 PASS + import OK
검수: bkit:security-architect + bkit:code-analyzer 분업 PASS (Critical 봉합 후)
Codex 의뢰서: ops/codex_outbox/20260529T101341..., 20260529T194250...

연결: docs/02-design/p1-truth-pack-5-29.md, p1-residual-plan-5-29.md
```

### 1-8. 적용 전 승인 조건
- ✅ Codex 회신 ① P1-A 4건 PASS
- ✅ Codex 회신 ② P2 + 4건 PASS
- ✅ 사장님 묶음 A commit 별도 승인

### 1-9. 적용 후 검증 조건
- `git log -1 --oneline` 확인
- 회귀 25/25 + preflight 10/10 재확인
- `git status` clean
- VPS git pull 보류 (사장님 별도 결단)

---

## 2. 묶음 B — 운영 안전 문서

### 2-1. 변경 파일 목록 (5 파일)
| # | 파일 | 분량 |
|---|---|---|
| 1 | `docs/02-design/p1-truth-pack-5-29.md` | ~280 라인 |
| 2 | `docs/02-design/p1-residual-plan-5-29.md` | ~280 라인 |
| 3 | `docs/02-design/filelock-policy-5_29.md` | ~150 라인 |
| 4 | `docs/02-design/hmac-rotation-playbook-5_29.md` | ~210 라인 |
| 5 | `docs/02-design/restart-dry-run-spec-5_29.md` | ~250 라인 |

### 2-2. 변경 라인 수
- 신규 작성 약 **1,170 라인** (모두 신규 파일)

### 2-3. 목적
- 운영 재가동 심사 자료 명문화
- filelock 정책 / HMAC rotation 절차 / §9 dry-run 명세 확립

### 2-4. 위험
- 매우 낮음 (문서만)
- 단, 향후 운영 단계에서 본 문서를 기준으로 작업 → 정확성 의무

### 2-5. rollback 방법
```bash
git revert <commit-hash>
# 영향 0건 (문서)
```

### 2-6. 회귀 명령
- 해당 없음 (문서)

### 2-7. commit 메시지 후보
```
docs(safety): 운영 안전 문서 5건 (5/29 P1-A + P2 + 잔여 4건 계획서)

신규 5건:
- p1-truth-pack-5-29.md: P1 호출자 마이그레이션 검증 매트릭스
- p1-residual-plan-5-29.md: 잔여 4건 처리 계획 (P2 + filelock + HMAC + §9)
- filelock-policy-5_29.md: portalocker 미적용 결단 + POSIX 4중 보호
- hmac-rotation-playbook-5_29.md: 정상/응급 회전 절차 + 스크립트 명세
- restart-dry-run-spec-5_29.md: §9 4단계 dry-run 정식 명세

검수: bkit:security-architect + bkit:code-analyzer 분업 PASS
미세 보강 5건 반영 (POSIX 인용 / 매매 cron grep / journalctl grep / 정규식 / rollback)
Codex 의뢰서: ops/codex_outbox/20260529T194250...
```

### 2-8. 적용 전 승인 조건
- ✅ 묶음 A commit 완료
- ✅ Codex 회신 ② 문서 부분 PASS (또는 별도 권장사항 반영)
- ✅ 사장님 묶음 B commit 별도 승인

### 2-9. 적용 후 검증 조건
- `git log -1 --oneline` 확인
- 5 파일 모두 `git ls-files` 등록 확인
- 회귀 25/25 재확인 (영향 0 예상)

---

## 3. 묶음 C — Deletion/Quarantine 문서

### 3-1. 변경 파일 목록 (4 파일)
| # | 파일 | 분량 |
|---|---|---|
| 1 | `docs/02-design/deletion-quarantine-audit-5_29.md` | ~290 라인 |
| 2 | `docs/02-design/flask-liquidate-route-disable-diff-5_29.md` | ~160 라인 |
| 3 | `docs/01-plan/v2-backlog-5_29.md` | ~180 라인 (11 항목) |
| 4 | `docs/02-design/archive_move_precheck_5_29.md` | ~200 라인 (본 차수 추가) |
| 5 | `docs/02-design/flask_remote_queue_audit_5_29.md` | ~190 라인 (본 차수 추가) |
| 6 | `docs/02-design/etf_rotation_p3_audit_5_29.md` | ~210 라인 (본 차수 추가) |
| 7 | `docs/02-design/restart_dry_run_execution_checklist_5_29.md` | ~280 라인 (본 차수 추가) |
| 8 | `docs/02-design/quant_commit_bundle_plan_5_29.md` | (본 문서) |

### 3-2. 변경 라인 수
- 신규 작성 약 **1,510 라인** (모두 신규 파일)

### 3-3. 목적
- deletion/quarantine 5분류 결과 명문화
- archive 이동 사전 검증 + 묶음 패키지 + §9 체크리스트
- v2 backlog 11건 등록

### 3-4. 위험
- 매우 낮음 (문서만)
- 단, B-6/B-7/B-8 적용 시 본 문서를 기준으로 작업 → 정확성 의무

### 3-5. rollback 방법
```bash
git revert <commit-hash>
# 영향 0건 (문서)
```

### 3-6. 회귀 명령
- 해당 없음 (문서)

### 3-7. commit 메시지 후보
```
docs(audit): deletion/quarantine audit + diff 초안 + v2 backlog (5/29)

신규 8건:
- deletion-quarantine-audit-5_29.md: 8 카테고리 5분류 결과
- flask-liquidate-route-disable-diff-5_29.md: B-8 diff 초안 (1차 — liquidate만)
- v2-backlog-5_29.md: 11 항목 (B-1 ~ B-11)
- archive_move_precheck_5_29.md: DELETE 2건 6종 reference 검증
- flask_remote_queue_audit_5_29.md: Flask queue 2차 전수 감사 (3건 QUARANTINE 확장)
- etf_rotation_p3_audit_5_29.md: ETF broker 미연결 확정
- restart_dry_run_execution_checklist_5_29.md: §9 실측 전 체크리스트
- quant_commit_bundle_plan_5_29.md: 본 묶음 패키지

검수: Explore 서브에이전트 분업 (archive + cron + Flask + ETF)
신규 발견: Flask queue 3건 QUARANTINE / ETF broker 0건 확정
v2 backlog: MERGE 4 + DELETE 2 + QUARANTINE 1 + P3 회귀 2 + P3 audit 2
```

### 3-8. 적용 전 승인 조건
- ✅ 묶음 A + 묶음 B commit 완료
- ✅ 사장님 묶음 C commit 별도 승인

### 3-9. 적용 후 검증 조건
- `git log -1 --oneline` 확인
- 8 파일 모두 `git ls-files` 등록 확인
- B-1 ~ B-11 항목이 v2-backlog.md 안에 모두 등재 확인

---

## 4. 묶음 외 — DELETE/QUARANTINE 실제 적용 묶음 (별도)

### 4-1. 묶음 D (제안 — DELETE 2건 archive 이동)
- 사전 조건: 묶음 A/B/C commit 완료 + 사장님 B-6/B-7 적용 별도 승인
- 변경:
  - `git mv scripts/dashboard.py scripts/archive/deprecated/dashboard.py`
  - `mkdir -p scripts/archive/orphan_20260529`
  - `git mv scripts/one_off/integration_dryrun_5_20.py scripts/archive/orphan_20260529/integration_dryrun_5_20.py`
- 회귀: 25/25 + preflight 10/10 재확인
- commit 메시지: `chore(archive): DELETE 2건 archive 이동 (5/29 사장님 결단 + Codex 검수)`

### 4-2. 묶음 E (제안 — Flask 큐 3건 비활성화)
- 사전 조건: 묶음 A/B/C 완료 + 사장님 B-8 적용 별도 승인 (3건 확장 결단 포함)
- 변경:
  - `website/flask_app.py` L789-792 (`start`) 비활성화
  - `website/flask_app.py` L804-807 (`stop`) 비활성화
  - `website/flask_app.py` L956-968 (`liquidate`) 비활성화
  - `website/flask_app.py` L947-950 도움말 3 라인 제거
- 회귀: 25/25 + preflight 10/10 재확인 + (선택) Flask 단위 테스트
- commit 메시지: `feat(safety): Flask remote_queue dead-letter 3건 비활성화 (B-8)`

### 4-3. 묶음 F (제안 — preflight --simulate-paper 코드 작성)
- 사전 조건: 묶음 A/B/C 완료 + 사장님 별도 승인
- 변경: `tools/quant_preflight.py` S1~S6 가드 추가
- 회귀: 신규 simulate-paper 통과 + 기존 회귀 영향 0건
- commit 메시지: `feat(preflight): --simulate-paper S1~S6 6개 가드 추가 (§9 단계 3)`

### 4-4. 묶음 G (제안 — tools/rotate_hmac_key.py 코드 작성)
- 사전 조건: 묶음 A/B/C 완료 + 사장님 별도 승인
- 변경: `tools/rotate_hmac_key.py` 신규 작성 (~70 라인)
- 회귀: 신규 단위 테스트 + .env 백업 안전성 확인
- commit 메시지: `feat(security): HMAC key rotation 스크립트 추가 (정상/응급 회전)`

---

## 5. 적용 순서 (사장님 결단 후)

```
[현재] Codex 회신 ①+② 대기
   ↓
[필수] Codex 회신 도달 + 사장님 보고
   ↓
[필수] 사장님 묶음 A commit 별도 승인
   ↓
[적용] 묶음 A commit + 회귀 재확인
   ↓
[필수] 사장님 묶음 B commit 별도 승인
   ↓
[적용] 묶음 B commit
   ↓
[필수] 사장님 묶음 C commit 별도 승인
   ↓
[적용] 묶음 C commit
   ↓
[필수] 사장님 묶음 D/E 적용 별도 승인 (DELETE + Flask 비활성화)
   ↓
[적용] 묶음 D (archive 이동) + 묶음 E (Flask 비활성화) 각각 별도 commit
   ↓
[선택] 묶음 F + 묶음 G (preflight 확장 + HMAC 스크립트)
   ↓
[필수] 사장님 §9 dry-run 실측 별도 승인
   ↓
[적용] §9 4단계 실측 (`docs/02-design/restart_dry_run_execution_checklist_5_29.md` 참고)
   ↓
[필수] 사장님 paper cron 1줄 등록 결단
   ↓
[가능] 제한적 paper cron 1줄 가동 (전체 X)
```

---

## 6. 묶음간 의존성

| 묶음 | 의존 | 이유 |
|---|---|---|
| A | - | 코드 게이트 (선행 0) |
| B | A 완료 | 문서가 코드 변경 반영 |
| C | A + B 완료 | audit + diff가 A/B 결과 인용 |
| D (DELETE) | A + B + C 완료 + 별도 승인 | 사장님 B-6/B-7 결단 |
| E (Flask) | A + B + C 완료 + 별도 승인 | 사장님 B-8 3건 확장 결단 |
| F (preflight) | A + B + C 완료 + 별도 승인 | §9 단계 3 실행 전 |
| G (HMAC 스크립트) | A + B + C 완료 + 별도 승인 | 운영 시작 전 |

→ **F + G는 §9 dry-run 실측 진입 전 의무**.

---

## 7. 적용 금지 (본 패키지 작성 후)

- ❌ 본 패키지 이전 묶음 A/B/C 임의 commit X
- ❌ 사장님 별도 승인 없이 묶음 D/E/F/G 적용 X
- ❌ §9 dry-run 실측 실행 X
- ❌ paper cron 등록 X
- ❌ 단독 commit X

---

## 8. 표현 룰

### 사용 가능
- "묶음 A/B/C exact diff 패키지 작성 완료"
- "commit 후보 정확 분리"
- "사장님 결단 + Codex 회신 즉시 판단 가능 상태"

### 사용 금지
- "묶음 적용 완료" X (commit 0건)
- "운영 안전 완성" X
- "재가동 가능" X

---

## 9. 연결 문서
- `docs/02-design/p1-truth-pack-5-29.md`
- `docs/02-design/p1-residual-plan-5-29.md`
- `docs/02-design/filelock-policy-5_29.md`
- `docs/02-design/hmac-rotation-playbook-5_29.md`
- `docs/02-design/restart-dry-run-spec-5_29.md`
- `docs/02-design/restart_dry_run_execution_checklist_5_29.md`
- `docs/02-design/deletion-quarantine-audit-5_29.md`
- `docs/02-design/flask-liquidate-route-disable-diff-5_29.md`
- `docs/02-design/flask_remote_queue_audit_5_29.md`
- `docs/02-design/etf_rotation_p3_audit_5_29.md`
- `docs/02-design/archive_move_precheck_5_29.md`
- `docs/01-plan/v2-backlog-5_29.md`
- `ops/codex_outbox/20260529T101341_..._p1-a4-callers-migration_review-requested.md`
- `ops/codex_outbox/20260529T194250_..._p2-residual-4items-and-critical-fix_review-requested.md`
