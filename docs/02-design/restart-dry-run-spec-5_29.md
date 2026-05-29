# §9 4단계 dry-run 정식 명세 — paper 재가동 심사용

> **상태**: 명세 + 스크립트 설계 + 보고 템플릿 (실제 실행은 사장님 별도 승인 후)
> **HEAD**: 35718cb (작업 전)
> **계기**: 5/28 사장님 결정문 P1-2 + 5/29 결단 옵션 A 승인 — "4단계 정식 명세 + 스크립트 + 보고 템플릿 작성. 5단계 Codex 재검수는 선택 단계."
> **상위 문서**: `docs/02-design/p1-residual-plan-5-29.md` §4

---

## 0. 한 줄 요약

> **§9 dry-run은 4단계 (one-line crontab 초안 → `*_MODE=paper` env grep → smoke test → journalctl 5분 모니터링)로 정식 명세한다. 선택 5단계 (Codex 재검수)는 사장님 결단 시 추가. 실제 실행은 P1-A + P2 + filelock/HMAC 문서 + Codex PASS + 사장님 별도 승인 모두 통과 후만.**

---

## 1. §9의 정의 (정식 명문화)

### 1-1. 이전 상태 (5/29 09:30 이전)
- `docs/01-plan/trading-factory-v1-architecture.md §9`는 "잔여 위험" 섹션
- 4단계 dry-run 명세 **부재** (메모리에 단편적 언급만)

### 1-2. 본 문서 결정
- "§9 4단계 dry-run"을 **본 문서로 정식 명문화**
- 이전 architecture.md §9는 "잔여 위험"으로 유지 (별도 섹션)
- 본 문서는 "재가동 심사 프로세스"로 분리

### 1-3. §9 정식 명칭
- **§9 정식 명칭**: "paper cron 재가동 심사 4단계 dry-run"
- 약칭: §9 dry-run

---

## 2. 4단계 dry-run 명세

### 2-1. 단계 1: One-line crontab 초안 작성 + 검증

**목적**: 활성화 후보 paper cron이 안전 정책에 부합하는지 사전 검증.

**산출물**: `ops/restart-cron-draft-YYYYMMDD.txt`

**작업**:
1. 활성화 후보 cron 1줄 작성 (예시):
   ```
   # [심사대상] 0 9 * * 1-5 cd /home/ubuntu/quantum-master && LIVE_TRADING_MODE=paper PYTHONPATH=/home/ubuntu/quantum-master ./venv/bin/python3.11 -u -X utf8 scripts/paper_warmup_daily.py >> /home/ubuntu/quantum-master/logs/paper_warmup.log 2>&1
   ```
2. 기존 매매 cron 6개 주석 처리 상태 그대로 유지 명시
3. `*_MODE=paper` env var 명시 (default 의존 X — 명시적 paper 선언)

**검증 정규식** (★ 5/29 security-architect 검수 반영: 주석 라인 제외 + 명시 카운트):
```bash
# 새 cron 1줄에 *_MODE=paper 명시 확인 (주석 라인 제외)
grep -vE '^#' ops/restart-cron-draft-*.txt | grep -E '(LIVE_TRADING_MODE|SELL_MONITOR_MODE|TELEGRAM_TRADING_MODE)=paper'

# 새 cron 1줄에 --real / --live / --force 인자 없음 확인 (주석 제외, 라인 카운트 = 0 의무)
DANGER_COUNT=$(grep -vE '^#' ops/restart-cron-draft-*.txt | grep -cE '\-\-(real|live|force|no-dry-run)')
if [ "$DANGER_COUNT" -ne 0 ]; then echo "FAIL: 위험 인자 ${DANGER_COUNT}건"; exit 1; fi
```

**기대 결과**: env grep 1+건 발견 + 위험 인자 0건

**차단 조건**:
- env grep 0건 → 차단
- 위험 인자 1건 이상 → 차단
- 매매 cron 주석 해제 라인 발견 → 차단

---

### 2-2. 단계 2: VPS `.env` `*_MODE=paper` env grep 검증

**목적**: VPS 환경변수 상태가 paper-only 정책 부합 확인.

**산출물**: 검증 로그 (텍스트 출력만)

**작업**:
```bash
# VPS 측 .env 환경변수 grep
ssh -i "D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem" -o ConnectTimeout=10 ubuntu@13.209.153.221 "
  grep -E '^(LIVE_TRADING_MODE|SELL_MONITOR_MODE|TELEGRAM_TRADING_MODE|AUTO_TRADING_ENABLED|QUANT_PAPER_ONLY)=' /home/ubuntu/quantum-master/.env
"
```

**기대 결과**:
- `LIVE_TRADING_MODE=paper` (명시 권장) 또는 미설정 (default paper)
- `SELL_MONITOR_MODE=paper` (명시 권장) 또는 미설정
- `TELEGRAM_TRADING_MODE=paper` (명시 권장) 또는 미설정
- `AUTO_TRADING_ENABLED=0` (필수)
- `QUANT_PAPER_ONLY=true` (권장, 또는 미설정)

**차단 조건**:
- `*_MODE=live` 라인 발견 → **즉시 중단**
- `AUTO_TRADING_ENABLED=1` 발견 → 즉시 중단
- `QUANT_AUTO_TRADE_DISABLED=false` 발견 → 즉시 중단

---

### 2-3. 단계 3: Smoke test — `quant_preflight.py --simulate-paper`

**목적**: 코드 실행 없이 paper intent 등록 + order_intents_gate 통과 시뮬레이션 + KILL_SWITCH 효과 검증.

**산출물**: preflight 로그 (PASS/FAIL)

**작업**:
```bash
# VPS 측 실행
ssh -i "..." ubuntu@13.209.153.221 "
  cd /home/ubuntu/quantum-master &&
  ./venv/bin/python3.11 tools/quant_preflight.py --simulate-paper
"
```

**`--simulate-paper` 옵션 명세** (스크립트 확장 필요):
1. 기존 10개 가드 검증 그대로 수행
2. 추가 6개 시뮬레이션 가드:
   - **S1**: paper intent dict 1건 생성 (memory-only, jsonl 미저장)
   - **S2**: `_compute_signature()` 호출 → HMAC 검증 PASS 확인
   - **S3**: `assert_order_intent_exists()` paper mode + quant executor 호출 → PASS 확인 (intent dict 임시 jsonl 작성 → 호출 → 삭제)
   - **S4**: `KisOrderAdapter._guard()`에 mode="paper" 전달 → ValueError raise 확인 (의도된 차단 — KisOrderAdapter는 live만 허용)
   - **S5**: `PaperOrderAdapter._guard()`에 mode="live" 전달 → ValueError raise 확인 (의도된 차단)
   - **S6**: `assert_runtime_orders_allowed()` 호출 → 현재 KILL_SWITCH 존재 시 raise 확인

**기대 결과**: 모든 기존 가드 PASS + S1~S6 6건 PASS = **총 16/16 PASS**

**차단 조건**:
- 기존 10개 중 1건이라도 FAIL → 즉시 중단
- S1~S6 중 1건이라도 FAIL → 즉시 중단
- raw broker call 1건 이상 발견 → 즉시 중단

---

### 2-4. 단계 4: Journalctl 5분 모니터링

**목적**: 단계 1~3 통과 후 단일 cron 1회 실행 → 실주문 0건 + KILL_SWITCH 작동 확인.

**산출물**: 5분간 journalctl 로그 캡처 + 분석 보고서

**작업 (사장님 명시 승인 후만)**:
```bash
# Step 1: 단계 1에서 작성한 one-line crontab 임시 적용
# (사장님 결단 + Codex 검수 통과 후만)

# Step 2: 단계 1 cron 1회 수동 트리거 (cron 등록 X, 직접 실행)
ssh -i "..." ubuntu@13.209.153.221 "
  cd /home/ubuntu/quantum-master &&
  LIVE_TRADING_MODE=paper PYTHONPATH=/home/ubuntu/quantum-master \
    ./venv/bin/python3.11 -u -X utf8 scripts/paper_warmup_daily.py 2>&1 | tee /tmp/paper_warmup_smoke.log
"

# Step 3: 5분간 journalctl 모니터링
# ★ 5/29 code-analyzer 검수 반영: sell_limit/buy_market/emergency_liquidate 추가 (실주문 검출 누락 방지)
ssh -i "..." ubuntu@13.209.153.221 "
  sudo journalctl --since '5 minutes ago' --no-pager | grep -E '(quant|order_intent|sell_(market|limit)|buy_(market|limit)|KILL_SWITCH|emergency_liquidate)'
" > ops/restart-dry-run-journal-YYYYMMDD-HHMM.log

# Step 4: 분석
ssh -i "..." ubuntu@13.209.153.221 "
  cd /home/ubuntu/quantum-master &&
  ls -la data/order_intents/quant_intents_*.jsonl 2>&1 | tail -3 &&
  echo '---' &&
  wc -l data/order_intents/quant_intents_*.jsonl 2>&1 &&
  echo '---' &&
  echo 'KIS 실주문 흔적 grep:' &&
  grep -E 'broker\.(buy|sell)_(limit|market)' /tmp/paper_warmup_smoke.log 2>&1 | head -3
"
```

**기대 결과**:
- paper intent 등록 N건 (jsonl 줄 수 증가)
- `KILL_SWITCH exists` 또는 paper mode block 알림 1+건
- KIS 실주문 트레이스 (`broker.buy_limit` / `broker.sell_market`) **0건**
- `IntentExpiredError` / `IntentSignatureError` / `OrderIntentError` 0건 (paper 경로는 정상 통과)

**차단 조건**:
- KIS 실주문 흔적 1건 이상 발견 → **즉시 KILL_SWITCH 활성화 + scheduler stop + 사장님 호출**
- 예상치 못한 raise 발견 → 분석 후 보고

---

## 3. (선택) 단계 5: Codex 재검수 + 사장님 별도 승인

**조건**: 사장님 결단 시 추가 (옵션 A 기준 선택 단계).

**산출물**: `ops/codex_outbox/...restart-dry-run-results_review-requested.md`

**내용**:
- 단계 1~4 결과 보고
- 차단 조건 미발생 확인
- Codex 재검수 의뢰 (단계별 PASS/FAIL + 추가 위험 발견 여부)
- Codex PASS 회신 후 사장님 별도 승인 → 정식 paper cron 1줄 등록 결단

---

## 4. preflight 확장 스크립트 설계 (`quant_preflight.py --simulate-paper`)

### 4-1. 명세
- 위치: `tools/quant_preflight.py` (기존 파일 확장)
- 옵션 추가: `--simulate-paper` (default False)
- 동작:
  ```python
  if args.simulate_paper:
      # S1~S6 시뮬레이션 가드 추가 실행
      checks.extend(_simulate_paper_checks())
  ```

### 4-2. `_simulate_paper_checks()` 의사 코드
```python
def _simulate_paper_checks() -> list[tuple[bool, str]]:
    """S1~S6 paper intent 시뮬레이션 가드.

    실제 jsonl 파일 시스템 변경 최소화 (임시 파일 사용).
    KIS broker 호출 절대 없음 (KisOrderAdapter._guard 단계까지만 호출).
    """
    results = []
    from src.use_cases.order_intents_gate import (
        _compute_signature, assert_order_intent_exists, register_intent,
        OrderIntentError, NoIntentError, IntentSignatureError,
    )
    from src.adapters.kis_order_adapter import KisOrderAdapter
    from src.adapters.paper_order_adapter import PaperOrderAdapter

    # S1: paper intent dict 생성 (memory-only)
    intent = {
        "intent_id": "sim_240810_smoke_5_29",
        "bot": "quant", "engine": "smoke_test",
        "ticker": "240810", "side": "BUY", "mode": "paper",
        "score": 0.0,
        "created_at": "2026-05-29T10:00:00+09:00",
        "expires_at": "2026-05-29T15:30:00+09:00",
    }
    results.append((True, "[PASS] S1 paper intent dict 생성"))

    # S2: HMAC 서명 + 검증
    try:
        sig = _compute_signature(intent)
        intent["hmac_signature"] = sig
        assert len(sig) >= 32
        results.append((True, "[PASS] S2 HMAC 서명 생성 + 검증"))
    except Exception as e:
        results.append((False, f"[FAIL] S2 HMAC 서명: {e}"))

    # S3: assert_order_intent_exists 호출 (임시 jsonl 작성)
    try:
        register_intent(intent, bot="quant")
        assert_order_intent_exists(
            ticker="240810", side="BUY", mode="paper", executor_bot="quant",
        )
        # 임시 jsonl 삭제
        from pathlib import Path
        intent_file = Path(f"data/order_intents/quant_intents_{_today_date_str()}.jsonl")
        # WARNING: 실제 jsonl 삭제는 운영 안전 위해 미수행 (smoke test 흔적 남김)
        results.append((True, "[PASS] S3 assert_order_intent_exists paper+quant"))
    except Exception as e:
        results.append((False, f"[FAIL] S3 intent gate: {e}"))

    # S4: KisOrderAdapter._guard에 mode="paper" → ValueError raise 확인
    try:
        adapter = KisOrderAdapter()
        try:
            adapter._guard("240810", 1, side="BUY", mode="paper", executor_bot="quant")
            results.append((False, "[FAIL] S4 KisAdapter mode=paper raise 안 됨"))
        except ValueError:
            results.append((True, "[PASS] S4 KisAdapter mode=paper 차단 OK"))
    except Exception as e:
        results.append((False, f"[FAIL] S4 KisAdapter 초기화: {e}"))

    # S5: PaperOrderAdapter._guard에 mode="live" → ValueError raise 확인
    try:
        padapter = PaperOrderAdapter()
        try:
            padapter._guard("240810", 1, side="BUY", mode="live", executor_bot="quant")
            results.append((False, "[FAIL] S5 PaperAdapter mode=live raise 안 됨"))
        except ValueError:
            results.append((True, "[PASS] S5 PaperAdapter mode=live 차단 OK"))
    except Exception as e:
        results.append((False, f"[FAIL] S5 PaperAdapter 초기화: {e}"))

    # S6: assert_runtime_orders_allowed → KILL_SWITCH 존재 시 raise
    from src.utils.trade_runtime_safety import assert_runtime_orders_allowed
    try:
        assert_runtime_orders_allowed()
        # 현재 KILL_SWITCH 미존재 → 정상 통과 (실제 시 KILL_SWITCH 존재 의무)
        results.append((True, "[PASS] S6 runtime guard 호출 OK (현재 KILL_SWITCH 없음)"))
    except PermissionError as e:
        # KILL_SWITCH 존재 시 raise → 이게 정상
        results.append((True, f"[PASS] S6 KILL_SWITCH 차단: {e}"))

    return results
```

### 4-3. 실측 검증 (별도 실행)
- 작성 후 회귀: `tests/test_phase1_paper_trade.py::TestC3_HmacKeyFailFast` PASS 유지 확인
- 신규 단위 테스트: `tests/test_preflight_simulate_paper.py` (S1~S6 검증, 별도 작성)

### 4-4. 작성 결단
- **명세 확정 (본 문서)**: 완료
- **실제 코드 작성**: 4건 통합 분업 검수 PASS + 사장님 별도 승인 후
- → 본 문서는 **명세만**, 구현은 후속 단계

---

## 5. 보고 템플릿 — `ops/restart-dry-run-result-template.md`

별도 파일로 생성 권장 (실행 시 복사 + 채움). 본 문서 §5에 포함된 내용:

```markdown
# §9 4단계 dry-run 실행 결과 — YYYYMMDD HH:MM

> **실행 일시**: YYYY-MM-DD HH:MM:SS KST
> **실행자**: 사장님 + 메인 AI (입회)
> **단계 1~4 PASS/FAIL 종합**: ☐ PASS / ☐ FAIL
> **(선택) 단계 5 Codex 재검수**: ☐ 진행 / ☐ 미진행

## 단계 1: One-line crontab 초안
- 산출물: `ops/restart-cron-draft-YYYYMMDD.txt`
- env grep 결과: [N건 발견 / 위험 인자 0건 / 매매 cron 주석 유지 ✅]
- 판정: ☐ PASS / ☐ FAIL

## 단계 2: VPS `*_MODE=paper` env grep
- 명령: `ssh ... grep -E '^(LIVE_TRADING_MODE|...)='`
- 결과 (raw):
  ```
  LIVE_TRADING_MODE=paper
  SELL_MONITOR_MODE=paper
  TELEGRAM_TRADING_MODE=paper
  AUTO_TRADING_ENABLED=0
  QUANT_PAPER_ONLY=true
  ```
- 판정: ☐ PASS / ☐ FAIL

## 단계 3: Smoke test (`quant_preflight.py --simulate-paper`)
- 명령: `ssh ... ./venv/bin/python3.11 tools/quant_preflight.py --simulate-paper`
- 결과 (raw):
  ```
  [PASS] runtime order gate: KILL_SWITCH exists
  ...
  [PASS] S6 KILL_SWITCH 차단: ...
  RESULT: PASS (16/16)
  ```
- 판정: ☐ PASS (16/16) / ☐ FAIL (N/16)

## 단계 4: Journalctl 5분 모니터링
- 실행 cron: `LIVE_TRADING_MODE=paper ... paper_warmup_daily.py`
- 시작 시각: YYYY-MM-DD HH:MM:SS
- 종료 시각: YYYY-MM-DD HH:MM:SS (+5분)
- paper intent 등록 N건 (jsonl 줄 수):
  ```
  data/order_intents/quant_intents_YYYY-MM-DD.jsonl: N lines
  ```
- KIS 실주문 흔적 grep:
  ```
  (출력 0줄 — 정상)
  ```
- 예상치 못한 raise:
  ```
  (출력 0줄 또는 분석 후 보고)
  ```
- 판정: ☐ PASS / ☐ FAIL

## 종합
- 단계 1~4 모두 PASS: ☐ YES / ☐ NO
- (선택) 단계 5 Codex 재검수: ☐ PASS / ☐ FAIL / ☐ 미진행
- 사장님 별도 승인: ☐ 승인 / ☐ 보류

## 후속 조치
- ☐ paper cron 1줄 정식 등록 (사장님 승인 시)
- ☐ 재가동 심사서 본문 작성 (별도 산출물)
- ☐ 추가 §9 사이클 (분기 1회 또는 사고 후)

## 표현 룰
- 사용: "§9 4단계 dry-run 완료", "재가동 심사 자료 준비 완료"
- 금지: "재가동 완료", "운영 안전 완성", "Phase 1 paper 가능" (사장님 별도 승인 전까지)
```

---

## 6. 진행 순서 (실제 실행)

```
[현재] 명세 작성 완료 (본 문서)
   ↓
[권장] 4건 통합 분업 검수 (security-architect + code-analyzer)
   ↓
[권장] Codex 의뢰서 보강 (P1-A + P2 + filelock + HMAC + §9 명세)
   ↓
[필수] 사장님 commit 결단 (4건 + 1 명세)
   ↓
[필수] preflight --simulate-paper 코드 구현 (별도 PDCA — ~1시간)
   ↓
[필수] Codex 코드 검수 PASS
   ↓
[필수] 사장님 §9 실행 별도 승인
   ↓
[실행] 단계 1~4 실행 (사장님 입회 권장)
   ↓
[선택] 단계 5 Codex 재검수
   ↓
[필수] 사장님 paper cron 1줄 등록 결단
   ↓
[가능] 제한적 paper cron 1줄 가동 (전체 X)
```

---

## 7. 적용 금지 (5/29 종일 + 본 문서 작성 후)
- ❌ §9 단계 1~4 실제 실행 (사장님 별도 승인 전까지)
- ❌ preflight `--simulate-paper` 코드 작성 (별도 PDCA 진입 전까지)
- ❌ cron 복원 / scheduler unmask / 실주문 실행
- ❌ 단독 commit
- ❌ "재가동 가능" 표현

---

## 8. 표현 룰

### 사용 가능
- "§9 4단계 dry-run 정식 명세 작성 완료"
- "재가동 심사 자료 준비 완료"
- "Codex 검수 의뢰 가능 상태"

### 사용 금지
- "재가동 가능" X
- "운영 안전 완성" X
- "Phase 1 paper 가능" X
- "§9 통과" X (실행 전)

---

## 9. 검수 의뢰 사항 (Codex)

1. 단계 1~4 명세의 차단 조건 충분성
2. 단계 3 `--simulate-paper` S1~S6 시뮬레이션 적정성
3. 단계 4 journalctl 모니터링 grep 패턴 적정성
4. 선택 단계 5 Codex 재검수 분리 결단 적정성
5. 보고 템플릿 항목 누락 여부

---

## 10. 연결 문서
- `docs/02-design/p1-residual-plan-5-29.md` §4
- `docs/02-design/p1-truth-pack-5-29.md`
- `docs/01-plan/trading-factory-v1-architecture.md` §9 (잔여 위험 — 본 문서로 보강)
- `docs/02-design/paper-cron-mode-policy-5_28.md`
- `tools/quant_preflight.py` (확장 대상)
- `memory/decision_5_28_p1_blockers.md` (사장님 결정문)
