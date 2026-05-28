# Phase 1 종합 보안 검수 — Trading Factory v1 + P0 봉합 (5/28)

> **작성일**: 2026-05-28
> **HEAD**: `db5d731`
> **분류**: 종합 보안 권장안 (코드 수정 없음, 설계 문서만)
> **검수 관점**: OWASP 2021 + KISA 금융보안 + Defense-in-Depth + 표현 룰(코덱스 5/28)
> **상태**: 메인 AI 단독 결정 금지. 코덱스 외부 검증 + 사용자 결단 후 적용.
> **전제 문서**:
> - `docs/02-design/security/trading-safety-architecture-review.md` (5/27 사고 후속)
> - `docs/02-design/security/phase1-security-review-5_28.md` (5/28 1차 P0 봉합 직후)
> **본 문서의 위치**: 위 두 문서 통합 + Note 4(매도 차단 텔레그램 실패 처리) + Note 5(C2 silent-return 폐지) + 표현 룰까지 반영한 **종합 보안 검수**.

---

## 0. 결론 (TL;DR)

### 0-1. Phase 1 봉합 완료 항목 (코드 기준 검증)

| ID | 항목 | 현재 상태 | 위치 |
|----|------|----------|------|
| P0-1 | `ORDER_INTENTS_GATE_DISABLED` 환경변수 우회 영구 제거 | 완료 (코드 레벨) | `src/use_cases/order_intents_gate.py:262` Note |
| P0-2 | `mode` 기본값 제거 (paper/live 명시 강제) | 완료 | `order_intents_gate.py:238~246`, paper/kis adapter |
| P0-3 | `executor_bot` 인자 + `intent.bot` 매치 검증 | 완료 | `order_intents_gate.py:271, 317` |
| P0-4 | `expires_at` timezone-aware 강제 | 완료 | `order_intents_gate.py:144~150` |
| P0-5 | HMAC-SHA256 서명 + `compare_digest` | 완료 | `order_intents_gate.py:94~112` |
| Note 1 | quant+live register 단계 차단 | 완료 | `order_intents_gate.py:405~409` |
| Note 4 | 텔레그램 ALERT 실패 silent pass 폐지 | 완료 | `chart_hero_executor.py:458~463, 509~514` |
| C1 | D0 picker intent + D+1 executor 페어 (직전 거래일 lookup) | 완료 | `order_intents_gate.py:169~215` |
| C2 | `_execute_sell` / `_execute_add_buy` silent-return 폐지 | 완료 | `chart_hero_executor.py:432~522` |
| Defense-in-Depth | quant+live 차단 — register + assert + adapter 3중 | 완료 | `kis_order_adapter.py:125~135` 포함 |

### 0-2. 잔존 위험 (P1~P3)

| 영역 | 잔존 위험 | 심각도 |
|------|---------|-------|
| HMAC 키 관리 거버넌스 부재 (회전·노출 대응) | High | **P1** |
| append-only filesystem (chattr +a) 미적용 | High | **P1** |
| 텔레그램 API_BASE 토큰을 URL에 포함 (로깅 노출 가능성) | Medium | **P1** |
| paper → live 슬리피지/세율 가정 보정 메커니즘 부재 | Medium | P2 |
| 감사 로그 5년 보존 정책 미정의 (KISA) | Medium | P2 |
| `.env` 단일 신뢰 — Secrets Manager 미적용 | High | P2 |
| FDS (이상 거래 탐지) 부재 | Medium | P3 |
| 2-key system / Heartbeat 자동 OFF | Medium | P3 |

### 0-3. 표현 룰 적용 결과 (코덱스 5/28 결정)

본 문서는 다음 금지 표현을 사용하지 않습니다:
- **"전체 주문 경로 L10 완료"** — Phase 1은 8개 진입점 중 일부만 마이그레이션됨. row 3~6 호출자 + smart_entry / adaptive_* / telegram 등 잔여 진입점 다수.
- **"Phase 1 paper 재가동 가능"** — 코드 PASS와 운영 PASS는 분리. paper dry-run + 코덱스 검수 + 사용자 결단 3단계 필요.
- **"live 안전망 완성"** — Note 4 적용 후에도 P1-P3 잔존 위험 다수. SPOF-5 (HMAC 키 자체) 신규 도입.

대신 사용하는 표현:
- "Phase 1 봉합 commit 검증 완료 — 운영 PASS는 별도"
- "10층 정의 완료 — 전체 호출자 마이그레이션은 row별 PR 진행 중"
- "코덱스 4차 검수 GO — 사용자 결단 + paper dry-run 필요"

---

## 1. 위험 매트릭스 (질문 영역별 — A~J)

| ID | 영역 | 잔존 위험 | 심각도 | 우선순위 |
|----|------|---------|--------|---------|
| R-A1 | HMAC 키 관리 | 키 회전 정책 부재 | High | **P1** |
| R-A2 | HMAC 키 관리 | 노출 시 대응 절차 미정의 (rotation playbook 부재) | High | **P1** |
| R-A3 | HMAC 키 관리 | `.env` 평문 저장 (Phase 0 그대로) | High | P2 |
| R-A4 | HMAC 키 관리 | 키 길이 32+ chars 검증만 — entropy 검증 부재 (예: "a" 32회 통과) | Medium | P2 |
| R-B1 | quant+live | register + assert + adapter 3중 차단 — 충분 | Low | (해결됨) |
| R-B2 | quant+live | 외부 jsonl 직접 쓰기 → HMAC 서명 없음 → assert에서 차단 | Low | (해결됨) |
| R-B3 | quant+live | 공격자가 HMAC 키 탈취 + day intent 위조 시 우회 가능 | Medium | P2 |
| R-C1 | intent 무결성 | append-only filesystem 미적용 → 삭제·rewrite 공격 가능 | High | **P1** |
| R-C2 | intent 무결성 | 파일 삭제 → 당일 매매 차단 (의도된 fail-closed), 단 알람 부족 | Low | (해결됨, 알람 권장) |
| R-C3 | intent 무결성 | replay attack — 어제 만료 intent 재사용 | Low | (해결됨 — expires_at + timezone-aware) |
| R-C4 | intent 무결성 | 파일시스템 오류 시 fallback 정책 부재 (현재 결과적 fail-closed) | Medium | P2 |
| R-C5 | intent 무결성 | file lock 부재 — register 동시 호출 race condition | Medium | **P1** |
| R-D1 | 호출자 전달 | 모든 P0 호출자(adaptive_*/smart_entry/telegram/auto_buy_executor)에 mode/executor_bot 명시 마이그레이션 미완 | High | **P1** |
| R-D2 | 호출자 전달 | backward compat (mode=None) 분기 유지 → 무방비 호출 가능 | High | **P1** |
| R-D3 | 호출자 전달 | 환경변수 default=live + quant+live 차단 = silent 차단 위험 (운영자 인지 부재) | Medium | P2 |
| R-E1 | silent-return 폐지 | _execute_sell C2 fix 완료 — 추가로 매도 intent 자동 발급 selector 강화 필요 | Low | (해결됨, 강화 권장) |
| R-E2 | silent-return 폐지 | telegram 실패 시 logger.warning 추가됨 — fallback ALERT (로컬 파일) 권장 | Low | P2 |
| R-F1 | backward compat 폐지 | `ORDER_INTENTS_GATE_STRICT` 환경변수 vs 코드 레벨 폐지 결단 미정 | Medium | P1~P2 |
| R-F2 | backward compat 폐지 | 폐지 시점 = 모든 P0 + P1 호출자 마이그레이션 완료 후 — 일정 미확정 | Medium | P2 |
| R-G1 | paper-first 한계 | 코스닥 세율 자동 분기 부재 (기본 KOSPI) | Low | P2 |
| R-G2 | paper-first 한계 | 슬리피지 가정의 실측 보정 메커니즘 부재 | Medium | P2 |
| R-G3 | paper-first 한계 | paper 검증된 시그널이 live에서도 동작한다는 가정 | Medium | P2 |
| R-H1 | KISA 금융보안 | 감사 로그 5년 보존 + 외부 백업 (S3) 부재 | Medium | P2 |
| R-H2 | KISA 금융보안 | 비정상 거래 패턴 탐지 (FDS) 부재 | Medium | P3 |
| R-H3 | KISA 금융보안 | `.env` 단일 신뢰 — 다중 인증 부재 | Medium | P3 |
| R-H4 | KISA 금융보안 | KIS 키 평문 저장 (`KIS_APP_KEY`/`KIS_APP_SECRET` `.env`) | High | P2 |
| R-I1 | 텔레그램 토큰 | 단타봇 측 httpx 토큰 노출 (5/28 발견) — 본 프로젝트 영향 없음 | Low | 모니터링 |
| R-I2 | 텔레그램 토큰 | 본 프로젝트 `API_BASE = f"...bot{TOKEN}"` 패턴 (logger.error 시 노출 가능) | Medium | **P1** |
| R-I3 | 텔레그램 토큰 | 향후 httpx/aiohttp 도입 시 디폴트 URL 로깅 위험 | Medium | P2 |
| R-J1 | 표현 룰 | 운영 PASS vs 코드 PASS 분리 — 표현 사용 모니터링 필요 | Low | 영구 |
| R-J2 | 표현 룰 | "전체 주문 경로 L10 완료" 등 금지 표현이 향후 보고서/커밋 메시지에 새지 않도록 거버넌스 | Low | 영구 |

---

## 2. 영역별 상세 분석

### A. HMAC 서명 메커니즘

#### A-1. 현재 구현 (`src/use_cases/order_intents_gate.py:75~112`)

```python
def _get_hmac_key() -> bytes:
    key_str = os.getenv("ORDER_INTENTS_HMAC_KEY", "")
    if not key_str:
        raise IntentSignatureError(...)
    if len(key_str) < 32:
        raise IntentSignatureError(...)
    return key_str.encode("utf-8")

def _compute_signature(intent: dict) -> str:
    payload = {k: v for k, v in intent.items() if k not in HMAC_EXCLUDED_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    sig = hmac.new(key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return sig

def _verify_signature(intent: dict) -> bool:
    return hmac.compare_digest(stored_sig, expected_sig)
```

검증 OK 항목:
- **알고리즘**: HMAC-SHA256 (충분, 2026 기준 정설)
- **Deterministic JSON**: `sort_keys=True, separators=(",", ":")` → 정합성 OK (공백/순서 차이 무영향)
- **Timing attack 방어**: `hmac.compare_digest` 사용
- **키 길이**: 64-char hex = 256-bit entropy (충분)
- **서명 필드 제외**: `HMAC_EXCLUDED_FIELDS = ("hmac_signature",)` — 자기 참조 회피 OK

#### A-2. 잔존 위험

**R-A1 키 회전 정책 부재 (P1)**
- 키가 한 번도 회전되지 않음. 장기간 유출 시 무한 위조 가능.
- intent에 `key_id` 필드 없음 → 키 회전 시 기존 intent 전체 일괄 invalidate 필요.

**R-A2 노출 대응 절차 부재 (P1)**
- 키 노출 발견 시 playbook 미정의. 새 키 발급 → 기존 intent 무효화 → 재서명 절차 필요.

**R-A3 `.env` 평문 저장 (P2)**
- 백업 파일 (`*.bak.20260514_1622` 등) 에도 같은 키 노출 가능성.
- `git secrets` 또는 `pre-commit` hook으로 commit 차단 부재.

**R-A4 entropy 검증 부재 (P2)**
- 32+ chars 길이만 검증 → `"a" * 32` 같은 약한 키 통과 가능.
- 권장: `_get_hmac_key()`에 Shannon entropy 검증 추가 (>= 4.0 bits/char).

#### A-3. 권장 (P1~P2)

```
P1:
1. key_id 필드 도입:
   intent["key_id"] = "v1"
   환경변수: ORDER_INTENTS_HMAC_KEY_V1, V2 동시 보유 (1주 grace)
   _verify_signature가 key_id 기반 키 조회

2. 노출 대응 playbook (docs/security/hmac-key-rotation.md):
   STEP 1. KILL_SWITCH 즉시 활성화 (모든 매매 정지)
   STEP 2. 새 키 발급 (openssl rand -hex 32)
   STEP 3. .env 양쪽 (로컬 + VPS) 갱신
   STEP 4. 기존 intent 전체 invalidate (key_id 매칭 X → 거부)
   STEP 5. selector 재실행으로 새 intent 발급
   STEP 6. KILL_SWITCH 해제 + 텔레그램 알람

3. entropy 검증:
   def _get_hmac_key() -> bytes:
       key_str = os.getenv("ORDER_INTENTS_HMAC_KEY", "")
       ...
       if _shannon_entropy(key_str) < 4.0:
           raise IntentSignatureError("[HMAC] entropy 부족 (예측 가능 키 의심)")
       return key_str.encode("utf-8")

P2:
4. `.env` → AWS Secrets Manager 또는 OS keyring 이전.
5. `git secrets` pre-commit hook으로 키 평문 commit 차단.
```

---

### B. quant+live 영구 차단의 안전성

#### B-1. 현재 구현 (3중 방어)

**Layer 1 (register_intent)** `order_intents_gate.py:405~409`:
```python
if bot == "quant" and str(intent.get("mode", "")).lower() == "live":
    raise OrderIntentError("[REGISTER] quant bot은 live mode intent 등록 금지 — ...")
```

**Layer 2 (assert_order_intent_exists)** `order_intents_gate.py:284~289`:
```python
if executor_bot == "quant" and mode == "live":
    raise OrderIntentError("[INPUT] quant executor + live mode 조합 금지 — ...")
```

**Layer 3 (KisOrderAdapter._guard)** `kis_order_adapter.py:125~135`:
```python
if mode is not None or executor_bot is not None:
    if mode != "live":
        raise ValueError("KisOrderAdapter는 mode='live'만 허용 ...")
```

#### B-2. 우회 시나리오 분석

| 시나리오 | 차단되는가 | 근거 |
|---------|----------|------|
| 정상 호출 `register_intent({..., "bot":"quant", "mode":"live"})` | O | Layer 1 |
| 외부에서 `quant_intents_*.jsonl` 에 mode="live" 직접 쓰기 (HMAC 없음) | O | Layer 2 — `_verify_signature` False → `IntentSignatureError` |
| 외부에서 mode="live" + 위조 hmac_signature | O | Layer 2 — HMAC 키 모르면 검증 실패 |
| HMAC 키 탈취 + mode="live" intent 위조 + executor_bot="quant" | O | Layer 3 — adapter 진입 시 차단 |
| HMAC 키 탈취 + `day_intents_*.jsonl`에 mode="live"로 직접 등록 (executor_bot="day") | **△** | day 화이트리스트 (`AUTO_TRADING_WHITELIST_ONLY`) 미활성 시 우회 가능 |

#### B-3. 잔존 위험 + 권장

**R-B3 (Medium, P2)**: HMAC 키 탈취 + day intent 위조 시 차단 메커니즘 부재.

권장:
```
P2:
1. intent에 selector_signature 추가:
   - selector 코드의 git SHA + 실행 PID + timestamp 기반 이중 서명
   - day bot이 외부 호출되어도 selector 검증 통과 불가

2. day bot 화이트리스트 강제 (운영 디폴트 ON):
   - .env: AUTO_TRADING_WHITELIST_ONLY=1
   - kis_order_adapter._guard Layer 6 (line 158~164) 이미 존재
   - day bot은 ETF 화이트리스트 26개만 허용

3. live mode 종목별 일일 한도 (KISA 이중 통제):
   - 30만원 초과 시 텔레그램 응답 필요 (사용자 승인)
   - 현재 자동 통과 → 권한 분리 권장
```

---

### C. order_intents 파일 무결성

#### C-1. 현재 보호 메커니즘

- HMAC-SHA256 서명 (위조 방지) — OK
- append-only mode 파일 쓰기 (`f.write` mode "a") — OK (코드 레벨)
- expires_at 기반 만료 (replay 차단) — OK
- timezone-aware 강제 (P0-4) — OK
- D0 + D+1 양쪽 파일 로드 (C1 fix) — OK

#### C-2. 잔존 위험

**R-C1 (High, P1) — Append-only filesystem 미적용**

현재 `.jsonl` 파일은 **일반 ext4/NTFS 파일**. 다음 공격 가능:

| 공격 | 시나리오 | 차단되는가 |
|------|---------|----------|
| **삭제** | `rm data/order_intents/day_intents_20260528.jsonl` | △ 의도된 동작 (당일 매매 차단), 단 알람 없음 |
| **재기록 (truncate)** | 파일 비우고 위조 intent 작성 | △ HMAC 차단되나 감사 로그 손실 |
| **선택적 라인 제거** | sed로 특정 intent 삭제 | O (HMAC) 단 삭제 사실 미탐지 |
| **시간 backdate** | OS 파일 mtime 변조 | O (expires_at은 파일 내부) |
| **race condition** | register와 동시에 외부 write | 가능성 있음 (file lock 없음) |

**R-C5 (Medium, P1) — file lock 부재**

`register_intent` 라인 `out_path.open("a", ...)` → 동시 쓰기 시 잘림/덮어쓰기 가능성.

#### C-3. 권장 (P1)

```
P1:
1. append-only filesystem (Linux):
   sudo chattr +a data/order_intents/
   - 삭제 불가, 수정 불가, append만 가능
   - root만 chattr -a 가능
   VPS 배포 시 ansible/systemd ExecStartPre로 등록

2. 파일 lock (fcntl.flock):
   def register_intent(...):
       ...
       with out_path.open("a", encoding="utf-8") as f:
           fcntl.flock(f, fcntl.LOCK_EX)
           try:
               f.write(json.dumps(intent, ensure_ascii=False) + "\n")
           finally:
               fcntl.flock(f, fcntl.LOCK_UN)
   - Windows는 msvcrt.locking 사용 (개발 환경)

P2:
3. 파일 해시 체인 (Merkle-like):
   - 각 line에 prev_line_hash 필드 추가
   - 중간 라인 삭제 시 chain 깨짐 → 탐지 가능
   - 매일 17:00 검증 cron

4. 외부 백업 (KISA 5년 보존):
   data/audit/order_intents/{YYYY-MM-DD}.jsonl.gz → S3 또는 별도 VPS
   - 매일 16:30 cron으로 압축 + rclone copy

5. 파일시스템 오류 fallback (R-C4):
   if not ORDER_INTENTS_DIR.exists():
       # 현재: 빈 리스트 반환 → 결과적 fail-closed
       # 보강: 즉시 KILL_SWITCH 활성화 + 텔레그램 ALERT
```

---

### D. P0 호출자 전달 안전성

#### D-1. 현재 마이그레이션 진행 상태

`Grep order_intents_gate src/` 결과 → mode/executor_bot 명시 호출 모듈:

| 모듈 | 마이그레이션 | 비고 |
|-----|------------|------|
| `src/strategies/chart_hero_executor.py` | 완료 (`mode="paper" if self.paper else "live"`, `executor_bot="quant"`) | C2 fix 5/28 |
| `src/use_cases/adaptive_stop_loss.py` | 진행 중 (Grep hit, 명시 호출 확인 필요) | row 4 |
| `src/use_cases/adaptive_quick_profit.py` | 진행 중 | row 4 |
| `src/use_cases/adaptive_reentry.py` | 진행 중 | row 4 |
| `scripts/owner_rule_monitor.py` | **★ 마이그레이션 필요** (5/27 사고 원인지) | row 1 |
| `scripts/auto_buy_executor.py` | 마이그레이션 필요 | row 5 |
| `src/strategies/smart_entry/` | 마이그레이션 필요 | row 5 |
| `src/use_cases/adaptive_*` 기타 | 검수 필요 | row 4 |
| 텔레그램 매매 핸들러 | 마이그레이션 필요 | row 6 |

#### D-2. 잔존 위험

**R-D1 (High, P1) — 호출자 마이그레이션 미완**
- mode + executor_bot 둘 다 명시한 호출만 L10 통과 (assert_order_intent_exists 강제).
- 기존 호출 (둘 다 None)은 backward compat로 그대로 통과 → **L10 우회**.
- 즉 현재 시점에 "전체 주문 경로 L10 완료"는 사실과 다름 → 표현 룰 적용 (J).

**R-D2 (High, P1) — backward compat 분기 유지**
- `kis_order_adapter.py:125~147` `if mode is not None or executor_bot is not None:` 분기로 기존 호출 유지.
- 점진 마이그레이션 정책상 필요하나, 폐지 시점 결단 필요.

**R-D3 (Medium, P2) — silent 차단 위험**
- quant+live 호출이 register 단계에서 차단되면 운영자가 인지 못 할 수 있음.
- 권장: `OrderIntentError` 발생 시 텔레그램 ALERT 의무.

#### D-3. 권장 (P1)

```
P1:
1. 호출자 마이그레이션 일정 (PR 분할):
   PR-A (1주 내): owner_rule_monitor — 5/27 사고 직접 원인지, 최우선
   PR-B (1주 내): auto_buy_executor + smart_entry — 매수 경로
   PR-C (1~2주): adaptive_stop_loss / quick_profit / reentry — 매도 경로
   PR-D (1~2주): 텔레그램 매매 핸들러 — 수동 매매 진입점

2. 각 PR 체크리스트:
   [ ] mode + executor_bot 둘 다 명시 호출 (둘 다 None 호출 금지)
   [ ] tests/test_adapter_intents_integration.py PASS
   [ ] paper dry-run 1일 → intent 정합성 100%
   [ ] 코덱스 검수 GO
   [ ] 사용자 결단 확인

3. 마지막 PR 후 backward compat 폐지:
   if mode is None or executor_bot is None:
       raise ValueError("mode + executor_bot 필수 (backward compat 폐지 — YYYY-MM-DD)")
```

---

### E. silent-return 폐지 검증 (chart_hero_executor C2)

#### E-1. 현재 구현 (`chart_hero_executor.py:432~522`)

`_execute_add_buy` (line 432~469):
- 주문 예외 → `err_type` 분류 (NoIntentError/IntentError) → `action` 태그 결정
- `logger.error` 기록 (silent X)
- 텔레그램 send_message 시도
- **Note 4 (5/28 16:00)**: 텔레그램 실패 시 `logger.warning` 추가 ("발송 실패 (차단은 정상 수행)")
- 결과 dict 반환 → caller의 `results.append`로 추적

`_execute_sell` (line 471~522):
- 주문 예외 → `err_type` 분류
- `logger.error` 기록 (silent X)
- 텔레그램 ALERT 전송 (평단/손익 추정 포함)
- **Note 4**: 텔레그램 실패 시 `logger.warning` ("매도 차단 자체는 정상, 운영자 수동 확인 필요")
- 결과 dict 반환 (block_reason 포함)

#### E-2. 트레이드오프 검증 (5/27 사고 학습 반영)

| 시나리오 | 매도 silent-return의 결과 (이전) | C2 fix 후 |
|---------|--------------------------------|----------|
| 매도 intent 누락 (정상 시장) | 매도 안 됨 → 보유 지속 → 손실 확대 가능 | logger.error + 텔레그램 ALERT + results 기록 |
| 매도 intent 누락 (급락장) | **손절 못 함 → 큰 손실 위험** | 동일 알림 → 사용자 즉시 수동 매도 가능 |
| 추매 intent 누락 (수익 종목) | 추가 수익 기회 손실 | logger.error + 텔레그램 알림 + 안전 측 (포지션 유지) |

#### E-3. 잔존 위험

**R-E2 (Low, P2) — fallback ALERT 부재**
- 텔레그램 실패 시 logger.warning만 기록 → journalctl 확인 부재 시 누락 가능.
- 권장: `logs/emergency_alerts.log` 파일에 추가 기록 (이미 `telegram_sender.py:31`에 경로 정의됨).

권장:
```python
except Exception as tg_err:
    logger.warning("[%s] 텔레그램 ALERT 실패 ...", action_tag, d["ticker"], tg_err)
    # 추가: 로컬 fallback 파일
    try:
        ALERT_FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ALERT_FALLBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} [{action_tag}] {d['ticker']} {e}\n")
    except Exception:
        pass  # 최후 fallback도 실패 시 logger.warning에 의존
```

#### E-4. 추가 권장 — 매도 intent 자동 발급

`_execute_sell`이 NoIntentError로 차단되는 경우 매도 못 함 → 손실 위험.

```
P2:
1. selector 강화 (D0 16:30):
   - 보유 종목 매일 09:00 자동 매도 intent 발급 (D+5 만료)
   - STOPLOSS 트리거 시 즉시 발급 후 sell_limit 호출
   - 결과: SELL_BLOCKED_NO_INTENT 사례 0건 보장
```

---

### F. backward compat 폐지 시점

#### F-1. 현재 상태

`kis_order_adapter.py:125~147` + `paper_order_adapter.py:168~184, 221~235`:
- mode + executor_bot 둘 다 None → 기존 9중 가드만 적용 (L10 우회).
- 둘 중 하나만 명시 → ValueError.
- 둘 다 명시 → assert_order_intent_exists 강제.

#### F-2. 권장 폐지 절차

```
방식 비교:

A. ORDER_INTENTS_GATE_STRICT 환경변수 단계 도입
   - .env: ORDER_INTENTS_GATE_STRICT=1 시 mode=None 호출도 차단
   - 장점: 점진 적용 가능, 롤백 쉬움
   - 단점: 환경변수 누락 시 우회 가능 (P0-1과 동일 문제)

B. 코드 레벨 폐지 (deadline 명시)
   if mode is None or executor_bot is None:
       raise ValueError(f"mode + executor_bot 필수 (backward compat 폐지: {DEADLINE})")
   - 장점: 우회 불가
   - 단점: 마이그레이션 완료까지 활성화 불가

권장 (P1~P2):
- 단계 1 (P1, 1주 내): A 방식 도입 + .env 디폴트 OFF
  마이그레이션 진행 호출자 ONE-BY-ONE 검증
- 단계 2 (P2, 1개월 내): A 방식 .env 디폴트 ON
- 단계 3 (P2 말, 1.5개월): B 방식 코드 레벨 폐지
  - PR-D 완료 후 deadline=2026-07-15 명시
```

---

### G. paper-first 정책의 한계

#### G-1. 현재 가정값 (`paper_order_adapter.py:31~35`)

```python
DEFAULT_SLIPPAGE_BASE_PCT = 0.05
DEFAULT_SLIPPAGE_FALLBACK_PCT = 0.10
FEE_PCT = 0.015
TAX_PCT_KOSPI = 0.18
TAX_PCT_KOSDAQ = 0.20
```

호출 시 `market="KOSPI"` 디폴트 → 코스닥 종목도 코스피 세율 적용 위험.

#### G-2. 권장 (P2)

```
P2:
1. 종목코드 → 시장 자동 분기:
   from src.utils.ticker_market import get_market  # 기존 확인 필요
   def sell_limit(self, ticker, ...):
       market = get_market(ticker)  # "KOSPI" / "KOSDAQ"
       ...
   - 디폴트 KOSPI 폐지

2. 슬리피지 실측 보정 (5/21+ 백테스트 결과 반영):
   PAPER_MIRROR_SLIPPAGE_OBSERVED_PCT 환경변수 + 자동 학습:
   - 매주 일요일 paper vs live (동일 시그널) 비교 → 가정값 자동 조정
   - 슬리피지 가정이 실측보다 낙관적 → live 손실 위험

3. paper → live 승급 가드:
   - paper 백테스트 D+1 적중률 ≥50%
   - 추가: paper 슬리피지 가정의 ±20% 이내 실측 일치 시만 승급
   - paper 검증된 시그널이 live에서도 동작한다는 가정 검증 메커니즘
```

---

### H. KISA / 금융보안 관점

#### H-1. 거래 감사 로그 (KISA 5년 의무)

현재:
- `data/order_intents/*.jsonl` 영구 보관됨 — OK (append-only로 강화 시)
- `logs/emergency_alerts.log` — 폴백 알람만
- `journalctl -u quantum-scheduler` — VPS 시스템 로그 (rotation 정책 확인 필요)

권장 (P2):
```
data/audit/
├── orders/
│   └── {YYYY}/{MM}/{DD}_orders.jsonl.gz  # 실주문 5년 보존
├── intents/
│   └── {YYYY}/{MM}/{DD}_intents.jsonl.gz
└── decisions/
    └── {YYYY}/{MM}/{DD}_decisions.jsonl.gz  # 판단 근거

매일 16:30 cron:
1. data/order_intents/*.jsonl → gzip 압축
2. data/audit/intents/{YYYY}/{MM}/ 이동
3. rclone copy → S3 (또는 별도 VPS)
4. 5년 (1825일) 후 자동 삭제 (KISA)
```

#### H-2. 비정상 거래 패턴 탐지 (FDS)

권장 (P3):
```
src/utils/anomaly_detector.py 신설:

1. 시간대 이상:
   - 09:55:11 같은 비표준 분초 매도 → 알람 (5/27 사고 패턴)
   - 09:01 직후 또는 15:25 직전 매매 비율 비정상

2. 종목 이상:
   - 화이트리스트 외 종목 시도 (이미 차단)
   - 짧은 시간 내 같은 종목 반복 매매

3. 금액 이상:
   - 일일 누적 한도의 90% 도달 시 사용자 응답 요청
   - 단일 종목 손실 -3% 도달 시 알람

4. 빈도 이상:
   - 시간당 5건 초과 → 자동 KILL_SWITCH
```

#### H-3. 다중 인증

현재 `.env` 단일 신뢰 (KIS 키 + HMAC 키 + 텔레그램 토큰).

권장 (P3):
```
1. 2-key system (핵 발사 모델):
   - 활성화: 사용자 텔레그램 응답 + 코덱스 검수 PASS 토큰 (양쪽 필요)
   - .env에 ACTIVATION_TOKEN_USER + ACTIVATION_TOKEN_CODEX

2. Heartbeat 기반 자동 OFF:
   - 사용자가 6시간마다 텔레그램으로 "/heartbeat" 명령
   - 미수신 시 자동 KILL_SWITCH

3. 단계적 활성화:
   - 매일 09:00 자동 OFF (default)
   - 사용자가 "/activate today" → 당일만 ON
   - 다음날 자동 OFF (실수 방지)
```

#### H-4. KIS 키 평문 저장

```python
# kis_order_adapter.py:73~78
self.broker = mojito.KoreaInvestment(
    api_key=os.getenv("KIS_APP_KEY"),
    api_secret=os.getenv("KIS_APP_SECRET"),
    acc_no=os.getenv("KIS_ACC_NO"),
    mock=is_mock,
)
```

**R-H4 (High, P2)**: `.env` 평문 + `.env.bak.*` 백업 파일 동시 노출.

권장:
```
P2:
1. AWS Secrets Manager 또는 OS keyring:
   from src.utils.secrets_manager import get_secret
   api_key = get_secret("KIS_APP_KEY")
   - 로컬: keyring 라이브러리 (Windows Credential Manager)
   - VPS: AWS Secrets Manager 또는 systemd LoadCredential

2. .env.bak.* 정리:
   git rm .env.bak.20260514_1622 (이미 .gitignore 되어 있는지 확인)
```

---

### I. 텔레그램 토큰 노출 (5/28 발견)

#### I-1. 본 프로젝트 자체 검증 (실측)

`src/telegram_sender.py:14, 33`:
```python
import requests
...
API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
```

**현재 상태**:
- `requests` 라이브러리 사용 — 기본적으로 URL 전체 로깅 안 함 (logger 설정 의존)
- 그러나 **API_BASE 자체에 토큰 포함** → `logger.error("API call failed: %s", url)` 같은 패턴 발견 시 토큰 노출
- `logging.DEBUG` 활성화 시 urllib3가 전체 URL 로깅 가능성

#### I-2. 잔존 위험

**R-I2 (Medium → P1)** — `API_BASE`에 토큰 포함된 URL 패턴은 향후 디버깅·예외 처리 코드에서 자동 노출 위험.

권장 (P1):
```python
# 현재
API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# 권장
_API_HOST = "https://api.telegram.org"

def _api_url(method: str) -> str:
    return f"{_API_HOST}/bot{TELEGRAM_BOT_TOKEN}/{method}"

def _masked_url(method: str) -> str:
    if not TELEGRAM_BOT_TOKEN:
        return f"{_API_HOST}/bot<UNSET>/{method}"
    return f"{_API_HOST}/bot{TELEGRAM_BOT_TOKEN[:4]}...{TELEGRAM_BOT_TOKEN[-4:]}/{method}"

# 모든 logger.error/warning에서 _masked_url() 사용
```

**R-I3 (Medium, P2)** — 향후 httpx/aiohttp 도입 시:
```
- httpx는 디폴트로 URL 로깅 (logger="httpx")
- 적용 시 logging.getLogger("httpx").setLevel(WARNING) 강제
- 또는 httpx.Client(event_hooks=...) 로 URL sanitization hook
```

---

### J. 표현 룰 (코덱스 5/28 결정)

#### J-1. 금지 표현 + 권장 표현

| 금지 | 권장 | 근거 |
|------|------|------|
| "전체 주문 경로 L10 완료" | "10층 정의 완료 — row별 마이그레이션 진행 중" | row 3~6 + smart_entry/adaptive_*/telegram 호출자 미완 (D-1 참조) |
| "Phase 1 paper 재가동 가능" | "코덱스 검수 GO — 사용자 결단 + paper dry-run 필요" | 코드 PASS와 운영 PASS 분리 |
| "live 안전망 완성" | "Phase 1 봉합 commit 검증 완료 — P1~P3 잔존 위험 다수" | SPOF-5 신규 도입 + 호출자 마이그레이션 미완 |
| "안전" | "현재 식별된 위험은 ... 잔존 위험은 ..." | 단정 회피, 위험 명시 |
| "재가동 OK" | "재가동 조건: 1) ... 2) ... 3) ..." | 조건 명시 |

#### J-2. 향후 거버넌스

```
P1 (영구):
1. 보고서/커밋 메시지 표현 룰 추가:
   - "완료" 단정 → "검증 완료 / 운영 PASS는 별도" 분리
   - "안전" → "위험 식별 / 잔존 위험 명시"
2. 코덱스 검수 시 표현 룰 위반 자동 flag
3. CLAUDE.md 또는 AGENTS.md에 표현 룰 영구 등록
```

---

## 3. SPOF 재평가 — Phase 1 적용 후

`trading-safety-architecture-review.md §3`의 SPOF에 Phase 1 효과 반영:

| SPOF | Phase 0 (5/27 사고 직전) | Phase 1 (5/28 적용 후) |
|------|------------------------|---------------------|
| **SPOF-1: mojito raw 객체 노출** | Critical (owner_rule_monitor 우회) | △ Medium — adapter 진입 시 mode/executor_bot 강제, 단 mojito broker 객체 자체는 여전히 public, **owner_rule_monitor 마이그레이션 미완** |
| **SPOF-2: `.env` 평문 환경변수** | High | High (변화 없음) — P2-6 필요 |
| **SPOF-3: KILL_SWITCH 파일 기반** | Medium-High | Medium-High (변화 없음) — order_intents가 보조하나 KILL_SWITCH 자체는 동일 |
| **SPOF-4: 단일 자격증명** | Critical | High — HMAC 키 추가로 다중화 일부 진전, 단 둘 다 `.env` 평문 |
| **★ SPOF-5 (신규): HMAC 키 자체** | N/A | **High** — 단일 키 + 회전 정책 부재 |
| **★ SPOF-6 (신규): 호출자 마이그레이션 미완** | N/A | **High** — backward compat 분기로 L10 우회 가능 |

**핵심 평가**:
- SPOF-1 본질적 개선 (가장 위험했던 SPOF)
- SPOF-5, SPOF-6 신규 도입 — Phase 1 효과 > 신규 위험이지만 P1 우선 해소 필요
- "live 안전망 완성" 표현은 정확하지 않음 → 표현 룰 (J) 적용

---

## 4. 우선순위 매트릭스 (P0~P3)

### P0 — 즉시 (24시간 내, 5/29까지)

| ID | 항목 | 근거 |
|----|------|------|
| P0-A | **현재 Phase 1 (commit `db5d731` 5/28)** 적용 완료 검증 | 코덱스 4차 PASS 확인 (Note 4 포함) |
| P0-B | `.env ORDER_INTENTS_HMAC_KEY` 로컬 + VPS 양쪽 설정 + 동일성 검증 | 양쪽 불일치 시 모든 매매 차단 |
| P0-C | 5/20~5/27 기간 quant_intents 파일 전체 감리 (HMAC 검증 + 위조 흔적 확인) | 사고 발생 가능 기간 추적 |
| P0-D | `tests/test_order_intents_gate.py` + `tests/test_adapter_intents_integration.py` 회귀 PASS 확인 | C2 fix + Note 4 후 재검증 |

### P1 — 1주 내 (6/4까지)

| ID | 항목 | 영역 |
|----|------|------|
| P1-1 | **append-only filesystem 적용** (`chattr +a data/order_intents/`) — VPS 의무 | C |
| P1-2 | **owner_rule_monitor mode/executor_bot 명시 마이그레이션** — 5/27 사고 직접 원인지 | D |
| P1-3 | **auto_buy_executor + smart_entry 마이그레이션** | D |
| P1-4 | **HMAC 키 회전 playbook 작성** + key_id 필드 도입 | A |
| P1-5 | **telegram API_BASE 토큰 마스킹** (`_masked_url`) | I |
| P1-6 | **파일 lock (fcntl.flock)** register_intent에 적용 | C |
| P1-7 | **ORDER_INTENTS_GATE_STRICT 환경변수 도입** (단계 1) | F |
| P1-8 | **logs/emergency_alerts.log fallback** _execute_sell에 추가 | E |

### P2 — 1개월 내 (6월 말)

| ID | 항목 | 영역 |
|----|------|------|
| P2-1 | adaptive_stop_loss/quick_profit/reentry 마이그레이션 | D |
| P2-2 | 텔레그램 매매 핸들러 마이그레이션 (row 6) | D |
| P2-3 | 종목코드 → 시장 자동 분기 (KOSPI/KOSDAQ 세율) | G |
| P2-4 | 슬리피지 실측 보정 메커니즘 (paper vs live 자동 비교) | G |
| P2-5 | 감사 로그 5년 보존 정책 + 외부 백업 (S3) | H |
| P2-6 | selector_signature 이중 서명 (HMAC 키 탈취 방어) | B |
| P2-7 | day bot 종목 화이트리스트 강제 (`AUTO_TRADING_WHITELIST_ONLY=1`) | B |
| P2-8 | `.env` → AWS Secrets Manager (KIS 키 + HMAC 키) | A, H |
| P2-9 | 파일시스템 오류 시 즉시 KILL_SWITCH 활성화 fallback | C |
| P2-10 | 파일 해시 체인 (Merkle-like) + 매일 검증 | C |
| P2-11 | backward compat 폐지 단계 2~3 (.env 디폴트 ON → 코드 폐지) | F |
| P2-12 | HMAC entropy 검증 (Shannon ≥ 4.0) | A |

### P3 — Defense-in-Depth (3개월+)

| ID | 항목 | 영역 |
|----|------|------|
| P3-1 | FDS (이상 거래 탐지) 룰셋 — 시간/종목/금액/빈도 | H |
| P3-2 | 2-key system (사용자 + 코덱스 양쪽 활성화 토큰) | H |
| P3-3 | Heartbeat 기반 자동 OFF (6시간 미수신 시) | H |
| P3-4 | 단계적 활성화 (매일 09:00 자동 OFF + `/activate today`) | H |

---

## 5. row별 마이그레이션 안전성 (사용자 결단 대기)

각 row PR은 코덱스 검수 통과 후 commit. 사용자 결단 + paper dry-run 1일 필수.

| row | 대상 | 위험도 | 우선 |
|-----|------|------|------|
| row 1 | `scripts/owner_rule_monitor.py` (5/27 사고 직접 원인지) | High (mojito raw 호출 직접 경로) | **P1-2** |
| row 2 | `scripts/auto_buy_executor.py` | High (매수 진입점) | **P1-3** |
| row 3 | `register_intent` 신규 호출처 (selector 코드들) | Low (가장 안전) | P1 |
| row 4 | `paper_order_adapter`/`kis_order_adapter` 호출처 (adaptive_*) | Medium | P2 |
| row 5 | `ChartHeroExecutor._execute_add_buy` (silent-return) | Low (정책 결정, C2 완료) | (해결됨) |
| row 6 | `ChartHeroExecutor._execute_sell` (silent-return) | Low (C2 완료) | (해결됨) |
| row 7 | 텔레그램 매매 핸들러 | Medium (수동 매매 진입점) | P2 |

각 PR 체크리스트:
```
[ ] mode + executor_bot 둘 다 명시 호출
[ ] tests/test_adapter_intents_integration.py PASS
[ ] tests/test_order_intents_gate.py PASS
[ ] paper dry-run 1일 → intent 정합성 100%
[ ] 코덱스 검수 GO
[ ] 사용자 결단 확인
```

---

## 6. 메인 AI 자가 평가 (코덱스 검수 대상)

본 문서의 한계 자인 (`feedback_code_review_problem` + `feedback_codex_collaboration` 준수):

1. **실제 공격 시도 미수행**: append-only 우회·HMAC 키 탈취 시나리오는 이론만, PoC 미작성 → 코덱스에 실제 시도 요청
2. **mojito 라이브러리 내부 미실측**: `broker.create_market_sell_order` 예외 처리/재시도 정책 미검증
3. **KISA 전자금융감독규정 원문 미확인**: 5년 보존·이중 통제 의무는 일반 권고 수준, 본 시스템 적용 범위 (개인 vs 사업자) 확인 필요
4. **HMAC 키 회전 시 grace period 1주는 추정값**: 실제 intent 발급 빈도 기준 재산정 필요
5. **silent-return 폐지의 부작용 검증 부족**: 매도 차단이 안전한 케이스 (시장 비정상) 가 있을 수 있음 — 트레이드오프 백테스트 필요
6. **호출자 마이그레이션 진행도 정확 수치 미산정**: Grep만으로 추론, 실제 호출 라인 카운트 + 회귀 테스트 적용 부재
7. **표현 룰 위반 자동 탐지 부재**: 향후 보고서에 "완료" 단정이 새는지 감시 메커니즘 없음

본 문서는 메인 AI가 단독으로 결단할 사항이 아닙니다. **코덱스 외부 검증 + 사용자 결정** 후 적용해야 합니다.

---

## 7. 핵심 질문 답변 요약

| 질문 (A~J) | 답변 |
|------|------|
| A. HMAC 64-char hex 적정성? | OK (256-bit entropy). 단 회전 정책 + key_id 필드 + entropy 검증 필수 (P1-4) |
| A. 노출 시 대응 절차? | playbook 부재 → P1-4 작성 |
| A. ORDER_INTENTS_HMAC_KEY 32+ chars 검증? | 길이 검증 OK. entropy 검증 부재 (R-A4, P2-12) |
| B. quant+live 영구 차단 충분성? | 정상 경로 + 외부 jsonl 직접 쓰기 + HMAC 위조 → 모두 차단. HMAC 키 탈취 + day intent 위조 시나리오만 잔존 (P2-6) |
| C. HMAC만으로 충분? | 불충분. append-only filesystem 필요 (P1-1) — 삭제·rewrite 공격 일부 노출 |
| C. FS 오류 시 fallback? | 현재 결과적 fail-closed (NoIntentError), 알람 부재 → P2-9 |
| D. 모든 P0 호출자 마이그레이션? | **미완**. owner_rule_monitor (사고 원인지) + auto_buy_executor + smart_entry + adaptive_* + telegram 잔여 (P1-2~3, P2-1~2) |
| D. silent 차단 운영자 인지? | OrderIntentError 발생 시 텔레그램 ALERT 의무화 필요 (P2-3) |
| E. silent-return 폐지 완료? | C2 fix (5/28) 완료. Note 4로 텔레그램 실패 logger.warning 추가됨. fallback 파일 권장 (P1-8) |
| F. backward compat 폐지 시점? | 단계 1 (P1-7, .env STRICT 도입) → 단계 2~3 (P2-11) |
| G. paper-first 한계? | 코스닥 세율 자동 분기 부재 (P2-3) + 슬리피지 실측 보정 부재 (P2-4) |
| H. KISA 5년 보존 의무 충족? | 부분 충족 (`data/order_intents/` 영구). 압축·외부 백업 추가 필요 (P2-5) |
| H. KIS 키 평문 저장? | High 위험 (R-H4) → Secrets Manager (P2-8) |
| I. 텔레그램 토큰 현재 노출 위험? | 본 프로젝트 — requests 디폴트 로깅 안 됨, 단 API_BASE 패턴 위험 (P1-5 마스킹) |
| J. 표현 룰 자체 점검? | 본 문서 J-1 적용 — "완료/안전" 단정 회피, 운영 PASS vs 코드 PASS 분리 |

---

## 8. 다음 액션 (사용자 결정 대기)

1. **본 문서 코덱스 검수 의뢰** (`ops/codex_inbox/`)
2. **P0-B, P0-C, P0-D 즉시 실행** (HMAC 키 동기화 + 5/20~5/27 intent 감리 + 회귀 PASS 재확인)
3. **P1-2 (owner_rule_monitor 마이그레이션) 6/4까지 적용** — 5/27 사고 직접 원인지, 최우선
4. **P1-5 (telegram 토큰 마스킹) 6/4까지** — 향후 디버깅 코드에서 자동 노출 방지
5. **row 1~7 마이그레이션 P1-2 적용 후 진행** — owner_rule_monitor 우선 안전 확보
6. **표현 룰 (J) 코드 리뷰 prompts/agents 메모리에 영구 등록** — 향후 보고서/커밋 메시지에 누락 방지

---

> **참조 문서**:
> - `docs/02-design/security/trading-safety-architecture-review.md` (5/27 사고 후속, OWASP + KISA)
> - `docs/02-design/security/phase1-security-review-5_28.md` (5/28 1차 P0 봉합 직후 1차 검수)
> - `docs/01-plan/trading-factory-v1-architecture.md` §7
> - `docs/02-design/quant-runtime-truth-pack.md` §3 L9
> - `src/use_cases/order_intents_gate.py` (Trading Factory v1 L10, 463줄)
> - `src/strategies/chart_hero_executor.py` (C2 fix + Note 4)
> - `src/adapters/kis_order_adapter.py:125~135` (Layer 3 mode 검증)
> - `src/adapters/paper_order_adapter.py:168~184, 221~235` (paper guard)
> - `tests/test_order_intents_gate.py` + `tests/test_adapter_intents_integration.py`
> - `ops/codex_outbox/20260528T130158..._changes-requested.md` (코덱스 1차)
> - `ops/codex_outbox/20260528T160000..._note4_alert_fallback.md` (코덱스 5/28 16:00 — Note 4)

---

## 부록 A. 코드 검증 결과 (실측 라인 인용)

### A.1 HMAC 서명 deterministic JSON (`order_intents_gate.py:94~103`)
```python
def _compute_signature(intent: dict) -> str:
    key = _get_hmac_key()
    payload = {k: v for k, v in intent.items() if k not in HMAC_EXCLUDED_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    sig = hmac.new(key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return sig
```
검증: `sort_keys=True`, `separators=(",", ":")` 양쪽 모두 적용 — JSON 정합성 OK.

### A.2 quant+live register 차단 (`order_intents_gate.py:405~409`)
```python
if bot == "quant" and str(intent.get("mode", "")).lower() == "live":
    raise OrderIntentError(
        "[REGISTER] quant bot은 live mode intent 등록 금지 — research/selector 역할만 수행. "
        "live 매매는 day bot 또는 별도 승인된 executor만 가능."
    )
```

### A.3 quant+live assert 차단 (`order_intents_gate.py:284~289`)
```python
if executor_bot == "quant" and mode == "live":
    raise OrderIntentError(
        "[INPUT] quant executor + live mode 조합 금지 — "
        "quant bot은 research/selector 역할만, live 매매 권한 없음. "
        "live 매매는 day bot 또는 별도 승인된 executor만 가능."
    )
```

### A.4 KisOrderAdapter mode='live' 강제 (`kis_order_adapter.py:125~135`)
```python
if mode is not None or executor_bot is not None:
    if mode != "live":
        raise ValueError(
            f"[GUARD] KisOrderAdapter는 mode='live'만 허용 (received='{mode}'). "
            "paper 매매는 PaperOrderAdapter 사용. ..."
        )
```

### A.5 _execute_sell 텔레그램 실패 logger.warning (Note 4, `chart_hero_executor.py:509~514`)
```python
except Exception as tg_err:
    logger.warning(
        "[%s] 텔레그램 ALERT 발송 실패 (매도 차단 자체는 정상) %s: %s — 운영자 수동 확인 필요",
        action_tag, d["ticker"], tg_err,
    )
```

### A.6 _execute_sell 결과 dict 반환 (C2 fix, `chart_hero_executor.py:515~520`)
```python
return {
    "action": action_tag, "ticker": d["ticker"],
    "price": price, "qty": qty,
    "trigger_reason": action.get("reason", ""),
    "block_reason": str(e),
}
```
호출자 `monitor_positions` (line 406~409) 에서 `results.append(sell_result)` + `continue` — PnL/수량/close 변경 없이 결과 추적.

---

## 부록 B. 표현 룰 (J) 코덱스 5/28 결정 — 본 문서 자체 점검

본 문서가 다음 표현을 회피했는지 자체 점검:

| 금지 표현 | 본 문서 사용 여부 | 비고 |
|----------|---------------|------|
| "전체 주문 경로 L10 완료" | 미사용 | D-2 R-D1에 "마이그레이션 미완" 명시 |
| "Phase 1 paper 재가동 가능" | 미사용 | "코덱스 검수 GO + 사용자 결단 + paper dry-run" 조건 분리 |
| "live 안전망 완성" | 미사용 | SPOF-5, SPOF-6 신규 도입 명시 |
| "안전" (단정) | 미사용 (잔존 위험 명시와 함께만 사용) | 위험 매트릭스로 대체 |

→ 본 문서는 표현 룰 (J) 준수.
