# Phase 1 보안 검수 — Trading Factory v1 (10층 안전망)

> **작성일**: 2026-05-28
> **분류**: 보안 권장안 (코드 수정 없음, 설계 문서만)
> **검수 관점**: OWASP 2021 + KISA 금융보안 + Defense-in-Depth + Capability-based Security
> **상태**: 메인 AI 단독 결정 금지. 코덱스 외부 검증 후 적용.
> **전제 문서**: `docs/02-design/security/trading-safety-architecture-review.md` (5/27 사고 후속)

---

## 0. 결론 (TL;DR)

**Phase 1 안전망 10층화 = 5/27 사고 대비 본질적 개선이 맞다.** 그러나 다음 4개 영역에서 잔존 위험이 식별됨:

1. **HMAC 키 관리 거버넌스 부재** (회전·노출 대응 미정의) — **P1 High**
2. **append-only 파일시스템 (immutable) 미적용** — HMAC 서명만으로는 삭제·재기록 공격 일부 노출 — **P1 High**
3. **row 5/6 silent-return 정책의 양면성** — 매도 차단 시 손절 불가 위험 — **P2 Medium**
4. **paper → live 슬리피지/세율 가정 보정 메커니즘 부재** — **P2 Medium**

**즉시 차단되는 위협 (Phase 1로 해결됨)**:
- mojito raw broker 직접 호출 (P0-1: ORDER_INTENTS_GATE_DISABLED 우회 영구 제거)
- mode 누락으로 인한 paper/live 혼선 (P0-2: mode 필수 인자)
- 잘못된 executor가 다른 봇 intent 사용 (P0-3: executor_bot 매치)
- timezone-naive expires_at으로 인한 우회 (P0-4: 강제 timezone-aware)
- intent 파일 위조 (P0-5: HMAC-SHA256 서명)
- **quant + live 조합 영구 차단** (register 단계 + assert 단계 defense in depth)

---

## 1. 위험 매트릭스 (질문 영역별)

| ID | 영역 | 잔존 위험 | 심각도 | 우선순위 |
|----|------|---------|--------|---------|
| R-A1 | HMAC 키 관리 | 키 회전 정책 부재 | High | **P1** |
| R-A2 | HMAC 키 관리 | 노출 시 대응 절차 미정의 (rotation playbook 부재) | High | **P1** |
| R-A3 | HMAC 키 관리 | `.env` 평문 저장 (Phase 0 그대로) | High | P2 |
| R-B1 | quant+live 차단 | register + assert 양층 차단 — **충분** | Low | (해결됨) |
| R-B2 | quant+live 차단 | 외부 quant_intents_*.jsonl 직접 쓰기 시도 → HMAC 서명 없음 → assert에서 차단 | Low | (해결됨) |
| R-B3 | quant+live 차단 | 공격자가 HMAC 키도 탈취 시 → register 단계 우회 가능 | Medium | P2 |
| R-C1 | intent 파일 무결성 | append-only filesystem 미적용 → 삭제·rewrite 공격 가능 | High | **P1** |
| R-C2 | intent 파일 무결성 | 파일 삭제 → 당일 매매 차단 (의도된 동작, fail-closed) | Low | (해결됨, 단 알람 부족) |
| R-C3 | intent 파일 무결성 | replay attack — 어제 만료된 intent 재사용 | Low | (해결됨, expires_at + timezone-aware) |
| R-C4 | intent 파일 무결성 | 파일시스템 오류 시 fallback 정책 부재 | Medium | P2 |
| R-D1 | 텔레그램 토큰 | 단타봇 측 httpx 토큰 로깅 — 본 프로젝트 영향 없음 | Low | 모니터링 |
| R-D2 | 텔레그램 토큰 | 향후 httpx/aiohttp 도입 시 동일 위험 | Medium | P2 |
| R-D3 | 텔레그램 토큰 | 현재 requests 사용 — URL에 토큰 포함 (`API_BASE = f"...bot{TOKEN}"`) | Medium | **P1** |
| R-E1 | row 5/6 silent-return | 매도 intent 누락 → 손절 불가 → 손실 확대 위험 | High | **P1** |
| R-E2 | row 5/6 silent-return | 추매 intent 누락 → 기회 손실 (안전 측면) | Low | 정책 결정 |
| R-F1 | paper-first | paper 가정값 보정 메커니즘 부재 | Medium | P2 |
| R-F2 | paper-first | 코스닥 세율 0.20% 자동 분기 부재 (`TAX_PCT_KOSPI` 디폴트) | Low | P2 |
| R-G1 | KISA 금융보안 | 감사 로그 보존 정책 미정의 (5년 의무) | Medium | P2 |
| R-G2 | KISA 금융보안 | 비정상 거래 패턴 탐지 (FDS) 부재 | Medium | P3 |
| R-G3 | KISA 금융보안 | `.env` 단일 신뢰 — 다중 인증 부재 | Medium | P3 |

---

## 2. 영역별 상세 분석

### A. HMAC 키 관리

#### A-1. 현재 상태 (실측)

`src/use_cases/order_intents_gate.py:75~91`:
```python
key_str = os.getenv("ORDER_INTENTS_HMAC_KEY", "")
if not key_str:
    raise IntentSignatureError(...)
if len(key_str) < 32:
    raise IntentSignatureError(...)
return key_str.encode("utf-8")
```

- **알고리즘**: HMAC-SHA256 (충분)
- **키 길이**: 64-char hex = 256-bit entropy (충분)
- **검증**: deterministic JSON (`sort_keys=True, separators=(",", ":")`) → 정합성 OK
- **`hmac.compare_digest`** 사용 → timing attack 방어 OK

#### A-2. 잔존 위험

| 위험 | 시나리오 | 영향 |
|------|---------|------|
| **R-A1 키 회전 정책 부재** | 키가 한 번도 회전되지 않음. 장기간 유출 시 무한 위조 가능 | Critical (장기 노출 시) |
| **R-A2 노출 대응 절차 부재** | 키 노출 발견 시 어떻게? 새 키 발급 → 기존 intent 전체 무효화 → 재서명? 절차 없음 | High |
| **R-A3 `.env` 평문 저장** | 백업 파일(`*.bak.20260514_1622`)에도 같은 키 노출 가능성 | High |

#### A-3. 권장 (P1)

```
1. 키 ID 도입: 각 intent에 key_id 필드 추가
   {"hmac_signature": "...", "key_id": "v1"}
   - 검증 시 해당 key_id로 키 조회
   - 키 회전 시 v1 → v2 발급, v1은 deprecated 표시

2. 키 회전 주기: 분기당 1회 (3개월)
   - .env에 ORDER_INTENTS_HMAC_KEY_V1, _V2 동시 보유
   - 새 intent는 V2 서명, V1 intent는 검증 가능 (grace period 1주)
   - 1주 후 V1 키 삭제

3. 노출 대응 playbook (docs/security/hmac-key-rotation.md):
   STEP 1. KILL_SWITCH 즉시 활성화 (모든 매매 정지)
   STEP 2. 새 키 발급 (openssl rand -hex 32)
   STEP 3. .env 양쪽 (로컬 + VPS) 갱신
   STEP 4. 기존 intent 전체 invalidate (key_id 매칭 X → 거부)
   STEP 5. selector 재실행으로 새 intent 발급
   STEP 6. KILL_SWITCH 해제 + 텔레그램 알람

4. 단기 (P2): .env → AWS Secrets Manager 또는 OS keyring
```

---

### B. quant+live 영구 차단

#### B-1. 현재 구현 (defense in depth — 양층 차단)

**Layer 1 (register 단계)** `order_intents_gate.py:368~372`:
```python
if bot == "quant" and str(intent.get("mode", "")).lower() == "live":
    raise OrderIntentError(
        "[REGISTER] quant bot은 live mode intent 등록 금지 — research/selector 역할만 수행. "
        "live 매매는 day bot 또는 별도 승인된 executor만 가능."
    )
```

**Layer 2 (assert 단계)** `order_intents_gate.py:247~252`:
```python
if executor_bot == "quant" and mode == "live":
    raise OrderIntentError(
        "[INPUT] quant executor + live mode 조합 금지 — ..."
    )
```

**Layer 3 (adapter)** `kis_order_adapter.py:125~135`:
```python
if mode is not None or executor_bot is not None:
    if mode != "live":
        raise ValueError("KisOrderAdapter는 mode='live'만 허용...")
```

#### B-2. 우회 시나리오 분석

| 시나리오 | 차단되는가 | 근거 |
|---------|----------|------|
| 정상 호출 `register_intent({..., "bot":"quant", "mode":"live"})` | O | Layer 1 |
| 외부에서 `quant_intents_*.jsonl`에 mode="live" 직접 쓰기 (HMAC 없음) | O | Layer 2 — `_verify_signature` False → `IntentSignatureError` |
| 외부에서 mode="live" + 위조 hmac_signature 작성 | O | Layer 2 — HMAC 키 모르면 검증 실패 |
| HMAC 키까지 탈취한 공격자가 mode="live" intent 위조 | **△ 부분** | Layer 3 — adapter 진입 시 `executor_bot="quant"` + `mode="live"` 차단됨. **단, day intent에 quant 종목 등록하면 우회 가능** |
| `day_intents_*.jsonl`에 mode="live"로 직접 등록 | **△** | 정상 day bot 매매로 처리됨 — **B-3 추가 가드 필요** |

#### B-3. 잔존 위험 + 권장

**R-B3 (Medium, P2)**: HMAC 키 탈취 + day intent 위조 시 차단 메커니즘 부재.

권장:
```
1. intent에 selector_signature 추가 (HMAC + selector PR_ID + timestamp)
   - selector 코드의 git SHA + 실행 시각 기반 이중 서명
   - day bot이 외부에서 호출되어도 selector 검증 통과 불가

2. day bot 화이트리스트 종목 강제: AUTO_TRADING_WHITELIST_ONLY=1
   - kis_order_adapter._guard Layer 6 (line 158~164) 이미 존재
   - day bot은 ETF 화이트리스트만 허용 → 임의 종목 매매 불가

3. live mode 종목별 일일 한도 (KISA 이중 통제):
   - 30만원 초과 시 사용자 텔레그램 응답 필요
   - 현재 자동 통과 → 권한 분리 필요
```

---

### C. order_intents 파일 무결성

#### C-1. 현재 보호 메커니즘

- HMAC-SHA256 서명 (위조 방지) — OK
- append-only mode 파일 쓰기 (`f.write` mode "a") — OK (코드 레벨)
- expires_at 기반 만료 (replay 차단) — OK
- timezone-aware 강제 (P0-4) — OK

#### C-2. 잔존 위험

**R-C1 (High, P1) — Append-only filesystem 미적용**

현재 `.jsonl` 파일은 **일반 ext4/NTFS 파일**. 다음 공격이 가능:

| 공격 | 시나리오 | 차단되는가 |
|------|---------|----------|
| **삭제** | `rm data/order_intents/day_intents_20260528.jsonl` | △ 의도된 동작 (당일 매매 차단), 단 알람 없음 |
| **재기록 (truncate)** | 파일 비우고 위조 intent 작성 | △ HMAC 차단되나 감사 로그 손실 |
| **선택적 라인 제거** | sed로 특정 intent 삭제 | O (HMAC) 단 삭제 사실 미탐지 |
| **시간 backdate** | OS 파일 mtime 변조 | O (expires_at은 파일 내부) |
| **race condition** | register와 동시에 외부 write | 가능성 있음 (file lock 없음) |

#### C-3. 권장 (P1)

```
1. append-only 파일 속성 (Linux chattr +a):
   chattr +a data/order_intents/
   - 삭제 불가, 수정 불가, append만 가능
   - root만 chattr -a 가능
   VPS 배포 시 systemd 또는 ansible 등록

2. 파일 lock (fcntl.flock):
   register_intent에서 fcntl.flock(f, fcntl.LOCK_EX) 호출
   - 동시 쓰기 race condition 차단

3. 파일 해시 체인 (Merkle-like):
   각 line에 prev_line_hash 필드 추가
   - 중간 라인 삭제 시 chain 깨짐 → 탐지 가능
   - 매일 17:00 검증 cron

4. 외부 백업:
   data/audit/order_intents/{YYYY-MM-DD}.jsonl.gz → S3 (또는 별도 VPS)
   - 매일 16:30 cron으로 백업
   - 5년 보존 (KISA 의무)

5. 파일시스템 오류 fallback (R-C4):
   if not ORDER_INTENTS_DIR.exists():
       # 현재: 빈 리스트 반환 → 결과적 fail-closed (NoIntentError)
       # 보강: 즉시 KILL_SWITCH 활성화 + 텔레그램 ALERT
```

---

### D. 텔레그램 토큰 노출

#### D-1. 본 프로젝트 자체 검증 (실측)

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

#### D-2. 잔존 위험

**R-D3 (Medium → P1)** — `API_BASE`에 토큰 포함된 URL 패턴은 향후 디버깅·예외 처리 코드에서 자동 노출 위험.

권장:
```python
# 현재
API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# 권장
_API_HOST = "https://api.telegram.org"

def _api_url(method: str) -> str:
    # 토큰을 URL 안에 두되, 외부 노출 함수에서는 마스킹
    return f"{_API_HOST}/bot{TELEGRAM_BOT_TOKEN}/{method}"

def _masked_url(method: str) -> str:
    return f"{_API_HOST}/bot{TELEGRAM_BOT_TOKEN[:4]}...{TELEGRAM_BOT_TOKEN[-4:]}/{method}"

# 모든 logger.error/warning에서 _masked_url() 사용
```

**R-D2 (Medium, P2)** — 향후 httpx/aiohttp 도입 시:
```
- httpx는 디폴트로 URL 로깅 (logger="httpx")
- 적용 시 logging.getLogger("httpx").setLevel(WARNING) 강제
- 또는 httpx.Client(event_hooks=...) 로 URL sanitization hook
```

---

### E. row 5/6 추매·매도 silent-return 정책

#### E-1. 현재 동작 (실측)

`src/strategies/chart_hero_executor.py:398~438`:
```python
def _execute_add_buy(self, d: dict, price: int, qty: int):
    try:
        self.order.buy_limit(d["ticker"], price, qty,
            mode="paper" if self.paper else "live",
            executor_bot="quant")
    except Exception:
        return  # 추매 차단 — 평단/수량 변경 X

def _execute_sell(self, d: dict, price: int, qty: int, action: dict):
    try:
        self.order.sell_limit(d["ticker"], price, qty,
            mode="paper" if self.paper else "live",
            executor_bot="quant")
    except Exception:
        return  # 매도 차단
```

#### E-2. 트레이드오프 분석

| 시나리오 | silent-return의 결과 |
|---------|-------------------|
| **추매 intent 누락** | 매수 안 됨 → 기회 손실 (안전 측) |
| **매도 intent 누락 (정상 시장)** | 매도 안 됨 → 보유 지속 → 손실 확대 가능 |
| **매도 intent 누락 (급락장)** | **손절 못 함 → 큰 손실 위험** ★ |
| **추매 intent 누락 (수익 종목)** | 추가 수익 기회 손실 |

#### E-3. R-E1 (High, P1) — 매도 차단 위험

**핵심 모순**: order_intents_gate는 "매매 안 함"이 안전한데, 매도는 "안 함"이 위험할 수 있음.

권장 정책:
```
1. 매도 silent-return 즉시 폐지 → 에러 전파:
   def _execute_sell(self, d, price, qty, action):
       try:
           self.order.sell_limit(...)
       except NoIntentError:
           # ★ 매도 intent 누락 → SELL_BLOCKED 알람 + 사용자 즉시 통보
           telegram.send("[ALERT] 매도 intent 누락: %s — 수동 매도 검토" % d["ticker"])
           raise  # 호출자에게 전파, _save_positions 호출 X
       except Exception as e:
           # 다른 예외도 알람
           telegram.send("[ALERT] 매도 실패: %s — %s" % (d["ticker"], e))
           raise

2. 추매는 silent OK (안전 측):
   def _execute_add_buy(...):
       try: ...
       except NoIntentError:
           logger.info("[ADD_BUY SKIP] %s intent 없음", d["ticker"])
           return  # 안전 측 — 추매 안 함

3. 매도 intent 자동 발급 (selector 강화):
   - 보유 종목 매일 09:00 매도 intent 자동 발급 (D+5 만료)
   - 또는 STOPLOSS 트리거 시 즉시 발급 후 sell_limit 호출
```

---

### F. paper-first 정책의 한계

#### F-1. 현재 가정값 (실측)

`src/adapters/paper_order_adapter.py:31~35`:
```python
DEFAULT_SLIPPAGE_BASE_PCT = 0.05
DEFAULT_SLIPPAGE_FALLBACK_PCT = 0.10
FEE_PCT = 0.015
TAX_PCT_KOSPI = 0.18
TAX_PCT_KOSDAQ = 0.20
```

`sell_limit(market="KOSPI")` 디폴트 → 코스닥 종목도 코스피 세율 적용 위험.

#### F-2. 권장 (P2)

```
1. 종목코드 → 시장 자동 분기:
   def _resolve_market(ticker: str) -> str:
       # KOSPI: 0/3/5/9로 시작 (일부 예외)
       # KOSDAQ: 0이 아닌 1/2/4 등 + 6자리 마지막 0
       # 정확: 종목 마스터 파일 참조
       from src.utils.ticker_market import get_market
       return get_market(ticker)  # "KOSPI" / "KOSDAQ"
   
   def sell_limit(self, ticker, ...):
       market = self._resolve_market(ticker)  # 디폴트 폐지
       ...

2. 슬리피지 실측 보정 (5/21+ 백테스트 결과 반영):
   PAPER_MIRROR_SLIPPAGE_OBSERVED_PCT 환경변수 + 자동 학습:
   - 매주 일요일 paper vs live (당일 시뮬) 비교 → 가정값 자동 조정
   - 슬리피지 가정이 실측보다 낙관적 → live 손실 위험

3. paper → live 승급 가드:
   - paper 백테스트 D+1 적중률 ≥50% (현재 §13)
   - 추가: paper 슬리피지 가정의 ±20% 이내 실측 일치 시만 승급
```

---

### G. KISA 금융보안 관점

#### G-1. 거래 감사 로그

**KISA 의무**: 모든 매매 시도/실행 **5년 보존**.

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

매일 16:30 cron으로 압축 + 외부 백업 (S3 권장)
```

#### G-2. 비정상 거래 패턴 탐지 (FDS)

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

#### G-3. 다중 인증

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

---

## 3. 우선순위 매트릭스 (P0~P3)

### P0 — 즉시 (24시간 내, 5/29까지)

| ID | 항목 | 근거 |
|----|------|------|
| P0-A | **현재 Phase 1 (코드 변경 5/28 13:01~13:37 commit)** 모두 적용 완료 검증 | 코덱스 4차 PASS 확인 |
| P0-B | `.env ORDER_INTENTS_HMAC_KEY` 로컬 + VPS 양쪽 설정 확인 + 동일성 검증 | 양쪽 불일치 시 모든 매매 차단 |
| P0-C | 5/20~5/27 기간 quant_intents 파일 전체 감리 (HMAC 검증 + 위조 흔적 확인) | 사고 발생 가능 기간 추적 |

### P1 — 1주 내 (6/3까지)

| ID | 항목 | 영역 |
|----|------|------|
| P1-1 | **append-only filesystem 적용** (`chattr +a data/order_intents/`) | C |
| P1-2 | **매도 silent-return 폐지** → 에러 전파 + 텔레그램 ALERT | E |
| P1-3 | **HMAC 키 회전 playbook 작성** + key_id 필드 도입 | A |
| P1-4 | **telegram API_BASE 토큰 마스킹** (`_masked_url`) | D |
| P1-5 | **파일 lock (fcntl.flock)** register_intent에 적용 | C |

### P2 — 1개월 내 (6월 말)

| ID | 항목 | 영역 |
|----|------|------|
| P2-1 | 종목코드 → 시장 자동 분기 (KOSPI/KOSDAQ 세율) | F |
| P2-2 | 슬리피지 실측 보정 메커니즘 (paper vs live 자동 비교) | F |
| P2-3 | 감사 로그 5년 보존 정책 + 외부 백업 (S3) | G |
| P2-4 | selector_signature 이중 서명 (HMAC 키 탈취 방어) | B |
| P2-5 | day bot 종목 화이트리스트 강제 (`AUTO_TRADING_WHITELIST_ONLY=1`) | B |
| P2-6 | `.env` → AWS Secrets Manager (KIS 키 + HMAC 키) | A |
| P2-7 | 파일시스템 오류 시 즉시 KILL_SWITCH 활성화 fallback | C |
| P2-8 | 파일 해시 체인 (Merkle-like) + 매일 검증 | C |

### P3 — Defense-in-Depth (3개월+)

| ID | 항목 | 영역 |
|----|------|------|
| P3-1 | FDS (이상 거래 탐지) 룰셋 — 시간/종목/금액/빈도 | G |
| P3-2 | 2-key system (사용자 + 코덱스 양쪽 활성화 토큰) | G |
| P3-3 | Heartbeat 기반 자동 OFF (6시간 미수신 시) | G |
| P3-4 | 단계적 활성화 (매일 09:00 자동 OFF + `/activate today`) | G |

---

## 4. SPOF (단일 실패점) 재평가 — Phase 1 적용 후

`trading-safety-architecture-review.md §3` 의 SPOF 4개에 대한 Phase 1 효과:

| SPOF | Phase 0 (5/27 사고 직전) | Phase 1 (5/28 적용 후) |
|------|------------------------|---------------------|
| **SPOF-1: mojito raw 객체 노출** | Critical (owner_rule_monitor 우회) | △ Medium — 어댑터 진입 시 mode/executor_bot 강제, 단 mojito broker 객체 자체는 여전히 public |
| **SPOF-2: `.env` 평문 환경변수** | High | High (변화 없음) — P2-6 필요 |
| **SPOF-3: KILL_SWITCH 파일 기반** | Medium-High | Medium-High (변화 없음) — order_intents가 보조하나 KILL_SWITCH 자체는 동일 |
| **SPOF-4: 단일 자격증명** | Critical | High — HMAC 키가 추가되어 다중화 일부 진전, 단 둘 다 `.env` 평문 |
| **★ SPOF-5 (신규): HMAC 키 자체** | N/A | **High** — 단일 키 + 회전 정책 부재 |

**핵심 평가**: SPOF-1 (가장 위험) 본질적 개선. SPOF-5 (신규)가 새로 도입되었으나 Phase 1 효과 > 신규 위험.

---

## 5. row 3~6 단독 마이그레이션 안전성 (사용자 결단 대기)

코덱스 5차 PASS 후 메인 AI 자율 진행 시 — 다음 가드 권장:

```
PR 분할 (각각 코덱스 검수 통과 후 commit):
- row 3: register_intent 신규 호출처 (selector 코드들) — 가장 안전
- row 4: paper_order_adapter.buy_limit/sell_limit 호출처 — 중간 위험
- row 5: ChartHeroExecutor._execute_add_buy (silent-return) — 정책 결정 필요
- row 6: ChartHeroExecutor._execute_sell (silent-return) — ★ P1-2 우선 적용

각 PR 체크리스트:
[ ] tests/test_adapter_intents_integration.py PASS
[ ] tests/test_order_intents_gate.py PASS
[ ] paper dry-run 1일 → intent 정합성 100%
[ ] 코덱스 검수 GO
[ ] 사용자 결단 확인
```

---

## 6. 메인 AI 자가 평가 (코덱스 검수 대상)

본 문서의 한계 자인:

1. **실제 공격 시도 미수행**: append-only 우회·HMAC 키 탈취 시나리오는 이론만, PoC 미작성 → 코덱스에 실제 시도 요청
2. **mojito 라이브러리 내부 미실측**: `broker.create_market_sell_order` 예외 처리/재시도 정책 미검증
3. **KISA 전자금융감독규정 원문 미확인**: 5년 보존·이중 통제 의무는 일반 권고 수준, 본 시스템 적용 범위 (개인 vs 사업자) 확인 필요
4. **HMAC 키 회전 시 grace period 1주는 추정값**: 실제 intent 발급 빈도 기준 재산정 필요
5. **silent-return 폐지가 의도된 거래 차단 효과 손상 우려**: 매도 차단이 안전한 케이스(시장 비정상)가 있을 수 있음 — 트레이드오프 백테스트 필요

본 문서는 메인 AI가 단독으로 결단할 사항이 아닙니다. **코덱스 외부 검증 + 사용자 결정** 후 적용해야 합니다.

---

## 7. 핵심 질문 답변 요약

| 질문 | 답변 |
|------|------|
| HMAC 키 64-char hex 적정성? | OK (256-bit entropy 충분). 단 회전 정책 + key_id 필드 필수 (P1-3) |
| quant+live 영구 차단 충분성? | 정상 경로 OK. HMAC 키 탈취 시나리오만 잔존 위험 → selector_signature 이중 서명 권장 (P2-4) |
| append-only filesystem 필요성? | **필요 (P1-1)**. HMAC만으로는 삭제·rewrite 공격 일부 노출 |
| 텔레그램 토큰 현재 노출 위험? | 본 프로젝트 자체 검증 — requests 사용으로 디폴트 로깅 안 됨, 단 API_BASE 패턴 위험 (P1-4 마스킹) |
| 매도 silent-return 위험성? | **★ 가장 위험**. 손절 못 함 → 큰 손실 가능 → 폐지 + 에러 전파 + 알람 (P1-2) |
| paper-first 가정의 위험? | 코스닥 세율 자동 분기 부재 (P2-1) + 슬리피지 실측 보정 메커니즘 부재 (P2-2) |
| KISA 5년 보존 의무 충족? | 현재 부분 충족 (`data/order_intents/` 영구) — 압축·외부 백업 추가 (P2-3) |

---

## 8. 다음 액션 (사용자 결정 대기)

1. **본 문서 코덱스 검수 의뢰** (`ops/codex_inbox/`)
2. **P0-B, P0-C 즉시 실행** (HMAC 키 동기화 검증 + 5/20~5/27 intent 감리)
3. **P1-2 (매도 silent-return 폐지) 6/3까지 적용 여부** — 트레이드오프 백테스트 선행 권장
4. **row 3~6 마이그레이션 P1-2 적용 후 진행** — 매도 안전성 먼저 확보
5. **기존 `trading-safety-architecture-review.md`와 통합 여부** — Phase 1 적용 결과를 §11 Phase 1 항목에 반영

---

> **참조 문서**:
> - `docs/02-design/security/trading-safety-architecture-review.md` (5/27 사고 후속, OWASP + KISA)
> - `docs/01-plan/trading-factory-v1-architecture.md` §7
> - `docs/02-design/quant-runtime-truth-pack.md` §3 L9
> - `src/use_cases/order_intents_gate.py` (Trading Factory v1 L10)
> - `tests/test_order_intents_gate.py` + `tests/test_adapter_intents_integration.py`
