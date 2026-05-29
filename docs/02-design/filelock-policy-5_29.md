# filelock 정책 — `register_intent` jsonl 동시 쓰기 결단

> **상태**: 정책 결단 (코드 변경 0, 운영 룰 명시)
> **HEAD**: 35718cb (작업 전)
> **계기**: 5/28 사장님 결정문 P1-4 + 5/29 결단 옵션 A 승인 — "portalocker 도입 금지. POSIX append atomic 근거와 한계만 문서화."
> **상위 문서**: `docs/02-design/p1-residual-plan-5-29.md` §2

---

## 1. 결단 (한 줄)

> **`register_intent` jsonl append-write에 OS-level filelock(portalocker 등)을 도입하지 않는다.** POSIX `O_APPEND` atomic write 보장 + Python BufferedWriter 동작 분석 + 단일 cron 가동 단계 라는 3중 근거로 실전 안전을 확보하되, 다중 cron 동시 가동 진입 시점에 별도 PDCA로 portalocker 도입을 재검토한다.

---

## 2. 현재 상태

### 2-1. 코드 위치
- 파일: `src/use_cases/order_intents_gate.py:418-422`
- 코드:
  ```python
  ORDER_INTENTS_DIR.mkdir(parents=True, exist_ok=True)
  out_path = ORDER_INTENTS_DIR / f"{bot}_intents_{_today_date_str()}.jsonl"

  with out_path.open("a", encoding="utf-8") as f:
      f.write(json.dumps(intent, ensure_ascii=False) + "\n")
  ```
- 보호 메커니즘: **없음** (filelock / threading.Lock / asyncio.Lock 0건)

### 2-2. 위험 시나리오
다중 프로세스(cron 여러 개)가 같은 jsonl 파일에 동시 append할 경우:
- (A) 한 줄이 다른 줄 중간에 끼어들어 손상된 JSON 발생
- (B) 쓰기 완료 전 다른 프로세스가 읽기 시작 → partial JSON parse 실패
- (C) 파일 길이 / 줄 수 race condition

---

## 3. POSIX `O_APPEND` atomic write 보장 (1차 근거)

### 3-1. POSIX 표준 조항
- POSIX.1-2017 §2.9.7: `O_APPEND` flag로 열린 파일에 대한 `write()` 시스템 콜은 다음을 보장:
  > "Each `write()` request shall be performed atomically with respect to other (`O_APPEND`-using) writers."
- 즉, 같은 파일에 `O_APPEND`로 열려 있는 모든 writer 사이에는 `write()` 호출 단위로 atomic 보장.

### 3-2. Linux 일반 파일 atomic 보장 메커니즘 ★ (정확성 보강 — security-architect 5/29 검수 반영)
- Linux 일반 파일에 대한 `O_APPEND` atomic write 보장은 **PIPE_BUF (4096B)** 가 아니라 **kernel inode lock (`inode->i_rwsem`, Linux 3.14+)** 으로 보장됨.
- 핵심: `__generic_file_write_iter` (mm/filemap.c) 가 inode rwsem 락 안에서 단일 write 수행 → 다른 writer와 절대 섞이지 않음.
- PIPE_BUF는 엄밀히는 pipe/FIFO 한정 (POSIX.1-2017 §2.9.2). 일반 파일에는 적용 X.
- **결론**: Linux 일반 파일 + `O_APPEND` + 단일 `write()` 시스템 콜이면 페이로드 크기 무관 atomic (PIPE_BUF보다 더 강한 보장).
- 단, **Python BufferedWriter가 큰 페이로드를 여러 시스템 콜로 분할할 가능성** 은 §4-2 한계로 남음 → 페이로드 크기 < BufferedWriter buffer (8192B) 유지가 본질적 보호.

### 3-3. 본 시스템 적용 가능 여부
- **VPS 환경**: Ubuntu 24.04 Linux (AWS Lightsail Seoul) → POSIX 준수
- **로컬 환경**: Windows 11 PowerShell (개발 환경) → POSIX 미준수 영역. 단, 로컬에서는 cron 가동 0건이라 race 발생 가능성 0.
- **운영 (paper cron 1줄 가동 단계)**: VPS Linux 단일 프로세스 → race 미발생.
- **운영 (다중 cron 동시 가동 단계)**: VPS Linux + 페이로드 < 4096B → atomic 보장.

---

## 4. Python BufferedWriter 동작 분석 (2차 근거)

### 4-1. CPython 구현 검증
- `open(path, "a")` → `BufferedWriter` (default buffer 8192B)
- `f.write(s)`는 buffer에 누적 후 flush 시점에 단일 시스템 콜 `write()` 호출.
- `with` 블록 종료 시 `__exit__` → `close()` → `flush()` → 단일 `write()` 시스템 콜.
- **결론**: 단일 `f.write(...)` 호출 + with 블록 종료가 단일 시스템 콜로 귀결.

### 4-2. 잠재 위험
- buffer 크기(8192B)를 초과하는 write는 내부적으로 여러 `write()` 시스템 콜로 분할 가능.
- 본 시스템에서는 줄당 크기 < 8192B 보장됨 (다음 절 참고).

---

## 5. Intent 줄 크기 실측/추정 (3차 근거)

### 5-1. 실측 결과 (5/29 현재)
- `data/order_intents/` 디렉토리 미생성 (5/28 21:30 paper cron 긴급정지 이후 신규 intent 0건)
- → 실측 불가, 스키마 기반 추정으로 대체

### 5-2. 스키마 기반 추정
`REQUIRED_INTENT_FIELDS = {"intent_id", "bot", "engine", "ticker", "side", "mode", "score", "created_at", "expires_at"}` + 자동 추가 `hmac_signature` + 선택 필드 `engine_params` 등

샘플 JSON 1줄 (typical):
```json
{"intent_id":"q_240810_chart_hero_d1_2026-05-29","bot":"quant","engine":"chart_hero","ticker":"240810","side":"BUY","mode":"paper","score":85.5,"created_at":"2026-05-29T09:00:00+09:00","expires_at":"2026-05-29T15:30:00+09:00","hmac_signature":"a1b2c3d4e5f6...64hex","engine_params":{"premium":0.5,"timeout":300}}
```
**추정 크기**: 400~600 바이트/줄

### 5-3. 상한 계산
- 최악 시나리오 (모든 선택 필드 + 한글 ticker_name 등): **<1500 바이트/줄 예상**
- < 4096B (POSIX `PIPE_BUF`) **+** < 8192B (Python BufferedWriter default) → **이중 안전 여유**

---

## 6. 단일 cron 가동 단계 (4차 근거 — 운영 컨텍스트)

### 6-1. 현재 운영 상태 (5/29 09:30 기준)
- quantum-scheduler: `inactive + masked`
- 매매 cron 6개: 모두 `# [긴급정지 5/28]` 주석 처리
- 살아있는 cron 57개: paper/scanner/learning 등 비매매 (intent 미등록 경로)
- intent 등록 빈도: **0건/일** (paper cron 재가동 전까지)

### 6-2. 향후 단계 진입 시 (paper 재가동 심사 통과 시)
- 첫 가동: **단일 paper cron 1줄**만 허용 (사장님 정책)
- 이 단계: multi-writer race 발생 가능성 = 0
- 이후 단계 (다중 cron 동시 가동): race 가능성 발생 → 옵션 B (portalocker 도입) 별도 PDCA 진입

---

## 7. portalocker 미적용 결단의 합리적 근거 요약

| 근거 | 보호 수준 |
|---|---|
| (1) POSIX `O_APPEND` atomic | < 4096B write는 절대 섞이지 않음 |
| (2) Python BufferedWriter 단일 write() 호출 | with 블록 종료 시 단일 시스템 콜 |
| (3) Intent 줄 크기 < 4096B (추정 400~600B) | (1)+(2) 적용 범위 안 |
| (4) 단일 cron 가동 단계 (5/29~) | multi-writer race 미발생 |

**4중 보호 안전망**으로 portalocker 의존 추가 없이 실전 안전 확보 가능.

---

## 8. 한계 / 잠재 위험 (정직 명시)

### 8-1. 본 결단이 보장하지 않는 시나리오
- (A) **다중 프로세스 동시 가동** + intent 줄 크기 > 4096B (예: engine_params에 거대 dict 삽입)
  - 발생 가능성: 낮음 (현재 스키마 + 운영 단계)
  - 검출 방법: 향후 jsonl 파일 줄 크기 모니터링 권장
- (B) **Windows 로컬 환경**에서 다중 프로세스 동시 가동
  - 발생 가능성: 0 (로컬 cron 없음 + 개발 환경)
- (C) **NFS / SMB 등 네트워크 파일시스템** 사용
  - 발생 가능성: 0 (data/ 는 로컬 파일시스템)
- (D) **Python 3.x 미래 버전에서 BufferedWriter 동작 변경**
  - 발생 가능성: 매우 낮음 (POSIX 호환 유지가 표준)

### 8-2. 운영 중 모니터링 권장 항목 (P3)
- jsonl 파일 줄 크기 분포 (max / p99)
- HMAC 검증 실패 비율 (위조 외에 손상도 검출)
- intent 등록 후 `list_today_intents(verify_signatures=True)` 호출 시 `_signature_valid=False` 비율

---

## 9. 향후 portalocker 도입 트리거 (조건부 PDCA)

다음 중 **하나라도 해당 시** 옵션 B (portalocker 도입) PDCA 즉시 진입:

1. **다중 cron 동시 가동 결단** (paper 또는 live)
2. **intent 줄 크기 > 2048B** 관측 (안전 여유 절반 소진 시)
3. **HMAC 검증 실패 + 위조 아닌 손상** 1건 이상 관측
4. **운영 환경 NFS/SMB 마이그레이션**
5. **Python BufferedWriter 동작 변경** 공지 발견

---

## 10. 옵션 B (도입 시) 설계 미리보기

향후 PDCA 진입 시 적용 패턴 (참고):
```python
import portalocker

with out_path.open("a", encoding="utf-8") as f:
    portalocker.lock(f, portalocker.LOCK_EX)
    try:
        f.write(json.dumps(intent, ensure_ascii=False) + "\n")
        f.flush()
    finally:
        portalocker.unlock(f)
```
- 의존: `portalocker` (이미 P0-1 atomic write에 사용 중 — 신규 의존 X)
- 성능 비용: ~수십 마이크로초/intent (cron 등록 빈도가 매우 낮아 무시 가능)
- 회귀 영향: backward compat 보장 (functional 변경 없음)

---

## 11. 표현 룰

### 사용 가능
- "POSIX append atomic 4중 보호로 실전 안전 확보"
- "portalocker 미적용 결단의 합리적 근거 4건"
- "다중 cron 동시 가동 진입 전 옵션 B 별도 PDCA"

### 사용 금지
- "filelock 문제 해결 완료" X (해결 아님, 도입 미적용 결단)
- "어떤 동시성 시나리오도 안전" X (한계 §8 명시)
- "filelock 불필요" X (조건부 도입 트리거 §9 명시)

---

## 12. 검수 의뢰 사항 (Codex)

1. POSIX `O_APPEND` atomic + Python BufferedWriter 동작 분석 정확성
2. Intent 줄 크기 추정의 합리성 (실측 0건 한계)
3. portalocker 미적용 결단의 합리적 근거 4건 충분성
4. 향후 옵션 B 진입 트리거 5건 적정성
5. P3 모니터링 항목 적정성

---

## 13. 연결 문서
- `docs/02-design/p1-residual-plan-5-29.md` §2 (계획)
- `src/use_cases/order_intents_gate.py:370-430` (register_intent 구현)
- `src/use_cases/atomic_write.py` (P0-1 portalocker 사용처 — 참조)
- `memory/decision_5_28_p1_blockers.md` (P1-4 사장님 결정문)
