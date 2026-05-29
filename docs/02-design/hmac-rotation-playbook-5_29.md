# HMAC Rotation Playbook — `ORDER_INTENTS_HMAC_KEY` 회전 절차

> **상태**: 운영 절차 문서 (옵션 A 승인, 다중 키 운영 보류)
> **HEAD**: 35718cb (작업 전)
> **계기**: 5/28 사장님 결정문 P1-4 + 5/29 결단 옵션 A 승인 — "정상/응급 회전 절차 + 검증 스크립트 작성. 다중 키 운영은 보류."
> **상위 문서**: `docs/02-design/p1-residual-plan-5-29.md` §3

---

## 0. 한 줄 요약

> **HMAC 키 회전은 (정상) 분기 1회 / (응급) 키 유출 의심 즉시. 다중 키 운영은 보류 — 회전 중에는 모든 기존 intent를 만료 처리하고 새 키로 일괄 전환.**

---

## 1. 현재 상태

### 1-1. 키 설정
- 환경변수: `ORDER_INTENTS_HMAC_KEY`
- 위치: `.env` (로컬 + VPS 동기)
- 길이: 64자 (5/29 09:30 preflight 검증)
- 강제 검증: `src/use_cases/order_intents_gate.py:75-90` — 32+ 문자 미만 시 `IntentSignatureError` raise

### 1-2. 키 사용처 (5/29 기준)
| 파일 | 역할 |
|---|---|
| `src/use_cases/order_intents_gate.py` | HMAC 서명 + 검증 (gate 핵심) |
| `tools/quant_preflight.py` | 키 존재/길이 검증 (운영 진입 전 가드) |
| `tests/test_phase1_paper_trade.py` | 키 fail-fast 회귀 (4건) |
| `tests/test_order_intents_gate.py` | 서명/검증 단위 테스트 |
| `tests/test_adapter_intents_integration.py` | 어댑터 통합 테스트 |
| `docs/02-design/security/phase1-comprehensive-review-5_28.md` | 보안 검수 문서 (참조) |
| `docs/02-design/security/phase1-security-review-5_28.md` | 보안 검수 문서 (참조) |
| `docs/01-plan/trading-factory-v1-architecture.md` | 아키텍처 설계 (참조) |

### 1-3. 회전 절차 현황
- **정상 회전**: 미정의 (절차 부재)
- **응급 회전**: 미정의 (사고 대응 매뉴얼 부재)
- **회전 스크립트**: 미작성

---

## 2. 회전 주기

| 시나리오 | 주기 |
|---|---|
| 정상 회전 | **분기 1회** (3개월 주기) — 다음 회전 예정 2026-08-29 |
| 응급 회전 | **즉시** — 키 유출 의심 / .env 노출 / 백업 분실 |
| 운영 환경 변경 시 | VPS 마이그레이션 / SSH 인증 체계 변경 시 1회 |

---

## 3. 정상 회전 절차 (분기 1회)

### 3-1. 사전 점검 (작업 5분 전)
```bash
# 1) 현재 상태 확인
source venv/Scripts/activate
python -u -X utf8 tools/quant_preflight.py
# 기대: RESULT: PASS (ORDER_INTENTS_HMAC_KEY: present (len=64))

# 2) VPS 동기 상태 확인
ssh -i "D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem" -o ConnectTimeout=10 ubuntu@13.209.153.221 \
  "grep -c '^ORDER_INTENTS_HMAC_KEY=' /home/ubuntu/quantum-master/.env"
# 기대: 1

# 3) 매매 cron 정지 상태 확인 (★ 5/29 code-analyzer 검수 반영: adaptive + run_adaptive_cycle 추가)
ssh -i "..." ubuntu@13.209.153.221 \
  "crontab -l 2>/dev/null | grep -E '(auto_buy|owner_rule|smart_entry|sell_monitor|smart_sell|live_trading|adaptive|run_adaptive_cycle|paper_warmup)' | grep -v '^#'"
# 기대: (출력 0줄 — 모두 주석 처리)
```

### 3-2. 회전 실행 (단계별)

**Step 1: 신규 키 생성 + .env 교체 (로컬)**
```bash
python -u -X utf8 tools/rotate_hmac_key.py --confirm
# 산출물:
#   - .env.bak.YYYYMMDD_HHMM (현재 키 백업)
#   - .env (신규 키 적용)
#   - 화면: 신규 키 길이 검증 + 백업 경로 출력
```

**Step 2: 로컬 회귀 + preflight 검증**
```bash
python -u -X utf8 tools/quant_preflight.py
# 기대: RESULT: PASS

python -u -X utf8 -m pytest tests/test_phase1_paper_trade.py::TestC3_HmacKeyFailFast -v
# 기대: 4 passed
```

**Step 3: VPS 동기 (사장님 명시 승인 후)** ★ rollback 절차 보강 (security-architect 5/29 검수 반영)
```bash
# 3-1) 신규 .env scp
scp -i "D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem" \
  .env ubuntu@13.209.153.221:/home/ubuntu/quantum-master/.env.new

# 3-2) VPS 측 백업 + 교체 + 검증 + rollback 안전망 (SSH 단일 명령)
ssh -i "..." ubuntu@13.209.153.221 "
  BAK=/home/ubuntu/quantum-master/.env.bak.\$(date +%Y%m%d_%H%M)
  cp /home/ubuntu/quantum-master/.env \$BAK
  mv /home/ubuntu/quantum-master/.env.new /home/ubuntu/quantum-master/.env
  cd /home/ubuntu/quantum-master && ./venv/bin/python3.11 tools/quant_preflight.py
  RC=\$?
  if [ \$RC -ne 0 ]; then
    echo '[ROLLBACK] preflight FAIL → 백업 복구'
    cp \$BAK /home/ubuntu/quantum-master/.env
    exit \$RC
  fi
"
# 기대: RESULT: PASS (실패 시 자동 rollback)
```

**Step 4: 기존 intent 자동 만료 처리**
- 기존 등록된 intent는 신규 키로 검증 실패 → `IntentSignatureError` 자동 raise → 매매 거부
- **즉, 별도 삭제 작업 불필요** (D+1 자동 만료 + HMAC 검증 실패 이중 차단)
- 단, `data/order_intents/*.jsonl` 파일은 그대로 둠 (감리 기록)

**Step 5: VPS 단일 cron smoke test** (paper 재가동 단계에서만)
- 본 단계는 paper 재가동 심사 통과 후만 적용
- 단일 paper cron 1회 실행 + journalctl 5분 모니터링
- 기대: paper intent 등록 N건 + 실주문 0건

### 3-3. 사후 정리
```bash
# 1) 백업 무결성 확인
ls -la .env.bak.* 2>&1 | head -5

# 2) 회전 기록 작성
echo "$(date +%Y-%m-%d_%H:%M)  rotated by 사장님 + 메인 AI  (정상 분기)" >> ops/hmac_rotation_log.txt

# 3) 다음 회전 예정 일자 텔레그램 안내
# (자동 알림 cron 작성은 별도 PDCA)
```

---

## 4. 응급 회전 절차 (키 유출 의심 시 즉시)

### 4-1. 트리거 조건 (즉시 실행)
- `.env` 파일이 git에 실수 커밋된 흔적 발견
- `.env` 백업이 외부 저장소(GitHub/Google Drive)에 노출 의심
- VPS SSH key 분실 / 권한 외 접근 흔적
- 로그 파일에 키 일부 노출 발견
- 운영자 PC 분실/해킹 의심

### 4-2. 단계별 실행 (T+0 분 ~ T+30분)

**T+0: KILL_SWITCH 즉시 활성화**
```bash
# 로컬
touch data/KILL_SWITCH

# VPS
ssh -i "..." ubuntu@13.209.153.221 \
  "touch /home/ubuntu/quantum-master/data/KILL_SWITCH"
```
**효과**: `assert_runtime_orders_allowed()` 강제 raise → 모든 매매 즉시 차단

**T+5: 신규 키 생성 + 즉시 교체**
```bash
# 정상 회전 Step 1과 동일 (--emergency 옵션 추가)
python -u -X utf8 tools/rotate_hmac_key.py --confirm --emergency
# 차이: ops/hmac_rotation_log.txt에 [EMERGENCY] 태그 + 사유 기록
```

**T+10: VPS 동기 (사장님 명시 즉시 승인 + scp)**
```bash
scp -i "..." .env ubuntu@13.209.153.221:/home/ubuntu/quantum-master/.env.new
ssh -i "..." ubuntu@13.209.153.221 "
  cp /home/ubuntu/quantum-master/.env /home/ubuntu/quantum-master/.env.compromised.$(date +%Y%m%d_%H%M)
  mv /home/ubuntu/quantum-master/.env.new /home/ubuntu/quantum-master/.env
  cd /home/ubuntu/quantum-master && ./venv/bin/python3.11 tools/quant_preflight.py
"
```

**T+15: 기존 intent 무효화 확인**
```bash
# VPS jsonl 파일 모두 신규 키 검증 실패 확인
ssh -i "..." ubuntu@13.209.153.221 "
  cd /home/ubuntu/quantum-master &&
  ./venv/bin/python3.11 -c '
from src.use_cases.order_intents_gate import list_today_intents
intents = list_today_intents(verify_signatures=True)
invalid = [i for i in intents if not i.get(\"_signature_valid\", True)]
print(f\"총 intent: {len(intents)}, 신규 키로 검증 실패: {len(invalid)}\")
'
"
# 기대: 신규 키로 검증 실패 = 전체 intent 수 (모두 거부됨)
```

**T+20: 사고 보고서 작성**
- 파일: `ops/hmac_compromise_report_YYYYMMDD.md`
- 내용:
  1. 유출 의심 일시 + 발견 경위
  2. 노출 범위 (key / 로그 / 백업 / .env)
  3. 회전 실행 일시
  4. 후속 조치 (git 히스토리 정리 / 노출 채널 폐쇄 / 다음 회전 예정)

**T+25: KILL_SWITCH 유지 + 사장님 결단 대기**
- 키 회전 완료 후에도 KILL_SWITCH 유지 (paper 재개 결단은 별도)
- 매매 cron 6개 정지 상태 유지 (재가동 심사서 통과 전까지)

**T+30: 텔레그램 사고 알림** (선택)
- 사장님 + 메인 AI + Codex 채널에 사고 보고

### 4-3. 후속 조치 (T+1일 ~ T+1주)

- (1) Git 히스토리에 키 흔적 발견 시: `git filter-repo` 또는 `bfg-repo-cleaner` 적용 (force push 사장님 승인 필수)
- (2) 노출 채널 영구 폐쇄 (백업 위치 변경 / GitHub repo private / Google Drive 폴더 권한 강화)
- (3) 운영자 PC 보안 점검 (전체 디스크 검사 / 패스워드 일괄 변경)
- (4) 1주일 후 회귀 점검 + 정상 회전 cycle 재시작

---

## 5. 회전 스크립트 — `tools/rotate_hmac_key.py`

### 5-1. 명세
- 위치: `tools/rotate_hmac_key.py`
- 권한: 로컬 실행만 (VPS 동기는 별도 scp 명령)
- 의존: `python>=3.10`, `secrets` (stdlib), `argparse` (stdlib), `pathlib` (stdlib)
- 인자:
  - `--confirm` (필수): 실행 의도 확인
  - `--emergency` (선택): 응급 회전 모드 (로그에 [EMERGENCY] 태그)
  - `--length` (선택): 키 길이 (default 64, 32 미만 거부)
  - `--dry-run` (선택): 실제 .env 변경 없이 시뮬레이션
- 동작:
  1. `.env` 존재 확인 + 권한 검증
  2. 현재 `ORDER_INTENTS_HMAC_KEY=...` 줄 추출 + 백업
  3. `secrets.token_urlsafe(48)` → 64자 신규 키 생성
  4. `.env.bak.YYYYMMDD_HHMM` 작성
  5. `.env` 내 `ORDER_INTENTS_HMAC_KEY=신규키` 교체 (regex)
  6. `ops/hmac_rotation_log.txt` append
  7. preflight 자동 호출 + PASS 확인

### 5-2. 작성 결단 (옵션 A 승인)
스크립트 작성 권장 — 정상/응급 회전 절차 자동화 + 사고 시 인적 오류 방지.

→ **작성 시점**: 4건 통합 분업 검수 PASS 후 + 사장님 별도 승인 시 (지금은 명세만 확정)

---

## 6. 다중 키 운영 보류 사유 (옵션 B 미적용)

### 6-1. 보류 결단 (사장님 5/29 승인)
- 다중 키 운영 (KEY_V1 + KEY_V2 병행 검증) 미도입
- 회전 중에는 KILL_SWITCH로 매매 차단 + 모든 기존 intent 무효화 + 신규 키 일괄 전환

### 6-2. 보류 합리적 근거
1. **운영 부담**: 다중 키 운영은 intent 스키마에 `key_version` 필드 추가 + 검증 로직 복잡화
2. **회전 빈도**: 정상 분기 1회 → 짧은 차단 시간(30분) 허용 가능
3. **응급 회전**: 키 유출 시 zero-downtime보다 즉시 차단이 안전
4. **코드 변경 범위**: 다중 키 운영은 ~4시간 + 회귀 + 검수 추가 — 현 단계 미필요

### 6-3. 향후 옵션 B 도입 트리거 (조건부)
- live 매매 운영 진입 + 분기 회전 시 zero-downtime 요구 발생 시
- 회전 빈도가 월 1회 이상으로 증가 필요 시
- 다중 cron 동시 가동 단계에서 회전 윈도우 문제 발생 시

---

## 7. 표현 룰

### 사용 가능
- "HMAC 회전 정상/응급 절차 명문화 완료"
- "회전 스크립트 명세 확정 (작성은 분업 검수 PASS 후)"
- "다중 키 운영 보류 결단 — 회전 중 일괄 차단 + 일괄 전환"

### 사용 금지
- "HMAC 운영 완성" X (스크립트 미작성)
- "회전 자동화 완료" X (명세만 확정)
- "다중 키 운영 적용" X (보류)

---

## 8. 검수 의뢰 사항 (Codex)

1. 정상/응급 회전 절차 안전성 (T+0 ~ T+30분)
2. 회전 스크립트 명세 (인자/동작) 적정성
3. 다중 키 운영 보류 결단의 합리적 근거 4건 충분성
4. 응급 회전 시 KILL_SWITCH 유지 정책 적정성
5. 향후 옵션 B 도입 트리거 조건 적정성

---

## 9. 연결 문서
- `docs/02-design/p1-residual-plan-5-29.md` §3
- `docs/02-design/filelock-policy-5_29.md` (관련 정책)
- `src/use_cases/order_intents_gate.py:75-90` (`_get_hmac_key`)
- `tools/quant_preflight.py` (HMAC 키 가드)
- `tests/test_phase1_paper_trade.py::TestC3_HmacKeyFailFast` (회귀 4건)
