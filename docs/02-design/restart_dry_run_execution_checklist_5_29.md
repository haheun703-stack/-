# §9 dry-run 실측 전 실행 체크리스트 — 5/29(금)

> **상태**: 실행 체크리스트 (실측 실행 미수행, 사장님 별도 승인 후만)
> **계기**: 5/29 사장님 결단 — "실제 dry-run 아직 실행 X. 실행 전 체크리스트 완성"
> **상위 문서**: `docs/02-design/restart-dry-run-spec-5_29.md` (4단계 정식 명세)

---

## 0. 한 줄 결론

> **§9 4단계 dry-run 실측 실행을 위한 단계별 입력/명령/기대출력/성공기준/실패기준/중단조건/승인문구/로그보존/보고템플릿을 정식 체크리스트로 명문화. 실측 실행은 본 체크리스트 모든 항목 충족 + 사장님 별도 승인 후만.**

---

## 1. 실측 실행 사전 조건 (체크리스트 진입 전 충족 의무)

다음 **모든 조건** 충족 전까지 본 체크리스트 진입 금지:

| # | 조건 | 검증 명령 | 충족 여부 |
|---|---|---|---|
| 1 | Codex 회신 ① P1-A 4건 PASS | `ls ops/codex_inbox/ \| grep 20260529.*p1-a4` | ⏸️ 대기 |
| 2 | Codex 회신 ② P2 + 4건 PASS | `ls ops/codex_inbox/ \| grep 20260529.*p2-residual` | ⏸️ 대기 |
| 3 | 사장님 commit 묶음 A 승인 | (구두 또는 문서 명시) | ⏸️ 대기 |
| 4 | `tools/quant_preflight.py --simulate-paper` 코드 작성 + 회귀 PASS | 코드 작성 별도 PDCA | ⏸️ 미작성 |
| 5 | 매매 cron 6개 정지 상태 유지 검증 | `ssh ... crontab -l \| grep -E '(auto_buy\|owner_rule\|...)' \| grep -v '^#' \| wc -l` 결과 0 | ✅ 5/29 09:30 확인 |
| 6 | quantum-scheduler masked 상태 검증 | `ssh ... systemctl is-enabled quantum-scheduler` 결과 `masked` | ✅ 5/29 09:30 확인 |
| 7 | KILL_SWITCH 파일 존재 검증 | `ssh ... ls /home/ubuntu/quantum-master/data/KILL_SWITCH` 결과 존재 | ✅ 5/29 09:30 확인 |
| 8 | 회귀 25/25 + preflight 10/10 PASS | `pytest ... && python tools/quant_preflight.py` | ✅ 5/29 종일 유지 |
| 9 | 사장님 §9 dry-run 실측 실행 별도 승인 (구두 또는 문서) | "다음 보고에 §9 실행 결과 가져오세요" 등 명시 지시 | ⏸️ 대기 |

→ **9/9 충족 시에만 §9 본 실측 진입**.

---

## 2. 단계 1: One-line crontab 초안 확인

### 2-1. 입력
- 활성화 후보 cron 1줄 (paper 전용)
- 예시:
  ```
  # [심사대상] 0 9 * * 1-5 cd /home/ubuntu/quantum-master && LIVE_TRADING_MODE=paper PYTHONPATH=/home/ubuntu/quantum-master ./venv/bin/python3.11 -u -X utf8 scripts/paper_warmup_daily.py >> /home/ubuntu/quantum-master/logs/paper_warmup.log 2>&1
  ```
- 사장님 결단으로 어떤 cron이 첫 활성화 후보인지 명시 (paper_warmup_daily / chart_hero_picker_cycle / 기타)

### 2-2. 명령 후보 (로컬)
```bash
# 초안 작성 (사장님 결단 후)
mkdir -p ops
cat > ops/restart-cron-draft-$(date +%Y%m%d).txt << 'EOF'
# [심사대상] 0 9 * * 1-5 cd /home/ubuntu/quantum-master && LIVE_TRADING_MODE=paper PYTHONPATH=/home/ubuntu/quantum-master ./venv/bin/python3.11 -u -X utf8 scripts/paper_warmup_daily.py >> /home/ubuntu/quantum-master/logs/paper_warmup.log 2>&1

# [정지 유지] # [긴급정지 5/28] */5 9-15 * * 1-5 cd /home/ubuntu/quantum-master && ./venv/bin/python3.11 scripts/owner_rule_monitor.py ...
# [정지 유지] # [긴급정지 5/28] */5 14 * * 1-5 cd /home/ubuntu/quantum-master && ... scripts/auto_buy_executor.py ...
# (다른 매매 cron 4개 동일)
EOF

# 검증 정규식 (security-architect 5/29 검수 반영)
grep -vE '^#' ops/restart-cron-draft-*.txt | grep -E '(LIVE_TRADING_MODE|SELL_MONITOR_MODE|TELEGRAM_TRADING_MODE)=paper'
DANGER_COUNT=$(grep -vE '^#' ops/restart-cron-draft-*.txt | grep -cE '\-\-(real|live|force|no-dry-run)')
echo "DANGER_COUNT=$DANGER_COUNT"
```

### 2-3. 기대 출력
- env grep: **1+건 발견** (`LIVE_TRADING_MODE=paper` 등)
- `DANGER_COUNT=0`

### 2-4. 성공 기준
- env grep 결과 1+건 (paper 명시)
- DANGER_COUNT = 0
- 매매 cron 6개 모두 주석 처리 유지 라인 명시

### 2-5. 실패 기준 (중단)
- env grep 0건 → 즉시 중단
- DANGER_COUNT ≥ 1 → 즉시 중단
- 매매 cron 주석 해제 라인 발견 → 즉시 중단

### 2-6. 필요 승인 문구
- 사장님: "단계 1 통과. 단계 2 진행하세요"

### 2-7. 로그 보존 위치
- `ops/restart-cron-draft-YYYYMMDD.txt` (커밋 보류 — 사장님 결단 후 별도)

---

## 3. 단계 2: VPS `*_MODE=paper` env grep 검증

### 3-1. 입력
- VPS `.env` 파일 (Codex 회신 + 사장님 별도 승인 후 변경 가능)

### 3-2. 명령 후보
```bash
ssh -i "D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem" -o ConnectTimeout=10 ubuntu@13.209.153.221 "
  grep -E '^(LIVE_TRADING_MODE|SELL_MONITOR_MODE|TELEGRAM_TRADING_MODE|AUTO_TRADING_ENABLED|QUANT_PAPER_ONLY|QUANT_AUTO_TRADE_DISABLED)=' /home/ubuntu/quantum-master/.env
"
```

### 3-3. 기대 출력
```
LIVE_TRADING_MODE=paper          # 또는 미설정 (default paper)
SELL_MONITOR_MODE=paper          # 또는 미설정 (default paper)
TELEGRAM_TRADING_MODE=paper      # 또는 미설정 (default paper)
AUTO_TRADING_ENABLED=0           # 필수
QUANT_PAPER_ONLY=true            # 권장
```

### 3-4. 성공 기준
- `*_MODE=live` 라인 0건
- `AUTO_TRADING_ENABLED=0` 또는 미설정
- `QUANT_AUTO_TRADE_DISABLED=false` 라인 0건

### 3-5. 실패 기준 (즉시 중단)
- **`*_MODE=live` 라인 1건 이상** → 즉시 중단 + 사장님 호출
- **`AUTO_TRADING_ENABLED=1`** → 즉시 중단
- **`QUANT_AUTO_TRADE_DISABLED=false`** → 즉시 중단

### 3-6. 필요 승인 문구
- 사장님: "단계 2 통과. 단계 3 진행하세요"

### 3-7. 로그 보존 위치
- 출력을 `ops/restart-dry-run-step2-env-YYYYMMDD-HHMM.log`로 저장

---

## 4. 단계 3: Smoke test (`quant_preflight.py --simulate-paper`)

### 4-1. 입력
- preflight 확장 코드 작성 완료 (사전 조건 #4 충족)
- VPS `.env` 검증 완료 (단계 2 통과)

### 4-2. 명령 후보 (VPS)
```bash
ssh -i "..." ubuntu@13.209.153.221 "
  cd /home/ubuntu/quantum-master &&
  ./venv/bin/python3.11 tools/quant_preflight.py --simulate-paper 2>&1
" | tee ops/restart-dry-run-step3-preflight-$(date +%Y%m%d-%H%M).log
```

### 4-3. 기대 출력
```
Quantum preflight expect=blocked
[PASS] runtime order gate: KILL_SWITCH exists
[PASS] data/KILL_SWITCH: exists
[PASS] legacy kill_switch.flag: absent
[PASS] AUTO_TRADING_ENABLED: 0
[PASS] MODEL: REAL
[PASS] QM-E schedule live command: dry-run/no-live
[PASS] QM-P missing script guard: guarded
[PASS] cron/shell danger args (run_bat.sh): no danger args
[PASS] raw mojito broker calls (use_cases/scripts): no raw calls
[PASS] ORDER_INTENTS_HMAC_KEY (Phase 1 L10 가드): present (len=64)
[PASS] S1 paper intent dict 생성
[PASS] S2 HMAC 서명 생성 + 검증
[PASS] S3 assert_order_intent_exists paper+quant
[PASS] S4 KisAdapter mode=paper 차단 OK
[PASS] S5 PaperAdapter mode=live 차단 OK
[PASS] S6 KILL_SWITCH 차단: [RUNTIME-GUARD] live broker order blocked: KILL_SWITCH exists
RESULT: PASS (16/16)
```

### 4-4. 성공 기준
- **16/16 PASS** (기존 10 + S1~S6 6)
- `RESULT: PASS`

### 4-5. 실패 기준 (즉시 중단)
- 기존 10개 중 1건 FAIL → 즉시 중단
- S1~S6 중 1건 FAIL → 즉시 중단
- S6에서 KILL_SWITCH 미존재 detected → WARN (code-analyzer 5/29 권장 보강 적용 권장)

### 4-6. 필요 승인 문구
- 사장님: "단계 3 통과. 단계 4 진행하세요" (실제 cron 1회 실행 단계 — 더욱 신중)

### 4-7. 로그 보존 위치
- `ops/restart-dry-run-step3-preflight-YYYYMMDD-HHMM.log`

---

## 5. 단계 4: Journalctl 5분 모니터링

### 5-1. 입력
- 단계 1~3 통과 완료
- 사장님 단계 4 진행 별도 승인 (실제 cron 1회 실행 단계)

### 5-2. 명령 후보 (VPS, ★ 신중)
```bash
# Step 1: 단계 1 cron 1회 수동 트리거 (cron 등록 X, 직접 실행)
ssh -i "..." ubuntu@13.209.153.221 "
  cd /home/ubuntu/quantum-master &&
  LIVE_TRADING_MODE=paper PYTHONPATH=/home/ubuntu/quantum-master \
    ./venv/bin/python3.11 -u -X utf8 scripts/paper_warmup_daily.py 2>&1 | tee /tmp/paper_warmup_smoke.log
" | tee ops/restart-dry-run-step4-trigger-$(date +%Y%m%d-%H%M).log

# Step 2: 5분 대기 (백그라운드 작업 완료 대기)
sleep 300

# Step 3: 5분간 journalctl 모니터링 (★ 5/29 code-analyzer 검수 반영: sell_limit/buy_market/emergency_liquidate 추가)
ssh -i "..." ubuntu@13.209.153.221 "
  sudo journalctl --since '5 minutes ago' --no-pager | grep -E '(quant|order_intent|sell_(market|limit)|buy_(market|limit)|KILL_SWITCH|emergency_liquidate)'
" > ops/restart-dry-run-step4-journal-$(date +%Y%m%d-%H%M).log

# Step 4: 분석
ssh -i "..." ubuntu@13.209.153.221 "
  cd /home/ubuntu/quantum-master &&
  ls -la data/order_intents/quant_intents_*.jsonl 2>&1 | tail -3 &&
  echo '---' &&
  wc -l data/order_intents/quant_intents_*.jsonl 2>&1 &&
  echo '---' &&
  echo 'KIS 실주문 흔적 grep:' &&
  grep -E 'broker\\.(buy|sell)_(limit|market)' /tmp/paper_warmup_smoke.log 2>&1 | head -3 &&
  echo '---' &&
  echo '예상치 못한 raise:' &&
  grep -iE '(Traceback|Exception|Error)' /tmp/paper_warmup_smoke.log 2>&1 | head -10
"
```

### 5-3. 기대 출력
- paper intent 등록 N건 (jsonl 줄 수 증가)
- `KILL_SWITCH exists` 또는 paper mode block 알림 1+건
- **KIS 실주문 흔적 0건** (broker.buy_limit / broker.sell_market grep 결과 빈 출력)
- 예상치 못한 raise 0건 또는 paper 경로 의도된 raise (IntentExpiredError 등)

### 5-4. 성공 기준
- jsonl 줄 수 1+건 증가 (paper intent 등록 확인)
- `broker.(buy|sell)_(limit|market)` grep 0건
- 예상치 못한 raise 0건

### 5-5. 실패 기준 (★ 즉시 중단 + 사장님 호출 + KILL_SWITCH 재확인)
- **KIS 실주문 흔적 1건 이상 발견** → 즉시 KILL_SWITCH 강제 (`touch data/KILL_SWITCH`) + scheduler 정지 검증 + 사장님 호출
- 예상치 못한 raise (Traceback) 분석 후 사장님 보고
- jsonl 줄 수 증가 0건 → paper 경로 자체 미작동 (분석 후 보고)

### 5-6. 필요 승인 문구
- 사장님: "단계 4 통과. (선택) 단계 5 Codex 재검수 진행하세요"
- 또는: "단계 4 PASS. 본 §9 사이클 완료"

### 5-7. 로그 보존 위치
- `ops/restart-dry-run-step4-trigger-YYYYMMDD-HHMM.log`
- `ops/restart-dry-run-step4-journal-YYYYMMDD-HHMM.log`
- VPS `/tmp/paper_warmup_smoke.log` (사후 분석용, 재실행 시 덮어쓰임 주의)

---

## 6. (선택) 단계 5: Codex 재검수 + 사장님 별도 승인

### 6-1. 입력
- 단계 1~4 통과 결과

### 6-2. 명령 후보
```bash
# Codex 의뢰서 신규 작성
TS=$(date +%Y%m%dT%H%M%S)
TS_TARGET=$(date +%Y%m%dT%H%M00)
cat > ops/codex_outbox/${TS}_${TS_TARGET}_quant-bot_restart-dry-run-results_review-requested.md << 'EOF'
# [퀀트봇 → 코덱스] §9 4단계 dry-run 실측 결과 검수 요청

(단계 1~4 결과 + 보고 템플릿 채움)
EOF
```

### 6-3. 기대 출력
- Codex 회신: PASS / 추가 권장사항 / FAIL

### 6-4. 성공 기준
- Codex PASS 회신

### 6-5. 실패 기준
- Codex 추가 권장사항 → 적용 후 §9 사이클 재실행
- Codex FAIL → 분석 + 사장님 별도 결단

### 6-6. 필요 승인 문구
- 사장님: "Codex PASS 확인. paper cron 1줄 정식 등록 결단"

### 6-7. 로그 보존 위치
- `ops/codex_outbox/${TS}_..._review-requested.md`
- Codex 회신 도착 시 `ops/codex_inbox/`

---

## 7. 보고 템플릿 (단계 1~4 완료 후 채움)

```markdown
# §9 4단계 dry-run 실측 결과 — YYYYMMDD HH:MM

> **실행 일시**: YYYY-MM-DD HH:MM:SS KST
> **실행자**: 사장님 + 메인 AI (입회)
> **종합 판정**: ☐ PASS / ☐ FAIL / ☐ 단계 N 중단

## 단계 1: One-line crontab 초안
- 산출물: `ops/restart-cron-draft-YYYYMMDD.txt`
- env grep 결과: __건 발견
- DANGER_COUNT: 0
- 매매 cron 6 주석 유지: ☐ YES / ☐ NO
- 판정: ☐ PASS / ☐ FAIL

## 단계 2: VPS `*_MODE=paper` env grep
- 명령: `ssh ... grep -E '^(LIVE_TRADING_MODE|...)='`
- 결과 raw (붙여넣기):
  ```
  LIVE_TRADING_MODE=paper
  ...
  ```
- 판정: ☐ PASS / ☐ FAIL

## 단계 3: Smoke test (`--simulate-paper`)
- 명령: `ssh ... ./venv/bin/python3.11 tools/quant_preflight.py --simulate-paper`
- RESULT: __ / 16 PASS
- 판정: ☐ PASS / ☐ FAIL

## 단계 4: Journalctl 5분 모니터링
- 실행 cron: `LIVE_TRADING_MODE=paper ... paper_warmup_daily.py`
- 시작/종료: HH:MM:SS ~ HH:MM:SS
- paper intent 등록: __건 (jsonl 줄 수)
- KIS 실주문 흔적: __건 (grep 결과)
- 예상치 못한 raise: __건
- 판정: ☐ PASS / ☐ FAIL

## (선택) 단계 5 Codex 재검수
- 의뢰서: `ops/codex_outbox/...`
- Codex 회신: ☐ PASS / ☐ FAIL / ☐ 미진행

## 종합
- 1~4 모두 PASS: ☐ YES / ☐ NO
- (선택) Codex PASS: ☐ YES / ☐ NO / ☐ 미진행
- 사장님 paper cron 정식 등록 결단: ☐ 승인 / ☐ 보류

## 후속 조치 (사장님 결단 후)
- ☐ paper cron 1줄 정식 등록 (제한적 가동, 전체 X)
- ☐ 1주 관찰 + 사고 0건 누적 시 다음 cron 1줄 추가 결단
- ☐ 사고 시 즉시 KILL_SWITCH + 사장님 호출
```

---

## 8. 표현 룰

### 사용 가능
- "§9 dry-run 실측 전 체크리스트 완성"
- "단계 N 통과" / "단계 N 중단"
- "Codex 재검수 회신 대기"

### 사용 금지
- "§9 통과" X (실측 전)
- "재가동 가능" X
- "운영 안전 완성" X
- "Phase 1 paper 재가동 가능" X

---

## 9. 적용 금지 (본 체크리스트 작성 후)

- ❌ 본 체크리스트 사전 조건 9건 미충족 시 실측 진입 X
- ❌ 단계 1~4 임의 순서 변경 X
- ❌ 단계 4 실측 사장님 별도 승인 없이 진행 X
- ❌ 단독 commit X

---

## 10. 연결 문서
- `docs/02-design/restart-dry-run-spec-5_29.md` (4단계 정식 명세)
- `docs/02-design/p1-truth-pack-5-29.md`
- `docs/02-design/p1-residual-plan-5-29.md`
- `docs/02-design/filelock-policy-5_29.md`
- `docs/02-design/hmac-rotation-playbook-5_29.md`
- `docs/02-design/p2-residual-4items-and-critical-fix_review-requested.md` (Codex 의뢰서 ②)
- `tools/quant_preflight.py` (확장 대상)
