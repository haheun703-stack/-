# Paper Cron Mode 명시 정책 (5/28 코덱스 17:26 결정)

> **상태**: 정책 v1 (cron 재가동 전 적용 필수)
> **계기**: 코덱스 17:26 — "*_MODE default=live는 운영 정책상 애매. paper cron 재가동 전에는 각 cron에 *_MODE=paper 명시 필요"

## 1. 환경변수 default 정책

| 환경변수 | 영향 호출자 | default (현재) | paper cron 재가동 시 |
|---------|------------|---------------|---------------------|
| `OWNER_RULE_MODE` | scripts/owner_rule_monitor.py | live | **paper 명시 필수** |
| `TELEGRAM_TRADING_MODE` | src/telegram_command_handler.py | live | (수동, paper 권장 시 명시) |
| `AUTO_BUY_EXECUTOR_MODE` | scripts/auto_buy_executor.py | live | **paper 명시 필수** |
| smart_entry mode | scripts/smart_entry_runner.py | dry_run 기반 자동 | dry_run=True → mode='paper' |
| run_adaptive_cycle mode | scripts/run_adaptive_cycle.py | --paper/--real 기반 | --paper 명시 |

## 2. Cron 등록 시 명시 (paper 재가동 시 예시)

### owner_rule_monitor

**올바른 형태** — env 변수를 python 실행 앞에 둘 것 (cd 앞에 두면 별도 셸로 분기되어 python 프로세스에 안 들어감).

```bash
*/5 9-15 * * 1-5 \
  cd /home/ubuntu/quantum-master && \
  OWNER_RULE_MODE=paper ./venv/bin/python3.11 scripts/owner_rule_monitor.py \
  >> /tmp/owner_rule_monitor.log 2>&1
```

**잘못된 형태 (env 전달 안 됨)**:

```bash
# ❌ OWNER_RULE_MODE=paper cd ... && python ... → env가 cd에만 영향, python 프로세스에 미전달
OWNER_RULE_MODE=paper cd /home/ubuntu/quantum-master && ./venv/bin/python3.11 ...
```

### auto_buy_executor

```bash
*/5 14 * * 1-5 \
  cd /home/ubuntu/quantum-master && \
  AUTO_BUY_EXECUTOR_MODE=paper \
  PYTHONPATH=/home/ubuntu/quantum-master \
  ./venv/bin/python3.11 -u -X utf8 scripts/auto_buy_executor.py \
  >> /home/ubuntu/quantum-master/logs/auto_buy.log 2>&1
```

### run_adaptive_cycle (--paper 그대로)

```bash
*/5 9-15 * * 1-5 \
  cd /home/ubuntu/quantum-master && \
  PYTHONPATH=/home/ubuntu/quantum-master \
  ./venv/bin/python3.11 -u -X utf8 scripts/run_adaptive_cycle.py --paper \
  >> /home/ubuntu/quantum-master/logs/adaptive_cycle.log 2>&1
```

## 3. quant + live 자동 차단 (Note 1 정합)

- 모든 퀀트봇 호출자(`executor_bot="quant"`)는 `mode="live"` 시 register 단계 거부
- 즉 default=live여도 매매 0건 (안전, fail-close)
- 단 운영자 인지 부재 위험 → docs/cron에 명시 권장

## 4. 텔레그램 매매 정책 (P0-2)

- `executor_bot="day"` (수동 매매는 day bot 권한)
- `TELEGRAM_TRADING_MODE` default=live → day+live 가능 (단타봇 권한 가정)
- paper 운영 권장 시 사용자가 명시 설정 (`export TELEGRAM_TRADING_MODE=paper`)

## 5. 코덱스 결단 요청

1. default=live vs paper 권장
   - default=live: 코덱스 정책 (현재) — quant+live 차단으로 fail-close
   - default=paper: 운영 안전성 (실수 방지) — 모든 cron에 mode=paper 명시 + 실거래 시 명시 변경
2. paper cron 재가동 시점에 명시 검증
   - quant_preflight에 cron 환경변수 검사 추가 권장?
3. day bot live intent 권한 정책 (telegram_command_handler)

## 6. 표현 금지 (코덱스 결정문)

- "전체 주문 경로 L10 완료" 사용 X
- "Phase 1 paper 재가동 가능" 사용 X
- 사용 가능: "P0 5건 호출자 mode/executor_bot 전달 완료"
- 사용 가능: "잔여 P1 호출자 + backward compat 폐지 미진행"

## 7. crontab 등록 — 보기용 vs 실제 (코덱스 5/28 19:11 정정)

위의 multi-line `\` 형식은 **가독성용 표시**일 뿐. 실제 crontab 등록은 **한 줄 엔트리**가 안전.

### 보기용 (docs/PR/리뷰 시 줄바꿈 OK)

```bash
*/5 9-15 * * 1-5 \
  cd /home/ubuntu/quantum-master && \
  OWNER_RULE_MODE=paper ./venv/bin/python3.11 scripts/owner_rule_monitor.py \
  >> /tmp/owner_rule_monitor.log 2>&1
```

### 실제 crontab 등록 (single-line)

```bash
*/5 9-15 * * 1-5 cd /home/ubuntu/quantum-master && OWNER_RULE_MODE=paper ./venv/bin/python3.11 scripts/owner_rule_monitor.py >> /tmp/owner_rule_monitor.log 2>&1
```

```bash
*/5 14 * * 1-5 cd /home/ubuntu/quantum-master && AUTO_BUY_EXECUTOR_MODE=paper PYTHONPATH=/home/ubuntu/quantum-master ./venv/bin/python3.11 -u -X utf8 scripts/auto_buy_executor.py >> /home/ubuntu/quantum-master/logs/auto_buy.log 2>&1
```

```bash
*/5 9-15 * * 1-5 cd /home/ubuntu/quantum-master && PYTHONPATH=/home/ubuntu/quantum-master ./venv/bin/python3.11 -u -X utf8 scripts/run_adaptive_cycle.py --paper >> /home/ubuntu/quantum-master/logs/adaptive_cycle.log 2>&1
```

**이유**: crontab은 백슬래시(`\`) 줄바꿈 처리가 일부 환경에서 다름. 한 줄 엔트리가 가장 호환성 높음.

## 8. 테스트 적용 범위 정정 (코덱스 5/28 19:11)

코덱스 검수 결과 표현 정정:

| 테스트 | 실제 범위 | 표현 정정 |
|--------|----------|----------|
| `test_smart_entry_buy_limit_calls_include_kwargs_static` | smart_entry.py 정적 regex 매치 (multi-line buy_limit 2곳) | **"실제 buy_limit 호출 2곳에 대한 정적 회귀"** (런타임 실제 매수 플로우 X) |
| `test_auto_buy_executor_buy_limit_call_runtime` | KisOrderAdapter mock + adapter.buy_limit kwargs 검증 스니펫 | **"env + adapter kwargs 전달 스니펫 검증"** (script main 전체 실행 X) |

## 9. Phase 1 paper cron 재가동 직전 추가 검증 (필수)

코드 회귀와 별도로 cron 재가동 직전:

1. **One-line crontab 등록 후 dry-run** — `bash -c "{crontab line}"` 직접 실행 + 로그 확인
2. **Smoke test** — register_intent 1건 + 매매 호출 시뮬레이션 → NoIntentError 또는 FILLED 확인
3. **환경변수 검증** — `env | grep -E "(OWNER_RULE|AUTO_BUY|TELEGRAM_TRADING|ORDER_INTENTS_HMAC)_MODE|_KEY"` 출력
4. **journalctl 모니터** — 첫 cron 실행 5분간 실시간 모니터 (`journalctl -u cron -f`)

이 4단계 통과 후 코덱스 검수 의뢰. 그 전까지 "Phase 1 paper 재가동 가능" 표현 사용 X.
