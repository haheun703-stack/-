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

```bash
*/5 9-15 * * 1-5 \
  OWNER_RULE_MODE=paper \
  cd /home/ubuntu/quantum-master && ./venv/bin/python3.11 scripts/owner_rule_monitor.py \
  >> /tmp/owner_rule_monitor.log 2>&1
```

### auto_buy_executor

```bash
*/5 14 * * 1-5 \
  AUTO_BUY_EXECUTOR_MODE=paper \
  cd /home/ubuntu/quantum-master && \
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
