# 서버 배포 + E-0 배선 실행 계획 — 2026-06-18

작성: 2026-06-18 (목) · 상태: **GO·15:40 수동배포 대기** (사장님 결정) · 연계: [unfreeze-checklist.md](unfreeze-checklist.md) §2-E
> ★6/18 정정: E-0 배선 완료(`617fcf0`) · 자동화 정본=cron(scheduler아님)·restart 금지 · 윈도우 15:40~16:05.
>
> **결정 1줄**: freeze→unfreeze 경로의 첫 마일스톤. 서버 HEAD `12ca898`(6/10) → 현재 `242a8a2` 전체 pull(63커밋)
> = 리스크엔진·실행게이트 **production 첫 배포** + 페이퍼 게이트 드라이런(E-0) 가동 → **20거래일 페이퍼 시계 시작**.
> 실주문은 KILL_SWITCH(5/19)가 계속 차단. 이건 "실탄 발사"가 아니라 "안전장치 가동 + 증거 수집 시작".

---

## 0. 핵심 재인식 (아침에 정정된 것)

- ❌ 당초 "저녁에 버튼만 누르면 됨" — **틀림.**
- ✅ **E-0는 반나절급 코딩**이다: 실제 도는 페이퍼 엔진(`scripts/paper_trading_unified.py`, 1883줄)이
  `gate_check`를 **미경유**(gate/balance/ohlcv 참조 0). 이 배선이 깔려야 `gate_log_*.jsonl`이 쌓이기 시작.
- ✅ 따라서 순서: **① E-0 배선 코딩(낮·페이퍼·드라이런=freeze 양립) → ② 저녁 배포(E-0 포함 pull) → ③ gate_log 적재 확인.**
  세 개를 동시에 안 하면 "E가 왜 0?"을 또 재발견(6/17 오전 반복). [unfreeze-checklist §2-E line 102]

---

## 1. 선행 의존 (둘 다 충족돼야 저녁 배포 GO)

| # | 선행조건 | 담당 | 상태 |
|---|----------|------|------|
| P1 | **KRX_PW 갱신** (연기금/금융투자 bulk 복구·서버 .env) | 사장님 (KRX 사이트 비번 확인) | ⏳ 대기 |
| P2 | **E-0 배선 코딩 + 로컬 테스트** (페이퍼 엔진 → gate_check) | 나 (낮) | ✅ 완료 (`617fcf0`, 오전) |

- P1이 없어도 배포 자체는 가능하나, 연기금/금융투자 bulk가 계속 비어 "3종 동시"가 반쪽. **권장=P1 먼저.**
- P2는 freeze 양립(페이퍼 엔진은 자체 시뮬·실주문 경로 아님). 게이트는 **드라이런 로그만** 남김(enforce=False).

---

## 2. E-0 배선 설계 (P2 — 낮 작업)

**목표**: 페이퍼 엔진의 매 가상매수 시도 → `gate_wiring.gate_check` 경유 → `gate_log_*.jsonl`에 GATE-DRYRUN 기록.

**배선 지점**: `paper_trading_unified.py`의 가상 체결 직전 (매수 후보 확정 후).

**어댑팅 필요 3종** (페이퍼 엔진엔 어댑터/잔고포트 없음):
1. `balance_port.fetch_balance()` → 페이퍼 잔고(현금+보유)로 equity 합성하는 얇은 셰임(shim).
2. `ohlcv_loader`(adv20) → 페이퍼 엔진이 이미 쓰는 pykrx OHLCV 재사용(line 241) → adv20 계산.
3. **enforce 모드** = `False`(드라이런): REJECT/RESIZE를 **기록만** 하고 가상체결은 그대로 진행
   (페이퍼는 게이트 거동을 *관측*하는 게 목적·차단이 목적 아님). 단 로그엔 verdict 전부 남김.

**불변식 보존**: 실매매 6호출처(smart_entry·adaptive_*·limit_up·live_trading·telegram)는 **무접촉**.
페이퍼 엔진에만 드라이런 배선 추가. 회귀 신규실패 0 목표.

**검증**: `tests/test_paper_gate_dryrun.py`(신규) — 페이퍼 매수 1건이 gate_log에 GATE-DRYRUN 1행 남기는지 +
enforce=False라 REJECT여도 가상체결 진행되는지 + 기존 페이퍼 테스트 무손상.

> ⚠️ 시장가 사이즈 바인딩(6/13 적대검증 P2)·chart_hero 재배선(D)은 E-0 범위 밖 — 실탄 시장가 켜기 직전 처리.

---

## 3. 배포 실행 절차 (★6/18 정정: cron 정본·scheduler restart 금지)

> ⚠️ **장중(09:00~15:30) 금지** + 페이퍼 cron(16:30 D·17:00 J) 전 권장 → **윈도우 15:40~16:05** (15:35 L 장마감 직후~16:10 O 전).
> ★**자동화 정본 = cron(`scripts/cron/run_bat.sh`)**. `quantum-scheduler.service`는 5/27부터 의도적 inactive. **배포 = `git pull`만**, restart 치지 말 것(cron 이중실행·freeze 위반). pull 후 다음 cron부터 새 코드 자동 적용.

```
# 0) 배포 전 서버 스냅샷 (롤백 대비)
ssh ... "cd ~/quantum-master && git rev-parse HEAD && git status --short"

# 1) 로컬 산출물·백업 임시보관 (brain_data_upload.json M 등 — incoming diff엔 없어 충돌 안나나 안전차원)
ssh ... "cd ~/quantum-master && git stash push -u"

# 2) 서버 pull (63커밋: 리스크엔진 Phase2 VaR·gate_wiring·밸류밴드·2층·collect_kis·sector_fire fix·E-0·확신모델 설계서)
ssh ... "cd ~/quantum-master && git pull origin main"   # → 242a8a2

# 3) 검증① 의존성 (리스크엔진 신규 import: 서버 venv 누락 점검) — 배포 직후 즉시 가능
ssh ... "cd ~/quantum-master && ./venv/bin/python3.11 -c 'import risk; from risk import gate_wiring, var_engine' 2>&1"

# ★ systemctl restart 없음. 검증②③은 저녁 cron(16:30 D·17:00 J) 후 §4 참조.
```

---

## 4. 배포 후 검증 (3종 — 하나라도 빠지면 재발견 함정)

> ★**[6/22 재검증 — KRX 사태로 묻혔던 §4를 사무실에서 추적]**: 6/18 계획서가 "저녁 배포"로 짰으나
> 실제 배포는 6/19 18:58(`5cfb45b`). 그런데 **6/19 페이퍼 cron이 18:38:52에 먼저 실행**(reflog 대조)
> → 그날 페이퍼는 E-0 없는 옛 코드(`12ca898`)로 돌아 gate_log 0. 이후 KRX 차단 사태(6/19~22)로
> §4 검증을 아무도 안 봐서 "E-0=완료"라 착각된 채 묻혔음. 6/22 08:11 `de03137`(E-0 포함) 재배포됨.

- [ ] **(1) gate_log 적재 시작** — ⏳ **오늘(6/22) 16:30 D cron 페이퍼 후 확인 예정**. ★배선 코드 자체는
  6/22 서버 격리 재현으로 **정상 확증**(`run_paper_gate_dryrun`→verdict=PASS·`/tmp/gate_test`에
  `gate_log_20260622.jsonl` 1182B 생성, 실카운트 경로 무접촉). 단 실카운트 경로
  `data/risk/gate_logs_paper`는 6/19 타이밍 누락 이후 **아직 빈 디렉토리 미생성** → 오늘 16:30 페이퍼가
  신규매수 1건 이상 하면 첫 줄 적재 = **E 20거래일 카운트 D-day**. (매수 0인 날은 line 899 미도달 → 그날 미카운트.)
- [x] **(2) 서버 배포 효력** ✅ (6/22) — `git rev-parse HEAD`=`de03137`(⊃ E-0 `617fcf0`, merge-base 조상 확인).
  서버 venv `import risk; from src.use_cases import paper_gate, gate_wiring` **전부 성공**.
- [ ] **(3) sector_fire 6/16 재업로드** — fix `93233d2`(_drop_nonfinite_floats) 효력으로 6/16 NaN FAIL 해소되는지(서버 재실행).
- [ ] 부가: KILL_SWITCH 여전히 active(실주문 0 불변)·quantum-scheduler 정상·기존 BAT-D 9테이블 무손상.

---

## 5. 롤백 계획

- 배포 후 import 실패/cron 크래시 시: `ssh ... "cd ~/quantum-master && git reset --hard 12ca898"` — **reset만**. 다음 cron이 옛 코드로 돎(restart 불필요).
- data/risk·gate_log은 신규 생성물이라 롤백해도 기존 데이터 무손상.
- E-0는 페이퍼 전용이라 롤백해도 실매매 경로 영향 0.

---

## 6. 불변식 영향 명시

| 불변식 | 배포 후 |
|--------|---------|
| 실주문 0 | **유지** (KILL_SWITCH active) |
| 매매로직 0 변경 | **유지** (E-0는 페이퍼 드라이런·실매매 6호출처 무접촉) |
| production 배선 0 | **의도적으로 깸** (리스크엔진 첫 배포 = unfreeze 경로 시작) |
| freeze | **유지** (unfreeze는 E 20거래일 완료 후 별도 결정) |

---

## 7. 타임라인 (오늘)

| 시각 | 작업 | 담당 |
|------|------|------|
| 오전 | E-0 배선 코딩 + 로컬 테스트 (P2) ✅ `617fcf0` | 나 |
| 병행 | KRX_PW 갱신 (P1) | 사장님 |
| **15:40~16:05** | **배포 실행 (§3 `git pull`만) + 검증①(import)** | 나 (노트북 ON) |
| 16:30 D·17:00 J cron 후 | 검증②(gate_log 적재)·③(sector_fire 6/16) | 나 |

> freeze 유지. 이 배포는 unfreeze가 아니라 **unfreeze까지의 20거래일 시계를 켜는 행위**.
