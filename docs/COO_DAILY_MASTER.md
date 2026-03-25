# COO DAILY MASTER — Quantum Master 운영 지시서
# 최종 업데이트: 2026-03-25
# 이 파일은 프로젝트 루트에 저장하고 절대 삭제하지 않는다.

---

## ⚠️ 최우선 원칙 (매일 반드시 읽을 것)

1. **COO가 모른다고 느끼면 멈추고 보고한다** — 추측으로 진행하지 않는다
2. **기존 파일 수정 전 반드시 백업** — _original 또는 _backup 접미사
3. **커밋은 단계별로** — 한꺼번에 몰아서 하지 않는다
4. **컨트롤타워(Claude.ai) 제안은 참고용** — 우리 코드 기준으로 판단한다
5. **SHIELD RED = 신규 진입 없음** — 예외 없음

---

## 📅 매일 실행 체크리스트

### 🌙 전일 야간 (자동 — 건드리지 말 것)

```
06:10  BAT-A   US 마감 데이터 (VIX, S&P500, NightWatch)
07:00  BAT-B   Morning 데이터 갱신
07:30  BAT-K_Safety  Safety Margin 사전 체크
07:55  BAT-M_NXT    NXT Pre-market
08:00  BAT-M   아침 브리핑 생성 + 텔레그램 발송
08:20  BAT-N   Signal Log
```

### 🌅 장 시작 전 (08:45) — 매일 수동 확인

```bash
python scripts/tomorrow_checklist.py
```

**확인 항목 8가지:**
- [ ] 1. BAT-D 어젯밤 정상 실행 여부
- [ ] 2. COO run_log 마지막 실행 결과 (7그룹 전부 OK?)
- [ ] 3. alpha_v2 현재 모드 (paper / live)
- [ ] 4. SHIELD 상태 (RED / YELLOW / GREEN)
- [ ] 5. BRAIN 결정 (레짐, 슬롯 수, confidence)
- [ ] 6. LENS 맥락 (공격 / 방어 / 관망)
- [ ] 7. scan_buy 통과 종목 수
- [ ] 8. 3조건 추적 (승률 / MDD / COO 실행 횟수)

**SHIELD RED 시 → 신규 진입 없음. 기존 포지션 손절선 확인만.**

### 🕘 장중 (09:30~15:20 — 자동)

```
08:50  BAT-E   SmartEntry (KILL_SWITCH 해제 전까지 OFF)
08:55  BAT-I   VWAP 모니터
08:55  BAT-K   IntradayEye (7개 감지기, 5분 주기)
09:05  BAT-P1  DaySignal
11:30  BAT-H   장중 AI 분석 + Market Pulse
14:00  BAT-G   FridayDip (금요일만)
15:20  BAT-P2  DayClose
```

### 🌆 장 마감 후 (16:30 — 자동, 절대 건드리지 말 것)

```
16:10  BAT-O   SignalTrack
16:30  BAT-D   ★ COO Orchestrator → 31단계 메인 파이프라인
               실패 시 BAT-D_original.bat 자동 폴백
17:00  BAT-J   Portfolio Outlook
17:30  BAT-F   Sniper Watch
```

### 🌙 저녁 (17:30 이후) — 매일 수동 확인

```bash
cat data/coo_run_log.json | python -c "
import json,sys
log = json.load(sys.stdin)
for g in log['groups']:
    status = '✅' if g['success'] else '❌'
    print(f\"{status} {g['group']}: {g.get('duration','?')}초\")
"
```

**문제 그룹 있으면 즉시 확인 — 다음날 BAT-D 전에 반드시 해결**

---

## 🔄 KILL_SWITCH 해제 3조건 (매일 추적)

| 조건 | 기준 | 현재 상태 | 충족 여부 |
|------|------|-----------|-----------|
| 1. Paper 승률 | 55% 이상 | 데이터 쌓이는 중 | ⏳ |
| 2. Paper MDD | -8% 이내 | 백테스트 -8.24% 기확인 | 🟡 근접 |
| 3. COO 실전 실행 | 10회 이상 | 2026-03-26 첫 실행 | ⏳ |

**3조건 동시 충족 시 → SmartEntry KILL_SWITCH 해제**
해제 방법:
```bash
del data/KILL_SWITCH
# 그 다음 settings.yaml:
# smart_entry.enabled: true
```

---

## 🔧 현재 토글 상태 (함부로 바꾸지 말 것)

```yaml
# settings.yaml 핵심 토글
alpha_v2:
  enabled: true           # ✅ 2026-03-25 활성화
  mode: "paper"           # paper → live 전환은 3조건 후
  lens_enabled: true      # ✅ 2026-03-25 활성화

execution_alpha:
  enabled: false          # EX-1~7 구현완료, KILL_SWITCH 해제 후 활성화

intraday_eye:
  enabled: true           # 장중 감시 가동 중

smart_entry:              # KILL_SWITCH가 막고 있음
  # data/KILL_SWITCH 파일 삭제 = 자동매수 활성화
```

---

## 📊 업그레이드 완료 현황 (건드리지 말 것)

### ✅ 완료 — 안정 운영 중

| 모듈 | 커밋 | 내용 |
|------|------|------|
| V2 스코어링 | STEP 0~6 | 레짐별 동적 가중치, PF 1.38 검증 |
| LENS Layer | 387d16a | 4렌즈 맥락, 비파괴 확인 |
| SHIELD SW-1~7 | 68667d4 | 레짐별 손절/공포매수/분할진입 |
| 거래일 가드 | ee81911 | 19개 BAT 전체 통일 |
| COO Orchestrator | 9631919 | 7그룹 66단계, 폴백 체계 |
| FLOWX 메시지 | b3a4a1d | 방어/공격/관망 자동 판별 |
| BAT-D 실전 연결 | 3020167 | COO 래핑, 원본 백업 |
| tomorrow_checklist | 3020167 | 8항목, 텔레그램 발송 |

### ⏳ 진행 중 — Paper 데이터 쌓이는 중

| 항목 | 조건 | 예상 시점 |
|------|------|-----------|
| V2 live 전환 | 3조건 충족 | 데이터 쌓이면 |
| KILL_SWITCH 해제 | 3조건 동시 충족 | 데이터 쌓이면 |
| Execution Alpha | KILL_SWITCH 후 | 단계적 |

### 📋 다음 작업 — COO STEP 3 이후

```
STEP 3  BAT-D 첫 실전 로그 검증 (2026-03-26 ~)
STEP 4  V2 + LENS 순차 최적화 (로그 기반)
PHASE 4 LLM Scenario Pipeline (JARVIS 연동)
```

---

## 🚨 긴급 상황 대응

### BAT-D 실패 시
```bash
# 즉시 원본으로 롤백
copy bat\BAT-D_original.bat bat\BAT-D.bat
# 오류 확인
cat data/coo_run_log.json
# 컨트롤타워(Claude.ai)에 로그 공유
```

### SHIELD 갑자기 RED 시
```
1. 신규 매수 중단
2. 기존 포지션 손절선 전부 확인
3. tomorrow_checklist.py 실행해서 원인 확인
4. 원인 파악 전까지 모든 자동 매수 OFF
```

### COO 7그룹 중 하나 실패 시
```
critical 그룹(데이터수집/지표계산/시그널엔진) 실패
→ FLOWX에 "데이터 불완전" 자동 발행됨
→ 당일 신규 진입 없음
→ 다음날 BAT-D 전에 해당 그룹 수동 테스트

non-critical 그룹 실패
→ 폴백 데이터로 자동 진행됨
→ coo_run_log.json에서 폴백 사유 확인
→ 3일 연속 같은 그룹 실패 시 컨트롤타워 보고
```

---

## 📝 매주 금요일 — 주간 점검

```bash
# 1. 이번 주 COO 실행 성공률
# 2. Paper Trading 성과 확인
# 3. KILL_SWITCH 3조건 진척도
# 4. 실패한 그룹 패턴 있는지
# 5. FLOWX 메시지 정확도 (방어/공격/관망이 실제 시장과 맞았나)
```

결과를 컨트롤타워(Claude.ai)에 공유하면 다음 최적화 지시 받을 수 있음.

---

## 💬 컨트롤타워(Claude.ai) 보고 기준

**즉시 보고:**
- COO 3일 연속 같은 그룹 실패
- SHIELD RED 3일 이상 지속
- Paper MDD -10% 초과

**주간 보고:**
- 주간 Paper Trading 성과
- 3조건 진척도
- 이상하다고 느끼는 것 전부

**보고 시 반드시 포함:**
- coo_run_log.json 최근 3일치
- data/paper_portfolio.json
- SHIELD 현재 상태

---

*이 파일은 프로젝트 루트 + docs/ 두 곳에 저장할 것*
*절대 삭제하지 않는다. 업데이트 시 날짜 기록.*
