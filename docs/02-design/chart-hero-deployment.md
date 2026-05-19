# 차트영웅 매매 시스템 — VPS 배포 가이드

**작성일**: 2026-05-19
**가동 목표**: 2026-05-22(금) 09:00 (paper mirror)
**실전 목표**: 2026-05-27(화) 사장님 GO 후

---

## 1. 배포 전 체크리스트

### 로컬 검증 (완료)
- [x] `kis_weekly_kit` 종목+지수 주봉 ✅
- [x] `four_signal_gate` 4-시그널 합성 ✅
- [x] `chart_hero_advisory` 역발상 advisory ✅
- [x] `chart_hero_tension_rule` 긴장 룰 ✅
- [x] `analyst_target_collector` + 시드 CSV ✅
- [x] `perplexity_catalyst` ✅
- [x] `surge_d1_picker` 5-Gate ✅
- [x] `d1_confirm` D+1 양봉 확인 ✅
- [x] `chart_hero_executor` paper/real 모드 + 고가주 필터 ✅
- [x] 스케줄러 2개 (close_cycle + morning_monitor) ✅
- [x] 텔레그램 알림 ✅
- [x] 백테스트 골격 ✅

### 5/20(수) 작업 예정
- [ ] Gate 2 Supabase quant_surge_pullback 직접 조회
- [ ] Gate 3 MA20 dev 계산 통합
- [ ] 정보봇 응답 통합 (Gate 4-A catalyst)
- [ ] VPS 배포 + cron 등록

### 5/21(목) 작업 예정
- [ ] 6개월 백테스트 실제 데이터 실행
- [ ] 원본 vs 긴장 타입 비교 리포트

---

## 2. VPS 배포 절차

### 2-1. 코드 동기화
```bash
# VPS SSH 접속
ssh -i "D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem" \
    -o ConnectTimeout=10 ubuntu@13.209.153.221

# 코드 pull
cd ~/quantum-master
git pull origin main

# 로그 확인 (최근 commit 12개 적용)
git log --oneline -12
```

### 2-2. 의존성 확인
```bash
source venv/bin/activate
pip install youtube-transcript-api yfinance  # 5/19 추가됨
pip list | grep -E "yfinance|requests|pandas"
```

### 2-3. 환경 변수 검증
```bash
# .env에 필수 키 확인
cd ~/quantum-master
cat .env | grep -E "PERPLEXITY_API_KEY|KIS_APP_KEY|TELEGRAM" | \
    awk -F'=' '{print $1 "=***"}'
```

### 2-4. 자가 검증 (각 모듈)
```bash
# 매크로 4-시그널
python -u -X utf8 -m src.macro.four_signal_gate

# 5-Gate 종목 선정 (5/19 결과: GATE1_BLOCKED 정상)
python -u -X utf8 scripts/surge_d1_picker.py

# 마감 사이클 (paper, force 무시)
python -u -X utf8 scripts/chart_hero_close_cycle.py --paper --force

# 모니터 (보유 없음 → 정상)
python -u -X utf8 scripts/chart_hero_morning_monitor.py --paper
```

### 2-5. cron 등록
```bash
crontab -e
```

추가:
```cron
# === 차트영웅 매매 시스템 (5/22 paper mirror 가동) ===
# 09:30 보유 포지션 모니터 (추매/익절/손절)
30 9 * * 1-5 cd ~/quantum-master && ./venv/bin/python3.11 scripts/chart_hero_morning_monitor.py --paper >> /tmp/chart_hero_morning.log 2>&1

# 14:55 마감 사이클 (5-Gate + D+1 양봉 + 진입)
55 14 * * 1-5 cd ~/quantum-master && ./venv/bin/python3.11 scripts/chart_hero_close_cycle.py --paper >> /tmp/chart_hero_close.log 2>&1
```

### 2-6. 첫 가동 검증 (5/22 09:30)
```bash
# 09:30 cron 실행 후
tail -50 /tmp/chart_hero_morning.log
cat data/logs/chart_hero/morning_monitor_2026-05-22.json
```

---

## 3. 5/22~5/26 paper mirror 일정

| 일자 | 09:30 모니터 | 14:55 사이클 | 사장님 확인 |
|---|---|---|---|
| 5/22(금) | 보유 0 → 종료 | 매크로 게이트 + 시드 데이터 활용 | 일일 결과 텔레그램 |
| 5/23(토) | (휴장) | (휴장) | - |
| 5/24(일) | (휴장) | (휴장) | - |
| 5/25(월) | 5/22 진입분 모니터 | 새 진입 후보 | 누적 PnL 검토 |
| 5/26(화) | 모니터 | 매크로 + 정보봇 catalyst 통합 | 1주 누적 검토 |

### 5/27(화) 결단 회의 자료
- 1주 paper mirror WR/PF/MDD
- 차트영웅 원본 vs 긴장 타입 (백테스트 비교)
- 실전 진입 사이즈 결단 (1주 ~ 5주 ~ 10주)

---

## 4. paper → real 전환 절차

사장님이 5/27 GO 결단 후:

```bash
# crontab 수정 (--paper → --real)
crontab -e
```

```cron
30 9 * * 1-5 ... chart_hero_morning_monitor.py --real --capital 25000000 ...
55 14 * * 1-5 ... chart_hero_close_cycle.py --real --capital 25000000 ...
```

**중요**: real 모드 첫날은 사장님이 직접 09:30 / 14:55 로그 실시간 확인 권장.

---

## 5. 롤백 절차 (긴급)

### 5-1. cron 즉시 정지
```bash
crontab -l > /tmp/cron_backup_$(date +%Y%m%d).txt
crontab -r  # 또는 chart_hero 라인만 # 처리
```

### 5-2. 보유 포지션 강제 청산
```bash
# data/chart_hero_positions.json 확인
cat data/chart_hero_positions.json

# real 모드라면 KIS HTS/MTS에서 수동 청산
# paper 모드는 파일 삭제만으로 OK
rm data/chart_hero_positions.json
```

### 5-3. 텔레그램 알림
사장님께 알림: "차트영웅 매매 중지 + 사유"

---

## 6. 모니터링 대시보드

매일 확인 항목:
- `/tmp/chart_hero_morning.log` (09:30 결과)
- `/tmp/chart_hero_close.log` (14:55 결과)
- `data/logs/chart_hero/*.json` (상세 로그)
- 텔레그램 알림 (사장님 단톡방)
- `data/chart_hero_positions.json` (현재 포지션)

---

## 7. 비상 연락처 & 룰

### 시스템 문제 시
1. cron 정지 → 코드 수정 → 재가동
2. 사장님께 텔레그램 보고
3. 다음날 새벽 패치

### KIS API 장애 시
- KIS 토큰 만료 → 자동 갱신 (kis_nxt_kit `get_token`)
- 화이트리스트 차단 → VPS IP 13.209.153.221 등록 확인

### Perplexity 비용 폭주 시
- catalyst 자동 분석 비활성화 (`run_picker(analyze_catalyst_live=False)`)
- 정보봇 데이터로 대체

---

## 8. 메모리 참조

- `project_chart_hero_quant_5_19.md` 핵심 결단 (옵션 A 100%, 5-Gate, 긴장 룰)
- `project_chart_hero_independence.md` 단타봇 advisory 격리 원칙
- `feedback_kis_api_direct_check.md` KIS API 직접 호출 원칙
- `feedback_no_wrapup_suggest.md` 마무리 제안 금지

---

**핵심 원칙 재확인**:
- 차트영웅 매매 코드 = `four_signal_gate`, `chart_hero_advisory`, `chart_hero_tension_rule` 만 참조
- 단타봇 advisory, brain.py PANIC 모드, regime BEARISH **절대 import 금지**
- 모순 발생 시 → 격리 우선
