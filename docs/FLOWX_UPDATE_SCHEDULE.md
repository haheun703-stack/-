# FLOWX 웹봇 데이터 업데이트 스케줄

> 퀀트봇 → Supabase → FLOWX 웹봇 데이터 파이프라인 전체 정리
> 최종 검증: 2026-04-07 (BAT-D 로그 기반)

---

## 1. 전체 아키텍처

```
퀀트봇 (VPS BAT-D)
  ├─ scan_*.py / use_cases/*.py  → data/*.json (로컬 생성)
  ├─ upload_flowx.py             → Supabase upsert (G3 + G4)
  └─ flowx_uploader.py           → 빌더 함수 + 업로드 로직

Supabase (wkvrhawzzmocpxjesvir)
  ├─ quant_*       테이블 (퀀트봇 전용)
  ├─ dashboard_*   테이블 (정보봇/웹봇 공용)
  └─ intelligence_* 테이블 (공통)

FLOWX 웹봇 → Supabase SELECT → 화면 표시
```

## 2. BAT 스케줄 (일별)

| BAT | 시각 | FLOWX 관련 작업 |
|-----|------|-----------------|
| **BAT-A (모닝)** | 06:10 | 모닝 브리핑 업로드, 시그널 기록 (QUANT PICK) |
| **BAT-I (장중)** | 08:55 | VWAP/장중 모니터링 (FLOWX 직접 업로드 없음) |
| **BAT-D (장후)** | 16:30~17:30 | **메인 업로드** — 11 테이블 + ETF/외국인/시나리오 |
| **BAT-HEALTH** | 18:00 | Supabase 오늘 데이터 확인 (검증만) |

## 3. BAT-D FLOWX 업로드 상세 (메인)

### 실행 구조
BAT-D에서 `upload_flowx.py`가 **2회** 실행됨:
- **G3** (~17:15): 1차 업로드 (picks 141건)
- **G4** (~17:27): 2차 업로드 (최종 picks 97건, 자비스 반영)

### 업로드 테이블 (11개)

| # | 테이블명 | Supabase 테이블 | 건수 (4/7 기준) | 상태 |
|---|---------|----------------|----------------|------|
| 1 | 시장브레인 | `quant_market_brain` | 1건 (4,799자) | OK |
| 2 | 업종수급 | `quant_sector_flow` | 1건 (379,406자) | OK |
| 3 | 업종모멘텀 | `quant_sector_momentum` | 1건 (6,761자) | OK |
| 4 | 섹터로테이션 | `quant_sector_rotation` | 20행 | OK |
| 5 | ETF자금흐름 | `quant_etf_fund_flow` | 1건 (5,781자) | OK |
| 6 | ETF추천 | `quant_etf_recommendation` | 1건 (2,119자) | OK |
| 7 | 스마트머니 | `quant_smart_money` | 50행 | OK |
| 8 | ETF시그널 | `quant_etf_signals` | 20행 | OK |
| 9 | 스나이퍼 | `dashboard_sniper` | 30행 | OK |
| 10 | **급락반등** | `dashboard_crash_bounce` | **30행** | **OK** |
| 11 | 릴레이 | `dashboard_relay` | 0건 | **조건부** |

### 추가 업로드 (G3/G4 외)

| 항목 | Supabase 테이블 | 건수 | 시점 |
|------|----------------|------|------|
| ETF 시그널 | `quant_etf_signal_*` | 20건 | G3/G4 |
| AI 추천 (내일 PICK) | `quant_picks` | 97~141건 | G3/G4 |
| 외국인 자금 | `quant_foreign_flow` | 31건 | G3/G4 |
| 시나리오 대시보드 | `dashboard_scenario` | 4 시나리오 | G3/G4 |
| 자비스 컨트롤타워 | `quant_jarvis` | 25종목 | G4 |
| Paper 매매기록 | `paper_trades` | BUY 3건 | G4 이후 |

## 4. 급락반등 포착기 (crash_bounce) 상세

### 데이터 흐름
```
scan_crash_bounce.py
  → data/crash_bounce_scan.json (30건)
  → upload_flowx.py → build_crash_bounce_rows()
  → Supabase dashboard_crash_bounce (upsert, PK: date+ticker)
```

### 업데이트 주기
- **매일 장마감 후 1회** (BAT-D G3 ~17:15)
- DDL: `dashboard_crash_bounce` (date, ticker, name, close, change_pct, gap_20ma, bb_position, volume_ratio, foreign_net, inst_net, foreign_days, inst_days, signal_type, grade, score, reasons)
- RLS: `anon_read` 정책 활성화

### 시그널 유형
| 유형 | 조건 | 백테스트 |
|------|------|---------|
| 복합급락 반등 | 볼린저 + 거래량 동시 | BEST (PF 2.64) |
| 볼린저급락 반등 | 20MA -15% + BB 하단이탈 + 수급 | +3.38%, 승률 60.2% |
| 거래량폭발 반등 | 20MA -15% + Vol 3x + 수급 | +3.31%, 승률 62.5% |
| 관심 | 이격도 -12%~-15% (조건 근접) | — |

### 등급 체계 (FLOWX 표현 변경 후, 4/5~)
| 등급 | 아이콘 | 조건 |
|------|--------|------|
| 강력 포착 (구: 적극매수) | ★ | 복합 시그널 또는 쌍끌이 수급 |
| 포착 (구: 매수) | ◎ | 단일 시그널 + 기관/외인 |
| 관심 | ○ | 조건 근접 |

### handoff 스펙 문서
- `website/data/handoff_crash_bounce_tab.json` — DDL, 컬럼 정의, UI 스펙, 백테스트 통계 포함
- 이 파일은 **스펙 문서**이며 매일 갱신되는 데이터가 아님 (Supabase가 실제 데이터)

## 5. relay 테이블 실패 원인

`relay=False`는 **에러가 아닌 정상 동작**:
- `build_relay_rows()`가 `group_relay_today.json`의 `fired_groups`를 읽음
- 릴레이 엔진에서 활성 그룹이 없으면 → 0건 → upsert 스킵 → `False`
- 4/7 기준: AI반도체(Phase4), 방산(Phase4) 활성이지만 래거드 매칭 0 → 정상

## 6. 검증 체크리스트 (장날 매일)

### 자동 검증 (BAT-HEALTH 18:00)
- `[FLOWX] Supabase 오늘 데이터 확인 — 정상` 로그 확인

### 수동 검증 (필요시)
```bash
# VPS에서 FLOWX 로그 확인
grep '\[FLOWX\]' logs/cron_$(date +%Y%m%d).log

# 급락반등 건수 확인
grep '급락반등' logs/cron_$(date +%Y%m%d).log

# 전체 업로드 결과 확인
grep '10/11\|11/11' logs/cron_$(date +%Y%m%d).log
```

## 7. 4/7 검증 결과

| 항목 | 결과 |
|------|------|
| BAT-A 모닝 (06:10) | OK — 17건 QUANT PICK 시그널 기록 |
| BAT-D G3 (17:15) | OK — 10/11 테이블 업로드 |
| BAT-D G4 (17:27) | OK — 10/11 테이블 업로드 (최종) |
| 급락반등 | OK — 30건 업로드 |
| 자비스 | OK — 25종목 업로드 |
| Paper Trading | OK — BUY 3건 (하이록코리아, HMM, AJ네트웍스) |
| BAT-HEALTH (18:11) | OK — Supabase 데이터 정상 확인 |
| relay | 0건 (래거드 매칭 없음, 정상 동작) |
| BAT-A 미장분석 (18:00) | FAIL — pykrx utf-8 디코드 에러 (수동 개입 필요) |
