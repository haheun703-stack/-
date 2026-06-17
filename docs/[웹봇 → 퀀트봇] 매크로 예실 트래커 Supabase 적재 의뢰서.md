# [웹봇 → 퀀트봇] 매크로 예실(예상 vs 실제) 트래커 Supabase 적재 의뢰서

- **작성일**: 2026-06-09 (화) KST
- **발신**: FLOWX 웹봇 운영
- **수신**: 퀀트봇 (quantum-master) — 미국 지표 / 정보봇 협업 — 한국 지표
- **배경**: 사장님이 "매크로 예실관리 트래커"(서규하 스펙)를 flowx에 통합 지시. 웹측 검수 + 아키텍처 확정 + 웹 UI 선구현 완료.

---

## 0. 결론 먼저 (아키텍처 확정)
- 스펙의 **FastAPI(app.py)·DuckDB는 버립니다.** flowx는 Supabase만 읽음. 독립 서버 무의미.
- 스펙의 **예실 로직(나우캐스트·선물내재·예실 score)만 흡수** → 기존 **퀀트봇 `src/macro` 확장**으로 Supabase 직적재.
- **미국 지표 = 퀀트봇**(이미 cpi_tracker·macro_aggregator 보유) / **한국 지표(BOK 등) = 정보봇** 강점.
- 웹은 이미 완료: `/economic-calendar` 페이지 + `/api/macro/forecast-actual`. **데이터만 오면 자동 표출**(현재 "적재 대기중" 표시).

## 1. ⚠️ 컨센서스 충돌 해소 (중요)
스펙은 "컨센서스 자동수집 금지(유료/라이선스)"인데 사장님은 "봇 자동 적재" 지시 → 충돌.
**해소책**: 유료 컨센서스(블룸버그/로이터) 대신 **무료 프록시 자동 적재** —
- CPI/PCE → **클리블랜드 연준 나우캐스트**
- FOMC → **Fed Funds 선물 내재 확률**(yfinance)
- 그 외 → 봇 자체 나우캐스트/모델
- **`consensus_source`에 출처 명기 + 라벨은 "시장 추정치"** (절대 "공식 컨센서스"라 하지 말 것)

## 2. Supabase 테이블 (웹이 마이그레이션 SQL 작성 완료)
- 파일: flowx repo `supabase/migrations/20260609_macro_forecast_actual.sql`
- **사장님이 Supabase SQL Editor에서 1회 실행** (DDL은 보안상 그 경로만)
- 스키마:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `indicator_code`* | text | FOMC, CPI_HEAD, CPI_CORE, NFP, UNRATE, PCE, PPI, ISM_MFG, RETAIL, GDP, JOLTS / **BOK_RATE, KR_CPI, KR_EXPORT**(한국) |
| `event_date`* | date | 지표 귀속/발표 기준일 (PK 일부) |
| `region` | text | 'US' \| 'KR' |
| `indicator_name_ko` | text | 한글명 ('미국 소비자물가', '한국 기준금리') |
| `impact` | smallint | 1~5 별점 |
| `frequency` | text | monthly \| quarterly \| per-meeting \| weekly |
| `event_datetime_kst` | timestamptz | 발표 시각 KST (미국은 **DST 반영** 21:30/22:30) |
| `consensus` | numeric | 시장 추정치(나우캐스트/선물) |
| `consensus_source` | text | 'cleveland_nowcast' \| 'fed_funds_futures' \| 'model' \| 'bok' \| 'kostat' |
| `actual` | numeric | 실제 발표 (**발표 전 null**) |
| `prior` | numeric | 직전치 |
| `surprise` | numeric | actual − consensus |
| `surprise_score` | numeric | 예실 점수 (스펙 score 공식) |
| `stance` | text | '긴축' \| '완화' \| '중립' |
| `market_impact` | text | '악재' \| '호재' \| '중립' (**증시 영향** 기준) |
| `unit` | text | '%' \| '천명' \| 'index' \| '%p' |
| `note` | text | |

(*) = PK `(indicator_code, event_date)` → upsert 안전.

## 3. ⚠️ 한국 색상 규칙 (웹 UI 이미 반영)
`market_impact`는 **증시 영향 기준**으로 채워주세요 (수치 방향 아님):
- 인플레 상회=긴축압력=증시 **악재** → 웹에서 **파랑**
- 인플레 하회/완화=증시 **호재** → 웹에서 **빨강**
- `stance`(긴축/완화)와 `market_impact`(악재/호재)는 별개 컬럼. 둘 다 채워주세요.

## 4. 적재 범위 분담
- **퀀트봇(미국)**: FOMC·CPI·NFP·실업률·PCE·PPI·ISM·소매·GDP·JOLTS — BLS/BEA/FRED + 나우캐스트/선물
- **정보봇(한국)**: BOK 기준금리·한국 CPI·수출입·산업생산 — 한은/통계청
  - 보너스: `BOK_RATE`의 최신 `actual` 값은 flowx 매크로 요약 "기준금리" 카드(`macro_dashboard.bok_base`)와도 연결됨 — 현재 그 카드가 비어있음(미국 폴백 중). 한국 기준금리 적재 시 자동 한국 전환.

## 5. 적재 주기
- 발표 **전**: `consensus`/`prior`/`event_datetime_kst` 미리 채움 (`actual` null)
- 발표 **후**: `actual`·`surprise`·`surprise_score`·`stance`·`market_impact` 업데이트 (같은 PK upsert)
- 미국 지표는 한국시간 새벽 발표 → DST 스케줄러 주의(스펙 TODO에 있음)

## 6. 웹측 완료분 (참고)
- `/economic-calendar` 페이지: 🇺🇸/🇰🇷 필터 + 발표예정(D-day) + 발표완료(서프라이즈) + 면책
- `/api/macro/forecast-actual`: 테이블 없어도 graceful(빈배열) — 적재 즉시 표출
- 면책 멘트(예측 아님·리스크관리·매매신호 금지) 페이지에 명시

---

**요청**: 위 스키마대로 적재 시작 부탁드립니다. 한국 지표는 정보봇과 협업분. 적재되면 웹측 `/api/health` + 페이지 확인하겠습니다. 🙏

---

## 퀀트봇 회신 (2026-06-09 KST)

**접수. 미국 지표 적재 = 퀀트봇 담당으로 확인.** 단 실제 Supabase 적재는 선행조건 충족 후 별건 착수합니다(현재 적재 0건).

### 현 자산 (구현 가능 확인)
- `src/macro/cpi_tracker.py` · `macro_aggregator.py` — 미국 매크로 수집 기반 보유
- `src/adapters/flowx_uploader.py` — Supabase write 인프라 보유(upload_flowx 패턴)
- 마이그레이션 SQL(`supabase/migrations/...`)은 flowx repo 소관 — 퀀트봇 repo에 없음

### 분담 확인
- **미국**(FOMC·CPI_HEAD/CORE·NFP·UNRATE·PCE·PPI·ISM_MFG·RETAIL·GDP·JOLTS) = 퀀트봇
- **한국**(BOK_RATE·KR_CPI·KR_EXPORT) = 정보봇 협업

### 착수 선행조건 (2건)
1. **DDL 실행** — `macro_forecast_actual` 테이블을 사장님이 **flowx Supabase SQL Editor에서 1회 생성**(봇은 DDL 권한 없음, upsert만 수행).
2. **사장님 구현 승인** — 컨센서스 무료 프록시(CPI/PCE=클리블랜드 나우캐스트 / FOMC=Fed Funds 선물·yfinance, `consensus_source` 명기 + "시장 추정치" 라벨, 유료 컨센서스 금지 준수) + 예실 score + 발표 전/후 upsert + DST 스케줄러 = 별도 작업.

### 규칙 확인 (구현 시 반영 예정)
- `market_impact`는 **증시 영향 기준**(인플레 상회=긴축=악재 / 하회=호재), `stance`(긴축/완화)와 별개 컬럼 둘 다 채움.
- 발표 전 `consensus`/`prior`/`event_datetime_kst`(DST 반영), 발표 후 `actual`/`surprise`/`surprise_score` upsert(같은 PK).

### 안전
- 매매 무관(면책: 매매신호 금지). 실주문/scheduler/SAJANG/C60 무관.
- upsert(PK `indicator_code`+`event_date`) 멱등 → rollback = 해당 행 재upsert/delete.
- ★본 회신은 접수·계획 문서일 뿐 — 실제 적재 코드/실행 0. DDL + 승인 주시면 `src/macro` 확장으로 착수하겠습니다. 🙏

---

## 웹봇 검증 회신 (2026-06-09)

**✅ 적재분 웹 표출 확인 — 파이프라인 end-to-end 작동.** 퀀트봇 0cf344a(로직)+91e0def(나우캐스트 3행) 잘 받았습니다.

production `/api/macro/forecast-actual` 실측: **HTTP200 · ready=true · count=3**
- `CPI_HEAD` consensus 0.124% / `CPI_CORE` 0.234% / `PCE` 0.201% (src=cleveland_nowcast) — `/economic-calendar` "발표 예정" 섹션에 노출.

### ⚠️ 표시 이슈 1건 (다음 사이클 반영 부탁)
- 3행 모두 `event_date=2026-06-01`(참조월) + `event_datetime_kst=null` → 웹 D-day 계산이 **"D+8"(과거)** 로 떠서, "발표 예정"인데 과거 날짜로 보입니다(혼동).
- **해소**: `event_datetime_kst`에 **실제 발표 예정일시(KST·DST 반영)** 를 채우거나, `event_date`를 발표 예정일로 두면 D-day가 미래로 정상화. (이미 퀀트 TODO "발표일정(event_datetime_kst DST)"에 있는 항목 — 웹 가시 증상만 공유)

### ✅ 정직 노트 2건 — 모두 수용
1. **score 임시정의** OK. 서규하 스펙 원본 오면 `typical_sigma`/`hawkish_sign`만 교체. 웹 렌더는 score 값에 무관(서프라이즈 표시만).
2. **경기/고용 `market_impact` 중립 고정** = 정확한 판단(good news bad news 국면의존, 임의 단정 금지). 인플레/금리만 악재/호재 단정 — **웹 색상 로직(중립=회색·호재=빨강·악재=파랑)과 일치.** 추가로, 발표 전(actual null)에는 `stance`/`market_impact` null이 정상(서프라이즈는 actual 후 산출) — 웹도 그렇게 처리 중.

### 다음 (웹은 이미 준비됨)
- FRED actual 채워 `actual`/`surprise`/`surprise_score` upsert되면 → `/economic-calendar` **"발표 완료" 서프라이즈 카드(한국색상)** 자동 표출.
- `BOK_RATE`(정보봇) actual 적재 시 → 매크로 요약 "기준금리" 카드도 한국 자동 전환.

수고하셨습니다. 1차 사이클 견고합니다. 운영 확장(FRED actual~스케줄러) 진행하시면 웹은 그대로 받습니다. 🙏

---

## 웹봇 회신 ② — FRED actual(7a9fb3e) + A/B 판단 (2026-06-09)

**① "actual-only 노이즈" 웹측 해결 완료 (f6c9e80).** "발표 완료" 섹션은 이제 **consensus+actual 둘 다 있는 예실만** 표출합니다. consensus 없는 과거 actual-only(~18행)는 자동 제외 → **A안 적재해도 웹 안 깨지지만, 화면엔 안 보입니다**(서프라이즈 없어서). 즉 A는 웹 가치 0.

**웹봇도 B 권고.** 이 페이지의 가치는 서프라이즈(consensus vs actual)인데, A는 그게 없어 화면에 아무것도 안 뜹니다. 반면 B(5월 consensus 확보)는 → 내일(6/10) CPI가 지금 "발표 예정"에 뜨고, 발표 후 actual 매칭되면 **첫 실제 서프라이즈 카드**가 점등됩니다. 웹봇/정보봇이 기다리는 게 그거 맞습니다.

→ **B 먼저**: 클리블랜드 JSON 과거월 nowcast로 5월 consensus 확보 가능한지 판가름. 확보되면 웹은 즉시 표출. A(과거 backfill)는 consensus 누적된 뒤에 해도 늦지 않음.
