# [퀀트봇 → 웹봇] /quant 페이지 탭 재구성 지시서

> **작성일**: 2026-04-12
> **목적**: 기존 탭 3개(대형주 피보/전체 피보/섹터 로테이션) → 새 탭 2개(대형주 점검/소형주 테마) + 섹터 로테이션 통합. 기존 피보나치 데이터 누락 없이 신규 데이터와 합침.

---

## 1. 탭 구조 변경

### Before (현재 5탭)
```
퀀트시스템 | 급락반등 | 대형주 피보나치 | 전체 피보나치 | 섹터 로테이션
```

### After (변경 후 5탭)
```
퀀트시스템 | 급락반등 | 대형주 점검(NEW) | 소형주 테마(NEW) | 섹터 로테이션
```

- `대형주 피보나치` 탭 삭제 → `대형주 점검` 탭으로 대체 (5축 스코어 + 피보나치 통합)
- `전체 피보나치` 탭 삭제 → `소형주 테마` 탭으로 대체 (테마 그룹핑 + 눌림목 통합)
- `섹터 로테이션` 탭 그대로 유지
- `퀀트시스템` 탭 Zone C에서 "대형주 종합점검" 아코디언 **제거** (탭으로 승격했으므로)
- `퀀트시스템` 탭 Zone C에서 "시장 순위" 아코디언은 **유지**

---

## 2. 퀀트시스템 탭 변경사항

### Zone B "대형주 건강 체크" 박스
- 현재대로 A등급/B등급/C등급 숫자 표시 유지
- **"대형주 점검 탭으로 →"** 링크 클릭 시 `대형주 점검` 탭으로 이동

### Zone C 변경
- "대형주 종합점검 — 30대형주 + 13소형주 전체" 아코디언 **삭제** (탭으로 이동)
- 나머지 아코디언(스마트머니/노다지/포트폴리오/시장순위/바닥잡이) 유지

---

## 3. 대형주 점검 탭 (신규)

### 데이터 소스
- **메인**: `quant_bluechip_checkup` 테이블 → 기존 API 사용 (`/api/bluechip-checkup` 또는 신규)
- **피보나치 상세**: `quant_fib_scanner` 테이블 → 기존 API (`/api/fib-scanner`)

### 화면 구성

#### 상단: 등급 요약 카드
```
┌──────────┬──────────┬──────────┬──────────┐
│  A등급    │  B등급    │  C등급    │  D등급    │
│    0     │    0     │   30     │    0     │
│  (녹색)   │  (파랑)   │  (주황)   │  (회색)   │
└──────────┴──────────┴──────────┴──────────┘
 전반적 약세 — 대형주 신중하게
```
클릭 시 해당 등급 필터링.

#### 중단: 30종목 테이블 (기본 접힌 상태)

| # | 종목 | 등급 | 총점 | 밸류 | 실적 | 수주 | 수급 | 기술 | 코멘트 |
|---|------|-----|------|------|------|------|------|------|--------|
| 1 | NAVER | C | 53 | 10 | 12 | 10 | 10 | 11 | 관망 추천 |

각 종목 행 클릭 시 **아코디언 펼치기**:

```
┌─────────────────────────────────────────────────────┐
│ NAVER (035420)  C등급 53점                            │
│                                                       │
│ 5축 스코어 바 차트                                     │
│  밸류에이션  ████████░░  10/20  PER 25.3 고평가         │
│  실적모멘텀  ██████████░░ 12/20  목표가 괴리 +34%       │
│  수주이벤트  ████████░░  10/20  최근 공시 없음          │
│  수급흐름    ████████░░  10/20  수급 데이터 부족         │
│  기술적위치  █████████░░ 11/20  MID 구간 (-20.3%)      │
│                                                       │
│ 피보나치 상세                                          │
│  52주 고점: 253,500  저점: 163,100  현재: 202,000      │
│  38.2%: 197,631  50%: 208,300  61.8%: 218,969         │
│  Zone: MID | 하락: -20.3% | 위치: 56%                 │
│                                                       │
│ 해외 시각                                              │
│  [+] AI 성장과 글로벌 협업에 긍정적, 목표주가 269,577원  │
│  출처: Bloomberg, Reuters                              │
│                                                       │
│ 주요 공시 (있는 경우)                                    │
│  • [수주] 네이버클라우드 대형 계약 (tier1, +15점)        │
└─────────────────────────────────────────────────────┘
```

#### 하단: 피보나치 상세 (기존 대형주 피보나치 탭 내용)

"피보나치 레벨 상세 보기 ▼" 아코디언:

| # | 종목 | 현재가 | 52주고점 | 하락률 | Zone | 38.2% | 50% | 61.8% | 목표가 | 상승여력 |
|---|------|--------|---------|--------|------|-------|-----|-------|--------|---------|
| 1 | 삼성전자 | 206,000 | 223,000 | -7.6% | NEAR_HIGH | ... | ... | ... | 186,599 | ... |

이 데이터는 `quant_fib_scanner.fib_leaders` 30종목.

### bluechips[] 데이터 스키마

```typescript
interface BluechipItem {
  code: string           // 종목코드
  name: string           // 종목명
  sector: string         // 업종 (반도체, 방산, 은행...)
  cap: number            // 시총 (억원)
  price: number          // 현재가
  grade: string          // A / B / C / D
  total_score: number    // 총점 (0~100)
  comment: string        // 한줄 코멘트
  scores: {              // 5축 상세
    valuation: { score: number, reason: string }    // 밸류에이션 (0~20)
    earnings: { score: number, reason: string }     // 실적 모멘텀 (0~20)
    events: { score: number, reason: string }       // 수주/이벤트 (0~20)
    supply_demand: { score: number, reason: string } // 수급 흐름 (0~20)
    technical: { score: number, reason: string }    // 기술적 위치 (0~20)
  }
  target_price: number   // 컨센서스 목표가
  upside_pct: number     // 상승여력 (%)
  per: number            // PER
  pbr: number            // PBR
  fib_zone: string       // DEEP / MID / MILD / NEAR_HIGH
  drop_52w: number       // 52주 고점 대비 하락률 (%)
  position_pct: number   // 52주 범위 내 위치 (0~100%)
  major_events: {        // DART 주요 공시 (최대 3건)
    event: string
    tier: string
    score: number
    url: string
    report: string
  }[]
  global_view: {         // 해외 시각 (상위 10종목만, 나머지 null)
    summary: string
    sentiment: string    // positive / neutral / negative
    source: string
  } | null
}
```

### fib_leaders[] 데이터 스키마 (기존 피보나치)

```typescript
interface FibLeader {
  code: string
  name: string
  cap: number            // 시총 (조원 단위)
  price: number
  w52h: number           // 52주 고점
  w52l: number           // 52주 저점
  drop: number           // 고점 대비 하락률 (%)
  fib_382: number        // 피보나치 38.2% 레벨
  fib_500: number        // 피보나치 50% 레벨
  fib_618: number        // 피보나치 61.8% 레벨
  fib_zone: string       // DEEP / MID / MILD / NEAR_HIGH
  fib_zone_label: string // "38.2% 아래 (깊은 하락)" 등
  fib_status: string     // "회복 중" / "하락 중"
  position_pct: number   // 52주 범위 위치 (0~100%)
  target: number         // 목표가
  upside: number         // 상승여력 (%)
  per: number
  pbr: number
  sector: string
}
```

---

## 4. 소형주 테마 탭 (신규)

### 데이터 소스
- **테마 소형주**: `quant_bluechip_checkup` 테이블 → `data.smallcaps[]`
- **눌림목 전체**: `quant_fib_scanner` 테이블 → `data.fib_stocks[]`

### 화면 구성

#### 상단: 안내 문구
```
소형주 테마 — 대형주에서 찾은 산업의 저렴한 종목
대형주 점검에서 "이 산업이 좋다" → 여기서 실제 매수 후보
```

#### 중단: 핫 섹터 테마별 그룹핑

테마 카드:
```
┌──────────────┬──────────────┬──────────────┐
│   2차전지     │    반도체     │     방산      │
│   3종목       │    2종목      │    2종목      │
│  HOT 섹터    │   HOT 섹터   │  시나리오 연동  │
│  (빨간 배경)   │  (빨간 배경)  │  (주황 배경)   │
├──────────────┼──────────────┼──────────────┤
│    건설       │     은행      │     증권      │
│   2종목       │    2종목      │    2종목      │
└──────────────┴──────────────┴──────────────┘
```

각 테마 카드 클릭 시 종목 리스트 펼치기:

```
┌─────────────────────────────────────────────────┐
│ 🔥 2차전지 — 3종목                               │
│                                                   │
│  엔켐      37,200원  시총17,047억  -64.6%  DEEP  │
│            수급: 관망                              │
│                                                   │
│  엘앤에프   [AI추천]                               │
│            양극재 생산, 삼성SDI 공급망 핵심         │
│                                                   │
│  (종목명 클릭 → /stock/[ticker] 이동)             │
└─────────────────────────────────────────────────┘
```

#### 하단: 눌림목 소형주 전체 (기존 전체 피보나치 탭 내용)

"눌림목 소형주 전체 (기존 피보나치) ▼" 섹션:

**필터 탭**: 전체 50 | DEEP | MID | MILD

| # | 종목 | 현재가 | 시총 | 하락률 | Zone | 52주고점 | 38.2% | 50% | 61.8% | 목표가 | 상승여력 |
|---|------|--------|------|--------|------|---------|-------|-----|-------|--------|---------|
| 1 | 에코프로비엠 | ... | ... | -42% | DEEP | ... | ... | ... | ... | ... | ... |

이 데이터는 `quant_fib_scanner.fib_stocks` 50종목.

### smallcaps[] 데이터 스키마

```typescript
interface SmallcapItem {
  code: string           // 종목코드
  name: string           // 종목명
  sector: string         // 업종
  theme: string          // 테마 (= 연결된 핫 섹터)
  cap: number            // 시총 (억원, AI추천은 0일 수 있음)
  price: number          // 현재가 (AI추천은 0)
  drop_52w: number       // 52주 고점 대비 하락률
  fib_zone: string       // DEEP / MID / MILD / "" (AI추천은 빈값)
  upside_pct: number     // 상승여력
  target_price: number   // 목표가
  supply_signal: string  // "쌍끌이" / "매집" / "관망" / "AI추천"
  per: number
  pbr: number
  ai_reason?: string     // AI 추천 이유 (Perplexity, supply_signal="AI추천"일 때만)
}
```

### fib_stocks[] 데이터 스키마 (기존 전체 피보나치)

```typescript
interface FibStock {
  code: string
  name: string
  cap: number            // 시총 (억원)
  price: number
  w52h: number
  w52l: number
  drop: number           // 고점 대비 하락률 (%)
  fib_382: number
  fib_500: number
  fib_618: number
  fib_zone: string       // DEEP / MID / MILD
  fib_zone_label: string
  fib_status: string
  position_pct: number
  target: number
  upside: number
  per: number
  pbr: number
  sector: string
}
```

---

## 5. 섹터 로테이션 탭 — 변경 없음

기존 그대로 유지. `quant_fib_scanner.sector_rotation` 20건.

---

## 6. API 정리

| 데이터 | API 엔드포인트 | 테이블 | 비고 |
|--------|---------------|--------|------|
| 대형주 체크업 | `/api/bluechip-checkup` | `quant_bluechip_checkup` | 신규 or 기존 |
| 피보나치 전체 | `/api/fib-scanner` | `quant_fib_scanner` | 기존 (fib_leaders + fib_stocks + sector_rotation) |
| 시장 순위 | `/api/market-ranking` | `quant_market_ranking` | 기존 |

대형주 점검 탭에서 2개 API를 **병렬 fetch**:
1. `bluechip-checkup` → 5축 스코어 30종목 + 소형주 13종목
2. `fib-scanner` → 피보나치 상세 (fib_leaders 30종목)

소형주 테마 탭에서 2개 API를 **병렬 fetch**:
1. `bluechip-checkup` → smallcaps (테마 소형주 13종목)
2. `fib-scanner` → fib_stocks (눌림목 50종목)

---

## 7. 핵심 동선 다이어그램

```
/quant 페이지 (SystemPage.tsx)
│
├── [퀀트시스템] 탭
│   ├── Zone A: 뭘 사라 (3초)
│   │   └── 파워스코어 TOP3 + 노다지 TOP2 + ETF 추천
│   ├── Zone B: 왜 사야 하나 (30초)
│   │   ├── 섹터 온도 (HOT/COLD)
│   │   ├── 스마트 머니 쌍끌이 TOP 3
│   │   ├── 오늘 폭발 (거래량/외인 TOP 3)
│   │   └── 대형주 건강 체크 (등급 요약 → "대형주 점검 탭으로 →" 링크)
│   └── Zone C: 펼쳐서 보기
│       ├── 스마트 머니 전체
│       ├── 노다지 리포트 전체
│       ├── 포트폴리오 배분
│       ├── 시장 순위 (거래량/급등/체결강도)
│       └── 바닥잡이 레이더 (COMING SOON)
│
├── [급락반등] 탭 — 변경 없음
│   └── dashboard_crash_bounce
│
├── [대형주 점검] 탭 — ★ 신규 (기존 "대형주 피보나치" 대체)
│   ├── 등급 요약 (A/B/C/D 카드)
│   ├── 30종목 5축 스코어 테이블 (행 클릭 → 아코디언)
│   │   └── 5축 바차트 + 피보나치 상세 + 해외시각 + 공시
│   └── 피보나치 레벨 상세 (기존 대형주 피보나치 데이터)
│       └── quant_fib_scanner.fib_leaders 30종목
│
├── [소형주 테마] 탭 — ★ 신규 (기존 "전체 피보나치" 대체)
│   ├── 핫 섹터 테마별 카드 (2차전지/반도체/방산...)
│   │   └── quant_bluechip_checkup.smallcaps 13종목
│   └── 눌림목 소형주 전체 (기존 전체 피보나치 데이터)
│       ├── DEEP / MID / MILD 필터 탭
│       └── quant_fib_scanner.fib_stocks 50종목
│
└── [섹터 로테이션] 탭 — 변경 없음
    └── quant_fib_scanner.sector_rotation 20건
```

---

## 8. 체크리스트

- [ ] 기존 `대형주 피보나치` 탭 컴포넌트 삭제
- [ ] 기존 `전체 피보나치` 탭 컴포넌트 삭제
- [ ] 신규 `대형주 점검` 탭 컴포넌트 생성
- [ ] 신규 `소형주 테마` 탭 컴포넌트 생성
- [ ] 퀀트시스템 탭 Zone C에서 "대형주 종합점검" 아코디언 제거
- [ ] Zone B "대형주 건강 체크" 박스에 탭 이동 링크 추가
- [ ] `/api/bluechip-checkup` 엔드포인트 확인 (없으면 신규 생성)
- [ ] 기존 피보나치 데이터(fib_leaders 30종목, fib_stocks 50종목) 누락 없이 새 탭에 포함
- [ ] 기존 피보나치 DEEP/MID/MILD 필터 기능 유지

---

*이 지시서는 퀀트봇에서 작성했습니다. 데이터 스키마 질문은 퀀트봇에게.*
