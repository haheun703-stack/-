# FLOWX 대시보드 V2 — 웹봇 통합 지시서

> **작성일**: 2026-03-24
> **대상**: FLOWX 웹 개발자 (웹봇)
> **목적**: Quantum Master 퀀트봇 대시보드를 ppwangga.com FLOWX 페이지에 통합

---

## 1. 아키텍처 개요

```
[BAT-D 스케줄러] (매일 16:30)
    │
    ├─ 데이터 수집 → 시스템 JSON 파일들 생성
    │
    └─ dashboard_data.py 실행
           │
           └─ website/data/dashboard_state.json 생성
                  │
                  └─ FLOWX 웹 페이지가 fetch()로 읽어서 렌더링
```

**핵심**: 웹봇은 `dashboard_state.json` 하나만 읽으면 됩니다. 나머지는 Python 백엔드가 처리합니다.

---

## 2. 데이터 소스

### 파일: `website/data/dashboard_state.json`

- **갱신 주기**: BAT-D 실행 시 (평일 16:30~17:00)
- **생성자**: `dashboard_data.py`
- **인코딩**: UTF-8
- **크기**: ~5KB

---

## 3. JSON 스키마 상세

### 최상위 구조

```json
{
  "generated_at": "2026-03-24T20:56:50.005315",
  "zone1": { ... },   // 오늘의 판단
  "zone2": [ ... ],   // 액션 리스트
  "zone3": { ... },   // 성과 (Paper Trading)
  "zone4": [ ... ],   // 섹터 흐름
  "zone5": { ... },   // 수급 레이더
  "zone6": { ... }    // 시스템 신뢰도
}
```

### Zone 1: 오늘의 판단 (BRAIN + LENS)

```json
{
  "verdict": "관망",           // "매수" | "관망" | "회피" | "중립"
  "cash_pct": 71,              // 현금 비중 (0~100)
  "buy_pct": 29,               // 매수 비중 (0~100)
  "regime": "CAUTION",         // PANIC | CRISIS | BEAR | CAUTION | NEUTRAL | RECOVERY | BULL
  "regime_transition": "BEAR 접근",  // 전환 방향 (빈 문자열 = 없음)
  "transition_prob": 70,       // 전환 확률 (0~100)
  "macro_grade": "현금비중 확대", // 매크로 등급 설명
  "vix": 26.1,                 // VIX 지수
  "kospi": 2480,               // KOSPI 종가 (0이면 데이터 없음)
  "kospi_chg": -1.2,           // KOSPI 등락률 (%)
  "brain_score": 100.0,        // BRAIN 9-ARM 합산 점수
  "shield_status": "RED",      // RED | YELLOW | GREEN
  "lens_summary": "...",       // LENS 요약 (max 100자, 빈 문자열 가능)
  "updated_at": "2026-03-24T19:13:34"
}
```

**verdict 색상 매핑**:
| verdict | 색상 | CSS 클래스 |
|---------|------|-----------|
| 매수 | #00e59b (green) | buy |
| 관망 | #f5a623 (amber) | watch |
| 회피 | #ff4d6d (red) | sell |
| 중립 | #f5a623 (amber) | watch |

### Zone 2: 액션 리스트

```json
[
  {
    "ticker": "042660",          // 종목코드 (6자리)
    "name": "한화오션",           // 종목명
    "action": "BUY",             // "BUY" | "WATCH" | "SELL"
    "grade": "A",                // "AA" | "A" | "B" | "C" | "F"
    "score": 80,                 // 총점 (0~100)
    "reason": "K-방산 수출...",   // 사유 (max 60자)
    "strategy": "AI_BRAIN",      // "AI_BRAIN" | "SCAN"

    // SCAN 전략인 경우 추가 필드:
    "close": 108400,             // 현재가 (optional)
    "stop_loss": 102980,         // 손절가 (optional)
    "target_price": 119240       // 목표가 (optional)
  }
]
```

**최대 6개 항목**. action별 스타일:
| action | 아이콘 | 배경색 | 테두리 |
|--------|--------|--------|--------|
| BUY | 초록 원 "BUY" | #001f16 | rgba(0,229,155,0.25) |
| WATCH | 황색 원 "WTC" | #1a1600 | rgba(245,166,35,0.2) |
| SELL | 적색 원 "SEL" | #3d0012 | rgba(255,77,109,0.25) |

### Zone 3: 성과 (Paper Trading)

```json
{
  "equity": 29987920,          // 현재 자산 (원)
  "initial_capital": 30000000, // 초기 자본 (원)
  "total_return_pct": -0.04,   // 전체 수익률 (%)
  "week_return_pct": 0,        // 주간 수익률 (%)
  "month_return_pct": 0,       // 월간 수익률 (%)
  "win_rate": 62.5,            // 승률 (%)
  "pf": 1.38,                  // Profit Factor
  "mdd": -4.2,                 // 최대 낙폭 (%)
  "total_trades": 16,          // 총 거래 수
  "wins": 10,                  // 승
  "losses": 6,                 // 패
  "positions": [               // 현재 보유 포지션
    {
      "ticker": "042660",
      "name": "한화오션",
      "pnl_pct": -0.12,       // 현재 손익 (%)
      "days": 3,               // 보유일
      "strategy": "AI_BRAIN",
      "grade": "A"
    }
  ],
  "recent_trades": []          // 최근 청산 거래 (최근 5건)
}
```

### Zone 4: 섹터 흐름

```json
[
  {
    "name": "건설",            // 섹터명
    "score": 93.4,             // 모멘텀 점수 (0~100)
    "ret_5d": 3.61,            // 5일 수익률 (%)
    "rsi": 55.8,               // RSI 14
    "rank": 1,                 // 순위
    "signal": "매수",          // "매수" | "관찰" | "주의" | "회피"
    "relay": null              // null | "FIRE" (릴레이 발화)
  }
]
```

**최대 10개 섹터**. signal별 색상:
| signal | 색상 |
|--------|------|
| 매수 | #00e59b |
| 관찰 | #38bdf8 |
| 주의 | #f5a623 |
| 회피 | #ff4d6d |

### Zone 5: 수급 레이더

```json
{
  "foreign_flow": [            // 외국인 자금 흐름 (상위 5)
    {
      "ticker": "068270",
      "name": "셀트리온",
      "direction": "NEUTRAL",  // "BUY" | "SELL" | "NEUTRAL"
      "score": 46,
      "z_score": 3.18          // Z-Score (이상치 감지)
    }
  ],
  "sd_patterns": [             // SD V2 매집/분산 패턴
    {
      "ticker": "005930",
      "name": "삼성전자",
      "grade": "A",            // A(매집) | B(관찰) | C(관찰) | F(분산)
      "pattern": "매집",       // "매집" | "관찰" | "분산"
      "pattern_name": "3연속 기관 순매수",
      "sd_score": 0.45         // SD 점수 (-1~+1)
    }
  ],
  "supply_summary": {          // KOSPI 전체 수급
    "foreign": 482000000000,   // 외국인 (원)
    "inst": -124000000000,     // 기관 (원)
    "indiv": 89000000000       // 개인 (원)
  }
}
```

### Zone 6: 시스템 신뢰도

```json
{
  "tomorrow_picks": 67.7,     // 시그널별 적중률 (%)
  "whale_detect": 43.3,
  "volume_spike": 36.1,
  "brain": 60.0,
  "recent_10": [0,0,0,0,0,0,0,0,0,0],  // 최근 10건 (1=적중, 0=미스)
  "active_signals": 7          // 활성 시그널 수
}
```

---

## 4. 레퍼런스 HTML

프로젝트 루트의 `quantum_master_dashboard.html`이 완전한 동작 레퍼런스입니다:
- 다크 테마 CSS + 6-Zone 레이아웃
- `dashboard_state.json`을 fetch()로 읽어 동적 렌더링
- 5분 자동 갱신 + REFRESH 버튼

**이 파일을 그대로 FLOWX에 통합하면 됩니다.**

---

## 5. FLOWX 통합 방법

### 옵션 A: 직접 통합 (권장)

1. `quantum_master_dashboard.html`의 CSS와 HTML 구조를 FLOWX 레이아웃에 이식
2. `<script>` 섹션의 렌더링 함수들을 그대로 사용
3. `DATA_URL`을 실제 데이터 경로에 맞게 수정:
   ```js
   const DATA_URL = '/quant/data/dashboard_state.json';
   ```

### 옵션 B: iframe 임베드 (빠른 적용)

```html
<iframe src="/quant/quantum_master_dashboard.html"
        style="width:100%;height:100vh;border:none;background:#0a0c10">
</iframe>
```

### 옵션 C: Supabase 연동 (향후)

현재 `dashboard_state.json`은 정적 파일이지만, 향후 Supabase 테이블에 업로드하여 실시간 API로 전환 가능:
```js
const DATA_URL = 'https://xxx.supabase.co/rest/v1/dashboard_state?select=*&order=id.desc&limit=1';
```

---

## 6. 기존 FLOWX 모듈과의 관계

| 기존 모듈 | Zone 대응 | 변경 사항 |
|-----------|-----------|-----------|
| ETF 시그널 | Zone 1 (판단) | 더 풍부한 BRAIN+LENS 판단으로 대체 |
| 그룹순환 | Zone 4 (섹터) | 10개 섹터 + 릴레이로 확장 |
| 외국인 자본흐름 | Zone 5 (수급) | SD V2 패턴 + 외/기/개 수급 통합 |
| 페이퍼 트레이딩 | Zone 3 (성과) | 실제 paper_trading_unified.py 결과 |
| *신규* | Zone 2 (액션) | AI 추천 + 스캔 결과 일간 할일 |
| *신규* | Zone 6 (신뢰도) | 시그널별 적중률 + 최근 10건 |

---

## 7. 디자인 스펙

### 색상 팔레트

```css
--bg:       #0a0c10    /* 배경 */
--bg2:      #0f1218    /* 카드 배경 */
--bg3:      #161b24    /* 헤더/셀 배경 */
--border:   #1e2736    /* 테두리 */
--text:     #e2e8f0    /* 텍스트 */
--muted:    #5a6a82    /* 보조 텍스트 */
--green:    #00e59b    /* 양봉/매수 */
--amber:    #f5a623    /* 경고/관망 */
--red:      #ff4d6d    /* 음봉/매도 */
--blue:     #38bdf8    /* 정보 */
--purple:   #a78bfa    /* 시그널 */
```

### 폰트

- 제목/숫자: `Space Mono` (monospace)
- 본문: `Noto Sans KR` (sans-serif)
- 기본 크기: 13px

### 반응형

현재 데스크탑 전용. 모바일 대응 시:
- Zone 4+5 하단 그리드: `grid-template-columns: 1fr` (1열)
- 신뢰도 바: 스크롤 가능 (이미 `overflow-x: auto` 적용)

---

## 8. 데이터 갱신 타이밍

| 시간 | 이벤트 | 갱신 내용 |
|------|--------|-----------|
| 08:00 | BAT-M 아침브리핑 | zone1 일부 (BRAIN 판단) |
| 09:05~15:20 | BAT-K Intraday Eye | zone1 실시간은 별도 |
| 16:30 | BAT-D 장마감 | **전체 갱신** (모든 Zone) |
| 17:00 | BAT-J Outlook | zone1 보강 |
| 18:30 | BAT-D2 수급 | zone5 수급 데이터 |

**프론트엔드 갱신**: 5분마다 자동 fetch (이미 구현)

---

## 9. 파일 구조

```
website/
├── data/
│   └── dashboard_state.json     ← Python이 생성, 웹이 소비
├── quantum_master_dashboard.html ← 완전한 레퍼런스 HTML
└── (기존 FLOWX 파일들)

dashboard_data.py                ← JSON 생성 스크립트 (BAT-D에서 실행)
```

---

## 10. 체크리스트

- [ ] `dashboard_state.json` 경로 확인 (서버 배포 시)
- [ ] CORS 설정 (같은 도메인이면 불필요)
- [ ] CSS 변수 충돌 확인 (FLOWX 기존 스타일과)
- [ ] 모바일 반응형 (필요 시)
- [ ] Supabase 실시간 전환 (향후)
