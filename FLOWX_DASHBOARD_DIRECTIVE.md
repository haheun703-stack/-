# FLOWX 대시보드 V3 — 웹봇 통합 지시서 (풀 공개 버전)

> **작성일**: 2026-03-24
> **대상**: FLOWX 웹 개발자 (웹봇)
> **목적**: Quantum Master 퀀트봇 대시보드를 ppwangga.com FLOWX 페이지에 통합
> **핵심 변경**: 종목 기술지표 · 수급 · 매집 · AI판단 전체 공개 + ETF 추천 추가

---

## 0. 변경사항 요약 (이전 대비)

| 항목 | 이전 (V2) | 현재 (V3) |
|------|-----------|-----------|
| Zone 2 종목 | 이름+등급+점수만 표시 | **풀 공개**: RSI/ADX/BB/Stoch/MA/SAR + 외인기관수급 + 매집단계 + 안전마진 + 점수분해 + AI판단 |
| Zone 4 섹터 | 섹터 이름만 | 섹터별 **ETF 코드 · 시그널** 연결 |
| Zone 7 ETF | 없음 | **신규**: 매크로/섹터/테마별 ETF 추천 + 시장 방향 |
| Zone 라벨 | 영어 (BRAIN+LENS, SD V2) | **전부 한글** (일반인이 이해 가능) |
| BUY/SELL | 영어 | **매수/매도/관찰** 한글 |

---

## 1. 아키텍처

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

---

## 2. 라벨 규칙 (한글 필수)

**모든 UI 라벨은 한글로. 영어 전문용어 사용 금지.**

| 위치 | 한글 라벨 |
|------|-----------|
| 헤더 | 퀀트 시스템 PRO |
| 신뢰도 바 | AI추천 / 세력탐지 / 거래량적중 / AI브레인 |
| Zone 1 | 오늘의 판단 · AI 종합 분석 |
| Zone 1 태그 | 시장 / 시장국면 / AI분석 |
| Zone 2 | 종목 추천 · 기술지표·수급·매집 전체 공개 |
| Zone 2 배지 | 매수 / 매도 / 관찰 (BUY/SELL/WATCH 아님) |
| Zone 2 전략 | AI 판단 / 스캔 발굴 (AI_BRAIN/SCAN 아님) |
| Zone 3 | 포트폴리오 · 모의투자 성과 |
| Zone 4 | 섹터 순환 · 업종별 흐름 TOP 10 |
| Zone 5 | 외국인·기관 수급 · 자금 흐름 분석 |
| Zone 5 하위 | 매집·분산 패턴 감지 |
| Zone 7 | ETF 추천 · 매크로·섹터·테마별 ETF |
| Zone 7 방향 | 상승/하락/횡보 (BULL/BEAR/NEUTRAL 아님) |
| 푸터 | 퀀트 시스템 · 10단계 시그널엔진 · N개 시그널 가동 |

---

## 3. JSON 스키마

### 파일: `website/data/dashboard_state.json` (~15KB)

### 최상위 구조

```json
{
  "generated_at": "2026-03-24T21:52:13",
  "zone1": { ... },
  "zone2": [ ... ],
  "zone3": { ... },
  "zone4": [ ... ],
  "zone5": { ... },
  "zone6": { ... },
  "zone7": [ ... ]
}
```

### Zone 1: 오늘의 판단

```json
{
  "verdict": "관망",
  "cash_pct": 71,
  "buy_pct": 29,
  "regime": "CAUTION",
  "regime_transition": "BEAR 접근",
  "transition_prob": 70,
  "macro_grade": "현금비중 확대",
  "vix": 26.1,
  "kospi": 2480,
  "kospi_chg": -1.2,
  "brain_score": 100.0,
  "shield_status": "RED",
  "lens_summary": "...",
  "updated_at": "2026-03-24T19:13:34"
}
```

### Zone 2: 종목 추천 (풀 공개) — 핵심 변경

```json
[
  {
    "ticker": "042660",
    "name": "한화오션",
    "action": "BUY",
    "grade": "A",
    "score": 80,
    "reason": "K-방산 수출 교두보 확대 기대...",
    "strategy": "AI_BRAIN",

    "close": 122300,
    "price_change": 2.86,
    "stop_loss": 117190,
    "target_price": 143650,
    "entry_price": 126010,
    "entry_condition": "MA5 하향이탈 -3.4%→반등확인 후",

    "rsi": 42.5,
    "adx": 21.7,
    "stoch_k": 32.7,
    "stoch_d": null,
    "bb_position": 0.26,
    "above_ma20": false,
    "above_ma60": false,
    "ma5_gap": -3.4,
    "sar_trend": -1,

    "foreign_5d": -32208000000,
    "inst_5d": -17047000000,

    "accum_phase": "재돌파",
    "accum_days": 23,
    "accum_return": 3.5,

    "safety_signal": "RED",
    "safety_label": "위험",

    "score_breakdown": {
      "multi": 0,
      "individual": 16.0,
      "tech": 12,
      "flow": 0,
      "safety": 8,
      "overheat": 3
    },

    "ai_action": "BUY",
    "ai_tag": "AI:BUY(80%,high)",
    "ai_bonus": 16.2,

    "overheat_flags": ["SAR↓"],
    "drawdown": -21.0,
    "consensus_upside": 38.3,
    "trade_strategy": "swing"
  }
]
```

**최대 10개 항목.**

### Zone 3: 포트폴리오

```json
{
  "equity": 29987920,
  "initial_capital": 30000000,
  "total_return_pct": -0.04,
  "week_return_pct": 0,
  "month_return_pct": 0,
  "win_rate": 62.5,
  "pf": 1.38,
  "mdd": -4.2,
  "total_trades": 16,
  "wins": 10,
  "losses": 6,
  "positions": [
    {
      "ticker": "042660",
      "name": "한화오션",
      "pnl_pct": -0.12,
      "days": 3,
      "strategy": "AI_BRAIN",
      "grade": "A"
    }
  ],
  "recent_trades": []
}
```

### Zone 4: 섹터 순환

```json
[
  {
    "name": "건설",
    "score": 93.4,
    "ret_5d": 3.61,
    "rsi": 55.8,
    "rank": 1,
    "signal": "매수",
    "relay": null,
    "etf_code": "139260",
    "etf_signal": "MOMENTUM_DIP",
    "etf_sizing": "QUARTER"
  }
]
```

### Zone 5: 수급 분석

```json
{
  "foreign_flow": [
    {
      "ticker": "068270",
      "name": "셀트리온",
      "direction": "NEUTRAL",
      "score": 46,
      "z_score": 3.18
    }
  ],
  "sd_patterns": [
    {
      "ticker": "005930",
      "name": "삼성전자",
      "grade": "A",
      "pattern": "매집",
      "pattern_name": "3연속 기관 순매수",
      "sd_score": 0.45
    }
  ],
  "supply_summary": {
    "foreign": 482000000000,
    "inst": -124000000000,
    "indiv": 89000000000
  }
}
```

### Zone 6: 신뢰도

```json
{
  "tomorrow_picks": 67.7,
  "whale_detect": 43.3,
  "volume_spike": 36.1,
  "brain": 60.0,
  "recent_10": [0,0,0,0,0,0,0,0,0,0],
  "active_signals": 7
}
```

### Zone 7: ETF 추천 (신규!)

```json
[
  {
    "_market_direction": "NEUTRAL",
    "_market_score": 0.13,
    "_market_confidence": 66,
    "_regime": "CAUTION",
    "_vix": 26.1,
    "_reasons": ["VIX 26.1 (공포)", "파생 STRONG_BULL (100점)"]
  },
  {
    "category": "섹터",
    "ticker": "143860",
    "name": "TIGER 헬스케어",
    "action": "BUY",
    "confidence": 40,
    "holding_period": "5~10일",
    "portfolio_pct": 8,
    "stop_loss": "섹터 수급 이탈 시",
    "target": "시나리오 Phase 전환 시 재평가",
    "entry_timing": "09:30~10:00 VWAP 기준 매수",
    "reasons": [
      "시나리오 미국 금리 인하 사이클 Phase1 (45점)",
      "스텔스 매집 (외인 400억, 가격 -1.6%)"
    ]
  }
]
```

---

## 4. Zone 2 종목 카드 — 풀 공개 렌더링

### 4-1. 카드 구조

```
┌────────────────────────────────────────────────────┐
│ [A] 한화오션 (042660)       [매수] 80점  AI 판단    │ ← 헤더
├────────────────────────────────────────────────────┤
│ K-방산 수출 교두보 확대 기대...                      │ ← 사유
├────────────────────────────────────────────────────┤
│ 현재가 122,300원 (+2.86%)  손절 117,190  목표 143,650 │ ← 가격
│                                                    │
│ [RSI 42] [ADX 22] [BB 26%] [Stoch 33] [MA5 -3.4%] [SAR↓] │ ← 기술 칩
│                                                    │
│ [재돌파 23일] [안전: 위험] [AI:BUY(80%)] [외인5d -322억] │ ← 태그
│                                                    │
│ 기술분석 12  수급 0  개별종목 16  안전도 8  과열 3    │ ← 점수 분해
└────────────────────────────────────────────────────┘
```

### 4-2. 카드 색상

| action | 한글 | 배경 | 테두리 |
|--------|------|------|--------|
| BUY | 매수 | #001f16 | rgba(0,229,155,0.3) |
| WATCH | 관찰 | #1a1600 | rgba(245,166,35,0.25) |
| SELL | 매도 | #3d0012 | rgba(255,77,109,0.3) |

### 4-3. 기술지표 칩 색상

| 지표 | 좋음 (초록) | 나쁨 (빨강) |
|------|------------|------------|
| RSI | ≤45 | ≥70 |
| ADX | ≥25 | — |
| BB% | <30% | >80% |
| Stoch | <20 | >80 |
| SAR | ↑ | ↓ |
| MA60 | MA60↑ | — |

### 4-4. 태그 색상

| 태그 | 배경 | 글자색 |
|------|------|--------|
| 매집단계 | #001f16 | #00e59b |
| 안전: 정상 | #001f16 | #00e59b |
| 안전: 주의 | #3d2800 | #f5a623 |
| 안전: 위험 | #3d0012 | #ff4d6d |
| AI 태그 | #1e1040 | #a78bfa |
| 외인/기관 매수 | #001f16 | #00e59b |
| 외인/기관 매도 | #3d0012 | #ff4d6d |
| 과열 경고 | #3d2800 | #f5a623 |

### 4-5. 수급 금액 표시

```
1억 이상: ±N억 (예: +322억, -170억)
1만 이상: ±N만
```

---

## 5. Zone 7 ETF 카드

### 5-1. 카드 구조

```
┌─ 시장 방향 헤더 ──────────────────────────────────┐
│ 시장 방향: 횡보 (점수 0.13, 신뢰도 66%)             │
│ VIX 26.1 (공포) · 파생 STRONG_BULL (100점)          │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│ [섹터] TIGER 헬스케어 (143860)     [매수] 신뢰 40%  │
│ 시나리오 미국 금리 인하... · 스텔스 매집...            │
│ 보유 5~10일 · 손절: 섹터 수급 이탈 시                │
└────────────────────────────────────────────────────┘
```

### 5-2. 카테고리 색상

| category | 배경 | 글자색 |
|----------|------|--------|
| 섹터 | #0c2a3d | #38bdf8 |
| 매크로 | #1e1040 | #a78bfa |
| 테마 | #3d2800 | #f5a623 |

### 5-3. 시장 방향 변환

| 원본 | 표시 |
|------|------|
| BULL | 상승 |
| BEAR | 하락 |
| NEUTRAL | 횡보 |

---

## 6. CSS (전체)

### 색상 팔레트

```css
:root {
  --bg: #0a0c10; --bg2: #0f1218; --bg3: #161b24;
  --border: #1e2736; --border2: #2a3545;
  --text: #e2e8f0; --muted: #5a6a82; --muted2: #3d4e63;
  --green: #00e59b; --green2: #003d2b; --green3: #001f16;
  --amber: #f5a623; --amber2: #3d2800;
  --red: #ff4d6d; --red2: #3d0012;
  --blue: #38bdf8; --blue2: #0c2a3d;
  --purple: #a78bfa; --purple2: #1e1040;
  --mono: 'Space Mono', monospace;
  --sans: 'Noto Sans KR', sans-serif;
}
```

### Zone 2 카드 CSS (핵심 — 복붙 가능)

```css
.action-list { display: flex; flex-direction: column; gap: 12px; }
.stock-card { border-radius: 6px; border: 1px solid; overflow: hidden; }
.stock-card.buy  { border-color: rgba(0,229,155,0.3); background: var(--green3); }
.stock-card.watch{ border-color: rgba(245,166,35,0.25); background: #1a1600; }
.stock-card.sell { border-color: rgba(255,77,109,0.3); background: var(--red2); }

.sc-header { display: flex; align-items: center; gap: 12px; padding: 10px 14px; border-bottom: 1px solid var(--border); }
.sc-grade { font-family: var(--mono); font-size: 18px; font-weight: 700; }
.stock-card.buy  .sc-grade { color: var(--green); }
.stock-card.watch .sc-grade { color: var(--amber); }
.stock-card.sell .sc-grade  { color: var(--red); }
.sc-name { font-size: 14px; font-weight: 600; flex: 1; }
.sc-name em { font-style: normal; color: var(--muted); font-size: 11px; margin-left: 6px; }
.sc-badge { font-family: var(--mono); font-size: 10px; padding: 2px 8px; border-radius: 3px; font-weight: 700; }
.stock-card.buy  .sc-badge { background: var(--green); color: #000; }
.stock-card.watch .sc-badge { background: var(--amber); color: #000; }
.stock-card.sell .sc-badge  { background: var(--red); color: #fff; }
.sc-score { font-family: var(--mono); font-size: 15px; font-weight: 700; }
.sc-strategy { font-size: 10px; color: var(--muted); }

.sc-reason { padding: 8px 14px; font-size: 12px; color: var(--muted); line-height: 1.5; border-bottom: 1px solid var(--border); }
.sc-detail { padding: 10px 14px; display: flex; flex-direction: column; gap: 8px; }
.sc-prices { display: flex; gap: 16px; flex-wrap: wrap; font-size: 11px; }

.sc-tech { display: flex; gap: 6px; flex-wrap: wrap; }
.tech-chip { font-family: var(--mono); font-size: 10px; padding: 2px 7px; border-radius: 3px; background: var(--bg3); border: 1px solid var(--border); white-space: nowrap; }
.tech-chip.good { border-color: rgba(0,229,155,0.3); color: var(--green); }
.tech-chip.bad  { border-color: rgba(255,77,109,0.3); color: var(--red); }

.sc-tags { display: flex; gap: 6px; flex-wrap: wrap; }
.sc-tag { font-size: 10px; padding: 2px 7px; border-radius: 3px; white-space: nowrap; }
.sc-tag.accum      { background: var(--green3); color: var(--green); }
.sc-tag.safe-green { background: var(--green3); color: var(--green); }
.sc-tag.safe-yellow{ background: var(--amber2); color: var(--amber); }
.sc-tag.safe-red   { background: var(--red2); color: var(--red); }
.sc-tag.ai         { background: var(--purple2); color: var(--purple); }
.sc-tag.flow-buy   { background: var(--green3); color: var(--green); }
.sc-tag.flow-sell  { background: var(--red2); color: var(--red); }

.sc-breakdown { display: flex; gap: 4px; font-family: var(--mono); font-size: 10px; flex-wrap: wrap; }
.sc-breakdown span { padding: 1px 5px; border-radius: 2px; background: var(--bg3); color: var(--muted); }
```

### Zone 7 ETF CSS (핵심 — 복붙 가능)

```css
.etf-list { display: flex; flex-direction: column; gap: 8px; }
.etf-card { display: flex; align-items: center; gap: 12px; padding: 10px 14px; border-radius: 5px; background: var(--bg3); border: 1px solid var(--border); }
.etf-cat { font-family: var(--mono); font-size: 10px; padding: 2px 7px; border-radius: 3px; white-space: nowrap; }
.etf-cat.sector { background: var(--blue2); color: var(--blue); }
.etf-cat.macro  { background: var(--purple2); color: var(--purple); }
.etf-cat.theme  { background: var(--amber2); color: var(--amber); }
.etf-main { flex: 1; }
.etf-name { font-size: 13px; font-weight: 500; }
.etf-name em { font-style: normal; color: var(--muted); font-size: 11px; margin-left: 6px; }
.etf-detail { font-size: 11px; color: var(--muted); margin-top: 2px; }
.etf-conf { font-family: var(--mono); font-size: 13px; font-weight: 700; }
.etf-pct { font-family: var(--mono); font-size: 11px; color: var(--muted); }
.etf-action-badge { font-family: var(--mono); font-size: 10px; padding: 2px 8px; border-radius: 3px; font-weight: 700; }
.etf-action-badge.buy  { background: var(--green); color: #000; }
.etf-action-badge.watch{ background: var(--amber); color: #000; }
.etf-market-bar { padding: 8px 14px; font-size: 11px; color: var(--muted); background: var(--bg3); border-radius: 5px; border: 1px solid var(--border); margin-bottom: 10px; font-family: var(--mono); }
```

---

## 7. 페이지 레이아웃

```
┌────────────────────────────────────────────────────────────┐
│ 퀀트 시스템 PRO                        ● 2026-03-24 20:56 │
├────────────────────────────────────────────────────────────┤
│ AI추천 67.7% │ 세력탐지 43.3% │ 거래량적중 36.1% │ ...    │
├──────────────────────────┬─────────────────────────────────┤
│ ● 오늘의 판단            │ ● 종목 추천                     │
│   AI 종합 분석    19:13  │   기술지표·수급·매집 전체 공개    │
│                          │                                 │
│ [시장] KOSPI 2480 (-1.2%)│   [종목 카드 1 — 풀 공개]       │
│ [시장국면] CAUTION → BEAR│   [종목 카드 2 — 풀 공개]       │
│ [AI분석] ...             │   [종목 카드 3 ...]             │
│                          │   ...최대 10개                  │
├──────────────────────────┴─────────────────────────────────┤
│ ● 포트폴리오 · 모의투자 성과                                │
├──────────────────────────┬─────────────────────────────────┤
│ ● 섹터 순환              │ ● 외국인·기관 수급               │
│   업종별 흐름 TOP 10     │   자금 흐름 분석                 │
├──────────────────────────┴─────────────────────────────────┤
│ ● ETF 추천 · 매크로·섹터·테마별 ETF                         │
├────────────────────────────────────────────────────────────┤
│ 퀀트 시스템 · 10단계 시그널엔진 · N개 시그널 가동           │
└────────────────────────────────────────────────────────────┘
```

---

## 8. 렌더링 JS (복붙 가능)

### 유틸리티

```javascript
function moneyStr(v) {
  if (!v) return '0';
  const abs = Math.abs(v);
  const sign = v >= 0 ? '+' : '-';
  if (abs >= 100000000) return sign + (abs / 100000000).toFixed(0) + '억';
  if (abs >= 10000) return sign + (abs / 10000).toFixed(0) + '만';
  return sign + abs.toLocaleString();
}
```

### Zone 2 렌더링

```javascript
function renderZone2(items) {
  const el = document.getElementById('zone2List');
  if (!items || !items.length) {
    el.innerHTML = '<div style="padding:12px;color:#5a6a82">오늘 액션 없음</div>';
    return;
  }

  el.innerHTML = items.map(a => {
    const act = (a.action || 'WATCH').toUpperCase();
    const cls = act === 'BUY' ? 'buy' : act === 'SELL' ? 'sell' : 'watch';
    const actKr = act === 'BUY' ? '매수' : act === 'SELL' ? '매도' : '관찰';
    const stratKr = a.strategy === 'AI_BRAIN' ? 'AI 판단' : '스캔 발굴';

    // 기술지표 칩
    let techChips = '';
    if (a.rsi) {
      const rc = a.rsi <= 30 ? 'good' : a.rsi >= 70 ? 'bad' : a.rsi <= 45 ? 'good' : '';
      techChips += `<span class="tech-chip ${rc}">RSI ${a.rsi}</span>`;
    }
    if (a.adx) techChips += `<span class="tech-chip ${a.adx >= 25 ? 'good' : ''}">ADX ${a.adx}</span>`;
    if (a.bb_position != null) {
      const bc = a.bb_position < 0.3 ? 'good' : a.bb_position > 0.8 ? 'bad' : '';
      techChips += `<span class="tech-chip ${bc}">BB ${(a.bb_position*100).toFixed(0)}%</span>`;
    }
    if (a.stoch_k) {
      const sc = a.stoch_k < 20 ? 'good' : a.stoch_k > 80 ? 'bad' : '';
      techChips += `<span class="tech-chip ${sc}">Stoch ${a.stoch_k}</span>`;
    }
    if (a.ma5_gap) techChips += `<span class="tech-chip">MA5 ${a.ma5_gap > 0 ? '+' : ''}${a.ma5_gap}%</span>`;
    if (a.above_ma60) techChips += `<span class="tech-chip good">MA60↑</span>`;
    if (a.sar_trend === -1) techChips += `<span class="tech-chip bad">SAR↓</span>`;
    else if (a.sar_trend === 1) techChips += `<span class="tech-chip good">SAR↑</span>`;

    // 태그
    let tags = '';
    if (a.accum_phase) tags += `<span class="sc-tag accum">${a.accum_phase} ${a.accum_days}일</span>`;
    if (a.safety_signal) {
      const sc = a.safety_signal === 'GREEN' ? 'safe-green' : a.safety_signal === 'YELLOW' ? 'safe-yellow' : 'safe-red';
      tags += `<span class="sc-tag ${sc}">안전: ${a.safety_label || a.safety_signal}</span>`;
    }
    if (a.ai_tag) tags += `<span class="sc-tag ai">${a.ai_tag}</span>`;
    if (a.foreign_5d) {
      tags += `<span class="sc-tag ${a.foreign_5d > 0 ? 'flow-buy' : 'flow-sell'}">외인5d ${moneyStr(a.foreign_5d)}</span>`;
    }
    if (a.inst_5d) {
      tags += `<span class="sc-tag ${a.inst_5d > 0 ? 'flow-buy' : 'flow-sell'}">기관5d ${moneyStr(a.inst_5d)}</span>`;
    }
    if (a.overheat_flags && a.overheat_flags.length) {
      tags += a.overheat_flags.map(f => `<span class="sc-tag safe-yellow">${f}</span>`).join('');
    }

    // 점수 분해
    let breakdown = '';
    if (a.score_breakdown) {
      const sb = a.score_breakdown;
      breakdown = `<div class="sc-breakdown">
        <span>기술분석 ${sb.tech}</span><span>수급 ${sb.flow}</span>
        <span>개별종목 ${sb.individual}</span><span>안전도 ${sb.safety}</span>
        <span>과열 ${sb.overheat}</span><span>복합 ${sb.multi}</span>
      </div>`;
    }

    // 가격
    let prices = '';
    if (a.close) prices += `<span>현재가 <b>${a.close.toLocaleString()}원</b> (${a.price_change > 0 ? '+' : ''}${a.price_change}%)</span>`;
    if (a.stop_loss) prices += `<span>손절 ${a.stop_loss.toLocaleString()}</span>`;
    if (a.target_price) prices += `<span>목표 ${a.target_price.toLocaleString()}</span>`;
    if (a.drawdown) prices += `<span>52주낙폭 ${a.drawdown}%</span>`;
    if (a.consensus_upside) prices += `<span>컨센 +${a.consensus_upside}%</span>`;

    return `<div class="stock-card ${cls}">
      <div class="sc-header">
        <span class="sc-grade">${a.grade}</span>
        <span class="sc-name">${a.name} <em>${a.ticker}</em></span>
        <span class="sc-badge">${actKr}</span>
        <span class="sc-score">${a.score}점</span>
        <span class="sc-strategy">${stratKr}</span>
      </div>
      <div class="sc-reason">${a.reason || ''}</div>
      <div class="sc-detail">
        ${prices ? `<div class="sc-prices">${prices}</div>` : ''}
        ${techChips ? `<div class="sc-tech">${techChips}</div>` : ''}
        ${tags ? `<div class="sc-tags">${tags}</div>` : ''}
        ${breakdown}
      </div>
    </div>`;
  }).join('');
}
```

### Zone 7 렌더링

```javascript
function renderZone7(items) {
  const el = document.getElementById('zone7Content');
  if (!items || !items.length) {
    el.innerHTML = '<div style="padding:12px;color:#5a6a82">ETF 추천 없음</div>';
    return;
  }

  let html = '';

  // 시장 방향 헤더
  if (items[0] && items[0]._market_direction) {
    const m = items[0];
    const dirKr = m._market_direction === 'BULL' ? '상승' : m._market_direction === 'BEAR' ? '하락' : '횡보';
    const dc = m._market_direction === 'BULL' ? '#00e59b' : m._market_direction === 'BEAR' ? '#ff4d6d' : '#f5a623';
    html += `<div class="etf-market-bar">
      시장 방향: <span style="color:${dc};font-weight:700">${dirKr}</span>
      (점수 ${m._market_score}, 신뢰도 ${m._market_confidence}%)
      · 레짐: ${m._regime} · VIX ${m._vix}
      ${m._reasons ? '<br>' + m._reasons.join(' · ') : ''}
    </div>`;
    items = items.slice(1);
  }

  html += '<div class="etf-list">';
  html += items.map(e => {
    const catCls = e.category === '섹터' ? 'sector' : e.category === '매크로' ? 'macro' : 'theme';
    const actCls = e.action === 'BUY' ? 'buy' : 'watch';
    const details = [];
    if (e.holding_period) details.push(`보유 ${e.holding_period}`);
    if (e.stop_loss) details.push(`손절: ${e.stop_loss}`);
    if (e.entry_timing) details.push(e.entry_timing);

    return `<div class="etf-card">
      <span class="etf-cat ${catCls}">${e.category}</span>
      <div class="etf-main">
        <div class="etf-name">${e.name} <em>${e.ticker}</em></div>
        <div class="etf-detail">${(e.reasons || []).join(' · ')}</div>
        ${details.length ? `<div class="etf-detail">${details.join(' · ')}</div>` : ''}
      </div>
      <span class="etf-action-badge ${actCls}">${e.action === 'BUY' ? '매수' : '관찰'}</span>
      <span class="etf-conf" style="color:${e.confidence >= 60 ? '#00e59b' : '#f5a623'}">신뢰 ${e.confidence}%</span>
      ${e.portfolio_pct ? `<span class="etf-pct">${e.portfolio_pct}%</span>` : ''}
    </div>`;
  }).join('');
  html += '</div>';

  el.innerHTML = html;
}
```

---

## 9. 레퍼런스 파일

| 파일 | 용도 |
|------|------|
| `quantum_master_dashboard.html` | 완전 동작하는 레퍼런스 (1,000줄, CSS+JS 인라인) |
| `website/data/dashboard_state.json` | 실제 데이터 샘플 (9종목 + ETF 4종목) |
| `dashboard_data.py` | JSON 생성 Python 스크립트 |

**`quantum_master_dashboard.html`을 그대로 FLOWX에 통합하면 됩니다.**

---

## 10. 체크리스트

- [ ] `dashboard_state.json` 배포 경로 설정
- [ ] 기존 "액션 리스트" 심플 테이블 → 풀 공개 stock-card 교체
- [ ] Zone 7 ETF 섹션 추가
- [ ] 모든 영어 라벨 → 한글 변환 (섹션 2 참조)
- [ ] Zone 4 섹터에 ETF 코드 표시 추가
- [ ] CSS 변수 충돌 확인
- [ ] 모바일 반응형 (필요 시)
