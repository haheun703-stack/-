# JARVIS Control Tower — 차트 모듈 설계서

> **목적**: ppwangga.com Control Tower에 네이버금융 스타일 인터랙티브 대시보드 추가
> **스택**: Flask + SQLite (기존) + Plotly.js + Tailwind CSS
> **날짜**: 2025-03-05

---

## 1. 전체 구조 (네이버금융 스타일 네비게이션)

```
Control Tower (ppwangga.com)
├── 기존 페이지들 (캘린더, 시그널 등)
│
└── 📊 성과 대시보드 (/dashboard)
    ├── [탭1] 종합 Overview ← 메인 랜딩
    ├── [탭2] 봇1: 포물선 v10.3
    ├── [탭3] 봇2: 그룹ETF 순환매
    ├── [탭4] 봇3: (세번째 봇 이름)
    └── [탭5] 매매 내역 (전체)
```

### 네비게이션 UX
- 상단: 기존 Control Tower 메뉴바
- 서브탭: 가로 탭 바 (종합 | 봇1 | 봇2 | 봇3 | 매매내역)
- 각 탭 안에서: 기간 필터 버튼 (1주 | 1개월 | 3개월 | 6개월 | 1년 | 전체)
- 클릭으로 drill-down: 차트 포인트 클릭 → 해당 일자 매매 상세로 이동

---

## 2. 데이터 파이프라인

### 2-1. 현재 상태
```
봇1 (로컬PC) → SQLite/CSV (봇1 자체 DB)
봇2 (로컬PC) → SQLite/CSV (봇2 자체 DB)
봇3 (로컬PC) → SQLite/CSV (봇3 자체 DB)
```

### 2-2. 목표 구조
```
봇1 ──┐                    ┌──────────────────────┐
봇2 ──┼── API POST ──────→ │ Control Tower Server │
봇3 ──┘   (매일 장마감 후)   │ (ppwangga.com)       │
                            │                      │
                            │ central.db (SQLite)   │
                            │  ├─ trades            │
                            │  ├─ daily_returns     │
                            │  └─ bot_summary       │
                            └──────────────────────┘
```

### 2-3. 중앙 DB 스키마

```sql
-- 매매 내역 (개별 거래)
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_name TEXT NOT NULL,        -- 'v10.3_swing', 'group_etf', 'bot3'
    trade_date TEXT NOT NULL,      -- 'YYYY-MM-DD'
    stock_code TEXT NOT NULL,      -- '005930'
    stock_name TEXT NOT NULL,      -- '삼성전자'
    side TEXT NOT NULL,            -- 'BUY' / 'SELL'
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    amount REAL NOT NULL,          -- quantity * price
    pnl REAL,                      -- 실현손익 (SELL 시에만)
    pnl_pct REAL,                  -- 수익률 % (SELL 시에만)
    created_at TEXT DEFAULT (datetime('now'))
);

-- 일별 성과 (봇별)
CREATE TABLE daily_returns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_name TEXT NOT NULL,
    date TEXT NOT NULL,             -- 'YYYY-MM-DD'
    daily_return_pct REAL NOT NULL, -- 당일 수익률 %
    cumulative_pct REAL NOT NULL,   -- 누적 수익률 %
    portfolio_value REAL,           -- 평가금액
    cash REAL,                      -- 현금잔고
    drawdown_pct REAL,              -- 당일 DD %
    max_drawdown_pct REAL,          -- 누적 MDD %
    UNIQUE(bot_name, date)
);

-- 주간/월간 요약 (자동 집계용 뷰 또는 테이블)
CREATE TABLE period_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_name TEXT NOT NULL,
    period_type TEXT NOT NULL,      -- 'weekly' / 'monthly'
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    return_pct REAL,               -- 기간 수익률
    win_count INTEGER,             -- 익절 횟수
    lose_count INTEGER,            -- 손절 횟수
    win_rate REAL,                 -- 승률
    profit_factor REAL,            -- PF
    max_drawdown_pct REAL,         -- 기간 MDD
    avg_hold_days REAL,            -- 평균 보유일
    total_trades INTEGER,          -- 총 거래 수
    UNIQUE(bot_name, period_type, period_start)
);

-- 봇 메타 정보
CREATE TABLE bot_info (
    bot_name TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,     -- '포물선 v10.3 스윙'
    description TEXT,
    allocation_pct REAL,            -- 배분 비율 (60%, 40% 등)
    start_date TEXT,                -- 운용 시작일
    status TEXT DEFAULT 'active'    -- 'active' / 'paused' / 'stopped'
);
```

### 2-4. 데이터 업로드 API (Flask 엔드포인트)

```python
# 각 봇이 장마감 후 호출하는 API
@app.route('/api/upload/trades', methods=['POST'])
def upload_trades():
    """
    POST body:
    {
        "bot_name": "v10.3_swing",
        "trades": [
            {
                "trade_date": "2025-03-05",
                "stock_code": "005930",
                "stock_name": "삼성전자",
                "side": "SELL",
                "quantity": 10,
                "price": 82000,
                "amount": 820000,
                "pnl": 50000,
                "pnl_pct": 6.5
            }
        ]
    }
    """
    pass

@app.route('/api/upload/daily', methods=['POST'])
def upload_daily_return():
    """
    POST body:
    {
        "bot_name": "v10.3_swing",
        "date": "2025-03-05",
        "daily_return_pct": 0.8,
        "cumulative_pct": 12.5,
        "portfolio_value": 11250000,
        "cash": 3200000,
        "drawdown_pct": -0.3,
        "max_drawdown_pct": -4.5
    }
    """
    pass
```

---

## 3. 페이지별 상세 설계

### 3-1. [탭1] 종합 Overview

**레이아웃:**
```
┌─────────────────────────────────────────────────┐
│  종합 | 봇1 | 봇2 | 봇3 | 매매내역              │  ← 탭 바
├─────────────────────────────────────────────────┤
│                                                 │
│  [KPI 카드 4개 가로 배치]                         │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐            │
│  │총수익률│ │ MDD  │ │ 승률  │ │  PF  │            │
│  │+12.5%│ │-4.2% │ │ 68%  │ │ 1.85 │            │
│  └──────┘ └──────┘ └──────┘ └──────┘            │
│                                                 │
│  [Plotly 차트: 봇 3개 누적수익률 곡선 오버레이]     │
│  ───── v10.3 스윙                                │
│  ───── 그룹ETF 순환매                             │
│  ───── 봇3                                       │
│  ───── 합산 포트폴리오                             │
│                                                 │
│  기간: [1주] [1개월] [3개월] [6개월] [1년] [전체]   │
│                                                 │
│  [최근 매매 테이블 - 최근 10건]                     │
│  날짜 | 봇 | 종목 | 매수/매도 | 수익률             │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Plotly 차트 사양:**
- 타입: `scatter` (line mode)
- X축: 날짜
- Y축: 누적 수익률 (%)
- 봇별 색상: v10.3=#00D4AA, 그룹ETF=#FF6B6B, 봇3=#4ECDC4, 합산=#FFD93D
- hover: 날짜, 봇 이름, 수익률, 평가금액
- 줌/팬 가능, rangeslider 포함
- 0% 기준선 표시 (회색 대시)

### 3-2. [탭2~4] 개별 봇 상세

**레이아웃:**
```
┌─────────────────────────────────────────────────┐
│  [봇 이름 + 상태 뱃지(🟢운용중)]                   │
│  배분: 60% | 운용시작: 2025-02-19                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  [KPI 카드 6개]                                   │
│  총수익률 | MDD | 승률 | PF | 평균보유일 | 총거래수 │
│                                                 │
│  ── 차트 영역 ──                                  │
│                                                 │
│  [서브탭] 수익률곡선 | Drawdown | 월별히트맵        │
│                                                 │
│  <수익률곡선>                                     │
│  - 누적수익률 라인 + 일별수익률 바 차트 듀얼축       │
│                                                 │
│  <Drawdown>                                     │
│  - Drawdown % area 차트 (빨간 fill)               │
│                                                 │
│  <월별히트맵>                                     │
│  - 월별 수익률 heatmap (초록=+, 빨강=-)            │
│  - 행: 연도, 열: 1~12월                           │
│                                                 │
│  ── 주간/월간 성과 테이블 ──                       │
│  [주간] [월간] 토글                               │
│  기간 | 수익률 | 승률 | PF | MDD | 거래수          │
│                                                 │
│  ── 보유 종목 현황 ──                              │
│  종목명 | 매수가 | 현재가 | 수익률 | 보유일          │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Plotly 차트 사양:**

수익률 곡선 (듀얼축):
```javascript
// trace1: 누적수익률 (좌축, 라인)
{
    x: dates, y: cumulative_pct,
    type: 'scatter', mode: 'lines',
    name: '누적수익률',
    yaxis: 'y1'
}
// trace2: 일별수익률 (우축, 바)
{
    x: dates, y: daily_return_pct,
    type: 'bar', name: '일별수익률',
    marker: { color: daily_return_pct.map(v => v >= 0 ? '#00D4AA' : '#FF6B6B') },
    yaxis: 'y2'
}
```

Drawdown 차트:
```javascript
{
    x: dates, y: drawdown_pct,
    type: 'scatter', fill: 'tozeroy',
    fillcolor: 'rgba(255,107,107,0.3)',
    line: { color: '#FF6B6B' }
}
```

월별 히트맵:
```javascript
{
    z: monthly_returns_matrix,  // [연도][월]
    x: ['1월','2월',...,'12월'],
    y: ['2025', '2026'],
    type: 'heatmap',
    colorscale: [[0,'#FF6B6B'],[0.5,'#1a1a2e'],[1,'#00D4AA']],
    zmid: 0
}
```

### 3-3. [탭5] 매매 내역

**레이아웃:**
```
┌─────────────────────────────────────────────────┐
│  [필터 바]                                       │
│  봇: [전체▼] 기간: [시작일]~[종료일]  종목: [검색]  │
│  매매: [전체|매수|매도]  결과: [전체|익절|손절]      │
│                                                 │
│  [매매 내역 테이블 - Plotly Table or DataTable]    │
│  #  | 날짜 | 봇 | 종목 | 매수/매도 | 수량 |       │
│     | 단가 | 금액 | 수익률 | 손익금액              │
│                                                 │
│  페이지네이션: < 1 2 3 4 5 ... 20 >               │
│                                                 │
│  [하단 통계]                                      │
│  필터 적용 기준: 총 N건 | 평균수익률 X% |           │
│  익절 N건 / 손절 N건 | 총손익 ±XXXX원              │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 4. 프론트엔드 기술 스택

```
Plotly.js (CDN)     ← 인터랙티브 차트
Tailwind CSS (CDN)  ← JARVIS 다크 테마 스타일링
Jinja2 템플릿        ← Flask 기존 구조 유지
Vanilla JS          ← 탭 전환, 필터, API 호출
```

### 4-1. 다크 테마 컬러 팔레트 (기존 JARVIS 테마)

```css
:root {
    --bg-primary: #0d1117;      /* 메인 배경 */
    --bg-card: #161b22;         /* 카드/패널 */
    --bg-hover: #21262d;        /* hover */
    --border: #30363d;          /* 테두리 */
    --text-primary: #e6edf3;    /* 주 텍스트 */
    --text-secondary: #8b949e;  /* 보조 텍스트 */
    --accent-green: #00D4AA;    /* 수익 / 양봉 */
    --accent-red: #FF6B6B;      /* 손실 / 음봉 */
    --accent-yellow: #FFD93D;   /* 경고 / 합산 */
    --accent-blue: #4ECDC4;     /* 보조 강조 */
}
```

### 4-2. Plotly 다크 레이아웃 공통 설정

```javascript
const JARVIS_LAYOUT = {
    paper_bgcolor: '#0d1117',
    plot_bgcolor: '#161b22',
    font: { color: '#e6edf3', family: 'Pretendard, sans-serif' },
    xaxis: {
        gridcolor: '#21262d',
        linecolor: '#30363d',
        rangeslider: { visible: true }
    },
    yaxis: {
        gridcolor: '#21262d',
        linecolor: '#30363d',
        zeroline: true,
        zerolinecolor: '#30363d'
    },
    hoverlabel: {
        bgcolor: '#161b22',
        bordercolor: '#30363d',
        font: { color: '#e6edf3' }
    },
    legend: {
        bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#8b949e' }
    },
    margin: { l: 60, r: 30, t: 40, b: 40 }
};
```

---

## 5. Flask 라우트 구조

```python
# ── 페이지 라우트 ──
@app.route('/dashboard')
@app.route('/dashboard/<tab>')           # tab = overview, bot1, bot2, bot3, trades
def dashboard(tab='overview'):
    return render_template('dashboard.html', active_tab=tab)

# ── 차트 데이터 API (JSON) ──
@app.route('/api/chart/cumulative')      # 누적수익률 (전체 봇)
@app.route('/api/chart/cumulative/<bot>') # 누적수익률 (개별 봇)
@app.route('/api/chart/drawdown/<bot>')   # Drawdown
@app.route('/api/chart/monthly/<bot>')    # 월별 히트맵 데이터
@app.route('/api/chart/daily/<bot>')      # 일별 수익률 바
@app.route('/api/kpi/<bot>')              # KPI 카드 데이터
@app.route('/api/kpi/total')              # 종합 KPI

# ── 테이블 데이터 API ──
@app.route('/api/trades')                 # 매매내역 (필터 params: bot, start, end, side)
@app.route('/api/summary/<period>')       # 주간/월간 요약 (period = weekly/monthly)
@app.route('/api/holdings/<bot>')         # 현재 보유종목

# ── 데이터 업로드 API (봇 → 서버) ──
@app.route('/api/upload/trades', methods=['POST'])
@app.route('/api/upload/daily', methods=['POST'])
```

---

## 6. 봇 → 서버 데이터 전송 스크립트

각 봇에 추가할 업로더 모듈 (장마감 후 자동 실행):

```python
# bot_uploader.py — 각 봇 프로젝트에 추가
import requests
import json

CONTROL_TOWER_URL = "https://ppwangga.com"
BOT_NAME = "v10.3_swing"  # 봇마다 변경

def upload_daily_return(date, daily_pct, cum_pct, port_value, cash, dd, mdd):
    """장마감 후 일일 성과 업로드"""
    payload = {
        "bot_name": BOT_NAME,
        "date": date,
        "daily_return_pct": daily_pct,
        "cumulative_pct": cum_pct,
        "portfolio_value": port_value,
        "cash": cash,
        "drawdown_pct": dd,
        "max_drawdown_pct": mdd
    }
    resp = requests.post(f"{CONTROL_TOWER_URL}/api/upload/daily", json=payload)
    return resp.status_code == 200

def upload_trades(date, trades_list):
    """당일 매매 내역 업로드"""
    payload = {
        "bot_name": BOT_NAME,
        "trades": trades_list
    }
    resp = requests.post(f"{CONTROL_TOWER_URL}/api/upload/trades", json=payload)
    return resp.status_code == 200
```

---

## 7. 구현 순서 (Claude Code 작업 단계)

```
Phase 1: DB + API (1일)
  ├─ central.db 스키마 생성
  ├─ /api/upload/* 엔드포인트 구현
  ├─ /api/chart/* 데이터 조회 API 구현
  └─ 더미 데이터 시드 스크립트

Phase 2: 종합 Overview 페이지 (1일)
  ├─ dashboard.html 템플릿 (탭 구조)
  ├─ KPI 카드 컴포넌트
  ├─ 누적수익률 오버레이 차트
  └─ 최근 매매 테이블

Phase 3: 개별 봇 상세 페이지 (1일)
  ├─ 듀얼축 수익률 차트
  ├─ Drawdown 차트
  ├─ 월별 히트맵
  └─ 주간/월간 성과 테이블

Phase 4: 매매 내역 페이지 (0.5일)
  ├─ 필터 바 UI
  ├─ 매매 테이블 + 페이지네이션
  └─ 하단 통계 집계

Phase 5: 봇 연동 (0.5일)
  ├─ bot_uploader.py 각 봇에 배포
  ├─ 기존 데이터 마이그레이션 스크립트
  └─ 스케줄러 연동 (장마감 후 자동)
```

---

## 8. 기존 데이터 마이그레이션

각 봇의 로컬 SQLite/CSV에서 central.db로 초기 데이터 이관:

```python
# migrate_bot_data.py
"""
사용법:
  python migrate_bot_data.py \
    --bot-name v10.3_swing \
    --source-db /path/to/bot1/trades.db \
    --target-db /path/to/central.db \
    --source-type sqlite  # 또는 csv

봇별 소스 DB 구조가 다를 수 있으므로
컬럼 매핑을 봇별로 설정해야 함
"""
```

---

## 9. 보안 고려

```python
# API 인증 (간단한 토큰 방식)
API_TOKEN = os.environ.get('JARVIS_API_TOKEN')

@app.before_request
def check_api_auth():
    if request.path.startswith('/api/upload'):
        token = request.headers.get('Authorization')
        if token != f'Bearer {API_TOKEN}':
            return jsonify({'error': 'unauthorized'}), 401
```

---

## 10. 향후 확장

- **NIGHTWATCH 연동**: 나이트워치 🟢/🟡/🔴 시그널을 대시보드에 오버레이
- **알림 연동**: MDD 임계치 돌파 시 텔레그램 + 대시보드 배너 경고
- **벤치마크 비교**: KOSPI 수익률 라인을 차트에 함께 표시
- **JGIS 레이어**: 글로벌 시그널 상태를 사이드바로 표시
