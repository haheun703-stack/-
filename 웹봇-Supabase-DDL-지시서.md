# 웹봇 Supabase DDL 지시서

> 퀀트봇에서 Row 테이블 4개 신규 빌더 + sector_rotation 스키마 수정 완료.
> 웹봇에서 아래 DDL을 Supabase에 실행해주세요.

## 1. 신규 테이블 4개 CREATE

### D-1. `dashboard_smart_money` (스마트머니 추적)

```sql
CREATE TABLE IF NOT EXISTS dashboard_smart_money (
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  name TEXT,
  sector TEXT,
  foreign_consec_days INT DEFAULT 0,
  inst_consec_days INT DEFAULT 0,
  foreign_net_5d FLOAT DEFAULT 0,
  inst_net_5d FLOAT DEFAULT 0,
  signal_type TEXT,          -- 'DUAL_BUY' / 'FOREIGN_BUY' / 'INST_BUY'
  price FLOAT DEFAULT 0,
  change_pct FLOAT DEFAULT 0,
  score FLOAT DEFAULT 0,
  PRIMARY KEY (date, ticker)
);

-- RLS (anon 읽기 허용)
ALTER TABLE dashboard_smart_money ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_read_smart_money" ON dashboard_smart_money
  FOR SELECT TO anon USING (true);
```

### D-2. `dashboard_etf_signals` (ETF 시그널)

```sql
CREATE TABLE IF NOT EXISTS dashboard_etf_signals (
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  name TEXT,
  sector TEXT,
  close FLOAT DEFAULT 0,
  change_pct FLOAT DEFAULT 0,
  aum FLOAT DEFAULT 0,
  aum_change FLOAT DEFAULT 0,
  aum_change_pct FLOAT DEFAULT 0,
  volume BIGINT DEFAULT 0,
  value FLOAT DEFAULT 0,
  signal_type TEXT,          -- '대량 자금유입'/'자금유입'/'강세 급등'/'강세'/'대량 자금유출'/'자금유출'/'약세 급락'/'약세'/'보합'
  score FLOAT DEFAULT 0,
  PRIMARY KEY (date, ticker)
);

ALTER TABLE dashboard_etf_signals ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_read_etf_signals" ON dashboard_etf_signals
  FOR SELECT TO anon USING (true);
```

### D-3. `dashboard_relay` (릴레이 시그널)

```sql
CREATE TABLE IF NOT EXISTS dashboard_relay (
  date TEXT NOT NULL,
  lead_sector TEXT NOT NULL,
  lag_sector TEXT NOT NULL,
  lead_return_1d FLOAT DEFAULT 0,
  lead_return_5d FLOAT DEFAULT 0,
  lead_breadth FLOAT DEFAULT 0,
  lag_return_1d FLOAT DEFAULT 0,
  lag_return_5d FLOAT DEFAULT 0,
  gap FLOAT DEFAULT 0,
  signal_type TEXT,          -- '강한 매수 기회'/'매수 기회'/'관심 구간'/'추격 진행중'/'선행 하락'/'대기'
  score FLOAT DEFAULT 0,
  PRIMARY KEY (date, lead_sector, lag_sector)
);

ALTER TABLE dashboard_relay ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_read_relay" ON dashboard_relay
  FOR SELECT TO anon USING (true);
```

### D-4. `dashboard_sniper` (스나이퍼 워치)

```sql
CREATE TABLE IF NOT EXISTS dashboard_sniper (
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  name TEXT,
  sector TEXT,
  close FLOAT DEFAULT 0,
  change_pct FLOAT DEFAULT 0,
  rsi FLOAT DEFAULT 0,
  ma20_gap FLOAT DEFAULT 0,
  bb_position FLOAT DEFAULT 0,    -- 0~1 (볼린저밴드 위치)
  adx FLOAT DEFAULT 0,
  foreign_days INT DEFAULT 0,
  inst_days INT DEFAULT 0,
  exec_strength FLOAT DEFAULT 0,  -- 체결강도 (0~100)
  vol_ratio FLOAT DEFAULT 0,
  signal_type TEXT,               -- '골든크로스'/'과매도 반등'/'수급 반전'/'볼밴 하단'/'추세 시작'
  score FLOAT DEFAULT 0,
  PRIMARY KEY (date, ticker)
);

ALTER TABLE dashboard_sniper ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_read_sniper" ON dashboard_sniper
  FOR SELECT TO anon USING (true);
```

---

## 2. 기존 테이블 ALTER: `sector_rotation`

기존 컬럼(`etf_code`, `signal`, `data`)을 지시서 D-5 스키마로 변경.

```sql
-- 기존 컬럼 삭제 (없으면 무시)
ALTER TABLE sector_rotation DROP COLUMN IF EXISTS etf_code;
ALTER TABLE sector_rotation DROP COLUMN IF EXISTS signal;
ALTER TABLE sector_rotation DROP COLUMN IF EXISTS data;

-- 신규 컬럼 추가 (없으면 추가)
ALTER TABLE sector_rotation ADD COLUMN IF NOT EXISTS score FLOAT DEFAULT 0;
ALTER TABLE sector_rotation ADD COLUMN IF NOT EXISTS ret_5d FLOAT DEFAULT 0;
ALTER TABLE sector_rotation ADD COLUMN IF NOT EXISTS ret_20d FLOAT DEFAULT 0;
ALTER TABLE sector_rotation ADD COLUMN IF NOT EXISTS momentum FLOAT DEFAULT 0;
ALTER TABLE sector_rotation ADD COLUMN IF NOT EXISTS flow FLOAT DEFAULT 0;
ALTER TABLE sector_rotation ADD COLUMN IF NOT EXISTS breadth FLOAT DEFAULT 0;

-- PK 확인: (date, sector) — 기존 PK가 다르면 재생성 필요
-- 현재 PK가 (date, rank, sector)이면:
-- ALTER TABLE sector_rotation DROP CONSTRAINT sector_rotation_pkey;
-- ALTER TABLE sector_rotation ADD PRIMARY KEY (date, sector);
```

### D-5. `dashboard_crash_bounce` (급락반등 포착기) — **신규 추가**

```sql
CREATE TABLE IF NOT EXISTS dashboard_crash_bounce (
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  name TEXT,
  close FLOAT DEFAULT 0,
  change_pct FLOAT DEFAULT 0,        -- 전일대비 등락률 (%)
  gap_20ma FLOAT DEFAULT 0,          -- 20일 이동평균 이격도 (%, 음수=급락)
  bb_position FLOAT DEFAULT 0,       -- 볼린저밴드 위치 (0~1, 음수=하단이탈)
  volume_ratio FLOAT DEFAULT 0,      -- 거래량 배수 (20일 평균 대비)
  foreign_net FLOAT DEFAULT 0,       -- 외국인 당일 순매수 (억원)
  inst_net FLOAT DEFAULT 0,          -- 기관 당일 순매수 (억원)
  foreign_days INT DEFAULT 0,        -- 외국인 연속 매수 일수
  inst_days INT DEFAULT 0,           -- 기관 연속 매수 일수
  signal_type TEXT,                  -- '복합급락 반등'/'볼린저급락 반등'/'거래량폭발 반등'/'관심'
  grade TEXT,                        -- '적극매수'/'매수'/'관심'
  score INT DEFAULT 0,               -- 0~100
  reasons TEXT,                      -- JSON 배열 문자열 (포착 이유)
  PRIMARY KEY (date, ticker)
);

ALTER TABLE dashboard_crash_bounce ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_read_crash_bounce" ON dashboard_crash_bounce
  FOR SELECT TO anon USING (true);
```

---

### D-6. `quant_nxt_picks` (🟡 NXT 주목 종목 — 퀀트시스템 메인)

```sql
CREATE TABLE IF NOT EXISTS quant_nxt_picks (
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  name TEXT,
  close INT DEFAULT 0,
  ret_d0 FLOAT DEFAULT 0,          -- 당일 수익률 (%)
  vol_ratio FLOAT DEFAULT 0,       -- 거래량/MA20 배수
  ma20_dev FLOAT DEFAULT 0,        -- 20MA 이격도 (%)
  rsi FLOAT DEFAULT 0,
  tv FLOAT DEFAULT 0,              -- 거래대금 (억원)
  foreign_net FLOAT DEFAULT 0,     -- 외인 당일 순매수 (억원)
  inst_net FLOAT DEFAULT 0,        -- 기관 당일 순매수 (억원)
  foreign_streak INT DEFAULT 0,    -- 외인 연속 매수 일수
  inst_streak INT DEFAULT 0,       -- 기관 연속 매수 일수
  dual_streak INT DEFAULT 0,       -- 쌍끌이 연속 일수
  foreign_cum FLOAT DEFAULT 0,     -- 외인 누적 순매수 (억원)
  inst_cum FLOAT DEFAULT 0,        -- 기관 누적 순매수 (억원)
  accum_score FLOAT DEFAULT 0,     -- 축적 점수 (0~100)
  final_score FLOAT DEFAULT 0,     -- 최종 점수 (0~100)
  PRIMARY KEY (date, ticker)
);

ALTER TABLE quant_nxt_picks ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_read_nxt_picks" ON quant_nxt_picks
  FOR SELECT TO anon USING (true);
```

### D-7. `quant_bottom_picks` (🟢 바닥에서 고개 든 종목)

```sql
CREATE TABLE IF NOT EXISTS quant_bottom_picks (
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  name TEXT,
  close INT DEFAULT 0,
  ret_d0 FLOAT DEFAULT 0,          -- 당일 수익률 (%)
  fib_zone TEXT,                    -- 'DEEP'/'BOTTOM'/'LOW'/'MID'/'HIGH'
  drop_pct FLOAT DEFAULT 0,        -- 52주 고점 대비 하락률 (%, 음수)
  vol_ratio FLOAT DEFAULT 0,       -- 거래량/MA20 배수
  tv FLOAT DEFAULT 0,              -- 거래대금 (억원)
  rsi FLOAT DEFAULT 0,
  foreign_turn BOOLEAN DEFAULT false, -- 외인 수급 양전환
  inst_turn BOOLEAN DEFAULT false,    -- 기관 수급 양전환
  supply_score FLOAT DEFAULT 0,    -- 수급 점수 (0~100)
  final_score FLOAT DEFAULT 0,     -- 최종 점수 (0~100)
  PRIMARY KEY (date, ticker)
);

ALTER TABLE quant_bottom_picks ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_read_bottom_picks" ON quant_bottom_picks
  FOR SELECT TO anon USING (true);
```

### D-8. `quant_etf_strategy` (📊 내일의 ETF 전략)

```sql
CREATE TABLE IF NOT EXISTS quant_etf_strategy (
  date TEXT NOT NULL PRIMARY KEY,
  regime TEXT,                      -- 'BULL'/'CAUTION'/'BEAR'/'CRISIS' 등
  shield TEXT,                      -- 'GREEN'/'YELLOW'/'RED'
  direction TEXT,                   -- 'BULL'/'NEUTRAL'/'BEAR'
  vix FLOAT DEFAULT 0,
  fear_index FLOAT DEFAULT 0,
  contrarian BOOLEAN DEFAULT false,
  bull_etfs JSONB,                  -- [{ticker, name, desc}] 레버리지/상승 ETF
  bear_etfs JSONB,                  -- [{ticker, name, desc}] 인버스/곱버스 ETF
  safe_etfs JSONB,                  -- [{ticker, name, desc}] 금/채권/달러 ETF
  message TEXT                      -- 코멘트 (공포에 떨지 마세요 등)
);

ALTER TABLE quant_etf_strategy ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_read_etf_strategy" ON quant_etf_strategy
  FOR SELECT TO anon USING (true);
```

---

## 3. 퀀트시스템 메인 화면 UI 설계

### 탭 구성: 퀀트시스템 (1탭에 3박스)

```
┌─────────────────────────────────────────────────────┐
│  🟡 NXT 주목 종목 (형광 노란색 #FFD700)              │
│  ─────────────────────────────────────              │
│  quant_nxt_picks → final_score DESC                 │
│  없는 날: "주목 종목 없음 — 쉬는 것도 전략입니다"      │
├─────────────────────────────────────────────────────┤
│  🟢 바닥에서 고개 든 종목 (초록)                      │
│  ─────────────────────────────────────              │
│  quant_bottom_picks → final_score DESC              │
├─────────────────────────────────────────────────────┤
│  📊 내일의 ETF 전략                                  │
│  ─────────────────────────────────────              │
│  quant_etf_strategy → 최신 1행                      │
│  direction에 따라 bull_etfs/bear_etfs 강조           │
│  message 표시 (공포 대응 포함)                        │
└─────────────────────────────────────────────────────┘
```

### 색상 가이드
- NXT 주목: `#FFD700` (형광 노란색)
- 바닥 반등: `#4CAF50` (초록)
- ETF 전략: `#2196F3` (파란색)
- 공포 메시지: `#FF5722` (빨간/주황)

### ETF 전략 UI 로직
- `direction == "BULL"` → 오를 때 ETF 강조 (bull_etfs)
- `direction == "BEAR"` → 내릴 때 ETF 강조 (bear_etfs)
- `direction == "NEUTRAL"` → 양쪽 동일하게 표시 + 안전자산(safe_etfs) 추가
- `contrarian == true` → 🔥 역발상 배지 표시
- `message` → 하단에 큰 글씨로 표시

---

## 4. 퀀트봇 업로드 현황

| 테이블 | 상태 | 데이터 건수 | 비고 |
|--------|------|------------|------|
| `dashboard_smart_money` | 운영중 | ~50건/일 | accumulation_alert 기반 |
| `dashboard_etf_signals` | 운영중 | ~20건/일 | 섹터 ETF 20개 |
| `dashboard_relay` | 운영중 | 0~10건/일 | 발화 시에만 데이터 |
| `dashboard_sniper` | 운영중 | ~30건/일 | pullback_scan 기반 |
| `dashboard_crash_bounce` | 운영중 | 0~50건/일 | 급락반등 포착 |
| `sector_rotation` | 운영중 | ~20건/일 | 섹터 로테이션 |
| **`quant_nxt_picks`** | **신규** | 0~30건/일 | 🟡 NXT 주목 (수급 릴레이 시드) |
| **`quant_bottom_picks`** | **신규** | 0~30건/일 | 🟢 바닥 반등 (피보나치 바닥+빨간봉) |
| **`quant_etf_strategy`** | **신규** | 1건/일 | 📊 ETF 전략 (Brain/Shield 기반) |

## 5. 퀀트봇 업로드 코드

- `upload_all_quant_tables()` → 14테이블 일괄 업로드 (JSONB 6 + Row 5 + 메인 3)
- BAT-D G4 단계에서 자동 실행

## 6. 프론트엔드 API 연동

| 테이블 | API 엔드포인트 | 쿼리 |
|--------|--------------|------|
| `dashboard_smart_money` | `/api/smart-money` | `SELECT * WHERE date = (최신) ORDER BY score DESC LIMIT 50` |
| `dashboard_etf_signals` | `/api/etf-signals` | `SELECT * WHERE date = (최신) ORDER BY score DESC` |
| `dashboard_relay` | `/api/relay` | `SELECT * WHERE date = (최신) ORDER BY score DESC` |
| `dashboard_sniper` | `/api/sniper` | `SELECT * WHERE date = (최신) ORDER BY score DESC LIMIT 100` |
| `dashboard_crash_bounce` | `/api/crash-bounce` | `SELECT * WHERE date = (최신) ORDER BY score DESC LIMIT 50` |
| `sector_rotation` | `/api/sector-rotation` | `SELECT * WHERE date = (최신) ORDER BY rank ASC` |
| **`quant_nxt_picks`** | **`/api/nxt-picks`** | `SELECT * WHERE date = (최신) ORDER BY final_score DESC` |
| **`quant_bottom_picks`** | **`/api/bottom-picks`** | `SELECT * WHERE date = (최신) ORDER BY final_score DESC` |
| **`quant_etf_strategy`** | **`/api/etf-strategy`** | `SELECT * WHERE date = (최신) LIMIT 1` |
