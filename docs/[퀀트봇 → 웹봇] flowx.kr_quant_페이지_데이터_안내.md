# [퀀트봇 → 웹봇] flowx.kr/quant 페이지 데이터 안내 v1

**작성일**: 2026-05-16 (토)
**작성자**: 퀀트봇 (quantum-master)
**대상**: 웹봇 (FLOWX 페이지 운영자)
**긴급도**: ★★★★ — 5/18 (월) 첫 paper 실전 결과 노출 필요

---

## 1. 요약

퀀트봇이 매일 평일 종가 후 매매 결과 + ETF + Bluechip + 위험감지 메타를 **Supabase에 모두 푸시**하고 있습니다. 하지만 `flowx.kr/quant` 페이지가 이 데이터를 조회·렌더링하지 않는 상태입니다.

이 문서는 페이지 컴포넌트에 표시해야 할 **4개 섹션** + 각 섹션별 Supabase 쿼리 + UI 권장사항을 제공합니다.

---

## 2. 표시해야 할 4개 섹션

### 🎯 섹션 1: 시장 위험도 헤더 카드 (최상단)

매일 16:49 정보봇이 갱신하는 한국시장 위험점수를 페이지 최상단에 카드로.

**데이터 소스**: `macro_risk_daily` 테이블 (정보봇 관할, 정보봇 가이드 참조)

**Supabase 쿼리**:
```typescript
const { data } = await supabase
  .from('macro_risk_daily')
  .select('*')
  .order('date', { ascending: false })
  .limit(1)
  .single();
```

**렌더링 권장**:
```
┌─────────────────────────────────────────────────────────┐
│ 🛡️ 한국시장 위험도        2026-05-15                    │
├─────────────────────────────────────────────────────────┤
│  [큰 등급] 위험 (78점)  매수금액 ×0.4                    │
│                                                          │
│  외부:15  외인:25  이벤트:14  디커플:24                  │
│                                                          │
│  🚨 핵심 시그널:                                          │
│   • 환율 1461.1원 강세                                   │
│   • 원자재 3종 동반 -2% 하락                             │
│   • 외국인 일일 5.2조원 매도                             │
│                                                          │
│  👉 권장: 신규 진입 40%로 축소, 단타봇 진입 중단         │
└─────────────────────────────────────────────────────────┘
```

**색상 가이드** (level 컬럼 기준):
- NORMAL (정상): 녹색 `#10b981`
- CAUTION (주의): 연두색 `#84cc16`
- WARNING (경고): 노랑 `#eab308`
- DANGER (위험): 주황 `#f97316`
- CRISIS (위기): 빨강 `#ef4444`

**컬럼 명세**:
| 컬럼 | 타입 | 용도 |
|---|---|---|
| date | text | 일자 (YYYY-MM-DD) |
| total_score | int | 0~100 위험점수 |
| level | text | NORMAL/CAUTION/WARNING/DANGER/CRISIS |
| level_kr | text | 정상/주의/경고/위험/위기 |
| external_score | int | 외부환경 점수 |
| foreign_flow_score | int | 외인수급 점수 |
| event_score | int | 이벤트 점수 |
| decoupling_score | int | 디커플링 점수 |
| key_signals | jsonb (list[text]) | 핵심 시그널 텍스트 배열 |
| recommended_action | text | 권장 행동 문구 |

---

### 📊 섹션 2: 페이퍼 트레이딩 매매 내역 (메인 테이블)

퀀트봇이 매일 종가 후 paper_trades 테이블에 INSERT.

**데이터 소스**: `paper_trades` 테이블

**Supabase 쿼리** (최근 30일):
```typescript
const { data } = await supabase
  .from('paper_trades')
  .select('*')
  .gte('trade_date', '2026-04-16')  // 30일 전부터
  .order('created_at', { ascending: false });
```

**컬럼 명세**:
| 컬럼 | 타입 | 예시 | 설명 |
|---|---|---|---|
| id | uuid | - | PK |
| trade_date | date | 2026-05-15 | 거래일 |
| code | text | "005930" | 종목코드 (zfill 6자리) |
| name | text | "삼성전자" | 종목명 |
| side | text | BUY / SELL | 매수/매도 |
| price | numeric | 73500 | 가격 |
| quantity | int | 100 | 수량 |
| pnl_pct | numeric (nullable) | +2.5 | SELL일 때 손익률 % |
| **strategy** | text | "SCAN" / "BLUECHIP_A_쌍끌이" / "ETF_LONG_BUY" | **분류 핵심 컬럼** |
| cumulative_pf | numeric | 1.85 | 누적 PF |
| cumulative_mdd | numeric | -8.2 | 누적 MDD |
| win_rate | numeric | 62.5 | 누적 승률 % |
| **memo** | text | "등급:AA \| 위험:위험(78점) ×0.4" | **위험감지 메타 포함** |
| created_at | timestamptz | - | 입력 시각 |

#### Strategy 분류 (필터/탭 권장)

`strategy` 컬럼으로 3개 봇 활동을 구분:

| Strategy 패턴 | 봇 | 의미 |
|---|---|---|
| `SCAN`, `REBALANCE`, `ALPHA`, `INTRADAY_LEARNED` | Paper | 일반 페이퍼 매매 |
| `BLUECHIP_*` (예: `BLUECHIP_A_쌍끌이`, `BLUECHIP_TRAILING_STOP`) | Bluechip | 우량주 TOP 30 |
| `ETF_*` (예: `ETF_LONG_BUY`, `ETF_STRONG_LONG_SWITCH`, `ETF_STOP_LOSS`, `INVERSE_MAX_HOLD`) | ETF 방향 | JARVIS 기반 |
| `TAKE_PROFIT_T1`, `TAKE_PROFIT_T2`, `TRAILING_STOP`, `STOP_LOSS`, `MAX_HOLD` | 공통 | 청산 사유 |

#### memo 필드 파싱 가이드

5/18부터 모든 신규 매매의 memo 끝에 `| 위험:{등급}({점수}점) ×{배수}` 자동 첨부:

예시:
- `"등급:AA | 위험:위험(78점) ×0.4"`
- `"부분 | 위험:정상(15점) ×1.0"`
- `"JARVIS | 위험:경고(55점) ×0.6"`

→ 매매 행 옆에 위험 등급 칩(chip) 표시 권장.

**렌더링 권장**:
```
┌──────────────────────────────────────────────────────────────────┐
│ 📊 페이퍼 매매 내역                  [전체|Paper|Bluechip|ETF] ▼ │
├──────────────────────────────────────────────────────────────────┤
│ 일자        종목         사이드  가격       수량  손익   전략     │
│ 05/15 09:33 🟠로보스타   BUY    88,800    50   -    SCAN        │
│ 05/15 09:33 🟠GS리테일   BUY    29,450    152  -    SCAN        │
│ 05/15 09:33 🟠LG전자우   SELL   84,016    -    +2.1 REBALANCE   │
│   ↑ 🟠 = 그날 위험등급 (memo 파싱)                                │
└──────────────────────────────────────────────────────────────────┘
```

---

### 🎯 섹션 3: 시그널/전략별 누적 적중률 (분석 카드)

`paper_trades` 테이블에서 `strategy`별 집계 → 효과 측정.

**Supabase 쿼리** (SQL 함수 또는 클라이언트 집계):
```sql
SELECT
  strategy,
  COUNT(*) FILTER (WHERE side='SELL') as total_exits,
  COUNT(*) FILTER (WHERE side='SELL' AND pnl_pct > 0) as wins,
  AVG(pnl_pct) FILTER (WHERE side='SELL') as avg_pnl,
  SUM(pnl_pct) FILTER (WHERE side='SELL' AND pnl_pct > 0) /
    NULLIF(ABS(SUM(pnl_pct) FILTER (WHERE side='SELL' AND pnl_pct < 0)), 0) as pf
FROM paper_trades
WHERE trade_date >= NOW() - INTERVAL '90 days'
GROUP BY strategy
HAVING COUNT(*) FILTER (WHERE side='SELL') >= 3
ORDER BY pf DESC NULLS LAST;
```

**렌더링 권장**:
```
┌─────────────────────────────────────────────────────────┐
│ 🎯 전략별 누적 적중률 (90일)                             │
├─────────────────────────────────────────────────────────┤
│ 전략                  거래  WR    PF    평균수익  배지   │
│ BLUECHIP_A_쌍끌이     24    67%   2.42  +3.8%    🟢    │
│ SCAN                  18    61%   1.95  +2.5%    🟢    │
│ INTRADAY_LEARNED      5     60%   1.28  +1.4%    🟡    │
│ TRAILING_STOP         8     38%   0.72  -0.9%    🔴    │
└─────────────────────────────────────────────────────────┘
```

**색상**:
- PF ≥ 1.5 → 녹색 🟢
- PF 1.0~1.5 → 노랑 🟡
- PF < 1.0 → 빨강 🔴

---

### 📈 섹션 4: 자산 곡선 (라인차트)

paper_portfolio.json 측 daily_equity는 아직 Supabase로 푸시 안 됨.

**옵션 A (즉시)**: paper_trades에서 합성 — `(BUY 비용 누계 - SELL 수익 누계)` 추적

**옵션 B (개선)**: 별도 테이블 `paper_daily_equity` 신설 후 일별 자산 푸시 — 퀀트봇 측에서 추가 가능

→ **웹봇 측이 옵션 A로 시작 권장**. 옵션 B 필요 시 퀀트봇에 요청.

---

## 3. 페이지 레이아웃 권장 (와이어프레임)

```
╔══════════════════════════════════════════════════════════╗
║  FLOWX                                              ☰    ║
╠══════════════════════════════════════════════════════════╣
║  [홈] [퀀트] [정보] [단타] ...                            ║
╠══════════════════════════════════════════════════════════╣
║                                                           ║
║  ╔══ 섹션 1: 시장 위험도 ══════════════════════════════╗ ║
║  ║ 🟠 위험 (78점)  ×0.4  | 외부15 외인25 이벤14 디커24║ ║
║  ║ 🚨 환율 1461원 / 외인 5.2조 매도 / 원자재 -2%      ║ ║
║  ║ 👉 권장: 신규 진입 40% 축소                         ║ ║
║  ╚════════════════════════════════════════════════════╝ ║
║                                                           ║
║  ╔══ 섹션 4: 자산 곡선 ════════════════════════════════╗ ║
║  ║  [Paper/Bluechip/ETF 라인차트]                      ║ ║
║  ╚════════════════════════════════════════════════════╝ ║
║                                                           ║
║  ╔══ 섹션 3: 전략별 누적 적중률 ═══════════════════════╗ ║
║  ║  [표 또는 카드들]                                    ║ ║
║  ╚════════════════════════════════════════════════════╝ ║
║                                                           ║
║  ╔══ 섹션 2: 매매 내역 (최신순) ═══════════════════════╗ ║
║  ║  [필터: 전체/Paper/Bluechip/ETF] [날짜 범위]        ║ ║
║  ║  [테이블 — 페이지네이션]                             ║ ║
║  ╚════════════════════════════════════════════════════╝ ║
╚══════════════════════════════════════════════════════════╝
```

---

## 4. 데이터 갱신 주기

| 데이터 | 갱신 시각 (KST, 평일) | 출처 |
|---|---|---|
| macro_risk_daily | 16:49 | 정보봇 |
| paper_trades (paper) | 15:35~ | 퀀트봇 paper_trading_unified |
| paper_trades (bluechip) | 15:35~ | 퀀트봇 bluechip_timing |
| paper_trades (ETF) | 15:35~ | 퀀트봇 manage_etf_position |
| etf_signals | 16:35~ | 퀀트봇 BAT-F |
| foreign_flow | 16:35~ | 퀀트봇 BAT-F |

페이지 새로고침: **5분 간격 자동 refetch** 권장 (Supabase realtime subscription도 가능).

---

## 5. 실시간 검증 — 5/18 (월) 이후

5/18 paper_trading_unified 첫 실전 실행 후:

```sql
-- 5/18 매매 카운트 확인
SELECT
  strategy,
  COUNT(*) as cnt,
  COUNT(*) FILTER (WHERE memo LIKE '%위험:%') as with_risk_memo
FROM paper_trades
WHERE trade_date = '2026-05-18'
GROUP BY strategy;
```

기대 결과:
- `SCAN`/`REBALANCE`: paper 진입/청산
- `BLUECHIP_*`: bluechip 활동
- `ETF_LONG_*`: JARVIS 방향 ETF
- 모든 행의 memo에 위험 메타 포함

---

## 6. FAQ

**Q1. paper_trades에 너무 많은 데이터가 쌓이면?**
A. 30일 기본 필터링 + 사용자 선택 시 전체 조회. 인덱스: `trade_date`, `strategy`.

**Q2. memo 파싱이 복잡한데?**
A. 별도 컬럼 `risk_level` 추가 요청 시 퀀트봇이 스키마 마이그레이션 가능. 임시는 정규식 `/위험:(.+?)\(/` 파싱.

**Q3. ETF strategy 종류가 너무 많은데?**
A. prefix `ETF_`로 통일. 페이지에서 prefix 매칭으로 그룹.

**Q4. realtime 구독?**
A. `paper_trades` 테이블 INSERT 이벤트 구독 가능 (Supabase realtime). 신규 매매 즉시 노출.

**Q5. 정보봇 위험감지 카드는 다른 페이지에도 노출?**
A. 정보봇 측에 별도 요청. 이 카드는 다른 봇 페이지(/daytrading, /info)에도 공통.

---

## 7. 협업 연락

웹봇 구현 중 막히면:
- 퀀트봇 git: `https://github.com/haheun703-stack/-`
- FlowxUploader 어댑터: `src/adapters/flowx_uploader.py`
- 매매 진입/청산 코드: `scripts/paper_trading_unified.py`, `scripts/bluechip_timing.py`
- 위험감지 SDK: `src/utils/risk_gate.py` (정보봇 SDK 래퍼)

---

## 변경 이력

| 일자 | 내용 |
|---|---|
| 2026-05-16 | 초안 — paper_trades 4개 섹션 + 위험감지 카드 안내 |
