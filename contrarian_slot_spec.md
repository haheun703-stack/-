# 역발상 저점매집 전략 슬롯 구현 지시서

## 개요

기존 시스템(스윙 60% + 그룹ETF 40%)을 건드리지 않고, **독립된 3번째 전략 슬롯**을 추가한다.
이 슬롯은 기존 시스템과 정반대 — **과매도/공포 구간에서 진입**하는 역발상 저점매집 전략이다.

### 핵심 철학
- 기존 봇: "위험하면 안 들어간다" (방어형)
- 이 슬롯: "다들 무서워할 때 들어간다" (공격형)
- 단, **제한된 자본** 안에서만 운용 (전체 계좌의 10~20%)

---

## 1. 자본 배분 구조 변경

```
[기존]
├── 스윙 전략: 60%
└── 그룹ETF 로테이션: 40%

[변경]
├── 스윙 전략: 50% (기존 대비 -10%)
├── 그룹ETF 로테이션: 35% (기존 대비 -5%)
└── 역발상 저점매집: 15% (신규)
```

> 초기에는 15%로 시작. 3개월 성과 평가 후 Portfolio Orchestrator 자연선택 메커니즘에 따라 비중 자동 조정.

---

## 2. 진입 조건 (공포 감지 시스템)

기존 시스템이 AND-게이트(모든 조건 충족 시 진입)인 반면,
이 슬롯은 **스코어링 방식** — 공포 신호를 점수화해서 임계치 초과 시 진입.

### 2-1. 시장 레벨 공포 지표 (매크로 필터)

| 지표 | 데이터 소스 | 공포 조건 | 점수 |
|------|-----------|----------|------|
| KOSPI 낙폭 | KIS API | 5일 누적 -5% 이상 | 15점 |
| 투자자별 매매동향 | KRX | 외국인 5일 연속 순매도 | 10점 |
| VIX(VKOSPI) | KRX | 25 이상 | 10점 |
| 신용잔고 급감 | KRX | 5일 내 -10% 이상 | 10점 |
| 공포탐욕지수 | 자체 계산 | 30 이하 (Extreme Fear) | 15점 |

**매크로 필터 통과 기준: 30점 이상** (100점 만점 중)
→ 30점 미만이면 이 슬롯은 현금 보유 상태 유지 (쉬는 것도 전략)

### 2-2. 종목 레벨 과매도 신호 (종목 스크리닝)

매크로 필터 통과 시, 아래 기준으로 KRX 전 종목 스캔:

| 지표 | 조건 | 점수(30점 만점) |
|------|------|----------------|
| RSI(14) | 30 이하 | 8점 |
| 볼린저밴드 | 하단 이탈 | 5점 |
| 52주 고점 대비 낙폭 | -30% 이상 | 7점 |
| 거래량 | 20일 평균 대비 200% 이상 (투매 신호) | 5점 |
| 이격도(20일) | 92 이하 | 5점 |

**종목 진입 기준: 18점 이상**

---

## 3. 종목 선정 자율성 레이어 (핵심 차별점)

단순 기술적 과매도만으로는 "떨어지는 칼날"을 잡게 된다.
**왜 이 종목이 반등할 수 있는지** 근거를 종합 판단하는 레이어가 핵심.

### 3-1. 펀더멘털 안전장치 (필수 통과)

아래 중 **모두 충족**해야 매수 후보에 진입:

- 영업이익: 최근 4분기 중 3분기 이상 흑자
- 부채비율: 200% 이하
- 시가총액: 1,000억 이상 (유동성 확보)
- 관리종목/투자경고 아님

### 3-2. 정보 우위 신호 (가산점 — 자율 탐색 영역)

여기서 봇에게 **자율성**을 부여한다.
아래 데이터 소스를 스스로 조합해서 "매수 근거 리포트"를 생성하게 한다.

| 데이터 소스 | 탐색 내용 | 가산점 |
|------------|----------|--------|
| DART 공시 | 자사주 매입, 대주주 장내매수, 무상증자 | +10점 |
| JARVIS_SECTOR_MAP | 해당 섹터 US 앵커 종목 최근 반등 여부 | +8점 |
| 위성 스크리너 | 조용한 축적 패턴 감지 (quiet accumulation) | +7점 |
| NIGHTWATCH 신호 | 해당 섹터 GREEN 전환 | +10점 |
| 신용잔고/대차잔고 | 공매도 잔고 급감 (숏커버 신호) | +5점 |

**자율 탐색 결과물**: 각 종목에 대해 아래 형식의 매수 근거 리포트를 자동 생성

```json
{
  "종목코드": "005930",
  "종목명": "삼성전자",
  "과매도점수": 22,
  "정보우위점수": 18,
  "매수근거": [
    "RSI 28 극단적 과매도",
    "DART: 3/5 자사주 500억 매입 공시",
    "US앵커(NVDA) 최근 5일 +8% 반등",
    "NIGHTWATCH 반도체섹터 YELLOW→GREEN 전환"
  ],
  "리스크요인": [
    "외국인 7일 연속 순매도 지속",
    "52주 신저가 갱신 중"
  ],
  "확신도": "B+"  // A+, A, B+, B, C (C는 자동 탈락)
}
```

> **C등급은 자동 탈락** — 기존 시스템에서 배운 교훈. C-grade 신호는 전체 성과를 끌어내린다.

---

## 4. 매수 실행 (분할매수)

한 종목에 대해 **최대 3회 분할매수**:

```
1차 매수: 슬롯 배정 자본의 15% (탐색 진입)
  → 진입 조건 충족 시 즉시

2차 매수: 슬롯 배정 자본의 20% (확인 매수)
  → 1차 매수 후 추가 -3% 하락 시 OR 3영업일 후 반등 확인 시

3차 매수: 슬롯 배정 자본의 25% (확신 매수)
  → 2차 매수 후 추가 -5% 하락 시 OR 거래량 동반 양봉 출현 시
```

- 종목당 최대 투입: 슬롯 자본의 60%
- 동시 보유 종목: 최대 3개
- 1종목 집중 금지: 종목당 슬롯 자본의 60% 상한

---

## 5. 매도 실행 (하이브리드 엑싯)

### 5-1. 손절 (비가역성 방어)

| 조건 | 행동 |
|------|------|
| 평균단가 대비 -7% | 전량 손절 (무조건) |
| 펀더멘털 훼손 (적자전환 공시 등) | 전량 즉시 매도 |

### 5-2. 익절 (수익 극대화)

```
구간 1: +5% ~ +10%
  → 보유량의 1/3 익절

구간 2: +10% ~ +15%
  → 보유량의 1/3 추가 익절

구간 3: +15% 이상
  → 나머지 1/3은 트레일링 스탑 전환
  → 고점 대비 -5% 이탈 시 청산
```

---

## 6. 킬스위치 & 리스크 관리

### 6-1. 슬롯 레벨 킬스위치

| 조건 | 행동 |
|------|------|
| 슬롯 MDD -10% 도달 | 슬롯 전체 매도, 5영업일 냉각기 |
| 월간 손실 -8% | 해당 월 슬롯 비활성화 |
| 동시 2종목 손절 발생 | 슬롯 일시 중지, 수동 재개 필요 |

### 6-2. 주문 안전장치

- 매수 전 반드시 예수금 확인 (미수금 방지 — 기존 이슈 재발 방지)
- 슬리피지 임계치: 평균 0.3%, 최대 0.5% (기존 QA 검수 기준 동일)
- 장 시작 후 5분, 장 마감 전 10분 매매 금지

---

## 7. Control Tower 연동

### 7-1. 대시보드 추가 항목

기존 Control Tower(ppwangga.com)에 "역발상 슬롯" 서브 대시보드 추가:

- 현재 매크로 공포 점수 (실시간)
- 스크리닝 중인 후보 종목 리스트
- 보유 종목별 매수 근거 리포트
- 분할매수 진행 상태 (1차/2차/3차)
- 슬롯 수익률 곡선 & MDD

### 7-2. 텔레그램 알림

기존 텔레그램 봇을 통해 알림 발송:

```
[역발상 슬롯] 매크로 공포점수 42점 — 스크리닝 시작
[역발상 슬롯] 후보 발견: 삼성전자(005930) 과매도22 + 정보우위18 = 40점 (B+)
[역발상 슬롯] 1차 매수 실행: 삼성전자 57,800원 x 50주
[역발상 슬롯] 킬스위치 발동: MDD -10% 도달, 전량 청산
```

### 7-3. SQLite 테이블 추가

```sql
-- 매크로 공포 점수 이력
CREATE TABLE contrarian_macro_score (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    kospi_drop_score INTEGER,
    foreign_sell_score INTEGER,
    vkospi_score INTEGER,
    credit_score INTEGER,
    fear_greed_score INTEGER,
    total_score INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 종목 스크리닝 결과
CREATE TABLE contrarian_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT NOT NULL,
    oversold_score INTEGER,
    info_edge_score INTEGER,
    total_score INTEGER,
    confidence_grade TEXT,  -- A+, A, B+, B, C
    buy_reasons TEXT,       -- JSON array
    risk_factors TEXT,      -- JSON array
    status TEXT DEFAULT 'candidate',  -- candidate/entered/rejected
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 분할매수 추적
CREATE TABLE contrarian_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    stock_name TEXT NOT NULL,
    entry_phase INTEGER,    -- 1, 2, 3
    entry_price REAL,
    quantity INTEGER,
    entry_date TEXT,
    avg_price REAL,         -- 평균단가 (분할매수 반영)
    total_quantity INTEGER,  -- 누적 수량
    status TEXT DEFAULT 'holding',  -- holding/partial_exit/closed
    exit_price REAL,
    exit_date TEXT,
    pnl_pct REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 슬롯 일일 성과
CREATE TABLE contrarian_daily_pnl (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    slot_capital REAL,
    invested_amount REAL,
    cash_amount REAL,
    daily_return_pct REAL,
    cumulative_return_pct REAL,
    mdd_pct REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 8. 파일 구조

기존 프로젝트 구조에 아래 디렉토리/파일 추가:

```
strategies/
├── swing/              (기존)
├── group_etf/          (기존)
└── contrarian/         (신규)
    ├── __init__.py
    ├── macro_fear_scanner.py      # 매크로 공포 점수 계산
    ├── stock_screener.py          # 종목 레벨 과매도 스크리닝
    ├── info_edge_analyzer.py      # 정보 우위 자율 탐색 (DART, 섹터맵, NIGHTWATCH 연동)
    ├── buy_reason_reporter.py     # 매수 근거 리포트 자동 생성
    ├── split_buyer.py             # 분할매수 실행 로직
    ├── hybrid_exit.py             # 하이브리드 엑싯 (손절 + 구간익절 + 트레일링)
    ├── killswitch.py              # 슬롯 레벨 킬스위치
    ├── contrarian_main.py         # 전략 메인 오케스트레이터
    └── config.py                  # 파라미터 설정 (점수 임계치, 비중 등)
```

---

## 9. 실행 사이클

```
매일 장 시작 전 (08:30)
│
├─ 1) macro_fear_scanner 실행
│   └─ 매크로 공포 점수 계산 → DB 저장
│
├─ 2) 공포 점수 ≥ 30?
│   ├─ YES → stock_screener 실행 (KRX 전 종목 스캔)
│   │   └─ 과매도 18점 이상 종목 필터
│   │       └─ info_edge_analyzer 실행 (자율 탐색)
│   │           └─ buy_reason_reporter로 리포트 생성
│   │               └─ B등급 이상만 → 텔레그램 알림
│   │
│   └─ NO → "시장 평온, 대기 중" 로그 → 종료
│
├─ 3) 장중 (09:05 ~ 15:20)
│   ├─ 보유 종목 모니터링
│   ├─ 분할매수 조건 체크 (2차, 3차)
│   ├─ 엑싯 조건 체크 (손절/익절/트레일링)
│   └─ 킬스위치 조건 체크
│
└─ 4) 장 마감 후 (15:40)
    ├─ 일일 성과 기록 → DB
    ├─ 텔레그램 일일 리포트 발송
    └─ Control Tower 대시보드 업데이트
```

---

## 10. 백테스트 우선 실행

**실전 투입 전 반드시 백테스트:**

- 기간: 최근 2년 (2024.03 ~ 2026.03)
- 포함 이벤트: 공매도 재개(2025.03), 미국-이란 전쟁(2026.03)
- 비교 기준: 기존 스윙 전략 동일 기간 수익률 대비 알파
- 통과 기준: PF 1.5 이상, MDD -15% 이내, 승률 45% 이상

백테스트 통과 후 → 1개월 페이퍼 트레이딩 → 실전 (최소 자본) → 점진적 증액

---

## 11. BRAIN 레짐 연동 규칙

역발상이라고 해서 시스템 전체 방어 체계를 무시하면 안 된다.
BRAIN이 "위험하다"고 할 때 역발상 전략만 공격하면 계좌 전체 리스크가 올라간다.

### 11-1. 레짐별 자본 캡

| BRAIN 레짐 | 역발상 슬롯 자본 사용 | 비고 |
|-----------|---------------------|------|
| CRISIS / PANIC | **0%** (신규진입 금지) | 보유 종목은 엑싯 룰만 적용 |
| BEAR | **7.5%** (15% 중 절반) | 분할매수 1차만 허용 |
| CAUTION | **15%** (풀 사용) | 정상 운용 |
| BULL | **15%** (풀 사용) | 정상 운용 |

### 11-2. 연동 방식

```python
# brain_allocation.json 읽어서 자동 적용
brain = load_json("data/brain_allocation.json")
regime = brain.get("regime", "CAUTION")

if regime in ("CRISIS", "PANIC"):
    contrarian_cap = 0.0       # 신규진입 완전 금지
elif regime == "BEAR":
    contrarian_cap = 0.075     # 7.5%
else:
    contrarian_cap = 0.15      # 15%
```

### 11-3. 핵심 원칙

- BRAIN은 계좌 전체 방어 체계 → **역발상 슬롯도 BRAIN 우산 아래**
- CRISIS에서 "바닥 찍었으니 매수" 유혹이 가장 크지만, **그때가 가장 위험**
- 역발상 슬롯의 진짜 기회는 CAUTION→BEAR 전환 초기 (아직 패닉 전)
- BEAR에서 절반만 쓰는 건 "물 타기 한 발" 허용하되 과노출 방지

---

## 주의사항

1. **미수금 절대 방지**: 매수 전 예수금 체크 로직 필수 (기존 이슈 재발 방지)
2. **기존 전략과 독립 운용**: 기존 스윙/ETF 코드 일절 수정 금지
3. **C등급 신호 자동 탈락**: 과거 교훈 — C-grade는 전체 성과를 끌어내림
4. **자율성은 종목 선정 단계에서만**: 리스크 관리(손절/킬스위치)는 절대 규칙 기반 유지
5. **BRAIN 레짐 연동 필수**: 역발상 슬롯도 BRAIN 방어 체계 안에서 운용
