# ALADDIN — Autonomous Live Analyst for Deep-Dip Investment Navigation

> "숫자만 보는 퀀트봇이 아니라, 맥락을 읽는 살아있는 애널리스트"

---

## ⚠️ 실행 지침 — 반드시 읽고 따를 것

```
이 문서(ALADDIN 스펙)는 최종 목표 설계서다.
지금 당장 풀스펙으로 구현하지 마라.

즉시 실행: Phase A만
  - 기존 백테스트 검증된 역발상 뼈대 유지
    (BRAIN 연동 + 공포점수 + 분할매수 + 트레일링 익절)
  - 시총 1,000억+ 유니버스 확장 (룰 기반 스크리너) — 백테스트 완료
  - 페이퍼 트레이딩 진행

Phase B (1개월 후, 별도 지시 시):
  - ALADDIN 스펙의 나머지 (PANIC/FEAR 분류,
    매수 논증문, thesis 검증, 동적 사이징)

ALADDIN md는 장기 로드맵으로 참고만 해라.
Phase A 코드에 Phase B 로직 미리 넣지 마라.

페이퍼 트레이딩 설정:
- 유니버스: 시총 1,000억+ (1,028종목)
- BRAIN 연동: ON (CRISIS 차단, BEAR 반감)
- 자본: 가상 1,500만원 (전체 자본의 15% 가정)
- MAX_STOCKS: 3
- 기간: 최소 4주

실행 방식:
- 매일 BAT 스케줄에 맞춰 스캔 실행
- 매수 시그널 발생 시 실제 주문 안 하고 가상 포트폴리오에 기록
- paper_portfolio.json에 진입/청산/손익 기록
- 텔레그램으로 시그널 알림 (실행은 안 함)

시그널이 한 건도 안 뜨면 그것도 정상 —
공포 이벤트가 없으면 안 사는 게 맞으니까.
```

---

## 철학: 왜 기존 퀀트봇과 달라야 하는가

기존 봇은 **"이 조건이면 사고, 저 조건이면 판다"** — 기계다.
엘리트 애널리스트는 이렇게 생각한다:

```
"삼성전자가 -15% 빠졌다. 근데 이게 반도체 다운사이클 때문인가,
 트럼프 관세 공포 때문인가, 아니면 실적 미스 때문인가?
 원인에 따라 대응이 완전히 다르다."
```

같은 -15%인데 **왜 빠졌는지**에 따라 매수/관망/도망이 갈린다.
이걸 판단하는 게 ALADDIN의 핵심이다.

---

## Phase 1. 시장 체온 측정 — "지금 무슨 장인가?"

숫자를 보기 전에, **시장의 서사(narrative)**를 먼저 파악한다.

### 1-1. 레짐 분류 (4가지 시장 상태)

| 레짐 | 정의 | ALADDIN 행동 |
|------|------|-------------|
| PANIC | 시스템 리스크 (전쟁, 금융위기, 블랙스완) | 관망. 칼날 안 잡는다. 패닉은 저점이 아니라 시작이다. |
| FEAR | 과도한 공포, 펀더멘털 대비 과매도 | **핵심 사냥 구간**. 여기서 진입한다. |
| NEUTRAL | 평범한 조정 or 횡보 | 대기. 특별 기회 없으면 현금 보유. |
| GREED | 과열, 유포리아 | 절대 진입 금지. 보유 중이면 익절 가속. |

### 1-2. 레짐 판정 로직

단순 점수제가 아니라, **복합 컨텍스트 판단**:

```python
def classify_regime():
    """
    엘리트 애널리스트의 사고 흐름을 코드로 구현
    """

    # Layer 1: 숫자 — 정량 지표
    quantitative = {
        "kospi_5d_return": get_kospi_5d_return(),      # KOSPI 5일 수익률
        "vkospi": get_vkospi(),                         # 변동성 지수
        "foreign_flow_5d": get_foreign_net_5d(),        # 외국인 5일 순매수
        "credit_balance_change": get_credit_change(),   # 신용잔고 변화
        "put_call_ratio": get_put_call_ratio(),         # 풋콜비율
        "kospi_distance_from_200ma": get_ma200_gap(),   # 200일선 이격도
    }

    # Layer 2: 맥락 — 왜 이 숫자가 나왔는가
    context = {
        "trigger_event": identify_trigger(),
        # 예: "US_IRAN_WAR", "FED_RATE_SHOCK", "EARNINGS_MISS", "TRADE_WAR"
        #     "LIQUIDITY_CRISIS", "SECTOR_ROTATION", "NO_CLEAR_TRIGGER"

        "trigger_severity": assess_severity(),
        # 1~10 스케일. 전쟁=9, 관세=6, 실적=4, 수급=3

        "trigger_duration_estimate": estimate_duration(),
        # SHORT(1~2주), MEDIUM(1~3개월), LONG(6개월+)

        "global_sync": is_global_selloff(),
        # True면 한국만의 문제가 아님 → 더 신중
    }

    # Layer 3: 역사 — 과거 유사 패턴
    historical = {
        "similar_episodes": find_similar_episodes(context["trigger_event"]),
        # 과거 유사 이벤트에서 저점까지 걸린 기간, 낙폭, 회복 기간 반환
        # 예: 2020 코로나 → 저점까지 23일, -35%, 회복 5개월

        "avg_recovery_days": calc_avg_recovery(),
        "current_vs_typical_drawdown": compare_drawdown(),
        # 현재 낙폭이 과거 유사 사례 평균 대비 어느 수준인지
        # 예: "현재 -12%, 과거 평균 저점 -18% → 아직 65% 지점"
    }

    # 판정
    if context["trigger_severity"] >= 8 and context["trigger_duration_estimate"] == "LONG":
        return "PANIC"  # 시스템 리스크 — 손대지 마라

    if (quantitative["vkospi"] > 25
        and quantitative["kospi_5d_return"] < -0.05
        and context["trigger_severity"] <= 7
        and historical["current_vs_typical_drawdown"] > 0.6):
        # 공포는 있지만 시스템 리스크는 아니고,
        # 과거 패턴 대비 낙폭이 60% 이상 진행됨
        return "FEAR"  # 사냥 구간

    if quantitative["put_call_ratio"] < 0.7 and quantitative["kospi_distance_from_200ma"] > 0.08:
        return "GREED"

    return "NEUTRAL"
```

> **핵심**: PANIC과 FEAR를 구분하는 것. 2026년 3월 미-이란 전쟁 초기는 PANIC이었다.
> 거기서 뛰어들었으면 계좌가 녹았다. PANIC이 FEAR로 전환되는 시점을 포착하는 게 진짜 실력이다.

---

## Phase 2. 종목 사냥 — "무엇을, 왜 사는가?"

레짐이 FEAR로 판정되면 사냥 시작.
단순 스크리닝이 아니라, **애널리스트의 3단계 사고법**을 따른다.

### 2-1. 1차 그물 — 정량 과매도 필터 (2600+ → ~50개)

빠르게 걸러내는 단계. 기계적으로 처리.

```python
def cast_net():
    """KRX 전 종목에서 기본 과매도 조건 필터"""
    candidates = []
    for stock in all_krx_stocks():
        score = 0
        score += 8 if stock.rsi_14 < 30 else 0
        score += 7 if stock.drawdown_from_52w_high < -0.25 else 0
        score += 5 if stock.bollinger_position == "below_lower" else 0
        score += 5 if stock.volume_ratio_20d > 2.0 else 0  # 투매 흔적
        score += 5 if stock.disparity_20d < 92 else 0

        if score >= 18:
            candidates.append(stock)
    return candidates  # ~50개
```

### 2-2. 2차 체질 검사 — 펀더멘털 안전망 (~50개 → ~15개)

"싸다고 다 사는 게 아니다. 싸고 살아남을 놈만 산다."

```python
def health_check(candidates):
    """죽어가는 회사를 걸러낸다"""
    survivors = []
    for stock in candidates:
        # 절대 금지 조건 (하나라도 해당 시 탈락)
        if stock.is_administrative_issue: continue      # 관리종목
        if stock.is_investment_warning: continue         # 투자경고
        if stock.market_cap < 100_000_000_000: continue  # 시총 1000억 미만
        if stock.debt_ratio > 200: continue              # 부채비율 200% 초과
        if stock.operating_profit_positive_quarters < 3: continue  # 최근 4분기 중 3분기 미만 흑자

        # 추가 건전성 (가산)
        stock.health_bonus = 0
        if stock.current_ratio > 150: stock.health_bonus += 3      # 유동비율 양호
        if stock.interest_coverage > 5: stock.health_bonus += 3    # 이자보상배율 양호
        if stock.free_cash_flow > 0: stock.health_bonus += 4       # FCF 양호

        survivors.append(stock)
    return survivors  # ~15개
```

### 2-3. 3차 애널리스트 브리핑 — 핵심 차별 구간 (~15개 → ~3~5개)

**여기서 AI에게 자율성을 준다.**
기존 봇은 "이 조건 충족하면 매수". ALADDIN은 **"왜 이 종목이 반등하는가?"를 스스로 논증**한다.

```python
def analyst_briefing(survivors):
    """
    각 종목에 대해 5-Dimensional Analysis 수행.
    1D(표면) → 5D(구조적)까지 자율 탐색.
    """
    briefings = []

    for stock in survivors:
        analysis = {}

        # ── 1D: 표면 (무슨 일이 있었나?) ──
        analysis["1D_surface"] = {
            "price_action": describe_recent_price_action(stock),
            "immediate_trigger": identify_stock_drop_trigger(stock),
            # "실적 미스", "섹터 전반 하락", "대주주 매도", "공매도 공격" 등
        }

        # ── 2D: 수급 (누가 팔고 누가 사고 있나?) ──
        analysis["2D_flow"] = {
            "foreign_trend": get_foreign_flow(stock, days=20),
            "institutional_trend": get_institutional_flow(stock, days=20),
            "retail_trend": get_retail_flow(stock, days=20),
            "short_interest_change": get_short_interest_trend(stock),
            "dark_pool_signals": get_quiet_accumulation(stock),
            # 위성 스크리너 연동 — 조용한 축적 패턴
        }

        # ── 3D: 공시/이벤트 (내부자는 뭘 하고 있나?) ──
        analysis["3D_insider"] = {
            "dart_disclosures": get_recent_dart(stock, days=30),
            # 자사주 매입, 대주주 장내매수, CB/BW 전환, 유상증자 등
            "insider_trading": get_insider_trades(stock),
            "upcoming_events": get_event_calendar(stock),
            # 실적발표일, 주총, 배당락일 등
        }

        # ── 4D: 섹터 연결 (이 종목이 속한 생태계는 어떤가?) ──
        analysis["4D_sector"] = {
            "sector": stock.wics_sector,
            "us_anchor_status": get_us_anchor_trend(stock),
            # JARVIS_SECTOR_MAP 연동
            # 예: 삼성전자 → US앵커 NVDA +8% → 긍정
            "sector_rotation_signal": get_sector_rotation_phase(stock),
            # 자금이 이 섹터로 유입 중인가, 유출 중인가
            "nightwatch_signal": get_nightwatch_status(stock.sector),
            # GREEN/YELLOW/RED
            "peer_comparison": compare_with_peers(stock),
            # 같은 섹터 내 다른 종목 대비 낙폭이 과도한가
        }

        # ── 5D: 매크로 구조 (큰 그림에서 이 종목의 위치는?) ──
        analysis["5D_macro"] = {
            "policy_tailwind": check_policy_support(stock),
            # 정부 정책 수혜 여부 (K-반도체 전략, 방산 예산 등)
            "global_supply_chain": check_supply_chain_position(stock),
            # 글로벌 공급망에서의 위치, 대체 불가능성
            "currency_impact": assess_fx_impact(stock),
            # 원달러 환율 방향이 이 종목에 유리한가
            "commodity_link": check_commodity_correlation(stock),
            # 원자재 가격과의 연동성
        }

        # ── 종합 판정: 매수 논증문 생성 ──
        briefing = generate_conviction_report(stock, analysis)
        briefings.append(briefing)

    return rank_by_conviction(briefings)  # 확신도 순 정렬
```

### 매수 논증문 (Conviction Report) 형식

```json
{
  "stock_code": "005930",
  "stock_name": "삼성전자",
  "conviction_grade": "A",
  "conviction_score": 82,

  "thesis": "반도체 다운사이클 공포로 -25% 하락했으나, HBM3E 수주 모멘텀은 건재.
             US앵커 NVDA가 선반등 중이며, 외국인 순매도 둔화 + 자사주 5000억 매입 공시.
             과거 유사 패턴(2019Q1 반도체 다운사이클) 대비 현재 낙폭이 이미 80% 수준 도달.",

  "bull_case": {
    "핵심_촉매": "HBM3E 양산 본격화 (Q2), NVDA 선반등",
    "수급_전환": "외국인 순매도 5일→2일로 둔화, 기관 3일 연속 순매수",
    "내부자_행동": "자사주 5000억 + 임원 장내매수 2건",
    "예상_반등폭": "+15~25% (3개월 기준)",
    "반등_시나리오": "NVDA 실적 서프라이즈 → 반도체 섹터 리레이팅 → 삼성 동반 상승"
  },

  "bear_case": {
    "최대_리스크": "중국 CXMT HBM 자체 개발 성공 시 ASP 하락 압력",
    "추가_하락_시나리오": "파운드리 적자 지속 + 메모리 가격 추가 하락",
    "예상_추가_낙폭": "-10% 추가 가능",
    "킬_조건": "분기 영업적자 전환 시 즉시 손절"
  },

  "sizing_recommendation": "HIGH",
  // HIGH: 3회 분할매수 모두 실행 가능
  // MEDIUM: 2회까지만
  // LOW: 1회 탐색 매수만

  "time_horizon": "2~3개월",

  "key_monitoring_triggers": [
    "NVDA 실적발표 (4/23) — 서프라이즈 시 2차 매수 즉시",
    "삼성전자 잠정실적 (4/5) — 미스 시 thesis 재검토",
    "외국인 순매수 전환 시 — 확신 매수 트리거"
  ]
}
```

> **이게 기존 봇과의 결정적 차이.**
> 기존: "RSI 28이고 볼린저 하단 이탈이니까 매수" (기계)
> ALADDIN: "HBM3E 모멘텀은 살아있고, NVDA가 선반등 중이며, 내부자가 사고 있고,
>           과거 패턴 대비 낙폭 80% 도달했으니까 매수" (애널리스트)

---

## Phase 3. 포지션 구축 — "어떻게, 얼마나 사는가?"

### 3-1. 확신도 기반 동적 사이징

고정 비율이 아니라, **확신도에 따라 베팅 크기가 달라진다.**

```
확신도 A+ (90점+) → 슬롯 자본의 최대 25% 투입 가능
확신도 A  (80~89) → 최대 20%
확신도 B+ (70~79) → 최대 15%
확신도 B  (60~69) → 최대 10% (탐색 수준)
확신도 C  (60 미만) → 자동 탈락. 진입 불가.
```

### 3-2. 적응형 분할매수

기계적 "1차/2차/3차"가 아니라, **상황에 따라 전술이 바뀐다.**

```python
def execute_split_buy(stock, conviction_report):
    """
    매수 논증문의 시나리오에 따라 분할매수 전술을 자동 결정
    """

    sizing = conviction_report["sizing_recommendation"]

    if sizing == "HIGH":
        # 확신도 높음 — 적극적 3단계
        phases = [
            {"pct": 0.40, "trigger": "즉시 (진입 조건 충족)"},
            {"pct": 0.30, "trigger": "monitoring_trigger 발생 시 OR 추가 -3% 시"},
            {"pct": 0.30, "trigger": "반등 확인 양봉 출현 시"},
        ]

    elif sizing == "MEDIUM":
        # 중간 확신 — 신중한 2단계
        phases = [
            {"pct": 0.40, "trigger": "즉시"},
            {"pct": 0.60, "trigger": "bull_case 촉매 실현 확인 시"},
        ]

    elif sizing == "LOW":
        # 탐색 — 소량 1회
        phases = [
            {"pct": 1.00, "trigger": "즉시 (탐색 매수, 소규모)"},
        ]

    return phases
```

### 3-3. 타이밍 미세조정

매수 결정 후에도 **장중 최적 타이밍**을 잡는다:

```python
def find_entry_timing(stock):
    """
    VWAP 기반 장중 최적 진입점 탐색
    """
    current_price = stock.price
    vwap = stock.intraday_vwap

    if current_price < vwap * 0.995:
        # VWAP 대비 -0.5% 이하 → 기관 매집 단가 이하, 즉시 진입
        return "EXECUTE_NOW"

    elif current_price > vwap * 1.01:
        # VWAP 대비 +1% 이상 → 오늘은 비싸다, 내일로 연기
        return "DEFER_TO_TOMORROW"

    else:
        # VWAP 근처 → 14:30 이후 장 후반 진입 (종가 매매)
        return "WAIT_FOR_CLOSING_AUCTION"
```

---

## Phase 4. 포지션 관리 — "살아있는 모니터링"

매수 후가 진짜 중요하다. **보유 중에도 thesis를 계속 검증**한다.

### 4-1. Thesis Alive Check (매일)

```python
def daily_thesis_check(position):
    """
    매수 논증문의 핵심 가정이 아직 유효한지 매일 확인
    """
    report = position.conviction_report
    alerts = []

    # bull_case 촉매가 실현됐는가?
    for trigger in report["key_monitoring_triggers"]:
        status = check_trigger_status(trigger)
        if status == "REALIZED_POSITIVE":
            alerts.append(f"촉매 실현: {trigger} → 확신도 상향")
        elif status == "REALIZED_NEGATIVE":
            alerts.append(f"촉매 부정: {trigger} → thesis 재검토 필요")

    # bear_case 리스크가 현실화됐는가?
    kill_condition = report["bear_case"]["킬_조건"]
    if is_kill_condition_met(kill_condition):
        alerts.append(f"킬 조건 충족: {kill_condition} → 즉시 청산")
        return "KILL"

    # 섹터 환경 변화
    nightwatch = get_nightwatch_status(position.sector)
    if nightwatch == "RED" and position.original_nightwatch != "RED":
        alerts.append("NIGHTWATCH RED 전환 → 포지션 축소 고려")

    # US 앵커 동향
    us_anchor_trend = get_us_anchor_recent(position.stock)
    if us_anchor_trend == "BREAKDOWN":
        alerts.append("US 앵커 급락 → thesis 위협")

    return alerts
```

### 4-2. 적응형 엑싯

기계적 %가 아니라, **thesis 상태에 따라 엑싯 전략이 바뀐다:**

```
[Thesis 강화 시] — 촉매 실현, 수급 전환 확인
  → 엑싯 기준 완화: 트레일링 스탑 -7%로 넓힘
  → "달리는 말에서 너무 빨리 내리지 마라"

[Thesis 유지 시] — 특별한 변화 없음
  → 기본 엑싯: +5% 1/3, +10% 1/3, +15%~ 트레일링(-5%)

[Thesis 약화 시] — 리스크 현실화 조짐
  → 엑싯 기준 강화: 본전 근처에서도 축소 시작
  → 트레일링 -3%로 타이트하게

[Thesis 붕괴 시] — 킬 조건 충족
  → 즉시 전량 청산. 예외 없음.
```

---

## Phase 5. 학습과 진화 — "매 거래에서 배운다"

### 5-1. 거래 사후 분석 (Post-Trade Review)

모든 청산 완료 포지션에 대해 자동 리뷰:

```json
{
  "trade_id": "CTR-2026-003",
  "stock": "삼성전자",
  "entry_thesis": "HBM3E 모멘텀 + NVDA 선반등...",
  "result": "+12.3%",
  "duration": "18 영업일",

  "what_worked": [
    "NVDA 선반등 → 삼성 동반 상승 thesis 적중",
    "자사주 매입이 하방 지지 역할"
  ],
  "what_didnt": [
    "2차 매수 타이밍이 1일 늦었음 (추가 -2% 구간 놓침)",
    "익절 1/3이 너무 빨랐음 (+5%에서, 결국 +18%까지 감)"
  ],
  "lesson": "확신도 A 이상일 때 첫 익절 구간을 +7%로 올려도 될 듯",
  "parameter_suggestion": {
    "first_take_profit": "5% → 7% (확신도 A 이상 한정)"
  }
}
```

### 5-2. 파라미터 자가 조정 제안

매월 말 전체 거래 리뷰 후, 파라미터 조정 제안을 생성:

```
[ALADDIN 월간 리뷰 — 2026년 3월]

총 거래: 5건 | 승률: 60% | 평균 수익: +8.2% | PF: 2.1

발견된 패턴:
- 확신도 A+ 거래 (2건): 승률 100%, 평균 +14%
- 확신도 B+ 거래 (3건): 승률 33%, 평균 +2%
→ 제안: B+ 이하 진입을 줄이고, A 이상에 자본 집중

- NIGHTWATCH GREEN 섹터 진입 (3건): 승률 67%
- NIGHTWATCH YELLOW 섹터 진입 (2건): 승률 50%
→ 제안: YELLOW에서는 sizing을 한 단계 낮추기

수동 승인 필요. 자동 변경하지 않음.
```

---

## 리스크 관리 — 절대 규칙 (자율성 없음)

**여기에는 자율성을 주지 않는다. 한 줄도 양보 없다.**

```
■ 종목당 손절: 평균단가 -8% → 전량 즉시 청산
■ 슬롯 MDD: -12% → 전 포지션 청산 + 5영업일 냉각기
■ 월간 손실: -10% → 해당 월 슬롯 비활성화
■ 동시 보유: 최대 4종목
■ 종목 집중: 슬롯 자본의 25% 상한 (A+ 한정, 그 외 20%)
■ 미수금: 매수 전 예수금 100% 확인 (절대 원칙)
■ 매매 금지 시간: 장 시작 후 5분, 장 마감 전 10분
■ thesis 킬 조건 충족 시: 수익 중이어도 즉시 청산
```

---

## 텔레그램 알림 설계

기존 봇의 단순 "매수/매도" 알림이 아니라, **애널리스트 브리핑 스타일**:

```
━━━━━━━━━━━━━━━━━━━━━━
📊 ALADDIN 시장 브리핑 (08:30)
━━━━━━━━━━━━━━━━━━━━━━
레짐: FEAR (공포점수 67/100)
트리거: 트럼프 반도체 관세 25% 발표
심각도: 6/10 | 예상 지속: 2~4주
과거 유사: 2018 무역전쟁 (저점까지 15일, -12%)
현재 진행: 낙폭 -8% (과거 대비 67% 지점)

→ 사냥 모드 활성화. 스크리닝 시작.
━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━
🎯 ALADDIN 종목 브리핑
━━━━━━━━━━━━━━━━━━━━━━
삼성전자 (005930) — 확신도 A (82점)

핵심 논증:
"반도체 다운사이클 공포로 -25% 하락.
 그러나 HBM3E 수주잔고 건재, NVDA 선반등 중.
 자사주 5000억 + 임원 장내매수.
 과거 패턴 대비 낙폭 80% 도달."

사이징: HIGH (3단계 분할매수)
1차 진입: 57,800원 x 86주 (슬롯 자본 10%)
다음 트리거: NVDA 실적(4/23) or 추가 -3%

리스크: 분기 영업적자 전환 시 즉시 청산
━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━
⚡ ALADDIN Thesis 업데이트
━━━━━━━━━━━━━━━━━━━━━━
삼성전자 — 보유 7일차 | 현재 +4.2%

Thesis 상태: 강화 ⬆️
- NVDA +5.2% 실적 서프라이즈 (촉매 실현)
- 외국인 3일 연속 순매수 전환

행동: 2차 매수 실행 예정 (내일 09:05)
엑싯 기준: 트레일링 -7%로 완화
━━━━━━━━━━━━━━━━━━━━━━
```

---

## 파일 구조

```
strategies/contrarian/
├── __init__.py
├── config.py                    # 파라미터 설정
│
├── phase1_regime/
│   ├── regime_classifier.py     # 레짐 판정 (PANIC/FEAR/NEUTRAL/GREED)
│   ├── trigger_identifier.py    # 하락 원인 식별
│   ├── historical_matcher.py    # 과거 유사 패턴 매칭
│   └── fear_score.py            # 공포 점수 정량화
│
├── phase2_hunting/
│   ├── quantitative_net.py      # 1차 정량 과매도 필터
│   ├── health_check.py          # 2차 펀더멘털 안전망
│   ├── analyst_5d.py            # 3차 5D 분석 엔진 (핵심)
│   ├── conviction_report.py     # 매수 논증문 생성
│   └── candidate_ranker.py      # 확신도 순위 정렬
│
├── phase3_execution/
│   ├── dynamic_sizer.py         # 확신도 기반 동적 사이징
│   ├── adaptive_split_buy.py    # 적응형 분할매수
│   └── entry_timer.py           # VWAP 기반 장중 타이밍
│
├── phase4_management/
│   ├── thesis_monitor.py        # 일일 Thesis Alive Check
│   ├── adaptive_exit.py         # Thesis 연동 적응형 엑싯
│   └── killswitch.py            # 절대 규칙 리스크 관리
│
├── phase5_learning/
│   ├── post_trade_review.py     # 거래 사후 분석
│   └── monthly_review.py        # 월간 파라미터 조정 제안
│
├── integrations/
│   ├── dart_connector.py        # DART 공시 연동
│   ├── sector_map_connector.py  # JARVIS_SECTOR_MAP 연동
│   ├── nightwatch_connector.py  # NIGHTWATCH 연동
│   ├── satellite_connector.py   # 위성 스크리너 연동
│   └── telegram_briefing.py     # 텔레그램 애널리스트 브리핑
│
├── data/
│   ├── historical_episodes.json # 과거 위기 패턴 DB
│   └── sector_anchors.json      # US-KR 섹터 앵커 매핑
│
└── contrarian_main.py           # 메인 오케스트레이터
```

---

## 실행 전 필수 사항

1. **백테스트** (2024.03~2026.03, 주요 이벤트 포함)
   - 통과 기준: PF ≥ 1.5, MDD ≤ -15%, 승률 ≥ 45%
2. **1개월 페이퍼 트레이딩** — 실제 브리핑만 발송, 매매 미집행
3. **실전 최소 자본** — 슬롯 자본 15%로 시작
4. **기존 시스템 무간섭** — 스윙/ETF 코드 수정 절대 금지
