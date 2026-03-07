# 섹터 릴레이 엔진 구현 계획서

## 목표
**"US 선행주 확인 → 한국 대장주 확인 → 한국 2차 연동주 진입"** 4단계 릴레이 시스템 구축.
5개 섹터(AI반도체, 방산, 정유/에너지, 배터리/ESS, 조선/LNG) 동시 지원.

---

## 현재 시스템 vs 신규 릴레이 엔진

| 항목 | 현재 | 릴레이 엔진 (신규) |
|------|------|-------------------|
| US 추적 | ETF/지수 17개 (SOXX, XLE 등) | **개별 대장주** (NVDA, LMT, XOM 등) + 기존 ETF |
| 경보 방향 | Kill만 (방어) | **Boost(공격) + Kill(방어)** 양방향 |
| 릴레이 체인 | 한국 그룹 내부 (대장→계열사) | **US→KR 크로스마켓** 4단계 |
| 실행 레벨 | 기술적 지표 기반 | **전일 고가/VWAP/15분고가** 레벨 기반 |
| 섹터 성격 구분 | 동일 기준 | **지속형/이벤트형/조건부** 차등 |

---

## 아키텍처

```
config/relay_sectors.yaml        ← 5개 섹터 정의 (종목, 레벨, 룰)
src/relay/
  ├── __init__.py
  ├── config.py                  ← relay_sectors.yaml 로더
  ├── us_tracker.py              ← US 개별 대장주 가격/레벨 추적 (yfinance)
  ├── relay_engine.py            ← 4단계 릴레이 판정 엔진
  ├── alert_classifier.py        ← 3단계 경보 (예비/본/실행)
  └── execution_rules.py         ← 매수/매도 실행 규칙 (VWAP/전일고가)
scripts/
  ├── run_relay_engine.py        ← 릴레이 엔진 실행 스크립트
  └── relay_us_update.py         ← US 개별주 데이터 업데이트
data/relay/
  ├── us_leaders.json            ← US 대장주 최신 데이터 + 레벨
  ├── relay_signal.json          ← 릴레이 경보 결과
  └── relay_history.json         ← 경보 이력 (학습용)
```

### 클린 아키텍처 준수
- `src/relay/` = use_cases 레벨 (비즈니스 로직)
- 외부 의존 (yfinance, KIS API) = adapters 경유
- AI Brain, US Overnight = 기존 데이터 재활용 (data_bridge 확장)

---

## 파일별 구현 상세

### 1. `config/relay_sectors.yaml` — 5개 섹터 전체 정의

```yaml
relay_engine:
  enabled: true
  alert_min_count: 3        # 경보 3개 이상 동시 충족 시만 거래
  capital_split:
    kr_leader_pct: 50       # 한국 대장주 배분
    kr_secondary_pct: 50    # 한국 2차 연동주 배분
  max_trades_per_sector: 2  # 섹터당 일일 최대 거래 수

  sectors:
    ai_semiconductor:
      name: "AI 반도체"
      type: "persistent"          # 지속형
      us_leaders:
        - {ticker: "NVDA", name: "NVIDIA", role: "AI accelerator"}
        - {ticker: "AVGO", name: "Broadcom", role: "AI networking"}
        - {ticker: "AMD", name: "AMD", role: "GPU challenger"}
        - {ticker: "MU", name: "Micron", role: "HBM/memory"}
      us_leader_min_strong: 2     # 최소 2개 이상 강세
      us_secondaries_etf: "SOXX"  # 기존 US Overnight에서 재활용
      kr_leaders:
        - {ticker: "000660", name: "SK하이닉스"}
        - {ticker: "005930", name: "삼성전자"}
      kr_secondaries:
        - {ticker: "042700", name: "한미반도체"}
        - {ticker: "403870", name: "HPSP"}
        - {ticker: "095340", name: "ISC"}
      alert_keywords: ["HBM", "AI chip", "AI investment", "반도체 수출", "AI CAPEX"]
      kill_switch: "KR 대장주 2개 모두 VWAP 아래 30분 고착"
      take_profit_leader: [4, 6]    # 1차 익절 %
      take_profit_secondary: [5, 8]

    defense:
      name: "방산"
      type: "event"               # 이벤트형
      us_leaders:
        - {ticker: "LMT", name: "Lockheed Martin", role: "prime defense"}
        - {ticker: "RTX", name: "RTX", role: "defense/aerospace"}
      us_leader_min_strong: 2
      us_secondaries_etf: "XLI"
      kr_leaders:
        - {ticker: "012450", name: "한화에어로스페이스"}
        - {ticker: "079550", name: "LIG넥스원"}
      kr_secondaries:
        - {ticker: "272210", name: "한화시스템"}
      alert_keywords: ["전쟁", "지정학", "방산 수출", "국방비", "NATO", "무기 수출"]
      kill_switch: "지정학 뉴스 완화 OR 대장주 시초고가 돌파 2회 실패"
      take_profit_leader: [3, 5]
      take_profit_secondary: [5, 8]

    energy:
      name: "정유/에너지"
      type: "event"
      us_leaders:
        - {ticker: "XOM", name: "Exxon Mobil", role: "major oil"}
        - {ticker: "CVX", name: "Chevron", role: "major oil"}
      us_leader_min_strong: 2
      us_secondaries_etf: "XLE"
      kr_leaders:
        - {ticker: "010950", name: "S-Oil"}
        - {ticker: "096770", name: "SK이노베이션"}
      kr_secondaries:
        - {ticker: "004090", name: "한국석유"}
        - {ticker: "000440", name: "중앙에너비스"}
        - {ticker: "024060", name: "흥구석유"}
      alert_keywords: ["유가 급등", "호르무즈", "OPEC", "원유", "정제마진", "이란"]
      kill_switch: "유가 선물 장중 음전 OR 대장주 시초 저점 이탈"
      take_profit_leader: [3, 4]
      take_profit_secondary: [5, 8]

    battery_ess:
      name: "배터리/ESS"
      type: "conditional"          # 조건부
      condition: "ESS/AI전환 뉴스 필수 (EV 반등만으로 불가)"
      us_leaders:
        - {ticker: "TSLA", name: "Tesla", role: "EV/ESS leader"}
        - {ticker: "ENPH", name: "Enphase", role: "energy storage"}
      us_leader_min_strong: 2
      us_secondaries_etf: "XLK"
      kr_leaders:
        - {ticker: "373220", name: "LG에너지솔루션"}
        - {ticker: "006400", name: "삼성SDI"}
      kr_secondaries:
        - {ticker: "247540", name: "에코프로비엠"}
        - {ticker: "003670", name: "포스코퓨처엠"}
      alert_keywords: ["ESS", "AI 데이터센터 배터리", "에너지저장", "전력 인프라"]
      negative_keywords: ["EV 수요 부진", "전기차 둔화"]
      kill_switch: "EV 수요 부진 헤드라인 부각"
      entry_rule: "전일고가 위 2일 연속 안착 후만 스윙 진입"
      take_profit_leader: [5, 7]
      take_profit_secondary: [5, 8]

    shipbuilding_lng:
      name: "조선/LNG"
      type: "conditional"
      condition: "LNG 장기계약/발주 뉴스 필수"
      us_leaders:
        - {ticker: "LNG", name: "Cheniere Energy", role: "LNG export"}
        - {ticker: "GLNG", name: "Golar LNG", role: "LNG infra"}
      us_leader_min_strong: 2
      us_secondaries_etf: "XLE"
      kr_leaders:
        - {ticker: "329180", name: "HD현대중공업"}
        - {ticker: "042660", name: "한화오션"}
      kr_secondaries:
        - {ticker: "082740", name: "HSD엔진"}
        - {ticker: "071970", name: "STX중공업"}
      alert_keywords: ["LNG 계약", "선박 발주", "조선 수주", "해양 인프라", "운임"]
      kill_switch: "발주 뉴스 단발성 + 거래대금 미동반"
      entry_rule: "주봉 눌림 스윙 전용 (전일저점 미이탈 + 전일고가 돌파)"
      take_profit_leader: [6, 10]
      take_profit_secondary: [5, 8]
```

### 2. `src/relay/us_tracker.py` — US 대장주 가격 추적

**역할**: 미국 개별 대장주의 최신 가격/레벨을 수집하고 strength/weakness 판정
**데이터 소스**: yfinance (이미 us_overnight_backfill.py에서 사용 중)
**출력**: `data/relay/us_leaders.json`

```python
# 핵심 로직 (수도코드)
for sector in relay_sectors:
    for leader in sector.us_leaders:
        data = yfinance.download(leader.ticker, period="5d")
        # 레벨 계산
        prev_high = data[-2].high      # 전일 고가
        prev_low = data[-2].low        # 전일 저가
        close = data[-1].close         # 최종 종가
        sma20 = 20일 이평

        # 강화/약화 판정
        strength_level = prev_high * 1.001  # 전일고가 +0.1%
        weakness_level = prev_low            # 전일 저가

        is_strong = close > strength_level
        is_weak = close < weakness_level

        # 추세 확인 (5일 연속)
        trend_5d = (close / data[-6].close - 1) * 100
```

### 3. `src/relay/relay_engine.py` — 4단계 릴레이 판정

**핵심**: 각 섹터별로 현재 릴레이 Phase를 판정

```
Phase 0: INACTIVE  — 조건 미충족
Phase 1: WATCH     — US 대장주 1개 강세 (예비 경보)
Phase 2: CONFIRM   — US 대장주 2개+ 강세 + US 2차 확산 (본 경보)
Phase 3: KR_READY  — 한국 대장주 전일 강세/시간외 강세 확인 (실행 준비)
Phase 4: EXECUTE   — 한국 대장주 레벨 회복 → 매수 가능 (실행 경보)
```

**입력 데이터 (기존 시스템 재활용)**:
- `data/relay/us_leaders.json` ← us_tracker.py 생성
- `data/us_market/overnight_signal.json` ← 기존 US Overnight (SOXX, XLE 등)
- `data/ai_brain_judgment.json` ← sector_outlook (뉴스 분석)
- `data/market_news.json` ← 키워드 매칭
- `data/sector_rotation/sector_momentum.json` ← 한국 섹터 모멘텀

### 4. `src/relay/alert_classifier.py` — 경보 분류기

**역할**: 섹터 유형별(지속형/이벤트형/조건부) 차등 경보 로직

| 유형 | 경보 ON 조건 | 경보 OFF 조건 |
|------|-------------|---------------|
| persistent | US 2개+ 강세 + 뉴스 지속 | US 대장주 전원 약세 |
| event | 뉴스 트리거 + US 동반 강세 | 뉴스 해소 + 1일 경과 |
| conditional | 특정 키워드 + negative 키워드 부재 + US 확인 | negative 키워드 출현 |

**경보 레벨 (1~5)**:
- 5: 전 조건 충족 (Phase 4 가능)
- 4: US 확인 + 뉴스 강 (Phase 3)
- 3: US 확인만 (Phase 2)
- 2: 뉴스만 (Phase 1)
- 1: 약한 신호
- 0: 비활성

### 5. `src/relay/execution_rules.py` — 매수/매도 실행 규칙

**역할**: SmartEntry와 연동하여 레벨 기반 매수/매도 판정

**매수 규칙 (A형 재돌파 / B형 눌림)**:
```
A형: 전일 고가 × 1.001~1.005 재돌파 시
B형: VWAP 회복 + 첫 15분 고가 재돌파
```

**매도 규칙**:
```
손절: 첫 눌림 저점 -0.3% OR VWAP 종가 이탈 (먼저 도달)
1차 익절: 손절폭의 2배(2R) OR +4~6% (대장주) / +5~8% (2차 연동주)
최종 청산: 5EMA 종가 이탈 OR 전일 고가 재이탈 OR 대장주 동반 붕괴
```

**금지 규칙**:
- 시초가 +7% 이상 갭상승 종목 첫 5분 추격 금지
- 대장주 VWAP 아래 30분 고착 시 해당 섹터 매매 중단
- 2차 연동주는 대장주 강세 유지 시만 진입

### 6. `scripts/run_relay_engine.py` — 실행 스크립트

```
사용법:
  python scripts/run_relay_engine.py --update     # US 데이터 업데이트 + 릴레이 판정
  python scripts/run_relay_engine.py --signal      # 릴레이 경보 출력
  python scripts/run_relay_engine.py --telegram    # 텔레그램 전송
  python scripts/run_relay_engine.py --all         # 전부 실행
```

### 7. `scripts/relay_us_update.py` — US 데이터 업데이트

- yfinance로 US 대장주 5일 데이터 다운로드
- strength/weakness 레벨 계산
- `data/relay/us_leaders.json` 저장
- BAT-A에 통합 (06:10 미장 마감 직후)

---

## 기존 시스템 수정 사항

### 수정 1: `scripts/schedule_A_us_close.bat`
- Phase 1에 `relay_us_update.py` 추가 (US Overnight 직후)

### 수정 2: `scripts/schedule_B_morning.bat`
- 아침에 `run_relay_engine.py --signal --telegram` 추가

### 수정 3: `src/etf/data_bridge.py`
- `load_relay_signal()` 함수 추가 — relay_signal.json 로드

### 수정 4: `config/settings.yaml`
- `relay_engine` 섹션 추가 (enable/disable 및 기본 파라미터)

---

## 구현 순서

| 단계 | 작업 | 파일 |
|------|------|------|
| 1 | 설정 파일 생성 | `config/relay_sectors.yaml` |
| 2 | US 대장주 추적기 | `src/relay/us_tracker.py` + `scripts/relay_us_update.py` |
| 3 | 릴레이 엔진 코어 | `src/relay/relay_engine.py` + `config.py` |
| 4 | 경보 분류기 | `src/relay/alert_classifier.py` |
| 5 | 실행 규칙 | `src/relay/execution_rules.py` |
| 6 | 통합 실행 스크립트 | `scripts/run_relay_engine.py` |
| 7 | 기존 시스템 연동 | BAT 수정, data_bridge 확장, settings.yaml |
| 8 | 텔레그램 리포트 | 경보 결과 텔레그램 전송 |

---

## 출력 예시 (`data/relay/relay_signal.json`)

```json
{
  "date": "2026-03-03",
  "generated_at": "2026-03-03 06:30",
  "sectors": {
    "ai_semiconductor": {
      "phase": 2,
      "phase_name": "CONFIRM",
      "alert_level": 4,
      "us_leaders_status": {
        "NVDA": {"close": 183.50, "strong": true, "ret_1d": 0.56},
        "AVGO": {"close": 320.10, "strong": true, "ret_1d": 0.40},
        "AMD": {"close": 197.80, "strong": false, "ret_1d": -0.41},
        "MU": {"close": 415.20, "strong": true, "ret_1d": 0.61}
      },
      "us_strong_count": 3,
      "us_secondary_confirm": true,
      "news_keywords_matched": ["HBM", "AI CAPEX"],
      "news_score": 8,
      "kr_leaders_action": {
        "SK하이닉스": {"prev_close": 245000, "prev_high": 248000, "action": "WATCH_BREAKOUT"},
        "삼성전자": {"prev_close": 72000, "prev_high": 73200, "action": "WATCH_BREAKOUT"}
      },
      "kr_secondaries_action": {
        "한미반도체": {"action": "STANDBY"},
        "HPSP": {"action": "STANDBY"},
        "ISC": {"action": "STANDBY"}
      },
      "execution_rules": {
        "buy_type": "A_BREAKOUT",
        "stop_rule": "VWAP_OR_PULLBACK_LOW",
        "take_profit": [4, 6]
      },
      "kill_switch_active": false,
      "summary": "AI반도체 본경보 — US 3/4 강세, HBM/CAPEX 뉴스. KR 대장주 전일고가 돌파 대기"
    },
    "defense": {
      "phase": 1,
      "phase_name": "WATCH",
      "alert_level": 2,
      "summary": "방산 예비경보 — LMT 강세, 지정학 뉴스 약"
    },
    "energy": {
      "phase": 3,
      "phase_name": "KR_READY",
      "alert_level": 5,
      "summary": "에너지 실행준비 — XOM/CVX 동반 강세, 호르무즈 뉴스, S-Oil 전일 강세"
    },
    "battery_ess": {
      "phase": 0,
      "phase_name": "INACTIVE",
      "alert_level": 0,
      "summary": "배터리 비활성 — ESS 전환 뉴스 부재"
    },
    "shipbuilding_lng": {
      "phase": 0,
      "phase_name": "INACTIVE",
      "alert_level": 0,
      "summary": "조선 비활성 — LNG 발주 뉴스 없음"
    }
  },
  "active_alerts": ["ai_semiconductor", "energy"],
  "execution_ready": ["energy"],
  "total_alert_score": 11,
  "recommendation": "에너지 섹터 실행 준비 완료. AI 반도체는 KR 대장주 전일고가 돌파 대기.",
  "telegram_summary": "🚨 릴레이 경보 [03-03]\n✅ AI반도체: 본경보(US 3/4↑) — KR 돌파 대기\n✅ 에너지: 실행준비(XOM/CVX↑+호르무즈) — S-Oil 진입 가능\n⏸ 방산: 예비(LMT↑)\n⬜ 배터리: 비활성\n⬜ 조선: 비활성"
}
```

---

## 텔레그램 리포트 형식

```
🚨 섹터 릴레이 경보 [2026-03-03 06:30]

━━━ AI 반도체 ━━━ [본경보 ★★★★☆]
🇺🇸 NVDA $183.5↑ | AVGO $320.1↑ | MU $415.2↑
📰 HBM 수요 + AI CAPEX 확대
🇰🇷 SK하이닉스 248,000 돌파 대기 | 삼성전자 73,200 돌파 대기
📋 대장50%+후행50% | 손절: VWAP이탈 | 익절: +4~6%

━━━ 에너지 ━━━ [실행준비 ★★★★★]
🇺🇸 XOM $154.8↑ | CVX $190.1↑
📰 호르무즈 해협 봉쇄 + 유가 급등
🇰🇷 S-Oil 전일고가 재돌파 시 진입
📋 대장50%+후행50% | 손절: 시초저점이탈 | 익절: +3~4%

━━━ 방산 ━━━ [예비경보 ★★☆☆☆]
🇺🇸 LMT $679↑ — RTX 미확인
⏸ KR 진입 보류

⬜ 배터리/ESS: 비활성
⬜ 조선/LNG: 비활성
```

---

## 의존성
- yfinance (이미 설치됨 — us_overnight_backfill.py에서 사용)
- 기존 데이터: overnight_signal.json, ai_brain_judgment.json, market_news.json, sector_momentum.json
- 신규 의존성: 없음
