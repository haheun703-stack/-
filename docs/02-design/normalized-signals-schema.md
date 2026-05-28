# normalized_signals 공통 스키마 (Trading Factory v1)

> **상태**: Draft (퀀트봇 측 제안, 단타봇/정보봇/Codex 합의 대기)
> **연결**: docs/01-plan/trading-factory-v1-architecture.md Step 3

## 1. 목적

봇 간 signal 데이터 표준화 → selector가 통일된 입력 수신.
"LLM은 종목을 임의 선택하지 않는다" 원칙 강제.

## 2. 스키마 v1 (jsonl, append-only)

```jsonl
{
  "signal_id": "q_2026-05-28_sector_fire_001",
  "bot": "quant",
  "engine": "sector_fire_v3",
  "ticker": "240810",
  "name": "원익IPS",
  "side_bias": "BUY",
  "timeframe": "swing_3d",
  "score": 81.7,
  "confidence": "strong",
  "components": {
    "earnings_value": 12,
    "sector_theme": 25,
    "investor_3_5d_accumulation": 18,
    "bottom_structure": 10,
    "volume_increase": 8,
    "event_confidence": 8.7,
    "short_term_overheat": -0,
    "short_sell_supply": -0,
    "market_regime_risk": -0,
    "stop_loss_width": -0
  },
  "reasons": [
    "AI반도체 섹터 53점",
    "외인+기관 동시매수 (3일 누적 +120만주)",
    "MA5 위 + 양봉 65%"
  ],
  "raw_data_ref": "data/scan/sector_fire_v3_20260528_1730.json",
  "created_at": "2026-05-28T17:30:00+09:00",
  "expires_at": "2026-05-31T15:30:00+09:00"
}
```

## 3. 필수 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `signal_id` | string | 봇 prefix + 날짜 + 엔진 + 일련번호 |
| `bot` | enum | "quant" / "day" / "info" |
| `engine` | string | 신호 발생 엔진 (예: sector_fire_v3, limit_up_recovery, foreign_accumulation) |
| `ticker` | string | 6자리 종목코드 |
| `name` | string | 한글 종목명 |
| `side_bias` | enum | "BUY" / "SELL" / "NEUTRAL" |
| `timeframe` | enum | "intraday" / "swing_1d" / "swing_3d" / "swing_5d" |
| `score` | float | 점수 (0~100, 엔진별 정규화) |
| `confidence` | enum | "weak" / "medium" / "strong" |
| `components` | dict | 점수 구성 요소 (감리용 투명성) |
| `reasons` | list[string] | 사람이 읽을 이유 (3~5개) |
| `created_at` | ISO8601 | 신호 발생 시각 |
| `expires_at` | ISO8601 | 유효 만료 시각 |

## 4. 옵션 필드

- `raw_data_ref`: 원본 데이터 파일 경로 (감리용)
- `event_links`: 정보봇 event_signals 연결 (`event_id` 목록)
- `historical_winrate`: 동일 엔진의 과거 30일 적중률
- `notes`: 자유 텍스트 (디버깅용)

## 5. 봇별 engine 목록 (예시)

### 5-1. 퀀트봇 (swing)
- `sector_fire_v3` (섹터발화)
- `bottom_rebound` (바닥반등)
- `supply_surge` (수급전환)
- `earnings_gap` (실적괴리)
- `etf_theme_rotation` (ETF/테마 회전)
- `foreign_accumulation` (외인 누적매집)
- `fibonacci_level` (피보나치)
- `chart_hero_5gate` (차트영웅)

### 5-2. 단타봇 (intraday)
- `limit_up_recovery` (상한가 풀림 후 회복)
- `nxt_strength` (NXT 야간 강도)
- `dual_accumulation` (외인+기관 동시매집)
- `volume_spike` (거래대금/거래량 급증)
- `vwap_recovery` (VWAP 회복)
- `rule_d` (Rule D 우선순위)

### 5-3. 정보봇 (event)
- `news_sentiment` (뉴스 감성)
- `dart_disclosure` (DART 공시)
- `macro_event` (글로벌 매크로)
- `sector_news_cluster` (섹터 뉴스 클러스터)

## 6. selector 입력 흐름

```
normalized_signals.jsonl (각 봇 출력)
  ↓
selector (approved_swing_selector / approved_intraday_selector)
  - 봇별 source_weight 적용
  - TOP N 종목 선정
  ↓
candidate_snapshot.json (선정 후보 + 점수)
  ↓
order_intents.jsonl (실제 주문 의도)
  ↓
order_intents_gate.assert_order_intent_exists()
  ↓
KisOrderAdapter / PaperOrderAdapter (주문 집행)
```

## 7. score 정규화 (봇 간 비교 가능)

- 0~100 범위
- 50 = 중립
- 80+ = 강력 포착 (action 권장)
- 65~79 = 포착 (관찰)
- 50~64 = 관심 (대기)
- <50 = 패스

## 8. 검수 (Codex 검수 요청)

1. 스키마 필드 추가/삭제 권장 (특히 risk 측면)
2. signal_id 중복 방지 메커니즘
3. expires_at 기본값 (engine별 권장)
4. components 표준화 (봇 간 비교 가능성)
5. 정보봇 event_signals와 normalized_signals 통합 가능성

## 9. 단타봇 측 합의 사항

- `engine` enum 확정 (단타봇이 운영 중인 엔진 목록)
- `timeframe` 추가 옵션 (예: `intraday_5min`, `intraday_30min`)
- `components` 가중치 표준 (예: 상한가 +30, NXT +20 등)

## 10. 정보봇 측 합의 사항

- `event_signals.jsonl`을 normalized_signals 변형으로 통일 가능?
- 또는 별도 스키마 유지 + `event_links`로 연결
