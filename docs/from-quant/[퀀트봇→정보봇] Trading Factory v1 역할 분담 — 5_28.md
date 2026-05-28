# [퀀트봇 → 정보봇] Trading Factory v1 역할 분담 지시서

> **작성일**: 2026-05-28 12:10 KST
> **연결**: 퀀트봇 docs/01-plan/trading-factory-v1-architecture.md (사용자 5/28 결단)

## 1. 정보봇 역할 (확정)

**Event Intelligence Bot** — 정보/뉴스/공시/글로벌 이벤트 공급

### 책임 (Have)
- 뉴스, 공시, 글로벌 지수, 환율, 섹터 이벤트 → `event_signals.jsonl` 표준화
- 각 이벤트에 `ticker`, `sector`, `freshness`, `impact_score`, `confidence` 부착
- 단타봇/퀀트봇 selector가 읽을 수 있게 Supabase 또는 파일로 공급

### 영구 금지 (Have NOT)
- ❌ KIS 주문/매수/매도 함수 import 절대 금지
- ❌ 자기 판단으로 매매 후보 확정 금지
- ❌ order_intents 생성 금지 (단타봇/퀀트봇만 생성)

## 2. event_signals 스키마 (제안)

```jsonl
{"event_id":"e_2026-05-28_001","source":"news","ticker":"240810","name":"원익IPS","sector":"AI반도체","headline":"...","sentiment":"positive","impact_score":0.85,"confidence":"high","freshness_minutes":15,"published_at":"2026-05-28T09:00:00+09:00"}
```

## 3. 검수 요청 (정보봇 측 응답 의무)

1. 정보봇 측에 KIS 주문 함수 import 잔존 여부 (Codex 검수 + 정보봇 self-audit)
2. `intelligence_supply_demand` 테이블 스키마 명확화 (단타봇이 `WHERE ticker = %s` 쿼리 시도 → ticker 컬럼 없음 에러)
3. `event_signals.jsonl` 출력 일정 (실시간 vs 배치)
4. 단타봇/퀀트봇 selector가 정보봇 데이터 읽는 표준 인터페이스 (Supabase REST vs 파일 vs MCP)

## 4. 퀀트봇 ↔ 정보봇 데이터 흐름 (현재 5/28)

- VPS `/home/ubuntu/jgis/stock_data_daily/` 39컬럼 OHLCV (퀀트봇이 매일 활용)
- Supabase 7테이블 (sector_investor_flow, etf_investor_flow, program_trading, intelligence_supply_demand, stock_technicals, stock_valuations, treemap_stocks)
- 신규: `event_signals.jsonl` 추가 (Trading Factory v1)

## 5. 산출물 (의무)

| 파일 | 용도 |
|------|------|
| `event_signals.jsonl` | 실시간 이벤트 스트림 |
| `sector_event_heatmap.json` | 섹터별 이벤트 강도 |
| `morning_event_brief.md` | 09:00 사전 브리핑 |

## 6. 영구 금지 검증 (정보봇 self-audit)

정보봇 측 코드에서 다음 패턴 grep + 0건 확인 보고 의무:
```bash
grep -rn "create_market_sell_order\|create_market_buy_order\|create_limit_sell_order\|create_limit_buy_order\|sell_market\|buy_market\|sell_limit\|buy_limit\|order_sell\|order_buy" .
grep -rn "from.*kis_order_adapter\|import.*mojito\|KoreaInvestment" .
```

기대 결과: **모든 grep 0건**.
