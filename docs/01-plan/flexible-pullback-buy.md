# 유연 눌림목 매수 + 4수급 결합 룰 — PDCA Plan

> **상태**: Plan (5/26 08:35 작성, 16:00 장 마감 후 정식 착수)
> **배경**: 5/26 09:00 1주차 가동 직전 퐝가님 지시
> **원칙**: 메모리 [[feedback-backtest-first]] — 백테스트 숫자 의무, 추정/통념 표현 금지

---

## 1. 배경 (퐝가님 지시 5/26 08:30)

> "꼭 우리가 기준잡은 금액대에 도달을 못하더라도 눌림목이니 너가 유연하게 적용을 해야되.
> 내일 산일전기 같은경우에는 올라 갈껏 같은데....
> 장중에 4번의 수급들도 잘 확인하고 적용을 해야돼."

핵심:
1. **고정 임계값(-10/-20/-30%) 미도달이어도 눌림목 형성 시 유연 매수**
2. **장중 4수급(외인/기관/금투/연기금) 시그널 결합**
3. **상승 추세 종목은 깊은 눌림목 기다리지 말고 매수**

---

## 2. 현재 시스템 자산 (5/26 기준)

| 기능 | 위치 | 가동 |
|------|------|------|
| 분할매수 큐 L1/L2/L3 (-10/-20/-30% 고정) | `src/use_cases/adaptive_buy_queue.py` | ✓ |
| 4수급 1/2/3단계 (ETF+금투+연기금+기타법인) | `scripts/scan_sector_fire.py` V3 (ab3b010) | ✓ |
| C2 진입 점수 (L1 필수+L2 점수) | `src/use_cases/entry_score.py` | ✓ |
| smart_money DUAL_FLOW 105 | 정보봇 통합 (HPSP 검증) | ✓ |
| sniper 수급반전 | 정보봇 통합 | ✓ |
| MA5 위치 / 양봉 50% / 체결강도 100 | `src/use_cases/signal_filter.py` | ✓ |
| **눌림목 자동 트리거 (가격 고정 X)** | **❌ 미구현** | **본 Plan 핵심** |
| **4수급 + 눌림목 결합** | **❌ 미구현** | **본 Plan 핵심** |

---

## 3. 문제 정의

### 3-1. 현행 큐 시스템의 한계
- 디아이티: peak 28,050 → L1 25,245 (-10%) 도달 필요
- 산일전기: peak 341,000 → L2 272,800 (-20%) 도달 필요
- **문제**: 강세 종목은 -10%까지 안 떨어지면 매수 기회 영원히 못 잡음
- **5/21 사례**: 067310(하나마이크론) +15.47% 폭등 시 -10% 미도달 → 매수 0건 → 기회 손실

### 3-2. 4수급 활용도 부족
- V3 시그널은 일별 (BAT-D 16:30) 발화 — **장중 실시간 4수급 변화 미반영**
- 큐는 가격 임계값만 봄 — 4수급 시그널 무시

---

## 4. 가설 (백테스트 검증 필요)

### H1: 얕은 눌림목 + 4수급 만족 = 매수 정당
- 4수급 1단계(ETF+금투+연기금+기타법인) 발화 + **-3~-5% 눌림목** = L1 (-10%) 대체 매수

### H2: 4수급 3단계 (외인 확인) = 즉시 매수
- 외인 매수 진입 시 **눌림목 깊이 무관 + 가격 도달 무관** = 즉시 추가 매수
- 메모리 5/14 22:30: "3단계 완성: D+1 +2.02%, 적중률 59.1%, D+3 +4.00%"

### H3: 깊은 눌림목 + 4수급 미발화 = 매수 보류
- L2 (-20%) 도달이어도 4수급 0단계면 매수 보류 (가치 함정 회피)

---

## 5. Design 골격 (장 마감 후 정식 착수)

### 5-1. 새 함수 `evaluate_flexible_buy_signal()`
```python
# src/use_cases/adaptive_buy_queue.py 신규
def evaluate_flexible_buy_signal(
    ticker: str,
    current_price: int,
    peak_price: int,
    supply_signal_level: int,  # 0/1/2/3
) -> dict:
    """가격 + 4수급 결합 매수 시그널.

    Returns:
        {"action": "BUY"|"WAIT"|"SKIP",
         "level": "L1"|"L1.5"|"L2"|"L3"|"FLEX",
         "reason": str,
         "alloc_pct": float}
    """
    pullback_pct = (peak_price - current_price) / peak_price * 100
    # H2: 외인 진입 시 즉시 (가격 무관)
    if supply_signal_level >= 3:
        return {"action": "BUY", "level": "FLEX", "alloc_pct": 0.3, ...}
    # H1: 얕은 눌림목 + 2단계 이상
    if 3 <= pullback_pct <= 5 and supply_signal_level >= 1:
        return {"action": "BUY", "level": "L1.5", "alloc_pct": 0.15, ...}
    # 기존 L1/L2/L3 (4수급 0단계는 보류)
    ...
```

### 5-2. 큐 데이터 모델 확장
```json
{
    "stages": [
        {"level": "FLEX", "trigger": "4수급_3단계", "alloc_ratio": 0.3, ...},
        {"level": "L1.5", "trigger": "얕은눌림목_4수급_1단계", "alloc_ratio": 0.15, ...},
        {"level": "L1", "target_pct": 0.9, ...},
        {"level": "L2", "target_pct": 0.8, ...},
        {"level": "L3", "target_pct": 0.7, ...}
    ]
}
```

### 5-3. 4수급 실시간 fetch
- BAT-D 일별 → **매 30분 cron 사이클에 4수급 fetch 추가**
- 단타봇 4유형 (외인/기관/개인/기타) 실시간 데이터 활용
- 정보봇 OHLCV 39컬럼 `Foreign_Net`, `Inst_Net` 활용

---

## 6. 백테스트 시나리오 (의무)

### 데이터 소스
- 정보봇 OHLCV CSV (2,632종목) — `data/external/jgis_ohlcv/`
- 자체 5min/15min parquet — `data/intraday/{5min,15min}/`
- Supabase `quant_sector_fire` (4/25~)

### 시나리오
| 룰 | 표본 | D+1 평균 | 승률 | +10%↑ 비율 | MDD | PF | 평균보유일 |
|----|------|---------|------|----------|-----|-----|-----------|
| 기존 L1/L2/L3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **H1 얕은눌림목+1단계** | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **H2 즉시 (3단계)** | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **H3 깊은+0단계 SKIP** | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Acceptance Criteria
- D+1 평균 ≥ 기존 +0.84% (MVP-2.6 백테스트 R2)
- 승률 ≥ 56.9% (기존)
- +10%↑ 비율 ≥ 46.5% (C2 백테스트)
- MDD ≤ 기존 +2%p (악화 허용 한도)
- PF ≥ 1.5

---

## 7. 일정

| 날짜 | 작업 | 산출물 |
|------|------|--------|
| 5/26 16:00~22:00 | Plan 정식화 + 백테스트 데이터 준비 | 본 문서 보강 |
| 5/27(수) | 백테스트 실행 (3개월 + 6개월) | `docs/02-design/flexible-pullback-backtest.md` |
| 5/28(목) | Design + 코드 구현 | `src/use_cases/adaptive_buy_queue.py` 확장 |
| 5/29(금) | 회귀 테스트 + paper mirror 검증 | 회귀 통과 보고 |
| 6/2(월) 09:00 | 정식 가동 | — |

---

## 8. 즉시 적용 (오늘 5/26)

### 산일전기 EYE-07 워치리스트 추가 ✓
- `config/settings.yaml` `eye_watchlist`에 `062040` 추가 (commit 직후 적용)
- 3종 큐 종목(디아이티/산일전기/원익IPS) 모두 워치리스트 등록 (4수급 시그널 발화 시 EYE 알림)
- 매수 X — 정보 알림만 (안전망 보존)

### 09:00 가동
- 현행 안전망 그대로 (워밍업 의미 보존)
- 디아이티 L1 25,245 / 산일전기 L2 272,800 / 원익IPS L2 121,200 기존 큐 룰 적용
- EYE-07 워치리스트 +3% 급등 알림으로 4수급 종목 모니터링

---

## 9. 리스크 + 메모리 규칙 준수

- [[feedback-backtest-first]]: 백테스트 결과 숫자 없이 룰 변경 X
- [[feedback-warmup-meaning-first]]: 1주차 워밍업 의미 보장 (현행 안전망 유지)
- [[feedback-subagent-team-first]]: Plan 완료 시 서브에이전트 분업 (백테스트/구현/회귀)
- [[project-quantbot-strategy-taxonomy]]: 본 Plan은 **단기 트랙** (차트영웅+정석단타+3파/4파) 강화

---

## 10. 미해결 질문 (16:00 작업 시 결단)

1. **4수급 fetch 빈도**: 매 30분 vs 매 5분 (KIS API rate limit + 정확도 trade-off)
2. **눌림목 정의**: 단순 -3~-5% vs MA20/MA60 이탈 + RSI 50 이하 등 복합
3. **FLEX 매수 alloc**: L1/L2/L3 (각 30%/30%/30%)에 추가 vs 대체
4. **산일전기 L1 FAILED 복구**: 5/25 13:00 buy_limit 미구현 이슈로 FAILED — PENDING 복구 vs 그대로 두고 신규 룰 적용

---

(끝)
