# [퀀트봇 → 웹봇] flowx.kr/quant 한국어화 + 차트 보완 v2

**작성일**: 2026-05-17 (일)
**작성자**: 퀀트봇 (quantum-master)
**대상**: 웹봇 (FLOWX 페이지 운영자)
**긴급도**: ★★★ — 사용자 가시성 직접 영향

---

## 🎯 사용자(퐝가님) 지시

> **"주린이들이 이해 못 하잖아. 모든 작업은 한글로 풀어서 작성하고 게시"**

- 코드 내부 식별자(strategy 값): 영어 유지 (개발 표준)
- 페이지/텔레그램/리포트 사용자 표시: **반드시 한국어 풀이**
- 약어(WR/PF 등): **한국어로 풀어쓰기**

---

## 1. 자산 곡선 차트 — 라벨 잘림 수정

현재: 우측 끝 "+29.8%" 라벨이 차트 영역 밖으로 잘림.

### 권장 해결책 (3개 중 택일)

**A. 우측 padding 추가 (권장)**:
```typescript
// recharts 사용 시
<LineChart margin={{ top: 20, right: 70, bottom: 20, left: 10 }}>
```

**B. 라벨을 점 위로 표시 (배지 스타일)**:
```typescript
<LabelList dataKey="cumulative_pct" position="top" 
  formatter={(v) => `${v > 0 ? '+' : ''}${v}%`} />
```

**C. 마지막 데이터 포인트 별도 강조**:
- 마지막 점만 큰 dot + 텍스트 위쪽 배치
- 텍스트 배경에 흰색 박스(stroke)로 가독성 ↑

---

## 2. 표 헤더 한국어화

| 현재 (영어 약어) | 변경 (한국어 풀이) | 의미 설명 |
|---|---|---|
| WR | **승률** | 익절 거래 비율 |
| PF | **손익비** | 총 수익 ÷ 총 손실 |
| 평균수익 | 평균수익률 (유지) | |
| 등급 | 평가 (유지) | |
| 거래 | 거래수 (유지) | |

### 추가: 헤더 호버 시 툴팁 권장
```typescript
<Tooltip content="승률: SELL 거래 중 익절(pnl>0) 비율">
  <th>승률</th>
</Tooltip>
<Tooltip content="손익비(Profit Factor): 총 수익 합계 ÷ 총 손실 합계">
  <th>손익비</th>
</Tooltip>
```

---

## 3. Strategy 값 한국어 매핑 사전 ⭐ 핵심

### 추가할 사전 (PaperTradesView.tsx)

```typescript
const STRATEGY_KR: Record<string, string> = {
  // ── 청산 사유 ──
  'TRAILING_STOP': '트레일링 스톱 (고점 -2%)',
  'TAKE_PROFIT_T1': '1차 익절 (+10% 부분 매도)',
  'TAKE_PROFIT_T2': '2차 익절 (+20% 전량)',
  'TAKE_PROFIT': '익절',
  'STOP_LOSS': '손절 (-7%)',
  'MAX_HOLD': '보유 기간 만료',
  'SUPPLY_EXIT': '수급 이탈 (3일 연속)',
  'NEUTRAL_EXIT': '방향 중립 전환',
  'DIRECTION_SWITCH': 'JARVIS 방향 전환',

  // ── Bluechip (우량주 TOP 30) ──
  'BLUECHIP_STOP_LOSS': '우량주 손절',
  'BLUECHIP_SUPPLY_EXIT': '우량주 수급 이탈',
  'BLUECHIP_TRAILING_STOP': '우량주 트레일링',
  'BLUECHIP_TAKE_PROFIT_T1': '우량주 1차 익절',
  'BLUECHIP_TAKE_PROFIT_T2': '우량주 2차 익절',
  'BLUECHIP_MAX_HOLD': '우량주 보유 만료',
  'BLUECHIP_A_쌍끌이': '우량주 쌍끌이 (외인+기관 동시 매수)',
  'BLUECHIP_B_기관연기금': '우량주 기관·연기금 매수',
  'BLUECHIP_C_3주체합류': '우량주 3주체 합류',
  'BLUECHIP_D_외인폭발': '우량주 외인 폭발 매수',
  'BLUECHIP_E_연기금매집': '우량주 연기금 매집',
  'BLUECHIP_F_금투기타': '우량주 금투·기타법인',

  // ── Paper (일반 페이퍼) ──
  'SCAN': '일반 스캔 진입',
  'ALPHA': '알파 시그널 진입',
  'REBALANCE': '리밸런싱',
  'AI_BRAIN': 'AI 두뇌 추천',
  'AI_LARGECAP': 'AI 대형주 추천',
  'INTRADAY_LEARNED': '장중 학습 시그널',
  'PB15_BB': '15% 눌림목 + 볼린저밴드',
  'PULLBACK15_VOL3x': '눌림목 거래량 3배 폭증',
  'PULLBACK15_DUAL': '눌림목 + 양매수',
  'PHASE5_STAGE3': '3단계 완성 시그널',
  'PHASE8_THEME_DUAL': '테마 양매수',
  'FOREIGN_SURGE_PB': '외인 폭발 + 눌림목',
  'SILENT_GOLD_COMBO': '사일런트 골든 콤보',
  'LAGGARD_FOLLOW': '래거드 추격',

  // ── ETF 방향 (JARVIS) ──
  'ETF_LONG_BUY': 'ETF 롱 매수 (KODEX 200)',
  'ETF_STRONG_LONG_BUY': 'ETF 강한 롱 (KODEX 레버리지)',
  'ETF_SHORT_BUY': 'ETF 인버스 매수 (KODEX 인버스)',
  'ETF_STRONG_SHORT_BUY': 'ETF 강한 인버스 (200선물인버스2X)',
  'ETF_LONG_SWITCH': 'ETF 롱 스위칭',
  'ETF_STRONG_LONG_SWITCH': 'ETF 강한 롱 스위칭',
  'ETF_SHORT_SWITCH': 'ETF 인버스 스위칭',
  'ETF_STRONG_SHORT_SWITCH': 'ETF 강한 인버스 스위칭',
  'ETF_LONG_SELL': 'ETF 롱 청산',
  'ETF_STOP_LOSS': 'ETF 손절',
  'ETF_TAKE_PROFIT_T2': 'ETF 2차 익절 (+10%)',
  'ETF_TRAILING_STOP': 'ETF 트레일링 (고점 -2%)',
  'INVERSE_MAX_HOLD': '인버스 강제 청산 (D+2 만료)',
};

function getStrategyKr(strategy: string): string {
  if (!strategy) return '미분류';
  return STRATEGY_KR[strategy] || strategy;
}
```

### 적용

```typescript
// 전략별 적중률 표 (PaperTradesView.tsx:75-103)
<td>{getStrategyKr(row.strategy)} <span className="text-xs text-gray-400">({row.strategy})</span></td>

// 매매 내역 테이블
<td>{getStrategyKr(trade.strategy)}</td>
```

→ 표시는 한국어, 코드값은 작은 회색으로 보조 표시 (디버깅용).

---

## 4. 등급 chip 한국어

```typescript
function getGradeKr(pf: number): { emoji: string, text: string, color: string } {
  if (pf >= 1.5) return { emoji: '🟢', text: '우수', color: 'text-green-600' };
  if (pf >= 1.0) return { emoji: '🟡', text: '보통', color: 'text-yellow-600' };
  return { emoji: '🔴', text: '부진', color: 'text-red-600' };
}
```

표시:
- 🟢 우수 (손익비 1.5↑)
- 🟡 보통 (손익비 1.0~1.5)
- 🔴 부진 (손익비 1.0↓)

---

## 5. 카테고리 필터 라벨

| 현재 | 한국어 권장 |
|---|---|
| 전체 | 전체 (유지) |
| Paper | **일반 페이퍼** |
| Bluechip | **우량주** |
| ETF | ETF (유지) |
| 청산만 | 청산만 (유지) |

---

## 6. memo 필드 파싱 chip 한국어

`memo` 필드의 chip 표시 한국어화:

| memo 패턴 | chip 표시 |
|---|---|
| `backfill_20260516` | 📦 과거 데이터 |
| `보유중` | 📌 보유 중 |
| `등급:AA` | ⭐ 최고 등급 |
| `등급:A` | ⭐ 우수 등급 |
| `점수:XX` | 🎯 점수 XX |
| `위험:정상(NN점) ×1.0` | ✅ 시장 정상 |
| `위험:주의(NN점) ×0.8` | 🟢 시장 주의 |
| `위험:경고(NN점) ×0.6` | 🟡 시장 경고 |
| `위험:위험(NN점) ×0.4` | 🟠 시장 위험 |
| `위험:위기(NN점) ×0.2` | 🔴 시장 위기 |
| `부분` | 부분 매도 |
| `전량` | 전량 매도 |
| `JARVIS` | 🤖 JARVIS 방향 |

---

## 7. 사이드 라벨 한국어

차트 상단/하단 라벨:
- `최근 30일 · SELL 합성 · 한국 컨벤션` → `최근 30일 · 매도 합산 · 한국 통화`
- 또는 그대로 유지 (이미 한국어 + 영어 혼합)

---

## 8. 페이지 헤더 설명문 추가 (권장)

페이퍼 매매 탭 상단에 짧은 설명:

```
📊 페이퍼 매매
실전 자금이 아닌 가상 자금으로 시뮬레이션한 결과입니다.
모든 매매는 실제 시장 가격으로 계산되며, 전략의 유효성을 검증합니다.
```

자산 곡선 위:
```
누적 수익률: 시작 자본 1억원 대비 현재까지의 변화율
4/16 시작 → 5/15 +29.8% (페이퍼 = 1.298억)
```

전략별 적중률 위:
```
각 진입/청산 전략이 얼마나 효과적인지 보여줍니다.
승률 50%↑ + 손익비 1.5↑ = 신뢰할 수 있는 전략.
```

---

## 9. 전체 페이지 한국어화 원칙 (모든 탭 공통)

페이지에 영어 약어/식별자가 등장하는 곳마다 동일 패턴 적용:

```typescript
// 패턴 A: 한국어 (식별자)
<span>{한국어풀이} <small className="text-gray-400">({식별자})</small></span>

// 패턴 B: 한국어만 (식별자 숨김, 호버 시 표시)
<Tooltip content={식별자}><span>{한국어풀이}</span></Tooltip>

// 패턴 C: 약어 + 한국어 (헤더용)
<th>승률<br/><small>(WR)</small></th>
```

---

## 10. 다른 탭의 영어 약어 점검 대상

| 탭 | 점검 필요 약어 |
|---|---|
| 섹터 발화(FIRE) | FIRE, RSI, MA20 |
| 테마 수급 | 수급 (외인/기관/연기금/금투/기타법인 → 풀이 OK) |
| 테마 모멘텀 | 모멘텀, 가속도 |
| 실적 괴리(GAP) | GAP, EPS, PER |
| 대형주 피보나치 | 피보나치, 0.382 / 0.618 (수준) |

각 탭별 별도 한국어화 가이드 필요 시 별개 지시서 작성.

---

## 11. 변경 우선순위 (요청 순서)

1. **🔴 즉시**: 자산 곡선 "+29.8%" 라벨 잘림 수정
2. **🔴 즉시**: 전략 한국어 매핑 사전 적용 (STRATEGY_KR)
3. **🟠 빠르게**: 표 헤더 WR/PF → 승률/손익비
4. **🟡 보통**: 카테고리 필터 + memo chip + 헤더 설명문
5. **🟢 점진적**: 다른 탭 영어 약어 점검

---

## 변경 이력

| 일자 | 내용 |
|---|---|
| 2026-05-16 | v1: 4개 섹션 데이터 안내 |
| 2026-05-17 | **v2: 한국어화 + 차트 보완 (이 문서)** |
