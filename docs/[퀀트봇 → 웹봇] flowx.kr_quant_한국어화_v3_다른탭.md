# [퀀트봇 → 웹봇] flowx.kr/quant 한국어화 v3 — 다른 탭 약어

**작성일**: 2026-05-17 (일)
**작성자**: 퀀트봇 (quantum-master)
**대상**: 웹봇 (FLOWX 페이지 운영자)
**전제**: v2 페이퍼 매매 탭 한국어화 완료 (커밋 3778911) 이후 후속

---

## 🎯 범위

페이퍼 매매 탭 외 **8개 탭에 등장하는 영어 약어** 한국어 풀이.

탭 목록:
1. 퀀트시스템
2. ~~페이퍼 매매~~ (v2 완료)
3. **섹터 발화(FIRE)**
4. **테마 수급**
5. **테마 모멘텀**
6. **상한가 눌림목**
7. **실적 괴리(GAP)**
8. **급락 반응**
9. **대형주 피보나치**
10. **소형주 피보나치**

---

## 1. 공통 약어 매핑 (전 탭 공용)

```typescript
const COMMON_ABBR_KR: Record<string, string> = {
  // 기술 지표
  "RSI": "상대 강도 지수 (RSI)",
  "MA20": "20일 이동평균 (MA20)",
  "MA60": "60일 이동평균 (MA60)",
  "MA120": "120일 이동평균 (MA120)",
  "BB": "볼린저밴드 (BB)",
  "MACD": "MACD 지표",
  "ATR": "변동성 지표 (ATR)",
  "VWAP": "거래량 가중 평균가 (VWAP)",
  
  // 재무 지표
  "PER": "주가수익비율 (PER)",
  "PBR": "주가순자산비율 (PBR)",
  "ROE": "자기자본이익률 (ROE)",
  "EPS": "주당순이익 (EPS)",
  "EV": "기업가치 (EV)",
  "EBITDA": "EBITDA (영업이익+감가상각)",
  
  // 수익률 / 성과
  "WR": "승률",
  "PF": "손익비",
  "MDD": "최대 손실폭",
  "ROI": "수익률",
  "PnL": "손익",
  "Sharpe": "샤프 비율",
  
  // 시장 액션
  "GAP": "괴리율 (예상 vs 실적)",
  "FIRE": "섹터 발화",
  "BUY": "매수",
  "SELL": "매도",
  "HOLD": "보유",
  "STRONG_BUY": "강한 매수",
  "STRONG_SELL": "강한 매도",
};

function abbrKr(code: string): string {
  return COMMON_ABBR_KR[code] || code;
}
```

---

## 2. 탭별 한국어화 권장

### 2-1. 섹터 발화(FIRE) 탭

```typescript
const FIRE_LABELS: Record<string, string> = {
  "fire_score": "발화 점수",
  "fire_level": "발화 등급",
  "STRONG": "강한 발화 🔥🔥🔥",
  "MODERATE": "중간 발화 🔥🔥",
  "WEAK": "약한 발화 🔥",
  "EARLY": "초기 발화",
  "MATURE": "성숙 발화",
  "rotation_rank": "순환 순위",
  "momentum_acceleration": "모멘텀 가속도",
  "breadth": "확산도",
  "leader_count": "주도주 개수",
};
```

표 헤더 권장:
- "Sector" → "섹터"
- "Score" → "발화 점수"
- "Rank" → "순위"
- "Momentum" → "모멘텀"
- "Breadth" → "확산도"
- "Leaders" → "주도주"

### 2-2. 테마 수급 탭

```typescript
const SUPPLY_LABELS: Record<string, string> = {
  "foreign_net": "외인 순매수",
  "institution_net": "기관 순매수",
  "pension_net": "연기금 순매수",
  "finance_net": "금융투자 순매수",
  "other_corp_net": "기타법인 순매수",
  "individual_net": "개인 순매수",
  "dual_buy": "양매수 (외인+기관 동반)",
  "triple_buy": "3주체 합류 (외인+기관+연기금)",
  "smart_money": "스마트 머니",
  "exhaustion_rate": "외인 소진율",
};
```

표시 권장:
- 금액: `+1.2조` / `-840억` 형식 (조/억 단위)
- 양매수: `🟢🟢 양매수`, 3주체: `🟢🟢🟢 3주체 합류`

### 2-3. 테마 모멘텀 탭

```typescript
const MOMENTUM_LABELS: Record<string, string> = {
  "momentum_score": "모멘텀 점수",
  "acceleration": "가속도",
  "deceleration": "감속",
  "trend_strength": "추세 강도",
  "ret_1d": "1일 수익률",
  "ret_5d": "5일 수익률",
  "ret_20d": "20일 수익률",
  "volume_surge": "거래량 폭증",
  "vol_ratio": "거래량 비율 (전일 대비)",
};
```

### 2-4. 상한가 눌림목 탭

```typescript
const PULLBACK_LABELS: Record<string, string> = {
  "pullback_pct": "눌림 폭",
  "days_from_high": "고점 이후 일수",
  "ma20_distance": "20일선 이격도",
  "rsi_oversold": "RSI 과매도",
  "volume_dry_up": "거래량 고갈",
  "support_level": "지지선",
  "resistance_level": "저항선",
  "bb_lower": "볼린저 하단",
  "ULIMIT_DAY": "상한가 발생일",
};
```

### 2-5. 실적 괴리(GAP) 탭

```typescript
const GAP_LABELS: Record<string, string> = {
  "actual_eps": "실적 EPS (확정)",
  "consensus_eps": "예상 EPS (컨센서스)",
  "eps_surprise": "EPS 서프라이즈 (실적 - 예상)",
  "surprise_pct": "서프라이즈 비율",
  "POSITIVE_GAP": "긍정 괴리 (실적 > 예상)",
  "NEGATIVE_GAP": "부정 괴리 (실적 < 예상)",
  "guidance_up": "가이던스 상향",
  "guidance_down": "가이던스 하향",
};
```

### 2-6. 급락 반응 탭

```typescript
const REBOUND_LABELS: Record<string, string> = {
  "drop_pct": "낙폭",
  "rebound_pct": "반등폭",
  "rebound_days": "반등 소요 일수",
  "bottom_signal": "바닥 시그널",
  "volume_climax": "거래량 절정 (투매)",
  "RSI_OVERSOLD": "RSI 과매도 (30 이하)",
  "BB_BOTTOM": "볼린저 하단 터치",
  "smart_money_inflow": "스마트 머니 유입",
};
```

### 2-7. 대형주/소형주 피보나치 탭

```typescript
const FIB_LABELS: Record<string, string> = {
  "fib_0": "기준선 (0%)",
  "fib_236": "1단 되돌림 (23.6%)",
  "fib_382": "2단 되돌림 (38.2%)",
  "fib_500": "절반 되돌림 (50%)",
  "fib_618": "황금 되돌림 (61.8%)",
  "fib_786": "깊은 되돌림 (78.6%)",
  "fib_1000": "전저점 (100%)",
  "current_fib_level": "현재 위치",
  "next_target": "다음 목표가",
  "stop_level": "손절 수준",
};
```

표시 권장:
- 도형 라벨: `0.618` → `61.8% (황금)` 식 표시
- 호버 시: "황금 되돌림선 = 강한 지지/저항 + 매수 타이밍"

---

## 3. 공통 UI 패턴

### 3-1. 약어 + 풀이 동시 표시
```tsx
<th title="상대 강도 지수: 14일 기간 동안 종가 변동의 강도를 0-100으로 측정">
  RSI (상대강도)
</th>
```

### 3-2. 등급 chip 색상 표준
```typescript
const GRADE_COLORS = {
  "🟢 우수": "bg-green-100 text-green-700",
  "🟡 보통": "bg-yellow-100 text-yellow-700",
  "🟠 주의": "bg-orange-100 text-orange-700",
  "🔴 부진": "bg-red-100 text-red-700",
  "🔥 발화": "bg-red-100 text-red-700 font-bold",
  "💎 양매수": "bg-blue-100 text-blue-700",
};
```

### 3-3. 숫자 표기 한국 컨벤션
- 1,200,000원 (천 단위 콤마)
- 1.2조 / 840억 / 4,500만 (대용량은 한국식)
- 수익률: +5.23% (소수점 2자리 + 부호)

---

## 4. 우선순위

| 탭 | 우선순위 | 이유 |
|---|---|---|
| 섹터 발화(FIRE) | 🔴 즉시 | 가장 많이 봄, FIRE 약어 다수 |
| 테마 수급 | 🔴 즉시 | 핵심 지표, dual_buy 등 영어 표시 |
| 실적 괴리(GAP) | 🟠 빠르게 | EPS/PER 약어 |
| 상한가 눌림목 | 🟠 빠르게 | RSI/MA20/BB 약어 |
| 테마 모멘텀 | 🟡 보통 | 일부 영어 약어 |
| 급락 반응 | 🟡 보통 | 일부 영어 약어 |
| 피보나치 | 🟢 점진 | 숫자 위주, 도형 라벨만 |

---

## 5. 일관성 체크리스트

웹봇 적용 시 다음 일관성 확인:

- [ ] 모든 약어가 한국어 풀이로 노출
- [ ] 동일 약어가 다른 탭에서 동일 한국어로 번역 (예: RSI는 모든 탭에서 "상대 강도 지수")
- [ ] 호버 툴팁에 짧은 설명 제공
- [ ] 색상 등급 표준 통일 (🟢 우수 = 녹색 등)
- [ ] 숫자 한국 컨벤션 (조/억/만)

---

## 변경 이력

| 일자 | 내용 |
|---|---|
| 2026-05-16 | v1: paper 탭 4섹션 안내 |
| 2026-05-17 | v2: paper 탭 한국어화 + 차트 보완 |
| 2026-05-17 | **v3: 다른 8개 탭 약어 한국어화 (이 문서)** |
