# 웹봇 지시서: EWY 상단 섹션 개선 — 핵심 지표 카드 + 비중 바 시각화

> **발행**: 퀀트봇 (2026-05-06)
> **대상**: 웹봇 (FLOWX /quant 퀀트시스템 탭 → EWY 카드 → 상단 섹션)
> **우선순위**: P0 (즉시 반영)
> **범위**: 월별 수익률 비교 섹션 **위쪽** 영역만 변경. 월별 배지 + 필터 칩 + 테이블은 건드리지 않음.

---

## 1. 문제점 (현재)

현재 상단은 4개 블록이 나열되어 있음:
- "비중 TOP 5" → 숫자만 나열, 비중 차이가 시각적으로 안 드러남
- "주요 변동" → +0.1%p 같은 미미한 변동도 같은 크기로 표시
- "신규 편입" → 없으면 "없음" 텍스트만 → 공간 낭비
- "편출" → 없으면 "없음" 텍스트만 → 공간 낭비

**핵심 문제**: EWY가 뭔지 모르는 사람은 이 페이지가 왜 중요한지 알 수 없음.

---

## 2. 변경 개요

| 기존 | 변경 |
|------|------|
| "비중 TOP 5" 텍스트 리스트 | **비중 바(bar) 차트** — CSS width%로 시각화 |
| "주요 변동" 별도 블록 | **TOP 5 바 안에 통합** (변동 있으면 뱃지 표시) |
| "신규 편입" 항상 표시 | **있을 때만 표시** (빨간 강조) |
| "편출" 항상 표시 | **있을 때만 표시** (파란 강조) |
| 설명 없음 | **EWY 한 줄 설명 추가** |
| 핵심 지표 없음 | **3칸 지표 카드** (보유종목, IT집중도, TOP2비중) |

---

## 3. Supabase 데이터

### 테이블: `quant_ewy_holdings`

### 사용 컬럼:

```
top20: JSONB — [{"rank","code","name","weight","weight_prev","weight_change","sector","signal"}, ...]
changes: JSONB — [{"code","name","weight","weight_change","direction","magnitude"}, ...]
new_entries: JSONB — [{"code","name","weight","sector","impact"}, ...]
removed: JSONB — [{"code","name","weight","sector","impact"}, ...]
total_stocks: integer — 80
```

### 섹터별 비중은 top20에서 계산:

```typescript
// top20 + 나머지 holdings에서 sector별 합산
// 또는 monthly_perf.stocks 에서 sector별 weight 합산
const sectorWeights: Record<string, number> = {};
stocks.forEach(s => {
  sectorWeights[s.sector] = (sectorWeights[s.sector] || 0) + s.weight;
});
```

---

## 4. 섹션 구조 (위 → 아래)

```
┌─────────────────────────────────────────────────────────┐
│  EWY 펀드 보유종목                        2026-05-06     │
│  MSCI Korea ETF — 외인 패시브 자금 흐름의 창               │
├─────────────────────────────────────────────────────────┤
│  [A] 핵심 지표 카드 3칸                                   │
├─────────────────────────────────────────────────────────┤
│  [B] 비중 TOP 5 바 차트                                   │
├─────────────────────────────────────────────────────────┤
│  [C] 변동 알림 (신규 편입/편출 — 있을 때만)                  │
├─────────────────────────────────────────────────────────┤
│  [기존] 월별 수익률 비교 (배지 + 필터 칩 + 테이블)           │
└─────────────────────────────────────────────────────────┘
```

---

## 5. [A] 핵심 지표 카드 — 가로 3칸

### 레이아웃: `grid grid-cols-3 gap-3 mb-6`

### 스타일: 섹터발화 월별 요약 배지와 동일 (SectorFireView.tsx 참조)

```tsx
<div className="grid grid-cols-3 gap-3 mb-6">

  {/* 카드 1: 보유 종목 수 */}
  <div className="rounded-xl p-4 border bg-[#F5F4F0]/60 border-[#e5e7ef]/40">
    <div className="text-[13px] text-[#9ca3b8] mb-1">보유 종목</div>
    <div className="text-[28px] font-extrabold text-[#1A1A2E] tabular-nums">
      {totalStocks}<span className="text-[16px] font-normal text-[#9ca3b8] ml-1">개</span>
    </div>
  </div>

  {/* 카드 2: IT 집중도 */}
  <div className="rounded-xl p-4 border bg-[#F5F4F0]/60 border-[#e5e7ef]/40">
    <div className="text-[13px] text-[#9ca3b8] mb-1">IT 집중도</div>
    <div className="text-[28px] font-extrabold text-[#1A1A2E] tabular-nums">
      {itWeight.toFixed(1)}<span className="text-[16px] font-normal text-[#9ca3b8] ml-0.5">%</span>
    </div>
    <div className="text-[12px] text-[#9ca3b8] mt-0.5">
      {itCount}종목
    </div>
  </div>

  {/* 카드 3: TOP 2 집중도 */}
  <div className="rounded-xl p-4 border bg-[#F5F4F0]/60 border-[#e5e7ef]/40">
    <div className="text-[13px] text-[#9ca3b8] mb-1">TOP 2 비중</div>
    <div className="text-[28px] font-extrabold tabular-nums"
         style={{ color: top2Weight >= 40 ? '#DC2626' : '#1A1A2E' }}>
      {top2Weight.toFixed(1)}<span className="text-[16px] font-normal text-[#9ca3b8] ml-0.5">%</span>
    </div>
    <div className="text-[12px] text-[#9ca3b8] mt-0.5">
      {top20[0]?.name} + {top20[1]?.name}
    </div>
  </div>

</div>
```

### 데이터 계산:

```typescript
const totalStocks = data.total_stocks;  // 80

// IT 집중도: monthly_perf.stocks 에서 sector === 'Information Technology'
const itStocks = stocks.filter(s => s.sector === 'Information Technology');
const itWeight = itStocks.reduce((sum, s) => sum + s.weight, 0);
const itCount = itStocks.length;

// TOP 2 비중
const top2Weight = (top20[0]?.weight || 0) + (top20[1]?.weight || 0);
```

---

## 6. [B] 비중 TOP 5 바 차트

### 제목: 없음 (카드와 바 사이에 제목 불필요 — 바 자체가 자명)

### 레이아웃: `flex flex-col gap-2 mb-6`

```tsx
<div className="flex flex-col gap-2 mb-6">
  {top20.slice(0, 5).map((stock, i) => {
    const maxWeight = top20[0]?.weight || 1;   // 1위 비중 = 100% 기준
    const barPct = (stock.weight / maxWeight) * 100;
    const hasChange = Math.abs(stock.weight_change || 0) >= 0.05;

    return (
      <div key={stock.code} className="flex items-center gap-3">

        {/* 순위 */}
        <span className="text-[14px] font-bold text-[#9ca3b8] w-6 text-right tabular-nums">
          {i + 1}
        </span>

        {/* 종목명 + 코드 */}
        <div className="w-[130px] flex-shrink-0">
          <span className="font-bold text-[#1A1A2E] text-[14px]">{stock.name}</span>
          <span className="text-[11px] text-[#9ca3b8] ml-1">{stock.code}</span>
        </div>

        {/* 바 차트 영역 */}
        <div className="flex-1 flex items-center gap-2">
          {/* 바 배경 */}
          <div className="flex-1 h-[28px] bg-[#F5F4F0] rounded-lg overflow-hidden relative">
            {/* 채워진 바 */}
            <div
              className="h-full rounded-lg transition-all duration-500"
              style={{
                width: `${barPct}%`,
                background: i === 0
                  ? 'linear-gradient(90deg, #FF6B35, #FF4444)'   // 1위: 주황→빨강
                  : i === 1
                  ? 'linear-gradient(90deg, #FF8C00, #FF6B35)'   // 2위: 진주황→주황
                  : 'linear-gradient(90deg, #F59E0B, #FF8C00)',  // 3~5위: 노랑→주황
                opacity: 1 - i * 0.1,  // 순위별 약간 연하게
              }}
            />
            {/* 바 안의 비중 텍스트 */}
            <span className="absolute inset-0 flex items-center px-3 text-[13px] font-bold text-white tabular-nums"
                  style={{ textShadow: '0 1px 2px rgba(0,0,0,0.3)' }}>
              {stock.weight.toFixed(2)}%
            </span>
          </div>
        </div>

        {/* 비중 변동 뱃지 (변동 있을 때만) */}
        {hasChange ? (
          <span
            className="text-[12px] font-semibold tabular-nums px-1.5 py-0.5 rounded"
            style={{
              color: stock.weight_change > 0 ? '#DC2626' : '#2563EB',
              background: stock.weight_change > 0 ? 'rgba(220,38,38,0.08)' : 'rgba(37,99,235,0.08)',
            }}
          >
            {stock.weight_change > 0 ? '+' : ''}{stock.weight_change.toFixed(2)}
          </span>
        ) : (
          <span className="w-[50px]" />  // 정렬용 빈 공간
        )}

      </div>
    );
  })}

  {/* 나머지 종목 요약 */}
  <div className="flex items-center gap-3 mt-1">
    <span className="w-6" />
    <span className="text-[13px] text-[#9ca3b8]">
      나머지 {totalStocks - 5}종목 = {remainingWeight.toFixed(1)}%
    </span>
  </div>
</div>
```

### 데이터 계산:

```typescript
const remainingWeight = 100 - top20.slice(0, 5).reduce((s, h) => s + h.weight, 0);
// 100 - 53.94 = 46.06
```

### 바 색상 규칙:

| 순위 | 그라데이션 | 의미 |
|------|----------|------|
| 1위 | `#FF6B35 → #FF4444` (주황→빨강) | 최대 비중 강조 |
| 2위 | `#FF8C00 → #FF6B35` (진주황→주황) | 두 번째 강조 |
| 3~5위 | `#F59E0B → #FF8C00` (노랑→주황) | 일반 |

### 바 너비 계산:

- 1위 비중을 100%로 놓고 나머지는 비례
- 예: SK하이닉스 24.25% = 100%, 삼성전자 22.38% = 92.3%, SK스퀘어 3.11% = 12.8%
- 이렇게 하면 TOP2의 압도적 비중이 시각적으로 바로 드러남

---

## 7. [C] 변동 알림 — 편입/편출 있을 때만 표시

### 규칙:
- `new_entries` 배열이 비어있으면 → **이 섹션 전체를 렌더링하지 않음**
- `removed` 배열이 비어있으면 → **이 섹션 전체를 렌더링하지 않음**
- 둘 다 비어있으면 → **이 영역 자체가 없음** (공간 절약)
- 하나라도 있으면 → **눈에 띄는 알림 박스로 표시**

### 주요 변동 (changes):
- `magnitude === "LARGE"` (0.3%p 이상) 인 변동만 표시
- MEDIUM 이하는 TOP 5 바의 변동 뱃지로 충분

```tsx
{/* 신규 편입 — 있을 때만 */}
{newEntries.length > 0 && (
  <div className="mb-4 p-4 rounded-xl border-2 border-[#DC2626]/30 bg-[#DC2626]/5">
    <div className="flex items-center gap-2 mb-2">
      <span className="text-[14px] font-bold text-[#DC2626]">신규 편입</span>
      <span className="text-[12px] text-[#DC2626]/70">패시브 강제매수 예상</span>
    </div>
    <div className="flex flex-wrap gap-2">
      {newEntries.map(e => (
        <span key={e.code}
              className="px-3 py-1.5 rounded-lg bg-[#DC2626]/10 text-[#DC2626] text-[13px] font-semibold">
          {e.name} {e.weight.toFixed(2)}%
        </span>
      ))}
    </div>
  </div>
)}

{/* 편출 — 있을 때만 */}
{removed.length > 0 && (
  <div className="mb-4 p-4 rounded-xl border-2 border-[#2563EB]/30 bg-[#2563EB]/5">
    <div className="flex items-center gap-2 mb-2">
      <span className="text-[14px] font-bold text-[#2563EB]">편출</span>
      <span className="text-[12px] text-[#2563EB]/70">패시브 강제매도 예상</span>
    </div>
    <div className="flex flex-wrap gap-2">
      {removed.map(e => (
        <span key={e.code}
              className="px-3 py-1.5 rounded-lg bg-[#2563EB]/10 text-[#2563EB] text-[13px] font-semibold">
          {e.name} {e.weight.toFixed(2)}%
        </span>
      ))}
    </div>
  </div>
)}

{/* 대형 변동 — magnitude LARGE만 (있을 때만) */}
{largeChanges.length > 0 && (
  <div className="mb-4 p-4 rounded-xl border border-[#F59E0B]/30 bg-[#F59E0B]/5">
    <div className="text-[14px] font-bold text-[#F59E0B] mb-2">비중 급변</div>
    <div className="flex flex-wrap gap-2">
      {largeChanges.map(c => (
        <span key={c.code}
              className="px-3 py-1.5 rounded-lg text-[13px] font-semibold"
              style={{
                background: c.direction === 'UP' ? 'rgba(220,38,38,0.1)' : 'rgba(37,99,235,0.1)',
                color: c.direction === 'UP' ? '#DC2626' : '#2563EB',
              }}>
          {c.direction === 'UP' ? '▲' : '▼'} {c.name} {c.weight_change > 0 ? '+' : ''}{c.weight_change.toFixed(2)}%p
        </span>
      ))}
    </div>
  </div>
)}
```

### 데이터:

```typescript
const largeChanges = (data.changes || []).filter(c => c.magnitude === 'LARGE');
const newEntries = data.new_entries || [];
const removed = data.removed || [];
```

---

## 8. 삭제하는 것

| 기존 요소 | 처리 |
|----------|------|
| "비중 TOP 5" 텍스트 리스트 | **삭제** → [B] 바 차트로 대체 |
| "주요 변동" 블록 | **삭제** → TOP 5 바의 변동 뱃지 + [C] 대형 변동 알림으로 대체 |
| "신규 편입" 항상 표시 | **삭제** → [C] 있을 때만 알림 |
| "편출" 항상 표시 | **삭제** → [C] 있을 때만 알림 |

---

## 9. 변경하지 않는 것

- 월별 요약 배지 3칸 그리드 (이미 완성)
- 섹터 필터 칩 (이미 완성)
- 종목 테이블 (이미 완성)
- 접기/펼치기 (이미 완성)

---

## 10. EWY 한 줄 설명

페이지 상단 제목 아래에 다음 설명 추가:

```tsx
<div className="mb-6">
  <h3 className="text-[20px] font-bold text-[#1A1A2E]">EWY 펀드 보유종목</h3>
  <p className="text-[14px] text-[#9ca3b8] mt-1">
    MSCI Korea ETF — 외인 패시브 자금이 어디로 흐르는지 보는 창
  </p>
</div>
```

---

## 11. 스타일 매핑 요약

| 요소 | 값 | 출처 |
|------|---|------|
| 카드 배경 | `bg-[#F5F4F0]/60` | SectorFireView.tsx 헤더 |
| 카드 테두리 | `border-[#e5e7ef]/40` | SectorFireView.tsx 행 구분선 |
| 라벨 색상 | `text-[#9ca3b8]` | SectorFireView.tsx 보조 텍스트 |
| 수치 색상 | `text-[#1A1A2E]` | SectorFireView.tsx 메인 텍스트 |
| 수치 크기 | `text-[28px] font-extrabold` | 월별 배지 가중평균 스타일 |
| 양수 색상 | `#DC2626` | 통일 |
| 음수 색상 | `#2563EB` | 통일 |
| 경고 색상 | `#F59E0B` | 통일 |
| 바 높이 | `h-[28px]` | 터치/클릭 영역 확보 |
| 바 배경 | `bg-[#F5F4F0]` | 카드 배경과 통일 |
| 라운딩 | `rounded-xl` (카드), `rounded-lg` (바/뱃지) | 통일 |

---

## 12. 반응형

- 모바일 (< 640px): 카드 3칸 → `grid-cols-1` 세로 나열
- 바 차트: 종목명 너비 `w-[100px]`로 축소
- 그 외: 기존 월별 테이블과 동일 반응형 룰 적용

---

## 13. 검증 체크리스트

- [ ] 핵심 지표 카드 3칸 표시 (보유종목 80, IT집중도, TOP2 비중)
- [ ] TOP 5 바 차트: 1위 = 100% 너비 기준, 비중 텍스트가 바 안에
- [ ] 바 색상 그라데이션 (1위 빨강, 2위 주황, 3~5위 노랑)
- [ ] 비중 변동 뱃지: 변동 있을 때만 (0.05%p 이상)
- [ ] "나머지 75종목 = 46.1%" 텍스트 표시
- [ ] 신규 편입: 있을 때만 빨간 알림 박스
- [ ] 편출: 있을 때만 파란 알림 박스
- [ ] 편입/편출 모두 없을 때: 알림 영역 자체 없음
- [ ] 기존 4개 블록(비중 TOP 5, 주요 변동, 신규 편입, 편출) 삭제
- [ ] EWY 한 줄 설명 표시
- [ ] 모바일에서 카드 세로 나열
- [ ] 월별 수익률 비교 섹션은 변경 없음
