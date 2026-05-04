# 웹봇 지시서: EWY 보유종목 — 섹터발화 picks 테이블과 동일 스타일 적용

> **발행**: 퀀트봇 (2026-05-05)
> **대상**: 웹봇 (FLOWX /quant 퀀트시스템 탭 → EWY 카드)
> **우선순위**: P0 (즉시 반영)
> **핵심 원칙**: SectorFireView.tsx의 AccordionPicksTable **코드를 그대로 복사**해서 EWY에 적용

---

## 1. 문제점

현재 EWY 카드의 월별 수익률 테이블이 **섹터발화 picks 테이블과 스타일이 다름**:
- 글자가 너무 작아서 안 보임
- 색상, 굵기, 행 간격이 섹터발화와 다름
- 월별 요약 배지가 세로 나열 (가로 배치 필요)
- 섹터 필터 탭 없음

---

## 2. 수정 원칙

> **SectorFireView.tsx (614줄)의 스타일을 100% 복사**
>
> - 같은 색상 코드 (#DC2626, #2563EB, #1A1A2E, #F5F4F0, #9ca3b8)
> - 같은 CSS 클래스 (py-2.5 px-3, tabular-nums, font-bold)
> - 같은 행 높이, hover 효과
> - 같은 배지 스타일

---

## 3. Supabase 데이터

### 테이블: `quant_ewy_holdings`

### 컬럼: `monthly_perf` (JSONB)

```json
{
  "months": [
    {"key": "2026-03", "label": "3월"},
    {"key": "2026-04", "label": "4월"},
    {"key": "2026-05", "label": "5월(MTD)"}
  ],
  "stocks": [
    {
      "rank": 1, "code": "000660", "name": "SK하이닉스",
      "weight": 22.78, "weight_change": 0.63,
      "close": 1447000, "sector": "Information Technology",
      "returns": {"2026-03": -14.06, "2026-04": 44.01, "2026-05": 2.15}
    }
  ],
  "summary": {
    "2026-03": {"avg": -11.59, "wavg": -13.02, "up": 7, "dn": 72, "total": 80},
    "2026-04": {"avg": 14.70, "wavg": 25.13, "up": 64, "dn": 16, "total": 80},
    "2026-05": {"avg": 2.15, "wavg": 1.83, "up": 45, "dn": 35, "total": 80}
  }
}
```

---

## 4. 섹션 A: 월별 요약 배지 — 가로 3칸 그리드

### 레이아웃: `grid grid-cols-3 gap-3 mb-6`

```tsx
{/* ── 월별 요약 배지 ── */}
<div className="grid grid-cols-3 gap-3 mb-6">
  {months.map(m => {
    const s = summary[m.key];
    const wavg = s?.wavg ?? 0;
    return (
      <div
        key={m.key}
        className="rounded-xl p-4 border"
        style={{
          background: wavg >= 0 ? 'rgba(220,38,38,0.06)' : 'rgba(37,99,235,0.06)',
          borderColor: wavg >= 0 ? 'rgba(220,38,38,0.2)' : 'rgba(37,99,235,0.2)',
        }}
      >
        {/* 월 라벨 — 16px 굵게 */}
        <div className="text-[16px] font-bold text-[#1A1A2E] mb-1">{m.label}</div>

        {/* 가중평균 — 24px 매우 굵게 */}
        <div
          className="text-[24px] font-extrabold tabular-nums"
          style={{ color: wavg >= 0 ? '#DC2626' : '#2563EB' }}
        >
          {wavg >= 0 ? '+' : ''}{wavg.toFixed(2)}%
        </div>

        {/* 단순평균 — 14px */}
        <div
          className="text-[14px] tabular-nums mt-0.5"
          style={{ color: (s?.avg ?? 0) >= 0 ? '#DC2626' : '#2563EB', opacity: 0.7 }}
        >
          단순 {(s?.avg ?? 0) >= 0 ? '+' : ''}{(s?.avg ?? 0).toFixed(2)}%
        </div>

        {/* 상승/하락 */}
        <div className="text-[14px] mt-2 flex gap-3">
          <span style={{ color: '#DC2626' }}>상승 {s?.up ?? 0}</span>
          <span style={{ color: '#2563EB' }}>하락 {s?.dn ?? 0}</span>
        </div>
      </div>
    );
  })}
</div>
```

---

## 5. 섹션 B: 필터 칩 — 섹터별

### SectorFireView.tsx 필터 칩 코드를 그대로 복사

```tsx
// 섹터 한글 매핑
const SECTOR_KR: Record<string, string> = {
  'Information Technology': 'IT',
  'Financials': '금융',
  'Industrials': '산업재',
  'Consumer Discretionary': '소비재',
  'Communication Services': '통신',
  'Materials': '소재',
  'Health Care': '헬스케어',
  'Energy': '에너지',
  'Consumer Staples': '필수소비재',
  'Utilities': '유틸리티',
  'Real Estate': '부동산',
};

// 섹터별 종목 수 카운트
const sectorCounts = useMemo(() => {
  const counts: Record<string, number> = {};
  stocks.forEach(s => {
    const kr = SECTOR_KR[s.sector] || s.sector;
    counts[kr] = (counts[kr] || 0) + 1;
  });
  return counts;
}, [stocks]);

const [sectorFilter, setSectorFilter] = useState('전체');

// 필터링된 종목
const filteredStocks = useMemo(() => {
  if (sectorFilter === '전체') return stocks;
  return stocks.filter(s => (SECTOR_KR[s.sector] || s.sector) === sectorFilter);
}, [stocks, sectorFilter]);
```

### 필터 칩 UI — SectorFireView.tsx 559~577줄 스타일 그대로

```tsx
<div className="flex gap-1.5 mb-4 flex-wrap">
  {/* 전체 탭 */}
  <button
    onClick={() => setSectorFilter('전체')}
    className={`py-1.5 px-3.5 rounded-lg text-[14px] font-semibold transition-all ${
      sectorFilter === '전체'
        ? 'bg-[#00FF88] text-[#1A1A2E] shadow-sm'
        : 'bg-[#F5F4F0] text-[#9CA3AF] hover:text-[#1A1A2E]'
    }`}
  >
    전체 {stocks.length}
  </button>

  {/* 섹터별 탭 */}
  {Object.entries(sectorCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([sector, count]) => (
      <button
        key={sector}
        onClick={() => setSectorFilter(sector)}
        className={`py-1.5 px-3.5 rounded-lg text-[14px] font-semibold transition-all ${
          sectorFilter === sector
            ? 'bg-[#00FF88] text-[#1A1A2E] shadow-sm'
            : 'bg-[#F5F4F0] text-[#9CA3AF] hover:text-[#1A1A2E]'
        }`}
      >
        {sector} {count}
      </button>
    ))}
</div>
```

---

## 6. 섹션 C: 종목 테이블 — SectorFireView.tsx 273~332줄 스타일 그대로

### 헬퍼 함수 (SectorFireView.tsx에서 복사)

```tsx
// 수익률 색상 — retClr() 함수 그대로
function retClr(v: number | null): string {
  if (v == null || v === 0) return '#6B7280';
  if (v > 0) return '#DC2626';   // 빨강
  return '#2563EB';              // 파랑
}

// 수익률 굵기 (EWY 추가 — 월간 수익률이므로 임계값 확대)
function retWeight(v: number | null): string {
  if (v == null) return 'normal';
  const abs = Math.abs(v);
  if (abs >= 30) return '800';   // extrabold
  if (abs >= 15) return '700';   // bold
  if (abs >= 5) return '600';    // semibold
  return 'normal';
}

// 가격 포맷 — priceFmt() 그대로
function priceFmt(v: number): string {
  return v.toLocaleString();
}

// 수익률 포맷
function retFmt(v: number | null): string {
  if (v == null) return 'N/A';
  if (v === 0) return '0.00%';
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
}

// 비중변동 색상
function wcClr(v: number | null): string {
  if (v == null || v === 0) return '#6B7280';
  if (v > 0) return '#DC2626';
  return '#2563EB';
}
```

### 테이블 구조 — SectorFireView.tsx thead/tbody 그대로 복사

```tsx
<div className="overflow-x-auto">
  <table className="w-full" style={{ minWidth: '700px' }}>

    {/* ── 헤더 — SectorFireView.tsx 273~285줄과 동일 ── */}
    <thead>
      <tr className="bg-[#F5F4F0]/60">
        <th className="text-center py-2.5 px-3 font-bold text-[#1A1A2E] w-10">#</th>
        <th className="text-left py-2.5 px-3 font-bold text-[#1A1A2E]">종목</th>
        <th className="text-right py-2.5 px-3 font-bold text-[#1A1A2E]">비중</th>
        <th className="text-right py-2.5 px-3 font-bold text-[#1A1A2E]">종가</th>
        <th className="text-right py-2.5 px-3 font-bold text-[#1A1A2E]">변동</th>
        {/* 월별 헤더 — 동적 생성 */}
        {months.map(m => (
          <th key={m.key} className="text-right py-2.5 px-3 font-bold text-[#1A1A2E]">
            {m.label}
          </th>
        ))}
      </tr>
    </thead>

    {/* ── 바디 — SectorFireView.tsx 289~327줄과 동일 패턴 ── */}
    <tbody>
      {displayStocks.map(stock => (
        <tr
          key={stock.code}
          className="border-t border-[#e5e7ef]/40 hover:bg-[#fafbfc]"
        >
          {/* # 순위 */}
          <td className="py-2.5 px-3 text-center text-[#9ca3b8]">
            {stock.rank}
          </td>

          {/* 종목명 + 코드 — SectorFireView와 동일 */}
          <td className="py-2.5 px-3">
            <span className="font-bold text-[#1A1A2E]">{stock.name}</span>
            <span className="text-[12px] text-[#9ca3b8] ml-1.5">{stock.code}</span>
          </td>

          {/* 비중 */}
          <td className="py-2.5 px-3 text-right tabular-nums font-bold text-[#1A1A2E]">
            {stock.weight.toFixed(2)}%
          </td>

          {/* 종가 — priceFmt() */}
          <td className="py-2.5 px-3 text-right tabular-nums text-[#1A1A2E]">
            {priceFmt(stock.close)}
          </td>

          {/* 비중 변동 */}
          <td
            className="py-2.5 px-3 text-right tabular-nums"
            style={{
              color: wcClr(stock.weight_change),
              fontWeight: Math.abs(stock.weight_change || 0) >= 0.3 ? 'bold' : 'normal',
            }}
          >
            {stock.weight_change == null || stock.weight_change === 0
              ? '-'
              : `${stock.weight_change > 0 ? '+' : ''}${stock.weight_change.toFixed(2)}`}
          </td>

          {/* ★ 월별 수익률 — 핵심 컬럼 */}
          {months.map(m => {
            const ret = stock.returns?.[m.key] ?? null;
            return (
              <td
                key={m.key}
                className="py-2.5 px-3 text-right tabular-nums"
                style={{
                  color: retClr(ret),
                  fontWeight: retWeight(ret),
                }}
              >
                {retFmt(ret)}
              </td>
            );
          })}
        </tr>
      ))}
    </tbody>
  </table>
</div>
```

---

## 7. 접기/펼치기

```tsx
const [showAll, setShowAll] = useState(false);
const displayStocks = showAll ? filteredStocks : filteredStocks.slice(0, 20);

{/* 하단 펼치기 버튼 */}
{filteredStocks.length > 20 && (
  <div className="text-center py-3 border-t border-[#e5e7ef]/40">
    <button
      onClick={() => setShowAll(!showAll)}
      className="text-[14px] font-semibold transition-all bg-[#F5F4F0] text-[#9CA3AF] hover:text-[#1A1A2E] py-1.5 px-3.5 rounded-lg"
    >
      {showAll ? '접기' : `전체 ${filteredStocks.length}개 보기`}
    </button>
  </div>
)}
```

---

## 8. 변경 안 하는 것

- 비중 TOP 5 섹션 → 그대로
- 신규 편입 / 편출 → 그대로
- 하단 요약 텍스트 → 그대로
- iShares EWY ETF 링크 → 그대로

---

## 9. 스타일 매핑 요약 (복붙 체크리스트)

| 요소 | SectorFireView.tsx 원본 | EWY 적용 |
|------|------------------------|----------|
| 헤더 배경 | `bg-[#F5F4F0]/60` | **동일** |
| 헤더 글자 | `font-bold text-[#1A1A2E]` | **동일** |
| 셀 패딩 | `py-2.5 px-3` | **동일** |
| 행 구분선 | `border-t border-[#e5e7ef]/40` | **동일** |
| 행 호버 | `hover:bg-[#fafbfc]` | **동일** |
| 종목명 | `font-bold text-[#1A1A2E]` | **동일** |
| 종목 코드 | `text-[12px] text-[#9ca3b8] ml-1.5` | **동일** |
| 숫자 정렬 | `tabular-nums` | **동일** |
| 양수 색상 | `#DC2626` | **동일** |
| 음수 색상 | `#2563EB` | **동일** |
| 중립 색상 | `#6B7280` | **동일** |
| 필터 선택 | `bg-[#00FF88] text-[#1A1A2E]` | **동일** |
| 필터 미선택 | `bg-[#F5F4F0] text-[#9CA3AF]` | **동일** |

---

## 10. null 처리

1. `monthly_perf`가 null → "월별 수익률 데이터 준비 중" 표시
2. `returns[month_key]`가 null → "N/A" `#6B7280` 회색
3. `close`가 0 → "-"
4. `weight_change`가 null 또는 0 → "-"
