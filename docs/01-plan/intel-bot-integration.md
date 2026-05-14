# 정보봇 데이터 통합 활용 — Plan

**작성**: 2026-05-14
**기준**: [정보봇 → 퀀트봇] 2026-05-14_수급_데이터_안내.md (정보봇 저장소 `796116a`)
**전략**: **B. 병행 운영** — 퀀트봇 자체 데이터 유지 + 정보봇 데이터는 백테스트/팩터 연구용

## 1. 배경 (5/14)

정보봇이 [정보봇 → 퀀트봇] 지시서를 통해 시계열 데이터 제공 안내:
- OHLCV CSV 39컬럼 (2,629종목, 1~2년)
- Supabase 7테이블 (수급/기술/밸류에이션/매크로)
- 16:15 갱신 완성 (15:45 KIS + 16:00 단타봇 폴백 + 16:08 pykrx 폴백)

5/14 진단 결과: 퀀트봇 update_daily_data와 일부 중복. 다만 안전 위해 **병행 운영** 결정.

## 2. 현재 상태

### 데이터 접근 (5/14 16:48 구축)
- VPS 심볼릭 링크: `~/quantum-master/data/external/jgis_ohlcv` → `/home/ubuntu/jgis/stock_data_daily`
- 같은 VPS이므로 rsync 불필요, 항상 최신 자동
- 디스크 공간 절약 (사본 X)

### 컬럼 호환성 검증 완료

| 항목 | 퀀트봇 자체 | 정보봇 |
|------|-----------|------|
| 종목 수 | 2,859 | 2,629 (메이저 종목만) |
| 컬럼 수 | **38** | **39** |
| 추가 컬럼 | - | **Vol_Ratio, Regime_Tag** |
| 호환 | (기준) | **퀀트봇 전체 컬럼 + 2개 보너스** |

→ 정보봇 데이터를 그대로 백테스트에 활용 가능. 퀀트봇 형식 완전 호환.

## 3. 단계별 진행 (정보봇 권장 Phase)

### Phase 1: 데이터 접근 인프라 ✅ 완료 (5/14)
- VPS 심볼릭 링크 생성
- 컬럼 호환성 검증
- 종목 일치 검증 (2,629 ⊂ 2,859, 메이저 종목 일치)

### Phase 2: 외국인 매수 강도 팩터 백테스트 (다음 세션)

**가설**: Foreign_Net 5d/20d 누적 매수 → 상승 알파
```python
import pandas as pd
from pathlib import Path

JGIS_DIR = Path('/home/ubuntu/quantum-master/data/external/jgis_ohlcv')

for csv in JGIS_DIR.glob('*.csv'):
    df = pd.read_csv(csv)
    df['fn_5d'] = df['Foreign_Net'].rolling(5).sum()
    df['fn_20d'] = df['Foreign_Net'].rolling(20).sum()
    # 진입: fn_5d > 0 AND RSI < 40
    df['signal'] = ((df['fn_5d'] > 0) & (df['RSI'] < 40)).astype(int)
    # D+5/D+10 수익률 검증
```

**검증 기준**: 1년+ 데이터, WR 55%+, PF 1.5+

### Phase 3: 섹터 로테이션 모델

**소스**: Supabase `sector_investor_flow` (35일 × 40섹터)
```python
sql = """
SELECT sector, SUM(foreign_net_amt) AS fn_30d
FROM sector_investor_flow
WHERE date >= NOW() - INTERVAL '30 days'
GROUP BY sector
ORDER BY fn_30d DESC LIMIT 5
"""
# Top 5 섹터 = 다음 분기 아웃퍼폼 후보
```

**검증**: 분기별 섹터 ETF 매수 → 1년 누적 수익률 비교

### Phase 4: ETF 수급 → 종목 매수 모델

**소스**: Supabase `etf_investor_flow` (5/6~ 누적, 27 ETF)
- 컬럼: `foreign_net_amt`, `institution_net_amt`, `expected_beneficiaries`
- ETF에 외국인 매수 → ETF 구성종목 매수 시그널

**검증**: ETF 매수 D+1 vs 구성종목 D+5 수익률

### Phase 5: 공매도 squeeze 탐지

**소스**:
- 퀀트봇: `scan_short_factor.py` (이미 존재)
- 정보봇: Supabase `supply_short_balance` (잔고비율)

**조합**: 공매도 잔고 ↑ + 가격 모멘텀 ↑ → squeeze 후보

## 4. 권장 데이터 활용 패턴

### 패턴 A: 백테스트 (Phase 2-5)
- VPS 직접 실행 (정보봇 CSV 직접 읽기)
- 의존성: VPS 접근만 필요

### 패턴 B: Supabase 실시간 조회
```python
from supabase import create_client
sb = create_client(SUPABASE_URL, SUPABASE_KEY)  # 퀀트봇 .env에 있음
# stock_technicals 21지표
res = sb.table('stock_technicals').select('*').gte('date', '2024-01-01').execute()
```

### 패턴 C: 단순 컬럼 추가 (퀀트봇 기존 CSV 보강)
- 퀀트봇 `stock_data_daily/`는 그대로 유지
- 정보봇 `Vol_Ratio`, `Regime_Tag`만 별도 dict로 로드해 분석 시 join

## 5. 시간 흐름 안전 운영

```
15:30  KOSPI 장 마감
15:45  정보봇 KIS Async daily_csv (1차, ~1700종목)
16:00  정보봇 단타봇 daily/ 폴백 (+700종목 → 2400)
16:08  정보봇 pykrx 폴백 (남은 미갱신 → 95%+)
─────  16:15 정보봇 OHLCV 완성 ─────
16:30  퀀트봇 BAT-D 안전 시작 (정보봇 데이터 활용 가능)
17:00  Supabase stock_technicals 21지표 완성
17:08  stock_valuations 완성
17:20  AutoRecovery (Supabase 27테이블 무결성)
```

→ 퀀트봇 백테스트 작업은 **17:30 이후** 권장 (정보봇 모든 단계 완성 보장)

## 6. 위험 관리

| 위험 | 대응 |
|------|------|
| 정보봇 다운 시 데이터 stale | 퀀트봇 자체 update_daily_data 유지 → 영향 없음 (B 옵션) |
| 컬럼 변경 (정보봇 측) | 백테스트 스크립트에서 컬럼 존재 체크 + 미존재 시 fallback |
| 종목 차이 (2,629 vs 2,859) | 정보봇 데이터는 메이저만, 보조 분석용으로 사용 |
| Supabase 비용 | 일별 조회 → 캐싱, 빈도 제한 |

## 7. 다음 세션 액션

1. **Phase 2 Foreign_Net 팩터 백테스트 스크립트 작성** (`scripts/backtest/foreign_net_factor.py`)
2. **백테스트 결과 → `memory/backtest_results.md`에 기록**
3. **시그널 통합 검토** (FLOWX 시그널 보완 가능성)

## 8. 의존 작업

- BAT-D 시간 최적화 (5/15 검증 대기)
- 효성 fix 완료 ✅
- 자동매매 PDCA P0 가드레일 (별개 PDCA)

## 9. 측정 지표

- **백테스트 성과**: PF, WR, MDD, Sharpe
- **실전 시그널 정확도**: 정보봇 `stock_picks_verify` 일별 적중률 (17:25 실행)
- **데이터 신선도**: 매일 16:15 정보봇 완성 여부 모니터링

---

**현재 단계**: **Phase 1 완료** (인프라 구축), **다음 → Phase 2 백테스트** (다음 세션 권장)
