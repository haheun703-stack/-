# WICS 슈퍼섹터 릴레이 엔진 — 클로드코드 구현 스펙

## 배경: 왜 이걸 만드는가

2/19 증권 섹터 +70% 폭발 → 2/20 보험 섹터 +12.5% (미래에셋생명 +29.97% 상한가)
우리 시스템은 보험 종목을 **Zone B(진입불가)로 필터링해서 전부 놓침.**

근본 원인: "증권"과 "보험"이 같은 **"금융" 슈퍼섹터**라는 연결고리를 데이터로 안 가지고 있었음.
돈은 개별 섹터가 아니라 **슈퍼섹터 단위로 유입**되고, 섹터 간 릴레이가 일어남.

---

## Phase 1: wics_sector_mapper.py (신규 파일)

### 목적
WICS(Wise Industry Classification Standard) API를 호출하여 전 상장종목의 3층 섹터 매핑 테이블 생성

### 데이터 소스
- **API URL**: `https://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt={YYYYMMDD}&sec_cd={코드}`
- **인증**: 불필요 (공개 API)
- **업데이트 주기**: 매일 장마감 후 1회 (기존 daily_auto_update.bat에 편입)

### WICS 대분류 코드 (10개)
```python
WICS_SECTORS = {
    'G10': '에너지',
    'G15': '소재',
    'G20': '산업재',
    'G25': '경기소비재',
    'G30': '필수소비재',
    'G35': '건강관리',
    'G40': '금융',
    'G45': 'IT',
    'G50': '커뮤니케이션서비스',
    'G55': '유틸리티'
}
```

### API 호출 로직
```python
import requests
import pandas as pd
import time
from datetime import datetime

def fetch_wics_mapping(date=None):
    """WICS API에서 전 종목 섹터 매핑 가져오기"""
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    sector_codes = ['G10','G15','G20','G25','G30','G35','G40','G45','G50','G55']
    all_data = []
    
    for code in sector_codes:
        url = f'https://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt={date}&sec_cd={code}'
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            df = pd.json_normalize(data['list'])
            all_data.append(df)
        except Exception as e:
            print(f"[WICS] {code} 호출 실패: {e}")
        time.sleep(2)  # rate limit 존중
    
    if not all_data:
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    return result
```

### API 응답 필드 (사용할 것만)
```
CMP_CD       : 종목코드 (6자리, ex: '005930')
CMP_KOR      : 종목명 (ex: '삼성전자')
IDX_CD       : 대분류 코드 (ex: 'G45')
IDX_NM_KOR   : 대분류명 (ex: 'WICS IT')
SEC_CD       : 중분류 코드 (= IDX_CD와 동일할 수 있음)
SEC_NM_KOR   : 중분류명 (ex: 'IT', '은행', '증권', '보험')
MKT_VAL      : 시가총액 (백만원)
WGT          : 섹터 내 비중 (%)
```

### 출력 파일
- **경로**: `data/sector_rotation/wics_mapping.csv`
- **컬럼**:

| 컬럼 | 설명 | 예시 |
|------|------|------|
| stock_code | 종목코드 | 005930 |
| stock_name | 종목명 | 삼성전자 |
| super_sector_code | 대분류 코드 | G45 |
| super_sector_name | 대분류명 | IT |
| sector_name | 중분류명 | 반도체와반도체장비 |
| market_cap | 시총(백만원) | 315204519 |
| sector_weight | 섹터 내 비중(%) | 57.11 |

### 슈퍼섹터 → TIGER ETF 매핑 (추가 테이블)

기존 `sector_etf_builder.py`의 21개 ETF와 WICS 중분류를 연결하는 브릿지 테이블도 함께 생성:

```python
# data/sector_rotation/wics_etf_bridge.csv
WICS_ETF_BRIDGE = {
    # WICS 중분류명: TIGER ETF 종목코드
    '은행': '091220',           # TIGER 은행
    '증권': '091230',           # TIGER 증권  
    '보험': '140710',           # TIGER 보험
    '반도체와반도체장비': '091160',  # TIGER 반도체
    # ... 기존 21개 ETF 전부 매핑
}
```

**이 매핑이 없는 WICS 중분류** = ETF가 없음 → 개별종목 직접매매로만 커버 (기존 scan_krx_sector_full.py)

---

## Phase 2: 슈퍼섹터 릴레이 감지 엔진

### 목적
같은 대분류(슈퍼섹터) 내에서 선행 섹터 → 후행 섹터로 자금이 넘어가는 "릴레이"를 자동 감지

### 핵심 로직

#### 2-1. 슈퍼섹터별 합산 모멘텀

기존 `sector_momentum.py`의 21개 ETF 모멘텀 데이터를 WICS 대분류 기준으로 그룹핑:

```python
# 금융(G40) 슈퍼섹터 합산 모멘텀 계산 예시
# 은행 ETF 모멘텀 + 증권 ETF 모멘텀 + 보험 ETF 모멘텀 → 가중평균
def calc_supersector_momentum(sector_momentum_df, wics_mapping):
    """
    sector_momentum_df: 기존 sector_momentum.py 출력
    wics_mapping: wics_mapping.csv에서 ETF↔대분류 연결
    
    출력: 슈퍼섹터별 합산 모멘텀, RSI, 거래대금 합계
    """
    pass
```

#### 2-2. 릴레이 감지 규칙

```python
def detect_relay(supersector_data, sector_data):
    """
    릴레이 감지 조건:
    1. 슈퍼섹터 내 1개 섹터가 모멘텀 Top 3 진입
    2. 해당 섹터 RSI > 70 (과열 시작)
    3. 같은 슈퍼섹터의 다른 섹터 거래대금이 전일 대비 +30% 이상 증가
    
    출력: {
        'supersector': '금융',
        'leader': '증권',          # 선행 섹터
        'leader_rsi': 82,
        'relay_candidates': ['보험', '은행', '다각화된금융'],
        'relay_signals': {
            '보험': {'volume_change': +45%, 'momentum_rank_change': +3},
            '은행': {'volume_change': +12%, 'momentum_rank_change': +1},
        }
    }
    """
    pass
```

#### 2-3. Zone B 오버라이드 규칙

기존 `scan_krx_sector_full.py`의 Zone A/B/C 판정에 릴레이 오버라이드 추가:

```python
def apply_relay_override(zone_result, relay_signal):
    """
    기존: Zone B = 진입 불가 (무조건)
    수정: Zone B + 슈퍼섹터 릴레이 활성 = Zone A로 오버라이드
    
    오버라이드 조건 (모두 충족):
    1. 슈퍼섹터 릴레이 감지됨 (detect_relay에서 True)
    2. 슈퍼섹터 내 선행 섹터가 당일 +5% 이상
    3. 해당 종목의 거래대금이 전일 대비 2배 이상
    
    오버라이드 시:
    - Zone B → Zone A
    - 단, 사이즈는 HALF로 제한 (FULL 불가)
    - 손절 -3% 엄수 (테마머니 규칙 적용)
    """
    pass
```

---

## Phase 3: 모멘텀 가속도 + 선행주 감지

### 3-1. 모멘텀 가속도 (sector_momentum.py 수정)

기존 sector_momentum.py 출력에 다음 컬럼 추가:

```python
# 추가 컬럼
'momentum_rank_prev'    : 전일 모멘텀 순위
'momentum_rank_change'  : 순위 변화 (음수 = 상승)
'volume_change_pct'     : 거래대금 전일 대비 변화율(%)
'acceleration_flag'     : True if 순위 +3칸 이상 상승 AND 거래대금 +30% 이상
```

**구현 방법**: 전일 데이터 파일(`data/sector_rotation/sector_momentum_prev.csv`)을 저장해두고, 오늘 데이터와 비교

### 3-2. 선행주 감지 (sector_zscore.py 수정)

기존 sector_zscore.py 출력에 다음 컬럼 추가:

```python
# 추가 컬럼  
'z5_rank_in_sector'     : 섹터 내 z_5 순위 (1 = 가장 먼저 반등)
'z5_reversal'           : True if z_5 > z_20 전환 (단기가 장기 역전)
'leader_candidate'      : True if z5_rank == 1 AND z5_reversal == True
```

---

## Phase 4: 리포트 통합 + 텔레그램 알림

### sector_daily_report.py 수정

기존 리포트에 다음 섹션 추가:

```
=== 슈퍼섹터 릴레이 감지 ===

🔥 [금융] 릴레이 활성!
  선행: 증권 (모멘텀 1위, RSI 82, +70%)
  → 증권은 과열, 추격 금지
  
  릴레이 후보:
  1. 보험 — 거래대금 +45%↑, 순위 3칸↑ ⚡
  2. 은행 — 거래대금 +12%↑, 순위 1칸↑
  3. 다각화금융 — 변화 미미

  보험 내 래깅 종목 (Zone B→A 오버라이드):
  - 동양생명: HALF 사이즈, 손절 -3%
  - 미래에셋생명: HALF 사이즈, 손절 -3%

=== 모멘텀 가속도 TOP 5 ===
  1. 보험: 순위 8→5 (+3↑), 거래대금 +45%
  2. 조선: 순위 6→4 (+2↑), 거래대금 +22%
  ...

=== 선행주 감지 ===
  [조선] HD현대마린엔진 — z_5 최초 반등, 대장 후보
  [2차전지] 에코프로비엠 — z_5 > z_20 전환
```

### 텔레그램 알림 (기존 채널 활용)

릴레이 감지 시 별도 메시지:
```
🔥 슈퍼섹터 릴레이 감지: 금융

선행: 증권 → RSI 82 과열
릴레이: 보험 거래대금 +45%↑

오버라이드: 동양생명 Zone B→A (HALF)
```

---

## 파일 구조

```
scripts/
├── wics_sector_mapper.py          ← [신규] WICS API 매핑
├── sector_etf_builder.py          ← [수정 없음]
├── sector_momentum.py             ← [수정] 가속도 컬럼 추가
├── sector_zscore.py               ← [수정] 선행주 감지 컬럼 추가
├── sector_investor_flow.py        ← [수정 없음]
├── sector_relay_engine.py         ← [신규] 릴레이 감지 + Zone B 오버라이드
├── sector_daily_report.py         ← [수정] 릴레이 섹션 추가
├── scan_krx_sector_full.py        ← [수정] 오버라이드 로직 연동

data/sector_rotation/
├── wics_mapping.csv               ← [신규] 전종목 3층 매핑
├── wics_etf_bridge.csv            ← [신규] WICS↔ETF 브릿지
├── sector_momentum_prev.csv       ← [신규] 전일 모멘텀 (가속도 계산용)
├── (기존 파일들 유지)
```

---

## 실행 순서 (daily_auto_update.bat 수정)

```batch
:: === 섹터 순환매 파이프라인 ===
:: Step 0: WICS 매핑 갱신 (하루 1회)
python scripts/wics_sector_mapper.py

:: Step 1: ETF 데이터 수집 (기존)
python scripts/sector_etf_builder.py

:: Step 2: 모멘텀 계산 + 가속도 (수정)
python scripts/sector_momentum.py

:: Step 3: z-score + 선행주 감지 (수정)
python scripts/sector_zscore.py

:: Step 4: 수급 분석 (기존)
python scripts/sector_investor_flow.py

:: Step 5: 릴레이 감지 (신규)
python scripts/sector_relay_engine.py

:: Step 6: KRX 스캔 + 오버라이드 (수정)
python scripts/scan_krx_sector_full.py

:: Step 7: 통합 리포트 (수정)
python scripts/sector_daily_report.py
```

---

## 구현 우선순위

| 순서 | 작업 | 난이도 | 효과 |
|------|------|--------|------|
| **1** | wics_sector_mapper.py | 쉬움 | 3층 매핑 = 모든 후속 작업의 기초 |
| **2** | sector_relay_engine.py | 중간 | 릴레이 감지 = 오늘 같은 실수 방지 |
| **3** | sector_momentum.py 가속도 추가 | 쉬움 | 전일 비교 컬럼 추가만 하면 됨 |
| **4** | scan_krx_sector_full.py 오버라이드 | 중간 | Zone B→A 오버라이드 로직 |
| **5** | sector_daily_report.py 릴레이 섹션 | 쉬움 | 출력 포맷 추가 |
| **6** | sector_zscore.py 선행주 감지 | 쉬움 | z_5 순위/반전 컬럼 추가 |

**1번(wics_sector_mapper.py)부터 시작하자.** 이게 없으면 나머지가 다 안 된다.

---

## 검증 방법

### 역사적 검증 (어제/오늘 데이터로)
```
2/19 장마감 데이터로 엔진 돌렸을 때:
- 증권 RSI > 70 + 모멘텀 1위 → 금융 슈퍼섹터 활성화 ✓
- 보험 거래대금 전일 대비 증가 확인 → 릴레이 후보 ✓
- 동양생명 Zone B → Zone A 오버라이드 → HALF 매수 시그널 ✓

2/20 실제 결과:
- 미래에셋생명 +29.97%, 동양생명 +19.67% → 시그널 적중 확인
```

---

## 주의사항

1. **WICS API rate limit**: 각 호출 사이 2초 sleep 필수
2. **WICS 커버리지**: 시총 상위 500개 종목만 포함 (소형주 미포함) → KRX 업종 폴백 필요
3. **기존 코드 안 건드리기**: sector_etf_builder.py, sector_investor_flow.py 등은 수정 없음
4. **릴레이 오버라이드는 HALF 사이즈만**: 절대 FULL로 오버라이드하지 않음
5. **오버라이드 시에도 손절 -3% 엄수**: 테마머니 규칙 그대로 적용
