# [정보봇 → 퀀트봇·단타봇·웹봇] ETF 시그널 신뢰도 어필 어댑터 v1

**발신**: 정보봇 (조언자/공급자)
**수신**: 퀀트봇 (`quantum-master`) · 단타봇 (`bodyhunter`) · 웹봇 (`flowx-web`)
**발행일**: 2026-05-18 (월)
**커밋**: `289167d`

---

## 0. 한 줄

정보봇이 ETF 수급 시그널(외인+기관 순매수)의 **D+1 적중률을 자동 검증해서 카테고리별 신뢰도 가중치를 공급**합니다. 봇 측은 어댑터 한 번 호출로 가중치를 받아 자체 매매/표시 로직에 적용하면 됩니다.

**정보봇 본질**: 데이터 + 신뢰도 공급. 매매·표시 결정은 봇 자체.

---

## 1. 무엇이 새로 생겼나

### 1-1. 일별 검증 잡 (17:23 평일, AWS 가동 중)
- 신규 잡: `etf_signal_accuracy_verify` 매일 17:23 평일
- 처리: D-1 etf_investor_flow에서 외인+기관 순매수 절대값 ≥5억원 ETF → D 등락률 방향 일치 검증
- 적재: `etf_signal_accuracy` 테이블 (16컬럼)

### 1-2. 봇 어필용 view 3개 (Supabase)
| view | 용도 |
|------|------|
| `etf_signal_accuracy_summary` | 카테고리별 전 기간 누적 적중률 |
| `etf_signal_accuracy_30d` | 카테고리별 최근 30일 롤링 |
| `etf_signal_accuracy_by_sector` | 섹터+카테고리 (표본 ≥3건만) |

### 1-3. 봇 호출용 어댑터
- 정보봇 측 모듈: `src/infrastructure/adapters/etf_signal_reliability.py`
- 봇 측은 **REST API로 view 직접 조회** 또는 **이 어댑터 패턴을 복사**

---

## 2. 현재 적중률 (6 거래일 누적, 2026-05-15 기준)

| 카테고리 | 적중률 | 가중치 (산식 산출) | 활용 권장 |
|---------|--------|-------------------|---------|
| theme | **72.7%** | **1.38** | 강한 가중 (매수 시그널 신뢰도 ↑) |
| global | **62.5%** | **1.21** | 가중 |
| group | 100% (n=2) | 1.00 (표본 보수) | 중립 |
| sector | 45.7% | 0.93 | 약간 감산 |
| direction | 40.6% | 0.84 | 감산 (인덱스/레버리지/인버스) |
| bond_commodity | 28.6% | **0.64** | 강한 감산 (역방향 가까움) |

가중치 산식: 적중률 50% = 1.0, 80%+ = 1.5, 20%- = 0.5, 그 사이 선형 보간.

---

## 3. 봇 측 사용법 — 3가지 방법

### 3-A. 가장 간단 (REST API 직접 호출)

```python
from supabase import create_client

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
r = sb.table("etf_signal_accuracy_summary").select(
    "category,hit_rate_pct,total_signals"
).execute()

weights = {}
for row in r.data:
    if (row.get("total_signals") or 0) < 3:
        continue
    hr = row["hit_rate_pct"]
    # 산식: 50%=1.0, 80%+=1.5, 20%-=0.5
    if hr >= 80: w = 1.5
    elif hr <= 20: w = 0.5
    else: w = 0.5 + (hr - 20) * 1.0 / 60.0
    weights[row["category"]] = w

# 사용
etf_category = "theme"  # 매수 후보 ETF의 카테고리
final_score = raw_signal_score * weights.get(etf_category, 1.0)
```

### 3-B. 정보봇 어댑터 패턴 그대로 복사

`src/infrastructure/adapters/etf_signal_reliability.py` 파일을 봇 프로젝트에 복사. 의존성 없음 (supabase 클라이언트만).

```python
from infrastructure.adapters.etf_signal_reliability import ETFSignalReliability

r = ETFSignalReliability(supa)
weights = r.get_category_weights(recent_30d=True)  # 최근 30일
sector_w = r.get_sector_weight("대형주")
appeal_msg = r.summarize_for_bot()  # 텔레그램 첨부용 한 줄
```

### 3-C. 봇 메시지에 한 줄 어필 첨부

```python
appeal = r.summarize_for_bot()
# "ETF 적중률 — theme 73%, global 62%, sector 46%, direction 41%, bond_commodity 29% (정보봇 누적, ~2026-05-15)"

telegram_msg = f"""
[퀀트봇] 매수 시그널 X

{appeal}

(theme ETF는 정보봇 73% 적중률, 신뢰도 가중 ×1.38 적용)
"""
```

---

## 4. 봇별 권장 활용

### 퀀트봇 (`quantum-master`)
- **매수 후보 점수 보정**: ETF 카테고리 매핑 → 가중치 곱
- **권장**: ETF 외 종목도 같은 섹터 ETF의 카테고리로 분류 → 섹터 가중치 활용
- 예: 삼성전자 매수 → sector="반도체" → `get_sector_weight("반도체")` 활용

### 단타봇 (`bodyhunter`)
- **단발 매매 시그널 보정**: theme/global 가중치 ≥1.2 이면 진입 후보 우선순위 ↑
- **데이트레이딩 회피**: bond_commodity 가중치 0.64 → 해당 카테고리 ETF 단타 후보 제외 권장
- **선택적**: 텔레그램 알림에 정보봇 어필 메시지 첨부

### 웹봇 (`flowx-web`)
- **대시보드 ETF 페이지**: 카테고리별 적중률 카드 노출 ("정보봇 6일 누적 73% 적중")
- **시그널 강도 표시**: 매수 시그널 옆에 신뢰도 배지 (★★★ 73%+ / ★★ 50~70% / ★ 50%-)
- **REST API**: `/rest/v1/etf_signal_accuracy_summary` 직접 호출 (pg 라이브러리 도입 불필요, 기존 supabase-js 그대로)

---

## 5. 일정 + 갱신 주기

- **17:23 평일** 검증 잡 자동 가동 (5/18 첫 회차부터)
- view는 매번 최신 데이터 자동 집계 (별도 갱신 잡 불필요)
- **4주 후 (~2026-06-15)** 정보봇 적중률 보고 + 카테고리별 임계 차등 권고 예정

---

## 6. 회신 요청 (있으면 회신, 없으면 무응답)

각 봇 운영자께:
1. 어댑터 사용 방식 (A/B/C 중 어느 것 채택?) 또는 자체 패턴 사용?
2. 카테고리별 가중치 산식(50%=1.0, 80%+=1.5, 20%-=0.5 선형)이 적절한지, 또는 별도 가중치 곡선 선호?
3. 봇 메시지에 정보봇 어필 한 줄 첨부 의향?
4. 추가로 정보봇이 공급했으면 좋겠는 시그널 신뢰도(예: us_kr_theme_signals, supply_scoring 적중률)?

회신 위치:
- 퀀트봇 → `quantum-master/docs/from-jgis/`
- 단타봇 → `bodyhunter/scalper-agent/docs/from-jgis/`
- 웹봇 → 별도 협의

---

## 부록. 정보봇 본질 (재확인)

정보봇은 **데이터·시그널·신뢰도·근거(evidence)를 공급**합니다.
- 매매 결정 (퀀트봇/단타봇)
- 표시 결정 (웹봇)
- 콘텐츠 발행 (블로그봇)

모두 봇 자체 판단. 정보봇은 "이 정보가 가치 있음"을 어필할 뿐.
