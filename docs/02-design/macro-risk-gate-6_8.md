# Macro Risk Gate — FLOWX 보조 veto 설계 (2026-06-08)

작성: 2026-06-08 · 상태: **설계만(코드 0, 매매 로직 0, freeze 유지)** · 구현/배선 = 사장님 승인 별건

## 0. 한 줄 원칙

> **미국 10년물은 종목 선정 기준이 아니다. LIVE 진입을 막거나 늦추는 상위 리스크 스위치다.**
> US10Y 단독 veto **금지**. MacroRiskScore의 구성요소 중 하나로만 쓴다.

## 1. 배경 — 기존 검증과의 정합성

6/3 거시 적대검증 결론(메모리)과 **완전 일치**:
- "거시 = 시나리오 지도(맥락) / C60 = 핸들(타이밍). 거시 hard gate 승격 0개."
- "장단기역전 = 상시점등 노이즈(base 0.46), VIX = 거짓경보多(prec 0.29)."
- → **primary hard gate = C60 단독 유지. 거시는 보조/맥락.**

사장님 6/8 제안("US10Y 단독 veto 금지, MacroRisk는 보조 게이트")은 이 검증된 결론과 같은 방향. 흥분이 아니라 데이터로 뒷받침된 직관.

## 2. 인과 사슬 (왜 보는가)

미국채 금리 급등 → 달러 강세/원화 약세 → 외국인 환차손 리스크 → 한국 대형주·반도체 매도 → 개인이 물량 받음 → **D0 종가추종이 갭다운에 직격**.

→ 6/8 실관측과 일치: 반도체 급락일 D0 전멸, D1(시가)이 갭다운 회피로 +6~8%p 방어.

## 3. MacroRiskScore 구성요소 (변화율 중심, 절대값 아님)

| 항목 | 위험 신호 | 데이터 소스 |
|---|---|---|
| US10Y | 1일 +8bp 또는 5일 +20bp | macro_four_signal_daily.csv (us10y) ✅ |
| US30Y | 1일 +8bp 또는 5일 +20bp | **없음 — 추가 수집 필요** ❌ |
| USDKRW | 1일 +1% 또는 20일 신고가 | macro_four_signal_daily.csv (krw) ✅ |
| 외국인 | 전기전자/반도체 2일 연속 순매도 | scan_nationality / sector_investor_flow ✅ |
| KOSPI | 갭하락 + 외인 매도 동시 | kospi_index.csv (⚠️가공본 가능성) |

신규 관측 항목명: `us10y_change_bp_1d`, `us10y_change_bp_5d`, `us30y_change_bp_1d`, `usdkrw_change_pct_1d`, `usdkrw_20d_high`, `foreign_electronics_net`, `semi_foreign_5d`, `macro_risk_score`.

## 4. 게이트 단계 (복합 조건, 절대 단독 금지)

| MacroRiskScore | 조치 |
|---|---|
| ≤ 1 | 기존 LIVE 조건 그대로 적용 |
| ≥ 2 (위험신호 2개+) | 신규 매수 축소 / **D0 종가 진입 금지, D1 검증 필요** |
| ≥ 3 (위험신호 3개+) | **freeze 유지 / D0 금지 / D1만 관찰 / LIVE 전환 보류** |

예시:
- US10Y 상승 + 환율 안정 + 외국인 매수 유지 → 그냥 금리 부담, 매도 근거 부족 (score 낮음)
- US10Y 상승 + 환율 급등 + 외국인 전기전자 매도 → 위험, 신규 진입 축소 (score≥2)
- + 지수 갭하락까지 → **Macro Risk Gate ON, D0 금지/LIVE 보류** (score≥3)

## 5. LIVE 7조건과의 연결 (하드룰 아닌 보조 veto)

```
LIVE 후보라도 MacroRisk >= 3 이면 → 보류
MacroRisk >= 2 이면 → D0 금지, D1 검증 필요
MacroRisk <= 1 이면 → 기존 LIVE 7조건 적용
```

→ LIVE 7조건에 8번째 하드룰로 박지 않는다. **상위 veto 레이어**로 얹는다.

## 6. 반도체/AI 고밸류 페널티 (관측 레이어)

금리 급등일엔 미래 기대값이 할인됨 → 성장주 직격.
```
if US10Y 급등 and USDKRW 급등:
    반도체/AI/고PER/고PBR 종목에 risk_penalty (점수 차감)
```
삼성전자·SK하이닉스·유리기판·AI 장비주는 외국인 수급과 함께 본다.

## 7. 적대적 갭 / 선결 과제

1. **US30Y 데이터 없음** → fetch_ecos_macro/macro_adapter에 추가 수집 필요
2. **macro_four_signal_daily.csv 5/19 stale** → 갱신 파이프라인 점검 (BAT-D 연동 확인)
3. **6/3 검증서 거시 시그널 노이즈·거짓경보 많음** → 변화율 기준 + 복합 2개/3개가 그 보완책 (절대값 단독의 거짓경보 제거)
4. **kospi_k 가공본 가능성** → 실제 코스피 지수 확인 후 사용

## 8. 도입 단계 (freeze 양립)

1. **관측 레이어 먼저**: SHOW ME 리포트에 MacroRiskScore 패널 추가 (매매 영향 0, 6/7 가격축 레이어처럼)
2. 누적 관측으로 임계값 검증 (변화율 bp/% 기준이 실제로 falling knife를 예고하는가)
3. **검증 후에만** LIVE 보조 veto로 배선 (사장님 승인 별건)
4. **절대 hard gate로 승격 금지** — C60 단독 hard gate 원칙 불변

## 9. 금지선

- US10Y/US30Y 단독 veto 금지 (절대값 "5%면 매도" 류 금지)
- 매매 자동화 로직 변경 0 (현 freeze 유지)
- 거시를 primary hard gate로 승격 0
- 구현/갱신/배선은 사장님 승인 후
