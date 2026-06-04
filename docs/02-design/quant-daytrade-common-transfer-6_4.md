# 퀀트봇·단타봇·공통 적용 전달서 — 주봉/수급/방향 게이트 샘플

작성일: 2026-06-04  
목적: 첨부 대화에서 나온 좋은 요소를 현재 퀀트봇·단타봇·공통 인프라에 섞지 않고, 샘플 관측으로 검증한다.  
상태: 실주문 0 / 봇 OFF / SAJANG 무변경 / scheduler 무연결

## 0. 핵심 결론

MA는 매수 타점을 맞히는 도구가 아니라 방향 필터다.  
가격이 방향을 만들고, 주봉이 허가하고, 수급은 동행 여부를 확인한다.

따라서 이번 샘플은 룰 승격이 아니라 다음 3개를 병행 기록한다.

1. 퀀트봇: 주봉 게이트 + 일봉 눌림/지지 반등 + 수급 동행 확인
2. 단타봇: 강한 섹터/그룹 + RS + 거래량/마감강도 + 이격도 필터
3. 공통: NXT/매크로 차단기 + 시장폭 + 수급 fact layer

## 1. 공통 원칙

### 1.1 신호 역할 분리

| 요소 | 역할 | 승격 상태 |
| --- | --- | --- |
| 일봉 가격/캔들 | 실행 트리거 | 기존 로직 유지 |
| 주봉 20/60선 | 롱 허가/관망/회피 게이트 | 샘플 관측 |
| ADX/DMI | 횡보장 휩쏘 차단 | 샘플 관측 |
| 수급 | 동행 확인 점수 | hard gate 금지 |
| 외국인/기관/연기금 지분율 | 보유 구조 fact | 매매판단 직접 연결 금지 |
| 시장폭 | 공통 장세 건강도 | NXT 보조 관측 |
| 매크로 GAP/HY/VIX 등 | 공통 위험 맥락 | 가격 C60 대체 금지 |

### 1.2 수급 해석 원칙

수급은 확정 신호가 아니다. 누가 같은 방향으로 밀고 있는지 보는 동행 확인이다.

| 주체 | 해석 | 사용처 |
| --- | --- | --- |
| 외국인 | 대형주 방향 엔진. 환율/글로벌 위험선호와 같이 확인 | 대형주/반도체/레버리지 보조 |
| 연기금 | 느린 바닥 지지. 타이밍 신호 아님 | 눌림/바닥권 신뢰도 보조 |
| 금융투자 | 프로그램/차익거래 노이즈 많음 | 단독 방향 신호 금지 |
| 개인 | 극단에서 역방향 참고 | 천장/패닉 보조 |
| 기타법인 | 자사주/특수 이벤트만 의미 | 이벤트 fact |

중요: 수급 동행은 외국인+기관 합계 부호로 판단하지 않는다. 기관합계는 금융투자/투신/보험/연기금/사모/은행/기타금융 등 서로 다른 목적의 주체가 섞인 값이라 세력 족적을 상쇄할 수 있다. 샘플 기간에는 가능한 한 11주체 세부 데이터를 원본 기준으로 기록하고, 합계는 보조 표시만 한다.

데이터 레벨도 분리한다.

| 레벨 | 현재 사용 가능 데이터 | 샘플 처리 |
| --- | --- | --- |
| 시장/regime 레벨 | 11주체 세부 flow_market | 금융투자/연기금/외국인등록/기타외국인/개인 divergence 관측 가능 |
| 종목 레벨 | 현재 4주체 중심(외국인/기관/개인/기타법인) | hard gate 금지, feature/fact로만 기록 |
| 종목별 11주체 | 미연결. 키움 opt10059 등 별도 소스 필요 | 연결 전까지 사용 금지 |

샘플에서 반드시 분리할 항목:

| 항목 | 이유 |
| --- | --- |
| 금융투자 단독 | 프로그램/차익거래 노이즈가 많지만, 당일 세력성 물량의 대부분을 차지할 수 있어 합계에 묻히면 안 됨 |
| 연기금 단독 | 바닥 지지/floor 성격. 기관합계와 분리해야 의미가 살아남 |
| 투신/사모 | 국내 펀드/사모성 수급. 외국인/연기금과 성격 다름 |
| 기타외국인 vs 외국인등록 | 외국인 내부 divergence가 생길 수 있어 합산 금지 |
| 개인 | 극단 흡수/분산매도 판단용 |

강한 조합:

- 정배열 + 일봉 지지 + 외국인/연기금 동반 3일 매수 = 강한 동행
- 신고가 + 외국인 매도 전환 + 개인 순매수 흡수 = 분산매도 경고
- 역배열 + 개인 투매 + 외국인/연기금 매수 시작 = 바닥 후보, 단 반전 캔들 전 진입 금지

## 2. 퀀트봇 전달서

### 2.1 적용 철학

퀀트봇은 약할 때 사는 봇이다. 단, 싸 보여서 사는 것이 아니라 주봉이 허가하고 일봉에서 지지 테스트 후 반등이 확인될 때만 산다.

### 2.2 샘플 기록 필드

| 필드 | 설명 |
| --- | --- |
| ticker / name / date | 종목, 기준일 |
| weekly_gate | LONG_ALLOWED / WATCH / AVOID |
| weekly_close_vs_ma20 | 주봉 종가의 20주선 대비 괴리 |
| weekly_close_vs_ma60 | 주봉 종가의 60주선 대비 괴리 |
| weekly_ma20_slope | 20주선 기울기 |
| weekly_ma60_slope | 60주선 기울기 |
| daily_setup | PULLBACK / SUPPORT_TEST / BOUNCE_CONFIRMED / NONE |
| daily_ma_stack | 5>20>60>120 여부 |
| adx | 추세 강도 |
| di_direction | +DI 우위 / -DI 우위 |
| distance_to_ma20 | 일봉 20선 이격 |
| support_price | 지지 기준가 |
| rr_estimate | 예상 손익비 |
| supply_alignment | FOREIGN_INST / PENSION_SUPPORT / RETAIL_DISTRIBUTION / NONE |
| flow_11_actor_detail | 시장/regime 레벨 11주체 원본 상세. 종목별 값으로 오인 금지 |
| flow_financial_investment | 금융투자 단독 수급 |
| flow_pension | 연기금 단독 수급 |
| flow_foreign_registered | 외국인등록 수급 |
| flow_other_foreign | 기타외국인 수급 |
| flow_institution_sum | 기관합계. 단독 판단 금지, 참고용 |
| stock_flow_4_actor | 종목 레벨 4주체 수급. hard gate 금지 |
| decision_shadow | ENTER_CANDIDATE / WAIT / AVOID |

### 2.3 샘플 판정

| 조건 | 판정 |
| --- | --- |
| weekly_gate=LONG_ALLOWED + daily_setup=BOUNCE_CONFIRMED + rr>=2 | ENTER_CANDIDATE |
| weekly_gate=WATCH + daily_setup=BOUNCE_CONFIRMED | WAIT 또는 half score |
| weekly_gate=AVOID | 일봉 신호가 좋아도 AVOID |
| ADX<20 | 횡보장 의심, 신규 진입 감점 |
| distance_to_ma20 과열 | 눌림 대기 |

### 2.4 금지

- 주봉만 보고 진입 금지
- 수급 단독 진입 금지
- ADX를 검증 전 hard gate로 승격 금지
- 기존 C60/분할매수 룰 변경 금지

## 3. 단타봇 전달서

### 3.1 적용 철학

단타봇은 강할 때 타는 봇이다. 단, 오르니까 타는 것이 아니라 강한 섹터/그룹 안에서 상대강도·거래량·마감강도·이격도 필터를 통과한 종목만 paper로 탄다.

### 3.2 3-Type paper training 연결

| 타입 | 역할 | 이번 샘플 추가 |
| --- | --- | --- |
| A STEADY_EVENT_RIDE | 완만 신고가 + S급 명분 | 주봉 게이트, RS, 수급 동행 기록 |
| B ROTATION_PULLBACK | 강한 그룹 눌림 매수 | 주봉 게이트, ADX, 지지/수급 기록 |
| C ROTATION_RIDE | 강한 그룹 올라타기 | RS, 거래량, 마감강도, 이격도 필터 기록 |

### 3.3 샘플 기록 필드

| 필드 | 설명 |
| --- | --- |
| type | A/B/C |
| ticker / name / date | 종목, 기준일 |
| hot_group_status | HOT / WARMING / RELAY / NONE |
| sector_rank | 섹터/그룹 강도 순위 |
| rs_vs_market | 코스피/코스닥 대비 상대강도 |
| rs_vs_sector | 섹터 내 상대강도 |
| volume_ratio | 거래량 배율 |
| close_strength | 종가 위치/마감강도 |
| candle_quality | 양봉/윗꼬리/전일고가 돌파 |
| distance_to_ma20 | 과열 추격 방지 |
| weekly_gate | LONG_ALLOWED / WATCH / AVOID |
| supply_alignment | 외국인/기관/연기금 동행 |
| flow_11_actor_detail | 시장/regime 레벨 11주체 원본 수급. 종목별 세부로 오인 금지 |
| flow_financial_investment | 금융투자 단독. 세력성/프로그램성 분리 관찰 |
| flow_pension | 연기금 단독. 눌림/바닥 지지 확인용 |
| flow_foreign_registered | 외국인등록 |
| flow_other_foreign | 기타외국인. 외국인등록과 divergence 관찰 |
| flow_institution_sum | 기관합계. 단독 gate 금지 |
| stock_flow_4_actor | 종목 레벨 4주체 수급. feature/fact로만 사용 |
| event_reason | DART_S / NEWS_POSITIVE / SUPPLY / NONE |
| entry_variant | T0_CLOSE / T1_OPEN |
| stop_variant | -3 / -5 / -7 / -8 / -10 / ATR |
| exit_variant | D+1 / D+2 / D+3 / MA10 / STRUCTURE |
| paper_decision | PAPER_ENTER / WAIT / AVOID |

### 3.4 C 타입 올라타기 임시 판정

검증 전 단일값 확정 금지. 아래는 paper 관측용이다.

| 조건 | 해석 |
| --- | --- |
| hot_group_status in HOT/WARMING/RELAY | 그룹 힘 있음 |
| rs_vs_market > 0 | 시장보다 강함 |
| volume_ratio >= 1.5 | 수급/관심 유입 |
| close_strength 양호 | 당일 힘 유지 |
| distance_to_ma20 과열 아님 | 작전/끝물 추격 방지 |
| weekly_gate=AVOID | paper 진입 금지 또는 별도 관찰 |

### 3.5 금지

- NEWS 제목만으로 hard gate 승격 금지
- DART 아닌 뉴스 명분은 forward 기록만
- 수급 단독 진입 금지
- 단타봇을 실주문과 연결 금지
- scheduler 자동배선 금지

## 4. 공통 전달서

### 4.1 공통 feature layer

퀀트봇과 단타봇은 신호 엔진은 다르지만 아래 feature는 공유한다.

| 공통 feature | 공급 주체 | 목적 |
| --- | --- | --- |
| weekly_ma20/60 gate | 퀀트봇 데이터/일봉 기반 | 큰 방향 허가 |
| adx/dmi | 가격 feature | 횡보/추세 구분 |
| market_breadth | NXT/시장폭 | 장세 건강도 |
| ownership fact | 정보봇 stock_ownership | 구조적 보유 확인 |
| macro_gap watch | 정보봇 macro_gap | 위험 맥락 |
| sector/group status | sector_relay | 강한 돈의 위치 |
| event_signal | 정보봇/단타봇 | 명분 기록 |

### 4.2 공통 차단기

| 조건 | 공통 처리 |
| --- | --- |
| 시장 전체 위험 급등 | 신규 paper 진입도 별도 태그 |
| C60/주봉 대세 회피 | 레버/대형주 신규 진입 금지 |
| 휴장/비거래일 | scan/ledger date 보정 |
| 데이터 stale | 신호 생성 금지, stale 기록 |

## 5. 샘플 진행안

### 5.1 기간

1차 샘플: 2026-06-04 ~ 2026-06-12  
목적: 수익 확정이 아니라 어떤 feature가 살아있는지 확인

### 5.2 샘플 대상

| 영역 | 대상 |
| --- | --- |
| 퀀트봇 | 대형주/소부장/섹터발화 후보 중 눌림 발생 종목 |
| 단타봇 A | STEADY + S급 DART/이벤트 후보 |
| 단타봇 B | HOT/WARMING/RELAY 그룹 내 -3/-5/-7 눌림 |
| 단타봇 C | HOT/WARMING/RELAY 그룹 내 +5/+7 돌파/양봉 |
| 공통 | 488080, 삼성 레버, SK 레버, KOSPI/KOSDAQ breadth |

### 5.3 매일 보고 형식

```text
[6/4 paper 샘플]
공통: 시장폭 / 매크로 / 주봉 게이트 이상 여부
퀀트봇: 후보 N건, ENTER_CANDIDATE N건, AVOID 이유 TOP3
단타봇 A/B/C: 각 후보/진입/회피 수, 회피 이유 TOP3
수급: 외국인/연기금 동행 N건, 개인흡수 경고 N건
성과: paper PnL, MFE/MAE, hit/miss, feature별 기여
결론: 오늘 돈이 어디로 돌았는가, 어떤 타입이 유리했는가
```

### 5.4 6/12 1차 판정 기준

| 질문 | 판정 방법 |
| --- | --- |
| 주봉 게이트가 도움 됐나 | AVOID 종목이 실제로 약했는지 |
| ADX가 휩쏘를 줄였나 | ADX<20 회피군 성과 비교 |
| RS가 올라타기에 도움 됐나 | C타입 RS 상위/하위 MFE 비교 |
| 수급 동행이 의미 있었나 | 동행군과 비동행군 hit/miss 비교 |
| 뉴스/DART 명분이 도움 됐나 | event_reason별 성과 비교 |
| B와 C 중 어디가 나았나 | 조정매수 vs 올라타기 PnL/MDD/MFE/MAE |

## 6. 구현 우선순위

1. 공통 ledger schema 확장: weekly/adx/rs/supply/event fields 추가
2. 퀀트봇 sample scanner: 주봉 게이트 + 눌림 상태 기록
3. 단타봇 3-Type paper scanner: A/B/C 병행 기록
4. 매일 리포트: feature별 성과 요약
5. 6/12 1차 SHOW ME: 타입별 equity/MFE/MAE/회피이유 차트

## 7. 최종 원칙

이 전달서의 모든 항목은 처음에는 관측 필드다.  
돈을 버는 feature만 룰로 승격한다.  
좋아 보이는 말보다, forward paper ledger가 이긴다.
