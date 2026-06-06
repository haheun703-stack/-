# 한국 장기 사이클·밸류 레짐 레이어 — 6/6 검수 및 적용 지시

## 결론

붙여넣은 글의 핵심은 단타 진입법이 아니라 **한국장 큰 물때 판단법**이다.

- 경기선행지수 순환변동치 102 이상: 과열 주의
- 경기선행지수 순환변동치 첫 감소: 장기 투자 주의보
- 경기선행/동행 순환변동치 데드크로스: 위험 경보
- 경기선행지수 순환변동치 98 근처 + KOSPI PBR 1.0 이하: 장기 축적권
- KOSPI PBR 0.8 근처: 강한 축적 후보

단, 이 레이어는 **C60 hard gate 대체가 아니다**. 월간 장기 레짐 관측용이며, 실주문·스케줄러·SAJANG 권한은 0이다.

## 구현

신규 파일:

- `src/use_cases/kr_cycle_valuation_regime.py`
- `scripts/kr_cycle_valuation_regime.py`
- `tests/test_kr_cycle_valuation_regime.py`

수동 실행:

```powershell
.\venv\Scripts\python.exe -u -X utf8 scripts\kr_cycle_valuation_regime.py --no-write
.\venv\Scripts\python.exe -u -X utf8 scripts\kr_cycle_valuation_regime.py
```

저장 위치:

- `data_store/regime/kr_cycle_valuation_YYYY-MM-DD.json`

## 현재 검수 결과

현재 `data/ecos_macro.json`에는 기존 금리·환율·CPI·채권 데이터만 있고, 다음 필수 입력은 없다.

- `leading_cycle`: 경기선행지수 순환변동치
- `coincident_cycle`: 경기동행지수 순환변동치
- `kospi_pbr`: KOSPI PBR

따라서 현재 산출은 정상적으로 `DATA_UNAVAILABLE`이다. 이 상태에서 매매판단으로 쓰면 안 된다.

## 퀀트봇 적용 지시

사용 목적:

- 장기 투자/레버리지 비중 판단 보조
- C60 강세/약세와 함께 보는 월간 온도계
- 488080·삼성전자 단일레버·SK하이닉스 단일레버의 장기 비중 조절 참고

금지:

- C60 hard gate 대체 금지
- `engine_policy_map`의 R1/R4를 이 지표로 직접 변경 금지
- 98/PBR 1.0 이하라는 이유만으로 자동 매수 금지
- 102/첫 감소라는 이유만으로 자동 매도 금지

권장 처리:

- `CYCLE_ACCUMULATION`: 조정분할(C) 장기 축적 후보 라벨
- `CYCLE_FLOOR_WATCH`: 바닥권 관찰 라벨
- `CYCLE_OVERHEAT_WARNING`: 레버 신규투자 주의 라벨
- `CYCLE_RISK_OFF`: 장기 신규투자 금지 라벨
- `DATA_UNAVAILABLE`: 판단 금지

## 단타봇 적용 지시

사용 목적:

- 시장 온도 라벨
- 보유기간/추격강도 조절 참고
- A/B/C paper training 결과 해석 보조

금지:

- 개별 종목 진입 신호로 사용 금지
- A/B/C 후보를 이 지표로 hard gate 차단 금지
- 뉴스/EVENT나 수급처럼, 이 지표도 직접 매수 버튼이 아니다.

권장 처리:

- 과열권: C 타입 올라타기 보유기간 짧게 관찰
- 위험권: B/C 모두 후보는 기록하되 해석 보수화
- 바닥권: B 타입 눌림/반등형 관측 강화

## 후속 데이터 수집 과제

이 레이어를 실사용하려면 다음 데이터 연결이 필요하다.

1. KOSIS 또는 ECOS에서 경기선행지수 순환변동치 월별 시계열 수집
2. KOSIS 또는 ECOS에서 경기동행지수 순환변동치 월별 시계열 수집
3. pykrx `get_index_fundamental` 기반 KOSPI PBR 안정 수집
4. 과거 1998/2000/2007/2008/2011/2017/2020/2021 구간 백테스트
5. C60 대비 선행성 비교: 가격 C60보다 빠른가, 아니면 느린가

## 안전선

- 실주문 0
- scheduler 변경 0
- SAJANG 변경 0
- hard gate authority 0
- 자동승격 금지
- 결과가 좋아도 1차는 shadow/log만

## 검증

```powershell
.\venv\Scripts\python.exe -u -X utf8 -m pytest tests\test_kr_cycle_valuation_regime.py tests\test_engine_policy_map.py tests\test_regime_router_v1.py -q
```

결과:

- `21 passed`
- 현재 로컬 데이터 산출: `DATA_UNAVAILABLE`

## 판정

이 글에서 가져올 가치는 있다. 하지만 **종목 발굴 기능이 아니라 시장 체온계**다.

다음 단계는 데이터 수집기 연결 후, `98/PBR` 바닥권과 `102/첫 감소/데드크로스` 위험권이 실제 KOSPI·삼성전자·레버리지 운용 성과에 어떤 영향을 줬는지 검증하는 것이다.
