# Blind Test — v10.3 + Group Rotation

## 테스트 개요
- **전략**: v10.3 스윙(60%) + 그룹순환매 D모드(40%)
- **자본**: 1억 (v10: 6,000만 / 순환매: 4,000만)
- **시작**: 2026-02-19
- **최소 기간**: 8주 (40거래일)
- **벤치마크**: KOSPI 수익률

## 규칙 (절대 준수)

1. **시스템 추천만 따른다.** 감으로 추가 진입/청산 금지.
2. **추천 시 다음 거래일 시가 매수.** 슬리피지 0.5% 가정.
3. **청산 조건 도달 시 다음 거래일 시가 매도.**
4. **일일 리포트는 장마감 후 1시간 내 기록.**
5. **파라미터 변경 금지** — 최소 4주간.
   - 4주 후 중간 점검에서 조정 여부 판단.
6. **벤치마크**: KOSPI 수익률 대비 초과수익 측정.
7. **중단 조건**: MDD -15% 도달 시 1주 중단 후 재검토.
8. **추천 0종목 = 시스템이 기회 없다고 판단.** 정상 동작.

## 일일 프로세스

```
[07:00] US Signal 확인 → signal_YYYYMMDD.json
[09:00] 장 시작 → 시초가 기록 (장중 행동 없음)
[15:30] 장 마감 → KOSPI 종가/레짐 기록
[17:00] 데이터 업데이트 → daily_auto_update.bat
[17:30] 스캔 실행 → scan_buy_candidates.py + blind_test_daily.py
[18:00] 블라인드 테스트 기록 → daily_log JSON + 텔레그램
```

## 실행 명령

```bash
# 일일 기록 (자동)
python scripts/blind_test_daily.py

# 포지션 진입 기록
python scripts/blind_test_daily.py --enter v10 005930 삼성전자 55000 100

# 포지션 청산 기록
python scripts/blind_test_daily.py --exit 005930 56000 target

# 주간 리포트 생성
python scripts/blind_test_daily.py --weekly
```

## 디렉토리 구조

```
blind_test/
├── README.md           # 이 파일
├── config.json         # 테스트 설정
├── positions.json      # 현재 보유 (실시간)
├── trades.csv          # 전체 거래 (누적)
├── equity_curve.csv    # 일별 자산 추이
├── daily_log/          # 일별 기록 JSON
└── weekly_report/      # 주간 성과 보고
```

## 백테스트 기준치 (목표)

| 지표 | v10.3 단독 | 그룹순환매 | 결합 60/40 |
|------|-----------|-----------|-----------|
| 수익률 | +21.7% | +18.2% | +20.5% |
| MDD | -4.5% | -6.8% | -4.2% |
| PF | 1.78 | 2.00 | - |
| Sharpe | 2.20 | 1.28 | 2.22 |
