# 퀀트봇 자동매매 사전 테스트 PDCA — Plan

**작성**: 2026-05-14
**상태**: Plan 완료, **Do 단계 진입 전 다단계 검증 필수**
**목표**: 안전한 퀀트봇 자동매매 시작 (계좌 47339014-01)

## 1. 배경

5/14 16:30 KIS API 키 양방향 교환 완료:
- 퀀트봇 .env: `PSXv...KDDz / 43879566-01` → **`PSEf...ZqsS / 47339014-01`** (옛 단타봇 키)
- 단타봇 .env: 반대 교환
- **이제부터 퀀트봇이 자동매매 주체**, 단타봇 자동매매 종료

퐝가님 지시: **"아직 자동매매 ON 하지 말고 테스트 더"** — 활성화 전 충분 테스트 필요.

## 2. 현재 상태 (2026-05-14 진단)

### 코드 자산
- `src/adapters/kis_order_adapter.py` (16KB, 3/25 작성)
  - `KisOrderAdapter` 클래스: mojito2 래핑
  - 6개 주문 함수: `buy_limit`, `sell_limit`, `buy_market`, `sell_market`, `cancel`, `modify`
  - `BalancePort`, `CurrentPricePort` 동시 구현
  - **MODEL=REAL → 실전 / MODEL!=REAL → 모의투자 (`mock=True`)**

### 환경 설정
- `.env`: `MODEL=REAL` (현재 실전 모드 활성)
- VPS crontab: **매매 스크립트 없음** (자동매매 OFF, 안전)
- systemd: `quantum-scheduler.service` (BAT만, 매매 X)

### 위험 신호
- 만약 누군가 `KisOrderAdapter().buy_market(...)` 호출하면 **즉시 실전 매매 발생**
- 현재 cron 미등록이라 자동 호출은 없음
- 그러나 실수로 스크립트 실행 시 위험

## 3. 목표

| 단계 | 목표 |
|------|------|
| **사전 테스트 1**: MOCK 모드 | 모의투자 계좌로 모든 주문 함수 동작 검증 |
| **사전 테스트 2**: 실전 1주 소액 | 실전 계좌(47339014-01)에 1주 단위 매수/매도 (~수만원 손실 한도) |
| **사전 테스트 3**: 손절/익절 | 자동 손절 -3% / 익절 +5% 동작 검증 |
| **사전 테스트 4**: 슬리피지 측정 | 지정가 vs 시장가 체결 차이 정량화 |
| **단계적 활성화** | 1주 → 5주 → 10주 → 정상 운영 |

## 4. 단계별 진행 계획

### P0: 안전 가드레일 구축 (1세션)

- [ ] **`.env`에 자동매매 ON/OFF 스위치 추가**
  ```
  AUTO_TRADING_ENABLED=false   # 명시적 활성화 전까지 false
  AUTO_TRADING_MAX_QTY=1       # 종목당 최대 수량
  AUTO_TRADING_MAX_AMOUNT=100000  # 일일 최대 매수 금액 (10만원)
  ```
- [ ] `KisOrderAdapter.buy_*`/`sell_*` 진입 시 가드레일 검사
  ```python
  if not int(os.getenv("AUTO_TRADING_ENABLED", "0")):
      raise PermissionError("자동매매 미활성화 (.env AUTO_TRADING_ENABLED=true 필요)")
  if quantity > int(os.getenv("AUTO_TRADING_MAX_QTY", "1")):
      raise ValueError(f"최대 수량 초과 (요청 {quantity} > 한도 {max})")
  ```
- [ ] 일일 누적 매수 금액 추적 (`data/auto_trading/daily_volume.json`)
- [ ] Telegram 알림: 매수/매도 시 즉시 발송

### P1: MOCK 모드 검증 (1세션)

- [ ] `.env.test` 별도 파일 생성: `MODEL=MOCK`, `KIS_ACC_NO=모의투자_계좌번호`
  - **모의투자 계좌 발급 필요** (KIS 개발자센터)
- [ ] 테스트 스크립트 작성: `scripts/test/test_kis_order_mock.py`
  ```python
  # 1. 잔고 조회
  # 2. 005930 1주 매수 (지정가)
  # 3. 잔고 재확인
  # 4. 005930 1주 매도
  # 5. 결과 검증
  ```
- [ ] 모의투자 결과 비교: KIS 모의 vs 예상

### P2: 실전 1주 검증 (1세션, 시장 영업일)

- [ ] `.env`로 복귀 (`MODEL=REAL`, `AUTO_TRADING_ENABLED=true`)
- [ ] **1주만 매수 → 즉시 매도** (시장가, 약 5초 후)
  - 005930 1주 = 약 30만원 → 손실 한도 10만원 이하
- [ ] 슬리피지 측정: 지정가 vs 체결가 차이
- [ ] 텔레그램 알림 수신 확인
- [ ] 일일 누적 매수 금액 정상 추적
- [ ] **만족 시에만 P3 진행**

### P3: 손절/익절 검증 (1주 단위, 1-2주 모니터링)

- [ ] 자동 손절 로직 추가 (-3% 손실 시 매도)
- [ ] 자동 익절 로직 추가 (+5% 이익 시 매도 또는 trailing stop)
- [ ] 1종목 1주 단위로 실제 시장 노출 → 1-2주 결과 모니터링
- [ ] 슬리피지/체결 지연 누적 측정

### P4: 단계적 활성화 (점진 확대)

- [ ] 1주 → 5주 (50만원 한도)
- [ ] 5주 → 10주 (100만원 한도)
- [ ] 결과 만족 시 정상 운영 한도로 확대
- [ ] cron 등록 (BAT-D 후속 자동 실행)

## 5. 위험 관리 (Critical)

| 위험 | 대응 |
|------|------|
| **실수 매매 (대량 주문)** | `AUTO_TRADING_MAX_QTY=1` 환경변수 강제 |
| **고가 매수 (지정가 오타)** | 현재가 ±5% 범위 검증 |
| **반복 매수 (무한루프)** | 일일 매수 횟수 한도 (5회) |
| **API 에러 시 재시도 폭주** | exponential backoff, 최대 3회 |
| **계좌 잔고 부족** | 사전 잔고 체크 |
| **장 마감 시 주문** | 거래 시간 09:00~15:30 외 차단 |
| **휴장일 주문** | KRX 휴장일 캘린더 사전 체크 |

### 비상 정지 (Emergency Stop)

```bash
# 즉시 자동매매 중지
ssh ubuntu@13.209.153.221 "echo 'AUTO_TRADING_ENABLED=false' >> ~/quantum-master/.env"
# 또는 systemd 서비스 중지
sudo systemctl stop quantum-scheduler
```

## 6. 가드레일 (코드 패치 위치)

`src/adapters/kis_order_adapter.py` 라인 ~45 (buy_limit 진입 시):

```python
def buy_limit(self, ticker: str, price: int, quantity: int) -> Order:
    # === 자동매매 가드레일 ===
    if not int(os.getenv("AUTO_TRADING_ENABLED", "0")):
        raise PermissionError("[GUARD] AUTO_TRADING_ENABLED=true 필요")
    
    max_qty = int(os.getenv("AUTO_TRADING_MAX_QTY", "1"))
    if quantity > max_qty:
        raise ValueError(f"[GUARD] 수량 초과 ({quantity} > {max_qty})")
    
    # 거래 시간 체크
    from datetime import datetime
    now = datetime.now().time()
    from datetime import time as dtime
    if not (dtime(9, 0) <= now <= dtime(15, 30)):
        raise RuntimeError(f"[GUARD] 거래 시간 외 ({now})")
    
    # === 실제 주문 ===
    logger.info("[주문] 지정가 매수: %s %d주 @ %d원", ticker, quantity, price)
    ...
```

## 7. 검증 방법

### P1 (MOCK) 검증
- mojito 모의투자 잔고 조회 정상
- 1주 매수/매도 응답 코드 0 (성공)
- 잔고 변동 정상 추적

### P2 (실전 1주) 검증
- 실제 KIS 계좌 47339014-01 잔고 변동 확인
- 텔레그램 알림 수신
- 체결가 예상 범위 내 (±0.5%)

### P3 (손절/익절) 검증
- 1-2주 동안 손절/익절 시그널 동작
- 예상 손익 vs 실제 손익 일치율 90%+

## 8. 진행 절차 (다음 세션부터)

### 다음 세션 (Session 1)
1. P0 가드레일 코드 작성 (`kis_order_adapter.py` 수정)
2. `.env`에 AUTO_TRADING_* 변수 추가
3. 단위 테스트 작성

### Session 2
1. KIS 개발자센터에서 모의투자 계좌 신청
2. `.env.test` 작성
3. P1 MOCK 테스트 실행

### Session 3 (시장 영업일)
1. P2 실전 1주 검증
2. 결과 비교
3. P3 진행 결정

### Session 4+
- P3 손절/익절 검증
- P4 단계적 활성화
- 정상 운영 진입

## 9. 의존 작업 / 보류 사항

- BAT-D 시간 최적화 (현재 진행 중) — 완료 후 자동매매 진입
- 효성_004800.csv fix — 완료 ✅ (5/14)
- 사전 백테스트 결과 검토 (PF 3.35 우량주 매매타이밍 등)

## 10. 기준 / 통과 조건

- **P1 통과**: MOCK에서 5회 매수/매도 모두 성공
- **P2 통과**: 실전 1주 5회 매수/매도 + 슬리피지 < 0.5%
- **P3 통과**: 1-2주 자동 손절/익절 7건+ 시그널 정상
- **P4 진입**: 누적 손익 -5% 이내 (실전 안전성 검증)

## 11. 비용 / 리스크 한도

- **P1 (MOCK)**: 비용 0원
- **P2 (실전 1주)**: 종목당 약 30만원 (1주 매수+매도), 손실 한도 1만원/일
- **P3 (실전 1주, 1-2주)**: 누적 손실 한도 5만원
- **P4 (점진)**: 사용자 결정에 따른 점진 확대

---

**현재 단계**: **Plan 완료**, **다음 → P0 가드레일 코드 작성** (다음 세션 권장)

**중요**: 본 PDCA는 자금 손실 위험이 있는 작업입니다. 각 단계 통과 후에만 다음 단계로 진행하며, 사용자 명시적 승인이 필요합니다.
