# P0 가드레일 — Design 문서

**작성일**: 2026-05-17 (일)
**Plan 참조**: `docs/01-plan/quant-auto-trading-test.md` 섹션 4-P0
**현재 단계**: P0 보강 (5/14 기 구현분 + 누락 5건 추가)
**대상**: `src/adapters/kis_order_adapter.py` + 의존 모듈

---

## 1. 배경

5/14에 P0 가드레일 6개 검사 중 6개 구현 완료 (`_guard()` L82~119):

| # | 검사 | 5/14 구현 | 본 Design 추가 |
|---|---|---|---|
| 1 | AUTO_TRADING_ENABLED 활성화 | ✅ | - |
| 2 | AUTO_TRADING_MAX_QTY 종목당 한도 | ✅ | - |
| 3 | AUTO_TRADING_WHITELIST_ONLY 화이트리스트 | ✅ (26 종목) | - |
| 4 | 거래시간 09:00~15:30 | ✅ | - |
| 5 | 주말 차단 | ✅ | 공휴일 확장 |
| 6 | 통과 로그 | ✅ | - |
| 7 | **일일 매수 금액 한도** | ❌ | **신규** |
| 8 | **일일 누적 추적 (daily_volume.json)** | ❌ | **신규** |
| 9 | **현재가 ±5% 범위 검증** | ❌ | **신규** |
| 10 | **Telegram 매수/매도 알림** | ❌ | **신규** |
| 11 | **공휴일 캘린더 (`is_kr_trading_day`)** | ❌ (주말만) | **신규** |
| 12 | **단위 테스트** | ❌ | **신규** |

추가 1건 — **5/25 부처님오신날 대체공휴일 누락 발견 → trading_calendar.py 수정 완료** (별도 fix).

---

## 2. 코드 변경 사항

### 2-1. `_guard()` 메서드 확장

**위치**: `src/adapters/kis_order_adapter.py` L85~119

**변경 후 시그니처**:

```python
def _guard(self, ticker: str, quantity: int, price: int | None = None, side: str = "BUY"):
```

`price`를 추가 — 지정가 매수/매도 시 현재가와 비교용.

**6개 → 9개 검사로 확장**:

1. `AUTO_TRADING_ENABLED=1` (기존)
2. `AUTO_TRADING_MAX_QTY` (기존)
3. `AUTO_TRADING_WHITELIST_ONLY` (기존)
4. 거래시간 09:00~15:30 (기존)
5. **거래일 (주말 + 공휴일, `is_kr_trading_day()` 활용)** ← 확장
6. **일일 매수 금액 누적 한도** ← 신규
7. **현재가 ±5% 범위 (지정가만, price 전달 시)** ← 신규
8. **일일 매수 횟수 한도 (선택, MAX_TRADES_PER_DAY)** ← 신규
9. 통과 로그 (기존)

### 2-2. 일일 누적 추적 모듈

**파일**: `src/utils/auto_trading_volume.py` (신규)

**책임**:

- `record_buy(ticker, amount, qty)` — 매수 시 호출, 누적값 업데이트
- `get_today_volume()` → `{"date": "2026-05-26", "total_amount": 200000, "total_trades": 2, "buys": [...]}`
- 파일 경로: `data/auto_trading/daily_volume.json`
- 날짜 변경 시 자동 초기화 (전날 파일은 `daily_volume_20260526.json`으로 백업)
- atomic write (tempfile + rename) — 동시성 안전

**JSON 스키마**:

```json
{
  "date": "2026-05-26",
  "total_amount": 200000,
  "total_trades": 2,
  "buys": [
    {"ts": "2026-05-26T09:15:23", "ticker": "487240", "qty": 1, "price": 12500, "amount": 12500},
    {"ts": "2026-05-26T10:30:11", "ticker": "395160", "qty": 1, "price": 187500, "amount": 187500}
  ]
}
```

### 2-3. Telegram 알림

**의존**: `src/telegram_sender.py` (기존)

**호출 위치**: `buy_limit` / `sell_limit` / `buy_market` / `sell_market` 의 주문 접수 성공 직후 (`_parse_order_response` 후)

**메시지 포맷**:

```
[자동매매] 매수 접수
종목: KODEX AI전력 (487240)
수량: 1주
가격: 12,500원 (시장가 / 지정가)
금액: 12,500원
일일 누적: 12,500원 / 300,000원 (4.2%)
일일 횟수: 1회 / 5회
시각: 09:15:23
```

매도도 동일 형식 (`매수` → `매도`).

### 2-4. 현재가 ±5% 범위 검증

**조건**: `price`가 제공된 경우 (지정가 주문)

**구현**:

```python
if price is not None:
    current = self.fetch_current_price(ticker).get("current_price", 0)
    if current > 0:
        diff_pct = abs(price - current) / current * 100
        if diff_pct > 5:
            raise ValueError(
                f"[GUARD] 지정가 현재가 ±5% 초과: 지정 {price:,} vs 현재 {current:,} (편차 {diff_pct:.1f}%)"
            )
```

**이유**: 오타 (영순위) 또는 데이터 오류로 인한 비정상 가격 차단. ±5%는 일일 변동 폭 한계 ±30%보다 보수적.

### 2-5. 공휴일 캘린더

**현재 (5/14 구현)**:

```python
if date.today().weekday() >= 5:
    raise RuntimeError(f"[GUARD] 주말 휴장: {date.today()}")
```

**변경 후**:

```python
from src.trading_calendar import is_kr_trading_day
if not is_kr_trading_day():
    raise RuntimeError(f"[GUARD] KRX 휴장일: {date.today()}")
```

`is_kr_trading_day()`는 주말 + 공휴일 (2026년 17개) 모두 처리. 5/25 부처님오신날 대체공휴일 포함 (방금 추가).

---

## 3. 환경변수 (.env)

```bash
# === P0 자동매매 가드레일 (2026-05-17 등록) ===
# 5/25까지: ENABLED=0 (페이퍼 트레이딩 유지)
# 5/26부터: ENABLED=1 (실전 1주 매수 시작)
AUTO_TRADING_ENABLED=0
AUTO_TRADING_MAX_QTY=1
AUTO_TRADING_MAX_AMOUNT=300000          # 일일 최대 매수 금액 (30만원 = ~10종목 1주)
AUTO_TRADING_MAX_TRADES_PER_DAY=5       # 일일 매수 횟수 한도
AUTO_TRADING_WHITELIST_ONLY=1           # 화이트리스트만 매매
AUTO_TRADING_PRICE_RANGE_PCT=5          # 지정가 현재가 ±5% 한도
AUTO_TRADING_TELEGRAM_ALERT=1           # 매수/매도 시 텔레그램 알림
```

기본값 (env 미설정 시):

- `ENABLED=0` (안전 기본값, 매매 차단)
- `MAX_QTY=1`, `MAX_AMOUNT=300000`, `MAX_TRADES=5`, `RANGE_PCT=5`
- `WHITELIST_ONLY=0` (운영 시 1로 활성)
- `TELEGRAM_ALERT=1` (운영 시 즉시 알림)

---

## 4. 단위 테스트 (`tests/test_kis_order_guardrail.py`)

### 4-1. 시나리오 (총 12 케이스)

| # | 시나리오 | 기대 |
|---|---|---|
| 1 | `ENABLED=0` 매수 시도 | `PermissionError("AUTO_TRADING_ENABLED=1 필수")` |
| 2 | `ENABLED=1`, 수량 2 시도 (`MAX_QTY=1`) | `ValueError("수량 한도 초과")` |
| 3 | `WHITELIST_ONLY=1`, 비화이트리스트 종목 (예: `005930`) | `PermissionError("화이트리스트 외 종목")` |
| 4 | 거래시간 외 (예: 18:00) | `RuntimeError("거래시간 외")` |
| 5 | 주말 (5/23 토) | `RuntimeError("KRX 휴장일")` |
| 6 | **공휴일 (5/24 일 석가탄신일)** | `RuntimeError("KRX 휴장일")` |
| 7 | **대체공휴일 (5/25 월)** | `RuntimeError("KRX 휴장일")` |
| 8 | **일일 매수 금액 한도 초과 (310,000원)** | `ValueError("일일 금액 한도 초과")` |
| 9 | **일일 횟수 한도 초과 (6회째)** | `ValueError("일일 횟수 한도 초과")` |
| 10 | **현재가 ±5% 초과 (현재 1만원, 지정 1만 6천원 = +60%)** | `ValueError("지정가 현재가 ±5% 초과")` |
| 11 | 정상 케이스 (모두 통과) | 통과, `_guard()` 정상 종료 |
| 12 | Telegram 알림 발송 검증 | mock 으로 발송 호출 확인 |

### 4-2. 검증 도구

- `pytest` (이미 설치됨)
- `freezegun`: 시간/날짜 mock (`pip install freezegun` 필요)
- `unittest.mock`: KIS API mock

### 4-3. 실행

```bash
./venv/Scripts/python.exe -m pytest tests/test_kis_order_guardrail.py -v
```

---

## 5. 비상 정지 (Emergency Stop)

`.env` 에서 `AUTO_TRADING_ENABLED=0` 변경 + systemd 재시작 (또는 단순히 시스템에서 모든 매매 호출 차단).

```bash
# VPS에서 즉시 자동매매 중지
ssh ubuntu@13.209.153.221 "sed -i 's/AUTO_TRADING_ENABLED=1/AUTO_TRADING_ENABLED=0/' ~/quantum-master/.env"
# systemd가 .env 다시 읽도록 재시작
ssh ubuntu@13.209.153.221 "sudo systemctl restart quantum-scheduler"
```

---

## 6. 통과 조건 (P0 → P1 진입)

- [ ] 12개 단위 테스트 모두 통과
- [ ] `.env` 변수 등록 + 안전 기본값 (`ENABLED=0`)
- [ ] `data/auto_trading/` 디렉토리 생성
- [ ] Telegram 알림 mock 으로 발송 검증 통과
- [ ] 기존 `KisOrderAdapter` 사용처 영향 없음 (paper_trading 등)
- [ ] VPS 배포는 **5/25 일요일 직전 (5/24 정도)** — 평일 배포 금지 (자동매매 의도치 않게 ON 위험)

---

## 7. 변경 영향 평가

### 영향 받는 코드

- `src/adapters/kis_order_adapter.py` (P0 핵심)
- `src/utils/auto_trading_volume.py` (신규)
- `tests/test_kis_order_guardrail.py` (신규)
- `.env` (변수 7개 추가)
- `data/auto_trading/` (디렉토리 신설)

### 영향 받지 않는 코드 (검증 대상)

- `src/use_cases/paper_trading_orchestrator.py` — 페이퍼 매매는 `KisOrderAdapter`를 사용하지 않거나 mock 사용. 영향 없음 가정.
- BAT-D 파이프라인 — 매매 호출 없음, 영향 없음.
- 단타봇 — 별도 프로젝트, 영향 없음.

### 회귀 위험

- **HIGH**: 시간/공휴일 검사 강화로 기존 작동 중인 매매 코드가 차단될 수 있음 → 사용처 grep 후 영향 확인
- **MEDIUM**: 일일 누적 JSON 파일 부재 시 처음 실행에 빈 파일 생성 — `record_buy` 첫 호출에서 안전 처리 필요

---

## 8. 다음 단계 (P1 진입)

P0 통과 후 P1 (MOCK 모드 검증) 진입:

1. KIS 개발자센터에서 모의투자 계좌 신청
2. `.env.test` 작성 (MOCK 계좌 정보)
3. `scripts/test/test_kis_order_mock.py` 작성 (1주 매수/매도 5회)
4. 모의투자 결과 비교

P1 통과 → **5/26 (화) P2 실전 1주 시작** (5/25 부처님오신날 대체공휴일이므로).
