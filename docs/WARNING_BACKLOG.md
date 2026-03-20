# Warning Backlog — Paper Trading 중 점진적 개선 대상

> 생성일: 2026-03-20 (전체 파이프라인 검수 후)
> Critical 15건 → 전부 수정 완료 (fb128a5)
> Paper Trading 영향 4건 → 수정 완료 (본 커밋)
> 아래 ~26건 → 급하지 않음, Paper Trading 하면서 하나씩 정리

---

## 카테고리 A: 데이터 정확성 (기능은 작동, 값이 부정확)

| # | 파일:라인 | 이슈 | 우선순위 |
|---|----------|------|---------|
| A1 | kis_intraday_adapter.py:110 | 체결강도 필드명 `seln_cntg_smtn` → `tday_rltv`로 수정 필요 | P2 |
| A2 | kis_intraday_adapter.py:425-426 | 업종 상승/하락 종목수 필드 부정확 (`stck_prpr_updn`은 부호) | P2 |
| A3 | intraday_eye.py:238-239 | 20일 평균 거래량을 전일 1일치로 대용 → EYE-03 오탐 가능 | P2 |
| A4 | market_journal.py:234 | `foreign_net_bil` 변수명이 실제로는 억 단위 (`/1e8`) | P3 |
| A5 | market_journal.py:1033 | 승률 계산이 날 단위 → 종목별로 변경 권장 | P3 |

## 카테고리 B: 코드 안전성 (현재 작동하지만 취약)

| # | 파일:라인 | 이슈 | 우선순위 |
|---|----------|------|---------|
| B1 | signal_engine.py:918-928 | tc_cfg dict 직접 변경 (mutable state) → copy() 사용 | P2 |
| B2 | signal_engine.py:943,1312 | atr_14=0일 때 명시적 차단 + 로그 | P3 |
| B3 | execution_alpha.py:41 | prev_close=0 방어 없음 → 0원 주문 가능 | P2 |
| B4 | sell_monitor.py:59-62 | 비원자적 JSON 쓰기 → tempfile+rename 패턴 적용 | P3 |
| B5 | kis_order_adapter.py:31-33 | 환경변수 None 시 mojito에 None 전달 | P3 |
| B6 | smart_entry.py:310 | max(qty, 1) 극소 비중 → 의도 대비 과매수 가능 | P3 |
| B7 | _check_and_analyze_defense.py:51 | bb_up==bb_low일 때 0 나누기 | P3 |

## 카테고리 C: 코드 정리 (기능 무관, 깔끔함 개선)

| # | 파일:라인 | 이슈 | 우선순위 |
|---|----------|------|---------|
| C1 | smart_entry.py:148 | max_candle_wait 로드만 하고 미사용 (dead code) | P3 |
| C2 | smart_entry.py:813 | `import json as _json` 불필요 재import | P3 |
| C3 | smart_entry.py:583 | dry_run 로그에서 initial_discount 고정 출력 (exec_alpha 시 오해) | P3 |
| C4 | scan_buy_candidates.py:165-168 | 미사용 변수 `tick` | P3 |
| C5 | scan_buy_candidates.py:60,636,828,838 | 함수 내부 import 반복 → top-level로 통합 | P3 |
| C6 | scan_buy_candidates.py:1707-1708 | O(n^2) 순위 탐색 → enumerate 사용 | P3 |
| C7 | quality_roe.py:218 | `from scipy.stats import norm` 루프 내 import → 상단 이동 | P2 |
| C8 | lens_layer.py:62 | 문서 "4개 렌즈" → "5개 렌즈"로 업데이트 | P3 |
| C9 | models.py:36-44 | `_DEFAULTS` 인스턴스 변수 dead code | P3 |

## 카테고리 D: 설계/아키텍처 개선 (장기)

| # | 파일 | 이슈 | 우선순위 |
|---|------|------|---------|
| D1 | ports.py:250-277 | IntradayDataPort에 5개 메서드 누락 (fetch_orderbook 등) | P2 |
| D2 | telegram_sender.py | 클린 아키텍처 위반 → adapters/로 이동 + NotificationPort | P3 |
| D3 | risk_manager.py vs smart_sell.py | ExitRuleType.name("X1_HARD_STOP") vs SmartSell("X1") 매핑 불일치 | P2 |
| D4 | position_sizer_v2.py:153-159 | Half Kelly b=1.5 하드코딩 → 실데이터 기반 | P2 |
| D5 | value_composite.py:101 | weight_sum 임계치 0.3 vs quality 0.5 불일치 | P3 |
| D6 | signal_engine.py:80-81 | config_path 상대 경로 (BAT에서 cd 필수) | P2 |

## 카테고리 E: 슬러지 (설정값 미참조)

| # | 설정 키 | 비고 |
|---|--------|------|
| E1 | `smart_sell.x1_method: "market_immediate"` | 코드에서 미참조 |
| E2 | `alpha_v2.sizing.risk_pct` | position_sizer_v2에서 미참조 |
| E3 | `alpha_v2.sizing.stop_atr_mult` | position_sizer_v2에서 미참조 |
| E4 | `alpha_v2.sizing.use_correlation` | position_sizer_v2에서 미참조 |

---

## 기타 참고

- `kis_intraday_adapter.py:139`: mojito private 메서드 `_fetch_today_1m_ohlcv()` 직접 호출 → mojito 업데이트 시 깨질 수 있음
- `kis_intraday_adapter.py:496-501`: `is_market_open` 독스트링 "09:00~15:30" vs 코드 "08:30~15:30"
- `telegram_sender.py:28`: API_BASE가 모듈 로드 시 고정 → lazy init 권장
- `smart_sell.py:159`: time.sleep(30) 블로킹 → asyncio 전환 시 주의
- `risk_manager.py:56`: 주간 PnL maxlen=5 → 공휴일 주간 부정확
- `flow_map.py:89`: hot/cold 겹침 가능 (섹터 ≤ 6개 시)
- `engine.py:176`: `check_portfolio()`가 `risk._peak_equity` private 접근
- `_check_and_analyze_defense.py:9`: 하드코딩 날짜 `2026-03-18` (일회성 스크립트)
- `sell_monitor.py:127`: 모의투자 역조건 `MODEL != "REAL"` → 가독성 개선
- `smart_entry.py:1612-1626`: `_wait_until()` 자정 경계 미처리 (운영 시간대에선 무관)
- `smart_entry.py:697-698,784-795`: HOLDING 종목 적응형 정정 → 주석 "초기 지정가 유지"와 불일치
- `scan_buy_candidates.py:1326`: `SignalEngine("config/settings.yaml")` 상대 경로
- `_tick_round()` 5중 중복: smart_entry.py, telegram_command_handler.py, trade_approval.py, sell_monitor.py (trading_models.tick_round import로 통일)
