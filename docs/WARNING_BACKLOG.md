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

---

## 카테고리 F: 5/15 BAT-D 실전 검증 발견 (2026-05-17 등록)

| # | 이슈 | 영향 | 우선순위 |
| --- | --- | --- | --- |
| F1 | `scripts.etf_transmission` import 잔존 (실제 모듈은 `archive/deprecated/`로 폐기됨, 커밋 `d88d131`) — `ETF 전파 모델 실패` WARNING 2회 (06:10, 17:08) | LOCK 위반 시도 + 매일 WARNING 노이즈 (기능 미수행) | **P1** |
| F2 | `scripts.theme_scan_runner` import 잔존 (실제 모듈은 archive 폐기됨) — `테마 스캔 스킵` | 같은 사유 (LOCK 위반 시도 + 노이즈) | **P1** |
| F3 | `scripts.morning_briefing_generator` import 잔존 (실제 모듈은 archive 폐기됨) — BAT-M_morning 실패 1건의 직접 원인 | BAT-M_morning 매일 `[FAIL]` 카운트 (실패 4건의 일부) | **P1** |
| F4 | BAT-D 소요시간 회귀: 5/14 106분 → 5/15 **123분** (+17분), update_daily_data 978초 → **1471초** (+50%) | 새 KIS 키 첫 정규 운영 → 캐시 부재 가능, 5/18~5/19 추이 관찰 후 확정 | P2 |
| F5 | 스케줄러 메시지 충돌: `❌ 스케줄러: BAT-D 완료, 실패 4건` 직후 `=== BAT-D 완료 (실패: 0건) ===` | 카운트 정의 불일치 (sub-task vs 최종) — 정의 통일 필요 | P3 |

### F1~F3 후속조치 (2026-05-17 호출 위치 식별 완료)

모듈 3건은 의도적으로 `scripts/archive/deprecated/`로 폐기 완료 (커밋 `d88d131`). **CLAUDE.md LOCK 규칙에 따라 archive 참조·실행·import 금지** → 호출 코드에서 import/실행 부분을 제거해야 합니다.

| 호출 파일 | 라인 | 내용 | try/except 보호 | 실패 영향 |
| --- | --- | --- | --- | --- |
| `scripts/us_overnight_signal.py` | L1326~1343 | ETF 전파 모델 호출 + 결과 활용 (L1562) | ✅ 보호됨 | WARNING 로그만 (signal 처리 계속) |
| `scripts/run_morning_briefing.py` | L23~29 | 테마 스캔 (실험적 호출, 본격 morning briefing 아님) | ✅ 보호됨 | WARN 후 다음 단계 진행 |
| `scripts/cron_morning_briefing.py` | L64~69 | 테마 스캔 | ✅ 보호됨 | WARN 후 다음 단계 진행 |
| `scripts/cron_morning_briefing.py` | L83~93 | **morning_briefing_generator 호출 — 실패 시 `sys.exit(1)`로 종료** | ⚠️ 보호되지만 종료 | **BAT-M_morning 실패 1건의 직접 원인** |

### F1~F3 처리 옵션 (퐝가님 결정 필요)

**옵션 A. 단순 import 제거 (보수적)**

- 4개 try/except 블록을 통째로 들어내고 결과 사용 부분 (us_overnight_signal L1562, cron_morning_briefing L84~ 이후 STEP 3/4) 정리
- 효과: WARNING/FAIL 노이즈 0건, BAT-M_morning 실패 0건
- 단점: morning_briefing 기능 자체가 사라짐 (이미 아무도 사용 안 함이 확실하면 OK)

**옵션 B. 모듈 복구 + 재활성화**

- archive 모듈 3건을 복구해서 `scripts/` 본가로 이동
- 효과: 기능 활성화
- 단점: 폐기 결정 (`d88d131`) 사유 확인 필요, 복구 비용 발생

**옵션 C. 호출만 제거 + cron 등록도 제거 (BAT-M_morning 통째 폐지)**

- 옵션 A + crontab/run_bat에서 BAT-M_morning 자체 제거
- 효과: 가장 깔끔 (이미 결과를 안 쓰면 cron만 돌 가치 없음)
- 권장도: ★★★ — 결과를 정말 안 쓴다면 cron 자체를 빼는 게 맞음

### F4 후속조치

- 5/18 (월) BAT-D 시간 확인 — 5/14 수준(106분) 회복 여부

---

## 카테고리 F-2: 5/17 전체 검수 발견 archive LOCK 위반 4건 (2026-05-17 등록·해결)

c095e9a (F1~F3) 처리 후 잔존 검수 결과 **importlib 동적 로딩** 4건 추가 발견. 본 세션에서 일괄 해결.

| # | 파일:라인 | archive 의존 | 위험도 | 처리 |
| --- | --- | --- | --- | --- |
| F-2-1 | `scripts/run_morning_briefing.py:37` | `archive/legacy_wrappers/send_market_briefing.py` (build_unified_morning) | 상 (BAT-B 07:00) | ✅ `src/use_cases/morning_briefing.py` 추출 + 정규 import |
| F-2-2 | `scripts/us_overnight_signal.py:413` | `archive/backfill/backfill_us_kr_history.py` (PatternMatcher) | 중 (try 보호) | ✅ `src/utils/us_kr_history.py` 추출 + 정규 import |
| F-2-3 | `scripts/update_us_kr_daily.py:40` | `archive/backfill/backfill_us_kr_history.py` (US_SYMBOLS 등) | 상 (모듈 최상위 import) | ✅ 위와 동일 모듈에서 import |
| F-2-4 | `scripts/group_relay_detector.py:30` | `archive/analysis/group_relay_backtest.py` (find_csv_by_ticker 등) | 상 (모듈 최상위 import) | ✅ `src/utils/group_relay_loaders.py` 추출 + 정규 import |

### 처리 결과
- archive importlib 호출: **4건 → 0건**
- LOCK 위반 grep: 코드 0건 (주석 2건만 잔존 — 위반 시도 차단 명시용)
- 가드레일 단위테스트 13/13 PASSED 유지
- 호출자 4건 import smoke test 통과

### F-3: cron_morning_briefing.py DEPRECATED 처리 (옵션 C 완수)
F3 처리(`c095e9a`)에서 cron 폐지는 됐으나 파일 자체는 dead code로 잔존, 내부에 archive import 시도(L65, L84) 잔존. 본 세션에서 `main()` 진입 즉시 종료 + DEPRECATED 명시로 실행 차단 (파일은 보존). 향후 정리는 별도 결정.
- 회복 안 되면 KIS API rate limit / 토큰 캐시 검토

