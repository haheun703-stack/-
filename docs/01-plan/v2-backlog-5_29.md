# 퀀트봇 v2 Backlog — 5/29(금) 등록

> **상태**: backlog 등록 (실행 시점은 별도 결단)
> **계기**: 5/29 사장님 결단 — "MERGE 4건은 9월 v2 리팩 후보로 유지. 지금은 운영 안정화가 먼저. run_etf_rotation.py 별도 P3 audit 등록"
> **상위 문서**: `docs/02-design/deletion-quarantine-audit-5_29.md`

---

## 0. 한 줄 요약

> **5/29 deletion/quarantine audit 결과 즉시 처리 보류 7건 + 후속 P3 권장 4건 = 총 11건을 v2 backlog에 등록. 운영 안정화 우선 후 별도 PDCA 진입.**

---

## 1. v2 backlog 항목 매트릭스

| # | 항목 | 종류 | 우선순위 | 예상 진입 시점 | 예상 작업량 | 비고 |
|---|---|---|---|---|---|---|
| **B-1** | `paper_warmup_daily.py` → canonical paper evaluation 파이프라인 흡수 | MERGE | P3 | 9월 v2 | ~4시간 | 별도 PDCA + 회귀 + Codex |
| **B-2** | `paper_trading_unified.py` → FLOWX portfolio tracking 모듈 흡수 | MERGE | P3 | 9월 v2 | ~3시간 | 동일 |
| **B-3** | `run_limit_up_scanner.py` → `run_adaptive_cycle --limit-up` 흡수 | MERGE | P3 | 9월 v2 | ~3시간 | 동일 |
| **B-4** | `run_quant_3day_pilot.py` → `run_adaptive_cycle --paper --pilot` 흡수 | MERGE | P3 | 9월 v2 | ~2시간 | 동일 |
| **B-5** | `scripts/run_etf_rotation.py` 단독 deletion/quarantine audit | P3 audit | P3 | 운영 안정화 후 | ~30분 | ETF orders → broker 변환 경로 확인 |
| **B-6** | `scripts/dashboard.py` → `scripts/archive/deprecated/` 이동 | DELETE | P2 | Codex 회신 + 별도 승인 후 | ~10분 | reference 0건 검증 완료 (5/29) |
| **B-7** | `scripts/one_off/integration_dryrun_5_20.py` → `scripts/archive/orphan_*/` 이동 | DELETE | P2 | Codex 회신 + 별도 승인 후 | ~10분 | reference 0건 검증 완료 (5/29) |
| **B-8** | Flask "청산" 라우트 비활성화 (diff 초안 작성 완료) | QUARANTINE | P2 | Codex 회신 + 별도 승인 후 | ~30분 (적용+회귀) | diff: `docs/02-design/flask-liquidate-route-disable-diff-5_29.md` |
| **B-9** | `tests/test_live_trading_import.py` 신규 회귀 (SyntaxError 자동 검출) | P3 회귀 | P3 | 운영 안정화 후 | ~20분 | 5/29 P2 Critical 봉합 재발 방지 |
| **B-10** | 시간 의존 fixture (`TestPhase1ChartHeroExecutor` 등) 동적화 | P3 회귀 | P3 | 운영 안정화 후 | ~1시간 | `freezegun` 도입 vs 동적 timedelta |
| **B-11** | Flask `시작`/`정지` 큐 단독 P3 audit | P3 audit | P3 | 운영 안정화 후 | ~30분 | `remote_queue.json` polling 소비자 부재 (B-8과 동일 패턴) — `시작`은 KILL_SWITCH 삭제, `정지`는 KILL_SWITCH/STOP.signal 생성 트리거 가능. canonical 정합 확인 |

---

## 2. 항목별 상세

### B-1. paper_warmup_daily → canonical paper evaluation 흡수
**현재**: `scripts/paper_warmup_daily.py` — 일일 warmup 추적 (paper intent 등록 + 평가)
**문제**: warmup 평가 로직이 cron 스크립트 안에 분산되어 있음
**v2 방향**: `src/use_cases/paper_evaluation_pipeline.py` 신규 모듈 → 스크립트는 얇은 CLI 래퍼만
**영향**: VPS cron `0 9 * * 1-5 paper_warmup_daily.py` → `0 9 * * 1-5 python -m paper_evaluation_pipeline daily`
**위험**: paper 평가 로직 변경 시 5/29까지 누적된 warmup 데이터 호환성 필요

### B-2. paper_trading_unified → FLOWX portfolio tracking 흡수
**현재**: `scripts/paper_trading_unified.py` — 1주 rolling paper portfolio
**문제**: portfolio rolling 로직이 스크립트에 매몰
**v2 방향**: FLOWX portfolio tracking 모듈 (Supabase 연동) → 스크립트는 BAT-D 트리거만
**영향**: BAT-D 워크플로 변경
**위험**: rolling 윈도우 정의 변경 시 기존 데이터 정합성

### B-3. run_limit_up_scanner → run_adaptive_cycle --limit-up
**현재**: 독립 cron 주기
**문제**: 상한가 풀림 전담 스캐너 + adaptive cycle 별개 entrypoint
**v2 방향**: `run_adaptive_cycle --limit-up` 플래그로 흡수
**영향**: VPS cron 1줄 통합
**위험**: 두 cycle의 빈도/스케줄 차이

### B-4. run_quant_3day_pilot → run_adaptive_cycle --paper --pilot
**현재**: PILOT_START/PILOT_END 하드코드, PAPER 강제
**문제**: 3일 paper pilot 특수 케이스가 별도 스크립트
**v2 방향**: 플래그 기반 구성 (`--pilot 5-27:5-29`)
**영향**: 추후 pilot 재실행 시 CLI 일관성
**위험**: pilot 기간 외 실행 시 의도 확인

### B-5. run_etf_rotation.py 단독 audit
**현재**: 본 audit (5/29) 범위 외
**확인 필요**:
- `ETFOrchestrator.decide()` orders dict → 실제 KIS 주문 변환 경로 존재 여부
- 변환 경로 있으면 mode/executor_bot 명시 여부 + KILL_SWITCH 가드 적용 여부
- 변환 경로 없으면 의사결정 라이브러리로 KEEP
**결단 필요**: 운영 안정화 (Codex 회신 + paper 재가동 결단) 통과 후 별도 PDCA

### B-6 / B-7. DELETE 2건 archive 이동
**reference 재확인 결과 (5/29 검증)**:
- `scripts/dashboard.py`: 외부 import 0건 / VPS cron 참조 0건 / 헤더 "DEPRECATED" + `sys.exit(0)` 명시
- `scripts/one_off/integration_dryrun_5_20.py`: 외부 import 0건 / VPS cron 참조 0건 / 5/20 1회 시뮬 완료
**적용 방안**:
- B-6: `scripts/dashboard.py` → `scripts/archive/deprecated/dashboard.py` (헤더 이동 사유 추가)
- B-7: `scripts/one_off/integration_dryrun_5_20.py` → `scripts/archive/orphan_20260529/integration_dryrun_5_20.py`
**위험**: 0 (이동 후에도 `git log` / `git show` 추적 가능)

### B-8. Flask "청산" 라우트 비활성화
**diff 초안**: `docs/02-design/flask-liquidate-route-disable-diff-5_29.md` (작성 완료)
**적용 시점**: Codex 회신 + 사장님 별도 승인 후만
**위험**: 0 (dead-letter queue → 안전 응답 교체)

### B-9. live_trading import 회귀 (Critical 봉합 재발 방지)
**계기**: 5/29 P2 작업 시 `replace_all=true` Edit으로 들여쓰기 결함 발생 → 회귀가 import 0건이라 검출 못함
**작성안**:
```python
# tests/test_live_trading_import.py
def test_live_trading_import_no_syntax_error():
    """5/29 P2 Critical 봉합 재발 방지: live_trading.py SyntaxError 검출"""
    from src.use_cases.live_trading import LiveTradingEngine, create_live_engine
    assert LiveTradingEngine is not None
    assert create_live_engine is not None
```
**영향**: 회귀 25/25 → 26/26

### B-10. 시간 의존 fixture 동적화
**현재**: `TestPhase1ChartHeroExecutor` 등 fixture가 `expires_at=YYYY-MM-29T15:30:00+09:00` 고정 → KST 15:30 이후 실행 시 자연 만료 → 자연 실패
**옵션 A**: `freezegun` 라이브러리 도입 (`@freeze_time("2026-05-29T10:00:00+09:00")`)
**옵션 B**: fixture를 `now_kst + timedelta(hours=N)` 동적 계산
**권장**: 옵션 B (의존 추가 없음)
**위험**: 0 (테스트 코드만)

---

## 3. v2 진입 트리거 조건

다음 **모든 조건** 충족 시 v2 backlog 처리 PDCA 진입:

1. ✅ Codex 회신 ①+② 도달 + PASS
2. ✅ 사장님 commit 결단 (5/29 종일 작업 5건)
3. ✅ Paper 재가동 심사 1차 통과 (별도 단계)
4. ✅ B-6 / B-7 / B-8 (P2 항목) 처리 완료
5. ⏸️ 운영 1주일 이상 사고 0건 누적

→ **위 조건 부족 시 v2 backlog는 등록 상태 유지, 진입 X**.

---

## 4. 적용 금지 (본 backlog 등록 후)

- ❌ B-1 ~ B-10 항목 즉시 실행 X
- ❌ MERGE 4건 (B-1~B-4) 9월 진입 시점 자율 결단 X — 사장님 명시 결단 후만
- ❌ DELETE 2건 (B-6/B-7) 즉시 archive 이동 X — 사장님 별도 승인 후만
- ❌ Flask 비활성화 (B-8) 즉시 적용 X — Codex 회신 + 별도 승인 후만

---

## 5. 표현 룰

### 사용 가능
- "v2 backlog 10건 등록 완료"
- "운영 안정화 우선, v2 진입은 별도 트리거 충족 후"
- "DELETE 2건 + QUARANTINE 1건은 별도 commit 묶음 대기"

### 사용 금지
- "v2 진입 임박" X
- "운영 안정화 완료" X
- "Phase 2 가능" X

---

## 6. 다음 보고 시 갱신 권장 항목

| 보고 시점 | 갱신 항목 |
|---|---|
| Codex 회신 ①+② 도달 | B-6/B-7/B-8 우선순위 P2 → P1 승격 결단 |
| 사장님 commit 결단 | commit 묶음 결정 (3개 묶음 권장) |
| Paper 재가동 심사 통과 | v2 진입 트리거 조건 §3 #3 충족 표시 |
| 운영 1주 사고 0건 누적 | v2 진입 트리거 조건 §3 #5 충족 표시 |

---

## 7. 연결 문서
- `docs/02-design/deletion-quarantine-audit-5_29.md` (1차 audit)
- `docs/02-design/flask-liquidate-route-disable-diff-5_29.md` (B-8 diff 초안)
- `docs/02-design/p1-truth-pack-5-29.md`
- `docs/02-design/p1-residual-plan-5-29.md`
- `ops/codex_outbox/20260529T101341_..._p1-a4-callers-migration_review-requested.md` (의뢰서 ①)
- `ops/codex_outbox/20260529T194250_..._p2-residual-4items-and-critical-fix_review-requested.md` (의뢰서 ②)
- `memory/decision_5_28_p1_blockers.md` (사장님 결정문)
