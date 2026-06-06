# FLOWX Market OS v1 — 완성 승인 + 6/8 실관측 런북

작성일: 2026-06-06 · 상태: **1~8단계 구현 완료 / 구현 닫음 → 관측 단계 전환**
HEAD: `e0bbe04` (로컬=origin)

## 0. 완성 판정 (사장님 승인)

FLOWX Market OS v1 = **실주문 없는 관측 운영체제**로 완성. 더 구현할 것 없음.

| 항목 | 값 |
|---|---|
| 구현 단계 | 1~8 전부 |
| 코드 구성 | use_case 7 + 운영 CLI 7 + 테스트 8 (4단계 candidate_tiers는 정합검증이라 별도 CLI·use_case 없음, classify_tier SSOT 통합) |
| 누적 회귀 | 60 passed |
| 실주문 | 0 |
| scheduler 변경 | 0 |
| SAJANG 변경 | 0 |
| PAPER_OPEN | 금지 |
| 매도 자동화 | BLOCKED |
| 주문 심볼 grep | 0 (매 단계) |

**구현은 닫는다. 다음은 6/8 실데이터 1회전 관측이다. 월요일 결과로 바로 실전 전환하지 않는다.**

## 1. 6/8(월) 실관측 1회전 — 할 일

### 1.1 최신 데이터 확보
- `prefer_remote=True` 기본(CLI에서 `--no-remote`를 **주지 않는다**) → pykrx 최신 강제.
- 로컬 `data/raw/*.parquet`가 3/27 stale이므로 `--no-remote`로 돌리면 OHLCV가 비어 `DATA_UNAVAILABLE`/`HOLD_BLOCKED`로 차단됨(가짜 신호 0 = 안전장치). 실관측은 반드시 remote.
- pykrx 자체가 실패하면 그대로 `DATA_UNAVAILABLE`로 보수적 차단됨(오판보다 안전).

### 1.2 파이프라인 수동 1회전 (순서대로, 각 단계 산출물 저장)
```bash
source venv/Scripts/activate
export PYTHONPATH=/d/sub-agent-project_퀀트봇   # BAT는 set PYTHONPATH=D:\sub-agent-project_퀀트봇
python -u -X utf8 scripts/regime_router_v1.py      # → data_store/regime/regime_{date}.json
python -u -X utf8 scripts/engine_policy_map.py      # → data_store/policies/policy_{date}.json
python -u -X utf8 scripts/morning_plan_07.py        # → data_store/plans/plan_{date}.json + .md
python -u -X utf8 scripts/smart_entry_adapter.py    # → data_store/shadow_entries/shadow_entries_{date}.json
python -u -X utf8 scripts/exit_signal_observer.py   # → data_store/exit_observer/exit_observer_{date}.json + .md
python -u -X utf8 scripts/daily_review.py           # → data_store/reviews/daily_review_{date}.json + .md
python -u -X utf8 scripts/show_me_report.py         # → data_store/reports/show_me_{date}.json + .md + .png
```
- **4단계 candidate_tiers는 별도 CLI 없음** — `morning_plan_07`/`paper_smart_entry`의 `classify_tier` SSOT에 통합. 정합은 `pytest tests/test_candidate_tiers_alignment.py`로 보장(plan↔ledger 일치).
- **표준 운영 = 위 7개 CLI 순차 실행** (각 단계 산출물을 모두 저장하는 표준 루트).
- `show_me_report.py` 단독 실행은 1~8 체인을 한 번에 돌려보는 **빠른 확인용일 뿐**이다(중간 산출물 미저장). 전체 산출물 저장 표준 루트로 쓰지 않는다.

### 1.3 산출 확인
- 작전계획: `data_store/plans/plan_{date}.md`
- SHADOW_OPEN payload: `data_store/shadow_entries/shadow_entries_{date}.json`
- exit observer: `data_store/exit_observer/exit_observer_{date}.md`
- daily review: `data_store/reviews/daily_review_{date}.md`
- SHOW ME 그림/표: `data_store/reports/show_me_{date}.md` + `.png`

### 1.4 안전선 확인 (각 산출물 하단/safety 패널)
- `real_order=false`
- `scheduler_changed=false`
- `sajang_changed=false`
- `paper_open_allowed=false`
- `sell_automation=BLOCKED`

## 2. 계속 금지 (6/8에도, 월요일 결과가 좋아 보여도)

- 실주문 0 / KIS 주문 어댑터 접촉 0
- PAPER_OPEN 열기 금지 (SHADOW_OPEN만)
- scheduler 등록·systemctl 변경 금지
- SAJANG 변경 금지
- show_me 결과로 engine_policy_map/후보 승격/정책 변경 금지
- VPS 배포 별도 승인 전 금지

## 3. 6/12 사후비교 (누적 후)

6/8~6/12 관측을 누적해서 show_me 그림·숫자로 비교한다.
- 후보선정 성능 (as_of 종가 기준): tier별 D+1/3/5/10, missed_winner, false_positive
- SmartEntry 실행 성능 (virtual_entry_price 기준): 진입 타이밍 D+1/3/5/10, MFE/MAE
- exit 룰: 어떤 룰이 손실을 줄였나 / 수익을 빨리 끊었나
- **두 성능을 섞지 않는다** (사과 vs 오렌지).

승격·PAPER_OPEN·scheduler 배선은 6/12 비교 후 **사장님 승인 별건**.

## 4. 한 줄 결론

퀀트봇도 단타봇처럼 이제 개발이 아니라 **관측 단계**다.
단타봇 = A1/A2/B/C 전략 관측, 퀀트봇 = FLOWX 후보선정·진입타이밍·exit 관측.
둘 다 6/8 장 끝나고 **첫 장부·첫 그림**을 보는 게 다음 일이다.
