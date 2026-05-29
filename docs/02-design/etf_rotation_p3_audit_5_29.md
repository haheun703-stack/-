# ETF Rotation P3 Audit 1차 — 5/29(금)

> **상태**: read-only P3 audit (코드 변경 0건, B-5 결단 자료)
> **계기**: 5/29 사장님 결단 — "scripts/run_etf_rotation.py 별도 P3 audit. 지금 실행/수정 금지"
> **상위 문서**: `docs/02-design/deletion-quarantine-audit-5_29.md` §3-2 + `docs/01-plan/v2-backlog-5_29.md` B-5
> **검수 방식**: Explore 서브에이전트 분업 (run_etf_rotation 376 라인 + src/etf 7 파일 + VPS cron 전수)

---

## 0. 한 줄 결론

> **ETF rotation 전체 (run_etf_rotation.py + src/etf/ 7 파일)에서 broker 직접 호출 0건 / register_intent 0건 / 실제 매매 실행 0건 확정. VPS 실행은 `--blind-test --no-telegram` 관찰 전용. 분류: KEEP. 향후 live 활성화 결단 시 canonical 흡수 PDCA 필요.**

---

## 1. scripts/run_etf_rotation.py 전체 구조 분석

### 1-1. entrypoint
- 파일 376 라인
- L300+ `if __name__ == "__main__": main()` 정상 entrypoint
- CLI 인자:
  - `--dry-run` (관찰만)
  - `--no-telegram` (텔레그램 발송 비활성)
  - `--blind-test` (블라인드 테스트 모드)
  - `--no-ai` (AI 필터 비활성)

### 1-2. import 모듈
- `src.etf.orchestrator.ETFOrchestrator`
- `src.etf.data_bridge` (데이터 로딩)
- `src.agents.brain.Brain` (의사결정 보조)
- `src.utils.telegram_sender` (리포트 발송)
- `src.etf.ai_filter` (AI 필터)
- **broker / KisOrderAdapter / PaperOrderAdapter / register_intent / mode / executor_bot / KILL_SWITCH** : **import 0건**

### 1-3. orders dict 흐름
```
orchestrator.run()
    ↓
result["order_queue"] = list[dict]
    ↓
AI 필터 적용 (텍스트 필터링만)
    ↓
JSON 저장 (data/etf_rotation_result.json)
    ↓
텔레그램 리포트 발송 (텍스트 서식)
    ↓
🛑 broker 호출 지점: 없음
```

### 1-4. order_queue dict 스키마
```python
{
    "axis": "leverage" | "sector" | "index" | "all",
    "code": "ticker",
    "name": "종목명",
    "action": "BUY" | "SELL",
    "target_weight_pct": float,
    "reason": str,
    "priority": int,
}
```
- 매매 신호용 필드 (`code`, `action`, `target_weight_pct`)는 존재
- **그러나 이 dict를 broker로 변환하는 코드 0건**

---

## 2. src/etf/ 7 파일 전수 매트릭스

| 파일 | 함수/클래스 | broker 호출 | register_intent | mode/executor_bot | KILL_SWITCH | 분류 |
|---|---|---|---|---|---|---|
| `orchestrator.py` | ETFOrchestrator.run / decide / _build_order_queue | ❌ | ❌ | ❌ | ❌ | **KEEP** |
| `index_engine.py` | IndexETFEngine._close_all | ❌ | ❌ | ❌ | ❌ | **KEEP** |
| `leverage_engine.py` | LeverageEngine._emergency_close | ❌ | ❌ | ❌ | ❌ | **KEEP** |
| `sector_engine.py` | SectorETFEngine.run | ❌ | ❌ | ❌ | ❌ | **KEEP** |
| `risk_manager.py` | ETFRiskManager.run_checks (KILLSWITCH 결정만) | ❌ | ❌ | ❌ | (decision만) | **KEEP** |
| `data_bridge.py` | load_all / load_momentum | ❌ | ❌ | ❌ | ❌ | **KEEP** |
| `ai_filter.py` | apply_ai_filter | ❌ | ❌ | ❌ | ❌ | **KEEP** |

→ **7 파일 모두 의사결정 dict 생성만**. broker 호출 0건. KILLSWITCH는 risk_manager.py 의사결정 결과 ("이러한 액션 권장")만 표시.

---

## 3. VPS cron + 스케줄러 ETF 참조

### 3-1. VPS 살아있는 cron 21개 + 주석 5개
- `etf` 키워드 grep 결과: 0건 (직접 cron 등록 없음)

### 3-2. BAT 스케줄러 (schedule_D_original.bat / schedule_D_after_close.bat)
- `schedule_D_original.bat:178`: `python -u -X utf8 scripts\run_etf_rotation.py --blind-test --no-telegram >> logs\schedule.log 2>&1`
- → **관찰 전용 실행** (블라인드 + 텔레그램 OFF)

### 3-3. BAT-D 다른 ETF 단계 (관련 8건)
- 8: `sector_etf_builder.py --daily` (ETF 마스터 생성)
- 9c.7: `collect_etf_investor_flow.py` (수급 데이터)
- 9e: `collect_etf_volume.py` (거래량)
- 10: `update_etf_master.py` (마스터 업데이트)
- 11: `etf_trading_signal.py` (신호 생성)
- **11.3: `run_etf_rotation.py --blind-test --no-telegram`** ⭐
- 11.5: `leverage_etf_scanner.py` (레버리지 스캔)
- 20.95: `src.alpha.etf_engine` (추천)

→ 모두 데이터 수집 / 신호 생성 / 의사결정 출력. **broker 직접 호출 0건**.

---

## 4. 4분류 종합 결단

### 4-1. KEEP (전체 7 파일)

| 영역 | 근거 |
|---|---|
| src/etf/* 6 파일 | broker 호출 0건 + register_intent 0건 + 의사결정 dict 생성만 |
| scripts/run_etf_rotation.py | broker 호출 0건 + `--blind-test` 관찰 전용 + JSON + 텔레그램 리포트만 |

### 4-2. KEEP+GUARD / QUARANTINE / MERGE
- **현 단계 해당 없음** (broker 미연결)

### 4-3. 향후 live 활성화 시 MERGE 후보
- 시나리오: 사장님이 향후 ETF 자동매매 결단 시
- 변환 경로 신규 구축 필요:
  1. `order_queue` dict → `register_intent` 호출 → `data/order_intents/quant_intents_YYYY-MM-DD.jsonl` 등록
  2. 이후 매매 트리거 (별도 cron 또는 수동 명령) → `assert_order_intent_exists` 통과 → `KisOrderAdapter.buy_limit(mode='paper'/'live', executor_bot='quant')`
  3. canonical 경로 정확 부합 (signal → intent → gate → adapter)
- 이 PDCA는 별도 차수 (운영 안정화 후)

---

## 5. 1차 audit (5/29 오후) 결론 재확인

### 5-1. 1차 결론
- "ETF 엔진 close_all 경로는 의사결정 dict만 생성 (broker 호출 0건)"
- "scripts/run_etf_rotation.py B-5 별도 P3 audit 권장"

### 5-2. 2차 audit (본 문서) 결론
- ✅ **1차 결론 확정** — broker 호출 0건 변함없음
- ✅ **VPS 실행도 --blind-test 관찰 전용** 확인
- ✅ **7 파일 + 1 스크립트 전수 매트릭스 작성** 완료
- → B-5는 **현재 위험 0건**, 향후 live 활성화 결단 시점에 재진입

---

## 6. 권장 처리 방향

### 6-1. 즉시 행동 (P3)
- 본 audit 결과 v2-backlog (B-5) 갱신
- B-5 우선순위 P3 → **결단 보류** (위험 0건이므로 즉시 처리 불필요)

### 6-2. 향후 live 활성화 시 (별도 PDCA)
1. **Step 1**: canonical order pipeline 설계 (`order_queue` → `order_intent` 매핑 스키마)
2. **Step 2**: 변환 코드 작성 (paper mode 우선, mode/executor_bot 명시)
3. **Step 3**: 회귀 테스트 (paper 경로만)
4. **Step 4**: §9 4단계 dry-run 적용 (별도 ETF 전용)
5. **Step 5**: Codex 검수 PASS
6. **Step 6**: 사장님 별도 승인
7. **Step 7**: ETF paper cron 1줄 등록 (관찰)
8. **Step 8**: 1주 관찰 후 live 결단

### 6-3. 모니터링 권장 (P3)
- ETF 엔진이 향후 broker 호출 코드를 추가하는지 정기 grep 회귀:
  ```bash
  grep -rEn "(buy_limit|sell_limit|buy_market|sell_market|register_intent)\(" src/etf/ scripts/run_etf_rotation.py scripts/*etf*.py
  # 기대: 0건 (현 시점)
  ```

---

## 7. 잔여 위험 (정직 명시)

### 7-1. ETF 엔진이 broker 호출을 향후 추가하는 가능성
- 현 시점 위험 0건
- 추가 시 본 audit 무효화 → 신규 audit + canonical 적용 의무

### 7-2. order_queue dict의 외부 소비자
- JSON 저장 (`data/etf_rotation_result.json`) + 텔레그램 리포트만
- **그 외 외부 코드가 JSON을 읽어 broker로 변환하는지 별도 확인 권장** (P3 추가 검증)

### 7-3. risk_manager KILLSWITCH 의사결정
- `risk_manager.py:152` `"type": crash_rule.get("action", "close_all"), "severity": "KILLSWITCH"` 의사결정만 생성
- 의사결정이 실제 매매로 변환되는 경로 0건 확인됨
- 그러나 실제 ETF 매매 활성화 시 KILL_SWITCH와 의도된 동작 충돌 가능성 → 별도 PDCA 검토

---

## 8. 적용 금지 (본 audit 후)

- ❌ scripts/run_etf_rotation.py 수정 X
- ❌ src/etf/* 7 파일 수정 X
- ❌ ETF 매매 활성화 결단 X
- ❌ 단독 commit X

---

## 9. 표현 룰

### 사용 가능
- "ETF rotation P3 audit 1차 완료"
- "broker 미연결 + 의사결정 전용 구조 확인"
- "향후 live 활성화 결단 시 canonical 흡수 PDCA 필요"

### 사용 금지
- "ETF 안전 완성" X
- "ETF live 진입 가능" X
- "운영 안전 완성" X

---

## 10. 연결 문서
- `docs/02-design/deletion-quarantine-audit-5_29.md` §3-2 (1차 결과)
- `docs/01-plan/v2-backlog-5_29.md` B-5
- `scripts/run_etf_rotation.py` (376 라인)
- `src/etf/orchestrator.py` (`_build_order_queue` L204-254)
- `src/etf/index_engine.py` (`_close_all` L107)
- `src/etf/leverage_engine.py` (`_emergency_close` L258)
- `src/etf/risk_manager.py` (`run_checks` KILLSWITCH 의사결정)
- `scripts/cron/schedule_D_original.bat:178` (BAT 스케줄)
