# DELETE 2건 archive 이동 사전 검증 — 5/29(금)

> **상태**: 사전 검증 (코드/파일 이동 0건, 사장님 별도 승인 후만)
> **계기**: 5/29 사장님 결단 — "이동 전 깨질 것 없는지 증거만 더 쌓는다"
> **상위 문서**: `docs/02-design/deletion-quarantine-audit-5_29.md` §4-1 + `docs/01-plan/v2-backlog-5_29.md` B-6/B-7

---

## 0. 한 줄 결론

> **2건 모두 ARCHIVE_READY. 외부 import 0건 / VPS cron 참조 0건 / CI 참조 0건 / 대체 경로 명확. archive 이동 시 운영 영향 0건 예상.**

---

## 1. 검증 대상 (2건)

| # | 대상 | 이동 위치 후보 |
|---|---|---|
| 1 | `scripts/dashboard.py` | `scripts/archive/deprecated/dashboard.py` |
| 2 | `scripts/one_off/integration_dryrun_5_20.py` | `scripts/archive/orphan_20260529/integration_dryrun_5_20.py` |

---

## 2. 검증 항목 6종 (대상 1: `scripts/dashboard.py`)

### 2-1. repo import/reference
- `from scripts.dashboard` / `import scripts.dashboard` / `scripts.dashboard import` 패턴 grep: **0건**
- `scripts/dashboard.py` 문자열 grep: **0건** (audit 산출물 1건만 — 본 검증 문서 자체)
- **검증 결과**: PASS

### 2-2. CLI 문서 reference
- `README.md` / `docs/SYSTEM_MAP.md` 등 grep: **0건** (별도 검색)
- **검증 결과**: PASS

### 2-3. VPS cron reference
- VPS `crontab -l | grep -i dashboard`: **0건** (Bash 결과)
- **검증 결과**: PASS

### 2-4. scheduler reference (BAT/sh)
- `scripts/cron/run_bat.sh` L290: `# dashboard_data.py 단독실행 제거 — upload_flowx.py 내부 import로 통합`
  - **★ 주의**: 이 라인은 `dashboard_data.py` (루트의 다른 파일)을 가리킴. **`scripts/dashboard.py`와 무관**.
- `scripts/*.bat` grep: **0건**
- **검증 결과**: PASS (혼동 주의 사항 명시)

### 2-5. CI/test reference
- `tests/` 폴더 grep: **0건**
- `.github/workflows/` 또는 CI 설정: **존재하지 않음** (단일 개발자 환경)
- **검증 결과**: PASS

### 2-6. README/docs reference
- `docs/` 전체 grep: **1건** (`docs/02-design/deletion-quarantine-audit-5_29.md` — 본 audit 자체)
- 헤더 자체에 "DEPRECATED" 명시 + `sys.exit(0)` 코드 (즉시 종료)
- **검증 결과**: PASS

### 2-7. 대체 경로
- 모든 매매/통계 → `FlowxUploader` → Supabase → `https://www.flowx.kr/quant` (헤더 docstring 명시)
- 별도 운영 영향 0건

### 2-8. archive 이동 후 예상 영향
- 운영 영향: **0건** (외부 import 0 + cron 0 + scheduler 0)
- git 히스토리: 보존 (`git log --follow scripts/archive/deprecated/dashboard.py`)
- 향후 복구 필요 시: `git mv scripts/archive/deprecated/dashboard.py scripts/dashboard.py`
- **판정**: **ARCHIVE_READY**

---

## 3. 검증 항목 6종 (대상 2: `scripts/one_off/integration_dryrun_5_20.py`)

### 3-1. repo import/reference
- `from scripts.one_off.integration_dryrun_5_20` 패턴 grep: **0건**
- `integration_dryrun_5_20` 문자열 grep 결과 3건:
  1. `scripts/one_off/integration_dryrun_5_20.py` — **자기 자신**
  2. `docs/to-bodyhunter/2026-05-18_사장님-룰-EYE-dry-run-공유.md` — 단타봇 공유 문서 (파일명 인용만, 코드 import 아님)
  3. `docs/01-plan/v2-backlog-5_29.md` — 본 audit B-7 항목 (산출물)
- **외부 코드 import**: 0건
- **검증 결과**: PASS

### 3-2. CLI 문서 reference
- 사용 안내 (헤더 docstring): `source venv/Scripts/activate; python -u -X utf8 scripts/one_off/integration_dryrun_5_20.py` — 1회 시뮬용 명시
- 별도 CLI 문서: 0건
- **검증 결과**: PASS

### 3-3. VPS cron reference
- VPS `crontab -l | grep -i integration_dryrun`: **0건**
- **검증 결과**: PASS

### 3-4. scheduler reference (BAT/sh)
- `scripts/cron/run_bat.sh` grep: **0건**
- `scripts/*.bat` grep: **0건**
- **검증 결과**: PASS

### 3-5. CI/test reference
- `tests/test_no_raw_mojito_order_bypass.py` grep: **0건** (확인 완료)
- 그 외 tests/: **0건**
- **검증 결과**: PASS

### 3-6. README/docs reference
- `docs/` grep: 2건 (단타봇 공유 + audit 산출물) — 모두 인용/등록 메타데이터, 코드 의존 0
- 헤더 명시: "5/20 자동매매 통합 흐름 dry-run 시뮬레이션 (2026-05-18 작성, 1회성)"
- **검증 결과**: PASS

### 3-7. 대체 경로
- 5/20 가동 통합 시뮬 = 1회 검증 완료 (가동 9일 누적 = 검증 종료)
- 향후 유사 dry-run 필요 시: 본 audit B-10 (시간 의존 fixture 동적화 PDCA)에서 별도 작성 권장

### 3-8. archive 이동 후 예상 영향
- 운영 영향: **0건**
- git 히스토리: 보존
- 향후 참고 필요 시: archive 폴더에서 git show
- **판정**: **ARCHIVE_READY**

---

## 4. 판정 종합표

| # | 대상 | repo | 문서 | VPS cron | scheduler | CI | README/docs | 대체 경로 | 영향 | **최종** |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | dashboard.py | PASS | PASS | PASS | PASS (혼동 주의) | PASS | PASS | flowx.kr/quant | 0건 | **ARCHIVE_READY** |
| 2 | integration_dryrun_5_20.py | PASS | PASS | PASS | PASS | PASS | PASS | 1회 시뮬 종료 | 0건 | **ARCHIVE_READY** |

---

## 5. 이동 실행 시 권장 절차 (사장님 별도 승인 후)

### 5-1. dashboard.py 이동
```bash
cd "d:/sub-agent-project_퀀트봇"
git mv scripts/dashboard.py scripts/archive/deprecated/dashboard.py
# 이동 후 헤더 docstring에 다음 라인 추가 권장:
# "Moved to archive: 2026-05-29 (사장님 결단 + Codex 회신 후 별도 승인)"
# 회귀 25/25 + preflight 10/10 PASS 재확인
```

### 5-2. integration_dryrun_5_20.py 이동
```bash
# scripts/archive/orphan_20260529/ 폴더 생성
mkdir -p scripts/archive/orphan_20260529
git mv scripts/one_off/integration_dryrun_5_20.py scripts/archive/orphan_20260529/integration_dryrun_5_20.py
# 회귀 25/25 + preflight 10/10 PASS 재확인
```

### 5-3. commit (사장님 별도 승인 후)
- 묶음 C 일부로 commit 권장 (audit 문서 + archive 이동)
- 또는 별도 commit (audit 결과 적용 단일)

---

## 6. 잔여 위험 (정직 명시)

### 6-1. dashboard.py 혼동 위험
- `dashboard_data.py` (루트 별도 파일) 와 `scripts/dashboard.py` 혼동 가능
- `dashboard_data.py`는 `scripts/upload_flowx.py`에서 import 사용 중 (`run_bat.sh:290` 주석 참조)
- **archive 이동 대상은 `scripts/dashboard.py`만** (루트 `dashboard_data.py`는 KEEP)

### 6-2. 단타봇 공유 문서 인용
- `docs/to-bodyhunter/2026-05-18_사장님-룰-EYE-dry-run-공유.md`에 파일명 등장
- 단순 파일명 인용 (코드 의존 0) → archive 이동해도 문서 무효화 0건
- 단, 단타봇 운영진이 본 audit 결과 통보 필요 (이동 후 안내 권장)

---

## 7. 적용 금지 (본 검증 후)

- ❌ 즉시 archive 이동 X (Codex 회신 + 사장님 별도 승인 후만)
- ❌ git mv 즉시 실행 X
- ❌ 본 검증 결과 단독 commit X

---

## 8. 표현 룰

### 사용 가능
- "DELETE 2건 ARCHIVE_READY 판정 1차 완료"
- "외부 import + VPS cron + CI 참조 0건 확인"
- "운영 영향 0건 예상"

### 사용 금지
- "archive 이동 완료" X (이동 미수행)
- "DELETE 완료" X
- "운영 안전 완성" X

---

## 9. 연결 문서
- `docs/02-design/deletion-quarantine-audit-5_29.md` §4-1
- `docs/01-plan/v2-backlog-5_29.md` B-6, B-7
- `scripts/dashboard.py` (이동 대상)
- `scripts/one_off/integration_dryrun_5_20.py` (이동 대상)
- `scripts/cron/run_bat.sh:290` (혼동 주의 라인 — `dashboard_data.py`)
