# DELETE 2건 archive 이동 실행 계획 + rollback 절차 — 5/29(금)

> **상태**: 실행 계획 + rollback 절차 (코드/파일 이동 0건, 사장님 별도 승인 후만)
> **계기**: 5/29 사장님 결단 — "archive 이동 계획과 rollback 문서는 보강 가능"
> **상위 문서**:
> - `docs/02-design/archive_move_precheck_5_29.md` (사전 검증, ARCHIVE_READY 2/2)
> - `docs/01-plan/v2-backlog-5_29.md` B-6, B-7
> - `docs/02-design/quant_commit_bundle_plan_5_29.md` 묶음 D

---

## 0. 한 줄 결론

> **DELETE 2건 archive 이동 명령/순서/검증/rollback 절차 명문화. 적용은 묶음 A/B/C commit 완료 + Codex 회신 + 사장님 묶음 D 별도 승인 후만.**

---

## 1. 이동 대상 (2건)

| # | 원본 경로 | 이동 경로 | 사전 검증 |
|---|---|---|---|
| 1 | `scripts/dashboard.py` | `scripts/archive/deprecated/dashboard.py` | ARCHIVE_READY (외부 import 0, VPS cron 0) |
| 2 | `scripts/one_off/integration_dryrun_5_20.py` | `scripts/archive/orphan_20260529/integration_dryrun_5_20.py` | ARCHIVE_READY (외부 import 0, VPS cron 0) |

---

## 2. 실행 절차 (사장님 묶음 D 승인 후)

### 2-1. Step 1: 사전 상태 백업
```bash
cd "d:/sub-agent-project_퀀트봇"

# 백업 commit hash 기록
git rev-parse HEAD > ops/archive-move-rollback-hash.txt
cat ops/archive-move-rollback-hash.txt
# 예시 출력: abc1234567890

# 회귀 사전 통과 확인
source venv/Scripts/activate
python -u -X utf8 -m pytest tests/test_phase1_paper_trade.py::TestCallerPassesModeExecutorBot tests/test_phase1_paper_trade.py::TestC2_NoSilentReturn tests/test_phase1_paper_trade.py::TestC3_HmacKeyFailFast -q --tb=short
# 기대: 25 passed

python -u -X utf8 tools/quant_preflight.py
# 기대: RESULT: PASS (10/10)
```

### 2-2. Step 2: dashboard.py 이동
```bash
# 1) 디렉토리 확인 (이미 존재)
ls scripts/archive/deprecated/ | head -5

# 2) git mv (이력 보존)
git mv scripts/dashboard.py scripts/archive/deprecated/dashboard.py

# 3) 이동 후 헤더 업데이트 (선택) — 더 자세한 archive 사유 명시
# 현재 헤더: "DEPRECATED — flowx.kr/quant 페이지가 본진."
# 권장 추가:
# "Archived to: scripts/archive/deprecated/dashboard.py on 2026-05-29 (사장님 결단 + Codex 검수)"
# 별도 Edit 또는 명령:
# (이동 후 파일 안에서) sed -i '8a\\n# Archived: 2026-05-29 (사장님 결단)' scripts/archive/deprecated/dashboard.py

# 4) 이동 검증
ls scripts/dashboard.py 2>&1  # 기대: "ls: cannot access"
ls scripts/archive/deprecated/dashboard.py  # 기대: 파일 존재
```

### 2-3. Step 3: integration_dryrun_5_20.py 이동
```bash
# 1) 디렉토리 생성 (없다면)
mkdir -p scripts/archive/orphan_20260529

# 2) git mv (이력 보존)
git mv scripts/one_off/integration_dryrun_5_20.py scripts/archive/orphan_20260529/integration_dryrun_5_20.py

# 3) 이동 검증
ls scripts/one_off/integration_dryrun_5_20.py 2>&1  # 기대: "ls: cannot access"
ls scripts/archive/orphan_20260529/integration_dryrun_5_20.py  # 기대: 파일 존재

# 4) scripts/one_off/ 폴더 잔여 확인
ls scripts/one_off/
# 잔여: fix_hyosung_004800.py (별도 유지)
```

### 2-4. Step 4: 회귀 + preflight 사후 확인
```bash
# 회귀 25/25 재확인
python -u -X utf8 -m pytest tests/test_phase1_paper_trade.py::TestCallerPassesModeExecutorBot tests/test_phase1_paper_trade.py::TestC2_NoSilentReturn tests/test_phase1_paper_trade.py::TestC3_HmacKeyFailFast -q --tb=short
# 기대: 25 passed

# preflight 10/10 재확인
python -u -X utf8 tools/quant_preflight.py
# 기대: RESULT: PASS

# import 검증 (5/29 Critical 봉합 재발 방지)
python -c "from src.use_cases.live_trading import LiveTradingEngine, create_live_engine; print('import OK')"
# 기대: import OK
```

### 2-5. Step 5: git status 확인
```bash
git status
# 기대: 변경 사항:
#   renamed: scripts/dashboard.py -> scripts/archive/deprecated/dashboard.py
#   renamed: scripts/one_off/integration_dryrun_5_20.py -> scripts/archive/orphan_20260529/integration_dryrun_5_20.py
#
# 추적되지 않는 파일:
#   (해당 없음, git mv 사용)
```

### 2-6. Step 6: commit (묶음 D)
```bash
git commit -m "$(cat <<'EOF'
chore(archive): DELETE 2건 archive 이동 (B-6/B-7, 5/29 사장님 결단)

이동 (git mv, 이력 보존):
- scripts/dashboard.py -> scripts/archive/deprecated/dashboard.py
  사유: 5/16 결정 DEPRECATED, flowx.kr/quant 본진 (헤더 명시), sys.exit(0)
- scripts/one_off/integration_dryrun_5_20.py -> scripts/archive/orphan_20260529/
  사유: 5/20 가동 전 1회 시뮬 완료, 가동 9일 누적으로 용도 종료

사전 검증 (docs/02-design/archive_move_precheck_5_29.md):
- 외부 import 0건 / VPS cron 0건 / scheduler 0건 / CI 0건
- 대체 경로 명확 (dashboard: Supabase, dryrun: 가동 완료)

회귀: 25/25 + preflight 10/10 PASS 재확인
연결: docs/01-plan/v2-backlog-5_29.md B-6/B-7
EOF
)"
```

---

## 3. rollback 절차 (사고 시)

### 3-1. 즉시 rollback (commit 직후)
```bash
# 시나리오: commit 후 회귀 실패 발견 또는 사장님 결단으로 원복
git reset --hard $(cat ops/archive-move-rollback-hash.txt)

# 검증
ls scripts/dashboard.py  # 기대: 파일 존재
ls scripts/one_off/integration_dryrun_5_20.py  # 기대: 파일 존재
ls scripts/archive/deprecated/dashboard.py 2>&1  # 기대: 없음
ls scripts/archive/orphan_20260529/integration_dryrun_5_20.py 2>&1  # 기대: 없음
```

### 3-2. Push 후 rollback (원격 반영 후)
```bash
# 시나리오: push 후 사고 발생 (5/29 commit 정책상 사장님 별도 승인 후만 push, 즉 거의 발생 X)
# 단, 만약을 위해 명시:
git revert <archive-move-commit-hash>
# revert commit 생성 → 원본 파일 복구 + 이동 파일 삭제

git status
# 기대: revert commit 1건 추가
```

### 3-3. 부분 rollback (1건만 원복)
```bash
# 시나리오: dashboard.py만 archive 유지, integration_dryrun_5_20.py 복구
git mv scripts/archive/orphan_20260529/integration_dryrun_5_20.py scripts/one_off/integration_dryrun_5_20.py
git commit -m "chore: integration_dryrun_5_20.py archive 원복 (사장님 결단)"
```

### 3-4. rollback 후 안전 확인
- 회귀 25/25 + preflight 10/10 PASS 재확인
- `python -c "import scripts.dashboard"` 시도 → SystemExit(0) 정상 (dashboard.py가 복구된 경우)
- VPS git pull 0건 확인 (VPS는 본 commit 미반영 상태 유지)

---

## 4. 사후 검증 (commit 후)

### 4-1. 즉시 검증 (Step 4와 동일)
- 회귀 25/25 PASS
- preflight 10/10 PASS
- `python -c "from src.use_cases.live_trading import ..."` OK

### 4-2. 1일 후 검증 (선택)
```bash
# VPS는 본 commit 미반영 상태 유지 (사장님 별도 결단 후 git pull)
# 로컬에서:
ls scripts/dashboard.py 2>&1  # 기대: 없음
ls scripts/archive/deprecated/dashboard.py  # 기대: 존재
git log -1 --oneline scripts/archive/deprecated/dashboard.py  # 기대: 이동 commit hash
git log --follow scripts/archive/deprecated/dashboard.py | head  # 기대: 이동 + 원본 작성 commit 모두 표시
```

### 4-3. 1주 후 검증 (선택)
- 운영 중 archive 파일 참조 실패 사례 0건 확인
- VPS git pull 시 동기 (사장님 별도 결단 후만)

---

## 5. VPS 동기 결단 (별도 단계)

### 5-1. VPS git pull 시점 결단
- **기본 정책**: 본 archive 이동 commit은 **로컬에서만 적용**
- VPS pull 결단 조건:
  - 로컬 회귀 25/25 + preflight 10/10 PASS 1일 이상 유지
  - VPS cron이 archive로 이동된 파일을 참조하지 않음 (사전 검증에서 확인됨)
  - 사장님 별도 결단

### 5-2. VPS git pull 시 명령
```bash
ssh -i "..." ubuntu@13.209.153.221 "
  cd /home/ubuntu/quantum-master &&
  git fetch origin &&
  git log HEAD..origin/main --oneline &&
  # 사장님 확인 후 pull
  # git pull origin main
"
```

### 5-3. VPS pull 후 검증
- VPS 살아있는 cron 21개 실행 0 오류
- 매매 cron 6개 정지 상태 유지
- VPS preflight 10/10 PASS

---

## 6. 잠재 위험 + 완화책

### 6-1. dashboard.py와 dashboard_data.py 혼동
- **위험**: 외부 도구/문서에서 `dashboard.py`를 `dashboard_data.py`로 오인하고 archive 이동 → `dashboard_data.py` (루트의 다른 파일)이 깨질 가능성
- **완화**: 이동 명령에 절대 경로 (`scripts/dashboard.py`) 명시 + Step 5에서 git status로 정확한 이동 대상 확인
- **사전 검증**: `archive_move_precheck_5_29.md` §2-4에서 `run_bat.sh:290` 라인이 `dashboard_data.py`임을 명시 (혼동 주의)

### 6-2. one_off 폴더 잔여
- **현재**: `scripts/one_off/` 에 `fix_hyosung_004800.py` 1건 잔여 (별도 유지)
- **위험**: archive 이동 후 폴더가 비어 보이는 사례
- **완화**: Step 3-4에서 `ls scripts/one_off/` 확인 (`fix_hyosung_004800.py` 잔여 정상)

### 6-3. CLAUDE.md 정책 ("scripts/archive 절대 참조·실행·import 금지") 적용 확인
- archive로 이동된 파일은 이후 import/실행 금지 정책 적용
- 만약 누군가가 archive 안 파일을 import 시도 → 정책 위반 + 별도 사고 보고

### 6-4. 단타봇 측 문서 인용 (`docs/to-bodyhunter/`)
- **현재**: `docs/to-bodyhunter/2026-05-18_사장님-룰-EYE-dry-run-공유.md`에서 `integration_dryrun_5_20.py` 파일명 인용
- **완화**: archive 이동 commit 후 단타봇 측에 안내 (별도 통보)
- 단순 인용이라 단타봇 운영 영향 0건

---

## 7. 적용 금지 (본 실행 계획 작성 후)

- ❌ Step 1~6 즉시 실행 X (사장님 묶음 D 별도 승인 전까지)
- ❌ git mv 즉시 실행 X
- ❌ VPS git pull 즉시 실행 X
- ❌ rollback hash 보존 누락 X (사전 백업 의무)
- ❌ 단독 commit X

---

## 8. 표현 룰

### 사용 가능
- "archive 이동 실행 계획 + rollback 절차 명문화 완료"
- "Step 1~6 + rollback 3가지 시나리오 명시"
- "VPS 동기는 별도 단계"
- "적용은 묶음 A/B/C 완료 + Codex 회신 + 사장님 묶음 D 별도 승인 후"

### 사용 금지
- "archive 이동 완료" X (이동 미수행)
- "DELETE 완료" X
- "운영 안전 완성" X

---

## 9. 연결 문서
- `docs/02-design/archive_move_precheck_5_29.md` (사전 검증)
- `docs/02-design/deletion-quarantine-audit-5_29.md` §4-1 (1차 결단)
- `docs/01-plan/v2-backlog-5_29.md` B-6, B-7
- `docs/02-design/quant_commit_bundle_plan_5_29.md` 묶음 D
- `scripts/dashboard.py` (이동 대상 1)
- `scripts/one_off/integration_dryrun_5_20.py` (이동 대상 2)
- `scripts/archive/deprecated/` (이동 위치 1)
- `scripts/archive/orphan_20260529/` (이동 위치 2 — 폴더 신규 생성 필요)
- `CLAUDE.md` (scripts/archive 참조 금지 정책)
