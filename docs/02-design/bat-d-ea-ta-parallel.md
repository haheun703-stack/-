# B2 [EA]+[TA] 병렬화 — Design

**작성**: 2026-05-14
**참조**: [docs/01-plan/bat-d-time-optimization.md](../01-plan/bat-d-time-optimization.md) B2 항목
**목표**: scan_earnings_acceleration + scan_turnaround 11분 6초 → **약 5분 33초** (-5분 33초)

## 1. 현재 상태

`run_bat.sh` BAT-D L254-255:

```bash
run_py scripts/scan_earnings_acceleration.py    # [EA] 17:56:39 ~ ?
run_py scripts/scan_turnaround.py                # [TA] ? ~ 18:07:45
```

두 스크립트가 **순차 실행**되어 합산 11분 6초 소요.

## 2. 독립성 검증

### 파일 의존성

| 스크립트 | 입력 파일 (read) | 출력 파일 (write) |
|---------|----------------|-----------------|
| `scan_earnings_acceleration.py` | `data/v2_migration/financial_quarterly.json` | `data/earnings_acceleration.json` |
| (공통) | `data/universe/name_map.json` | - |
| `scan_turnaround.py` | `data/v2_migration/financial_quarterly.json` | `data/turnaround_candidates.json` |

**검증 결과**:
- ✅ **입력은 동일 파일이지만 두 스크립트 모두 read-only** → 충돌 없음
- ✅ **출력은 다른 파일** → 동시 쓰기 충돌 없음
- ✅ **외부 자원(API, DB) 사용 없음** → 외부 rate limit 충돌 없음
- ✅ **로거 prefix 분리** (`[EA]` vs `[TA]`) → 로그 동시 출력 가능

### 후속 단계 영향

`build_killer_picks.py`(L258)와 `run_cto.py`(L259)가 두 출력 파일을 모두 읽을 가능성. **두 파일이 모두 생성된 후** 다음 단계로 진행되어야 함 → `wait` 명령 필수.

## 3. 변경 제안

### Option A: run_bat.sh `&` + `wait` (권장)

```bash
# Before
run_py scripts/scan_earnings_acceleration.py
run_py scripts/scan_turnaround.py

# After (병렬 실행 + 동기화)
run_py scripts/scan_earnings_acceleration.py &
EA_PID=$!
run_py scripts/scan_turnaround.py &
TA_PID=$!
wait $EA_PID $TA_PID
```

**효과**:
- 두 스크립트 동시 실행
- 둘 다 끝날 때까지 다음 단계 대기 (build_killer_picks 안전)
- VPS 2 vCPU에서 한 코어씩 사용 → CPU 자원 효율적 사용

### Option B: Python wrapper로 multiprocessing (불필요)

- 더 복잡, 큰 이득 없음
- 실패 처리/로그 분리가 더 어려움
- **선택 안 함**

## 4. 코드 패치

### 변경 파일
- `scripts/cron/run_bat.sh` L254-255

### Before
```bash
    run_py scripts/track_pick_results.py
    run_py scripts/scan_earnings_acceleration.py
    run_py scripts/scan_turnaround.py
    run_py_long scripts/scan_tomorrow_picks.py
```

### After
```bash
    run_py scripts/track_pick_results.py
    # [EA]+[TA] 병렬 실행 (독립적, 입력=read-only/출력=분리)
    run_py scripts/scan_earnings_acceleration.py &
    EA_PID=$!
    run_py scripts/scan_turnaround.py &
    TA_PID=$!
    wait $EA_PID $TA_PID
    run_py_long scripts/scan_tomorrow_picks.py
```

### 주의: run_py 함수 호환성

`run_bat.sh` L21에서 `run_py()`가 bash 함수로 정의됨. 함수 백그라운드 실행 가능:
- `func &` 는 bash에서 함수를 백그라운드 subshell로 실행
- 실패 카운터(`run_py` 내부에서 사용)는 subshell 변수라 부모에 반영 안 됨 → **실패 추적 영향 검토 필요**

**해결**: run_py 내부에서 실패 시 stderr 로그 출력 + exit code 사용. 부모는 `wait $PID; status=$?`로 받기.

## 5. 변경된 run_bat.sh 안전 패치 (실패 추적 보존)

```bash
    run_py scripts/track_pick_results.py
    # [EA]+[TA] 병렬 실행 (B2 최적화)
    {
        run_py scripts/scan_earnings_acceleration.py
        echo "EA_EXIT=$?" > /tmp/ea_exit
    } &
    EA_PID=$!
    {
        run_py scripts/scan_turnaround.py
        echo "TA_EXIT=$?" > /tmp/ta_exit
    } &
    TA_PID=$!
    wait $EA_PID $TA_PID
    # 실패 검사 (선택)
    if [ -f /tmp/ea_exit ]; then source /tmp/ea_exit; rm /tmp/ea_exit; fi
    if [ -f /tmp/ta_exit ]; then source /tmp/ta_exit; rm /tmp/ta_exit; fi
    if [ "${EA_EXIT:-0}" != "0" ] || [ "${TA_EXIT:-0}" != "0" ]; then
        echo "[BAT-D] [EA]/[TA] 병렬 실행 중 실패: EA=$EA_EXIT TA=$TA_EXIT"
    fi
    run_py_long scripts/scan_tomorrow_picks.py
```

**단순화 (실패 추적 불필요 시)**:

```bash
    run_py scripts/scan_earnings_acceleration.py &
    run_py scripts/scan_turnaround.py &
    wait
```

## 6. 예상 효과

| 시나리오 | 시간 | 단축 |
|---------|------|------|
| 현재 (순차) | 11분 6초 | - |
| 병렬 (두 스크립트 비슷한 시간 가정) | ~5분 33초 | -5분 33초 (-50%) |
| 병렬 (한쪽이 더 길면) | max(EA, TA) | -3~5분 |

**보수 추정**: -3분 30초 (한쪽이 약간 더 긴 경우)
**낙관 추정**: -5분 33초 (이론치)

VPS 2 vCPU 동시 사용 → CPU 100% 사용 가능. 메모리 사용 ~2x (각 스크립트 ~150MB × 2 = 300MB), 1.9GB 여유 충분.

## 7. 위험 및 가드레일

| 위험 | 대응 |
|------|------|
| 두 스크립트가 같은 파일 read 시 IO 경합 | JSON 파일 작음(~수십 MB), pandas는 메모리에 한 번에 로드 → 영향 미미 |
| 동시 RAM 사용 OOM | 300MB × 2 = 600MB << 1.9GB, 안전 |
| 부분 실패 시 후속 단계 영향 | `wait` 후 종료 코드 검사 + 실패 시 로그 |
| `run_py` 함수 백그라운드 호환성 | 별도 subshell `{}` 블록으로 격리 |
| build_killer_picks가 한쪽 출력만 보고 진행 | `wait`가 두 PID 모두 기다리므로 안전 |

## 8. 검증 방법

### A/B 비교
- **Before**: 5/14 BAT-D 17:56:39 ~ 18:07:45 = 11분 6초
- **After (5/15부터)**: cron 로그에서 EA 시작 ~ TA 완료 시점 측정
- **목표**: < 7분

### 결과 무결성
- `data/earnings_acceleration.json`: 변경 전후 종목 수 일치
- `data/turnaround_candidates.json`: 변경 전후 종목 수 일치
- build_killer_picks 결과(`data/killer_picks.json`) 변경 전후 비교

### VPS CPU 사용량
- BAT-D 17:56~18:00 시점 `top` 또는 sysstat
- 두 Python 프로세스 동시 80%+ CPU 사용해야 정상

## 9. 롤백 계획

```bash
# 빠른 롤백 (순차로 복구)
git revert <commit-hash>
```

bash 패치라 코드 변경 작음 (5줄 정도), revert 안전.

## 10. 단계

1. **즉시 (P1)**: run_bat.sh 패치 + 단순 테스트 (테스트 환경에서 두 스크립트 단순 병렬 실행)
2. **검증 (P1)**: VPS 배포 → 5/15 BAT-D 자동 실행 결과 확인
3. **무결성 검증 (P1)**: 결과 JSON 파일 변경 전후 일치 확인

---

**현재 단계**: Design 완료, **다음 → run_bat.sh 패치 + 커밋**
