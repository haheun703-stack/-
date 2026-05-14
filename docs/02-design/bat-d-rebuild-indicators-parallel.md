# B3 rebuild_indicators 병렬화 — Design

**작성**: 2026-05-14
**참조**: [docs/01-plan/bat-d-time-optimization.md](../01-plan/bat-d-time-optimization.md) B3 항목
**목표**: rebuild_indicators 10분 40초 → 5분 30초 (-5분 10초)

## 1. 현재 상태

### 코드 진단 ([src/indicators.py](../../src/indicators.py))

```python
# L12: 이미 multiprocessing import 되어 있음
from multiprocessing import Pool

# L26-56: 워커 함수 정의됨
def _mp_process_stock(args):
    """Worker: 단일 종목 지표 계산 (multiprocessing)"""
    ...

# L829-876: process_all 함수
def process_all(self) -> int:
    """raw 디렉토리의 모든 parquet을 처리하여 processed에 저장 (멀티프로세싱)"""
    raw_files = sorted(self.raw_dir.glob("*.parquet"))
    ...
    # L840: ⚠️ 문제 지점
    num_workers = min(8, max(1, (os.cpu_count() or 4) - 1))
    ...
    with Pool(num_workers) as pool:
        for ticker, success, msg in tqdm(...):
            ...
```

### 실측 데이터 (VPS Lightsail Bodyhunter-60GB, 2026-05-14 BAT-D)

| 항목 | 값 |
|------|---|
| VPS CPU | **2 vCPU** |
| `os.cpu_count() - 1` | **1** |
| `num_workers` 계산 결과 | `min(8, max(1, 1))` = **1** |
| 실제 동작 worker | **1개 (= 사실상 sequential)** |
| 처리 대상 parquet | 1186개 |
| 실측 소요 시간 | 10분 40초 (17:20:04 ~ 17:30:44) |
| 종목당 평균 처리 시간 | ~0.54초 |
| VPS RAM | 1.9GB (284MB 사용 중, 14%) |
| VPS Swap | 5GB (184MB 사용, 여유 충분) |

## 2. 근본 원인

`max(1, (os.cpu_count() or 4) - 1)` 공식은 **CPU 부하를 1코어 남기는 보수적 설계**.
하지만 2 vCPU 시스템에서는 `2-1=1` → 단일 워커가 됨.

→ **multiprocessing.Pool 오버헤드만 발생하고 병렬 효과 0**.

## 3. 변경 제안

### Option A: 단순 공식 수정 (권장, 최소 위험)

```python
# Before (L840)
num_workers = min(8, max(1, (os.cpu_count() or 4) - 1))

# After
num_workers = int(os.getenv("INDICATORS_WORKERS", min(8, os.cpu_count() or 2)))
```

**효과**:
- VPS (2 vCPU): 1 → 2 worker (이론 -50%)
- 로컬 (8+ vCPU): 7 → 8 worker (변화 미미)
- 환경변수 override 가능 (테스트/디버깅 유연)

**위험**: 매우 낮음. multiprocessing 로직은 이미 검증됨, worker 수만 증가.

### Option B: 청크 크기 + imap_unordered 도입 (Option A 이후 검토)

```python
# 현재 Pool.imap()는 기본 chunk_size=1 → 매번 IPC 오버헤드
with Pool(num_workers) as pool:
    for result in pool.imap_unordered(_mp_process_stock, args_list, chunksize=10):
        ...
```

**효과**: IPC 오버헤드 감소 (-10~20% 추정)
**위험**: tqdm progress 정확도 영향 (순서 보장 안 됨), 큰 문제 아님

## 4. 코드 패치 (Option A 단일 변경)

### 변경 파일
- `src/indicators.py` L840 (1줄 변경)

### Before
```python
num_workers = min(8, max(1, (os.cpu_count() or 4) - 1))
```

### After
```python
num_workers = int(os.getenv("INDICATORS_WORKERS", min(8, os.cpu_count() or 2)))
```

### 추가 권장 (로그 가시성)
L843 부근에서 `num_workers` 값을 명시적으로 logger.info로 출력 (이미 출력 중이라면 그대로):
```python
logger.info(f"IndicatorEngine.process_all: {len(raw_files)}개 파일, {num_workers} workers (multiprocessing)")
```

## 5. 검증 방법

### A/B 비교 측정

| 시점 | 측정 | 기준 |
|------|------|------|
| **Before** | 2026-05-14 BAT-D rebuild_indicators | 10분 40초 |
| **After (변경 직후 수동 실행)** | `python scripts/rebuild_indicators.py` | < 6분 목표 |
| **After (5/15 BAT-D 자동 실행)** | cron 로그 17:20:04 ~ ?? 측정 | < 6분 |

### 결과 무결성 검증

- `data/processed/*.parquet` 파일 수: 1186개 유지
- 임의 종목 5개 sample (삼성전자 등): 컬럼 수, 행 수, 마지막 종가 등 변경 전후 동일
- BAT-D 후속 단계 (`run_ict_levels.py`, `run_relay_engine.py` 등) 정상 동작

### CPU 사용량 모니터링

- 변경 후 첫 실행 시 `top` 또는 `htop`으로 동시 워커 수 확인
- 2 워커 모두 80%+ CPU 사용해야 정상
- 메모리 spike 모니터링 (1GB 이하 유지)

## 6. 위험 및 가드레일

| 위험 | 대응 |
|------|------|
| 2 vCPU 100% 사용 → 동시 다른 cron 작업 영향 | BAT-D는 순차 실행이라 동시 작업 없음, OK |
| 워커 메모리 합산 → OOM | 현재 1워커 ~150MB 추정 × 2 = 300MB << 1.9GB, 안전 |
| Pool 종료 시 좀비 프로세스 | `with Pool(...)` 컨텍스트 매니저 사용 중, 자동 정리 |
| 결과 무결성 (parquet 동시 쓰기) | 각 워커가 다른 종목 파일 쓰므로 충돌 없음 |
| 변경 후 BAT-D 실패 1건 | 즉시 롤백 (`git revert`) — 1줄 변경이라 안전 |

## 7. 롤백 계획

```bash
# 변경 커밋
git revert <commit-hash>

# 또는 환경변수로 강제 1워커 (긴급)
INDICATORS_WORKERS=1 python scripts/rebuild_indicators.py
```

환경변수 override가 있어 코드 변경 없이도 1 워커로 복귀 가능.

## 8. 예상 효과

| 단계 | 값 |
|------|---|
| 현재 | 10분 40초 |
| 1→2 worker (Option A) | ~5분 30초 (이론 -50%) |
| + chunk_size 10 (Option B) | ~4분 30초 (추가 -15%) |

**보수 추정**: -5분
**낙관 추정**: -6분 10초

## 9. 다음 단계

1. **즉시 (P0)**: Option A 1줄 패치 적용 + 로컬 dry-run 측정
2. **검증 (P0)**: VPS 배포 → 5/15 BAT-D 자동 실행 결과 확인 (10분→? 측정)
3. **확장 (P1)**: 효과 만족 시 다른 single-thread 스크립트(B2, B5)에도 동일 패턴 적용
4. **고도화 (P2)**: Option B (chunk_size) 적용 검토

---

**현재 단계**: Design 완료, **다음 → 코드 패치 + 로컬 dry-run**
