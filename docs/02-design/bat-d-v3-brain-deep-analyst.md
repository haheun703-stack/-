# B1 v3 AI Brain — Deep Analyst 동시성 증대 (Design)

**작성**: 2026-05-14
**참조**: [docs/01-plan/bat-d-time-optimization.md](../01-plan/bat-d-time-optimization.md) B1 항목
**목표**: Phase 3+4 (Deep Analyst) 3분 37초 → 약 2분 10초 (-1분 27초)

## 1. v3_brain Phase 의존성 분석

```
Phase 0 (o1) → Phase 1 (Strategic) → Phase 2 (Sector) → Phase 3+4 (Deep) → Phase 5 (Portfolio) → Verifier
```

**결론**: 순수 직렬 의존성. Phase 단위 병렬화는 불가.

다만 **Phase 4 (Deep Analyst) 내부에서 종목 단위 비동기 병렬 처리** 중:
- `src/agents/deep_analyst.py` L166-216
- `asyncio.Semaphore(max_concurrent)` + `asyncio.gather` 사용
- 기본 `max_concurrent=3` (하드코딩)

## 2. 현황 측정

| 항목 | 값 |
|------|---|
| 5/14 실측 Phase 3+4 | 17:36:58 ~ 17:40:35 = **3분 37초** (217초) |
| 후보 종목 수 | 26종목 |
| 평균 종목당 시간 | 8.4초 |
| 현재 max_concurrent | 3 |
| 이론 최소 (3 동시) | 26 / 3 × 8.4 = 73초 |
| 실측 / 이론 = 차이 | 217 / 73 = **3배** (오버헤드 큼) |

→ 차트 렌더링, 컨텍스트 빌드 등 추가 시간. 동시성 증대만으로 비례 단축은 어렵지만 효과 있음.

## 3. Anthropic API rate limit 고려

| Tier | Sonnet 4 RPM | TPM |
|------|------|-----|
| Tier 2 (예상) | 1000 RPM | 80K TPM |

- 26종목 / 60초 = 0.43 RPS = 26 RPM << 1000 RPM
- `max_concurrent=5` 또는 8까지 안전

## 4. 변경 제안

### src/agents/deep_analyst.py

**Before (L172)**:
```python
max_concurrent: int = 3,
```

**After**:
```python
max_concurrent: int = int(os.getenv("DEEP_ANALYST_CONCURRENT", "5")),
```

**추가**: L17~22의 import에 `import os` 추가.

## 5. 예상 효과

| max_concurrent | Phase 4 예상 시간 | 단축 |
|---------------|----------------|-----|
| 3 (현재) | 217초 | - |
| 5 | ~150초 | -1분 7초 |
| 8 | ~110초 | -1분 47초 |

**보수 추정 (5 worker)**: -1~2분
**v3_brain 전체 (7분 55초)**: 6분대 진입

## 6. 위험 및 가드레일

| 위험 | 대응 |
|------|------|
| Anthropic API rate limit 초과 | 5 worker는 매우 안전 범위 (26 RPM << 1000 RPM) |
| 비용 증가 | 동시성과 무관, 총 토큰 동일 |
| Phase 5 (Portfolio) 의존성 영향 | Phase 4 결과 형식 동일, 영향 없음 |
| 부분 실패 (1~2 종목) | `asyncio.gather(return_exceptions=True)` 이미 처리 중 |
| 긴급 롤백 | `DEEP_ANALYST_CONCURRENT=3` 환경변수 |

## 7. 검증 방법

- **A/B 비교**: 5/14 Phase 3+4 = 3분 37초 → 5/15부터 < 3분 목표
- **결과 무결성**: `ai_v3_picks.json` 변경 전후 conviction 점수 분포 일치
- **API 응답 에러**: cron 로그에서 "Anthropic API timeout" 검색, 0건 유지

## 8. 다음 후속 작업 (별개)

v3_brain 추가 단축 후보:
- **Phase 1 (Strategic Brain)** 1분 38초 — Sonnet+Opus Advisor 호출. 모델 다운그레이드 검토
- **Phase 2 (Sector Strategist)** 28초 — 이미 빠름
- **Phase 5 (Portfolio Brain)** 1분 5초 — 6 후보 종목, 동시성 검토
- **Verifier (Perplexity)** 별도 API — 시간 측정 필요

총 v3_brain 추가 단축 잠재력: -2~3분 (이번 design 외)

---

**현재 단계**: Design 완료 + 패치 적용, **다음 → 커밋 + VPS 배포**
