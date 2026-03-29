[퀀트봇 → 웹봇] CFO/CTO/펀더멘탈 3개 섹션 추가

## 1. 테이블명: quant_jarvis (기존)

새 테이블 없음. 기존 `quant_jarvis.data` JSONB에 3개 키 추가됨.

## 2. 신규 데이터 구조

### `data.cfo` — 포트폴리오 건강 리포트

```json
{
  "generated_at": "2026-03-28 21:40",
  "health_score": 48.5,
  "risk_level": "aggressive",
  "positions_count": 8,
  "cash_ratio": 3.5,
  "max_sector_name": "기타",
  "max_sector_pct": 96.3,
  "var_95": -8.8,
  "warnings": [
    "현금 비중 4% < 권고 20%",
    "기타 섹터 96% 초과 집중",
    "HHI 1.00 — 포트폴리오 극도 집중"
  ],
  "recommendations": [
    "현금 비중을 20% 이상으로 확보 필요",
    "기타 비중 축소 또는 타 섹터 분산 필요"
  ],
  "drawdown_action": "유지",
  "drawdown_pct": 0.0,
  "investable": 0,
  "max_new_invest": 0,
  "regime": "CAUTION"
}
```

**화면 제안**: 성과탭 상단에 카드 형태
- 건강 점수 게이지 (0~100, 색상: <40 빨강, 40~70 노랑, >70 초록)
- 현금비율 / 섹터집중 / VaR 3개 미니 인디케이터
- warnings → 경고 배지 리스트
- drawdown_action → "유지/축소/중단/긴급" 상태 표시

### `data.cto` — 시스템 성과 리포트

```json
{
  "generated_at": "2026-03-28 ...",
  "total_records": 77,
  "source_performance": [
    {
      "source": "매집추적",
      "win_rate": 23.4,
      "avg_return": -2.05,
      "total": 47,
      "decay": false
    },
    {
      "source": "눌림목",
      "win_rate": 41.2,
      "avg_return": -0.08,
      "total": 17,
      "decay": false
    }
  ],
  "decay_alerts": [],
  "data_health_score": 93.3,
  "stale_count": 0,
  "missing_count": 1,
  "suggestions": [
    {
      "action": "REDUCE_WEIGHT",
      "detail": "매집추적: 승률 23.4% — 비중 축소 권고",
      "priority": "HIGH"
    }
  ]
}
```

**화면 제안**: 성과탭 하단 또는 별도 "시스템" 탭
- source_performance → 막대 차트 (소스별 승률 + 수익률)
- decay=true인 소스 → 빨간 경고 아이콘
- data_health_score → 게이지 (93/100)
- suggestions → 카드 리스트 (HIGH=빨강, MEDIUM=노랑, LOW=회색)

### `data.fundamentals` — 펀더멘탈 분석

```json
{
  "earnings": {
    "date": "2026-03-28",
    "total_analyzed": 1003,
    "status_counts": {
      "TURNAROUND_STRONG": 50,
      "TURNAROUND_EARLY": 80,
      "ACCELERATING": 241,
      "DECELERATING": 140,
      "DETERIORATING": 492
    },
    "turnaround_strong": [
      {
        "ticker": "010950", "name": "S-Oil",
        "status": "TURNAROUND_STRONG", "score": 81.0,
        "latest_op_income": 229200000000,
        "prev_op_income": -344000000000,
        "qoq_change": 1.666, "acceleration": 16.635
      }
    ],
    "turnaround_early": [...],
    "accelerating": [...]
  },
  "turnaround": {
    "date": "2026-03-28",
    "total_screened": 1008,
    "candidates_found": 39,
    "strong": [
      {
        "ticker": "096770", "name": "SK이노베이션",
        "turnaround_type": "STRONG", "score": 95,
        "op_income_q1": -417600000000,
        "op_income_latest": 573500000000,
        "debt_ratio": 64.0, "est_turnaround": "2025Q3"
      }
    ],
    "early": [...]
  }
}
```

**화면 제안**: `/quant` 새 탭 "펀더멘탈"
- status_counts → 도넛 차트 (5가지 상태 비율)
- turnaround_strong → 테이블 (적자→흑자 전환 종목, 점수/영업이익 변화)
- turnaround_early → 테이블 (적자 축소 중, 예상 전환 시점)
- accelerating → 테이블 (성장 가속 종목, QoQ/가속도)
- 금액 표시: 억원 단위 변환 필요 (원 단위로 들어옴, /1e8)

## 3. SQL 파일: 없음 (기존 테이블 활용)

## 4. 스케줄: 16:00~17:30 COO 파이프라인 (기존 업로드와 동시)

## 5. 데이터 설명

| 키 | 설명 | 갱신 |
|----|------|------|
| `cfo` | 포트폴리오 건강점수, 현금비율, 경고, 낙폭 조치 | 매일 |
| `cto` | 시그널 소스별 승률, 데이터 건강, 가중치 제안 | 매일 |
| `fundamentals.earnings` | DART 분기 재무제표 기반 실적 가속도 (1003종목) | 매일 |
| `fundamentals.turnaround` | 적자→흑자 턴어라운드 후보 (39종목) | 매일 |

## 6. 프론트 작업 요약

1. **성과탭 확장**: CFO 건강 카드 + CTO 소스 성과 차트
2. **새 탭 "펀더멘탈"**: 턴어라운드 + 실적가속도 테이블
3. API: 기존 `/api/quant` 엔드포인트에서 `data.cfo`, `data.cto`, `data.fundamentals` 꺼내면 됨 (추가 API 불필요)

## 7. 디자인 참고

- 기존 `/quant` 4탭 디자인 그대로 확장
- 건강 점수: 원형 게이지 (tailwind + CSS)
- 금액: `억원` 단위, 양수=초록, 음수=빨강
- 상태 배지: STRONG=emerald, EARLY=amber, ACCELERATING=blue, 나머지=gray
