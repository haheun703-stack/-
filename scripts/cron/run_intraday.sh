#!/bin/bash
# 장중 실시간 학습 엔진 (Phase 12)
# 평일 08:55 시작 → intraday_learner (15:35 자동 종료)
# 평일 15:40 → intraday_pattern_analyzer

set -e

PROJECT_DIR="/home/ubuntu/quantum-master"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR"
PY="$PROJECT_DIR/venv/bin/python3.11"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MODE="${1:-learner}"
TODAY=$(date +%Y-%m-%d)
DOW=$(date +%u)  # 1=월, 7=일

# 주말 가드
if [ "$DOW" -ge 6 ]; then
    echo "[$(date '+%H:%M:%S')] 주말 ($DOW) — 스킵"
    exit 0
fi

case "$MODE" in
    learner)
        # 08:55 시작 — 09:00까지 wait 후 15:35까지 수집
        echo "[$(date '+%H:%M:%S')] intraday_learner 시작 ($TODAY)"
        "$PY" -u -X utf8 "$PROJECT_DIR/scripts/intraday_learner.py" --wait-open \
            >> "$LOG_DIR/intraday_learner_$(date +%Y%m%d).log" 2>&1
        ;;
    analyzer)
        # 15:40 — 당일 분석 + 누적 + 내일 시그널 생성
        echo "[$(date '+%H:%M:%S')] intraday_pattern_analyzer 시작 ($TODAY)"
        "$PY" -u -X utf8 "$PROJECT_DIR/scripts/intraday_pattern_analyzer.py" --date "$TODAY" \
            >> "$LOG_DIR/intraday_analyzer_$(date +%Y%m%d).log" 2>&1
        ;;
    *)
        echo "Usage: $0 {learner|analyzer}"
        exit 1
        ;;
esac

echo "[$(date '+%H:%M:%S')] $MODE 완료"
