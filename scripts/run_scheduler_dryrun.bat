@echo off
chcp 65001 >nul
title Quantum Master v5.0 — Dry Run
color 0E

cd /d D:\sub-agent-project
call venv\Scripts\activate.bat

echo.
echo  ====================================================
echo   Quantum Master v5.0 — Dry Run (스케줄 확인)
echo  ====================================================
echo.

python -u -X utf8 scripts\daily_scheduler.py --dry-run

echo.
echo  ────────────────────────────────────────
echo  Phase 즉시 실행 예시:
echo    python scripts\daily_scheduler.py --run-now 3b    (장전 브리핑)
echo    python scripts\daily_scheduler.py --run-now snap1 (수급 1차)
echo    python scripts\daily_scheduler.py --run-now 10b   (장마감 리포트)
echo  ────────────────────────────────────────
echo.
pause
