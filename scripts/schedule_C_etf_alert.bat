@echo off
REM ============================================================
REM  [폐지] BAT-C: ETF 시그널 → BAT-B 아침 통합 브리핑에 흡수 (2026-02-28)
REM  스케줄 삭제: schtasks /delete /tn "QM_C_ETFAlert" /f
REM ============================================================

echo [%date% %time%] BAT-C 스킵 (BAT-B에 통합됨) >> D:\sub-agent-project\logs\schedule.log
