@echo off
chcp 65001 >nul
title Quantum Master v5.0 Scheduler
color 0A

echo.
echo  ====================================================
echo   Quantum Master v5.0 Daily Scheduler
echo   한국장 준비 ~ 미장 마감 자동 매매 시스템
echo  ====================================================
echo.
echo  [%date% %time%] 스케줄러 시작
echo.

REM 프로젝트 디렉토리 이동
cd /d D:\sub-agent-project

REM 가상환경 활성화
call venv\Scripts\activate.bat

REM 로그 디렉토리 확인
if not exist logs mkdir logs

REM 스케줄러 시작 (로그 동시 출력)
echo  모드: --dry-run 없이 실제 스케줄러 실행
echo  종료: Ctrl+C
echo  ────────────────────────────────────────
echo.

python -u -X utf8 scripts\daily_scheduler.py 2>&1 | tee logs\scheduler_%date:~0,4%%date:~5,2%%date:~8,2%.log

echo.
echo  [%date% %time%] 스케줄러 종료됨
pause
