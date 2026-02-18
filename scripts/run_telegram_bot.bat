@echo off
chcp 65001 >nul
title Quantum Master — Telegram Bot
color 0B

cd /d D:\sub-agent-project
call venv\Scripts\activate.bat

echo.
echo  ====================================================
echo   Quantum Master — 텔레그램 명령 봇 (독립 실행)
echo  ====================================================
echo.
echo  텔레그램에서 /ping 을 보내 연결을 확인하세요
echo  종료: Ctrl+C
echo  ────────────────────────────────────────
echo.

python -u -X utf8 src\telegram_command_handler.py

echo.
echo  [%date% %time%] 봇 종료됨
echo.
pause
