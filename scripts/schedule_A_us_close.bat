@echo off
REM ============================================================
REM  Quantum Master - BAT-A: 미장 마감 + 아침 재스캔 + 텔레그램
REM  스케줄: 매일 06:10 (월~토, 미장 마감 직후)
REM  등록: schtasks /create /tn "QM_A_USClose" /tr "D:\sub-agent-project\scripts\schedule_A_us_close.bat" /sc daily /st 06:10
REM
REM  [v2] 미장 데이터 반영 후 추천종목 재스캔 + 텔레그램 발송
REM       → BAT-E(08:50)가 최신 추천으로 자동매수
REM ============================================================

echo [%date% %time%] BAT-A 시작: 미장 마감 + 아침 재스캔 >> D:\sub-agent-project\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

REM ── PHASE 1: 미국장 데이터 업데이트 ──

REM 1) US 시장 데이터 업데이트 + Overnight Signal (원자재 포함)
echo [%date% %time%] [1/5] US Overnight Signal 업데이트 >> logs\schedule.log
python -u -X utf8 scripts\us_overnight_signal.py --update >> logs\schedule.log 2>&1

REM 2) US-KR 패턴DB 일일 누적
echo [%date% %time%] [2/5] US-KR 패턴DB 업데이트 >> logs\schedule.log
python -u -X utf8 scripts\update_us_kr_daily.py >> logs\schedule.log 2>&1

REM ── PHASE 2: 아침 재스캔 (미국장 반영) ──

REM 3) v3 AI Brain 재실행 (미장 데이터 반영 → ai_v3_picks.json 갱신)
echo [%date% %time%] [3/5] v3 AI Brain 아침 재스캔 >> logs\schedule.log
python -u -X utf8 scripts\run_v3_brain.py --no-telegram >> logs\schedule.log 2>&1
if errorlevel 1 echo [%date% %time%] v3 Brain 실패 (기존 picks 유지) >> logs\schedule.log

REM 4) 추천종목 재스캔 (overnight_signal + v3 picks 반영)
echo [%date% %time%] [4/5] 추천종목 아침 재스캔 >> logs\schedule.log
python -u -X utf8 scripts\scan_tomorrow_picks.py >> logs\schedule.log 2>&1

REM 5) 아침 통합 텔레그램 발송 (재스캔 결과 포함)
echo [%date% %time%] [5/5] 아침 텔레그램 발송 >> logs\schedule.log
python -u -X utf8 scripts\send_evening_summary.py --send --morning >> logs\schedule.log 2>&1

echo [%date% %time%] BAT-A 완료 (5단계) >> logs\schedule.log
