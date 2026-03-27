@echo off
REM FLOWX Phase C — Windows Task Scheduler 일괄 등록
REM 관리자 권한 필요

echo FLOWX BAT 스케줄 등록 중...

schtasks /create /tn "QM_M_MorningBriefing" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_M_morning_briefing.bat" /sc daily /st 08:00 /f
echo   [1/5] QM_M_MorningBriefing (08:00) 등록 완료

schtasks /create /tn "QM_N_SignalLog" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_N_signal_log.bat" /sc daily /st 08:20 /f
echo   [2/5] QM_N_SignalLog (08:20) 등록 완료

schtasks /create /tn "QM_P1_DaySignal" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_P_daytrading.bat log" /sc daily /st 09:05 /f
echo   [3/5] QM_P1_DaySignal (09:05) 등록 완료

schtasks /create /tn "QM_P2_DayClose" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_P_daytrading.bat close" /sc daily /st 15:20 /f
echo   [4/5] QM_P2_DayClose (15:20) 등록 완료

schtasks /create /tn "QM_O_SignalTrack" /tr "D:\sub-agent-project_퀀트봇\scripts\schedule_O_signal_track.bat" /sc daily /st 16:10 /f
echo   [5/5] QM_O_SignalTrack (16:10) 등록 완료

echo.
echo === 등록 확인 ===
schtasks /query /tn "QM_M_MorningBriefing" /fo list | findstr "작업 이름 상태 다음 실행"
schtasks /query /tn "QM_N_SignalLog" /fo list | findstr "작업 이름 상태 다음 실행"
schtasks /query /tn "QM_P1_DaySignal" /fo list | findstr "작업 이름 상태 다음 실행"
schtasks /query /tn "QM_P2_DayClose" /fo list | findstr "작업 이름 상태 다음 실행"
schtasks /query /tn "QM_O_SignalTrack" /fo list | findstr "작업 이름 상태 다음 실행"

echo.
echo FLOWX 스케줄 등록 완료!
pause
