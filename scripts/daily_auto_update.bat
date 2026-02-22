@echo off
REM ============================================================
REM  Quantum Master - 수동 실행용 백업 BAT
REM  (스케줄러가 놓쳤을 때 더블클릭으로 수동 실행)
REM
REM  정규 스케줄은 아래 4개 BAT로 분리됨:
REM    QM_A_USClose     06:10  schedule_A_us_close.bat
REM    QM_B_Morning     07:00  schedule_B_morning.bat
REM    QM_C_ETFAlert    08:00  schedule_C_etf_alert.bat
REM    QM_D_AfterClose  16:30  schedule_D_after_close.bat
REM ============================================================

echo 장마감 전체 데이터 수집을 수동 실행합니다...
call D:\sub-agent-project\scripts\schedule_D_after_close.bat
