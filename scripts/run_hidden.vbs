' run_hidden.vbs — BAT 파일을 숨김 창으로 실행
' 사용법: wscript.exe run_hidden.vbs "D:\sub-agent-project_퀀트봇\scripts\schedule_X.bat"
If WScript.Arguments.Count = 0 Then
    WScript.Echo "Usage: wscript run_hidden.vbs <bat_file_path>"
    WScript.Quit 1
End If

Dim batPath
batPath = WScript.Arguments(0)

Set shell = CreateObject("WScript.Shell")
' cmd /c로 실행 + stdin을 NUL에서 읽어 대화형 명령(time 등) 행(hang) 방지
shell.Run "cmd /c """ & batPath & """ < NUL", 0, True
