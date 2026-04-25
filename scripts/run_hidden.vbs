' run_hidden.vbs — BAT 파일을 숨김 창으로 실행
' 사용법: wscript.exe //B //Nologo run_hidden.vbs "schedule_X.bat" [arg1] [arg2] ...
If WScript.Arguments.Count = 0 Then
    WScript.Echo "Usage: wscript run_hidden.vbs <bat_file_path> [args...]"
    WScript.Quit 1
End If

Dim batPath, args, i
batPath = WScript.Arguments(0)
args = ""
For i = 1 To WScript.Arguments.Count - 1
    args = args & " " & WScript.Arguments(i)
Next

Set shell = CreateObject("WScript.Shell")
' cmd /c로 실행 + stdin을 NUL에서 읽어 대화형 명령(time 등) 행(hang) 방지
shell.Run "cmd /c """ & batPath & """" & args & " < NUL", 0, True
