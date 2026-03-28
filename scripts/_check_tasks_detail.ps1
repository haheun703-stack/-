Get-ScheduledTask | Where-Object {$_.TaskName -like "QM_*"} | ForEach-Object {
    $info = $_ | Get-ScheduledTaskInfo
    [PSCustomObject]@{
        TaskName   = $_.TaskName
        State      = $_.State
        LastResult = "0x{0:X}" -f $info.LastTaskResult
        LastRun    = $info.LastRunTime
        NextRun    = $info.NextRunTime
    }
} | Sort-Object TaskName | Format-Table -AutoSize
