@echo off
chcp 65001 >nul
echo ========================================
echo   新闻摘要系统 - 停止服务
echo ========================================
echo.

echo 正在停止后端(8000)与前端(5173)服务...
powershell -NoProfile -Command "$ports = @(8000,5173); $conns = Get-NetTCPConnection -LocalPort $ports -State Listen -ErrorAction SilentlyContinue; if ($conns) { $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique; foreach ($p in $pids) { try { $proc = Get-Process -Id $p -ErrorAction SilentlyContinue; Write-Host ('  停止进程: ' + $proc.ProcessName + ' (PID ' + $p + ')'); Stop-Process -Id $p -Force -ErrorAction SilentlyContinue } catch {} } } else { Write-Host '  没有正在运行的服务' }"

echo.
echo 已完成。
pause
