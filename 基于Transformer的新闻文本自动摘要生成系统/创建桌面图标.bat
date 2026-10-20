@echo off
chcp 65001 >nul
echo ========================================
echo   新闻摘要系统 - 创建桌面图标
echo ========================================
echo.

set BASE=%~dp0

if not exist "%BASE%启动系统.vbs" (
    echo 错误：找不到 %BASE%启动系统.vbs
    echo 请确保在项目根目录运行此脚本。
    pause
    exit /b 1
)

if not exist "%BASE%assets\app.ico" (
    echo 错误：找不到 %BASE%assets\app.ico
    echo 请先确认项目文件完整。
    pause
    exit /b 1
)

rem 通过环境变量传递路径，PowerShell 命令使用 Base64(UTF-16LE) 编码，
rem 避免中文路径在命令行中被错误编码导致快捷方式创建失败
set NEWS_BASE=%BASE%
powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand JABiAGEAcwBlACAAPQAgACQAZQBuAHYAOgBOAEUAVwBTAF8AQgBBAFMARQAKACQAdwBzACAAPQAgAE4AZQB3AC0ATwBiAGoAZQBjAHQAIAAtAEMAbwBtAE8AYgBqAGUAYwB0ACAAVwBTAGMAcgBpAHAAdAAuAFMAaABlAGwAbAAKACQAZABlAHMAawB0AG8AcAAgAD0AIABbAEUAbgB2AGkAcgBvAG4AbQBlAG4AdABdADoAOgBHAGUAdABGAG8AbABkAGUAcgBQAGEAdABoACgAIgBEAGUAcwBrAHQAbwBwACIAKQAKACQAbABuAGsAIAA9ACAAJAB3AHMALgBDAHIAZQBhAHQAZQBTAGgAbwByAHQAYwB1AHQAKAAiACQAZABlAHMAawB0AG8AcABcALBl+5VYZIGJ+3zffi4AbABuAGsAIgApAAoAJABsAG4AawAuAFQAYQByAGcAZQB0AFAAYQB0AGgAIAA9ACAAIgAkAHsAYgBhAHMAZQB9AC9UqFL7fN9+LgB2AGIAcwAiAAoAJABsAG4AawAuAFcAbwByAGsAaQBuAGcARABpAHIAZQBjAHQAbwByAHkAIAA9ACAAJABiAGEAcwBlAAoAJABsAG4AawAuAEkAYwBvAG4ATABvAGMAYQB0AGkAbwBuACAAPQAgACIAJAB7AGIAYQBzAGUAfQBhAHMAcwBlAHQAcwBcAGEAcABwAC4AaQBjAG8AIgAKACQAbABuAGsALgBEAGUAcwBjAHIAaQBwAHQAaQBvAG4AIAA9ACAAIgD6V45OVAByAGEAbgBzAGYAbwByAG0AZQByAIR2sGX7lYdlLGfqgahSWGSBiR91EGL7fN9+IgAKACQAbABuAGsALgBTAGEAdgBlACgAKQAKAFcAcgBpAHQAZQAtAEgAbwBzAHQAIAAiAEMAcgBlAGEAdABlAGQAIABzAGgAbwByAHQAYwB1AHQAOgAgACQAZABlAHMAawB0AG8AcABcALBl+5VYZIGJ+3zffi4AbABuAGsAIgAKAA==

if errorlevel 1 (
    echo.
    echo 创建失败，请尝试右键以管理员身份运行此脚本。
    pause
    exit /b 1
)

echo.
echo 完成！双击桌面"新闻摘要系统"图标即可一键启动。
pause
