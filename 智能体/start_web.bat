@echo off
chcp 65001 >nul
cd /d %~dp0
title 本地智能体网页服务

echo ============================================================
echo   本地智能体网页服务（FastAPI + 联网搜索）
echo ============================================================
echo.

echo [1/2] 检查端口 8000 是否已被占用...
netstat -ano -p tcp | findstr LISTENING | findstr ":8000 " >nul 2>&1
if %errorlevel%==0 (
    echo       检测到服务已在运行，直接打开浏览器...
    start "" http://127.0.0.1:8000
    echo.
    echo 浏览器地址：http://127.0.0.1:8000
    timeout /t 3 >nul
    exit /b 0
)

echo [2/2] 启动服务（启动后会自动打开浏览器，关闭本窗口即停止服务）...
echo.
python web_app.py

echo.
echo 服务已退出。如因报错退出，请查看上方错误信息（常见：未安装依赖请执行 pip install chromadb fastapi uvicorn）。
pause
