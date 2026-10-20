@echo off
chcp 65001 >nul
echo ========================================
echo   新闻摘要系统 - 后端服务启动脚本
echo ========================================
echo.

cd /d "%~dp0backend"

echo [1/3] 检查 Python 环境...
set PYTHON_EXE=..\..\Scripts\python.exe
if exist "%PYTHON_EXE%" (
    echo 使用项目虚拟环境: %PYTHON_EXE%
) else (
    echo 未找到项目虚拟环境，使用系统 Python
    set PYTHON_EXE=python
)

echo [2/3] 安装/检查依赖...
"%PYTHON_EXE%" -m pip install -r requirements.txt

echo [3/3] 启动后端服务 (http://localhost:8000)...
echo API 文档: http://localhost:8000/docs
echo.
"%PYTHON_EXE%" run.py

pause
