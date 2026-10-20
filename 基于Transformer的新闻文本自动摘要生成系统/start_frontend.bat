@echo off
chcp 65001 >nul
echo ========================================
echo   新闻摘要系统 - 前端服务启动脚本
echo ========================================
echo.

cd /d "%~dp0frontend"

echo [1/3] 检查 Node.js 环境...
node --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Node.js，请先安装 Node.js 16+
    pause
    exit /b 1
)

echo [2/3] 安装依赖...
if not exist "node_modules" (
    npm install
) else (
    echo node_modules 已存在，跳过安装
)

echo [3/3] 启动前端服务 (http://localhost:5173)...
echo.
npm run dev

pause
