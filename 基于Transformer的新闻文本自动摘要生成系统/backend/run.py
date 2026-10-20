"""
后端服务启动脚本（可直接在 PyCharm 中右键运行本文件）。

注意：不要直接运行 app/main.py，因为 app 是一个 Python 包，
main.py 内部使用相对导入（from .routers import ...），
直接运行文件会报 "attempted relative import with no known parent package"。

正确方式二选一：
    1. 直接运行本文件：python run.py
    2. 命令行方式：    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import os

import uvicorn

if __name__ == "__main__":
    # 确保工作目录为本文件所在目录（backend/），使 app 包可被正确导入
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
