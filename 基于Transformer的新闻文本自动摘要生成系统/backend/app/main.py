"""
FastAPI 应用入口。

启动方式:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import os

# 必须在导入 transformers 等库之前加载环境变量（镜像源、缓存目录等）
from . import runtime_env  # noqa: F401

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import summarize, evaluate, models, analysis, batch, feedback, dataset, news_query


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时预热，关闭时清理。"""
    logger.info("新闻摘要系统启动中...")
    # 预热默认模型（可注释掉以加快启动速度）
    # manager = get_model_manager()
    # manager._load_model(manager.default_model)
    logger.info("系统启动完成")
    yield
    logger.info("系统关闭")


app = FastAPI(
    title="基于 Transformer 的新闻文本自动摘要生成系统",
    description="使用 mT5 / Pegasus 等 Transformer 模型对中文新闻进行自动摘要生成。",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173")
origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(summarize.router)
app.include_router(evaluate.router)
app.include_router(models.router)
app.include_router(analysis.router)
app.include_router(batch.router)
app.include_router(feedback.router)
app.include_router(dataset.router)
app.include_router(news_query.router)  # 新闻数据查询库


# ---------------------------------------------------------------------------
# 前端静态文件托管（生产模式：单端口启动）
# 当 frontend/dist 存在时，后端自动托管前端页面与静态资源，
# 用户只需启动后端即可访问完整系统（如 http://localhost:8000/）。
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

_FRONTEND_DIST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "frontend", "dist",
)

if os.path.isdir(_FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
    logger.info(f"前端静态文件托管已启用: {_FRONTEND_DIST}")


if not os.path.isdir(_FRONTEND_DIST):
    @app.get("/", tags=["根路径"])
    async def root():
        """根路径，返回系统信息（仅开发模式，未托管前端时生效）。"""
        return {
            "name": "基于 Transformer 的新闻文本自动摘要生成系统",
            "version": "1.0.0",
            "docs": "/docs",
        }
