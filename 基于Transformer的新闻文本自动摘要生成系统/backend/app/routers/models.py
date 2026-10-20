"""
模型列表与健康检查路由。
"""
from fastapi import APIRouter

from ..model_manager import get_model_manager, AVAILABLE_MODELS
from ..schemas import ModelsResponse, ModelInfo, HealthResponse

router = APIRouter(tags=["系统信息"])


@router.get("/api/models", response_model=ModelsResponse, summary="获取可用模型列表")
async def get_models():
    """返回所有可用的摘要模型及其元信息。"""
    manager = get_model_manager()
    models = [
        ModelInfo(
            name=name,
            display_name=info["name"],
            description=info["description"],
            type=info["type"],
        )
        for name, info in AVAILABLE_MODELS.items()
    ]
    return ModelsResponse(models=models, default_model=manager.default_model)


@router.get("/api/health", response_model=HealthResponse, summary="健康检查")
async def health():
    """返回服务健康状态与当前设备信息。"""
    manager = get_model_manager()
    return HealthResponse(
        status="ok",
        device=manager.device,
        default_model=manager.default_model,
    )
