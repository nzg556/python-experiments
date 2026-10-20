"""
ROUGE 评估路由：POST /api/evaluate
"""
from fastapi import APIRouter, HTTPException
from loguru import logger

from ..utils import compute_rouge
from ..schemas import EvaluateRequest, EvaluateResponse

router = APIRouter(prefix="/api", tags=["摘要评估"])


@router.post("/evaluate", response_model=EvaluateResponse, summary="评估摘要质量（ROUGE）")
async def evaluate(request: EvaluateRequest):
    """
    计算生成摘要与参考摘要之间的 ROUGE 分数。

    - **hypothesis**: 生成的摘要
    - **reference**: 参考摘要（标准答案）
    """
    try:
        scores = compute_rouge(request.hypothesis, request.reference)
        return EvaluateResponse(
            rouge1=scores["rouge1"],
            rouge2=scores["rouge2"],
            rougeL=scores["rougeL"],
        )
    except Exception as e:
        logger.exception("ROUGE 评估失败")
        raise HTTPException(status_code=500, detail=f"评估失败: {e}")
