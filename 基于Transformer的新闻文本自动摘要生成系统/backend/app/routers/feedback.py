"""
人工反馈路由：保存用户在网页上编辑修正后的摘要（POST /api/feedback/save），
作为后续模型微调的补充数据集（JSONL 格式，每行一条）。
"""
import json
import os
import time
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api/feedback", tags=["人工反馈"])

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data")
_FEEDBACK_PATH = os.path.join(_DATA_DIR, "human_feedback.jsonl")


@router.post("/save", response_model=FeedbackResponse, summary="保存人工修正摘要")
async def save_feedback(request: FeedbackRequest):
    """记录人工修正内容：原文 / 模型初稿 / 人工修改稿，可作微调补充数据。"""
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "original_text": request.original_text,
            "model_summary": request.model_summary,
            "edited_summary": request.edited_summary,
            "model": request.model,
            "params": request.params or {},
        }
        with open(_FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        total = _count_records()
        logger.info(f"人工修正已保存，累计 {total} 条")
        return FeedbackResponse(
            saved=True, total_records=total, message="人工修正已保存，感谢反馈")
    except Exception as e:
        logger.exception("保存人工反馈失败")
        raise HTTPException(status_code=500, detail=f"保存失败: {e}")


@router.get("/list", summary="查看人工反馈记录数与最近记录")
async def list_feedback():
    total = _count_records()
    recent: List[Dict[str, Any]] = []
    try:
        if os.path.isfile(_FEEDBACK_PATH):
            with open(_FEEDBACK_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-5:]:
                if line.strip():
                    recent.append(json.loads(line))
    except Exception as e:
        logger.error(f"读取反馈失败: {e}")
    return {"total_records": total, "recent": recent}


def _count_records() -> int:
    if not os.path.isfile(_FEEDBACK_PATH):
        return 0
    try:
        with open(_FEEDBACK_PATH, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0
