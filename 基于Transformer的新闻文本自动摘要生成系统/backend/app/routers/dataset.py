"""
内置新闻示例数据集路由。
"""
from typing import List, Dict

from fastapi import APIRouter, HTTPException

from ..dataset import get_all_samples, get_sample_by_id

router = APIRouter(prefix="/api/dataset", tags=["数据集"])


@router.get("/list", summary="获取内置新闻示例数据集列表")
async def list_samples() -> List[Dict]:
    """返回所有内置新闻示例（含标题、分类、正文、参考摘要）。"""
    return get_all_samples()


@router.get("/load/{sample_id}", summary="按ID加载单条新闻示例")
async def load_sample(sample_id: int) -> Dict:
    """加载单条示例新闻的完整内容（标题、正文、参考摘要、分类）。"""
    sample = get_sample_by_id(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="示例ID不存在")
    return sample
