"""新闻数据查询库路由：检索、浏览、推荐。"""
from fastapi import APIRouter, HTTPException, Query

from ..news_query_db import get_db

router = APIRouter(prefix="/api/news-library", tags=["新闻数据查询库"])


@router.get("/search", summary="关键词/分类检索新闻数据集")
async def search(
    keyword: str = Query("", description="关键词"),
    category: str = Query("全部", description="分类"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    db = get_db()
    return db.search(keyword=keyword, category=category, page=page, page_size=page_size)


@router.get("/categories", summary="获取所有新闻分类")
async def categories():
    db = get_db()
    return {"categories": ["全部"] + db.categories()}


@router.get("/detail/{item_id}", summary="按ID获取新闻详情")
async def detail(item_id: int):
    db = get_db()
    item = db.get_by_id(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="新闻不存在")
    return item


@router.post("/recommend", summary="根据输入文本推荐相似新闻（互相推荐）")
async def recommend(body: dict):
    """输入一段新闻文本，返回数据集中最相似的若干条新闻。"""
    text = (body or {}).get("text", "")
    topk = (body or {}).get("topk", 5)
    db = get_db()
    return {"recommendations": db.recommend(text, topk=topk)}
