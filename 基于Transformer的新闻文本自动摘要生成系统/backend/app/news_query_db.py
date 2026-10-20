# -*- coding: utf-8 -*-
"""新闻文本数据查询库：加载外部中文新闻数据集，提供检索、浏览与互相推荐。

数据来源：LCSTS（Large Scale Chinese Short Text Summarization Dataset）。
"""
import json
import os
import re
from typing import List, Dict, Optional

from loguru import logger
from jieba import analyse

_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "news_query_dataset.json"
)

_JA_KANA_RE = re.compile(r"[぀-ゟ゠-ヿ]")
_CJK_RE = re.compile(r"[一-鿿]")


def _is_valid_chinese(item: Dict) -> bool:
    """校验条目是否为合规中文新闻（剔除日文/空文本）。"""
    text = (item.get("text") or "") + (item.get("title") or "")
    if len(text) < 30 or len(text) > 6000:
        return False
    if _JA_KANA_RE.search(text):
        return False
    return len(_CJK_RE.findall(text)) >= 10


class NewsQueryDB:
    """新闻数据查询库，支持关键词检索、分类浏览与相似推荐。"""

    def __init__(self) -> None:
        self._items: List[Dict] = []
        self._keywords: Dict[int, set] = {}  # id -> set of keywords
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(_DATASET_PATH):
            logger.warning(f"新闻数据查询库文件不存在: {_DATASET_PATH}")
            return
        try:
            with open(_DATASET_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._items = [
                {
                    "id": i + 1,
                    "title": item.get("title", ""),
                    "text": item.get("text", ""),
                    "summary": item.get("summary", ""),
                    "category": item.get("category", "综合"),
                }
                for i, item in enumerate(raw)
                if _is_valid_chinese(item)
            ]
            # 预计算关键词，避免推荐时重复提取
            for item in self._items:
                kws = {w for w in analyse.extract_tags(
                    item["title"] + item["text"], topK=15) if w}
                self._keywords[item["id"]] = kws
            logger.info(f"新闻数据查询库加载完成：{len(self._items)} 条")
        except Exception as e:
            logger.error(f"新闻数据查询库加载失败: {e}")

    def get_all(self) -> List[Dict]:
        return self._items

    def get_by_id(self, item_id: int) -> Optional[Dict]:
        for item in self._items:
            if item["id"] == item_id:
                return item
        return None

    def search(
        self,
        keyword: str = "",
        category: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict:
        """关键词 + 分类检索，分页返回。"""
        items = self._items
        if category and category != "全部":
            items = [x for x in items if x["category"] == category]
        if keyword:
            kw = keyword.strip()
            items = [
                x for x in items
                if kw in x["title"] or kw in x["text"] or kw in x["summary"]
            ]
        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = items[start:end]
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": [
                {
                    "id": x["id"],
                    "title": x["title"],
                    "category": x["category"],
                    "text": x["text"][:300] + ("..." if len(x["text"]) > 300 else ""),
                    "summary": x["summary"],
                    "full_text": x["text"],
                }
                for x in page_items
            ],
        }

    def categories(self) -> List[str]:
        return sorted({x["category"] for x in self._items if x["category"]})

    def recommend(
        self,
        text: str,
        topk: int = 5,
        exclude_id: Optional[int] = None,
    ) -> List[Dict]:
        """基于关键词重叠度推荐相似新闻（使用预计算关键词，响应快）。"""
        if not text or not self._items:
            return []
        cur_kws = {w for w in analyse.extract_tags(text, topK=15) if w}
        if not cur_kws:
            return []
        results = []
        for item in self._items:
            if exclude_id is not None and item["id"] == exclude_id:
                continue
            lib_kws = self._keywords.get(item["id"], set())
            if not lib_kws:
                continue
            inter = cur_kws & lib_kws
            if not inter:
                continue
            sim = len(inter) / (len(cur_kws | lib_kws) ** 0.5)
            results.append({
                "id": str(item["id"]),
                "title": item["title"],
                "snippet": item["text"][:80] + ("..." if len(item["text"]) > 80 else ""),
                "category": item["category"],
                "similarity": round(sim, 4),
                "matched_keywords": sorted(inter)[:6],
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:topk]


_db: Optional[NewsQueryDB] = None


def get_db() -> NewsQueryDB:
    global _db
    if _db is None:
        _db = NewsQueryDB()
    return _db
