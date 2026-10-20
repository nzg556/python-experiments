"""
历史新闻库：以 JSONL 文件（data/news_library.jsonl）持久化已摘要过的新闻，
用于"新闻关联推荐"的关键词检索。记录去重（按正文前缀哈希）。
"""
import hashlib
import json
import os
import threading
from typing import Dict, List, Optional

from loguru import logger

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_LIBRARY_PATH = os.path.join(_DATA_DIR, "news_library.jsonl")

_lock = threading.Lock()


def _ensure_dir() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)


def _doc_id(text: str) -> str:
    """以正文前 200 字符生成稳定文档 ID。"""
    return hashlib.md5((text or "")[:200].encode("utf-8")).hexdigest()[:12]


def load_library() -> List[Dict]:
    """读取全部历史新闻记录。"""
    if not os.path.isfile(_LIBRARY_PATH):
        return []
    items: List[Dict] = []
    try:
        with open(_LIBRARY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    except Exception as e:
        logger.error(f"读取新闻库失败: {e}")
    return items


def add_document(text: str, title: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 category: str = "") -> str:
    """
    将新闻加入历史库（正文去重），返回文档 ID。
    """
    doc_id = _doc_id(text)
    with _lock:
        _ensure_dir()
        try:
            if os.path.isfile(_LIBRARY_PATH):
                with open(_LIBRARY_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip() and json.loads(line).get("id") == doc_id:
                            return doc_id  # 已存在
        except Exception:
            pass
        record = {
            "id": doc_id,
            "title": title or (text or "")[:30],
            "text": (text or "")[:2000],
            "keywords": keywords or [],
            "category": category,
        }
        try:
            with open(_LIBRARY_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"写入新闻库失败: {e}")
    return doc_id


def get_document(doc_id: str) -> Optional[Dict]:
    for item in load_library():
        if item.get("id") == doc_id:
            return item
    return None
