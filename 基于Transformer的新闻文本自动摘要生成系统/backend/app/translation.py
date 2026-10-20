"""
多语言摘要模块：中英互译（Helsinki-NLP/opus-mt 系列, MarianMT）。

- 中译英：opus-mt-zh-en（生成英文摘要）
- 英译中：opus-mt-en-zh（英文新闻自动翻译为中文后生成摘要）

采用懒加载，首次调用时从 hf-mirror.com 镜像下载并缓存；下载或加载失败时
优雅降级（返回失败提示，不影响主摘要流程）。

注意：Windows 上 sentencepiece（C++）无法打开包含中文的路径，因此将
tokenizer 小文件复制到 ASCII 安全的临时目录后再加载分词器；模型权重
由 PyTorch 加载（不受路径编码影响），仍走统一缓存。
"""
import os
import shutil
import threading
from typing import Optional

from loguru import logger

from . import runtime_env  # noqa: F401  配置镜像与缓存（须在 transformers 前）

_ZH_EN_REPO = "Helsinki-NLP/opus-mt-zh-en"
_EN_ZH_REPO = "Helsinki-NLP/opus-mt-en-zh"
_TOKENIZER_FILES = ["config.json", "tokenizer_config.json",
                    "vocab.json", "source.spm", "target.spm"]

_lock = threading.Lock()
_tokenizer = None
_model = None
_load_failed = False

# 英译中模型状态（独立懒加载）
_enzh_tokenizer = None
_enzh_model = None
_enzh_load_failed = False


def _ascii_safe_tokenizer_dir(repo: str, tag: str) -> str:
    """把 tokenizer 文件复制到 ASCII 安全目录，返回该目录路径。"""
    import tempfile

    from huggingface_hub import hf_hub_download
    dst_dir = os.path.join(tempfile.gettempdir(), f"opus-mt-{tag}-tok")
    os.makedirs(dst_dir, exist_ok=True)
    for name in _TOKENIZER_FILES:
        dst = os.path.join(dst_dir, name)
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            continue  # 已复制过
        src = hf_hub_download(repo, name)
        shutil.copyfile(src, dst)
    return dst_dir


def _ensure_loaded() -> bool:
    """懒加载中译英模型（线程安全），成功返回 True。"""
    global _tokenizer, _model, _load_failed
    if _model is not None:
        return True
    if _load_failed:
        return False
    with _lock:
        if _model is not None:
            return True
        if _load_failed:
            return False
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info(f"加载英文翻译模型: {_ZH_EN_REPO}")
            tok_dir = _ascii_safe_tokenizer_dir(_ZH_EN_REPO, "zh-en")
            _tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=False)
            _model = AutoModelForSeq2SeqLM.from_pretrained(_ZH_EN_REPO)
            _model.eval()
            logger.info("英文翻译模型加载成功")
            return True
        except Exception as e:
            logger.error(f"翻译模型加载失败（英文摘要将不可用）: {e}")
            _load_failed = True
            return False


def translate_to_english(text: str, max_length: int = 256) -> Optional[str]:
    """
    将中文文本翻译为英文。

    Returns:
        英文译文；模型不可用时返回 None。
    """
    if not text or not text.strip():
        return ""
    if not _ensure_loaded():
        return None
    try:
        import torch
        inputs = _tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=512)
        with torch.no_grad():
            ids = _model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )
        out = _tokenizer.decode(ids[0], skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
        return out.strip() or None
    except Exception as e:
        logger.error(f"英文翻译失败: {e}")
        return None


def _ensure_enzh_loaded() -> bool:
    """懒加载英译中模型（线程安全），成功返回 True。"""
    global _enzh_tokenizer, _enzh_model, _enzh_load_failed
    if _enzh_model is not None:
        return True
    if _enzh_load_failed:
        return False
    with _lock:
        if _enzh_model is not None:
            return True
        if _enzh_load_failed:
            return False
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info(f"加载英译中翻译模型: {_EN_ZH_REPO}")
            tok_dir = _ascii_safe_tokenizer_dir(_EN_ZH_REPO, "en-zh")
            _enzh_tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=False)
            _enzh_model = AutoModelForSeq2SeqLM.from_pretrained(_EN_ZH_REPO)
            _enzh_model.eval()
            logger.info("英译中翻译模型加载成功")
            return True
        except Exception as e:
            logger.error(f"英译中模型加载失败: {e}")
            _enzh_load_failed = True
            return False


def is_english_text(text: str) -> bool:
    """判断文本是否以英文为主（ASCII 字母占非空白字符 60% 以上）。"""
    sample = (text or "").strip()[:500]
    if not sample:
        return False
    non_space = [c for c in sample if not c.isspace()]
    if not non_space:
        return False
    ascii_alpha = sum(1 for c in non_space if c.isascii() and c.isalpha())
    return ascii_alpha / len(non_space) >= 0.6


def translate_to_chinese(text: str, max_length: int = 512) -> Optional[str]:
    """
    将英文文本翻译为中文（用于英文新闻先翻译再摘要）。

    Returns:
        中文译文；模型不可用时返回 None。
    """
    if not text or not text.strip():
        return ""
    if not _ensure_enzh_loaded():
        return None
    try:
        import torch
        inputs = _enzh_tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=512)
        with torch.no_grad():
            ids = _enzh_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )
        out = _enzh_tokenizer.decode(ids[0], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
        return out.strip() or None
    except Exception as e:
        logger.error(f"英译中翻译失败: {e}")
        return None
